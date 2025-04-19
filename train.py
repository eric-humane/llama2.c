"""
This training script can be run both on a single GPU in debug mode,
and in larger training runs with DistributedDataParallel (DDP).

Usage examples:
- Single GPU (debug mode):
  $ python -m train.py --compile=False --eval_iters=10 --batch_size=8

- DDP on 4 GPUs on 1 node:
  $ torchrun --standalone --nproc_per_node=4 train.py
"""

import math
import os
import time
import importlib.util
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Transformer, ModelArgs

from tinystories import Task
from export import model_export

# -----------------------------------------------------------------------------
# I/O and Hyperparameters
out_dir = "out"
eval_interval = 5000
log_interval = 500
eval_iters = 100
eval_only = False  # if True, exit after first evaluation
always_save_checkpoint = False
init_from = "scratch"  # or "resume"

# Wandb logging (disabled by default)
wandb_log = False
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Data parameters
batch_size = (
    128  # micro-batch size (effective batch size depends on gradient accumulation)
)
max_seq_len = 256
vocab_source = "custom"  # either "llama2" or "custom"
vocab_size = 2048  # e.g., custom tokenizer's vocab size

# Model parameters
dim = 192
n_layers = 3
n_heads = 16
n_kv_heads = 4
multiple_of = 32
dropout = 0.05

# Training/optimizer parameters
gradient_accumulation_steps = 1
learning_rate = 5e-5  # Use a much smaller learning rate than 0.1
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0  # set to 0.0 to disable clipping
decay_lr = True  # Turn decay_lr back on, but with a safer learning rate
warmup_iters = 1000

# System parameters
device = "cuda"  # use 'cuda' for GPU training
dtype = "bfloat16"  # choose from "float32", "bfloat16", "float16"
use_compile = False  # use torch.compile if desired

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# Import config values from configurator.py instead of using exec
spec = importlib.util.spec_from_file_location("configurator", "configurator.py")
configurator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configurator)
if hasattr(configurator, "get_config_overrides"):
    overrides = configurator.get_config_overrides(config_keys)
    globals().update(overrides)
config = {k: globals()[k] for k in config_keys}

lr_decay_iters = max_iters
min_lr = 0.0

assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000, (
    "The vocab from Meta has 32K tokens"
)

# -----------------------------------------------------------------------------
# DDP and Device Setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    if not torch.distributed.is_initialized():
        init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    if gradient_accumulation_steps < ddp_world_size:
        print(
            f"WARNING: gradient_accumulation_steps ({gradient_accumulation_steps}) is less than ddp_world_size ({ddp_world_size})."
        )
        gradient_accumulation_steps = ddp_world_size
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = (
    gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
)
if master_process:
    print(f"tokens per iteration: {tokens_per_iter:,}")
    print(
        f"({gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len)"
    )

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# -----------------------------------------------------------------------------
# Data Loading Setup:
# Note: The num_workers parameter here refers to CPU worker processes for data loading
# (not GPU workers). On an H100 system, values between 8 and 16 often work well;
# tune based on your system's CPU and I/O performance.
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=16,
)

# -----------------------------------------------------------------------------
# Model Initialization
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)
if init_from == "scratch":
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "max_seq_len",
    ]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# -----------------------------------------------------------------------------
# Define external loss function.
# Assume the model returns logits of shape [B, T, vocab_size].
# For CrossEntropyLoss to work, we need to convert logits to shape [B, vocab_size, T]
# and the target should be [B, T] with each element being a class index.
loss_fn = torch.nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# Initialize GradScaler (without a device string) and optimizer.
scaler = torch.amp.GradScaler(enabled=dtype == "float16")
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

if use_compile:
    print("compiling the model... (takes ~1 minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    prefix = "_orig_mod." if use_compile else ""
    # pylint: disable=protected-access
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])


# -----------------------------------------------------------------------------
# Loss Estimation Function: compute loss externally.
@torch.no_grad()
def estimate_loss():
    """Evaluate model on train and validation splits and return average loss.

    Returns:
        dict: Dictionary containing average loss values for 'train' and 'val' splits
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        split_losses = torch.zeros(eval_iters)
        for idx in range(eval_iters):
            x_eval, y_eval = next(batch_iter)
            with ctx:
                eval_logits = model(x_eval, y_eval)  # output shape: [B, T, vocab_size]
                # Check if output is a tuple and extract the logits
                if isinstance(eval_logits, tuple):
                    eval_logits = eval_logits[
                        0
                    ]  # Assuming logits are the first element
                # Rearrange logits to [B, vocab_size, T] for cross_entropy
                eval_logits = eval_logits.transpose(1, 2)
                # Here, y_eval is assumed to be of shape [B, T] containing class indices.
                eval_loss = loss_fn(eval_logits, y_eval)
            split_losses[idx] = eval_loss.item()
        out[split] = split_losses.mean()
    model.train()
    return out


def get_lr(it):
    """Calculate learning rate based on iteration number using warmup and cosine decay.

    Args:
        it: Current iteration number

    Returns:
        Learning rate value
    """
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch initial batch

# -----------------------------------------------------------------------------
# Global iteration variables.
iter_num = 0  # Global training iteration counter.
best_val_loss = 1e9  # Best validation loss.
local_iter_num = 0  # For logging.
t0 = time.time()
raw_model = model.module if ddp else model  # Unwrap DDP if necessary.
running_mfu = -1.0

# -----------------------------------------------------------------------------
# Training Loop
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    },
                    step=iter_num,
                )
            except (wandb.errors.Error, ConnectionError) as e:
                print(f"wandb logging failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                # Uncomment below if using torch.save:
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
                model_export(raw_model, os.path.join(out_dir, "modelv1.bin"), version=1)

    if iter_num == 0 and eval_only:
        break

    # Forward-backward update with gradient accumulation.
    for micro_step in range(gradient_accumulation_steps):
        if ddp and micro_step < gradient_accumulation_steps - 1:
            context = model.no_sync()
        else:
            context = nullcontext()
        with context, ctx:
            output = model(X, Y)  # output: [B, T, vocab_size]
            # Check if output is a tuple and extract the logits
            if isinstance(output, tuple):
                output = output[0]  # Assuming logits are the first element
            # Rearrange logits to [B, vocab_size, T] for cross_entropy.
            output = output.transpose(1, 2)
            # Use Y as is (shape: [B, T]) containing target indices.
            loss = loss_fn(output, Y)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        X, Y = next(train_batch_iter)

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt * 1000:.2f}ms | mfu {running_mfu * 100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
