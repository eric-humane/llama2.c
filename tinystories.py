"""
Download, preprocess and serve datasets from Hugging Face as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
import shutil
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

# Detect device and set defaults for parallelism
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CPU_CORES = min(multiprocessing.cpu_count(), 64)  # Limit to avoid system overload
NUM_WORKERS_DEFAULT = max(1, NUM_CPU_CORES - 1)  # Leave one core for system processes
GPU_MEM_FRACTION = 0.95  # Use 95% of available GPU memory on A100s
MAX_SHARD_SIZE = 2000  # Larger shard size for high-end systems


def setup_multi_gpu():
    """
    Configure distributed training for multi-GPU setups.
    Returns the local rank or -1 if not using distributed training.
    """
    if not torch.cuda.is_available():
        return -1

    # Check if using distributed training
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize the distributed environment
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        print(
            f"Distributed training active - Rank: {rank}/{world_size}, Local rank: {local_rank}"
        )
        return local_rank

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(
            f"Multiple GPUs detected ({num_gpus}), but not using distributed training."
        )
        print(
            "For multi-GPU training, launch with: python -m torch.distributed.launch --nproc_per_node={num_gpus} tinystories.py ..."
        )

    return -1


# Set up distributed training on multi-GPU systems
LOCAL_RANK = setup_multi_gpu()


def download(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
    quiet: bool = False,
):
    """
    Downloads a dataset from Hugging Face and processes it to the expected format

    Args:
        dataset_name: Name of the dataset on Hugging Face
        dataset_config: Optional configuration name for the dataset
        split: Dataset split to download (train, validation, test, etc.)
        text_field: The field in the dataset that contains the text data
        quiet: Reduce verbosity of output logs
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_dir = os.path.join(DATA_CACHE_DIR, dataset_name.split("/")[-1])
    os.makedirs(data_dir, exist_ok=True)

    # Only the main process should download and process the dataset
    if LOCAL_RANK not in [-1, 0]:
        if dist.is_initialized():
            dist.barrier()  # Wait for rank 0 to finish downloading
        return

    if not quiet:
        print(f"Downloading {dataset_name} from Hugging Face...")

    # Check if config name is needed but not provided
    if dataset_config is None:
        configs = get_dataset_config_names(dataset_name)
        if configs and len(configs) > 0 and not quiet:
            print(f"Available configs for {dataset_name}: {configs}")
            dataset_config = configs[0]
            print(f"Using config: {dataset_config}")

    # Load the dataset
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    except (ValueError, FileNotFoundError, ConnectionError, IOError) as e:
        print(f"Error loading dataset: {e}")
        return

    # Process and save the dataset in batches, optimized for high-performance systems
    num_shards = max(1, len(dataset) // MAX_SHARD_SIZE)
    if not quiet:
        print(f"Processing dataset into {num_shards} shards...")

    for i in tqdm(range(num_shards), disable=quiet, desc="Processing shards"):
        shard_start = i * (len(dataset) // num_shards)
        shard_end = (
            (i + 1) * (len(dataset) // num_shards)
            if i < num_shards - 1
            else len(dataset)
        )

        shard_data = []
        for j in range(shard_start, shard_end):
            if text_field in dataset[j]:
                shard_data.append(dataset[j][text_field])
            else:
                print(f"Warning: '{text_field}' field not found in example {j}.")
                print(f"Available fields: {dataset[j].keys()}")
                if j == shard_start:  # Only print this message once
                    return

        shard_filename = os.path.join(data_dir, f"shard_{i:05d}.json")
        with open(shard_filename, "w", encoding="utf-8") as f:
            json.dump(shard_data, f, ensure_ascii=False)

    # Signal to other processes that download is complete
    if dist.is_initialized():
        dist.barrier()

    # Print a single example for debugging if not quiet
    if not quiet:
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        with open(shard_filenames[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        print("Download done.")
        print(f"Number of shards: {len(shard_filenames)}")
        print(f"Example text:\n{data[0]}")


def train_vocab(vocab_size, dataset_name: str, quiet=False, special_tokens=None):
    """
    Trains a custom sentencepiece tokenizer on the dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.

    Args:
        vocab_size: Size of the vocabulary
        dataset_name: Name of the dataset to train on
        quiet: Whether to reduce output verbosity
        special_tokens: List of special tokens to add to the vocabulary (e.g. ["[INST]", "[/INST]"])
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "corpus.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, dataset_name)
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not quiet:
        print(f"Writing temporary file {tiny_file} with {num_shards} shards...")

    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards], disable=quiet):
            with open(shard, "r", encoding="utf-8") as f:
                data = json.load(f)
            for example in data:
                text = example
                text = text.strip()
                of.write(text + "\n")

    if not quiet:
        print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")
        print("Will now train the vocab...")

    # Set up logging for sentencepiece (it's very verbose by default)
    # Hide most messages when quiet is True
    spm_train_kwargs = {
        "input": tiny_file,
        "model_prefix": prefix,
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "self_test_sample_size": 0,
        "input_format": "text",
        "character_coverage": 1.0,
        "num_threads": NUM_CPU_CORES,  # Use all CPU cores
        "split_digits": True,
        "allow_whitespace_only_pieces": True,
        "byte_fallback": True,
        "unk_surface": r" \342\201\207 ",
        "normalization_rule_name": "identity",
    }

    # Add special tokens if provided
    if special_tokens:
        if not quiet:
            print(f"Adding special tokens: {special_tokens}")
        spm_train_kwargs["user_defined_symbols"] = special_tokens

    # Add quiet output if needed
    if quiet:
        spm_train_kwargs["minloglevel"] = 2  # Error level only

    # 2) train the sentencepiece model
    spm.SentencePieceTrainer.Train(**spm_train_kwargs)

    if not quiet:
        print(f"Trained tokenizer is in {prefix}.model")
        print("Done.")


def process_shard(args, vocab_size, dataset_name, quiet=False):
    """
    Process a single data shard: tokenize text and save as binary file.

    Args:
        args: Path to the shard file
        vocab_size: Size of vocabulary to use
        dataset_name: Name of the dataset
        quiet: Whether to suppress output

    Returns:
        Status message about the processed shard
    """
    shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_tokens = []
    # Remove the tqdm progress bar to avoid multiple progress bars
    for example in data:
        text = example
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        bin_dir = os.path.join(DATA_CACHE_DIR, dataset_name)
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    # Only print stats if not quiet
    if not quiet:
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / (
            (all_tokens == 1).sum() + 1e-10
        )  # Add small epsilon to avoid division by zero
        # Use a return value instead of printing to avoid noise
        return f"Saved {os.path.basename(tokenized_filename)}, average seqlen: {avg_seq_len:.2f}"


def pretokenize(vocab_size, dataset_name: str, quiet=False):
    """
    Pretokenize all shards of the dataset
    """
    # Only main process needs to check if the files are already tokenized
    if LOCAL_RANK not in [-1, 0]:
        if dist.is_initialized():
            dist.barrier()  # Wait for rank 0 to complete
        return

    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, dataset_name)
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    print(f"Processing {len(shard_filenames)} shards...")

    # Calculate optimal chunksize for parallel processing based on num cores
    chunksize = max(1, len(shard_filenames) // (NUM_CPU_CORES * 2))

    # process all the shards in a process pool
    fun = partial(
        process_shard, vocab_size=vocab_size, dataset_name=dataset_name, quiet=quiet
    )

    # Use all available cores for processing with a single progress bar
    with ProcessPoolExecutor(max_workers=NUM_CPU_CORES) as executor:
        results = list(
            tqdm(
                executor.map(fun, shard_filenames, chunksize=chunksize),
                total=len(shard_filenames),
                disable=quiet,
                desc="Pretokenizing shards",
                unit="shard",
            )
        )

    # Signal to other processes that tokenization is complete
    if dist.is_initialized():
        dist.barrier()

    # Print summary stats if not quiet and if there are results
    if results:
        # Filter out None values (from quiet mode in process_shard)
        results = [r for r in results if r is not None]
        if results:
            print("\n".join(results))
    print("Done pretokenizing.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(
        self,
        split,
        max_seq_len,
        vocab_size,
        vocab_source,
        dataset_name: str,
        quiet=False,
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.dataset_name = dataset_name
        self.quiet = quiet

        # Determine the shard filenames
        if self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            bin_dir = os.path.join(DATA_CACHE_DIR, self.dataset_name)
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
        else:
            raise ValueError(
                f"Unexpected vocab_source: {self.vocab_source}. Expected 'llama2' or 'custom'."
            )

        self.shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        self.shard_filenames = (
            self.shard_filenames[1:]
            if self.split == "train"
            else self.shard_filenames[:1]
        )
        assert len(self.shard_filenames) > 0, f"No bin files found in {bin_dir}"

        # Calculate estimated total length for the dataset
        self.total_tokens = 0
        for shard in self.shard_filenames:
            self.total_tokens += os.path.getsize(shard) // 2  # uint16 = 2 bytes

        # Estimate number of sequences based on max_seq_len
        self.estimated_length = self.total_tokens // self.max_seq_len

    def __getitem__(self, idx):
        # Not used for IterableDataset, but adding to satisfy linter
        # IterableDataset uses __iter__ instead
        raise NotImplementedError("IterableDataset doesn't support indexing")

    def __len__(self):
        """Return an estimated length of the dataset for DistributedSampler"""
        return self.estimated_length

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        if not self.quiet:
            print(f"Created a PretokDataset with rng seed {seed}")
        rng = random.Random(seed)

        shard_filenames = self.shard_filenames.copy()

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


# -----------------------------------------------------------------------------
# public interface functions


def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")


class Task:
    """
    Utility class for handling dataset batch iteration.
    Provides methods for loading and iterating through batches of tokenized data.
    """

    @staticmethod
    def iter_batches(
        batch_size,
        device=None,
        num_workers=None,
        quiet=False,
        dataset_name="TinyStories",
        **dataset_kwargs,
    ):
        """
        Create and iterate through batches of tokenized data.

        Args:
            batch_size: Number of sequences per batch
            device: Target device for tensors (default: auto-detect)
            num_workers: Number of loader worker processes (default: auto-detect)
            quiet: Whether to suppress output messages
            dataset_name: Name of the dataset to load
            **dataset_kwargs: Additional arguments passed to PretokDataset

        Yields:
            Tuples of (input_tensor, target_tensor) batches
        """
        # Use default device if none provided
        if device is None:
            if LOCAL_RANK >= 0:
                device = torch.device(f"cuda:{LOCAL_RANK}")
            else:
                device = DEVICE

        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)

        # Use default number of workers if none provided
        if num_workers is None:
            # Scale workers based on whether we're using multi-GPU
            if dist.is_initialized():
                # Fewer workers per GPU when using multiple GPUs
                num_workers = max(1, NUM_WORKERS_DEFAULT // torch.cuda.device_count())
            else:
                num_workers = NUM_WORKERS_DEFAULT

        if not quiet:
            print(f"Using device: {device}, workers: {num_workers}")

        # Pass quiet to the dataset
        if "quiet" not in dataset_kwargs:
            dataset_kwargs["quiet"] = quiet

        # Ensure dataset_name is included in kwargs
        if "dataset_name" not in dataset_kwargs:
            dataset_kwargs["dataset_name"] = dataset_name

        ds = PretokDataset(**dataset_kwargs)

        # For distributed training, we'll use a sequential sampler
        # This prevents the DistributedSampler from trying to call len()
        loader_args = {
            "batch_size": batch_size,
            "pin_memory": True,
            "num_workers": num_workers,
        }

        if dist.is_initialized():
            # Custom handling for distributed mode - manually partition data
            rank = dist.get_rank()

            def worker_init_fn(worker_id):
                # Set unique seed for each worker and rank
                worker_seed = 42 + worker_id + 1337 * rank
                random.seed(worker_seed)
                np.random.seed(worker_seed)

            loader_args["worker_init_fn"] = worker_init_fn

        dl = torch.utils.data.DataLoader(ds, **loader_args)

        # Set optimal CUDA settings if GPU is available
        if device.type == "cuda":
            # Enable cudnn benchmark for faster training
            torch.backends.cudnn.benchmark = True
            # Set memory allocation settings - 95% for A100s
            torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)

        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


def fullclean(quiet=False):
    """
    Delete all downloaded datasets, tokenized files and output directories
    for starting fresh.
    """
    # Clean data directory
    if os.path.exists(DATA_CACHE_DIR):
        if not quiet:
            print(f"Cleaning {DATA_CACHE_DIR}/ directory...")
        try:
            # Remove all files and subdirectories in data/
            for item in os.listdir(DATA_CACHE_DIR):
                item_path = os.path.join(DATA_CACHE_DIR, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    if not quiet:
                        print(f"Deleted file: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    if not quiet:
                        print(f"Deleted directory: {item_path}")
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"Error while cleaning {DATA_CACHE_DIR}/: {e}")
    else:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        if not quiet:
            print(f"Created empty {DATA_CACHE_DIR}/ directory")

    # Clean miniout directory
    if os.path.exists("miniout"):
        if not quiet:
            print("Cleaning miniout/ directory...")
        try:
            # Remove all files and subdirectories in miniout/
            for item in os.listdir("miniout"):
                item_path = os.path.join("miniout", item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    if not quiet:
                        print(f"Deleted file: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    if not quiet:
                        print(f"Deleted directory: {item_path}")
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"Error while cleaning miniout/: {e}")
    else:
        os.makedirs("miniout", exist_ok=True)
        if not quiet:
            print("Created empty miniout/ directory")

    if not quiet:
        print("Clean complete.")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    # Command-line help text
    # These stages are designed to be run in order.
    #
    # Download any dataset from Hugging Face:
    # python tinystories.py download --dataset_name="roneneldan/TinyStories" --text_field="text"
    #
    # To tokenize data with the Llama 2 tokenizer:
    # python tinystories.py pretokenize --dataset_name="TinyStories"
    #
    # To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    # python tinystories.py train_vocab --vocab_size=2048 --dataset_name="TinyStories"
    # python tinystories.py pretokenize --vocab_size=2048 --dataset_name="TinyStories"
    #
    # To tokenize with custom special tokens (e.g., for instruction tuning):
    # python tinystories.py train_vocab --vocab_size=2048 --special_tokens="[INST],[/INST]"
    #
    # To clean all downloaded data and output files:
    # python tinystories.py fullclean

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        type=str,
        choices=["download", "pretokenize", "train_vocab", "fullclean"],
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=0,
        help="pretokenization vocab size. 0 = use Llama 2 tokenizer.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="roneneldan/TinyStories",
        help="Dataset name on Hugging Face",
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="Dataset configuration name"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to download"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field containing the text in the dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto)",
    )
    parser.add_argument(
        "--gpu_mem_fraction",
        type=float,
        default=None,
        help="Fraction of GPU memory to use (0.0-1.0)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce verbosity of output logs"
    )
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=None,
        help="Maximum number of examples per shard",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable optimizations for multi-GPU systems",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        default=None,
        help="Comma-separated list of special tokens to add (e.g. '[INST],[/INST]')",
    )
    args = parser.parse_args()

    # Update global settings if provided in arguments
    if args.num_workers is not None:
        NUM_WORKERS_DEFAULT = args.num_workers
    if args.gpu_mem_fraction is not None and 0.0 < args.gpu_mem_fraction <= 1.0:
        GPU_MEM_FRACTION = args.gpu_mem_fraction
    if args.max_shard_size is not None:
        MAX_SHARD_SIZE = args.max_shard_size

    # Only print system info from the main process
    if LOCAL_RANK in [-1, 0]:
        print(
            f"Device: {DEVICE}, CPU cores: {NUM_CPU_CORES}, Workers: {NUM_WORKERS_DEFAULT}"
        )
        if torch.cuda.is_available():
            print(
                f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
            if torch.cuda.device_count() > 1:
                print(f"Multi-GPU setup: {torch.cuda.device_count()} GPUs available")

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            text_field=args.text_field,
            quiet=args.quiet,
        )
    elif args.stage == "train_vocab":
        dataset_name = (
            args.dataset_name.split("/")[-1]
            if "/" in args.dataset_name
            else args.dataset_name
        )
        # Parse special tokens from comma-separated string to list
        special_tokens = None
        if args.special_tokens:
            special_tokens = [token.strip() for token in args.special_tokens.split(",")]
        train_vocab(
            vocab_size=args.vocab_size,
            dataset_name=dataset_name,
            quiet=args.quiet,
            special_tokens=special_tokens,
        )
    elif args.stage == "pretokenize":
        dataset_name = (
            args.dataset_name.split("/")[-1]
            if "/" in args.dataset_name
            else args.dataset_name
        )
        pretokenize(
            vocab_size=args.vocab_size, dataset_name=dataset_name, quiet=args.quiet
        )
    elif args.stage == "fullclean":
        fullclean(quiet=args.quiet)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
