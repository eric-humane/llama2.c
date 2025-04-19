"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval


# Function to get overrides from command line arguments
def get_config_overrides(config_keys):
    allowed_keys = set(config_keys)
    overrides = {}

    for arg in sys.argv[1:]:
        if "=" not in arg:
            # assume it's the name of a config file
            if not arg.startswith("--"):
                config_file = arg
                print(f"Overriding config with {config_file}:")
                with open(config_file) as f:
                    print(f.read())
                exec(open(config_file).read(), globals())
        else:
            # assume it's a --key=value argument
            assert arg.startswith("--")
            key, val = arg.split("=")
            key = key[2:]
            if key in allowed_keys:
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok if the key already exists in globals
                if (
                    key in globals()
                    and type(globals()[key]) != type(attempt)
                    and globals()[key] is not None
                ):
                    print(
                        f"Warning: type mismatch for {key}. Expected {type(globals()[key])}, got {type(attempt)}"
                    )
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                overrides[key] = attempt
            else:
                print(f"Warning: Unknown config key: {key} - skipping")

    return overrides


# For backwards compatibility with direct execution
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if "=" not in arg:
            # assume it's the name of a config file
            if not arg.startswith("--"):
                config_file = arg
                print(f"Overriding config with {config_file}:")
                with open(config_file) as f:
                    print(f.read())
                exec(open(config_file).read())
        else:
            # assume it's a --key=value argument
            assert arg.startswith("--")
            key, val = arg.split("=")
            key = key[2:]
            if key in globals():
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(globals()[key])
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                globals()[key] = attempt
            else:
                print(f"Warning: Unknown config key: {key} - skipping")
