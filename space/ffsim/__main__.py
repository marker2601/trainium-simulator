"""python -m ffsim <subcommand> ... (see ffsim/cli.py)."""
import sys

from ffsim.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
