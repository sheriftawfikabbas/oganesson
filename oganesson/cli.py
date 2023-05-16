import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, TextIO, TypeVar, Union
from . import VERSION


class CLI:
    def __init__(self, argv: Optional[str] = None) -> None:
        self.argv = argv or sys.argv[:]
        self.prog_name = Path(self.argv[0]).name
        self.formatter_class = argparse.RawDescriptionHelpFormatter
        self.epilog = 'Epilogue'
        self.version = VERSION

        parser = argparse.ArgumentParser(
            prog=self.prog_name,
            description=f"{self.prog_name} version {self.version}",
            epilog=self.epilog,
            formatter_class=self.formatter_class,
        )

        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {self.version}"
        )

        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="verbosity"
        )

        arguments = parser.parse_args(self.argv[1:])
        random.seed(arguments.seed)


def execute_cli(argv: Optional[str] = None) -> None:
    cli = CLI()


if __name__ == "__main__":
    execute_cli()
