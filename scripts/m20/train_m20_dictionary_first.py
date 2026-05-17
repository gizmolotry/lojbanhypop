from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_m20_dictionary import parse_args, run_train  # noqa: E402


if __name__ == "__main__":
    run_train(parse_args())
