"""Allow running the package as: python -m jodic"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from main import main

if __name__ == "__main__":
    main()
