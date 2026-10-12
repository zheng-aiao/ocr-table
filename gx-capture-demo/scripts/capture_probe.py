# -*- coding: utf-8 -*-
"""仅测海康 ISAPI 抓图。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.pipeline import capture_only  # noqa: E402


def main() -> int:
    load_config()
    result = capture_only()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
