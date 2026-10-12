# -*- coding: utf-8 -*-
"""
PI 最小联调脚本：连接 → 写测试点 → 读快照。

用法（先改 config.yaml 中 pi.* 连接与 capture.single.tag 测点）:
  python scripts/pi_probe.py
  python scripts/pi_probe.py --tag YOUR.TAG --value 123.45
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_capture_tag, load_config  # noqa: E402
from app.pi_client import get_pi_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="PI ctypes 最小联调")
    parser.add_argument("--tag", help="测点名，默认读 config capture.single.tag")
    parser.add_argument("--value", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config()
    tag = args.tag or get_capture_tag(cfg)
    client = get_pi_client(cfg)
    print("DLL:", Path(cfg["pi"]["dll_dir"]) / cfg["pi"].get("dll_name", "piapi.dll"))
    print("Server:", cfg["pi"]["server"], "User:", cfg["pi"]["user"])
    print("连接中...")
    client.connect()
    print("连接成功")
    now = datetime.now()
    print(f"写入 tag={tag} value={args.value} time={now}")
    rc = client.put_value(tag, args.value, when=now)
    print("put_value code:", rc)
    snap = client.get_snapshot(tag)
    print("snapshot value/istat/time:", snap[0], snap[1], snap[2])
    print("OK" if rc == 0 else "写入返回非 0，请检查测点权限与 DLL")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
