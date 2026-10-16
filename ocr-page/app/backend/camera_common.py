# -*- coding: utf-8 -*-
"""相机抓图共用：异常与落盘文件名。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


class HikvisionCaptureError(RuntimeError):
    """抓图失败（ISAPI / MVS 共用）。"""


def cam_log_prefix(side: str = "") -> str:
    """统一摄像头日志前缀，如 ``[left] `` / ``[right] `` / ``[single] ``。"""
    s = str(side or "").strip().lower()
    if not s:
        return ""
    return f"[{s}] "


def format_capture_stem(when: Optional[datetime] = None) -> str:
    """
    抓图文件名主干：H-MM（如 9-09、10-20）。
    Windows 文件名不能含 ':'，用 '-' 代替；日期由外层 YYYYMMDD 目录表达。
    """
    now = when or datetime.now()
    return f"{now.hour}-{now.minute:02d}"


def unique_time_path(
    out_dir: Path,
    when: Optional[datetime] = None,
    *,
    label: str = "",
) -> Path:
    """生成不冲突的抓图路径：优先 9-09.jpg，同分钟重复则 9-09_2.jpg …

    label 非空时插入到主干后：9-09_left.jpg / 9-09_61.jpg，便于双摄并行落盘。
    """
    stem = format_capture_stem(when)
    if label:
        safe = str(label).strip().replace(" ", "_")
        if safe:
            stem = f"{stem}_{safe}"
    path = out_dir / f"{stem}.jpg"
    if not path.exists():
        return path
    n = 2
    while True:
        cand = out_dir / f"{stem}_{n}.jpg"
        if not cand.exists():
            return cand
        n += 1
