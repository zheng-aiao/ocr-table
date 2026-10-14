# -*- coding: utf-8 -*-
"""OCR 结果导出 Excel（开发对照备份；写 PI 不依赖本文件）。"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ROOT_DIR, get_config, resolve_project_path

logger = logging.getLogger(__name__)


def _resolve_export_dir(cfg: Dict[str, Any], when: Optional[datetime] = None) -> Path:
    """data/exports 根目录下按日期分区，与 data/captures/YYYYMMDD 一致。"""
    excel = cfg.get("excel", {}) or {}
    raw = excel.get("dir", "data/exports")
    p = resolve_project_path(raw)
    day = (when or datetime.now()).strftime("%Y%m%d")
    out = p / day
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_ocr_excel(
    headers: List[str],
    rows: List[List[str]],
    *,
    image_path: Optional[str] = None,
    row_labels: Optional[List[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    将表头 + 多行写入 xlsx（路径：data/exports/YYYYMMDD/…）。excel.enabled=false 时跳过。
    """
    cfg = cfg or get_config()
    excel = cfg.get("excel", {}) or {}
    if not excel.get("enabled", False):
        logger.info("excel.enabled=false，跳过导出")
        return None

    try:
        from openpyxl import Workbook
    except ImportError as e:
        raise RuntimeError("写 Excel 需要 openpyxl，请 pip install openpyxl") from e

    now = datetime.now()
    out_dir = _resolve_export_dir(cfg, when=now)
    stem = Path(image_path).stem if image_path else "ocr"
    ts = now.strftime("%Y%m%d_%H%M%S")
    pattern = str(excel.get("filename") or "{stem}_{ts}.xlsx")
    name = pattern.format(stem=stem, ts=ts)
    path = out_dir / name

    labels = row_labels or [f"指标{i + 1}" for i in range(len(rows))]
    wb = Workbook()
    ws = wb.active
    ws.title = "OCR"
    ws.append(["指标"] + list(headers))
    for i, row in enumerate(rows):
        label = labels[i] if i < len(labels) else f"指标{i + 1}"
        cells = [row[j] if j < len(row) else "" for j in range(len(headers))]
        ws.append([label] + cells)
    wb.save(path)
    logger.info("OCR Excel 已保存: %s", path)
    return str(path)
