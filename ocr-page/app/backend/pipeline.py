# -*- coding: utf-8 -*-
"""抓图 → OCR → Excel 备份 → 写 PI 流水线。"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    ROOT_DIR,
    get_capture_tag,
    get_config,
    get_ocr_date_cfg,
    get_ocr_early_stop,
    get_ocr_early_stop_min_density,
    get_ocr_max_width,
    get_ocr_min_points,
    get_ocr_fix_decimal_cfg,
    get_ocr_sanitize_cfg,
    get_ocr_scale,
    get_ocr_table_crop,
    get_ocr_onnx_threads,
    get_pi_value_max,
    new_ocr_run_dir,
    resolve_ocr_workdir,
)
from .excel_export import export_ocr_excel
from .hikvision import build_camera_client
from .monitor_crop import prepare_monitor_work_image
from .ocr_bridge import extract_points_from_image, ocr_result_to_dict, row_tag_name
from .pi_client import get_pi_client

logger = logging.getLogger(__name__)


def _elapsed_ms(t0: float) -> int:
    return max(0, int(round((time.perf_counter() - t0) * 1000)))


def evaluate_ocr_point_count(
    point_count: int,
    min_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    有效点数与 ocr.min_points 比较。
    不足则本轮识别失败（切屏/熄屏/模糊等），调用方不应写 PI。
    """
    threshold = get_ocr_min_points() if min_points is None else max(0, int(min_points))
    n = int(point_count)
    ok = n >= threshold
    out: Dict[str, Any] = {
        "ok": ok,
        "point_count": n,
        "min_points": threshold,
    }
    if not ok:
        out["reason"] = "points_below_min"
        out["message"] = (
            f"OCR 有效点数 {n} < min_points={threshold}，本轮识别失败，不写 PI"
        )
        logger.warning(out["message"])
    return out


def capture_only() -> Dict[str, Any]:
    cfg = get_config()
    client = build_camera_client(cfg, ROOT_DIR)
    path = client.capture()
    return {
        "ok": True,
        "image_path": str(path),
        "url": client.picture_url,
        "size": path.stat().st_size,
    }


def _tag_prefix(cfg: Dict[str, Any]) -> str:
    return get_capture_tag(cfg)


def _copy_source(image_path: str, run_dir: Path, name: str = "00_source.jpg") -> str:
    import shutil

    src = Path(image_path)
    if not src.is_file():
        return ""
    if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        dst = run_dir / name
    else:
        dst = run_dir / f"00_source{src.suffix.lower()}"
    shutil.copy2(src, dst)
    return str(dst)


def _maybe_export_excel(
    ocr_dict: Dict[str, Any],
    image_path: str,
    row_labels: Optional[List[str]] = None,
) -> Optional[str]:
    try:
        labels = row_labels or list(ocr_dict.get("row_tags") or [])
        return export_ocr_excel(
            headers=list(ocr_dict.get("headers") or []),
            rows=[list(r) for r in (ocr_dict.get("rows") or [])],
            image_path=image_path,
            row_labels=labels or None,
        )
    except Exception:
        logger.exception("导出 Excel 失败")
        return None


def ocr_only(
    image_path: str,
    base_date: Optional[datetime] = None,
    *,
    trigger: str = "debug",
    persist: bool = True,
    upload_name: Optional[str] = None,
) -> Dict[str, Any]:
    from .run_store import persist_pipeline_result

    t_all = time.perf_counter()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = get_config()
    ocr_cfg = cfg.get("ocr", {})
    run_dir = new_ocr_run_dir("single", cfg)
    source_copy = _copy_source(image_path, run_dir)

    # 单屏同样先裁显示器：宽贴满、上下有环境时纠正日期/表区比例
    single_cfg = (cfg.get("capture") or {}).get("single") or {}
    dual_cfg = (cfg.get("capture") or {}).get("dual") or {}
    auto_monitor = single_cfg.get("auto_monitor")
    if auto_monitor is None:
        auto_monitor = dual_cfg.get("auto_monitor")
    if auto_monitor is None:
        auto_monitor = True
    work_path, monitor_meta, monitor_path = prepare_monitor_work_image(
        image_path, run_dir, enabled=bool(auto_monitor)
    )
    work_image = str(work_path)

    t_ocr = time.perf_counter()
    result = extract_points_from_image(
        image_path=work_image,
        workdir=resolve_ocr_workdir(cfg),
        scale=get_ocr_scale(cfg),
        min_confidence=float(ocr_cfg.get("min_confidence", 0.3)),
        crop=None,
        tag_prefix=_tag_prefix(cfg),
        base_date=base_date,
        debug_dir=run_dir,
        layout="single",
        sanitize=get_ocr_sanitize_cfg(cfg),
        fix_decimal=get_ocr_fix_decimal_cfg(cfg),
        early_stop=get_ocr_early_stop(cfg),
        early_stop_min_density=get_ocr_early_stop_min_density(cfg),
        max_ocr_width=get_ocr_max_width(cfg),
        date_cfg=get_ocr_date_cfg(cfg, layout="single"),
        table_crop=get_ocr_table_crop(cfg, layout="single"),
        onnx_threads=get_ocr_onnx_threads(cfg),
    )
    ocr_ms = _elapsed_ms(t_ocr)
    payload = {**ocr_result_to_dict(result)}
    payload["monitor"] = monitor_meta
    if monitor_path:
        payload["monitor_image"] = monitor_path
    if result.ok is False and (result.reason or "").startswith("date"):
        quality = {
            "ok": False,
            "point_count": 0,
            "min_points": get_ocr_min_points(cfg),
            "reason": result.reason or "date_ocr_failed",
            "message": result.message
            or "未识别到左上角日期筛选框，已跳过表格 OCR",
        }
        payload["ok"] = False
        payload["reason"] = quality["reason"]
        payload["message"] = quality["message"]
    else:
        quality = evaluate_ocr_point_count(len(result.points))
        payload["ok"] = bool(quality.get("ok"))
        if not payload["ok"]:
            payload["reason"] = quality.get("reason")
            payload["message"] = quality.get("message")
    payload["quality"] = quality
    if source_copy:
        payload["source_copy"] = source_copy
    payload["artifacts_dir"] = str(run_dir)
    excel_path = None
    if result.ok and result.points:
        excel_path = _maybe_export_excel(payload, image_path, result.row_tags)
        if excel_path:
            payload["excel_path"] = excel_path

    out: Dict[str, Any] = {
        "ok": payload["ok"],
        "image_path": image_path,
        "capture": {"image_path": image_path},
        "ocr": payload,
        "quality": quality,
        "point_count": quality.get("point_count"),
        "min_points": quality.get("min_points"),
        "pi": {"ok": None, "written": 0, "total": 0, "skipped": True, "reason": "ocr_only"},
        "artifacts_dir": str(run_dir),
        "monitor": monitor_meta,
        "timing": {
            "ocr_ms": ocr_ms,
            "monitor_ms": int(monitor_meta.get("ms") or 0),
            "date_ms": int((payload.get("timing") or {}).get("date_ms") or 0),
            "table_ms": int((payload.get("timing") or {}).get("table_ms") or 0),
            "total_ms": _elapsed_ms(t_all),
        },
    }
    if monitor_path:
        out["monitor_image"] = monitor_path
        out["work_image_path"] = work_image
    if not payload["ok"]:
        out["reason"] = payload.get("reason")
        out["message"] = payload.get("message")
    if source_copy:
        out["source_copy"] = source_copy
    if excel_path:
        out["excel_path"] = excel_path
    if upload_name:
        out["upload_name"] = str(upload_name).strip()
        out["uploaded"] = True
    # 保持原 API 扁平字段
    out.update({k: v for k, v in payload.items() if k not in out})
    if persist:
        persist_pipeline_result(
            out, mode="single", trigger=trigger, started_at=started_at
        )
    return out


def write_points_to_pi(
    points: List[Dict[str, Any]],
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    写入 PI：一行数据对应一个测点 前缀_1 … _N。
    按 timestamp 时间线升序依次写入（旧点先写，新点后写）。
    优先用点内 tag；否则用 tag 参数或 capture.single.tag 作前缀按 row_index 拼接。
    |value| 超过 pi.value_max 的点跳过不写（仍记入 items，不影响 OCR/前端展示）。
    """
    cfg = get_config()
    client = get_pi_client(cfg)
    client.connect()
    prefix = (tag or get_capture_tag(cfg)).strip()
    verify_write = bool((cfg.get("pi") or {}).get("verify_write", True))
    value_max = get_pi_value_max(cfg)

    ordered = sorted(
        list(points or []),
        key=lambda p: str((p or {}).get("timestamp") or ""),
    )

    results = []
    skipped_max = 0
    for p in ordered:
        label = p["time_label"]
        value = float(p["value"])
        ts = datetime.strptime(p["timestamp"], "%Y-%m-%d %H:%M:%S")
        point_tag = (p.get("tag") or "").strip()
        if not point_tag:
            row_i = int(p.get("row_index", 0))
            point_tag = row_tag_name(prefix, row_i)
        item: Dict[str, Any] = {
            "time_label": label,
            "tag": point_tag,
            "value": value,
            "timestamp": p["timestamp"],
            "row_index": p.get("row_index"),
        }
        if value_max is not None and abs(value) > value_max:
            skipped_max += 1
            item.update(
                {
                    "pi_code": None,
                    "ok": True,
                    "skipped": True,
                    "reason": "value_exceeds_max",
                    "message": (
                        f"|value|={abs(value):g} 超过 pi.value_max={value_max:g}，"
                        "跳过写入 PI（识别结果仍展示）"
                    ),
                }
            )
            logger.warning(
                "跳过写 PI：%s @ %s value=%s > value_max=%s",
                point_tag,
                p.get("timestamp"),
                value,
                value_max,
            )
            results.append(item)
            continue
        rc = client.put_value(point_tag, value, when=ts, verify=verify_write)
        item["pi_code"] = rc
        item["ok"] = rc == 0
        item["skipped"] = False
        if rc == -9001:
            item["message"] = (
                "PI API 返回成功但按写入时刻回读未确认（常见：早于测点建点时间，历史值未入库）"
            )
        elif rc != 0:
            item["message"] = f"piar_putvalue 失败 code={rc}"
        results.append(item)
    write_items = [r for r in results if not r.get("skipped")]
    ok_count = sum(1 for r in write_items if r.get("ok"))
    write_total = len(write_items)
    return {
        "ok": ok_count == write_total,
        "tag": prefix,
        "written": ok_count,
        "total": write_total,
        "skipped_max": skipped_max,
        "value_max": value_max,
        "items": results,
    }


def run_pipeline(
    base_date: Optional[datetime] = None,
    *,
    trigger: str = "manual",
) -> Dict[str, Any]:
    """一键：抓图 → OCR 整表 → Excel → 按行测点写 PI。"""
    from .run_store import persist_pipeline_result

    t_all = time.perf_counter()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = get_config()
    cam = build_camera_client(cfg, ROOT_DIR)
    t_cap = time.perf_counter()
    image_path = cam.capture()
    capture_ms = _elapsed_ms(t_cap)

    ocr_cfg = cfg.get("ocr", {})
    run_dir = new_ocr_run_dir("single", cfg)
    source_copy = _copy_source(str(image_path), run_dir)
    single_cfg = (cfg.get("capture") or {}).get("single") or {}
    dual_cfg = (cfg.get("capture") or {}).get("dual") or {}
    auto_monitor = single_cfg.get("auto_monitor")
    if auto_monitor is None:
        auto_monitor = dual_cfg.get("auto_monitor")
    if auto_monitor is None:
        auto_monitor = True
    work_path, monitor_meta, monitor_path = prepare_monitor_work_image(
        image_path, run_dir, enabled=bool(auto_monitor)
    )
    t_ocr = time.perf_counter()
    ocr = extract_points_from_image(
        image_path=str(work_path),
        workdir=resolve_ocr_workdir(cfg),
        scale=get_ocr_scale(cfg),
        min_confidence=float(ocr_cfg.get("min_confidence", 0.3)),
        crop=None,
        tag_prefix=_tag_prefix(cfg),
        base_date=base_date,
        debug_dir=run_dir,
        layout="single",
        sanitize=get_ocr_sanitize_cfg(cfg),
        fix_decimal=get_ocr_fix_decimal_cfg(cfg),
        early_stop=get_ocr_early_stop(cfg),
        early_stop_min_density=get_ocr_early_stop_min_density(cfg),
        max_ocr_width=get_ocr_max_width(cfg),
        date_cfg=get_ocr_date_cfg(cfg, layout="single"),
        table_crop=get_ocr_table_crop(cfg, layout="single"),
        onnx_threads=get_ocr_onnx_threads(cfg),
    )
    ocr_ms = _elapsed_ms(t_ocr)
    ocr_dict = ocr_result_to_dict(ocr)
    ocr_dict["monitor"] = monitor_meta
    if monitor_path:
        ocr_dict["monitor_image"] = monitor_path
    if source_copy:
        ocr_dict["source_copy"] = source_copy
    ocr_dict["artifacts_dir"] = str(run_dir)

    points_payload = [
        {
            "time_label": p.time_label,
            "value": p.value,
            "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": p.tag,
            "row_index": p.row_index,
        }
        for p in ocr.points
    ]
    if ocr.ok is False and (ocr.reason or "").startswith("date"):
        quality = {
            "ok": False,
            "point_count": 0,
            "min_points": get_ocr_min_points(cfg),
            "reason": ocr.reason or "date_ocr_failed",
            "message": ocr.message
            or "未识别到左上角日期筛选框，已跳过表格 OCR",
        }
        excel_path = None
    else:
        quality = evaluate_ocr_point_count(len(points_payload))
        excel_path = _maybe_export_excel(ocr_dict, str(image_path), ocr.row_tags)
    ocr_dict["quality"] = quality

    pi_ms = 0
    if not quality.get("ok"):
        write = {
            "ok": False,
            "written": 0,
            "total": 0,
            "items": [],
            "skipped": True,
            "reason": quality.get("reason"),
            "message": quality.get("message"),
            "point_count": quality.get("point_count"),
            "min_points": quality.get("min_points"),
        }
        out_ok = False
    else:
        t_pi = time.perf_counter()
        write = write_points_to_pi(points_payload)
        pi_ms = _elapsed_ms(t_pi)
        out_ok = bool(write.get("ok"))

    out: Dict[str, Any] = {
        "ok": out_ok,
        "capture": {"image_path": str(image_path), "size": Path(image_path).stat().st_size},
        "ocr": ocr_dict,
        "pi": write,
        "quality": quality,
        "point_count": quality.get("point_count"),
        "min_points": quality.get("min_points"),
        "artifacts_dir": str(run_dir),
        "monitor": monitor_meta,
        "timing": {
            "capture_ms": capture_ms,
            "ocr_ms": ocr_ms,
            "monitor_ms": int(monitor_meta.get("ms") or 0),
            "date_ms": int((ocr_dict.get("timing") or {}).get("date_ms") or 0),
            "table_ms": int((ocr_dict.get("timing") or {}).get("table_ms") or 0),
            "pi_ms": pi_ms,
            "total_ms": _elapsed_ms(t_all),
        },
    }
    if monitor_path:
        out["monitor_image"] = monitor_path
        out["work_image_path"] = str(work_path)
    if not quality.get("ok"):
        out["reason"] = quality.get("reason")
        out["message"] = quality.get("message")
    elif not out_ok:
        out["reason"] = "pi_write_failed"
        out["message"] = "写 PI 失败"
    if source_copy:
        out["source_copy"] = source_copy
    if excel_path:
        out["excel_path"] = excel_path
    return persist_pipeline_result(
        out, mode="single", trigger=trigger, started_at=started_at
    )


def curve(
    start: str,
    end: str,
    tag: Optional[str] = None,
    count: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = get_config()
    pi_cfg = cfg.get("pi", {})
    client = get_pi_client(cfg)
    client.connect()
    prefix = get_capture_tag(cfg)
    raw = (tag or "").strip()
    # 传入前缀时默认读 _1；也可显式传 GX.DEMO.OCR_2 / _3
    if not raw:
        point = row_tag_name(prefix, 0)
    elif raw.rsplit("_", 1)[-1].isdigit():
        point = raw
    else:
        point = row_tag_name(raw, 0)
    n = int(count or pi_cfg.get("curve_count", 96))
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    series = client.interp_values(point, start_dt, end_dt, n)
    return {
        "ok": True,
        "tag": point,
        "start": start,
        "end": end,
        "count": len(series),
        "points": series,
    }
