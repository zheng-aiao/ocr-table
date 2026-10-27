# -*- coding: utf-8 -*-
"""抓图 → OCR → Excel 备份 → 写 PI 流水线。"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    get_camera_table_search,
    get_capture_tag,
    get_config,
    get_ocr_max_width,
    get_ocr_min_points,
    get_ocr_fix_decimal_cfg,
    get_ocr_scale,
    get_ocr_onnx_threads,
    get_pi_value_max,
    new_ocr_run_dir,
    resolve_ocr_workdir,
)
from .excel_export import export_ocr_excel
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


def resolve_ocr_fail_fields(
    quality: Dict[str, Any],
    ocr: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    点数不足时的对外失败字段：优先蒙板/表格定位等具体原因，
    避免一律显示「有效点数 0 < min_points」。
    """
    ocr = ocr or {}
    crop_msg = str(ocr.get("table_crop_fallback_message") or "").strip()
    crop_reason = str(ocr.get("table_crop_fallback_reason") or "").strip()
    ocr_msg = str(ocr.get("message") or "").strip()
    ocr_reason = str(ocr.get("reason") or "").strip()
    q_msg = str(quality.get("message") or "").strip()
    q_reason = str(quality.get("reason") or "points_below_min").strip()

    def _is_generic_points(msg: str) -> bool:
        return ("min_points" in msg) or msg.startswith("OCR 有效点数")

    # 1) 表格区域/蒙板：最具体
    if crop_msg:
        return {
            "reason": crop_reason or ocr_reason or "table_crop_failed",
            "message": crop_msg,
        }
    # 2) OCR 引擎已给出非点数类原因
    if ocr_msg and not _is_generic_points(ocr_msg):
        return {
            "reason": ocr_reason or q_reason or "table_ocr_failed",
            "message": ocr_msg,
        }
    # 3) 退回点数门槛说明
    return {
        "reason": q_reason or "points_below_min",
        "message": q_msg
        or ocr_msg
        or "OCR 有效点数不足，本轮识别失败，不写 PI",
    }


def capture_only() -> Dict[str, Any]:
    """调试：并行抓两路摄像头。识别直接用整图，不预裁显示器有效区域。"""
    from .pipeline_dual import capture_dual_cameras, get_camera_defs
    cfg = get_config()
    cams = get_camera_defs(cfg)
    captured = capture_dual_cameras(cfg)

    sides_out: Dict[str, Any] = {}
    for side in ("left", "right"):
        cam = cams[side]
        src = captured.get(side)
        sides_out[side] = {
            "image_path": str(src) if src else "",
            "monitor_path": "",
            "work_image_path": str(src) if src else "",
            "host": cam.get("host"),
            "type": cam.get("type"),
            "tag": cam.get("tag"),
            "size": src.stat().st_size if src else 0,
            "monitor": {"ok": True, "mode": "skipped", "enabled": False},
        }

    return {
        "ok": True,
        "capture_mode": "double_camera",
        "left": sides_out["left"],
        "right": sides_out["right"],
        "left_image": sides_out["left"].get("image_path") or "",
        "right_image": sides_out["right"].get("image_path") or "",
        "image_path": sides_out["left"].get("image_path")
        or sides_out["right"].get("image_path")
        or "",
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


def _normalize_camera_side(side: Optional[str]) -> str:
    """联调单侧上传：left|right；其它值默认 left（兼容旧「单图」）。"""
    key = str(side or "").strip().lower()
    return key if key in ("left", "right") else "left"


def ocr_only(
    image_path: str,
    base_date: Optional[datetime] = None,
    *,
    trigger: str = "debug",
    persist: bool = True,
    upload_name: Optional[str] = None,
    side: Optional[str] = None,
) -> Dict[str, Any]:
    from .run_store import persist_pipeline_result
    from .pipeline_dual import get_camera_defs
    from .camera_common import cam_log_prefix

    t_all = time.perf_counter()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = get_config()
    ocr_cfg = cfg.get("ocr", {})
    cam_side = _normalize_camera_side(side)
    cams = get_camera_defs(cfg)
    cam = cams.get(cam_side) or {}
    tag_prefix = str(cam.get("tag") or "").strip() or _tag_prefix(cfg)
    search_crop = get_camera_table_search(cam_side, cfg)
    run_dir = new_ocr_run_dir("single", cfg)
    source_copy = _copy_source(image_path, run_dir)
    work_image = source_copy or image_path
    monitor_meta = {"ok": True, "mode": "skipped", "enabled": False}
    monitor_path = ""
    pfx = cam_log_prefix(cam_side)
    search_desc = (
        ",".join(f"{v:.3f}" for v in search_crop)
        if search_crop is not None
        else "整图"
    )
    logger.info(
        "%s单侧上传识别 side=%s search=%s tag=%s（auto 失败即止）",
        pfx,
        cam_side,
        search_desc,
        tag_prefix,
    )

    t_ocr = time.perf_counter()
    result = extract_points_from_image(
        image_path=work_image,
        workdir=resolve_ocr_workdir(cfg),
        scale=get_ocr_scale(cfg),
        min_confidence=float(ocr_cfg.get("min_confidence", 0.3)),
        tag_prefix=tag_prefix,
        debug_dir=run_dir,
        layout="double",
        fix_decimal=get_ocr_fix_decimal_cfg(cfg),
        max_ocr_width=get_ocr_max_width(cfg),
        onnx_threads=get_ocr_onnx_threads(cfg),
        log_prefix=pfx,
        search_crop=search_crop,
        camera_side=cam_side,
    )
    ocr_ms = _elapsed_ms(t_ocr)
    payload = {**ocr_result_to_dict(result)}
    payload["monitor"] = monitor_meta
    if monitor_path:
        payload["monitor_image"] = monitor_path
    quality = evaluate_ocr_point_count(len(result.points))
    payload["ok"] = bool(quality.get("ok"))
    if not payload["ok"]:
        fail = resolve_ocr_fail_fields(quality, payload)
        payload["reason"] = fail["reason"]
        payload["message"] = fail["message"]
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
    out["camera_side"] = cam_side
    out["table_search"] = search_desc if search_crop is not None else ""
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
    写入 PI：一行数据对应一个测点 前缀.01 … .0N。
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
    """本分支固定双摄流水线。"""
    del base_date
    from .pipeline_dual import run_pipeline_dual

    return run_pipeline_dual(trigger=trigger)


def curve(
    start: str,
    end: str,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """按测点读取 PI 档案中的时间与值，有多少返回多少。"""
    cfg = get_config()
    client = get_pi_client(cfg)
    client.connect()
    prefix = get_capture_tag(cfg)
    raw = (tag or "").strip()
    # 传入前缀时默认读 .01；已带 .01/.02 或旧式 _1 的完整测点名则原样查询
    if not raw:
        point = row_tag_name(prefix, 0)
    elif raw.rsplit(".", 1)[-1].isdigit() or raw.rsplit("_", 1)[-1].isdigit():
        point = raw
    else:
        point = row_tag_name(raw, 0)
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    series = client.archive_values(point, start_dt, end_dt)
    return {
        "ok": True,
        "tag": point,
        "start": start,
        "end": end,
        "count": len(series),
        "points": series,
    }
