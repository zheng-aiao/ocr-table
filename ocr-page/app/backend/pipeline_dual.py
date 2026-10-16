# -*- coding: utf-8 -*-
"""双摄像头流水线（double-camera）：两路并行抓拍 → 处理 → 写 PI。

本分支只服务双摄：
- cameras.left / cameras.right：各自 type（mvs|isapi）、连接参数、写 PI 的 tag 前缀
- capture.double.left / right：table_crop（表格固定框）、date_crop（日期搜索带）
- 流程：有效区域 → 纠偏 → 日期 → 表格识别
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from .camera_common import cam_log_prefix
from .config import (
    ROOT_DIR,
    get_config,
    get_auto_monitor,
    get_ocr_date_cfg,
    get_ocr_auto_enable,
    get_ocr_max_width,
    get_ocr_min_points,
    get_ocr_parallel_sides,
    get_ocr_fix_decimal_cfg,
    get_ocr_scale,
    get_ocr_table_crop,
    get_ocr_onnx_threads,
    new_ocr_run_dir,
    resolve_ocr_workdir,
)
from .excel_export import export_ocr_excel
from .hikvision import (
    HikvisionCaptureError,
    build_camera_client_for_side,
)
from .monitor_crop import prepare_monitor_work_image
from .ocr_bridge import extract_points_from_image, ocr_result_to_dict
from .pipeline import _elapsed_ms, evaluate_ocr_point_count, write_points_to_pi

logger = logging.getLogger(__name__)

DEFAULT_CAMERA: Dict[str, Dict[str, Any]] = {
    "left": {"type": "isapi", "host": "", "tag": "GX.DEMO.OCR.L"},
    "right": {"type": "mvs", "host": "", "tag": "GX.DEMO.OCR.R"},
}

DEFAULT_DOUBLE_CROP: Dict[str, Dict[str, Any]] = {
    "left": {
        "table_crop": "0.02,0.52,0.97,0.90",
        "date_crop": "0.02,0.02,0.55,0.34",
    },
    "right": {
        "table_crop": "0.03,0.52,1.0,0.90",
        "date_crop": "0.02,0.02,0.55,0.34",
    },
}


def get_capture_mode(cfg: Optional[Dict[str, Any]] = None) -> str:
    """本分支固定双摄。"""
    del cfg
    return "double"


def get_camera_defs(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """cameras.left / cameras.right（type、连接、tag）。"""
    cfg = cfg or get_config()
    raw = cfg.get("cameras") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for side in ("left", "right"):
        base = dict(DEFAULT_CAMERA[side])
        base.update(raw.get(side) or {})
        out[side] = base
    return out


def get_double_defs(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """capture.double：table_crop / date_crop（搜索带）；附带 cameras 的 tag/host。"""
    cfg = cfg or get_config()
    cap = cfg.get("capture") or {}
    raw = cap.get("double") or cap.get("dual") or {}
    cams = get_camera_defs(cfg)
    out: Dict[str, Dict[str, Any]] = {}
    for side in ("left", "right"):
        crop = dict(DEFAULT_DOUBLE_CROP[side])
        side_raw = raw.get(side) or {}
        if side_raw.get("table_crop") not in (None, ""):
            crop["table_crop"] = side_raw["table_crop"]
        if side_raw.get("date_crop") not in (None, ""):
            crop["date_crop"] = side_raw["date_crop"]
        cam = cams[side]
        out[side] = {
            **crop,
            "tag": cam.get("tag") or DEFAULT_CAMERA[side]["tag"],
            "host": cam.get("host") or "",
            "type": cam.get("type") or "",
            "id": side,
            "title": side,
        }
    return out


def get_dual_defs(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """兼容旧名，等同 get_double_defs。"""
    return get_double_defs(cfg)


def get_screen_defs(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    return get_double_defs(cfg)


def capture_dual_cameras(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """并行抓拍左右两路摄像头，返回 {left: Path, right: Path}（可能缺侧）。"""
    cfg = cfg or get_config()
    cams = get_camera_defs(cfg)
    clients = {
        side: build_camera_client_for_side(cfg, ROOT_DIR, side, cams[side])
        for side in ("left", "right")
    }
    out: Dict[str, Path] = {}
    errors: Dict[str, str] = {}

    def _one(side: str) -> Tuple[str, Path]:
        return side, Path(clients[side].capture())

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_one, side): side for side in ("left", "right")}
        for fut in as_completed(futs):
            side = futs[fut]
            try:
                s, path = fut.result()
                out[s] = path
            except Exception as e:
                errors[side] = str(e)
                logger.exception(
                    "%s抓图失败 type=%s host=%s",
                    cam_log_prefix(side),
                    cams[side].get("type"),
                    cams[side].get("host"),
                )

    if not out:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items()) or "未知错误"
        raise HikvisionCaptureError(f"双摄抓图全部失败 — {detail}")
    if errors:
        for side, err in errors.items():
            logger.warning("%s抓图失败 %s", cam_log_prefix(side), err)
    return out


def _ocr_one_side(
    *,
    pane_image: Path,
    tag_prefix: str,
    ocr_cfg: Dict[str, Any],
    workdir: Path,
    base_date: Optional[datetime],
    export_excel: bool = True,
    debug_dir: Optional[Path] = None,
    fix_decimal: Optional[Dict[str, Any]] = None,
    date_cfg: Optional[Dict[str, Any]] = None,
    table_crop: Optional[Tuple[float, float, float, float]] = None,
    side: str = "",
) -> Dict[str, Any]:
    """对一侧整屏图做 OCR（双摄 layout=double）。"""
    del base_date
    pfx = cam_log_prefix(side)
    try:
        scale = float(ocr_cfg.get("scale", get_ocr_scale()))
    except (TypeError, ValueError):
        scale = get_ocr_scale()
    max_w = ocr_cfg.get("max_ocr_width", None)
    if max_w in (None, ""):
        max_ocr_width = get_ocr_max_width()
    else:
        try:
            max_ocr_width = int(max_w)
        except (TypeError, ValueError):
            max_ocr_width = get_ocr_max_width()
    result = extract_points_from_image(
        image_path=str(pane_image),
        workdir=workdir,
        scale=scale,
        min_confidence=float(ocr_cfg.get("min_confidence", 0.3)),
        crop=None,
        tag_prefix=tag_prefix,
        base_date=None,
        debug_dir=debug_dir,
        layout="double",
        fix_decimal=(
            fix_decimal if fix_decimal is not None else get_ocr_fix_decimal_cfg()
        ),
        auto_enable=bool(ocr_cfg.get("auto_enable", get_ocr_auto_enable())),
        max_ocr_width=max_ocr_width,
        date_cfg=date_cfg
        if date_cfg is not None
        else get_ocr_date_cfg(layout="double", side=side or "left"),
        table_crop=table_crop,
        onnx_threads=int(ocr_cfg.get("onnx_threads", get_ocr_onnx_threads())),
        log_prefix=pfx,
    )
    payload = ocr_result_to_dict(result)
    if export_excel and result.ok and result.points:
        try:
            excel_path = export_ocr_excel(
                headers=list(payload.get("headers") or []),
                rows=[list(r) for r in (payload.get("rows") or [])],
                image_path=str(pane_image),
                row_labels=list(payload.get("row_tags") or []) or None,
            )
            if excel_path:
                payload["excel_path"] = excel_path
        except Exception:
            logger.exception("%sExcel 导出失败 image=%s", pfx, pane_image)
    return payload


def _points_payload(ocr_dict: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    """点列表；附带 side，便于前端区分，写 PI 仍只认 tag/timestamp。"""
    return [
        {
            "time_label": p["time_label"],
            "value": p["value"],
            "timestamp": p["timestamp"],
            "tag": p.get("tag") or "",
            "row_index": p.get("row_index"),
            "side": side,
        }
        for p in (ocr_dict.get("points") or [])
    ]


def _side_camera_failed(
    side: str,
    sc: Dict[str, Any],
    *,
    message: str,
    min_points: int,
) -> Dict[str, Any]:
    tag_prefix = str(sc.get("tag") or DEFAULT_CAMERA[side]["tag"]).strip()
    quality = {
        "ok": False,
        "point_count": 0,
        "min_points": min_points,
        "reason": "camera_failed",
        "message": message,
    }
    return {
        "ok": False,
        "id": side,
        "title": side,
        "tag": tag_prefix,
        "host": str(sc.get("host") or ""),
        "type": str(sc.get("type") or ""),
        "base_date": None,
        "date_source": "failed",
        "date_ocr": {},
        "image_path": "",
        "ocr": {"ok": False, "reason": "camera_failed", "message": message},
        "points": [],
        "point_count": 0,
        "ocr_filled": 0,
        "quality": quality,
        "min_points": min_points,
        "reason": "camera_failed",
        "message": message,
        "ocr_ms": 0,
        "date_ms": 0,
        "table_ms": 0,
    }


def ocr_side_images(
    left_path: Optional[str],
    right_path: Optional[str],
    *,
    write_pi: bool = False,
    export_excel: bool = True,
    trigger: str = "manual",
    persist: bool = True,
    capture_errors: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    对左右两路已抓到的整屏图分别 OCR（可选写 PI）。
    无整图分缝；可选各侧 auto_monitor。
    """
    t_all = time.perf_counter()
    cfg = get_config()
    screens = get_double_defs(cfg)
    cams = get_camera_defs(cfg)
    ocr_cfg = cfg.get("ocr") or {}
    workdir = resolve_ocr_workdir(cfg)
    run_dir = new_ocr_run_dir("double", cfg)
    auto_monitor = get_auto_monitor(cfg)
    min_points = get_ocr_min_points(cfg)
    capture_errors = dict(capture_errors or {})

    path_by_side: Dict[str, Optional[Path]] = {
        "left": Path(left_path) if left_path else None,
        "right": Path(right_path) if right_path else None,
    }
    for side, p in list(path_by_side.items()):
        if p is not None and not p.is_file():
            capture_errors.setdefault(side, f"图片不存在: {p}")
            path_by_side[side] = None

    sides: Dict[str, Any] = {}
    all_points: List[Dict[str, Any]] = []
    write_points: List[Dict[str, Any]] = []
    side_ocr_ms: Dict[str, int] = {}
    monitors: Dict[str, Any] = {}

    t_ocr_all = time.perf_counter()
    side_jobs: List[Dict[str, Any]] = []

    for side in ("left", "right"):
        sc = screens[side]
        cam = cams[side]
        src = path_by_side[side]
        if src is None:
            msg = capture_errors.get(side) or f"{side} 摄像头未抓到图"
            logger.warning("%s跳过识别：%s", cam_log_prefix(side), msg)
            sides[side] = _side_camera_failed(
                side, {**sc, **cam}, message=msg, min_points=min_points
            )
            side_ocr_ms[side] = 0
            continue

        pane_path = run_dir / f"{side}.jpg"
        try:
            shutil.copy2(src, pane_path)
        except Exception:
            logger.exception("%s复制侧图失败 src=%s", cam_log_prefix(side), src)
            pane_path = src

        side_debug = run_dir / side
        side_debug.mkdir(parents=True, exist_ok=True)
        work_path, monitor_meta, monitor_path = prepare_monitor_work_image(
            pane_path,
            side_debug,
            enabled=bool(auto_monitor),
            log_prefix=cam_log_prefix(side),
        )
        monitors[side] = {
            **(monitor_meta or {}),
            "monitor_image": monitor_path,
            "source": str(src),
        }
        tag_prefix = str(cam.get("tag") or DEFAULT_CAMERA[side]["tag"]).strip()
        side_jobs.append(
            {
                "side": side,
                "sc": sc,
                "cam": cam,
                "tag_prefix": tag_prefix,
                "pane_path": work_path,
                "source_path": str(src),
                "side_debug": side_debug,
            }
        )

    def _run_side_job(job: Dict[str, Any]) -> Tuple[str, Dict[str, Any], int]:
        side = job["side"]
        pfx = cam_log_prefix(side)
        logger.info("%s开始识别", pfx)
        t_side = time.perf_counter()
        ocr_dict = _ocr_one_side(
            pane_image=job["pane_path"],
            tag_prefix=job["tag_prefix"],
            ocr_cfg=ocr_cfg,
            workdir=workdir,
            base_date=None,
            export_excel=export_excel,
            debug_dir=job["side_debug"],
            fix_decimal=get_ocr_fix_decimal_cfg(cfg),
            date_cfg=get_ocr_date_cfg(cfg, layout="double", side=side),
            table_crop=get_ocr_table_crop(cfg, layout="double", side=side),
            side=side,
        )
        ms = _elapsed_ms(t_side)
        date_str = ocr_dict.get("base_date") or "-"
        pts_n = len(ocr_dict.get("points") or [])
        timing = ocr_dict.get("timing") or {}
        logger.info(
            "%s识别结束 ms=%s date_ms=%s table_ms=%s ok=%s date=%s points=%s",
            pfx,
            ms,
            timing.get("date_ms", 0),
            timing.get("table_ms", 0),
            bool(ocr_dict.get("ok")),
            date_str,
            pts_n,
        )
        return side, ocr_dict, ms

    ocr_by_side: Dict[str, Dict[str, Any]] = {}
    parallel = get_ocr_parallel_sides(cfg) and len(side_jobs) > 1
    if parallel:
        logger.info("双摄 OCR 并行 left|right")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = {pool.submit(_run_side_job, job): job["side"] for job in side_jobs}
            for fut in as_completed(futs):
                side, ocr_dict, ms = fut.result()
                ocr_by_side[side] = ocr_dict
                side_ocr_ms[side] = ms
    else:
        for job in side_jobs:
            side, ocr_dict, ms = _run_side_job(job)
            ocr_by_side[side] = ocr_dict
            side_ocr_ms[side] = ms

    for job in side_jobs:
        side = job["side"]
        sc = job["sc"]
        tag_prefix = job["tag_prefix"]
        pane_path = job["pane_path"]
        side_debug = job["side_debug"]
        ocr_dict = ocr_by_side[side]
        pts = _points_payload(ocr_dict, side)
        pfx = cam_log_prefix(side)
        date_failed = str(ocr_dict.get("date_source") or "") in ("failed", "disabled") or (
            ocr_dict.get("ok") is False
            and str(ocr_dict.get("reason") or "").startswith("date")
        )
        if date_failed:
            quality = {
                "ok": False,
                "point_count": 0,
                "min_points": min_points,
                "reason": ocr_dict.get("reason") or "date_ocr_failed",
                "message": ocr_dict.get("message")
                or "未识别到左上角日期筛选框，已跳过表格 OCR",
            }
            side_ok = False
            logger.warning(
                "%s跳过写 PI：日期识别失败 — %s",
                pfx,
                quality.get("message"),
            )
        else:
            quality = evaluate_ocr_point_count(len(pts), min_points)
            side_ok = bool(quality.get("ok"))
            if not side_ok:
                logger.warning(
                    "%s跳过写 PI：有效点数 %s < min_points=%s",
                    pfx,
                    quality.get("point_count"),
                    quality.get("min_points"),
                )
        ocr_dict["quality"] = quality
        all_points.extend(pts)
        if side_ok:
            write_points.extend(pts)
        cam = job.get("cam") or cams[side]
        side_info: Dict[str, Any] = {
            "ok": side_ok,
            "id": side,
            "title": side,
            "tag": tag_prefix,
            "host": str(cam.get("host") or ""),
            "type": str(cam.get("type") or ""),
            "base_date": ocr_dict.get("base_date"),
            "date_source": ocr_dict.get("date_source"),
            "date_ocr": ocr_dict.get("date_ocr") or {},
            "image_path": str(pane_path),
            "source_path": job.get("source_path"),
            "ocr": ocr_dict,
            "points": pts,
            "point_count": len(pts),
            "ocr_filled": int(ocr_dict.get("ocr_filled") or 0),
            "quality": quality,
            "min_points": min_points,
            "artifacts_dir": str(side_debug),
            "ocr_ms": side_ocr_ms[side],
            "date_ms": int((ocr_dict.get("timing") or {}).get("date_ms") or 0),
            "table_ms": int((ocr_dict.get("timing") or {}).get("table_ms") or 0),
            "monitor": monitors.get(side),
        }
        if not side_ok:
            side_info["reason"] = quality.get("reason") or ocr_dict.get("reason")
            side_info["message"] = quality.get("message") or ocr_dict.get("message")
        sides[side] = side_info

    ocr_ms = _elapsed_ms(t_ocr_all)
    left_ok = bool(sides["left"].get("ok"))
    right_ok = bool(sides["right"].get("ok"))
    out: Dict[str, Any] = {
        "ok": left_ok or right_ok,
        "mode": "double",
        "capture_mode": "double_camera",
        "left": sides["left"],
        "right": sides["right"],
        "left_ok": left_ok,
        "right_ok": right_ok,
        "points": all_points,
        "min_points": min_points,
        "artifacts_dir": str(run_dir),
        "timing": {
            "ocr_ms": ocr_ms,
            "left_ocr_ms": side_ocr_ms.get("left", 0),
            "right_ocr_ms": side_ocr_ms.get("right", 0),
            "left_date_ms": int(sides["left"].get("date_ms") or 0),
            "left_table_ms": int(sides["left"].get("table_ms") or 0),
            "right_date_ms": int(sides["right"].get("date_ms") or 0),
            "right_table_ms": int(sides["right"].get("table_ms") or 0),
            "pi_ms": 0,
            "total_ms": 0,
        },
    }
    if path_by_side["left"]:
        out["left_image"] = str(path_by_side["left"])
    if path_by_side["right"]:
        out["right_image"] = str(path_by_side["right"])
    if not left_ok and not right_ok:
        out["reason"] = "points_below_min"
        out["message"] = (
            f"左右屏有效点数均低于 min_points={min_points} 或抓拍失败，本轮不写 PI"
        )

    if write_pi:
        if write_points:
            t_pi = time.perf_counter()
            write = write_points_to_pi(write_points, tag=None)
            out["timing"]["pi_ms"] = _elapsed_ms(t_pi)
            write["sides_written"] = [
                s for s in ("left", "right") if sides[s].get("ok")
            ]
            write["sides_skipped"] = [
                s for s in ("left", "right") if not sides[s].get("ok")
            ]
        else:
            write = {
                "ok": False,
                "written": 0,
                "total": 0,
                "items": [],
                "skipped": True,
                "reason": "points_below_min",
                "message": out.get("message")
                or "左右屏 OCR 均未达到 min_points，不写 PI",
                "sides_written": [],
                "sides_skipped": ["left", "right"],
            }
        out["pi"] = write
        if write_points:
            out["ok"] = bool(write.get("ok")) and (left_ok or right_ok)
            if not write.get("ok"):
                out["reason"] = "pi_write_failed"
                out["message"] = "写 PI 失败"
        else:
            out["ok"] = False
    else:
        out["pi"] = {
            "ok": None,
            "written": 0,
            "total": 0,
            "skipped": True,
            "reason": "ocr_only",
            "message": None,
            "sides_written": [],
            "sides_skipped": ["left", "right"],
        }

    if left_ok != right_ok:
        out["reason"] = "partial_side_failed"
        fail_side = "左侧" if not left_ok else "右侧"
        ok_side = "右侧" if not left_ok else "左侧"
        fail_n = int(sides["left" if not left_ok else "right"].get("point_count") or 0)
        fail_detail = (
            sides["left" if not left_ok else "right"].get("message")
            or f"有效点数 {fail_n} < min_points={min_points}"
        )
        if write_pi:
            sides_written = (out.get("pi") or {}).get("sides_written") or []
            ok_key = "right" if not left_ok else "left"
            did_write = ok_key in sides_written and bool((out.get("pi") or {}).get("ok"))
            out["message"] = (
                f"{fail_side}未达标（{fail_detail}），该侧未写 PI；"
                + (
                    f"{ok_side}已达标并写入"
                    if did_write
                    else f"{ok_side}已达标但写 PI 未成功"
                )
            )
        else:
            out["message"] = f"{fail_side}未达标（{fail_detail}）；{ok_side}识别达标"

    out["timing"]["total_ms"] = _elapsed_ms(t_all)
    if persist:
        from .run_store import persist_pipeline_result

        return persist_pipeline_result(
            out, mode="double", trigger=trigger, started_at=None
        )
    return out


def split_and_ocr(
    image_path: str,
    *,
    base_date: Optional[datetime] = None,
    left_date: Optional[datetime] = None,
    right_date: Optional[datetime] = None,
    write_pi: bool = False,
    export_excel: bool = True,
    trigger: str = "manual",
    persist: bool = True,
    upload_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    兼容旧上传接口：若只给一张图，则左右都用同一张做 OCR（不再分缝）。
    正常抓拍请用 run_pipeline_dual / ocr_side_images。
    """
    del base_date, left_date, right_date
    path = str(image_path)
    out = ocr_side_images(
        path,
        path,
        write_pi=write_pi,
        export_excel=export_excel,
        trigger=trigger,
        persist=persist,
    )
    out["image_path"] = path
    out["legacy_single_upload"] = True
    if upload_name:
        out["upload_name"] = str(upload_name).strip()
        out["uploaded"] = True
    return out


def run_pipeline_dual(
    base_date: Optional[datetime] = None,
    left_date: Optional[datetime] = None,
    right_date: Optional[datetime] = None,
    *,
    trigger: str = "manual",
    do_ocr: bool = True,
    do_pi: bool = True,
) -> Dict[str, Any]:
    """一键双摄：并行抓两路 → 各自 OCR → 按点内 tag 写 PI。"""
    del base_date, left_date, right_date
    from datetime import datetime as _dt

    from .run_store import persist_pipeline_result

    t_all = time.perf_counter()
    started_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = get_config()

    t_cap = time.perf_counter()
    captured = capture_dual_cameras(cfg)
    capture_ms = _elapsed_ms(t_cap)

    left_path = str(captured["left"]) if "left" in captured else None
    right_path = str(captured["right"]) if "right" in captured else None
    capture_errors: Dict[str, str] = {}
    for side in ("left", "right"):
        if side not in captured:
            capture_errors[side] = f"{side} 摄像头抓拍失败"

    if not do_ocr:
        out = {
            "ok": True,
            "mode": "double",
            "capture_mode": "double_camera",
            "skipped_ocr": True,
            "capture": {
                "left_image": left_path,
                "right_image": right_path,
                "errors": capture_errors,
            },
            "timing": {"capture_ms": capture_ms, "total_ms": _elapsed_ms(t_all)},
        }
        return persist_pipeline_result(
            out, mode="double", trigger=trigger, started_at=started_at
        )

    result = ocr_side_images(
        left_path,
        right_path,
        write_pi=bool(do_pi),
        export_excel=True,
        trigger=trigger,
        persist=False,
        capture_errors=capture_errors,
    )
    cams = get_camera_defs(cfg)
    result["capture"] = {
        "left_image": left_path,
        "right_image": right_path,
        "errors": capture_errors,
        "left": {
            "host": cams["left"].get("host") or "",
            "type": cams["left"].get("type") or "",
            "tag": cams["left"].get("tag") or "",
        },
        "right": {
            "host": cams["right"].get("host") or "",
            "type": cams["right"].get("type") or "",
            "tag": cams["right"].get("tag") or "",
        },
    }
    if left_path:
        result["left_image"] = left_path
    if right_path:
        result["right_image"] = right_path
    timing = dict(result.get("timing") or {})
    timing["capture_ms"] = capture_ms
    timing["total_ms"] = _elapsed_ms(t_all)
    result["timing"] = timing
    return persist_pipeline_result(
        result, mode="double", trigger=trigger, started_at=started_at
    )


# 保留符号供旧代码 / 调试：分缝已从主流程移除
def _crop_to_file(*_args: Any, **_kwargs: Any) -> Path:
    raise RuntimeError("double-camera 分支不再整图分缝裁剪")
