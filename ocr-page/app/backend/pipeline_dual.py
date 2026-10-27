# -*- coding: utf-8 -*-
"""双摄像头流水线：抓拍 → 表格区域识别 → 表格数据识别 → 写 PI。

- cameras.left / cameras.right：各自 type、连接、测点前缀、table_search（auto 先验窗）
- 识别：table_search 内 auto，成败本轮即止
- 不识别画面日期（业务日用电脑系统日）
- 写库日 = 系统日按 pi.pi_date_back_days 提前，再与表头时刻拼接
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
    get_camera_table_search,
    get_config,
    get_ocr_max_width,
    get_ocr_min_points,
    get_ocr_parallel_sides,
    get_ocr_fix_decimal_cfg,
    get_ocr_scale,
    get_ocr_onnx_threads,
    new_ocr_run_dir,
    resolve_ocr_workdir,
)
from .excel_export import export_ocr_excel
from .hikvision import (
    HikvisionCaptureError,
    build_camera_client_for_side,
)
from .ocr_bridge import extract_points_from_image, ocr_result_to_dict
from .pipeline import (
    _elapsed_ms,
    evaluate_ocr_point_count,
    resolve_ocr_fail_fields,
    write_points_to_pi,
)
from .vote_store import apply_majority_votes

logger = logging.getLogger(__name__)

DEFAULT_CAMERA: Dict[str, Dict[str, Any]] = {
    "left": {"type": "isapi", "host": "", "tag": "GX.DEMO.OCR.L"},
    "right": {"type": "mvs", "host": "", "tag": "GX.DEMO.OCR.R"},
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
    """左右侧测点前缀、相机信息、table_search。"""
    cams = get_camera_defs(cfg)
    out: Dict[str, Dict[str, Any]] = {}
    for side in ("left", "right"):
        cam = cams[side]
        search = get_camera_table_search(side, cfg)
        out[side] = {
            "tag": cam.get("tag") or DEFAULT_CAMERA[side]["tag"],
            "host": cam.get("host") or "",
            "type": cam.get("type") or "",
            "id": side,
            "title": side,
            "table_search": (
                ",".join(f"{v:g}" for v in search) if search is not None else ""
            ),
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
    export_excel: bool = True,
    debug_dir: Optional[Path] = None,
    fix_decimal: Optional[Dict[str, Any]] = None,
    side: str = "",
    search_crop: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """对一侧图：search 窗内 auto，失败即止。"""
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
        tag_prefix=tag_prefix,
        debug_dir=debug_dir,
        layout="double",
        fix_decimal=(
            fix_decimal if fix_decimal is not None else get_ocr_fix_decimal_cfg()
        ),
        max_ocr_width=max_ocr_width,
        onnx_threads=int(ocr_cfg.get("onnx_threads", get_ocr_onnx_threads())),
        log_prefix=pfx,
        search_crop=search_crop,
        camera_side=str(side or "").strip().lower(),
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
    对左右两路已抓到的整图分别做表格区域识别和数据识别（可选写 PI）。
    只做 cameras.*.table_search 内 auto；失败本轮即止（不试 fixed/原图）。
    """
    t_all = time.perf_counter()
    cfg = get_config()
    screens = get_double_defs(cfg)
    cams = get_camera_defs(cfg)
    ocr_cfg = cfg.get("ocr") or {}
    workdir = resolve_ocr_workdir(cfg)
    run_dir = new_ocr_run_dir("double", cfg)
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
        work_path = pane_path
        monitors[side] = {
            "ok": True,
            "mode": "skipped",
            "enabled": False,
            "source": str(src),
        }
        tag_prefix = str(cam.get("tag") or DEFAULT_CAMERA[side]["tag"]).strip()
        search_crop = get_camera_table_search(side, cfg)
        side_jobs.append(
            {
                "side": side,
                "sc": sc,
                "cam": cam,
                "tag_prefix": tag_prefix,
                "pane_path": work_path,
                "source_path": str(src),
                "side_debug": side_debug,
                "search_crop": search_crop,
            }
        )

    def _run_side_job(job: Dict[str, Any]) -> Tuple[str, Dict[str, Any], int]:
        side = job["side"]
        pfx = cam_log_prefix(side)
        search_crop = job.get("search_crop")
        if search_crop is not None:
            search_desc = "search=" + ",".join(f"{v:.3f}" for v in search_crop)
        else:
            search_desc = "search=整图"
        logger.info("%s开始识别 %s（auto 失败即止）", pfx, search_desc)
        t_side = time.perf_counter()
        ocr_dict = _ocr_one_side(
            pane_image=job["pane_path"],
            tag_prefix=job["tag_prefix"],
            ocr_cfg=ocr_cfg,
            workdir=workdir,
            export_excel=export_excel,
            debug_dir=job["side_debug"],
            fix_decimal=get_ocr_fix_decimal_cfg(cfg),
            side=side,
            search_crop=search_crop,
        )
        ms = _elapsed_ms(t_side)
        date_str = ocr_dict.get("base_date") or "-"
        pts_n = len(ocr_dict.get("points") or [])
        timing = ocr_dict.get("timing") or {}
        logger.info(
            "%s识别结束 ms=%s 区域=%s 数据=%s ok=%s date=%s points=%s",
            pfx,
            ms,
            timing.get("region_ms", 0),
            timing.get("data_ms", 0),
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
        quality = evaluate_ocr_point_count(len(pts), min_points)
        side_ok = bool(quality.get("ok"))
        if not side_ok:
            fail = resolve_ocr_fail_fields(quality, ocr_dict)
            # 对外 reason/message 用具体原因（蒙板等），点数门槛仅作 quality 记录
            ocr_dict["reason"] = fail["reason"]
            ocr_dict["message"] = fail["message"]
            logger.warning(
                "%s跳过写 PI：%s（有效点数 %s < min_points=%s）",
                pfx,
                fail["message"],
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
            "region_ms": int((ocr_dict.get("timing") or {}).get("region_ms") or 0),
            "data_ms": int((ocr_dict.get("timing") or {}).get("data_ms") or 0),
            "monitor": monitors.get(side),
            "table_crop_fallback": bool(ocr_dict.get("table_crop_fallback")),
            "table_crop_fallback_reason": ocr_dict.get(
                "table_crop_fallback_reason"
            ),
            "table_crop_fallback_message": ocr_dict.get(
                "table_crop_fallback_message"
            ),
        }
        if not side_ok:
            side_info["reason"] = ocr_dict.get("reason") or quality.get("reason")
            side_info["message"] = ocr_dict.get("message") or quality.get("message")
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
            "left_region_ms": int(sides["left"].get("region_ms") or 0),
            "left_data_ms": int(sides["left"].get("data_ms") or 0),
            "right_date_ms": int(sides["right"].get("date_ms") or 0),
            "right_table_ms": int(sides["right"].get("table_ms") or 0),
            "right_region_ms": int(sides["right"].get("region_ms") or 0),
            "right_data_ms": int(sides["right"].get("data_ms") or 0),
            "pi_ms": 0,
            "total_ms": 0,
        },
    }
    if path_by_side["left"]:
        out["left_image"] = str(path_by_side["left"])
    if path_by_side["right"]:
        out["right_image"] = str(path_by_side["right"])
    if not left_ok and not right_ok:
        left_msg = str(sides["left"].get("message") or "").strip()
        right_msg = str(sides["right"].get("message") or "").strip()
        left_reason = str(sides["left"].get("reason") or "").strip()
        right_reason = str(sides["right"].get("reason") or "").strip()
        parts = []
        if left_msg:
            parts.append(f"左：{left_msg}")
        if right_msg:
            parts.append(f"右：{right_msg}")
        # 两侧同为蒙板/表格失败时，顶层 reason 用更具体的码，便于列表筛选
        if left_reason == right_reason and left_reason:
            out["reason"] = left_reason
        elif "black_osd" in (left_reason, right_reason):
            out["reason"] = "black_osd"
        elif "dimmed_modal" in (left_reason, right_reason):
            out["reason"] = "dimmed_modal"
        else:
            out["reason"] = "points_below_min"
        out["message"] = (
            "；".join(parts)
            if parts
            else (
                f"左右屏有效点数均低于 min_points={min_points} 或抓拍失败，本轮不写 PI"
            )
        )

    # 众数计票：质量门已过的点才进入；展示仍用 OCR 原值（sides/points），写库用众数胜出
    # 仅正式写 PI 时记票，避免联调 OCR-only 污染票箱
    vote_meta: Dict[str, Any] = {"enabled": False, "slots": {}}
    if write_points:
        write_points, vote_meta = apply_majority_votes(
            write_points, record=bool(write_pi), cfg=cfg
        )
        logger.info(
            "众数计票 record=%s past=%s forecast=%s slots=%s",
            bool(write_pi),
            vote_meta.get("past"),
            vote_meta.get("forecast"),
            len(vote_meta.get("slots") or {}),
        )
    out["vote"] = vote_meta

    if write_pi:
        if write_points:
            t_pi = time.perf_counter()
            write = write_points_to_pi(write_points, tag=None)
            pi_ms = _elapsed_ms(t_pi)
            out["timing"]["pi_ms"] = pi_ms
            logger.info(
                "写PI ms=%s written=%s/%s",
                pi_ms,
                write.get("written", 0),
                write.get("total", 0),
            )
            write["sides_written"] = [
                s for s in ("left", "right") if sides[s].get("ok")
            ]
            write["sides_skipped"] = [
                s for s in ("left", "right") if not sides[s].get("ok")
            ]
        else:
            logger.info("写PI ms=0（跳过：无有效测点）")
            write = {
                "ok": False,
                "written": 0,
                "total": 0,
                "items": [],
                "skipped": True,
                "reason": "points_below_min",
                "message": out.get("message")
                or "左右屏 OCR 均失败，不写 PI",
                "sides_written": [],
                "sides_skipped": ["left", "right"],
            }
        out["pi"] = write
        if write_points:
            out["ok"] = bool(write.get("ok")) and (left_ok or right_ok)
            if not write.get("ok"):
                out["reason"] = "pi_write_failed"
                fails = [
                    it
                    for it in (write.get("items") or [])
                    if not it.get("ok") and not it.get("skipped")
                ]
                first = fails[0] if fails else {}
                why = first.get("message") or (
                    f"code={first.get('pi_code')}"
                    if first.get("pi_code") is not None
                    else ""
                )
                tip = f"写 PI 未全部成功：{write.get('written', 0)}/{write.get('total', 0)}"
                if why:
                    tip += f" · {why}"
                if len(fails) > 1:
                    tip += f"（另有 {len(fails) - 1} 点失败）"
                out["message"] = tip
                write["message"] = tip
                write["reason"] = "pi_write_failed"
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
    logger.info(
        "抓图 ms=%s left=%s right=%s",
        capture_ms,
        "ok" if "left" in captured else "fail",
        "ok" if "right" in captured else "fail",
    )

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
        logger.info(
            "耗时汇总 抓图 ms=%s 总计 ms=%s（跳过识别）",
            capture_ms,
            out["timing"]["total_ms"],
        )
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
    logger.info(
        "耗时汇总 抓图 ms=%s 识别 ms=%s 写PI ms=%s 总计 ms=%s "
        "左区域=%s 左数据=%s 右区域=%s 右数据=%s",
        capture_ms,
        timing.get("ocr_ms", 0),
        timing.get("pi_ms", 0),
        timing.get("total_ms", 0),
        timing.get("left_region_ms", 0),
        timing.get("left_data_ms", 0),
        timing.get("right_region_ms", 0),
        timing.get("right_data_ms", 0),
    )
    return persist_pipeline_result(
        result, mode="double", trigger=trigger, started_at=started_at
    )


# 保留符号供旧代码 / 调试：分缝已从主流程移除
def _crop_to_file(*_args: Any, **_kwargs: Any) -> Path:
    raise RuntimeError("double-camera 分支不再整图分缝裁剪")
