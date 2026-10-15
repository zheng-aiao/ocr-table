# -*- coding: utf-8 -*-
"""双屏左右流水线：一帧抓图 → 裁显示器 → 左右裁剪 → 各自 OCR / Excel / 写 PI。

与 pipeline.run_pipeline（单屏）隔离，不修改其行为。

要点：
- 先 auto_monitor 收紧屏外环境（远景四周 / 近景上下留白），失败回退整图
- 再在屏内图上 auto_seam 左右分缝
- 左/右是两个机组，各自测点前缀 capture.dual.*.tag，行 → 前缀_1…_N
- 左右独立解析，点内已带 tag，写 PI 不互相覆盖
- 业务日只认各半屏左上角日期 OCR；失败则该侧直接失败并跳过表格 OCR
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import (
    ROOT_DIR,
    get_config,
    get_ocr_date_cfg,
    get_ocr_early_stop,
    get_ocr_early_stop_min_density,
    get_ocr_max_width,
    get_ocr_min_points,
    get_ocr_parallel_sides,
    get_ocr_fix_decimal_cfg,
    get_ocr_sanitize_cfg,
    get_ocr_scale,
    get_ocr_table_crop,
    get_ocr_onnx_threads,
    new_ocr_run_dir,
    parse_crop,
    resolve_ocr_workdir,
)
from .excel_export import export_ocr_excel
from .dual_seam import detect_dual_seam, seam_to_side_crops
from .hikvision import build_camera_client
from .monitor_crop import prepare_monitor_work_image
from .ocr_bridge import extract_points_from_image, ocr_result_to_dict
from .pipeline import _elapsed_ms, evaluate_ocr_point_count, write_points_to_pi

logger = logging.getLogger(__name__)

DEFAULT_DUAL: Dict[str, Dict[str, Any]] = {
    "left": {
        "id": "61",
        "title": "left",
        # 中缝内收，避免右缘吃到右屏表头/数据
        "crop": "0,0,0.545,1",
        "table_crop": "0.02,0.52,0.97,0.90",
        "tag": "GX.DEMO.OCR.L",
        "date_crop": "0.0,0.16,0.48,0.26",
    },
    "right": {
        "id": "62",
        "title": "right",
        # 中缝内收，避免左缘吃到左屏
        "crop": "0.555,0,1,1",
        "table_crop": "0.03,0.52,1.0,0.90",
        "tag": "GX.DEMO.OCR.R",
        "date_crop": "0.0,0.16,0.48,0.26",
    },
}


def get_capture_mode(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or get_config()
    mode = str((cfg.get("capture") or {}).get("mode") or "single").strip().lower()
    return "dual" if mode == "dual" else "single"


def get_dual_defs(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """合并默认与 capture.dual（兼容旧 capture.screens / 顶层 screens）。"""
    cfg = cfg or get_config()
    cap = cfg.get("capture") or {}
    raw = cap.get("dual") or cap.get("screens") or cfg.get("screens") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for side in ("left", "right"):
        base = dict(DEFAULT_DUAL[side])
        base.update(raw.get(side) or {})
        out[side] = base
    return out


# 兼容旧名
def get_screen_defs(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    return get_dual_defs(cfg)


def _parse_day(raw: Optional[str]) -> Optional[datetime]:
    """yyyy-MM-dd → 当天 00:00:00；空则 None。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if len(s) == 10:
        s = s + " 00:00:00"
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _crop_to_file(
    image_path: Path,
    crop: Tuple[float, float, float, float],
    out_path: Path,
) -> Path:
    """按相对比例裁剪整图并落盘（便于对照调试）。"""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("双屏裁剪需要 Pillow，请 pip install Pillow") from e

    with Image.open(image_path) as im:
        w, h = im.size
        x0, y0, x1, y1 = crop
        box = (
            max(0, int(x0 * w)),
            max(0, int(y0 * h)),
            min(w, int(x1 * w)),
            min(h, int(y1 * h)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"无效 crop 区域: {crop} → {box} (image {w}x{h})")
        cropped = im.crop(box)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() in {".jpg", ".jpeg"}:
            cropped.convert("RGB").save(out_path, quality=95)
        else:
            cropped.save(out_path)
    return out_path


def _ocr_one_side(
    *,
    pane_image: Path,
    tag_prefix: str,
    ocr_cfg: Dict[str, Any],
    workdir: Path,
    base_date: Optional[datetime],
    export_excel: bool = True,
    debug_dir: Optional[Path] = None,
    sanitize: Optional[Dict[str, Any]] = None,
    fix_decimal: Optional[Dict[str, Any]] = None,
    date_cfg: Optional[Dict[str, Any]] = None,
    table_crop: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """对半屏图做 OCR（layout=dual：半屏窄侧栏候选，不改动上游左右分屏）。

    业务日只认半屏左上角日期 OCR；失败则跳过表格 OCR。
    """
    del base_date
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
        layout="dual",
        sanitize=sanitize if sanitize is not None else get_ocr_sanitize_cfg(),
        fix_decimal=(
            fix_decimal if fix_decimal is not None else get_ocr_fix_decimal_cfg()
        ),
        early_stop=bool(ocr_cfg.get("early_stop", get_ocr_early_stop())),
        early_stop_min_density=int(
            ocr_cfg.get("early_stop_min_density", get_ocr_early_stop_min_density())
        ),
        max_ocr_width=max_ocr_width,
        date_cfg=date_cfg if date_cfg is not None else get_ocr_date_cfg(layout="dual"),
        table_crop=table_crop,
        onnx_threads=int(
            ocr_cfg.get("onnx_threads", get_ocr_onnx_threads())
        ),
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
            logger.exception("双屏 Excel 导出失败 side_image=%s", pane_image)
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


def _resolve_side_dates(
    *,
    base_date: Optional[datetime] = None,
    left_date: Optional[datetime] = None,
    right_date: Optional[datetime] = None,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    左右各自业务日回退参数已废弃（保留签名兼容）；实际只认画面日期 OCR。
    """
    del base_date, left_date, right_date
    return (None, None)


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
    对已有整图：先裁显示器区域 → 左右裁剪 → 独立 OCR（可选写 PI）。
    左右日期独立（优先各半屏左上角日期 OCR）；测点前缀独立；结果不混写。
    过程中间文件写入 ocr.artifacts_dir（默认 data/ocr_runs/dual/<会话>/）。
    upload_name：本机上传原始文件名（展示用，落盘预览仍用 00_source）。
    """
    import shutil

    t_all = time.perf_counter()
    cfg = get_config()
    screens = get_dual_defs(cfg)
    ocr_cfg = cfg.get("ocr") or {}
    workdir = resolve_ocr_workdir(cfg)
    full_path = Path(image_path)
    if not full_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    run_dir = new_ocr_run_dir("dual", cfg)
    work_root = run_dir

    src_copy = ""
    src_copy_path = run_dir / f"00_source{full_path.suffix.lower() or '.jpg'}"
    try:
        shutil.copy2(full_path, src_copy_path)
        src_copy = str(src_copy_path)
    except Exception:
        logger.exception("复制双屏源图失败")

    day_l, day_r = _resolve_side_dates(
        base_date=base_date, left_date=left_date, right_date=right_date
    )
    side_days = {"left": day_l, "right": day_r}

    sides: Dict[str, Any] = {}
    all_points: List[Dict[str, Any]] = []
    write_points: List[Dict[str, Any]] = []
    min_points = get_ocr_min_points(cfg)
    side_ocr_ms: Dict[str, int] = {}

    t_ocr_all = time.perf_counter()
    dual_cfg = (cfg.get("capture") or {}).get("dual") or {}

    # —— 0) 先裁显示器区域（相对整图），再在屏内做左右分缝 ——
    # 远景四周有环境 / 近景宽贴满但上下仍有墙桌时，避免日期表区比例错位。
    auto_monitor = dual_cfg.get("auto_monitor")
    if auto_monitor is None:
        auto_monitor = True
    work_path, monitor_meta, monitor_path = prepare_monitor_work_image(
        full_path, run_dir, enabled=bool(auto_monitor)
    )

    # —— 1) 自动窗缝：相对「屏内图」覆盖左右 crop；失败则用配置 crop ——
    auto_seam = dual_cfg.get("auto_seam")
    if auto_seam is None:
        auto_seam = True
    seam_meta: Dict[str, Any] = {"enabled": bool(auto_seam)}
    left_crop_override: Optional[Tuple[float, float, float, float]] = None
    right_crop_override: Optional[Tuple[float, float, float, float]] = None
    if auto_seam:
        try:
            fb = float(dual_cfg.get("seam_fallback") or 0.55)
        except (TypeError, ValueError):
            fb = 0.55
        try:
            gap = float(dual_cfg.get("seam_gap") or 0.012)
        except (TypeError, ValueError):
            gap = 0.012
        seam_meta = detect_dual_seam(work_path, fallback=fb)
        seam_meta["enabled"] = True
        seam_meta["gap"] = gap
        if seam_meta.get("ok"):
            left_crop_override, right_crop_override = seam_to_side_crops(
                float(seam_meta["seam"]), gap=gap
            )
            logger.info(
                "双屏采用自动缝 seam=%.4f mode=%s left=%s right=%s (on=%s)",
                seam_meta["seam"],
                seam_meta.get("mode"),
                left_crop_override,
                right_crop_override,
                work_path.name,
            )
        else:
            logger.info(
                "双屏自动缝未命中，使用配置 crop fallback_seam=%.4f",
                float(seam_meta.get("seam") or fb),
            )
    try:
        import json as _json

        (run_dir / "seam.json").write_text(
            _json.dumps(seam_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("写 seam.json 失败")

    side_jobs: List[Dict[str, Any]] = []
    for side in ("left", "right"):
        sc = screens[side]
        crop_raw = sc.get("crop") or DEFAULT_DUAL[side]["crop"]
        crop = parse_crop(str(crop_raw))
        if crop is None:
            raise ValueError(f"capture.dual.{side}.crop 无效: {crop_raw!r}")
        if side == "left" and left_crop_override is not None:
            crop = left_crop_override
        elif side == "right" and right_crop_override is not None:
            crop = right_crop_override
        tag_prefix = str(sc.get("tag") or DEFAULT_DUAL[side]["tag"]).strip()
        pane_path = work_root / f"{side}.jpg"
        side_debug = run_dir / side
        side_debug.mkdir(parents=True, exist_ok=True)
        # 左右裁相对屏内图（或整图回退）
        _crop_to_file(work_path, crop, pane_path)
        day = side_days[side]
        logger.info(
            "双屏 OCR %s crop=%s tag=%s fallback_date=%s → %s",
            side,
            crop,
            tag_prefix,
            day.strftime("%Y-%m-%d") if day else None,
            pane_path,
        )
        side_jobs.append(
            {
                "side": side,
                "sc": sc,
                "crop": crop,
                "tag_prefix": tag_prefix,
                "pane_path": pane_path,
                "side_debug": side_debug,
                "day": day,
            }
        )

    def _run_side_job(job: Dict[str, Any]) -> Tuple[str, Dict[str, Any], int]:
        t_side = time.perf_counter()
        ocr_dict = _ocr_one_side(
            pane_image=job["pane_path"],
            tag_prefix=job["tag_prefix"],
            ocr_cfg=ocr_cfg,
            workdir=workdir,
            base_date=job["day"],
            export_excel=export_excel,
            debug_dir=job["side_debug"],
            sanitize=get_ocr_sanitize_cfg(cfg),
            fix_decimal=get_ocr_fix_decimal_cfg(cfg),
            date_cfg=get_ocr_date_cfg(cfg, layout="dual", side=job["side"]),
            table_crop=get_ocr_table_crop(cfg, layout="dual", side=job["side"]),
        )
        return job["side"], ocr_dict, _elapsed_ms(t_side)

    ocr_by_side: Dict[str, Dict[str, Any]] = {}
    parallel = get_ocr_parallel_sides(cfg) and len(side_jobs) > 1
    if parallel:
        logger.info("双屏 OCR 并行 left|right")
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
        crop = job["crop"]
        tag_prefix = job["tag_prefix"]
        pane_path = job["pane_path"]
        side_debug = job["side_debug"]
        ocr_dict = ocr_by_side[side]
        pts = _points_payload(ocr_dict, side)
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
                "双屏 %s 日期 OCR 失败：%s，本侧跳过表格/不写 PI",
                side,
                quality.get("message"),
            )
        else:
            quality = evaluate_ocr_point_count(len(pts), min_points)
            side_ok = bool(quality.get("ok"))
            if not side_ok:
                logger.warning(
                    "双屏 %s 识别失败：有效点数 %s < min_points=%s，本侧不写 PI",
                    side,
                    quality.get("point_count"),
                    quality.get("min_points"),
                )
        ocr_dict["quality"] = quality
        all_points.extend(pts)
        if side_ok:
            write_points.extend(pts)
        resolved_day = ocr_dict.get("base_date")
        side_info: Dict[str, Any] = {
            "ok": side_ok,
            "id": str(sc.get("id") or DEFAULT_DUAL[side]["id"]),
            "title": str(sc.get("title") or side),
            "tag": tag_prefix,
            "base_date": resolved_day,
            "date_source": ocr_dict.get("date_source"),
            "date_ocr": ocr_dict.get("date_ocr") or {},
            "crop": list(crop),
            "image_path": str(pane_path),
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
        "mode": "dual",
        "image_path": str(full_path),
        "work_image_path": str(work_path),
        "left": sides["left"],
        "right": sides["right"],
        "left_ok": left_ok,
        "right_ok": right_ok,
        "points": all_points,
        "min_points": min_points,
        "artifacts_dir": str(run_dir),
        "monitor": monitor_meta,
        "seam": seam_meta,
        "timing": {
            "ocr_ms": ocr_ms,
            "monitor_ms": int(monitor_meta.get("ms") or 0),
            "seam_ms": int(seam_meta.get("ms") or 0),
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
    if monitor_path:
        out["monitor_image"] = monitor_path
    if not left_ok and not right_ok:
        out["reason"] = "points_below_min"
        out["message"] = (
            f"左右屏有效点数均低于 min_points={min_points}，本轮识别失败，不写 PI"
        )
    if src_copy:
        out["source_copy"] = src_copy
    if upload_name:
        out["upload_name"] = str(upload_name).strip()
        out["uploaded"] = True

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
        # 联调 / OCR-only：与单屏 ocr_only 一致，标记未写 PI
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

    # 部分成功：写完 PI 后再汇总说明，便于首页展示
    if left_ok != right_ok:
        out["reason"] = "partial_side_failed"
        fail_side = "左侧" if not left_ok else "右侧"
        ok_side = "右侧" if not left_ok else "左侧"
        fail_n = int(sides["left" if not left_ok else "right"].get("point_count") or 0)
        fail_detail = f"有效点数 {fail_n} < min_points={min_points}"
        if write_pi:
            sides_written = (out.get("pi") or {}).get("sides_written") or []
            ok_key = "right" if not left_ok else "left"
            did_write = ok_key in sides_written and bool((out.get("pi") or {}).get("ok"))
            out["message"] = (
                f"{fail_side}识别未达下限（{fail_detail}），该侧未写 PI；"
                + (
                    f"{ok_side}已达标并写入"
                    if did_write
                    else f"{ok_side}已达标但写 PI 未成功"
                )
            )
        else:
            out["message"] = (
                f"{fail_side}识别未达下限（{fail_detail}）；"
                f"{ok_side}识别达标"
            )

    out["timing"]["total_ms"] = _elapsed_ms(t_all)
    if persist:
        from .run_store import persist_pipeline_result

        return persist_pipeline_result(
            out, mode="dual", trigger=trigger, started_at=None
        )
    return out


def run_pipeline_dual(
    base_date: Optional[datetime] = None,
    left_date: Optional[datetime] = None,
    right_date: Optional[datetime] = None,
    *,
    trigger: str = "manual",
) -> Dict[str, Any]:
    """一键双屏：抓图 → 左右独立 OCR/Excel → 按点内 tag 写 PI。"""
    from datetime import datetime as _dt

    from .run_store import persist_pipeline_result

    t_all = time.perf_counter()
    started_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = get_config()
    cam = build_camera_client(cfg, ROOT_DIR)
    t_cap = time.perf_counter()
    full_path = Path(cam.capture())
    capture_ms = _elapsed_ms(t_cap)
    result = split_and_ocr(
        str(full_path),
        base_date=base_date,
        left_date=left_date,
        right_date=right_date,
        write_pi=True,
        export_excel=True,
        trigger=trigger,
        persist=False,
    )
    result["capture"] = {
        "image_path": str(full_path),
        "size": full_path.stat().st_size,
        "left_image": result["left"]["image_path"],
        "right_image": result["right"]["image_path"],
    }
    timing = dict(result.get("timing") or {})
    timing["capture_ms"] = capture_ms
    # 总耗时含抓图；split 内 total 不含外层 capture
    timing["total_ms"] = _elapsed_ms(t_all)
    result["timing"] = timing
    return persist_pipeline_result(
        result, mode="dual", trigger=trigger, started_at=started_at
    )
