# -*- coding: utf-8 -*-
"""执行履历：每次抓拍/OCR/写PI 落盘 run 摘要，供列表与详情页。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    DATA_DIR,
    ROOT_DIR,
    get_config,
    get_ocr_expected_cells,
    resolve_ocr_artifacts_root,
    resolve_project_path,
)

logger = logging.getLogger(__name__)

RUNS_DIR = DATA_DIR / "runs"


def _web_url(path: Optional[str | Path]) -> str:
    """把项目内绝对/相对路径转成可挂载的 URL；非项目内返回空。"""
    if not path:
        return ""
    try:
        p = resolve_project_path(path)
        root = ROOT_DIR.resolve()
        rel = p.relative_to(root)
        return "/" + rel.as_posix()
    except Exception:
        return ""


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _run_id(mode: str, artifacts_dir: Optional[str] = None) -> str:
    m = "dual" if str(mode).strip().lower() == "dual" else "single"
    if artifacts_dir:
        name = Path(artifacts_dir).name
        if name:
            return f"{m}_{name}"
    return f"{m}_{_now_stamp()}"


def _candidates_for_ui(ocr: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """各裁剪候选：含 OCR 表、胜负状态与可访问图片 URL。"""
    if not ocr:
        return []
    out: List[Dict[str, Any]] = []
    for raw in ocr.get("candidates") or []:
        if not isinstance(raw, dict):
            continue
        img = raw.get("image_path") or ""
        label = str(raw.get("label") or "")
        name = str(raw.get("name") or "").strip()
        if not name and label:
            name = label if label.startswith("cand_") else f"cand_{label}"
        out.append(
            {
                "index": raw.get("index"),
                "label": label,
                "name": name,
                "crop": raw.get("crop"),
                "status": raw.get("status") or "",
                "reason": raw.get("reason") or "",
                "score": raw.get("score"),
                "headers": list(raw.get("headers") or []),
                "rows": [list(r) for r in (raw.get("rows") or [])],
                "point_count": int(raw.get("point_count") or 0),
                "gap_filled": int(raw.get("gap_filled") or 0),
                "gap_fill_map": [
                    [str(c or "") for c in row]
                    for row in (raw.get("gap_fill_map") or [])
                ],
                "image_path": img,
                "image_url": _web_url(img),
            }
        )
    return out


def _scan_date_crop_paths(arts: str) -> List[str]:
    """从 artifacts 目录扫描日期过程图（不含 best）。"""
    if not arts:
        return []
    d = resolve_project_path(arts)
    if not d.is_dir():
        return []
    out: List[str] = []
    for name in (
        "date_locate.jpg",
        "date_crop.jpg",
        "date_crop_anchor.jpg",
        "date_crop_cfg.jpg",
    ):
        p = d / name
        if p.is_file():
            out.append(str(p))
    for p in sorted(d.glob("date_crop_enlarge_*.jpg")):
        out.append(str(p))
    for p in sorted(d.glob("date_crop_cfg_enlarge_*.jpg")):
        out.append(str(p))
    # 旧命名 date_crop_XX（排除已收录的别名）
    known = {Path(x).name.lower() for x in out}
    known.add("date_crop_best.jpg")
    for p in sorted(d.glob("date_crop_*.jpg")):
        n = p.name.lower()
        if n in known:
            continue
        if n.startswith("date_crop_enlarge_") or n.startswith("date_crop_cfg_enlarge_"):
            continue
        out.append(str(p))
    return out


def _date_step_meta(path: str) -> Dict[str, str]:
    """按文件名给日期过程图打角色标签（展示用）。"""
    name = Path(path).name.lower()
    if name == "date_locate.jpg":
        return {
            "role": "locate",
            "stage": "locate_main",
            "label": "定位框",
            "hint": "auto 找到的日期+筛选框",
        }
    if name == "date_crop.jpg":
        return {
            "role": "crop",
            "stage": "locate_main",
            "label": "OCR 主框",
            "hint": "自定位主框送入日期 OCR",
        }
    if name == "date_crop_anchor.jpg":
        return {
            "role": "anchor",
            "stage": "locate_main",
            "label": "标签右侧锚点",
            "hint": "主框未认全时，再裁「日期」右侧输入框",
        }
    if name == "date_crop_cfg.jpg":
        return {
            "role": "config",
            "stage": "config",
            "label": "配置固定裁剪",
            "hint": "config date_crop 回退主框",
        }
    if name.startswith("date_crop_enlarge_") and name.endswith(".jpg"):
        try:
            n = int(name.replace("date_crop_enlarge_", "").replace(".jpg", ""))
        except ValueError:
            n = 0
        return {
            "role": "enlarge",
            "stage": "enlarge",
            "label": f"放大尝试 {n}",
            "hint": "自定位主框四周略放大",
        }
    if name.startswith("date_crop_cfg_enlarge_") and name.endswith(".jpg"):
        try:
            n = int(name.replace("date_crop_cfg_enlarge_", "").replace(".jpg", ""))
        except ValueError:
            n = 0
        return {
            "role": "config_enlarge",
            "stage": "config",
            "label": f"配置框放大 {n}",
            "hint": "配置回退框略放大",
        }
    if name.startswith("date_crop_") and name.endswith(".jpg"):
        return {
            "role": "neighbor",
            "stage": "enlarge",
            "label": "裁剪尝试",
            "hint": "略放大或配置回退框",
        }
    if name.startswith("date_search_"):
        return {"role": "search", "stage": "", "label": "搜索带", "hint": "左上区域搜标签"}
    return {"role": "other", "stage": "", "label": "过程图", "hint": name}


_STAGE_DEFS: Tuple[Dict[str, str], ...] = (
    {
        "key": "locate_main",
        "title": "1. 自定位 OCR主框",
        "hint": "定位筛选框 + 送入 OCR 的主裁剪（2 张）",
    },
    {
        "key": "enlarge",
        "title": "2. OCR主框放大尝试",
        "hint": "主框未认出时，对裁剪区域略放大再认",
    },
    {
        "key": "config",
        "title": "3. 配置退回固定裁剪",
        "hint": "自定位整轮无命中后，改用 config date_crop",
    },
)


def _stage_key_from_attempt(attempt: Dict[str, Any]) -> str:
    """把 attempt.stage / 文件名映射到三阶段 key。"""
    st = str(attempt.get("stage") or "").strip()
    if st == "auto_main":
        return "locate_main"
    if st == "auto_enlarge":
        return "enlarge"
    if st == "config_fallback":
        return "config"
    path = str(attempt.get("image_path") or "")
    if path:
        return str(_date_step_meta(path).get("stage") or "enlarge")
    phase = attempt.get("phase")
    if phase == 0:
        li = attempt.get("local_index")
        if li == 0:
            return "locate_main"
        return "enlarge"
    if phase == 1:
        return "config"
    return "enlarge"


def _attempt_item_label(attempt: Dict[str, Any], stage_key: str) -> Dict[str, str]:
    """步骤内单张图的标题。"""
    skipped = str(attempt.get("skipped") or "")
    path = str(attempt.get("image_path") or "")
    name = Path(path).name.lower() if path else ""
    if skipped == "chart_axis":
        return {
            "role": "skip",
            "label": "跳过·曲线轴",
            "hint": "裁剪像曲线 Y 轴，未作日期解析",
        }
    if path:
        base = _date_step_meta(path)
    elif stage_key == "locate_main":
        base = {"role": "crop", "label": "OCR 主框", "hint": "自定位主框"}
    elif stage_key == "enlarge":
        li = int(attempt.get("local_index") or 0)
        base = {
            "role": "enlarge",
            "label": f"放大尝试 {li}",
            "hint": "主框略放大",
        }
    else:
        li = int(attempt.get("local_index") or 0)
        base = {
            "role": "config",
            "label": "配置固定裁剪" if li == 0 else f"配置框放大 {li}",
            "hint": "config date_crop",
        }
    if attempt.get("ok"):
        base["hint"] = (base.get("hint") or "") + " · 本步认出日期"
    elif not skipped:
        base["hint"] = (base.get("hint") or "") + " · 未认出"
    return base


def _paths_reached_for_date_ui(meta: Dict[str, Any], arts: str) -> List[str]:
    """
    只收集「实际走到」的过程图；提前成功后的后续步骤不展示。
    优先用 attempts；旧履历按 reason 截断。
    """
    paths: List[str] = []
    locate_meta = meta.get("locate") if isinstance(meta.get("locate"), dict) else {}
    loc_path = str((locate_meta or {}).get("image_path") or "").strip()
    if not loc_path and arts:
        p = resolve_project_path(arts) / "date_locate.jpg"
        if p.is_file():
            loc_path = str(p)
    # 仅当真正定位成功（有 crop/method）才展示定位图
    if (
        loc_path
        and locate_meta
        and locate_meta.get("ok") is not False
        and (locate_meta.get("crop") or locate_meta.get("method"))
    ):
        paths.append(loc_path)

    attempts = meta.get("attempts") if isinstance(meta.get("attempts"), list) else []
    if attempts:
        for att in attempts:
            if not isinstance(att, dict):
                continue
            # 增强轮常无新图 / 与首轮重复，只取有图且未重复的
            if att.get("enhance"):
                continue
            p = str(att.get("image_path") or "").strip()
            if p:
                paths.append(p)
            if att.get("ok") and att.get("date_str"):
                break
        return list(dict.fromkeys(paths))

    # 无 attempts 的旧履历：按 reason 截断，避免把锚点/放大误当成走过的步骤
    reason = str(meta.get("reason") or "")
    scanned = _scan_date_crop_paths(arts) if arts else []
    primary = str(meta.get("image_path") or "").strip()
    if reason.startswith("full_ymd_auto_locate") or reason.startswith("full_ymd_crop"):
        out: List[str] = []
        for p in scanned:
            n = Path(p).name.lower()
            if n == "date_locate.jpg":
                out.append(p)
        if primary:
            out.append(primary)
        else:
            for p in scanned:
                if Path(p).name.lower() == "date_crop.jpg":
                    out.append(p)
                    break
        return list(dict.fromkeys(out))

    return [p for p in scanned if not Path(p).name.lower().startswith("date_search_")]


def _humanize_date_reason(reason: Any, date_source: Any = None) -> str:
    """把 reason 码转成简短中文说明。"""
    r = str(reason or "").strip()
    src = str(date_source or "").strip()
    if not r and src == "ocr":
        return "OCR 识别成功"
    if r.startswith("full_ymd_auto_locate"):
        return "自定位主框认出完整日期"
    if r.startswith("full_ymd_crop"):
        return "配置退回裁剪认出完整日期"
    if "auto_locate" in r and "best_conf" in r:
        return "自定位阶段择优"
    if r.startswith("best_conf"):
        return "多命中按置信择优"
    if r == "merged_texts":
        return "合并多框原文解析"
    if r in ("date_not_found", "date_ocr_failed"):
        return "未识别到完整业务日期"
    if r == "date_ocr_disabled":
        return "日期 OCR 已关闭"
    return r or "-"


def _make_step_item(
    *,
    path: str,
    url: str,
    label: str,
    hint: str,
    role: str,
    stage: str,
    ok: bool = False,
    is_best: bool = False,
    skipped: str = "",
    date_str: str = "",
    name: str = "",
) -> Dict[str, Any]:
    return {
        "name": name or (Path(path).name if path else ""),
        "image_path": path,
        "image_url": url,
        "label": label,
        "hint": hint,
        "role": role,
        "stage": stage,
        "ok": ok,
        "is_best": is_best,
        "skipped": skipped,
        "date_str": date_str,
    }


def _date_fields_for_ui(
    *,
    artifacts_dir: str = "",
    date_ocr: Optional[Dict[str, Any]] = None,
    base_date: Any = None,
    date_source: Any = None,
) -> Dict[str, Any]:
    """日期 OCR 展示：主结果图 + 三阶段识别过程（仅已执行）。"""
    meta = dict(date_ocr or {}) if isinstance(date_ocr, dict) else {}
    arts = str(artifacts_dir or "").strip()

    best = str(meta.get("best_image_path") or "").strip()
    if not best and arts:
        cand = resolve_project_path(arts) / "date_crop_best.jpg"
        if cand.is_file():
            best = str(cand)

    primary = str(meta.get("image_path") or "").strip()
    attempts = meta.get("attempts") if isinstance(meta.get("attempts"), list) else []
    locate_meta = meta.get("locate") if isinstance(meta.get("locate"), dict) else {}

    # 按三阶段收集 items
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "locate_main": [],
        "enlarge": [],
        "config": [],
    }
    seen_urls: set[str] = set()

    def _push(stage_key: str, item: Dict[str, Any]) -> None:
        url = str(item.get("image_url") or "")
        if url and url in seen_urls:
            return
        if url:
            seen_urls.add(url)
        buckets.setdefault(stage_key, []).append(item)

    # —— 阶段1：定位图（若有）
    loc_path = str((locate_meta or {}).get("image_path") or "").strip()
    if not loc_path and arts:
        p = resolve_project_path(arts) / "date_locate.jpg"
        if p.is_file():
            loc_path = str(p)
    locate_ok = bool(
        loc_path
        and locate_meta
        and locate_meta.get("ok") is not False
        and (locate_meta.get("crop") or locate_meta.get("method"))
    )
    if locate_ok:
        sm = _date_step_meta(loc_path)
        _push(
            "locate_main",
            _make_step_item(
                path=loc_path,
                url=_web_url(loc_path),
                label=sm["label"],
                hint=sm["hint"],
                role=sm["role"],
                stage="locate_main",
                ok=True,
            ),
        )

    def _is_best_path(p: str, att: Optional[Dict[str, Any]] = None) -> bool:
        if att and att.get("ok") and att.get("date_str"):
            return True
        if not p or not best:
            return False
        try:
            return Path(p).resolve() == Path(best).resolve()
        except Exception:
            return Path(p).name == Path(best).name

    stop = False
    if attempts:
        for att in attempts:
            if not isinstance(att, dict) or stop:
                continue
            # 增强轮与首轮同框，不重复展示
            if att.get("enhance"):
                if att.get("ok") and att.get("date_str"):
                    stop = True
                continue
            stage_key = _stage_key_from_attempt(att)
            # 无自定位时，phase0 主框归入配置退回
            if (
                stage_key == "locate_main"
                and not locate_ok
                and str(att.get("stage") or "") != "auto_main"
            ):
                stage_key = "config"
            p = str(att.get("image_path") or "").strip()
            url = _web_url(p) if p else ""
            lab = _attempt_item_label(att, stage_key)
            _push(
                stage_key,
                _make_step_item(
                    path=p,
                    url=url,
                    label=lab.get("label") or "",
                    hint=lab.get("hint") or "",
                    role=lab.get("role") or "",
                    stage=stage_key,
                    ok=bool(att.get("ok")),
                    is_best=_is_best_path(p, att),
                    skipped=str(att.get("skipped") or ""),
                    date_str=str(att.get("date_str") or ""),
                ),
            )
            if att.get("ok") and att.get("date_str"):
                stop = True
    else:
        # 旧履历：按文件名归入三阶段
        for p in _paths_reached_for_date_ui(meta, arts):
            n = Path(p).name.lower()
            if n == "date_locate.jpg":
                if locate_ok:
                    continue  # 已在上面处理
                sm = _date_step_meta(p)
                _push(
                    "locate_main",
                    _make_step_item(
                        path=p,
                        url=_web_url(p),
                        label=sm["label"],
                        hint=sm["hint"],
                        role=sm["role"],
                        stage="locate_main",
                        ok=True,
                    ),
                )
                continue
            sm = _date_step_meta(p)
            sk = str(sm.get("stage") or "enlarge")
            if not sk:
                continue
            _push(
                sk,
                _make_step_item(
                    path=p,
                    url=_web_url(p),
                    label=sm["label"],
                    hint=sm["hint"],
                    role=sm["role"],
                    stage=sk,
                    ok=bool(meta.get("ok")),
                    is_best=_is_best_path(p),
                ),
            )

    # 组装 date_stages（只含已执行）
    date_stages: List[Dict[str, Any]] = []
    date_steps: List[Dict[str, Any]] = []
    step_i = 0
    for defn in _STAGE_DEFS:
        items = buckets.get(defn["key"]) or []
        if not items:
            continue
        # 放大尝试编号：按顺序改成 1..N
        if defn["key"] == "enlarge":
            for j, it in enumerate(items, start=1):
                if str(it.get("role") or "") in ("enlarge", "neighbor"):
                    it["label"] = f"放大尝试 {j}"
        staged_items = []
        for it in items:
            row = dict(it)
            row["index"] = step_i
            step_i += 1
            staged_items.append(row)
            date_steps.append(row)
        any_ok = any(bool(it.get("ok") and it.get("date_str")) or it.get("is_best") for it in staged_items)
        date_stages.append(
            {
                "key": defn["key"],
                "title": defn["title"],
                "hint": defn["hint"],
                "ok": any_ok,
                "items": staged_items,
            }
        )

    date_candidates = list(date_steps)

    if meta.get("image_path"):
        meta["image_url"] = _web_url(meta.get("image_path"))
    if best:
        meta["best_image_path"] = best
        meta["best_image_url"] = _web_url(best)

    src = date_source or meta.get("date_source") or ""
    reason = meta.get("reason") or ""
    reason_label = _humanize_date_reason(reason, src)

    return {
        "base_date": base_date if base_date not in (None, "") else meta.get("date_str"),
        "date_source": src,
        "date_ocr": meta,
        "date_reason_label": reason_label,
        "date_crop_path": primary or (date_steps[-1]["image_path"] if date_steps else ""),
        "date_crop_url": _web_url(primary)
        or (date_steps[-1]["image_url"] if date_steps else ""),
        "date_best_path": best,
        "date_best_url": _web_url(best),
        "date_stages": date_stages,
        "date_steps": date_steps,
        "date_candidates": date_candidates,
    }


def _enrich_run_date_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """给已落盘 run 补齐日期图 URL（兼容旧履历）。"""
    if not isinstance(record, dict):
        return record
    if record.get("mode") == "dual":
        for key in ("left", "right"):
            side = record.get(key)
            if not isinstance(side, dict):
                continue
            fields = _date_fields_for_ui(
                artifacts_dir=str(side.get("artifacts_dir") or ""),
                date_ocr=side.get("date_ocr") or {},
                base_date=side.get("base_date"),
                date_source=side.get("date_source"),
            )
            side.update(fields)
        return record

    ocr_meta = record.get("date_ocr") or {}
    fields = _date_fields_for_ui(
        artifacts_dir=str(record.get("artifacts_dir") or ""),
        date_ocr=ocr_meta if isinstance(ocr_meta, dict) else {},
        base_date=record.get("base_date"),
        date_source=record.get("date_source"),
    )
    # 旧单屏履历可能没有 date_ocr，尝试读 artifacts/date_ocr.json
    arts = str(record.get("artifacts_dir") or "")
    if arts and not (fields.get("date_candidates") or fields.get("date_best_url")):
        try:
            dj = resolve_project_path(arts) / "date_ocr.json"
            if dj.is_file():
                disk = json.loads(dj.read_text(encoding="utf-8"))
                nested = disk.get("date_ocr") if isinstance(disk, dict) else None
                fields = _date_fields_for_ui(
                    artifacts_dir=arts,
                    date_ocr=nested if isinstance(nested, dict) else disk,
                    base_date=record.get("base_date")
                    or (disk.get("base_date") if isinstance(disk, dict) else None),
                    date_source=record.get("date_source")
                    or (disk.get("date_source") if isinstance(disk, dict) else None),
                )
        except Exception:
            logger.exception("补齐日期 OCR 字段失败: %s", arts)
    record.update(fields)
    return record


def _side_summary(side: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not side:
        return None
    ocr = side.get("ocr") or {}
    arts = side.get("artifacts_dir") or ""
    best = ocr.get("best_crop_path") or ""
    if not best and arts:
        cand = resolve_project_path(arts) / "best_table.jpg"
        if cand.is_file():
            best = str(cand)
    cands = list(ocr.get("candidate_paths") or [])
    candidates = _candidates_for_ui(ocr)
    point_count = int(side.get("point_count") or 0)
    ocr_filled = int(
        side.get("ocr_filled")
        or ocr.get("ocr_filled")
        or 0
    )
    if ocr_filled <= 0:
        ocr_filled = sum(
            1
            for row in (ocr.get("rows") or [])
            for cell in row
            if cell and str(cell).strip()
        )
    headers = list(ocr.get("headers") or [])
    rows = [list(r) for r in (ocr.get("rows") or [])]
    gap_fill_map = [
        [str(c or "") for c in row]
        for row in (side.get("gap_fill_map") or ocr.get("gap_fill_map") or [])
    ]
    gap_filled = int(side.get("gap_filled") or ocr.get("gap_filled") or 0)
    if gap_filled <= 0 and gap_fill_map:
        gap_filled = sum(1 for row in gap_fill_map for c in row if c and str(c).strip())
    expected_cells = int(
        side.get("expected_cells")
        or ocr.get("expected_cells")
        or 0
    )
    if expected_cells <= 0:
        expected_cells = get_ocr_expected_cells(
            get_config(), headers=headers, rows=rows
        )
    date_fields = _date_fields_for_ui(
        artifacts_dir=arts,
        date_ocr=side.get("date_ocr") or ocr.get("date_ocr") or {},
        base_date=side.get("base_date") or ocr.get("base_date"),
        date_source=side.get("date_source") or ocr.get("date_source"),
    )
    return {
        "ok": bool(side.get("ok")),
        "id": side.get("id"),
        "title": side.get("title"),
        "tag": side.get("tag"),
        **date_fields,
        "point_count": point_count,
        "ocr_filled": ocr_filled,
        "expected_cells": expected_cells,
        "gap_filled": gap_filled,
        "gap_fill_map": gap_fill_map,
        "reason": side.get("reason"),
        "message": side.get("message"),
        "quality": side.get("quality"),
        "image_path": side.get("image_path") or "",
        "image_url": _web_url(side.get("image_path")),
        "artifacts_dir": arts,
        "artifacts_url": _web_url(arts),
        "best_table_path": best,
        "best_table_url": _web_url(best),
        "candidate_urls": [_web_url(c) for c in cands if _web_url(c)],
        "candidates": candidates,
        "excel_path": ocr.get("excel_path") or "",
        "excel_url": _web_url(ocr.get("excel_path")),
        "headers": headers,
        "rows": rows,
        "row_tags": list(ocr.get("row_tags") or []),
        "points": list(side.get("points") or []),
        "ocr_ms": side.get("ocr_ms"),
    }


def build_run_record(
    result: Dict[str, Any],
    *,
    mode: str,
    trigger: str = "manual",
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """从 pipeline 返回结果组装履历记录。"""
    finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    artifacts_dir = result.get("artifacts_dir") or ""
    if not artifacts_dir and isinstance(result.get("ocr"), dict):
        artifacts_dir = (result.get("ocr") or {}).get("artifacts_dir") or ""
    run_id = _run_id(mode, artifacts_dir or None)

    capture = result.get("capture") or {}
    image_path = (
        capture.get("image_path")
        or result.get("image_path")
        or ""
    )
    source_copy = result.get("source_copy") or ""
    if not source_copy and isinstance(result.get("ocr"), dict):
        source_copy = (result.get("ocr") or {}).get("source_copy") or ""

    excel_path = result.get("excel_path") or ""
    ocr = result.get("ocr") or {}
    if not excel_path:
        excel_path = ocr.get("excel_path") or ""

    quality = result.get("quality") or ocr.get("quality") or {}
    pi = result.get("pi") or {}
    # 本机上传原始文件名（展示用）；落盘预览仍用 source_copy / 00_source.*
    upload_name = str(
        result.get("upload_name")
        or ocr.get("upload_name")
        or ""
    ).strip()
    display_name = upload_name
    if not display_name:
        for cand in (image_path, source_copy):
            if not cand:
                continue
            base = Path(str(cand)).name
            if base and not base.lower().startswith("00_source"):
                display_name = base
                break
        if not display_name:
            display_name = Path(str(source_copy or image_path or "")).name or ""

    record: Dict[str, Any] = {
        "id": run_id,
        "mode": "dual" if mode == "dual" else "single",
        "trigger": trigger,
        "started_at": started_at or finished,
        "finished_at": finished,
        "ok": bool(result.get("ok")),
        "reason": result.get("reason") or quality.get("reason"),
        "message": result.get("message") or quality.get("message"),
        "point_count": int(
            result.get("point_count")
            or quality.get("point_count")
            or len(ocr.get("points") or [])
            or len(result.get("points") or [])
            or 0
        ),
        "min_points": int(
            result.get("min_points")
            or quality.get("min_points")
            or (get_config().get("ocr") or {}).get("min_points")
            or 8
        ),
        "upload_name": upload_name,
        "display_name": display_name,
        "capture_path": image_path,
        "capture_url": _web_url(image_path),
        "source_path": source_copy,
        "source_url": _web_url(source_copy) or _web_url(image_path),
        "artifacts_dir": artifacts_dir,
        "artifacts_url": _web_url(artifacts_dir),
        "excel_path": excel_path,
        "excel_url": _web_url(excel_path),
        "pi": {
            "ok": pi.get("ok"),
            "written": pi.get("written", 0),
            "total": pi.get("total", 0),
            "skipped": bool(pi.get("skipped")),
            "reason": pi.get("reason"),
            "message": pi.get("message"),
            "sides_written": pi.get("sides_written"),
            "sides_skipped": pi.get("sides_skipped"),
        },
        "quality": quality,
        "timing": result.get("timing") or {},
        "ocr_ms": (result.get("timing") or {}).get("ocr_ms"),
        "total_ms": (result.get("timing") or {}).get("total_ms"),
        "capture_ms": (result.get("timing") or {}).get("capture_ms"),
        "pi_ms": (result.get("timing") or {}).get("pi_ms"),
    }

    if record["mode"] == "dual":
        left = _side_summary(result.get("left"))
        right = _side_summary(result.get("right"))
        record["left_ok"] = bool(result.get("left_ok", left and left.get("ok")))
        record["right_ok"] = bool(result.get("right_ok", right and right.get("ok")))
        record["left"] = left
        record["right"] = right
        # 列表缩略用点数汇总
        record["point_count_left"] = int((left or {}).get("point_count") or 0)
        record["point_count_right"] = int((right or {}).get("point_count") or 0)
        record["point_count"] = record["point_count_left"] + record["point_count_right"]
        record["ocr_filled_left"] = int((left or {}).get("ocr_filled") or 0)
        record["ocr_filled_right"] = int((right or {}).get("ocr_filled") or 0)
        record["ocr_filled"] = record["ocr_filled_left"] + record["ocr_filled_right"]
        record["expected_cells_left"] = int((left or {}).get("expected_cells") or 0)
        record["expected_cells_right"] = int((right or {}).get("expected_cells") or 0)
        record["expected_cells"] = (
            record["expected_cells_left"] + record["expected_cells_right"]
        )
    else:
        best = ocr.get("best_crop_path") or ""
        if not best and artifacts_dir:
            cand = resolve_project_path(artifacts_dir) / "best_table.jpg"
            if cand.is_file():
                best = str(cand)
        cands = list(ocr.get("candidate_paths") or [])
        headers = list(ocr.get("headers") or [])
        rows = [list(r) for r in (ocr.get("rows") or [])]
        record["headers"] = headers
        record["rows"] = rows
        record["row_tags"] = list(ocr.get("row_tags") or [])
        record["gap_fill_map"] = [
            [str(c or "") for c in row]
            for row in (ocr.get("gap_fill_map") or result.get("gap_fill_map") or [])
        ]
        record["gap_filled"] = int(
            ocr.get("gap_filled") or result.get("gap_filled") or 0
        )
        if record["gap_filled"] <= 0 and record["gap_fill_map"]:
            record["gap_filled"] = sum(
                1
                for row in record["gap_fill_map"]
                for c in row
                if c and str(c).strip()
            )
        record["points"] = list(ocr.get("points") or result.get("points") or [])
        record["best_table_path"] = best
        record["best_table_url"] = _web_url(best)
        record["candidate_urls"] = [_web_url(c) for c in cands if _web_url(c)]
        record["candidates"] = _candidates_for_ui(ocr)
        ocr_filled = int(ocr.get("ocr_filled") or result.get("ocr_filled") or 0)
        if ocr_filled <= 0:
            ocr_filled = sum(
                1 for row in rows for cell in row if cell and str(cell).strip()
            )
        record["ocr_filled"] = ocr_filled
        record["expected_cells"] = get_ocr_expected_cells(
            get_config(), headers=headers, rows=rows
        )
        record.update(
            _date_fields_for_ui(
                artifacts_dir=artifacts_dir,
                date_ocr=ocr.get("date_ocr") or result.get("date_ocr") or {},
                base_date=ocr.get("base_date") or result.get("base_date"),
                date_source=ocr.get("date_source") or result.get("date_source"),
            )
        )

    return record


def save_run_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """写入 runs/{id}.json，若有 artifacts 目录则同步一份 run.json。"""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = str(record.get("id") or _run_id(str(record.get("mode") or "single")))
    record["id"] = run_id
    text = json.dumps(record, ensure_ascii=False, indent=2)
    index_path = RUNS_DIR / f"{run_id}.json"
    index_path.write_text(text, encoding="utf-8")
    record["record_path"] = str(index_path)
    record["record_url"] = _web_url(index_path)

    arts = record.get("artifacts_dir") or ""
    if arts:
        try:
            ad = resolve_project_path(arts)
            ad.mkdir(parents=True, exist_ok=True)
            (ad / "run.json").write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("写入 artifacts/run.json 失败: %s", arts)
    return record


def persist_pipeline_result(
    result: Dict[str, Any],
    *,
    mode: str,
    trigger: str = "manual",
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """组装并落盘履历，把 run 摘要挂回 result。"""
    try:
        record = build_run_record(
            result, mode=mode, trigger=trigger, started_at=started_at
        )
        save_run_record(record)
        result["run_id"] = record["id"]
        result["run"] = {
            "id": record["id"],
            "ok": record["ok"],
            "trigger": record["trigger"],
            "finished_at": record["finished_at"],
        }
    except Exception:
        logger.exception("落盘执行履历失败")
    return result


def collect_run_points(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从履历取出可写 PI 的点（双屏仅成功侧）。"""
    mode = str(data.get("mode") or "single").lower()
    if mode == "dual":
        points: List[Dict[str, Any]] = []
        for side_key in ("left", "right"):
            side = data.get(side_key) or {}
            if not side.get("ok"):
                continue
            for p in side.get("points") or []:
                if isinstance(p, dict) and p.get("timestamp") is not None:
                    points.append(p)
        return points
    return [
        p
        for p in (data.get("points") or [])
        if isinstance(p, dict) and p.get("timestamp") is not None
    ]


def run_ocr_succeeded(data: Dict[str, Any]) -> bool:
    """OCR 识别成功：已产出可写点（双屏至少一侧达标）。"""
    return bool(collect_run_points(data))


def _has_capture_image(data: Dict[str, Any]) -> bool:
    return bool(
        data.get("source_url")
        or data.get("capture_url")
        or data.get("source_path")
        or data.get("capture_path")
        or data.get("artifacts_dir")
    )


def classify_pi_status(data: Dict[str, Any]) -> str:
    """
    写 PI 状态（仅 OCR 成功时有意义）：
    - ok：已写入
    - unwritten：未写入（联调口子，尚未尝试写库）
    - fail：写失败
    - none：OCR/抓拍未成功，不涉及写 PI
    """
    if not run_ocr_succeeded(data):
        return "none"
    pi = data.get("pi") or {}
    if pi.get("ok") is True:
        return "ok"
    if pi.get("ok") is False and not pi.get("skipped"):
        return "fail"
    # skipped / ok=null → 未写入（调试补录口子）
    return "unwritten"


def _dual_one_side_ok(data: Dict[str, Any]) -> bool:
    """双屏仅一侧 OCR/可写达标（另一侧失败）。"""
    if str(data.get("mode") or "") != "dual":
        return False
    left = data.get("left_ok")
    right = data.get("right_ok")
    if left is not None or right is not None:
        return bool(left) != bool(right)
    return str(data.get("reason") or "").strip().lower() == "partial_side_failed"


def classify_result_status(data: Dict[str, Any]) -> str:
    """
    执行结果：
    - ok：抓拍+OCR+写PI 三步完成（双屏两侧均达标）
    - partial_ok：双屏仅一侧达标且该侧已写 PI
    - capture_fail：抓拍失败
    - ocr_fail：OCR 识别失败
    - pi_fail：写 PI 失败（与 pi_status=fail 同一回事）
    - pending_pi：OCR 成功但未写 PI（联调）
    """
    reason = str(data.get("reason") or "").strip().lower()
    if reason.startswith("capture") or reason in {
        "camera_error",
        "camera_failed",
        "capture_failed",
        "capture_error",
    }:
        return "capture_fail"
    if not _has_capture_image(data) and not bool(data.get("ok")):
        return "capture_fail"

    pi_st = classify_pi_status(data)
    if not run_ocr_succeeded(data):
        return "ocr_fail"
    if pi_st == "ok":
        if _dual_one_side_ok(data):
            return "partial_ok"
        return "ok"
    if pi_st == "fail":
        return "pi_fail"
    return "pending_pi"


def is_pi_rewrite_candidate(data: Dict[str, Any]) -> bool:
    """OCR 成功且尚未写成功 PI → 可补录（未写入或写失败）。"""
    return classify_pi_status(data) in ("unwritten", "fail")


def _status_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    pi = data.get("pi") or {}
    pi_status = classify_pi_status(data)
    result_status = classify_result_status(data)
    return {
        "result_status": result_status,
        "pi_status": pi_status,
        "pi_ok": pi.get("ok"),
        "pi_skipped": pi_status == "unwritten",
        "pi_reason": pi.get("reason"),
        "pi_written": pi.get("written"),
        "pi_total": pi.get("total"),
        "pi_pending": is_pi_rewrite_candidate(data),
        "ocr_ok": run_ocr_succeeded(data),
    }


def list_runs(
    *,
    limit: int = 50,
    offset: int = 0,
    mode: Optional[str] = None,
    ok: Optional[bool] = None,
    result: Optional[str] = None,
    day: Optional[str] = None,
    pi: Optional[str] = None,
    order: str = "desc",
) -> Dict[str, Any]:
    """扫描 runs/*.json。默认按完成时间倒序；order=asc 为时间线升序（旧→新）。

    result: ok|partial_ok|capture_fail|ocr_fail|pi_fail|pending_pi
    pi: ok|unwritten|fail（unwritten 仅未写入；fail 仅写失败；pending=未写入+写失败）
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    pi_filter = (pi or "").strip().lower() or None
    result_filter = (result or "").strip().lower() or None
    items: List[Dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("无法读取 run 记录: %s", f)
            continue
        if mode and str(data.get("mode") or "") != mode:
            continue
        if day:
            finished = str(data.get("finished_at") or data.get("started_at") or "")
            day_compact = day.replace("-", "")
            if day_compact not in finished.replace("-", "").replace(" ", "").replace(":", ""):
                if not finished.startswith(day):
                    continue
        st = _status_summary(data)
        rs = st["result_status"]
        ps = st["pi_status"]

        if result_filter:
            if rs != result_filter:
                continue
        elif ok is not None:
            # 兼容旧 ok 布尔：成功=三步完成；失败=其余
            if ok and rs != "ok":
                continue
            if not ok and rs == "ok":
                continue

        if pi_filter:
            if pi_filter in ("pending",):
                if not st["pi_pending"]:
                    continue
            elif pi_filter in ("unwritten", "skipped"):
                if ps != "unwritten":
                    continue
            elif pi_filter == "ok":
                if ps != "ok":
                    continue
            elif pi_filter == "fail":
                if ps != "fail":
                    continue
            elif ps != pi_filter:
                continue

        items.append(
            {
                "id": data.get("id"),
                "mode": data.get("mode"),
                "trigger": data.get("trigger"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "ok": rs == "ok",
                "left_ok": data.get("left_ok"),
                "right_ok": data.get("right_ok"),
                "reason": data.get("reason"),
                "message": data.get("message"),
                "point_count": data.get("point_count"),
                "point_count_left": data.get("point_count_left"),
                "point_count_right": data.get("point_count_right"),
                "ocr_filled": data.get("ocr_filled"),
                "ocr_filled_left": data.get("ocr_filled_left"),
                "ocr_filled_right": data.get("ocr_filled_right"),
                "expected_cells": data.get("expected_cells"),
                "expected_cells_left": data.get("expected_cells_left"),
                "expected_cells_right": data.get("expected_cells_right"),
                "min_points": data.get("min_points"),
                "capture_url": data.get("capture_url") or data.get("source_url"),
                "source_url": data.get("source_url"),
                "capture_path": data.get("capture_path") or "",
                "source_path": data.get("source_path") or "",
                "upload_name": data.get("upload_name") or "",
                "display_name": data.get("display_name")
                or data.get("upload_name")
                or "",
                "ocr_ms": data.get("ocr_ms")
                if data.get("ocr_ms") is not None
                else (data.get("timing") or {}).get("ocr_ms"),
                "total_ms": data.get("total_ms")
                if data.get("total_ms") is not None
                else (data.get("timing") or {}).get("total_ms"),
                **st,
            }
        )
    # 按业务时间线排序（finished_at），保证补录旧→新
    items.sort(
        key=lambda i: str(
            i.get("finished_at") or i.get("started_at") or i.get("id") or ""
        ),
        reverse=str(order or "desc").strip().lower() != "asc",
    )
    total = len(items)
    page = items[offset : offset + max(1, limit)]
    return {"ok": True, "total": total, "offset": offset, "limit": limit, "items": page}


def apply_pi_write_result(
    run_id: str,
    write: Dict[str, Any],
    *,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """把一次写 PI 结果回写到履历（联调写 PI / 首页补录共用）。"""
    data = get_run(run_id)
    if not data:
        return {"ok": False, "id": run_id, "reason": "not_found", "message": "执行记录不存在"}

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if error:
        data["pi"] = {
            **(data.get("pi") or {}),
            "ok": False,
            "written": int((write or {}).get("written") or 0),
            "total": int((write or {}).get("total") or 0),
            "skipped": False,
            "reason": "rewrite_error",
            "message": str(error),
            "rewritten_at": now,
        }
        data["pi_ms"] = None
        data["ok"] = False
        data["reason"] = "pi_write_failed"
        data["message"] = str(error)
        save_run_record(data)
        return {
            "ok": False,
            "id": run_id,
            "reason": "rewrite_error",
            "message": str(error),
            "pi": data["pi"],
            "pi_status": "fail",
            "result_status": "pi_fail",
        }

    write = write or {}
    write_ok = bool(write.get("ok"))
    data["pi"] = {
        "ok": write_ok,
        "written": int(write.get("written") or 0),
        "total": int(write.get("total") or 0),
        "skipped": False,
        "reason": None if write_ok else "pi_write_failed",
        "message": None if write_ok else "写 PI 未全部成功",
        "sides_written": write.get("sides_written"),
        "sides_skipped": write.get("sides_skipped"),
        "rewritten_at": now,
    }
    if str(data.get("mode") or "").lower() == "dual":
        written_sides = [
            s for s in ("left", "right") if bool((data.get(s) or {}).get("ok"))
        ]
        data["pi"]["sides_written"] = written_sides if write_ok else []
        data["pi"]["sides_skipped"] = [
            s for s in ("left", "right") if s not in written_sides
        ]
    # OCR 已成功时，写库成功则整体标成功（三步完成）
    if write_ok:
        data["ok"] = True
        data["reason"] = None
        data["message"] = None
    else:
        data["ok"] = False
        data["reason"] = "pi_write_failed"
        data["message"] = data["pi"].get("message") or "写 PI 失败"
    save_run_record(data)
    return {
        "ok": write_ok,
        "id": run_id,
        "written": write.get("written"),
        "total": write.get("total"),
        "pi": data["pi"],
        "pi_status": classify_pi_status(data),
        "result_status": classify_result_status(data),
        "message": None if write_ok else "写 PI 未全部成功",
    }


def rewrite_run_pi(run_id: str) -> Dict[str, Any]:
    """对单条履历用已存点数补写 PI，并更新落盘状态。"""
    from .pipeline import write_points_to_pi

    data = get_run(run_id)
    if not data:
        return {"ok": False, "id": run_id, "reason": "not_found", "message": "执行记录不存在"}
    if not is_pi_rewrite_candidate(data):
        status = classify_pi_status(data)
        return {
            "ok": False,
            "id": run_id,
            "reason": "not_eligible",
            "pi_status": status,
            "message": "非「OCR 成功且未写 PI」记录，跳过",
        }
    points = collect_run_points(data)
    try:
        write = write_points_to_pi(points, tag=None)
    except Exception as e:
        logger.exception("补写 PI 失败: %s", run_id)
        return apply_pi_write_result(
            run_id,
            {"written": 0, "total": len(points)},
            error=str(e),
        )
    return apply_pi_write_result(run_id, write)


def rewrite_pending_pi(
    *,
    mode: Optional[str] = None,
    day: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """一键补录：按时间线旧→新依次写 PI，最新记录最后写。"""
    listed = list_runs(
        limit=max(1, min(int(limit), 500)),
        offset=0,
        mode=mode,
        day=day,
        pi="pending",
        order="asc",
    )
    results: List[Dict[str, Any]] = []
    success = 0
    failed = 0
    for item in listed.get("items") or []:
        rid = str(item.get("id") or "")
        if not rid:
            continue
        logger.info(
            "补写 PI 按时间线 id=%s finished_at=%s",
            rid,
            item.get("finished_at") or item.get("started_at"),
        )
        one = rewrite_run_pi(rid)
        results.append(one)
        if one.get("ok"):
            success += 1
        else:
            failed += 1
    return {
        "ok": failed == 0,
        "total": len(results),
        "success": success,
        "failed": failed,
        "items": results,
    }


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    path = RUNS_DIR / f"{run_id}.json"
    data: Optional[Dict[str, Any]] = None
    if not path.is_file():
        # 兼容：从 ocr_runs 里找 run.json
        root = resolve_ocr_artifacts_root()
        for mode in ("single", "dual"):
            # id 形如 dual_20260916_185634_404
            stamp = run_id
            prefix = f"{mode}_"
            if run_id.startswith(prefix):
                stamp = run_id[len(prefix) :]
            cand = root / mode / stamp / "run.json"
            if cand.is_file():
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                except Exception:
                    logger.exception("读取 %s 失败", cand)
                break
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("读取 run 失败: %s", path)
            return None
    if data is None:
        return None
    return _enrich_run_date_fields(data)
