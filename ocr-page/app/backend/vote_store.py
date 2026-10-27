# -*- coding: utf-8 -*-
"""跨轮 OCR 众数计票（已公布时段）→ 写 PI 用多数决值。

规则（与产品约定一致）：
- 本机时间 ≥ 表头时刻：计票；每轮用当前众数胜出值写 PI；1 票也可胜出
- 本机时间 < 表头时刻：预测段，不计票，OCR 值直通
- 空值 / 无效解析 / 质量门未过的轮次：不计票（调用方只传入已达标点）
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config, get_vote_cfg, resolve_vote_tallies_dir

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_LOCK = threading.RLock()


def normalize_time_label(label: Any) -> Optional[str]:
    """表头时刻 → 规范 HH:MM（含 24:00）；非法则 None。"""
    if label is None:
        return None
    m = _TIME_RE.match(str(label).strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h == 24 and mi == 0:
        return "24:00"
    if h < 0 or h > 23 or mi < 0 or mi > 59:
        return None
    return f"{h:02d}:{mi:02d}"


def label_minutes(label: Any) -> Optional[int]:
    """表头 → 当日分钟数；24:00 → 1440。"""
    norm = normalize_time_label(label)
    if not norm:
        return None
    m = _TIME_RE.match(norm)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h == 24 and mi == 0:
        return 24 * 60
    return h * 60 + mi


def is_published_slot(time_label: Any, now: Optional[datetime] = None) -> bool:
    """本机时间 ≥ 表头时刻 → 已公布，可计票。"""
    mins = label_minutes(time_label)
    if mins is None:
        return False
    now = now or datetime.now()
    now_mins = now.hour * 60 + now.minute
    return now_mins >= mins


def normalize_value_key(value: Any) -> Optional[str]:
    """数值 → 计票键（去掉尾零，最多 4 位小数）。"""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    r = round(v, 4)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    s = f"{r:.4f}".rstrip("0").rstrip(".")
    return s or "0"


def value_from_key(key: str) -> float:
    return float(key)


def slot_key(tag: str, time_label: Any) -> Optional[str]:
    tag_s = (tag or "").strip()
    lab = normalize_time_label(time_label)
    if not tag_s or not lab:
        return None
    return f"{tag_s}|{lab}"


def _day_key(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d")


def _tally_path(day: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    root = resolve_vote_tallies_dir(cfg)
    return root / f"{day}.json"


def _empty_doc(day: str) -> Dict[str, Any]:
    return {"date": day, "updated_at": None, "slots": {}}


def load_tallies(day: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取某日票箱；不存在则返回空结构。"""
    day = str(day).replace("-", "")[:8]
    path = _tally_path(day, cfg)
    if not path.is_file():
        return _empty_doc(day)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("读取票箱失败 %s: %s", path, e)
        return _empty_doc(day)
    if not isinstance(raw, dict):
        return _empty_doc(day)
    slots = raw.get("slots")
    if not isinstance(slots, dict):
        slots = {}
    return {
        "date": str(raw.get("date") or day),
        "updated_at": raw.get("updated_at"),
        "slots": slots,
    }


def save_tallies(doc: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Path:
    day = str(doc.get("date") or _day_key()).replace("-", "")[:8]
    doc = dict(doc)
    doc["date"] = day
    doc["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = _tally_path(day, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _slot_counts(slot: Any) -> Dict[str, int]:
    if not isinstance(slot, dict):
        return {}
    counts = slot.get("counts")
    if isinstance(counts, dict):
        out: Dict[str, int] = {}
        for k, n in counts.items():
            try:
                out[str(k)] = int(n)
            except (TypeError, ValueError):
                continue
        return out
    # 兼容扁平 { "344.32": 3 }
    out = {}
    for k, n in slot.items():
        if k in ("counts", "winner", "total", "updated_at"):
            continue
        try:
            out[str(k)] = int(n)
        except (TypeError, ValueError):
            continue
    return out


def pick_winner(
    counts: Dict[str, int],
    *,
    previous_winner: Optional[str] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """次数最多者胜；并列时优先保留原胜出值（避免打平被本轮误识别翻盘），再字典序稳定。"""
    if not counts:
        return None, None
    best_n = max(int(n) for n in counts.values())
    cands = sorted(k for k, n in counts.items() if int(n) == best_n)
    if previous_winner and previous_winner in cands:
        win = previous_winner
    else:
        win = cands[0]
    try:
        return win, value_from_key(win)
    except ValueError:
        return win, None


def _rank_counts(counts: Dict[str, int]) -> List[Dict[str, Any]]:
    items = sorted(
        ((k, int(n)) for k, n in counts.items()),
        key=lambda kn: (-kn[1], kn[0]),
    )
    return [{"value": k, "count": n} for k, n in items]


def apply_majority_votes(
    points: List[Dict[str, Any]],
    *,
    record: bool = True,
    now: Optional[datetime] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对写 PI 点列表应用众数策略。

    record=True：本轮有效观测计入票箱（正式识别写库前）。
    record=False：仅按现有票箱取胜出值（补写 PI，避免重复计票）。

    返回 (写库用点列表, vote 摘要)；展示用 OCR 点应由调用方另行保留。
    """
    cfg = cfg or get_config()
    vote_cfg = get_vote_cfg(cfg)
    meta: Dict[str, Any] = {
        "enabled": bool(vote_cfg.get("enabled", True)),
        "date": None,
        "recorded": bool(record),
        "slots": {},
        "past": 0,
        "forecast": 0,
        "skipped": 0,
    }
    if not meta["enabled"]:
        return list(points or []), meta

    now = now or datetime.now()
    day = _day_key(now)
    meta["date"] = f"{day[:4]}-{day[4:6]}-{day[6:8]}"

    out: List[Dict[str, Any]] = []
    with _LOCK:
        doc = load_tallies(day, cfg)
        slots: Dict[str, Any] = dict(doc.get("slots") or {})
        dirty = False

        for raw in points or []:
            if not isinstance(raw, dict):
                continue
            p = dict(raw)
            tag = str(p.get("tag") or "").strip()
            label = p.get("time_label")
            lab = normalize_time_label(label)
            vkey = normalize_value_key(p.get("value"))
            if not tag or not lab or vkey is None:
                meta["skipped"] += 1
                out.append(p)
                continue

            try:
                ocr_val = float(vkey)
            except ValueError:
                meta["skipped"] += 1
                out.append(p)
                continue

            p["ocr_value"] = ocr_val
            sk = slot_key(tag, lab)
            assert sk is not None

            if not is_published_slot(lab, now):
                meta["forecast"] += 1
                p["vote_mode"] = "forecast"
                # 预测段：值保持 OCR，不入票箱
                out.append(p)
                meta["slots"][sk] = {
                    "tag": tag,
                    "time_label": lab,
                    "mode": "forecast",
                    "ocr_value": ocr_val,
                    "write_value": ocr_val,
                    "counts": {},
                    "rank": [],
                    "winner": None,
                    "total": 0,
                }
                continue

            meta["past"] += 1
            slot = slots.get(sk) if isinstance(slots.get(sk), dict) else {}
            counts = _slot_counts(slot)
            prev_winner = str(slot.get("winner") or "") or None
            if record:
                counts[vkey] = int(counts.get(vkey) or 0) + 1
                dirty = True
            elif not counts:
                # 补写且尚无票：退回本轮 OCR，并记 1 票视图（不落盘，除非 record）
                counts = {vkey: 1}

            win_key, win_val = pick_winner(counts, previous_winner=prev_winner)
            if win_val is None:
                win_val = ocr_val
                win_key = vkey

            total = sum(int(n) for n in counts.values())
            slots[sk] = {
                "counts": dict(counts),
                "winner": win_key,
                "total": total,
                "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if record:
                dirty = True

            p["value"] = float(win_val)
            p["vote_mode"] = "majority"
            p["vote_counts"] = dict(counts)
            p["vote_winner"] = win_key
            out.append(p)

            meta["slots"][sk] = {
                "tag": tag,
                "time_label": lab,
                "mode": "majority",
                "ocr_value": ocr_val,
                "write_value": float(win_val),
                "counts": dict(counts),
                "rank": _rank_counts(counts),
                "winner": win_key,
                "total": total,
            }

        if dirty:
            doc["slots"] = slots
            save_tallies(doc, cfg)
        else:
            # 补写路径也把最新 slots 快照带回
            doc["slots"] = slots

    return out, meta


def tallies_for_api(day: Optional[str] = None) -> Dict[str, Any]:
    """详情页 / API：某日票箱（含 rank）。"""
    cfg = get_config()
    vote_cfg = get_vote_cfg(cfg)
    if day:
        day_s = str(day).replace("-", "")[:8]
    else:
        day_s = _day_key()
    doc = load_tallies(day_s, cfg)
    slots_out: Dict[str, Any] = {}
    for sk, slot in (doc.get("slots") or {}).items():
        counts = _slot_counts(slot)
        win_key, _ = pick_winner(counts)
        tag, _, lab = sk.partition("|")
        slots_out[sk] = {
            "tag": tag,
            "time_label": lab,
            "mode": "majority",
            "counts": counts,
            "rank": _rank_counts(counts),
            "winner": win_key,
            "total": sum(counts.values()),
        }
    return {
        "ok": True,
        "enabled": bool(vote_cfg.get("enabled", True)),
        "date": f"{day_s[:4]}-{day_s[4:6]}-{day_s[6:8]}" if len(day_s) == 8 else day_s,
        "updated_at": doc.get("updated_at"),
        "slots": slots_out,
    }
