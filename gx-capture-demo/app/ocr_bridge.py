# -*- coding: utf-8 -*-
"""调用 image-table-extract 做表格 OCR，产出整表多行时段-数值列表。"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

# 现货 UI 日期筛选框常见写法；OCR 常把 2026 读成 026、2O26 等
_DATE_YMD_RE = re.compile(
    r"(20\d{2})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})\s*日?"
)
_DATE_FUZZY_YMD_RE = re.compile(
    r"(?<!\d)(\d{2,4})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})\s*日?(?!\d)"
)
_DATE_COMPACT_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
# 仅缺少年-月分隔、月日本身合法：202609/16（不把 00 猜成 09）
_DATE_GLUED_YM_D_RE = re.compile(
    r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])\s*[/\-.]\s*(\d{1,2})(?!\d)"
)
_DATE_MD_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[/\-.]\s*(\d{1,2})(?!\d)")
_DATE_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# 左上角「日期」标签常见 OCR 误读（反光/摩尔纹）——仅标签宽松，日期值另走严格解析
_DATE_LABEL_MISREADS = frozenset(
    {
        "日期",
        "曰期",
        "日朋",
        "日朝",
        "日欺",
        "日其",
        "目期",
        "自期",
        "自网",
        "日网",
        "门国",
        "门网",
        "门线",
        "门期",
    }
)


def _clamp_crop(
    x0: float, y0: float, x1: float, y1: float
) -> Tuple[float, float, float, float]:
    x0 = max(0.0, min(1.0, float(x0)))
    y0 = max(0.0, min(1.0, float(y0)))
    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    if x1 - x0 < 0.05:
        x1 = min(1.0, x0 + 0.05)
    if y1 - y0 < 0.03:
        y1 = min(1.0, y0 + 0.03)
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))


def _date_crop_neighbors(
    primary: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """
    主裁剪 + 略放大一档。
    放大=裁剪区域四周扩一点（斜拍时主框易切不全）；不再下移/右移（易滑进曲线轴）。
    """
    x0, y0, x1, y1 = (float(v) for v in primary)
    raw = [
        (x0, y0, x1, y1),
        (x0 - 0.03, y0 - 0.03, x1 + 0.04, y1 + 0.04),  # 略放大裁剪区域
    ]
    seen = set()
    out: List[Tuple[float, float, float, float]] = []
    for c in raw:
        box = _clamp_crop(*c)
        if box in seen:
            continue
        seen.add(box)
        out.append(box)
    return out


# 日期解析置信：完整 YYYY/MM/DD > 残缺年修复 > 年+月日拆分 > 仅月日
_DATE_CONF_FULL = 3
_DATE_CONF_FUZZY = 2
_DATE_CONF_SPLIT = 2
_DATE_CONF_MD = 1


@dataclass
class OcrPoint:
    time_label: str  # 如 18:15
    value: float
    timestamp: datetime
    tag: str = ""  # 如 GX.DEMO.OCR_1
    row_index: int = 0  # 数据行 0-based


@dataclass
class OcrResult:
    image_path: str
    headers: List[str]
    rows: List[List[str]]
    points: List[OcrPoint]
    row_tags: List[str] = field(default_factory=list)
    data_row_index: int = -1  # 兼容旧字段：主数据行（通常末行）
    artifacts_dir: str = ""  # OCR 过程图目录
    best_crop_path: str = ""  # 竞选胜出的表格截图
    candidate_paths: List[str] = field(default_factory=list)
    # 各裁剪候选的 OCR 结果与胜负说明（供详情页）
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    # 清洗前可解析填格数（OCR 识别数）；采纳数 = len(points)
    ocr_filled: int = 0
    # 胜出后同伴空格回填：格内为来源候选名（与 rows 同形，未补录为空串）
    gap_fill_map: List[List[str]] = field(default_factory=list)
    gap_filled: int = 0
    # 业务日：画面左上角日期 OCR（履历展示用）；写库时间戳按 pi_date_back_days 提前
    base_date: str = ""
    date_source: str = ""  # ocr | failed | disabled
    date_ocr: Dict[str, Any] = field(default_factory=dict)
    # 分阶段耗时（毫秒）：date_ms / table_ms / 各候选 ocr_ms
    timing: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    reason: str = ""
    message: str = ""


def _ensure_ocr_path(workdir: Path) -> None:
    w = str(workdir.resolve())
    if w not in sys.path:
        sys.path.insert(0, w)


def _crop_label(crop: Optional[Tuple[float, float, float, float]]) -> str:
    """未登记命名的裁剪回退标签（配置追加裁剪等）。"""
    if crop is None:
        return "table_auto"
    if crop == (0.0, 0.0, 1.0, 1.0):
        return "full"
    return "r{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(*crop).replace(".", "p")


def _cand_display_name(label: str) -> str:
    """规范展示名：cand_table_fixed / cand_table_auto …"""
    lab = (label or "").strip() or "unknown"
    return lab if lab.startswith("cand_") else f"cand_{lab}"


# 半屏/整图内默认固定表区（可被 config table_crop 覆盖）
_DEFAULT_TABLE_CROP_DUAL = (0.02, 0.52, 0.985, 0.90)
_DEFAULT_TABLE_CROP_SINGLE = (0.10, 0.52, 0.995, 0.90)


def _default_crop_candidates(
    layout: str,
    table_crop: Optional[Tuple[float, float, float, float]] = None,
) -> List[Tuple[str, Optional[Tuple[float, float, float, float]]]]:
    """
    表格 OCR 仅两个候选：先 auto（浅蓝表体+扩边），异常再回退固定比例框。
    固定框相对当前送入图（双屏=半屏，单屏=整图）。
    两路裁完后都进同一套纠偏再 OCR（斜拍时 fixed 同样必须纠偏）。
    """
    mode = "dual" if str(layout).strip().lower() == "dual" else "single"
    fixed = table_crop or (
        _DEFAULT_TABLE_CROP_DUAL if mode == "dual" else _DEFAULT_TABLE_CROP_SINGLE
    )
    return [
        ("table_auto", None),
        ("table_fixed", fixed),
    ]


def _materialize_crop_bgr(
    extractor: Any,
    image_path: str,
    crop_ratio: Optional[Tuple[float, float, float, float]],
    *,
    deskew: bool = False,
) -> Any:
    """
    按比例裁剪得到 BGR（未 scale）。
    deskew 仅表格预览用；日期定位/识别禁止纠偏（否则易把搜索带扭到曲线区）。
    """
    img = extractor.read_image_file(str(image_path))
    if crop_ratio is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = crop_ratio
        img = img[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)].copy()
    else:
        # 与 extract(crop_ratio=None) 同一路径，保证 last_auto_crop_ratio = 实际裁剪框
        img = extractor._resolve_table_auto_crop(img)
    if not deskew:
        return img
    try:
        return extractor._deskew_table_image(img)
    except Exception:
        return img


def _save_bgr(path: Path, bgr: Any) -> Optional[str]:
    try:
        import cv2  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), bgr)
        return str(path) if ok else None
    except Exception:
        logger.exception("保存 OCR 过程图失败: %s", path)
        return None


def _parse_float(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().replace(",", "").replace(" ", "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    if not s or s == "-":
        return None
    body0 = s[1:] if s.startswith("-") and len(s) > 1 else s
    digits0 = re.sub(r"\D", "", body0)
    if len(digits0) >= 10:
        return None
    if "." in body0 and len(re.sub(r"\D", "", body0.split(".", 1)[0])) >= 7:
        return None
    sign = -1.0 if s.startswith("-") and len(s) > 1 else 1.0
    body = s[1:] if (s.startswith("-") and len(s) > 1) else s
    # 漏点格式恢复：6 位 → xxx.xxx；5 位按量级
    if body.isdigit() and len(body) == 6:
        try:
            return sign * float(body[:3] + "." + body[3:])
        except ValueError:
            pass
    if body.isdigit() and len(body) == 5:
        try:
            if int(body[:2]) <= 40:
                return sign * float(body[:2] + "." + body[2:])
            return sign * float(body[:3] + "." + body[3:])
        except ValueError:
            pass
    try:
        v = float(s)
    except ValueError:
        return None
    if abs(v) >= 1_000_000.0:
        return None
    return v


def _label_to_datetime(label: str, base_date: Optional[datetime] = None) -> datetime:
    """将 18:15 / 24:00 映射到写库日（或次日 00:00）时间戳。"""
    base = base_date or datetime.now().replace(second=0, microsecond=0)
    day = base.date()
    m = _TIME_RE.match(label.strip())
    if not m:
        return base
    h, mi = int(m.group(1)), int(m.group(2))
    if h == 24 and mi == 0:
        return datetime.combine(day, datetime.min.time()) + timedelta(days=1)
    if h >= 24:
        h = h % 24
    return datetime.combine(day, datetime.min.time()).replace(hour=h, minute=mi)


def _pi_write_base_date(ocr_date: datetime, back_days: Optional[int] = None) -> datetime:
    """
    写 PI 用业务日 = OCR 识别日向前退 back_days 天（ocr.pi_date_back_days）。
    1=提前一天；0=当天不提前。再与表头时刻拼接。
    """
    if back_days is None:
        from .config import get_pi_date_back_days

        back_days = get_pi_date_back_days()
    try:
        n = max(0, int(back_days))
    except (TypeError, ValueError):
        n = 1
    return ocr_date - timedelta(days=n)


def _ymd_to_datetime(year: int, month: int, day: int) -> Optional[datetime]:
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def _repair_ocr_year(raw: str) -> Optional[int]:
    """修复 OCR 年份残缺：026→2026、2O24→2024、24→2024。"""
    s = (
        str(raw)
        .strip()
        .replace("O", "0")
        .replace("o", "0")
        .replace("l", "1")
        .replace("I", "1")
        .replace("丨", "1")
    )
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return None
    if re.fullmatch(r"20\d{2}", s):
        y = int(s)
    elif re.fullmatch(r"0\d{2}", s):
        y = int("2" + s)  # 024→2024, 026→2026
    elif re.fullmatch(r"\d{2}", s) and 20 <= int(s) <= 39:
        y = 2000 + int(s)
    elif re.fullmatch(r"2\d{3}", s):
        y = int(s)
    else:
        return None
    if 2020 <= y <= 2039:
        return y
    return None


def _normalize_date_ocr_text(text: str) -> str:
    """全角符号与常见数字误读归一；兼容现场 2026-08-12 与演示 2026/08/12。"""
    s = str(text or "")
    if not s:
        return ""
    s = s.translate(
        str.maketrans(
            {
                "／": "/",
                "－": "-",
                "—": "-",
                "–": "-",
                "−": "-",
                "：": ":",
                "．": ".",
                "O": "0",
                "o": "0",
                "О": "0",
                "〇": "0",
                "l": "1",
                "I": "1",
                "丨": "1",
            }
        )
    )
    # 日期框右侧日历图标常被读成「口」粘在串尾
    s = re.sub(r"口+$", "", s.strip())
    return s


def _date_format_preference(texts: Any) -> int:
    """
    业务日来自筛选框，优先 2026-08-12 / 2026/08/12；
    顶栏「2026年09月17日」虽也能解析，但常是系统日，优先级最低。
    """
    if isinstance(texts, str):
        blob = texts
    else:
        blob = " ".join(str(t) for t in (texts or []))
    s = _normalize_date_ocr_text(blob)
    if re.search(r"20\d{2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}", s):
        return 2
    if "年" in s and "月" in s:
        return 0
    return 1


def _parse_glued_ym_d(src: str) -> Optional[Tuple[datetime, int]]:
    """
    解析仅缺少年-月分隔、月日数字本身合法的 20xxMM/DD（如 202609/16）。
    不猜测、不改写任何一位数字；月非法（如 00）直接拒绝。
    """
    for m in _DATE_GLUED_YM_D_RE.finditer(src):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # 日至少两位，避免 202609/1 残缺串误采纳
        if len(m.group(3)) < 2:
            continue
        dt = _ymd_to_datetime(y, mo, d)
        if dt is not None:
            return dt, _DATE_CONF_FULL
    return None


def parse_ui_date_scored(
    text: str, *, allow_md_only: bool = True
) -> Optional[Tuple[datetime, int]]:
    """
    解析业务日期并返回置信档：
    3=完整 20xx/MM/DD（或等价紧凑/粘连且月日合法），
    2=残缺年修复或年+月日拆分，
    1=仅月日（本机年）。
    不猜测被读错的月/日数字。
    """
    if not text:
        return None
    s = _normalize_date_ocr_text(str(text).strip())
    if not s:
        return None
    compact = re.sub(r"\s+", "", s)

    for src in (s, compact):
        for m in _DATE_YMD_RE.finditer(src):
            dt = _ymd_to_datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if dt is not None:
                return dt, _DATE_CONF_FULL
        for m in _DATE_COMPACT_RE.finditer(src):
            dt = _ymd_to_datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if dt is not None:
                return dt, _DATE_CONF_FULL
        glued = _parse_glued_ym_d(src)
        if glued is not None:
            return glued
        for m in _DATE_FUZZY_YMD_RE.finditer(src):
            year = _repair_ocr_year(m.group(1))
            if year is None:
                continue
            dt = _ymd_to_datetime(year, int(m.group(2)), int(m.group(3)))
            if dt is not None:
                # 已是完整 20xx 则仍算 FULL
                raw_y = re.sub(r"[^\d]", "", str(m.group(1)))
                conf = (
                    _DATE_CONF_FULL
                    if re.fullmatch(r"20\d{2}", raw_y or "")
                    else _DATE_CONF_FUZZY
                )
                return dt, conf

    years = [_repair_ocr_year(m.group(1)) for m in _DATE_YEAR_RE.finditer(s + " " + compact)]
    years = [y for y in years if y is not None]
    for m in re.finditer(r"(?<!\d)(0\d{2})(?!\d)", compact):
        y = _repair_ocr_year(m.group(1))
        if y is not None:
            years.append(y)
    md_hits: List[Tuple[int, int]] = []
    for src in (s, compact):
        for m in _DATE_MD_RE.finditer(src):
            mo, d = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                md_hits.append((mo, d))
    if years and md_hits:
        dt = _ymd_to_datetime(years[0], md_hits[0][0], md_hits[0][1])
        if dt is not None:
            return dt, _DATE_CONF_SPLIT

    if allow_md_only and md_hits:
        mo, d = md_hits[0]
        dt = _ymd_to_datetime(datetime.now().year, mo, d)
        if dt is not None:
            return dt, _DATE_CONF_MD
    return None


def parse_ui_date_text(
    text: str, *, require_year: bool = False
) -> Optional[datetime]:
    """从 OCR 文本中解析业务日期（兼容年份残缺与月日拆分）。"""
    hit = parse_ui_date_scored(text, allow_md_only=not require_year)
    return hit[0] if hit else None


def _pick_best_date_hit(
    hits: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """置信优先；同置信优先筛选框格式(YYYY-MM-DD|/|)；再用多数；再并列取主裁剪。"""
    if not hits:
        return None
    from collections import Counter

    best_conf = max(int(h.get("confidence") or 0) for h in hits)
    tier = [h for h in hits if int(h.get("confidence") or 0) == best_conf]
    best_fmt = max(_date_format_preference(h.get("texts") or []) for h in tier)
    tier = [h for h in tier if _date_format_preference(h.get("texts") or []) == best_fmt]
    counts = Counter(str(h["date_str"]) for h in tier)
    best_str, _ = counts.most_common(1)[0]
    # 同置信多日期：多数辅助
    top_n = counts[best_str]
    tied = {k for k, n in counts.items() if n == top_n}
    pool = [h for h in tier if str(h["date_str"]) in tied]
    pool.sort(key=lambda h: int(h.get("crop_index", 99)))
    # 若并列日期不同，优先出现在主框上的日期
    if len(tied) > 1:
        by_idx = sorted(pool, key=lambda h: int(h.get("crop_index", 99)))
        winner = dict(by_idx[0])
    else:
        winner = dict(pool[0])
    winner["confidence"] = best_conf
    winner["vote_count"] = counts[str(winner["date_str"])]
    winner["vote_total"] = len(tier)
    return winner


def _is_chinese_ymd_value(text: Any) -> bool:
    """是否像顶栏「2026年09月22日」日期值（不是「日期」标签）。"""
    t = str(text or "").strip()
    if not t:
        return False
    # 年+月+日 连写，视为日期值，绝不能当标签
    if re.search(r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", t):
        return True
    cn = re.sub(r"[^\u4e00-\u9fff]", "", t)
    if cn in ("年月日", "年月", "月日") or (len(cn) <= 4 and "年" in cn and "月" in cn):
        return True
    return False


def _is_date_label_text(text: Any) -> bool:
    """
    是否像左上角「日期」筛选标签。
    必须像标签本身；顶栏「2026年09月22日」等日期值一律不算标签。
    """
    t = str(text or "").strip().replace(" ", "").replace("\u3000", "")
    if not t:
        return False
    # 顶栏中文日期值：禁止当标签（此前「日」+数字会误命中）
    if _is_chinese_ymd_value(t):
        return False
    if "日期" in t:
        # 「日期2026-09-22」粘连仍算标签；纯年月日已在上面拦掉
        return True
    head = re.split(r"[：:]", t, maxsplit=1)[0]
    head_cn = re.sub(r"[^\u4e00-\u9fff]", "", head)
    if head_cn in _DATE_LABEL_MISREADS:
        return True
    if t in _DATE_LABEL_MISREADS:
        return True
    # 极短误读：须含「期」（日朝/曰期），单「日」不够（避免 年月日）
    if len(head_cn) <= 3 and "期" in head_cn and "年" not in head_cn:
        return True
    # 合并框：标签误读 + 后面跟数字；必须有「期」，不能仅凭「日」
    if (
        len(head_cn) <= 4
        and head_cn
        and "期" in head_cn
        and "年" not in head_cn
        and re.search(r"\d", t)
    ):
        return True
    return False


def _locate_crop_looks_like_filter(texts: Any) -> bool:
    """
    定位裁剪是否像业务筛选框：应有「日期」标签，或筛出 YYYY-MM-DD|/|。
    仅有顶栏「yyyy年mm月dd日」不够格。
    """
    if isinstance(texts, str):
        blob_list = [texts]
    else:
        blob_list = [str(t) for t in (texts or [])]
    blob = " ".join(blob_list)
    # 明确有标签
    if any(_is_date_label_text(t) for t in blob_list):
        return True
    if "日期" in blob.replace(" ", ""):
        return True
    # 筛选框格式优先
    if _date_format_preference(blob_list) >= 2:
        return True
    # 只有中文年月日 → 顶栏，拒绝
    if _is_chinese_ymd_value(blob) or any(_is_chinese_ymd_value(t) for t in blob_list):
        return False
    return False


def _iter_ocr_boxes(
    ocr_result: Any,
) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """解析 PaddleOCR 结果为 (text, x0,y0,x1,y1)。"""
    out: List[Tuple[str, Tuple[float, float, float, float]]] = []
    if not ocr_result:
        return out
    for item in ocr_result:
        if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box, info = item[0], item[1]
        text = info[0] if isinstance(info, (list, tuple)) else info
        text_s = str(text or "").strip()
        if not text_s:
            continue
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            out.append((text_s, (min(xs), min(ys), max(xs), max(ys))))
        except Exception:
            continue
    return out


def _pick_date_label_box(
    boxes: List[Tuple[str, Tuple[float, float, float, float]]],
    *,
    region_w: float,
    region_h: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    在 OCR 框中选「日期」标签：优先上半、偏左、文本更短（纯「日期」优于长句）。
    坐标相对当前搜索裁剪图。
    """
    candidates: List[Tuple[float, float, float, float, float, str]] = []
    for text, box in boxes:
        if not _is_date_label_text(text):
            continue
        x0, y0, x1, y1 = box
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        # 过滤明显不在上半的命中
        if region_h > 1 and cy > region_h * 0.72:
            continue
        if region_w > 1 and cx > region_w * 0.85:
            continue
        # score 越小越好：上、左、短
        score = (cy / max(region_h, 1.0)) * 3.0 + (cx / max(region_w, 1.0)) + len(
            text.replace(" ", "")
        ) * 0.02
        candidates.append((score, x0, y0, x1, y1, text))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    _, x0, y0, x1, y1, _ = candidates[0]
    return (x0, y0, x1, y1)


def _filter_ratio_from_label_box(
    label_box: Tuple[float, float, float, float],
    *,
    img_w: int,
    img_h: int,
    search: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    将搜索裁剪内的「日期」标签框，扩成整图相对比例的「标签+右侧筛选框」裁剪。
    """
    sx0, sy0, sx1, sy1 = (float(v) for v in search)
    lx0, ly0, lx1, ly1 = label_box
    # 映射到整图像素
    ax0 = sx0 * img_w + lx0
    ay0 = sy0 * img_h + ly0
    ax1 = sx0 * img_w + lx1
    ay1 = sy0 * img_h + ly1
    lw = max(8.0, ax1 - ax0)
    lh = max(8.0, ay1 - ay0)
    # 含标签左侧一点 + 右侧日期框；宽度封顶避免吞掉半屏
    max_right = max(lw * 9.0, min(img_w * 0.28, lw * 14.0))
    rx0 = max(0.0, ax0 - lw * 0.35)
    # 少向上（避免吞进标题行），向下严格封顶，防止扩进曲线 Y 轴
    ry0 = max(0.0, ay0 - lh * 0.8)
    rx1 = min(float(img_w), ax1 + max_right, rx0 + img_w * 0.42)
    ry1 = min(float(img_h), ay1 + lh * 1.4, ry0 + max(lh * 3.2, img_h * 0.08))
    # 过扁时略加高，但仍不得超过标签下方约 2.5 行
    min_h = max(80.0, lh * 2.2, img_h * 0.04)
    max_h = max(min_h, lh * 3.5, img_h * 0.09)
    cur_h = ry1 - ry0
    if cur_h < min_h:
        need = min_h - cur_h
        up = need * 0.2
        down = need - up
        ry0 = max(0.0, ry0 - up)
        ry1 = min(float(img_h), ry1 + down, ry0 + max_h)
    elif cur_h > max_h:
        ry1 = ry0 + max_h
    return _clamp_crop(
        rx0 / img_w, ry0 / img_h, rx1 / img_w, ry1 / img_h
    )


def _filter_ratio_from_value_box(
    value_box: Tuple[float, float, float, float],
    *,
    img_w: int,
    img_h: int,
    search: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """无「日期」标签时：围住已识别出的年月日文本框（含左侧标签位）。"""
    sx0, sy0, sx1, sy1 = (float(v) for v in search)
    vx0, vy0, vx1, vy1 = value_box
    ax0 = sx0 * img_w + vx0
    ay0 = sy0 * img_h + vy0
    ax1 = sx0 * img_w + vx1
    ay1 = sy0 * img_h + vy1
    vw = max(8.0, ax1 - ax0)
    vh = max(8.0, ay1 - ay0)
    rx0 = max(0.0, ax0 - vw * 1.2)  # 左侧留「日期」标签空间
    ry0 = max(0.0, ay0 - vh * 0.7)
    rx1 = min(float(img_w), ax1 + vw * 0.45)
    ry1 = min(float(img_h), ay1 + vh * 0.7)
    min_h = max(100.0, img_h * 0.045)
    cur_h = ry1 - ry0
    if cur_h < min_h:
        pad = (min_h - cur_h) * 0.5
        ry0 = max(0.0, ry0 - pad)
        ry1 = min(float(img_h), ry0 + min_h)
        if ry1 - ry0 < min_h:
            ry0 = max(0.0, ry1 - min_h)
    return _clamp_crop(
        rx0 / img_w, ry0 / img_h, rx1 / img_w, ry1 / img_h
    )


def _pick_date_value_box(
    boxes: List[Tuple[str, Tuple[float, float, float, float]]],
    *,
    region_w: float,
    region_h: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    无「日期」标签时：只认筛选框格式 YYYY-MM-DD|/|。
    顶栏「yyyy年mm月dd日」一律不参与定位（不是业务筛选日）。
    """
    candidates: List[Tuple[float, Tuple[float, float, float, float]]] = []
    for text, box in boxes:
        if _is_chinese_ymd_value(text):
            continue
        fmt = _date_format_preference(text)
        if fmt < 2:
            continue
        scored = parse_ui_date_scored(text, allow_md_only=False) or parse_ui_date_scored(
            re.sub(r"\s+", "", text), allow_md_only=False
        )
        if scored is None:
            continue
        dt, conf = scored
        if conf < _DATE_CONF_FULL:
            continue
        if dt.year < 2018 or dt.year > 2100:
            continue
        x0, y0, x1, y1 = box
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        if region_h > 1 and cy > region_h * 0.75:
            continue
        if region_w > 1 and cx > region_w * 0.75:
            continue
        score = (
            (cy / max(region_h, 1.0)) * 2.0
            + (cx / max(region_w, 1.0))
            - conf * 0.1
            - fmt * 0.5
        )
        candidates.append((score, box))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def _date_label_search_regions(
    layout: str,
) -> List[Tuple[float, float, float, float]]:
    """
    「日期」搜索带。
    单屏：固定整图左上四分之一（日期必在此区），再在区内找标签/筛选框。
    双屏：相对已裁半屏的左上（保持原带，不动现场布局）。
    """
    mode = "dual" if str(layout).strip().lower() == "dual" else "single"
    if mode == "dual":
        return [
            (0.0, 0.06, 0.72, 0.42),
            (0.0, 0.10, 0.60, 0.48),
            (0.0, 0.02, 0.80, 0.55),
        ]
    # 单屏：屏幕四等分左上；略留边躲开浏览器顶栏/侧栏
    return [
        (0.02, 0.02, 0.50, 0.50),
    ]


def _texts_look_like_chart_axis(texts: Any) -> bool:
    """
    裁剪是否像曲线图 Y 轴（单位MW / 300/250/200），而非日期筛选框。
    这类框绝不当日期候选。
    """
    if isinstance(texts, str):
        blob = texts
    else:
        blob = " ".join(str(t) for t in (texts or []))
    s = _normalize_date_ocr_text(blob)
    if not s:
        return False
    # 已有完整年月日 → 不是纯轴
    if parse_ui_date_scored(s, allow_md_only=False):
        return False
    axis_hits = 0
    if re.search(r"单位\s*\(?\s*M\s*W", s, re.I) or "单位MM" in s.replace(" ", ""):
        axis_hits += 2
    if "实时发电" in s or "中标" in s:
        axis_hits += 1
    for tok in ("300", "250", "200", "150", "100"):
        if re.search(rf"(?<!\d){tok}(?!\d)", s):
            axis_hits += 1
    # 多个轴刻度且无日期标签
    if axis_hits >= 3 and "日期" not in s and "期" not in s:
        return True
    if axis_hits >= 2 and re.search(r"单位", s):
        return True
    return False


def locate_date_filter_crop(
    extractor: Any,
    ocr: Any,
    image_path: str,
    *,
    layout: str = "single",
    debug_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    自动定位左上角「日期」标签（或日期值文本），返回筛选框相对裁剪。
    单屏：先在整图左上四分之一内搜索，再裁标签+右侧筛选框。
    成功时 dict 含 crop / label_box / search / image_path / method。
    """
    try:
        full = extractor.read_image_file(str(image_path))
    except Exception:
        logger.exception("读取原图失败，无法自动定位日期: %s", image_path)
        return None
    h, w = full.shape[:2]
    if h < 20 or w < 20:
        return None

    run_dir = Path(debug_dir) if debug_dir else None
    last_texts: List[str] = []
    value_fallback: Optional[Dict[str, Any]] = None

    for si, search in enumerate(_date_label_search_regions(layout)):
        band = _materialize_crop_bgr(extractor, image_path, search)
        if band is None or getattr(band, "size", 0) == 0:
            continue
        bh, bw = band.shape[:2]
        # 放大 + 去摩尔纹变体，提高「日期」小字命中
        mapped: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for use in _locate_band_variants(band, loc_scale=2.5):
            result = _collect_ocr_result(ocr, use)
            boxes = _iter_ocr_boxes(result)
            scale_x = bw / max(1, use.shape[1])
            scale_y = bh / max(1, use.shape[0])
            for text, (x0, y0, x1, y1) in boxes:
                mapped.append(
                    (
                        text,
                        (
                            x0 * scale_x,
                            y0 * scale_y,
                            x1 * scale_x,
                            y1 * scale_y,
                        ),
                    )
                )
                last_texts.append(text)

            label = _pick_date_label_box(
                mapped, region_w=float(bw), region_h=float(bh)
            )
            method = "label"
            box_used = label
            if box_used is None:
                box_used = _pick_date_value_box(
                    mapped, region_w=float(bw), region_h=float(bh)
                )
                method = "value"
            if box_used is not None:
                break
        else:
            continue

        if method == "label":
            crop = _filter_ratio_from_label_box(
                box_used, img_w=w, img_h=h, search=search
            )
        else:
            crop = _filter_ratio_from_value_box(
                box_used, img_w=w, img_h=h, search=search
            )

        crop_bgr = _materialize_crop_bgr(extractor, image_path, crop)
        # 定位框校验：曲线轴 / 顶栏中文日（无「日期」标签）一律丢掉
        if crop_bgr is not None and getattr(crop_bgr, "size", 0) > 0:
            try:
                probe_txt = _collect_ocr_texts(
                    ocr, _date_ocr_variants(crop_bgr)[0]
                )
                if _texts_look_like_chart_axis(probe_txt):
                    logger.info(
                        "日期定位框疑似曲线轴，跳过 search=%s crop=%s texts=%s",
                        search,
                        crop,
                        probe_txt[:8],
                    )
                    continue
                if not _locate_crop_looks_like_filter(probe_txt):
                    logger.info(
                        "日期定位框不像筛选框（缺「日期」/仅有顶栏年月日），跳过 "
                        "method=%s crop=%s texts=%s",
                        method,
                        crop,
                        probe_txt[:8],
                    )
                    # 假标签命中不当真；继续搜，不把 value 兜底建成错框
                    if method == "label":
                        continue
                    continue
            except Exception:
                pass
        saved = ""
        if run_dir is not None and crop_bgr is not None:
            saved = _save_bgr(run_dir / "date_locate.jpg", crop_bgr) or ""
            _save_bgr(run_dir / f"date_search_{si:02d}.jpg", band)

        hit = {
            "crop": crop,
            "label_box": list(box_used),
            "search": list(search),
            "image_path": saved,
            "texts": last_texts[:16],
            "method": method,
        }
        # 仅真正的「日期」标签立即返回；value 兜底须已通过筛选框校验
        if method == "label":
            logger.info(
                "自动定位日期筛选框 layout=%s method=label search=%s crop=%s",
                layout,
                search,
                crop,
            )
            return hit
        if value_fallback is None:
            value_fallback = hit

    if value_fallback is not None:
        logger.info(
            "自动定位日期筛选框 layout=%s method=value crop=%s",
            layout,
            value_fallback.get("crop"),
        )
        return value_fallback

    logger.warning(
        "未在上半区找到「日期」标签 image=%s texts=%s",
        image_path,
        last_texts[:12],
    )
    return None


def _crop_right_of_date_label(bgr: Any, ocr_result: Any) -> Optional[Any]:
    """若识别到「日期」标签，裁其右侧输入框一带再认一次。"""
    if not ocr_result:
        return None
    h, w = bgr.shape[:2]
    label_box = _pick_date_label_box(
        _iter_ocr_boxes(ocr_result), region_w=float(w), region_h=float(h)
    )
    if label_box is None:
        return None
    _, y0, x1, y1 = label_box
    # 标签右侧、同高一行：向右延伸约 8～12 倍标签宽
    lw = max(8.0, label_box[2] - label_box[0])
    lh = max(8.0, y1 - y0)
    rx0 = int(max(0, x1 - lw * 0.1))
    ry0 = int(max(0, y0 - lh * 0.4))
    rx1 = int(min(w, x1 + lw * 10))
    ry1 = int(min(h, y1 + lh * 0.8))
    if rx1 - rx0 < 20 or ry1 - ry0 < 10:
        return None
    return bgr[ry0:ry1, rx0:rx1].copy()


def _demoire_preprocess_gray(gray: Any) -> List[Any]:
    """
    屏幕拍照摩尔纹抑制候选（灰度图列表，已放大前/后均可）。
    只做图像增强，不改日期数字；解析仍走严格规则。
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    out: List[Any] = []
    try:
        # 1) 双边：保边去纹
        soft = cv2.bilateralFilter(gray, d=7, sigmaColor=55, sigmaSpace=55)
        out.append(soft)
        # 2) 轻高斯压周期条纹后再锐化笔画
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        sharp = cv2.addWeighted(gray, 1.55, blur, -0.55, 0)
        out.append(sharp)
        soft_sharp = cv2.addWeighted(soft, 1.5, cv2.GaussianBlur(soft, (0, 0), 1.0), -0.5, 0)
        out.append(soft_sharp)
        # 3) NL-means 去噪 + CLAHE
        try:
            dn = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(dn)
            out.append(clahe)
        except Exception:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(soft)
            out.append(clahe)
        # 4) 自适应二值：黑字白底更利小数字
        ath = cv2.adaptiveThreshold(
            soft,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        out.append(ath)
        # 5) Otsu 二值
        _, otsu = cv2.threshold(soft, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out.append(otsu)
        # 6) 频域低通：削弱屏幕网格高频（摩尔纹主因）
        try:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            # 半径随图尺寸：保留笔画低频，去掉细网格
            radius = max(12, min(rows, cols) // 8)
            cv2.circle(mask, (ccol, crow), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, radius / 6.0))
            img_back = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
            low = np.abs(img_back)
            low = np.clip(low, 0, 255).astype(np.uint8)
            out.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(low))
        except Exception:
            pass
    except Exception:
        logger.exception("去摩尔纹预处理失败")
    return out


def _date_ocr_variants(bgr: Any) -> List[Any]:
    """日期小字区域增强：原图 / 放大 / CLAHE / 去摩尔纹 / 二值。"""
    import cv2  # type: ignore

    out: List[Any] = [bgr]
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # 过矮裁剪多放大，减轻摩尔纹下 09→00
        h = int(gray.shape[0])
        fx = 4.0 if h < 120 else 3.0
        up = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        out.append(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
        out.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
        for g in _demoire_preprocess_gray(up):
            if g is None or getattr(g, "size", 0) == 0:
                continue
            if len(getattr(g, "shape", ())) == 2:
                out.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            else:
                out.append(g)
    except Exception:
        logger.exception("日期 OCR 预处理失败，仅用原图")
    # 去重：过多变体拖慢；保留前 8 个足够覆盖常见摩尔纹
    return out[:5]


def _locate_band_variants(band: Any, *, loc_scale: float = 2.5) -> List[Any]:
    """日期标签搜索带：放大 + 少量去纹变体（定位阶段也要抗摩尔纹）。"""
    import cv2  # type: ignore

    out: List[Any] = []
    try:
        use = cv2.resize(
            band, None, fx=loc_scale, fy=loc_scale, interpolation=cv2.INTER_CUBIC
        )
        out.append(use)
        gray = cv2.cvtColor(use, cv2.COLOR_BGR2GRAY)
        # 定位只要少量强变体，避免每个搜索带跑太久
        for g in _demoire_preprocess_gray(gray)[:3]:
            if g is None or getattr(g, "size", 0) == 0:
                continue
            if len(getattr(g, "shape", ())) == 2:
                out.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            else:
                out.append(g)
    except Exception:
        logger.exception("日期定位预处理失败")
        out = [band]
    return out[:4]


def _collect_ocr_result(ocr: Any, img: Any) -> Any:
    result, _ = ocr(img)
    return result


def _texts_from_ocr_result(result: Any) -> List[str]:
    texts: List[str] = []
    if not result:
        return texts
    for item in result:
        if not item:
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            t = item[1]
            if isinstance(t, (list, tuple)) and t:
                t = t[0]
            text = str(t or "").strip()
            if text:
                texts.append(text)
    return texts


def _collect_ocr_texts(ocr: Any, img: Any) -> List[str]:
    return _texts_from_ocr_result(_collect_ocr_result(ocr, img))


def extract_ui_date_from_image(
    image_path: str,
    workdir: Path,
    *,
    crop: Optional[Tuple[float, float, float, float]] = None,
    crops: Optional[List[Tuple[float, float, float, float]]] = None,
    scale: float = 2.5,
    min_confidence: float = 0.3,
    debug_dir: Optional[Path] = None,
    layout: str = "single",
    auto_locate: bool = True,
) -> Dict[str, Any]:
    """
    OCR 半屏/整图左上角「日期」筛选框。
    优先 OCR 找「日期」标签再裁其右侧筛选框；失败才用配置/默认框 + 邻域。
    完整 YYYY/MM/DD 立即采纳；多命中时置信优先。
    """
    _ensure_ocr_path(workdir)
    from extract_table import TableImageExtractor  # type: ignore

    crop_list: List[Tuple[float, float, float, float]] = []
    if crops:
        crop_list.extend(crops)
    if crop is not None:
        crop_list.insert(0, crop)

    payload: Dict[str, Any] = {
        "ok": False,
        "date": None,
        "date_str": None,
        "raw_texts": [],
        "crop": [],
        "image_path": "",
        "best_image_path": "",
        "candidate_image_paths": [],
        "reason": "",
        "attempts": [],
        "locate": None,
    }
    candidate_image_paths: List[str] = []
    hits: List[Dict[str, Any]] = []
    unique_crops: List[Tuple[float, float, float, float]] = []
    run_dir: Optional[Path] = None
    extractor: Any = None

    def _accept(hit: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
        payload["ok"] = True
        payload["date"] = hit["date"]
        payload["date_str"] = hit["date_str"]
        payload["crop"] = list(hit["crop"])
        payload["raw_texts"] = list(hit.get("texts") or [])[:12]
        payload["confidence"] = hit.get("confidence")
        payload["reason"] = reason
        payload["candidate_image_paths"] = list(candidate_image_paths)
        if run_dir is not None and extractor is not None:
            src = str(hit.get("image_path") or "")
            try:
                if src and Path(src).is_file():
                    best_path = run_dir / "date_crop_best.jpg"
                    best_path.write_bytes(Path(src).read_bytes())
                    payload["best_image_path"] = str(best_path)
                else:
                    bgr_best = _materialize_crop_bgr(
                        extractor, image_path, tuple(hit["crop"])  # type: ignore[arg-type]
                    )
                    payload["best_image_path"] = (
                        _save_bgr(run_dir / "date_crop_best.jpg", bgr_best) or ""
                    )
            except Exception:
                logger.exception("保存日期 best 失败")
        logger.info(
            "日期 OCR 成功 date=%s conf=%s reason=%s crop=%s image=%s",
            payload["date_str"],
            hit.get("confidence"),
            reason,
            hit["crop"],
            image_path,
        )
        return payload

    try:
        from .config import get_ocr_onnx_threads

        extractor = TableImageExtractor(
            scale=float(scale),
            min_confidence=float(min_confidence),
            onnx_threads=get_ocr_onnx_threads(),
        )
        ocr = extractor._get_ocr()
        run_dir = Path(debug_dir) if debug_dir else None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)

        located_crop: Optional[Tuple[float, float, float, float]] = None
        if auto_locate:
            located = locate_date_filter_crop(
                extractor,
                ocr,
                image_path,
                layout=layout,
                debug_dir=run_dir,
            )
            payload["locate"] = (
                {
                    k: located.get(k)
                    for k in ("crop", "search", "label_box", "image_path", "method")
                    if located
                }
                if located
                else {"ok": False}
            )
            if located and located.get("crop"):
                located_crop = tuple(float(x) for x in located["crop"])  # type: ignore[assignment]
                loc_img = str(located.get("image_path") or "")
                if loc_img:
                    candidate_image_paths.append(loc_img)

        # 自动定位框优先；配置框仅作第二阶段回退
        auto_ordered: List[Tuple[float, float, float, float]] = []
        fallback_ordered: List[Tuple[float, float, float, float]] = []
        if located_crop is not None:
            auto_ordered.append(located_crop)
            auto_ordered.extend(_date_crop_neighbors(located_crop)[1:])
        if crop_list:
            primary = crop_list[0]
            for c in list(crop_list) + _date_crop_neighbors(primary):
                fallback_ordered.append(
                    (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                )
        elif located_crop is None:
            fb = (
                (0.0, 0.16, 0.48, 0.26)
                if str(layout).strip().lower() == "dual"
                else (0.16, 0.18, 0.50, 0.30)
            )
            fallback_ordered.extend(_date_crop_neighbors(fb))

        def _dedupe(
            items: List[Tuple[float, float, float, float]],
        ) -> List[Tuple[float, float, float, float]]:
            out: List[Tuple[float, float, float, float]] = []
            seen_local: set = set()
            for c in items:
                key = tuple(round(float(x), 4) for x in c)
                if key in seen_local:
                    continue
                seen_local.add(key)
                out.append((float(c[0]), float(c[1]), float(c[2]), float(c[3])))
            return out

        auto_crops = _dedupe(auto_ordered)
        fallback_crops = _dedupe(fallback_ordered)
        # 回退列表去掉已在自动定位里的框
        auto_keys = {tuple(round(float(x), 4) for x in c) for c in auto_crops}
        fallback_crops = [
            c
            for c in fallback_crops
            if tuple(round(float(x), 4) for x in c) not in auto_keys
        ]
        unique_crops = list(auto_crops)
        # 先只跑自动定位；若整轮无命中再追加配置回退
        phases: List[List[Tuple[float, float, float, float]]] = []
        if auto_crops:
            phases.append(auto_crops)
        if fallback_crops:
            phases.append(fallback_crops)
        if not phases:
            phases.append(
                _date_crop_neighbors(
                    (0.16, 0.18, 0.50, 0.30)
                    if str(layout).strip().lower() != "dual"
                    else (0.0, 0.16, 0.48, 0.26)
                )
            )

        if unique_crops:
            payload["crop"] = list(unique_crops[0])
        elif phases and phases[0]:
            payload["crop"] = list(phases[0][0])

        all_texts: List[str] = []
        crop_offset = 0

        def _attempt_stage(phase_i: int, crop_i: int) -> str:
            """识别过程三阶段：自定位主框 / 放大尝试 / 配置退回。"""
            if phase_i == 0 and located_crop is not None:
                return "auto_main" if crop_i == 0 else "auto_enlarge"
            return "config_fallback"

        # 两阶段 × 两轮增强；完整年月日立即返回
        for phase_i, phase_crops in enumerate(phases):
            unique_crops = phase_crops
            for pass_i, enhance in enumerate((False, True)):
                for i, c in enumerate(unique_crops):
                    global_i = crop_offset + i
                    stage = _attempt_stage(phase_i, i)
                    bgr = _materialize_crop_bgr(extractor, image_path, c)
                    crop_saved = ""
                    # 命名区分三阶段，便于详情页归类
                    if run_dir is not None and pass_i == 0:
                        if stage == "auto_main":
                            crop_saved = (
                                _save_bgr(run_dir / "date_crop.jpg", bgr) or ""
                            )
                            if crop_saved:
                                payload["image_path"] = crop_saved
                        elif stage == "auto_enlarge":
                            crop_saved = (
                                _save_bgr(
                                    run_dir / f"date_crop_enlarge_{i:02d}.jpg", bgr
                                )
                                or ""
                            )
                        elif i == 0:
                            # 配置退回主框
                            crop_saved = (
                                _save_bgr(run_dir / "date_crop_cfg.jpg", bgr) or ""
                            )
                            if crop_saved and not payload.get("image_path"):
                                payload["image_path"] = crop_saved
                        else:
                            crop_saved = (
                                _save_bgr(
                                    run_dir / f"date_crop_cfg_enlarge_{i:02d}.jpg",
                                    bgr,
                                )
                                or ""
                            )
                        # 兼容旧扫描：额外写 date_crop_XX
                        if crop_saved and Path(crop_saved).name != "date_crop.jpg":
                            _save_bgr(run_dir / f"date_crop_{global_i:02d}.jpg", bgr)
                    if crop_saved:
                        candidate_image_paths.append(crop_saved)

                    # 首轮即含去摩尔纹变体；增强轮跑满候选
                    all_vars = _date_ocr_variants(bgr)
                    variants = all_vars if enhance else all_vars[:5]
                    attempt_texts: List[str] = []
                    best_local: Optional[Tuple[datetime, int, List[str]]] = None
                    first_result = None
                    crop_hits: List[Dict[str, Any]] = []
                    # 先看原图 OCR：明显是曲线 Y 轴则跳过本框（配置 date_crop 拍屏常偏到图区）
                    probe = _texts_from_ocr_result(
                        _collect_ocr_result(ocr, variants[0] if variants else bgr)
                    )
                    if _texts_look_like_chart_axis(probe):
                        payload["attempts"].append(
                            {
                                "crop": list(c),
                                "texts": probe[:12],
                                "ok": False,
                                "enhance": enhance,
                                "skipped": "chart_axis",
                                "image_path": crop_saved,
                                "phase": phase_i,
                                "stage": stage,
                                "crop_index": global_i,
                                "local_index": i,
                            }
                        )
                        all_texts.extend(probe)
                        continue
                    for vi, variant in enumerate(variants):
                        # 日期框已在变体内 3x 放大；勿再套表格大图 scale（易糊掉 09）
                        use = variant
                        result = _collect_ocr_result(ocr, use)
                        if vi == 0 and not enhance:
                            first_result = result
                        texts = _texts_from_ocr_result(result)
                        attempt_texts.extend(texts)
                        all_texts.extend(texts)
                        joined = " ".join(texts)
                        scored = parse_ui_date_scored(
                            joined, allow_md_only=False
                        ) or parse_ui_date_scored(
                            re.sub(r"\s+", "", joined), allow_md_only=False
                        )
                        if not (
                            scored
                            and scored[1] >= _DATE_CONF_FULL
                            and _date_format_preference(texts) >= 2
                        ):
                            continue
                        dt, conf = scored
                        crop_hits.append(
                            {
                                "date": dt,
                                "date_str": dt.strftime("%Y-%m-%d"),
                                "confidence": conf,
                                "texts": texts[:12],
                            }
                        )
                    if crop_hits:
                        # 同裁剪多变体：多数票，避免个别二值把 16 读成 10
                        from collections import Counter

                        vote = Counter(h["date_str"] for h in crop_hits)
                        best_str, _ = vote.most_common(1)[0]
                        chosen = next(h for h in crop_hits if h["date_str"] == best_str)
                        best_local = (
                            chosen["date"],
                            int(chosen["confidence"]),
                            list(chosen["texts"]),
                        )

                    # 定位主框：再裁右侧输入框区域（去掉标题/标签干扰）做去纹识别
                    if i == 0 and not enhance and best_local is None:
                        try:
                            import cv2  # type: ignore

                            hh, ww = bgr.shape[:2]
                            if ww > 40 and hh > 20:
                                right_only = bgr[:, int(ww * 0.22) :].copy()
                                for variant in _date_ocr_variants(right_only)[:5]:
                                    texts = _collect_ocr_texts(ocr, variant)
                                    attempt_texts.extend(texts)
                                    all_texts.extend(texts)
                                    joined = " ".join(texts)
                                    scored = parse_ui_date_scored(
                                        joined, allow_md_only=False
                                    ) or parse_ui_date_scored(
                                        re.sub(r"\s+", "", joined), allow_md_only=False
                                    )
                                    if (
                                        scored
                                        and scored[1] >= _DATE_CONF_FULL
                                        and _date_format_preference(texts) >= 2
                                    ):
                                        dt, conf = scored
                                        best_local = (dt, conf, texts)
                                        break
                        except Exception:
                            logger.exception("日期右侧框增强识别失败")

                    # 主裁剪：锚「日期」标签右侧输入框再认一次（仅主框尚未认全时）
                    if (
                        i == 0
                        and not enhance
                        and first_result is not None
                        and best_local is None
                    ):
                        anchored = _crop_right_of_date_label(bgr, first_result)
                        if anchored is not None:
                            if run_dir is not None:
                                ap = _save_bgr(
                                    run_dir / "date_crop_anchor.jpg", anchored
                                )
                                if ap:
                                    candidate_image_paths.append(ap)
                            for variant in _date_ocr_variants(anchored)[:5]:
                                texts = _collect_ocr_texts(ocr, variant)
                                attempt_texts.extend(texts)
                                all_texts.extend(texts)
                                joined = " ".join(texts)
                                scored = parse_ui_date_scored(
                                    joined, allow_md_only=False
                                ) or parse_ui_date_scored(
                                    re.sub(r"\s+", "", joined), allow_md_only=False
                                )
                                if not (
                                    scored
                                    and scored[1] >= _DATE_CONF_FULL
                                    and _date_format_preference(texts) >= 2
                                ):
                                    continue
                                dt, conf = scored
                                best_local = (dt, conf, texts)
                                break

                    img_ref = crop_saved
                    if not img_ref and candidate_image_paths and global_i < len(
                        candidate_image_paths
                    ):
                        img_ref = candidate_image_paths[global_i]

                    if best_local is not None:
                        dt, conf, texts = best_local
                        hit = {
                            "crop": list(c),
                            "crop_index": global_i,
                            "date": dt,
                            "date_str": dt.strftime("%Y-%m-%d"),
                            "confidence": conf,
                            "texts": texts[:12],
                            "enhance": enhance,
                            "image_path": img_ref or "",
                            "phase": phase_i,
                            "stage": stage,
                        }
                        hits.append(hit)
                        payload["attempts"].append(
                            {
                                "crop": list(c),
                                "texts": texts[:12],
                                "ok": True,
                                "enhance": enhance,
                                "date_str": hit["date_str"],
                                "confidence": conf,
                                "image_path": img_ref or "",
                                "phase": phase_i,
                                "stage": stage,
                                "crop_index": global_i,
                                "local_index": i,
                            }
                        )
                        reason_prefix = (
                            "auto_locate#"
                            if phase_i == 0 and located_crop is not None
                            else "crop#"
                        )
                        # 仅当筛到筛选框格式(YYYY-MM-DD|/|)才立刻返回；
                        # 顶栏「yyyy年mm月dd日」继续找，避免盖过业务日
                        if (
                            conf >= _DATE_CONF_FULL
                            and _date_format_preference(texts) >= 2
                        ):
                            return _accept(
                                hit, reason=f"full_ymd_{reason_prefix}{global_i}"
                            )
                    else:
                        fail_attempt: Dict[str, Any] = {
                            "crop": list(c),
                            "texts": attempt_texts[:12],
                            "ok": False,
                            "enhance": enhance,
                            "phase": phase_i,
                            "stage": stage,
                            "crop_index": global_i,
                            "local_index": i,
                        }
                        if img_ref:
                            fail_attempt["image_path"] = img_ref
                        payload["attempts"].append(fail_attempt)

                # 原图轮已有筛选框完整年月日，不必再跑增强
                if not enhance and any(
                    int(h.get("confidence") or 0) >= _DATE_CONF_FULL
                    and _date_format_preference(h.get("texts") or []) >= 2
                    for h in hits
                ):
                    break

            # 自动定位阶段已有可用命中，不再跑配置回退
            if phase_i == 0 and hits:
                break
            crop_offset += len(unique_crops)

        payload["candidate_image_paths"] = list(candidate_image_paths)
        payload["raw_texts"] = all_texts[:24]

        winner = _pick_best_date_hit(hits)
        if winner is not None:
            reason = f"best_conf={winner.get('confidence')}"
            if located_crop is not None and int(winner.get("crop_index") or -1) == 0:
                reason = "auto_locate;" + reason
            if int(winner.get("vote_total") or 0) > 1:
                reason += (
                    f";tiebreak_votes={winner.get('vote_count')}"
                    f"/{winner.get('vote_total')}"
                )
            return _accept(winner, reason=reason)

        joined_all = " ".join(all_texts)
        scored = parse_ui_date_scored(
            joined_all, allow_md_only=False
        ) or parse_ui_date_scored(
            re.sub(r"\s+", "", joined_all), allow_md_only=False
        )
        if scored is not None and scored[1] >= _DATE_CONF_FULL:
            dt, conf = scored
            hit = {
                "crop": list(unique_crops[0]) if unique_crops else [],
                "crop_index": 0,
                "date": dt,
                "date_str": dt.strftime("%Y-%m-%d"),
                "confidence": conf,
                "texts": all_texts[:12],
                "image_path": payload.get("image_path") or "",
            }
            return _accept(hit, reason="merged_texts")

        payload["reason"] = "date_not_found"
        logger.warning(
            "日期 OCR 未解析到合法日期 image=%s texts=%s",
            image_path,
            all_texts[:12],
        )
        return payload
    except Exception as e:
        logger.exception("日期 OCR 失败 image=%s", image_path)
        payload["reason"] = f"date_ocr_error:{e}"
        payload["candidate_image_paths"] = list(candidate_image_paths)
        return payload


def resolve_business_date(
    *,
    image_path: str,
    workdir: Path,
    fallback_date: Optional[datetime] = None,
    date_cfg: Optional[Dict[str, Any]] = None,
    min_confidence: float = 0.3,
    debug_dir: Optional[Path] = None,
) -> Tuple[Optional[datetime], str, Dict[str, Any]]:
    """
    业务日只认画面左上角日期 OCR；不再回退接口/本机日。
    fallback_date 保留参数兼容，忽略。
    返回 (date_or_None, source, date_ocr_payload)；source 为 ocr / failed / disabled。
    """
    del fallback_date
    from .config import get_ocr_date_cfg

    cfg = date_cfg if date_cfg is not None else get_ocr_date_cfg()
    if not cfg.get("enabled", True):
        return None, "disabled", {
            "ok": False,
            "reason": "date_ocr_disabled",
            "message": "日期 OCR 已关闭，无法识别业务日期",
        }

    primary = cfg.get("crop")
    if isinstance(primary, list):
        primary = tuple(float(x) for x in primary)
    if primary is None:
        layout = str(cfg.get("layout") or "single")
        primary = (
            (0.0, 0.16, 0.48, 0.26)
            if layout == "dual"
            else (0.16, 0.18, 0.50, 0.30)
        )
    extra = cfg.get("crops") or []
    crops: List[Tuple[float, float, float, float]] = [primary]  # type: ignore[list-item]
    for item in extra:
        if isinstance(item, str):
            from .config import parse_crop

            p = parse_crop(item)
            if p:
                crops.append(p)
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            crops.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))

    meta = extract_ui_date_from_image(
        image_path,
        workdir,
        crop=None,
        crops=crops,
        scale=float(cfg.get("scale", 1.0)),
        min_confidence=min_confidence,
        debug_dir=debug_dir,
        layout=str(cfg.get("layout") or "single"),
        auto_locate=bool(cfg.get("auto_locate", True)),
    )
    if meta.get("ok") and meta.get("date") is not None:
        return meta["date"], "ocr", meta

    reason = str(meta.get("reason") or "date_not_found")
    texts = list(meta.get("raw_texts") or [])[:8]
    msg = "未识别到左上角日期筛选框中的业务日期，已跳过表格 OCR"
    if texts:
        msg += f"；OCR 原文：{' / '.join(str(t) for t in texts)}"
    meta = dict(meta)
    meta["ok"] = False
    meta["reason"] = reason
    meta["message"] = msg
    return None, "failed", meta


def _empty_date_fail_result(
    *,
    image_path: str,
    run_dir: Optional[Path],
    artifacts: List[str],
    date_source: str,
    date_ocr_out: Dict[str, Any],
    base_date_str: str = "",
    timing: Optional[Dict[str, Any]] = None,
) -> OcrResult:
    reason = str(date_ocr_out.get("reason") or "date_ocr_failed")
    message = str(
        date_ocr_out.get("message")
        or "未识别到左上角日期筛选框中的业务日期，已跳过表格 OCR"
    )
    if run_dir is not None:
        try:
            (run_dir / "date_ocr.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "base_date": base_date_str or None,
                        "date_source": date_source,
                        "date_ocr": date_ocr_out,
                        "reason": reason,
                        "message": message,
                        "table_ocr_skipped": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "best_meta.txt").write_text(
                f"date_source={date_source}\n"
                f"reason={reason}\n"
                f"message={message}\n"
                f"table_ocr_skipped=1\n",
                encoding="utf-8",
            )
        except Exception:
            logger.exception("写日期失败产物失败")
    return OcrResult(
        image_path=str(image_path),
        headers=[],
        rows=[],
        points=[],
        row_tags=[],
        data_row_index=-1,
        artifacts_dir=str(run_dir) if run_dir else "",
        best_crop_path="",
        candidate_paths=artifacts,
        candidates=[],
        ocr_filled=0,
        base_date=base_date_str,
        date_source=date_source,
        date_ocr=date_ocr_out,
        timing=dict(timing or {}),
        ok=False,
        reason=reason,
        message=message,
    )


def row_tag_name(tag_prefix: str, row_index: int) -> str:
    """GX.DEMO.OCR + _1/_2/_3（row_index 0-based）。"""
    prefix = (tag_prefix or "GX.DEMO.OCR").strip().rstrip("_")
    return f"{prefix}_{row_index + 1}"


def _points_from_full_table(
    headers: List[str],
    rows: List[List[str]],
    tag_prefix: str,
    base_date: Optional[datetime] = None,
) -> Tuple[List[OcrPoint], List[str]]:
    """整表每行数据 → 对应 PI 测点；base_date 为写库日（已按提前天数处理）。"""
    points: List[OcrPoint] = []
    row_tags: List[str] = []
    for i, row in enumerate(rows):
        tag = row_tag_name(tag_prefix, i)
        row_tags.append(tag)
        for j, header in enumerate(headers):
            cell = row[j] if j < len(row) else ""
            val = _parse_float(cell)
            if val is None:
                continue
            label = header.strip()
            points.append(
                OcrPoint(
                    time_label=label,
                    value=val,
                    timestamp=_label_to_datetime(label, base_date),
                    tag=tag,
                    row_index=i,
                )
            )
    return points, row_tags


def _header_minutes_in_grid(label: str) -> Optional[int]:
    """全日表刻度 00:15~24:00（步长 15）→ 分钟数；否则 None。"""
    m = _TIME_RE.match(str(label).strip())
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh == 24 and mm == 0:
        return 24 * 60
    if mm not in (0, 15, 30, 45) or hh < 0 or hh > 23:
        return None
    total = hh * 60 + mm
    if total < 15:  # 早于 00:15（如 00:00）不在网格内
        return None
    return total


def _header_quality(headers: List[str]) -> int:
    """表头质量 = 合法列数 − 100×坏跳 − 50×非法点 − 80×倒退。

    业务：全表 00:15~24:00 每 15 分钟一格；滚动后可见窗可从任意合法刻度起。
    相邻必须恰好差 15 分钟。坏跳（含邻屏漏进的 03:15→00:15）重罚，
    避免「列数更多但乱序」压过少一格的干净表头。

    评分前丢掉首尾非法刻度（常见误识 00:00）：只按剩余合法链打分，
    避免「整表可用却因一列 00:00 得 -42、选不出胜者」。
    """
    if not headers or len(headers) < 4:
        return -1
    # 去掉首尾连续非法头，保留中间业务刻度窗
    trimmed = list(headers)
    while trimmed and _header_minutes_in_grid(trimmed[0]) is None:
        trimmed.pop(0)
    while trimmed and _header_minutes_in_grid(trimmed[-1]) is None:
        trimmed.pop()
    if len(trimmed) < 4:
        return -1
    parsed: List[Optional[int]] = [_header_minutes_in_grid(h) for h in trimmed]
    illegal = sum(1 for m in parsed if m is None)
    valid_cols = len(trimmed) - illegal
    bad_jumps = 0
    backward = 0
    for a, b in zip(parsed, parsed[1:]):
        if a is None or b is None:
            continue
        if b - a != 15:
            bad_jumps += 1
            if b < a:
                backward += 1
    return valid_cols - 100 * bad_jumps - 50 * illegal - 80 * backward


def _sanitize_headers_for_write(
    headers: List[str],
    rows: List[List[str]],
    *,
    min_run: int = 2,
) -> Tuple[List[str], List[List[str]]]:
    """落库前表头清洗（与候选打分分离）。

    1. 丢掉非法列（解不出 / 不在 00:15~24:00 网格，如 00:00）
    2. 在剩余合法列上找所有「相邻 Δ=15、长度 ≥min_run」片段，并集保留
       （中间断档时两端都留，不因只取最长段而丢掉另一头）
    3. 同分钟去重（保留先出现的列）
    4. 时刻倒退 / 串屏：后段起点落在前段终点及以前 → 冲突，保留更长段；
       等长则保留左侧段
    """
    if not headers:
        return headers, rows

    n = len(headers)
    mins: List[Optional[int]] = [_header_minutes_in_grid(h) for h in headers]
    valid_idx = [i for i in range(n) if mins[i] is not None]
    if not valid_idx:
        empty_rows = [[] for _ in rows]
        return [], empty_rows

    runs: List[List[int]] = []
    cur: List[int] = [valid_idx[0]]
    for j in range(1, len(valid_idx)):
        i_prev, i_cur = valid_idx[j - 1], valid_idx[j]
        m_prev = mins[i_prev]
        m_cur = mins[i_cur]
        if m_prev is not None and m_cur is not None and (m_cur - m_prev) == 15:
            cur.append(i_cur)
        else:
            runs.append(cur)
            cur = [i_cur]
    runs.append(cur)
    runs = [r for r in runs if len(r) >= min_run]

    selected: List[List[int]] = []
    for r in runs:
        r_start = mins[r[0]]
        if r_start is None:
            continue
        conflict_at = [
            k
            for k, s in enumerate(selected)
            if mins[s[-1]] is not None and r_start <= int(mins[s[-1]])
        ]
        if not conflict_at:
            selected.append(r)
            continue
        conflict_len = sum(len(selected[k]) for k in conflict_at)
        if len(r) > conflict_len:
            for k in reversed(conflict_at):
                selected.pop(k)
            selected.append(r)
        # else：丢掉本段（串屏/逆序/重叠中的较短侧）

    keep: List[int] = []
    seen_m: set[int] = set()
    for r in selected:
        for i in r:
            m = mins[i]
            if m is None or m in seen_m:
                continue
            seen_m.add(m)
            keep.append(i)

    if not keep:
        return [], [[] for _ in rows]
    if keep == list(range(n)):
        return list(headers), [list(r) for r in rows]

    new_headers = [headers[i] for i in keep]
    new_rows = [[row[i] if i < len(row) else "" for i in keep] for row in rows]
    return new_headers, new_rows


def _strip_illegal_header_columns(
    headers: List[str], rows: List[List[str]]
) -> Tuple[List[str], List[List[str]]]:
    """兼容旧名：落库表头清洗（多段并集 + 倒退/串屏处理）。"""
    return _sanitize_headers_for_write(headers, rows)


def _leading_empty_penalty(rows: List[List[str]]) -> int:
    """数据行前导空列惩罚：错位表头常见「左侧空、右侧才有数」。"""
    if not rows:
        return 0
    penalties: List[int] = []
    for row in rows:
        lead = 0
        for cell in row:
            if cell and str(cell).strip():
                break
            lead += 1
        # 整行空不算错位
        if lead >= len(row):
            continue
        penalties.append(lead)
    if not penalties:
        return 0
    return int(max(penalties))


def _crop_area_ratio(
    crop: Optional[Tuple[float, float, float, float]],
    *,
    auto_ratio: Optional[Tuple[float, float, float, float]] = None,
) -> float:
    """裁剪面积占比；auto 优先用实测框。"""
    box = crop if crop is not None else auto_ratio
    if box is None:
        return 0.55
    x0, y0, x1, y1 = box
    return max(0.01, float(x1 - x0) * float(y1 - y0))


# 字典序：(表头质量, 实心行, 密度, 紧凑度)；同分优先更紧裁剪
_SCORE_DIM_NAMES = ("表头质量", "实心行数", "填格密度", "裁剪紧凑度")


def _literal_decimal_count(rows: Sequence[Sequence[Any]]) -> int:
    """
    统计 OCR 结果中「原文带小数且小数位≥2」的格子数。
    只认识别结果里已有的点（如 7.853 / 10.351），不把补点臆造算进去。
    """
    n = 0
    for row in rows or []:
        for cell in row:
            if cell is None:
                continue
            s = str(cell).strip()
            if re.fullmatch(r"-?\d+\.\d{2,}", s):
                n += 1
    return n


def _row_type_mix_penalty(rows: Sequence[Sequence[Any]]) -> int:
    """
    电力/计划（整型）与电价（小数）混在同一行时的扣分。
    返回建议从「表头质量」扣掉的整数分，避免混行脏候选靠多表头压过干净候选。
    """
    pen = 0
    for row in rows or []:
        filled = [str(c).strip() for c in row if c is not None and str(c).strip()]
        if len(filled) < 4:
            continue
        dec = 0
        pure_int = 0
        for s in filled:
            if "." in s:
                dec += 1
            elif re.fullmatch(r"-?\d{2,4}", s):
                pure_int += 1
        n = len(filled)
        # 同行同时有明显小数簇与整型簇 → 混行
        if dec >= max(2, n // 5) and pure_int >= max(2, n // 5):
            pen += 4
        elif dec >= 2 and pure_int >= 2 and abs(dec - pure_int) <= 2:
            pen += 3
    return pen


def _row_numeric_median(row: Sequence[Any]) -> Optional[float]:
    """同行非空可解析数值的中位数（绝对值）；样本不足返回 None。"""
    vals: List[float] = []
    for cell in row:
        if cell is None:
            continue
        s = str(cell).strip()
        if not s or s in ("0", "0.0"):
            continue
        try:
            vals.append(abs(float(s)))
        except ValueError:
            continue
    if len(vals) < 3:
        return None
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float(vals[mid - 1] + vals[mid]) / 2.0


def _gap_fill_from_peer_candidates(
    headers: List[str],
    rows: List[List[str]],
    peers: Sequence[Dict[str, Any]],
    *,
    ratio_k: float = 3.0,
) -> Tuple[List[List[str]], int, List[List[str]]]:
    """
    用同表头其他候选回填胜出结果空格：只补空、不改已有值；不猜修。

    按时刻标签对齐列；同伴按填格数/密度从高到低取票（避免弱候选多数票盖过强候选）。
    同行量级超出 ratio_k 的票丢弃。
    @return (回填后矩阵, 本次新补格数, 同源矩阵；格内为来源候选展示名，未补录为空串)
    """
    ncols = len(headers) if headers else 0
    gap_map: List[List[str]] = [[""] * ncols for _ in (rows or [])]
    if not headers or not rows or not peers:
        return [list(r) for r in rows], 0, gap_map
    out = [list(r) for r in rows]
    filled_n = 0
    k = max(1.0, float(ratio_k))

    ranked: List[Tuple[Tuple[int, int], str, Dict[str, int], List[List[str]]]] = []
    for rec in peers:
        if str(rec.get("status") or "") == "winner":
            continue
        ph = [str(h).strip() for h in (rec.get("headers") or [])]
        pr = rec.get("rows") or []
        if not ph or not pr:
            continue
        idx_by_h: Dict[str, int] = {}
        for j, h in enumerate(ph):
            if h and h not in idx_by_h:
                idx_by_h[h] = j
        if not idx_by_h:
            continue
        pub = rec.get("score") or []
        try:
            dens = int(pub[3]) if len(pub) > 3 and pub[3] is not None else 0
        except (TypeError, ValueError):
            dens = 0
        try:
            pts = int(rec.get("point_count") or 0)
        except (TypeError, ValueError):
            pts = 0
        name = str(rec.get("name") or "").strip()
        if not name:
            name = _cand_display_name(str(rec.get("label") or "unknown"))
        ranked.append(((pts, dens), name, idx_by_h, [list(r) for r in pr]))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if not ranked:
        return out, 0, gap_map

    for ri, row in enumerate(out):
        med = _row_numeric_median(row)
        for ci, header in enumerate(headers):
            if ci >= len(row):
                continue
            if row[ci] and str(row[ci]).strip():
                continue
            h = str(header).strip()
            if not h:
                continue
            picked = ""
            src_name = ""
            for _rank, peer_name, idx_by_h, pr in ranked:
                pj = idx_by_h.get(h)
                if pj is None or ri >= len(pr) or pj >= len(pr[ri]):
                    continue
                raw = pr[ri][pj]
                if raw is None:
                    continue
                s = str(raw).strip()
                if not s:
                    continue
                try:
                    fv = abs(float(s))
                except ValueError:
                    continue
                if med is not None and med > 0 and (fv > med * k or fv < med / k):
                    continue
                picked = s
                src_name = peer_name
                break
            if not picked:
                continue
            out[ri][ci] = picked
            gap_map[ri][ci] = src_name
            filled_n += 1
    return out, filled_n, gap_map


def _score_tuple_public(sc: Tuple[Any, ...]) -> List[Any]:
    """落盘用：前三维整数，第四维改为正面积占比便于阅读。"""
    if not sc or len(sc) < 4:
        return list(sc) if sc else []
    area = -float(sc[3]) if sc[3] is not None else None
    return [sc[0], sc[1], sc[2], round(area, 4) if area is not None else None]


def _explain_winner(sc: Tuple[Any, ...], label: str) -> str:
    pub = _score_tuple_public(sc)
    return (
        f"综合最优（{label}）："
        f"表头质量={pub[0]}，实心行={pub[1]}，密度={pub[2]}；"
        f"同分时优先更紧裁剪"
    )


def _explain_loser(
    mine: Tuple[Any, ...],
    winner: Tuple[Any, ...],
    winner_label: str,
    *,
    invalid: bool = False,
    skipped: bool = False,
) -> str:
    if skipped:
        return f"未参与比较：已提前选出足够好的候选（胜出者 {winner_label}）"
    # sc: (hq, solid, density, -area)
    if invalid or (mine and len(mine) > 1 and mine[0] < 0 and mine[1] <= 0):
        return "未识别到有效表头/数据行，视为无效候选"
    if not winner or (len(winner) > 0 and winner[0] < 0):
        return "综合分低于其他候选"
    for i, name in enumerate(_SCORE_DIM_NAMES[:3]):
        if mine[i] < winner[i]:
            return (
                f"相对胜出者「{winner_label}」：{name}较弱"
                f"（本候选 {mine[i]} < 胜出 {winner[i]}）"
            )
    if len(mine) > 3 and len(winner) > 3 and mine[3] < winner[3]:
        return (
            f"相对胜出者「{winner_label}」：表结构相当，但裁剪更松"
            f"（优先更紧表区）"
        )
    return f"综合分低于胜出者「{winner_label}」"


def extract_points_from_image(
    image_path: str,
    workdir: Path,
    scale: float = 2.5,
    min_confidence: float = 0.3,
    crop: Optional[Tuple[float, float, float, float]] = None,
    tag_prefix: str = "GX.DEMO.OCR",
    base_date: Optional[datetime] = None,
    data_row_index: int = -1,  # 保留参数兼容旧调用，整表模式下忽略
    debug_dir: Optional[Path] = None,
    layout: str = "single",
    sanitize: Optional[Dict[str, Any]] = None,
    early_stop: bool = False,
    early_stop_min_density: int = 70,
    skip_full_when_ok: bool = False,
    max_ocr_width: Optional[int] = None,
    date_cfg: Optional[Dict[str, Any]] = None,
    table_crop: Optional[Tuple[float, float, float, float]] = None,
    onnx_threads: Optional[int] = None,
) -> OcrResult:
    """
    按 image-table-extract 逻辑做整表识别（第 1 行时刻表头，其余均为数据行）。
    裁剪候选顺序：table_auto（蓝底定位+扩边）→ table_fixed（配置比例回退）；
    两路均：裁剪 → 纠偏 → OCR；按表头质量 / 实心行 / 填格密度择优，同分优先更紧裁剪。
    early_stop=True 时：仅当先行候选三行数据格全部填满才跳过后续；否则两边都 OCR 再择优。
    skip_full_when_ok 已废弃（不再跑整图 full），保留参数兼容旧调用。

    业务日只认左上角日期 OCR（完整年月日）；识别失败则整次失败并跳过表格 OCR。
    """
    del data_row_index
    del skip_full_when_ok
    # crop 旧参：单屏曾把 capture.single.crop 当额外候选；现统一用 table_crop
    if table_crop is None and crop is not None:
        table_crop = crop
    _ensure_ocr_path(workdir)
    from extract_table import TableImageExtractor  # type: ignore

    t0 = time.perf_counter()
    layout_mode = "dual" if str(layout).strip().lower() == "dual" else "single"
    try:
        dens_need = int(early_stop_min_density)
    except (TypeError, ValueError):
        dens_need = 70
    dens_need = max(50, min(100, dens_need))
    extractor_kw: Dict[str, Any] = {
        "scale": scale,
        "min_confidence": min_confidence,
        "sanitize": sanitize,
    }
    if max_ocr_width is not None:
        extractor_kw["max_ocr_width"] = max_ocr_width
    if onnx_threads is not None:
        extractor_kw["onnx_threads"] = onnx_threads
    else:
        try:
            from .config import get_ocr_onnx_threads

            extractor_kw["onnx_threads"] = get_ocr_onnx_threads()
        except Exception:
            extractor_kw["onnx_threads"] = 2
    extractor = TableImageExtractor(**extractor_kw)
    artifacts: List[str] = []
    best_crop_path = ""
    run_dir = Path(debug_dir) if debug_dir else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    # base_date 入参已废弃（兼容保留）；业务日只走画面 OCR，不猜测/不对侧回填
    del base_date
    if date_cfg is None:
        from .config import get_ocr_date_cfg

        date_cfg = get_ocr_date_cfg(layout=layout_mode)
    resolved_date, date_source, date_ocr_meta = resolve_business_date(
        image_path=image_path,
        workdir=workdir,
        fallback_date=None,
        date_cfg=date_cfg,
        min_confidence=min_confidence,
        debug_dir=run_dir,
    )
    date_ms = int((time.perf_counter() - t0) * 1000)
    date_crop_path = str(date_ocr_meta.get("image_path") or "")
    if date_crop_path:
        artifacts.append(date_crop_path)
    for p in date_ocr_meta.get("candidate_image_paths") or []:
        if p and p not in artifacts:
            artifacts.append(str(p))
    best_date = str(date_ocr_meta.get("best_image_path") or "")
    if best_date and best_date not in artifacts:
        artifacts.append(best_date)
    date_ocr_out = {k: v for k, v in date_ocr_meta.items() if k != "date"}
    if date_ocr_meta.get("date") is not None and not date_ocr_out.get("date_str"):
        date_ocr_out["date_str"] = date_ocr_meta["date"].strftime("%Y-%m-%d")

    if date_source != "ocr" or resolved_date is None:
        logger.warning(
            "日期 OCR 失败，跳过表格 OCR image=%s source=%s reason=%s",
            image_path,
            date_source,
            date_ocr_out.get("reason"),
        )
        return _empty_date_fail_result(
            image_path=image_path,
            run_dir=run_dir,
            artifacts=artifacts,
            date_source=date_source,
            date_ocr_out=date_ocr_out,
            timing={"date_ms": date_ms, "table_ms": 0, "ocr_ms": date_ms},
        )

    base_date_str = resolved_date.strftime("%Y-%m-%d")

    crop_candidates: List[Tuple[str, Optional[Tuple[float, float, float, float]]]] = list(
        _default_crop_candidates(layout_mode, table_crop=table_crop)
    )

    seen = set()
    unique: List[Tuple[str, Optional[Tuple[float, float, float, float]]]] = []
    for label, c in crop_candidates:
        key = c if c is not None else ("auto",)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, c))

    def _score(
        result: Any, cand: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[int, int, int, float]:
        # (表头质量, 实心行, 密度, -面积)
        if not result.headers or not result.rows:
            return (-1, -1, -1, 0.0)
        auto_ratio = (
            getattr(extractor, "last_auto_crop_ratio", None) if cand is None else None
        )
        hq = _header_quality(list(result.headers))
        # 前导空列：错位表头虚高列数时扣分（每空列 -2，最多扣掉大半）
        lead_pen = _leading_empty_penalty([list(r) for r in result.rows])
        if hq >= 0 and lead_pen > 0:
            hq = hq - min(hq, lead_pen * 2)
        # 整型/小数混行：扣表头分，避免 auto 多列但行搅在一起压过干净 fixed
        mix_pen = _row_type_mix_penalty([list(r) for r in result.rows])
        if mix_pen > 0:
            hq = hq - mix_pen
        ncols = max(len(result.headers), 1)
        filled = 0
        solid = 0
        # 窄表：按「填满约 2/3 列」计实心行（旧 max(6,…) 在 ≤5 列时永远为 0）
        solid_need = max(3, (ncols * 2 + 2) // 3)
        for row in result.rows:
            nfill = 0
            for cell in row:
                if cell and str(cell).strip():
                    nfill += 1
                    filled += 1
            if nfill >= solid_need:
                solid += 1
        density = int(100.0 * filled / max(len(result.rows) * ncols, 1))
        # 认 OCR 原文小数：带 ≥2 位小数的格子略加密度分
        dot_n = _literal_decimal_count(result.rows)
        dens_adj = min(160, density + dot_n)
        # 粘连怪物格扣密度：避免「脏但填满」压过稍空的干净候选
        dirty = 0
        for row in result.rows:
            for cell in row:
                if not cell:
                    continue
                s = str(cell).strip()
                digits = re.sub(r"\D", "", s)
                if len(digits) >= 10:
                    dirty += 1
                elif "." in s and len(re.sub(r"\D", "", s.split(".", 1)[0])) >= 7:
                    dirty += 1
        dens_adj = max(0, dens_adj - dirty * 25)
        # 混行再扣密度，与表头混行罚分双保险
        if mix_pen > 0:
            dens_adj = max(0, dens_adj - mix_pen * 8)
        return (
            hq,
            solid,
            dens_adj,
            -_crop_area_ratio(cand, auto_ratio=auto_ratio),
        )

    cand_records: List[Dict[str, Any]] = []
    best = None
    # 初值须低于「表头带罚分但仍可用」的候选（如误含 00:00 时 hq≈-42）。
    # 若用 (-1,-1,-1,0)，所有 hq<0 的可用候选都选不中 → 0 点「未达下限」。
    best_score: Tuple[int, int, int, float] = (-10**9, -1, -1, 0.0)
    best_idx = -1
    best_cand_path = ""
    stop_early = False
    allow_early_stop = bool(early_stop)

    def _good_enough(
        sc: Tuple[int, int, int, float],
        dens_floor: int,
        rows: Optional[List[Any]] = None,
    ) -> bool:
        # 仅当表头可用且每一数据行每一格都填满（100%）才跳过后续候选（fixed 回退）
        del dens_floor
        if sc[0] < 3:
            return False
        row_list = list(rows or [])
        if len(row_list) < 3:
            return False
        for row in row_list:
            if not row:
                return False
            for c in row:
                if c is None or not str(c).strip():
                    return False
        # 电价行（末行）仍有漏点 4 位整数或疑似串位长整数部 → 不 early_stop
        price = row_list[-1]
        undotted4 = 0
        long_int = 0
        for c in price:
            s = str(c).strip().replace("−", "-").replace("–", "-")
            if re.fullmatch(r"-?\d{4}", s):
                undotted4 += 1
            elif "." in s:
                try:
                    fv = abs(float(s))
                except ValueError:
                    continue
                ip = s.lstrip("-").split(".", 1)[0]
                if len(re.sub(r"\D", "", ip)) >= 3 and fv >= 80:
                    long_int += 1
        if undotted4 >= 1 or long_int >= 2:
            return False
        return True

    t_table = time.perf_counter()
    for i, (label, c) in enumerate(unique):
        skip_reason = ""
        if stop_early:
            skip_reason = "early_stop"
        if skip_reason:
            cand_records.append(
                {
                    "index": i,
                    "label": label,
                    "name": _cand_display_name(label),
                    "crop": list(c) if c is not None else None,
                    "status": "skipped",
                    "reason": "",
                    "score": None,
                    "headers": [],
                    "rows": [],
                    "point_count": 0,
                    "image_path": "",
                    "skip_reason": skip_reason,
                    "ocr_ms": 0,
                }
            )
            continue

        logger.info(
            "OCR 识别 #%s layout=%s scale=%s min_confidence=%s label=%s crop=%s image=%s",
            i,
            layout_mode,
            scale,
            min_confidence,
            label,
            c,
            image_path,
        )
        cand_saved = ""
        cand_name = _cand_display_name(label)
        if run_dir is not None:
            try:
                bgr = _materialize_crop_bgr(
                    extractor, image_path, c, deskew=True
                )
                cand_path = run_dir / f"{cand_name}.jpg"
                cand_saved = _save_bgr(cand_path, bgr) or ""
                if cand_saved:
                    artifacts.append(cand_saved)
            except Exception:
                logger.exception("保存候选裁剪失败 #%s label=%s crop=%s", i, label, c)

        t_cand = time.perf_counter()
        result = extractor.extract(str(image_path), crop_ratio=c)
        cand_ms = int((time.perf_counter() - t_cand) * 1000)
        sc = _score(result, c)
        # sc: (hq, solid, density, -area)
        invalid = sc[0] < 0 and sc[1] <= 0
        headers = list(result.headers or [])
        rows = [list(r) for r in (result.rows or [])]
        filled_n = sum(
            1 for row in rows for cell in row if cell and str(cell).strip()
        )

        rec: Dict[str, Any] = {
            "index": i,
            "label": label,
            "name": _cand_display_name(label),
            "crop": list(c) if c is not None else None,
            "status": "rejected",  # 稍后统一标 winner
            "reason": "",
            "score": _score_tuple_public(sc) if not invalid else list(sc[:3]) + [None],
            "headers": headers,
            "rows": rows,
            "point_count": filled_n,
            "image_path": cand_saved,
            "ocr_ms": cand_ms,
        }
        if invalid:
            logger.warning("OCR 裁剪无有效表 #%s crop=%s ocr_ms=%s", i, c, cand_ms)
            rec["invalid"] = True
            cand_records.append(rec)
            continue

        logger.info(
            "OCR 候选 #%s score=%s headers=%s ocr_ms=%s head=%s…%s",
            i,
            sc,
            len(headers),
            cand_ms,
            (headers[:2] if headers else None),
            (headers[-2:] if headers else None),
        )
        cand_records.append(rec)
        if sc > best_score:
            best_score = sc
            best = result
            best_idx = i
            best_cand_path = cand_saved
        if allow_early_stop and _good_enough(
            sc, dens_need, rows=list(getattr(result, "rows", None) or [])
        ):
            stop_early = True
            logger.info(
                "OCR early_stop #%s label=%s hq=%s 数据行全满，跳过后续候选",
                i,
                label,
                sc[0],
            )

    table_ms = int((time.perf_counter() - t_table) * 1000)
    timing_out: Dict[str, Any] = {
        "date_ms": date_ms,
        "table_ms": table_ms,
        "ocr_ms": date_ms + table_ms,
    }

    winner_label = ""
    if best_idx >= 0 and best_idx < len(cand_records):
        winner_label = _cand_display_name(
            str(cand_records[best_idx].get("label") or "")
        )

    for rec in cand_records:
        idx = int(rec["index"])
        if rec.get("status") == "skipped":
            rec["reason"] = _explain_loser(
                (-1, -1, -1, 0.0),
                best_score,
                winner_label or "—",
                skipped=True,
            )
            continue
        if idx == best_idx:
            rec["status"] = "winner"
            rec.pop("invalid", None)
            rec["reason"] = _explain_winner(best_score, winner_label)
            continue
        if rec.get("invalid"):
            rec["status"] = "invalid"
            rec["reason"] = _explain_loser(
                (-1, -1, -1, 0.0),
                best_score,
                winner_label or "—",
                invalid=True,
            )
            continue
        rec["status"] = "loser"
        pub = rec.get("score") or []
        try:
            mine_internal = (
                int(pub[0]),
                int(pub[1]),
                int(pub[2]),
                -float(pub[3]) if pub[3] is not None else 0.0,
            )
        except (TypeError, ValueError, IndexError):
            mine_internal = (-1, -1, -1, 0.0)
        rec["reason"] = _explain_loser(
            mine_internal, best_score, winner_label or "—", invalid=False
        )

    # 胜出裁剪定结构；空格用其他候选同表头回填（例：auto 认出漏格，fixed 漏了）
    # 只改最终 rows，不覆盖胜出者候选原始 rows（弹窗「本候选」仍看原识别）
    gap_filled_n = 0
    gap_fill_map: List[List[str]] = []
    if best is not None and best_idx >= 0:
        clean_h, clean_r = _sanitize_headers_for_write(
            list(best.headers), [list(r) for r in best.rows]
        )
        if clean_h != list(best.headers) or clean_r != [list(r) for r in best.rows]:
            best.headers = clean_h
            best.rows = clean_r
            logger.info(
                "OCR 落库表头清洗后 cols=%s headers=%s…%s",
                len(clean_h),
                clean_h[:2] if clean_h else None,
                clean_h[-2:] if clean_h else None,
            )
        filled_rows, gap_filled_n, gap_fill_map = _gap_fill_from_peer_candidates(
            list(best.headers),
            [list(r) for r in best.rows],
            cand_records,
        )
        if gap_filled_n > 0:
            best.rows = filled_rows
            best.ocr_filled = sum(
                1
                for row in filled_rows
                for cell in row
                if cell and str(cell).strip()
            )
            if best_idx < len(cand_records):
                wrec = cand_records[best_idx]
                # 保留胜出者原始识别表；补录只体现在最终结果 + gap_fill_map
                base_reason = str(wrec.get("reason") or "")
                wrec["reason"] = (
                    f"{base_reason}；同伴候选回填空格 {gap_filled_n} 处"
                    if base_reason
                    else f"同伴候选回填空格 {gap_filled_n} 处"
                )
                wrec["gap_filled"] = gap_filled_n
                wrec["gap_fill_map"] = [list(r) for r in gap_fill_map]
            logger.info(
                "OCR 同伴回填 gap_filled=%s winner=%s",
                gap_filled_n,
                winner_label or best_idx,
            )

    if best is None:
        if run_dir is not None and cand_records:
            try:
                (run_dir / "candidates.json").write_text(
                    json.dumps(cand_records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (run_dir / "date_ocr.json").write_text(
                    json.dumps(
                        {
                            "base_date": base_date_str,
                            "date_source": date_source,
                            "date_ocr": date_ocr_out,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                logger.exception("写 candidates.json 失败（无胜出）")
        return OcrResult(
            image_path=str(image_path),
            headers=[],
            rows=[],
            points=[],
            row_tags=[],
            data_row_index=-1,
            artifacts_dir=str(run_dir) if run_dir else "",
            best_crop_path="",
            candidate_paths=artifacts,
            candidates=cand_records,
            ocr_filled=0,
            base_date=base_date_str,
            date_source=date_source,
            date_ocr=date_ocr_out,
            timing=timing_out,
            ok=False,
            reason="table_ocr_failed",
            message="日期已识别，但表格 OCR 未得到有效表头/数据行",
        )

    if run_dir is not None and best_cand_path:
        try:
            best_out = run_dir / "best_table.jpg"
            shutil.copy2(best_cand_path, best_out)
            best_crop_path = str(best_out)
            artifacts.append(best_crop_path)
            (run_dir / "best_meta.txt").write_text(
                f"layout={layout_mode}\n"
                f"winner={winner_label or f'cand_{best_idx:02d}'}\n"
                f"score={best_score}\n"
                f"source={best_cand_path}\n"
                f"base_date={base_date_str}\n"
                f"date_source={date_source}\n",
                encoding="utf-8",
            )
            (run_dir / "candidates.json").write_text(
                json.dumps(cand_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (run_dir / "date_ocr.json").write_text(
                json.dumps(
                    {
                        "base_date": base_date_str,
                        "date_source": date_source,
                        "date_ocr": date_ocr_out,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("复制 best_table / 写 candidates.json 失败")
            best_crop_path = best_cand_path

    # 履历 base_date 仍记 OCR 识别日；写库时间戳按 ocr.pi_date_back_days 提前后再与表头时刻拼接
    pi_base_date = _pi_write_base_date(resolved_date)
    points, row_tags = _points_from_full_table(
        list(best.headers),
        [list(r) for r in best.rows],
        tag_prefix,
        pi_base_date,
    )
    ocr_filled = int(getattr(best, "ocr_filled", 0) or 0)
    if ocr_filled <= 0:
        ocr_filled = sum(
            1
            for row in (best.rows or [])
            for cell in row
            if cell and str(cell).strip()
        )
    return OcrResult(
        image_path=str(image_path),
        headers=list(best.headers),
        rows=[list(r) for r in best.rows],
        points=points,
        row_tags=row_tags,
        data_row_index=len(best.rows) - 1 if best.rows else -1,
        artifacts_dir=str(run_dir) if run_dir else "",
        best_crop_path=best_crop_path,
        candidate_paths=artifacts,
        candidates=cand_records,
        ocr_filled=ocr_filled,
        gap_fill_map=[list(r) for r in gap_fill_map],
        gap_filled=int(gap_filled_n),
        base_date=base_date_str,
        date_source=date_source,
        date_ocr=date_ocr_out,
        timing=timing_out,
        ok=True,
        reason="",
        message="",
    )


def ocr_result_to_dict(r: OcrResult) -> Dict[str, Any]:
    return {
        "image_path": r.image_path,
        "headers": r.headers,
        "rows": r.rows,
        "row_tags": r.row_tags,
        "data_row_index": r.data_row_index,
        "artifacts_dir": r.artifacts_dir,
        "best_crop_path": r.best_crop_path,
        "candidate_paths": list(r.candidate_paths or []),
        "candidates": list(r.candidates or []),
        "ocr_filled": int(r.ocr_filled or 0),
        "gap_fill_map": [list(row) for row in (r.gap_fill_map or [])],
        "gap_filled": int(r.gap_filled or 0),
        "base_date": r.base_date or None,
        "date_source": r.date_source or None,
        "date_ocr": dict(r.date_ocr or {}),
        "timing": dict(r.timing or {}),
        "ok": bool(r.ok),
        "reason": r.reason or None,
        "message": r.message or None,
        "points": [
            {
                "time_label": p.time_label,
                "value": p.value,
                "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "tag": p.tag,
                "row_index": p.row_index,
            }
            for p in r.points
        ],
    }
