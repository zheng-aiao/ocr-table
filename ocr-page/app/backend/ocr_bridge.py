# -*- coding: utf-8 -*-
"""调用 app/ocr_engine 做表格 OCR，产出整表多行时段-数值列表。"""

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
    date_source: str = ""  # ocr | system | failed
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


def _cand_short_name(label: str) -> str:
    """日志短名：fixed / auto。"""
    lab = (label or "").strip().lower()
    if "fixed" in lab:
        return "fixed"
    if "auto" in lab:
        return "auto"
    if "full" in lab:
        return "full"
    name = _cand_display_name(label)
    return name.replace("cand_table_", "").replace("cand_", "") or "unknown"


# 双摄单侧图默认固定表区（可被 config table_crop 覆盖）
_DEFAULT_TABLE_CROP = (0.02, 0.52, 0.985, 0.90)


def _default_crop_candidates(
    table_crop: Optional[Tuple[float, float, float, float]] = None,
) -> List[Tuple[str, Optional[Tuple[float, float, float, float]]]]:
    """
    双摄表格 OCR 候选：先 fixed，auto 是否再跑由 auto_enable 决定。

    fixed 用该侧 table_crop（未配则用默认框）。
    auto_enable 开启且 fixed 表头不够好时，再跑蓝底自动定位。

    裁完后都进同一套纠偏再 OCR（斜拍时 fixed 同样必须纠偏）。
    """
    fixed = table_crop or _DEFAULT_TABLE_CROP
    return [
        ("table_fixed", fixed),
        ("table_auto", None),
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
    写 PI 用业务日 = OCR 识别日向前退 back_days 天（pi.pi_date_back_days）。
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
    loose_band: bool = False,
) -> Optional[Tuple[float, float, float, float]]:
    """
    在 OCR 框中选「日期」标签：优先上半、偏左、文本更短（纯「日期」优于长句）。
    坐标相对当前搜索裁剪图。

    loose_band=True：搜索带本身已是顶栏窄带，不再用 0.72/0.85 位置硬过滤
    （否则标题占上沿时，「日期」行易被误杀）。
    """
    candidates: List[Tuple[float, float, float, float, float, str]] = []
    for text, box in boxes:
        if not _is_date_label_text(text):
            continue
        x0, y0, x1, y1 = box
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        if not loose_band:
            # 大图/大半屏搜索时：过滤明显不在上半的命中
            if region_h > 1 and cy > region_h * 0.72:
                continue
            if region_w > 1 and cx > region_w * 0.85:
                continue
        elif region_w > 1 and cx > region_w * 0.95:
            # 窄带内仍丢掉贴右边缘的噪点框
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
    loose_band: bool = False,
) -> Optional[Tuple[float, float, float, float]]:
    """
    无「日期」标签时：只认筛选框格式 YYYY-MM-DD|/|。
    顶栏「yyyy年mm月dd日」一律不参与定位（不是业务筛选日）。

    loose_band=True：日期值常在窄带右侧，不再用 cx<0.75 硬过滤。
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
        if loose_band:
            # 窄带内：只丢掉贴底噪点；日期值允许靠右
            if region_h > 1 and cy > region_h * 0.95:
                continue
        else:
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
    「日期」搜索带（单屏/双屏/双摄共用）。
    只覆盖标题栏附近左上窄带（约半宽 × 约 28% 高），避免旧版左上 1/4 过大拖慢 det。
    """
    del layout  # 与单屏同一搜索带，避免双屏超大 ROI 拖慢定位
    # 略留边；比旧 1/4 更矮，但比 0.28 略高，避免标题把「日期」行挤出带外
    return [
        (0.02, 0.02, 0.55, 0.34),
    ]


def _boxes_same_row(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    tol: float = 0.75,
) -> bool:
    """两框是否同一行（中心 Y 差不超过较大框高的 tol 倍）。"""
    ay0, ay1 = float(a[1]), float(a[3])
    by0, by1 = float(b[1]), float(b[3])
    ah = max(1.0, ay1 - ay0)
    bh = max(1.0, by1 - by0)
    acy = 0.5 * (ay0 + ay1)
    bcy = 0.5 * (by0 + by1)
    return abs(acy - bcy) <= max(ah, bh) * float(tol)


# 日志/展示用中文；payload 内仍存英文码便于逻辑分支
_DATE_METHOD_CN = {
    "label_same_row": "搜索带同行日期",
    "search_band_value": "搜索带直接认日期",
    "label_right": "标签右侧再识别",
}


def _date_method_cn(method: Any) -> str:
    key = str(method or "").strip()
    return _DATE_METHOD_CN.get(key, key or "-")


def _score_filter_date_text(
    text: Any,
) -> Optional[Tuple[datetime, int, str]]:
    """
    解析筛选框格式日期（YYYY-MM-DD|/|）；顶栏中文日返回 None。
    成功返回 (date, conf, date_str)。
    """
    raw = str(text or "").strip()
    if not raw or _is_chinese_ymd_value(raw):
        return None
    if _date_format_preference(raw) < 2:
        return None
    scored = parse_ui_date_scored(raw, allow_md_only=False) or parse_ui_date_scored(
        re.sub(r"\s+", "", raw), allow_md_only=False
    )
    if scored is None:
        return None
    dt, conf = scored
    if conf < _DATE_CONF_FULL or dt.year < 2018 or dt.year > 2100:
        return None
    return dt, int(conf), dt.strftime("%Y-%m-%d")


def _pick_same_row_date_right_of_label(
    boxes: List[Tuple[str, Tuple[float, float, float, float]]],
    label_box: Tuple[float, float, float, float],
) -> Optional[Dict[str, Any]]:
    """
    标签同行、偏右的筛选框日期（搜索带 OCR 一次命中即可早停）。
    也认「日期2026-10-16」粘在同一 OCR 框内的写法。
    """
    lx0, ly0, lx1, ly1 = (float(v) for v in label_box)
    lh = max(1.0, ly1 - ly0)
    best: Optional[Tuple[float, datetime, int, str, Tuple[float, float, float, float], List[str]]] = None

    for text, box in boxes:
        x0, y0, x1, y1 = (float(v) for v in box)
        scored = _score_filter_date_text(text)
        if scored is None:
            continue
        dt, conf, ds = scored
        # 标签框自身粘连日期
        same_label_box = (
            abs(x0 - lx0) < 2
            and abs(y0 - ly0) < 2
            and abs(x1 - lx1) < 2
            and abs(y1 - ly1) < 2
        )
        if same_label_box:
            best = (0.0, dt, conf, ds, box, [text])
            break
        if not _boxes_same_row(label_box, box):
            continue
        # 同行且在标签右侧（允许少量重叠）
        if (x0 + x1) * 0.5 < lx1 - lh * 0.3:
            continue
        if x1 <= lx0:
            continue
        dist = max(0.0, x0 - lx1)
        cand = (dist, dt, conf, ds, box, [text])
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        return None
    _, dt, conf, ds, vbox, texts = best
    return {
        "date": dt,
        "date_str": ds,
        "confidence": conf,
        "value_box": list(vbox),
        "texts": texts,
    }


def _pick_search_band_early_date(
    boxes: List[Tuple[str, Tuple[float, float, float, float]]],
    *,
    region_w: float,
    region_h: float,
    label_box: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    搜索带 OCR 结果里直接取完整筛选日期（理想最短路径，免二次裁剪 OCR）。
    优先：标签同行右侧；否则：带内任意 YYYY-MM-DD|/| 值框。
    """
    if label_box is not None:
        same = _pick_same_row_date_right_of_label(boxes, label_box)
        if same is not None:
            return {
                **same,
                "method": "label_same_row",
                "label_box": list(label_box),
            }

    value_box = _pick_date_value_box(
        boxes,
        region_w=float(region_w),
        region_h=float(region_h),
        loose_band=True,
    )
    if value_box is not None:
        for text, box in boxes:
            if (
                abs(box[0] - value_box[0]) < 1
                and abs(box[1] - value_box[1]) < 1
                and abs(box[2] - value_box[2]) < 1
                and abs(box[3] - value_box[3]) < 1
            ):
                scored = _score_filter_date_text(text)
                if scored is None:
                    break
                dt, conf, ds = scored
                return {
                    "date": dt,
                    "date_str": ds,
                    "confidence": conf,
                    "value_box": list(value_box),
                    "texts": [text],
                    "method": "search_band_value",
                    "label_box": list(label_box) if label_box else list(value_box),
                }

    # 兜底：扫全部文本（框选失败时仍可用）
    best: Optional[Tuple[float, datetime, int, str, Tuple[float, float, float, float], str]] = None
    for text, box in boxes:
        scored = _score_filter_date_text(text)
        if scored is None:
            continue
        dt, conf, ds = scored
        cy = 0.5 * (float(box[1]) + float(box[3]))
        cx = 0.5 * (float(box[0]) + float(box[2]))
        score = (cy / max(region_h, 1.0)) * 2.0 + (cx / max(region_w, 1.0))
        cand = (score, dt, conf, ds, box, text)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        return None
    _, dt, conf, ds, vbox, text = best
    return {
        "date": dt,
        "date_str": ds,
        "confidence": conf,
        "value_box": list(vbox),
        "texts": [text],
        "method": "search_band_value",
        "label_box": list(label_box) if label_box else list(vbox),
    }


def _filter_ratio_right_of_label_box(
    label_box: Tuple[float, float, float, float],
    *,
    img_w: int,
    img_h: int,
    search: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    仅裁「日期」标签右侧输入框一带（不含大块左侧标题），供后续 OCR。
    """
    sx0, sy0, sx1, sy1 = (float(v) for v in search)
    lx0, ly0, lx1, ly1 = label_box
    ax0 = sx0 * img_w + lx0
    ay0 = sy0 * img_h + ly0
    ax1 = sx0 * img_w + lx1
    ay1 = sy0 * img_h + ly1
    lw = max(8.0, ax1 - ax0)
    lh = max(8.0, ay1 - ay0)
    max_right = max(lw * 8.0, min(img_w * 0.26, lw * 12.0))
    rx0 = max(0.0, ax1 - lw * 0.15)
    ry0 = max(0.0, ay0 - lh * 0.55)
    rx1 = min(float(img_w), ax1 + max_right, rx0 + img_w * 0.38)
    ry1 = min(float(img_h), ay1 + lh * 1.15, ry0 + max(lh * 2.8, img_h * 0.07))
    min_h = max(64.0, lh * 2.0, img_h * 0.035)
    max_h = max(min_h, lh * 3.2, img_h * 0.085)
    cur_h = ry1 - ry0
    if cur_h < min_h:
        need = min_h - cur_h
        ry0 = max(0.0, ry0 - need * 0.25)
        ry1 = min(float(img_h), ry0 + max_h)
    elif cur_h > max_h:
        ry1 = ry0 + max_h
    return _clamp_crop(
        rx0 / img_w, ry0 / img_h, rx1 / img_w, ry1 / img_h
    )


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
    log_prefix: str = "",
) -> Optional[Dict[str, Any]]:
    """
    自动定位左上角「日期」筛选框。

    理想最短路径：搜索带 OCR 一次就读出 YYYY-MM-DD|/| → early_date 直接返回。
    仅当带内看不到完整日期时，才裁标签右侧再交给后续 OCR。
    """
    pfx = str(log_prefix or "")
    try:
        full = extractor.read_image_file(str(image_path))
    except Exception:
        logger.exception("%s读取原图失败，无法自动定位日期: %s", pfx, image_path)
        return None
    h, w = full.shape[:2]
    if h < 20 or w < 20:
        return None

    run_dir = Path(debug_dir) if debug_dir else None
    last_texts: List[str] = []

    def _save_locate(
        *,
        crop: Tuple[float, float, float, float],
        band: Any,
        si: int,
    ) -> str:
        crop_bgr = _materialize_crop_bgr(extractor, image_path, crop)
        saved = ""
        if run_dir is not None and crop_bgr is not None:
            saved = _save_bgr(run_dir / "date_locate.jpg", crop_bgr) or ""
            _save_bgr(run_dir / f"date_search_{si:02d}.jpg", band)
        return saved

    def _early_hit(
        *,
        early: Dict[str, Any],
        search: Tuple[float, float, float, float],
        band: Any,
        si: int,
        mapped: List[Tuple[str, Tuple[float, float, float, float]]],
        label: Optional[Tuple[float, float, float, float]],
    ) -> Dict[str, Any]:
        method = str(early.get("method") or "search_band_value")
        label_box = label
        if label_box is None and early.get("label_box"):
            try:
                label_box = tuple(float(x) for x in early["label_box"])  # type: ignore[misc]
            except Exception:
                label_box = None
        if method == "label_same_row" and label_box is not None:
            crop = _filter_ratio_from_label_box(
                label_box, img_w=w, img_h=h, search=search
            )
        else:
            vbox = early.get("value_box") or early.get("label_box")
            crop = _filter_ratio_from_value_box(
                tuple(float(x) for x in vbox),  # type: ignore[arg-type]
                img_w=w,
                img_h=h,
                search=search,
            )
        saved = _save_locate(crop=crop, band=band, si=si)
        texts = list(early.get("texts") or [])
        for t, _ in mapped:
            if t not in texts:
                texts.append(t)
        hit = {
            "crop": crop,
            "label_box": list(label_box)
            if label_box is not None
            else list(early.get("value_box") or []),
            "search": list(search),
            "image_path": saved,
            "texts": texts[:16],
            "method": method,
            "early_date": early["date"],
            "early_date_str": early["date_str"],
            "confidence": early["confidence"],
        }
        logger.info(
            "%s日期定位 搜索带直接命中 方式=%s date=%s",
            pfx,
            _date_method_cn(method),
            early["date_str"],
        )
        return hit

    for si, search in enumerate(_date_label_search_regions(layout)):
        band = _materialize_crop_bgr(extractor, image_path, search)
        if band is None or getattr(band, "size", 0) == 0:
            continue
        bh, bw = band.shape[:2]
        mapped: List[Tuple[str, Tuple[float, float, float, float]]] = []
        label: Optional[Tuple[float, float, float, float]] = None
        for use in _locate_band_variants(band):
            result = _collect_ocr_result(ocr, use)
            boxes = _iter_ocr_boxes(result)
            scale_x = bw / max(1, use.shape[1])
            scale_y = bh / max(1, use.shape[0])
            mapped = []
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
                mapped,
                region_w=float(bw),
                region_h=float(bh),
                loose_band=True,
            )
            # 理想路径：本轮搜索带 OCR 已读出完整筛选日期 → 立刻返回
            early = _pick_search_band_early_date(
                mapped,
                region_w=float(bw),
                region_h=float(bh),
                label_box=label,
            )
            if early is not None:
                return _early_hit(
                    early=early,
                    search=search,
                    band=band,
                    si=si,
                    mapped=mapped,
                    label=label,
                )
            # 有标签但带内无完整日期：不再试更多变体，改裁右侧
            if label is not None:
                break
        else:
            # 变体都无标签也无日期
            continue

        assert label is not None
        # 有标签但搜索带未读出完整日期 → 裁右侧输入区，交给后续 OCR
        crop = _filter_ratio_right_of_label_box(
            label, img_w=w, img_h=h, search=search
        )
        saved = _save_locate(crop=crop, band=band, si=si)
        logger.debug(
            "%s日期定位 方式=%s crop=%s",
            pfx,
            _date_method_cn("label_right"),
            crop,
        )
        return {
            "crop": crop,
            "label_box": list(label),
            "search": list(search),
            "image_path": saved,
            "texts": last_texts[:16],
            "method": "label_right",
        }

    logger.warning(
        "%s日期定位未在搜索带读到日期 texts=%s",
        pfx,
        last_texts[:8],
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


def _locate_band_variants(band: Any, *, loc_scale: float = 2.0) -> List[Any]:
    """日期标签搜索带：先放大，找不到「日期」再上少量去纹变体。"""
    import cv2  # type: ignore

    out: List[Any] = []
    try:
        use = cv2.resize(
            band, None, fx=loc_scale, fy=loc_scale, interpolation=cv2.INTER_CUBIC
        )
        out.append(use)
        gray = cv2.cvtColor(use, cv2.COLOR_BGR2GRAY)
        # 定位优先轻量：双边/锐化各一；NL-means/FFT 留给日期数字增强轮
        for g in _demoire_preprocess_gray(gray)[:2]:
            if g is None or getattr(g, "size", 0) == 0:
                continue
            if len(getattr(g, "shape", ())) == 2:
                out.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            else:
                out.append(g)
    except Exception:
        logger.exception("日期定位预处理失败")
        out = [band]
    return out[:3]


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
    layout: str = "double",
    auto_locate: bool = True,
    log_prefix: str = "",
) -> Dict[str, Any]:
    """
    双摄日期识别（简化三步）：
    1) date_crop 搜索带 OCR，直接认出 YYYY-MM-DD 则成功（落盘 date_search.jpg）
    2) 否则找「日期」标签，裁右侧输入区再 OCR（落盘 date_locate.jpg）
    3) 无日期也无标签则失败
    成功时胜选图复制为 date_best.jpg
    """
    del auto_locate
    del layout
    del scale
    del min_confidence
    pfx = str(log_prefix or "")
    _ensure_ocr_path(workdir)
    from extract_table import TableImageExtractor  # type: ignore

    search: Optional[Tuple[float, float, float, float]] = None
    if crop is not None:
        search = (float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3]))
    elif crops:
        c0 = crops[0]
        search = (float(c0[0]), float(c0[1]), float(c0[2]), float(c0[3]))
    if search is None:
        search = (0.02, 0.02, 0.55, 0.34)

    payload: Dict[str, Any] = {
        "ok": False,
        "date": None,
        "date_str": None,
        "raw_texts": [],
        "crop": list(search),
        "image_path": "",
        "best_image_path": "",
        "candidate_image_paths": [],
        "reason": "",
        "attempts": [],
        "locate": None,
    }
    candidate_image_paths: List[str] = []
    run_dir: Optional[Path] = None

    def _copy_best(src: str) -> str:
        if run_dir is None or not src:
            return ""
        try:
            sp = Path(src)
            if not sp.is_file():
                return ""
            best = run_dir / "date_best.jpg"
            best.write_bytes(sp.read_bytes())
            return str(best)
        except Exception:
            logger.exception("%s保存 date_best 失败", pfx)
            return ""

    def _accept(
        *,
        dt: datetime,
        conf: int,
        texts: List[str],
        reason: str,
        method: str,
        image_path: str,
        crop_ratio: Tuple[float, float, float, float],
        label_box: Any = None,
    ) -> Dict[str, Any]:
        best = _copy_best(image_path)
        payload["ok"] = True
        payload["date"] = dt
        payload["date_str"] = dt.strftime("%Y-%m-%d")
        payload["crop"] = list(crop_ratio)
        payload["raw_texts"] = list(texts)[:12]
        payload["confidence"] = conf
        payload["reason"] = reason
        payload["image_path"] = image_path
        payload["best_image_path"] = best
        payload["candidate_image_paths"] = list(candidate_image_paths)
        payload["locate"] = {
            "crop": list(crop_ratio),
            "search": list(search),
            "label_box": list(label_box) if label_box is not None else [],
            "image_path": image_path,
            "method": method,
            "early_date_str": payload["date_str"],
        }
        payload["attempts"].append(
            {
                "crop": list(crop_ratio),
                "texts": list(texts)[:12],
                "ok": True,
                "date_str": payload["date_str"],
                "confidence": conf,
                "image_path": image_path,
                "stage": method,
            }
        )
        logger.info(
            "%s日期定位 成功 方式=%s date=%s",
            pfx,
            _date_method_cn(method),
            payload["date_str"],
        )
        return payload

    try:
        from .config import get_ocr_onnx_threads

        extractor = TableImageExtractor(
            scale=1.0,
            min_confidence=0.3,
            onnx_threads=get_ocr_onnx_threads(),
        )
        ocr = extractor._get_ocr()
        run_dir = Path(debug_dir) if debug_dir else None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)

        full = extractor.read_image_file(str(image_path))
        h, w = full.shape[:2]
        if h < 20 or w < 20:
            payload["reason"] = "date_not_found"
            return payload

        band = _materialize_crop_bgr(extractor, image_path, search)
        if band is None or getattr(band, "size", 0) == 0:
            payload["reason"] = "date_not_found"
            return payload

        search_saved = ""
        if run_dir is not None:
            search_saved = _save_bgr(run_dir / "date_search.jpg", band) or ""
            if search_saved:
                candidate_image_paths.append(search_saved)
                payload["image_path"] = search_saved

        bh, bw = band.shape[:2]
        all_texts: List[str] = []
        label: Optional[Tuple[float, float, float, float]] = None
        mapped: List[Tuple[str, Tuple[float, float, float, float]]] = []

        # 1) 搜索带 OCR：直接认 YYYY-MM-DD
        for use in _locate_band_variants(band):
            result = _collect_ocr_result(ocr, use)
            boxes = _iter_ocr_boxes(result)
            scale_x = bw / max(1, use.shape[1])
            scale_y = bh / max(1, use.shape[0])
            mapped = []
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
                all_texts.append(text)

            label = _pick_date_label_box(
                mapped,
                region_w=float(bw),
                region_h=float(bh),
                loose_band=True,
            )
            early = _pick_search_band_early_date(
                mapped,
                region_w=float(bw),
                region_h=float(bh),
                label_box=label,
            )
            if early is not None:
                method = str(early.get("method") or "search_band_value")
                payload["raw_texts"] = all_texts[:24]
                return _accept(
                    dt=early["date"],
                    conf=int(early["confidence"]),
                    texts=list(early.get("texts") or []) + all_texts,
                    reason=f"full_ymd_{method}",
                    method=method,
                    image_path=search_saved or "",
                    crop_ratio=search,
                    label_box=label or early.get("value_box"),
                )
            if label is not None:
                break

        payload["raw_texts"] = all_texts[:24]

        # 2) 有「日期」标签但带内无完整日期 → 裁右侧再 OCR
        if label is not None:
            right_crop = _filter_ratio_right_of_label_box(
                label, img_w=w, img_h=h, search=search
            )
            right_bgr = _materialize_crop_bgr(extractor, image_path, right_crop)
            locate_saved = ""
            if run_dir is not None and right_bgr is not None:
                locate_saved = _save_bgr(run_dir / "date_locate.jpg", right_bgr) or ""
                if locate_saved:
                    candidate_image_paths.append(locate_saved)
            payload["locate"] = {
                "crop": list(right_crop),
                "search": list(search),
                "label_box": list(label),
                "image_path": locate_saved,
                "method": "label_right",
            }
            if right_bgr is not None and getattr(right_bgr, "size", 0) > 0:
                for variant in _date_ocr_variants(right_bgr)[:3]:
                    texts = _collect_ocr_texts(ocr, variant)
                    all_texts.extend(texts)
                    payload["raw_texts"] = all_texts[:24]
                    scored = None
                    for t in texts:
                        scored = _score_filter_date_text(t)
                        if scored is not None:
                            break
                    if scored is None:
                        scored = _score_filter_date_text(" ".join(texts))
                    if scored is None:
                        continue
                    dt, conf, _ds = scored
                    return _accept(
                        dt=dt,
                        conf=conf,
                        texts=texts,
                        reason="full_ymd_label_right",
                        method="label_right",
                        image_path=locate_saved or search_saved or "",
                        crop_ratio=right_crop,
                        label_box=label,
                    )

        # 3) 失败
        payload["reason"] = "date_not_found"
        payload["candidate_image_paths"] = list(candidate_image_paths)
        if label is None:
            logger.warning(
                "%s日期识别失败：搜索带无完整日期且无「日期」标签 texts=%s",
                pfx,
                all_texts[:8],
            )
        else:
            logger.warning(
                "%s日期识别失败：标签右侧未认出完整日期 texts=%s",
                pfx,
                all_texts[:8],
            )
        return payload
    except Exception as e:
        logger.exception("%s日期 OCR 异常", pfx)
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
    log_prefix: str = "",
) -> Tuple[Optional[datetime], str, Dict[str, Any]]:
    """
    业务日优先画面左上角日期 OCR。
    ocr.date_ocr_enabled=false 时不跑识别，直接返回本机系统日（source=system）。
    fallback_date 保留参数兼容，忽略。
    返回 (date_or_None, source, date_ocr_payload)；source 为 ocr / failed / system。
    """
    del fallback_date
    from .config import get_ocr_date_cfg

    pfx = str(log_prefix or "")
    cfg = date_cfg if date_cfg is not None else get_ocr_date_cfg()
    if not cfg.get("enabled", True):
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        date_str = today.strftime("%Y-%m-%d")
        logger.info("%s日期 OCR 已关闭(date_ocr_enabled=false)，用系统日 %s", pfx, date_str)
        return today, "system", {
            "ok": True,
            "reason": "date_ocr_disabled",
            "message": f"日期 OCR 已关闭，使用系统日 {date_str}",
            "date": today,
            "date_str": date_str,
            "fallback": "system",
            "skipped": True,
        }

    primary = cfg.get("crop")
    if isinstance(primary, list):
        primary = tuple(float(x) for x in primary)
    if primary is None:
        primary = (0.02, 0.02, 0.55, 0.34)

    meta = extract_ui_date_from_image(
        image_path,
        workdir,
        crop=primary,  # type: ignore[arg-type]
        crops=None,
        scale=float(cfg.get("scale", 1.0)),
        min_confidence=min_confidence,
        debug_dir=debug_dir,
        layout="double",
        auto_locate=True,
        log_prefix=pfx,
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


def _winner_looks_all_uncleared(rows: Sequence[Sequence[Any]]) -> bool:
    """
    胜出表像整表未出清（屏上全 -）：无真实数值。

    此时禁止同伴回填，否则弱候选的 133/22.30 等噪声会灌进空表。
    """
    if not rows:
        return False
    for row in rows:
        for cell in row:
            s = str(cell).strip() if cell is not None else ""
            if not s or s in ("0", "0.0", "0.00", "0.000", "-0", "-0.0", "-"):
                continue
            return False
    return True


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
    # 整表未出清：保持全空，不让同伴噪声填进来
    if _winner_looks_all_uncleared(rows):
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
    skip_reason: str = "",
) -> str:
    if skipped:
        if skip_reason == "auto_disabled":
            return "未参与比较：auto_enable 关闭，未跑自动裁剪"
        return f"未参与比较：fixed 已满足要求，跳过自动裁剪（胜出者 {winner_label}）"
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
    fix_decimal: Optional[Dict[str, Any]] = None,
    auto_enable: bool = True,
    skip_full_when_ok: bool = False,
    max_ocr_width: Optional[int] = None,
    date_cfg: Optional[Dict[str, Any]] = None,
    table_crop: Optional[Tuple[float, float, float, float]] = None,
    onnx_threads: Optional[int] = None,
    log_prefix: str = "",
) -> OcrResult:
    """
    按 ocr_engine 逻辑做整表识别（第 1 行时刻表头，其余均为数据行）。
    裁剪候选顺序：先 table_fixed，再按需 table_auto。
    两路均：裁剪 → 纠偏 → OCR；按表头质量 / 实心行 / 填格密度择优，同分优先更紧裁剪。
    auto_enable=True：fixed 表头可用且列数≥4 则跳过 auto，否则再跑 auto。
    auto_enable=False：只跑 fixed，fixed 不够好也不再跑 auto。
    skip_full_when_ok 已废弃（不再跑整图 full），保留参数兼容旧调用。

    业务日优先左上角日期 OCR（完整年月日）。
    ocr.date_ocr_enabled=false：跳过日期识别，直接用本机系统日继续表格 OCR。
    ocr.date_ocr_enabled=true 且识别失败：若 ocr.date_fallback_system=true，
    用本机系统日继续表格 OCR；否则整次失败并跳过表格 OCR。
    """
    del data_row_index
    del skip_full_when_ok
    pfx = str(log_prefix or "")
    # crop 旧参：单屏曾把 capture.single.crop 当额外候选；现统一用 table_crop
    if table_crop is None and crop is not None:
        table_crop = crop
    _ensure_ocr_path(workdir)
    from extract_table import TableImageExtractor  # type: ignore

    t0 = time.perf_counter()
    _lay = str(layout).strip().lower()
    layout_mode = "dual" if _lay in ("dual", "double") else "single"
    if fix_decimal is None:
        try:
            from .config import get_ocr_fix_decimal_cfg

            fix_decimal = get_ocr_fix_decimal_cfg()
        except Exception:
            fix_decimal = {"enabled": True}
    extractor_kw: Dict[str, Any] = {
        "scale": scale,
        "min_confidence": min_confidence,
        "fix_decimal": fix_decimal,
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

    # base_date 入参已废弃（兼容保留）；业务日只走画面 OCR / 系统日开关，不猜测/不对侧回填
    del base_date
    if date_cfg is None:
        from .config import get_ocr_date_cfg

        date_cfg = get_ocr_date_cfg(layout=layout_mode)
    date_enabled = bool((date_cfg or {}).get("enabled", True))
    if date_enabled:
        logger.info("%s日期识别 开始", pfx)
    else:
        logger.info("%s日期识别 已关闭，跳过 OCR 直接用系统日", pfx)
    resolved_date, date_source, date_ocr_meta = resolve_business_date(
        image_path=image_path,
        workdir=workdir,
        fallback_date=None,
        date_cfg=date_cfg,
        min_confidence=min_confidence,
        debug_dir=run_dir,
        log_prefix=pfx,
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

    if date_source == "system" and resolved_date is not None:
        # date_ocr_enabled=false，或后续扩展的其它系统日路径
        base_date_str = resolved_date.strftime("%Y-%m-%d")
        logger.info(
            "%s日期识别 ms=%s 使用系统日 date=%s reason=%s",
            pfx,
            date_ms,
            base_date_str,
            date_ocr_out.get("reason"),
        )
    elif date_source != "ocr" or resolved_date is None:
        use_system = bool((date_cfg or {}).get("fallback_system", False))
        if not use_system:
            try:
                from .config import get_ocr_date_fallback_system

                use_system = get_ocr_date_fallback_system()
            except Exception:
                use_system = False
        if use_system:
            resolved_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            date_source = "system"
            base_date_str = resolved_date.strftime("%Y-%m-%d")
            date_ocr_out = dict(date_ocr_out)
            date_ocr_out["fallback"] = "system"
            date_ocr_out["date_str"] = base_date_str
            date_ocr_out["message"] = (
                str(date_ocr_out.get("message") or "未识别到左上角业务日期")
                + f"；已启用 date_fallback_system，用系统日 {base_date_str} 继续表格 OCR"
            )
            logger.warning(
                "%s日期识别 ms=%s 失败 reason=%s — 回退系统日 %s 继续表格识别",
                pfx,
                date_ms,
                date_ocr_out.get("reason"),
                base_date_str,
            )
        else:
            logger.warning(
                "%s日期识别 ms=%s 失败 source=%s reason=%s — 跳过表格识别",
                pfx,
                date_ms,
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
    else:
        base_date_str = resolved_date.strftime("%Y-%m-%d")
        logger.info("%s日期识别 ms=%s 成功 date=%s", pfx, date_ms, base_date_str)

    crop_candidates: List[Tuple[str, Optional[Tuple[float, float, float, float]]]] = list(
        _default_crop_candidates(table_crop=table_crop)
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
    fixed_good = False
    allow_auto = bool(auto_enable)

    def _good_enough(
        sc: Tuple[int, int, int, float],
        rows: Optional[List[Any]] = None,
        headers: Optional[List[Any]] = None,
    ) -> bool:
        """
        fixed 是否够格，够格则不再跑 auto。
        表头可用、列数够、且至少有填格；若上两行像摩尔纹假读，视为不够。
        """
        if sc[0] < 3:
            return False
        # 列过少：贴边误砍后的短表头不算够，开启 auto 时留给自动裁剪
        # 抓拍半屏常只有 5~6 列有效，门槛用 4（勿用 7）
        n_headers = len(list(headers or []))
        if n_headers < 4:
            return False
        # 只认出表头、数据全空：不算够（常见于漏检 0/电价），交给 auto 再裁一刀
        if int(sc[2]) <= 0:
            return False
        row_list = list(rows or [])
        # 电力/计划行（末行电价除外）若像全 0 被读成垃圾，不算够
        upper = row_list[:-1] if len(row_list) >= 2 else row_list
        for row in upper:
            try:
                if extractor._row_looks_like_moire_zero_junk(row):
                    return False
            except Exception:
                break
        return True

    t_table = time.perf_counter()
    n_cands = len(unique)
    logger.info("%s表格识别 开始 候选者%s位", pfx, n_cands)
    for i, (label, c) in enumerate(unique):
        short = _cand_short_name(label)
        ord_i = i + 1  # 日志从 1 起
        skip_reason = ""
        is_auto = "auto" in (label or "").lower()
        if is_auto and not allow_auto:
            skip_reason = "auto_disabled"
        elif is_auto and fixed_good:
            skip_reason = "fixed_ok"
        if skip_reason:
            logger.info(
                "%s表格识别 候选%s %s：跳过",
                pfx,
                ord_i,
                short,
            )
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

        logger.debug(
            "%s表格识别 候选%s %s crop=%s",
            pfx,
            ord_i,
            short,
            c,
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
                logger.exception(
                    "%s保存候选裁剪失败 候选%s %s", pfx, ord_i, short
                )

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
            logger.warning(
                "%s表格识别 候选%s %s：无效 ms=%s",
                pfx,
                ord_i,
                short,
                cand_ms,
            )
            rec["invalid"] = True
            cand_records.append(rec)
            continue

        logger.info(
            "%s表格识别 候选%s %s：ms=%s",
            pfx,
            ord_i,
            short,
            cand_ms,
        )
        cand_records.append(rec)
        if sc > best_score:
            best_score = sc
            best = result
            best_idx = i
            best_cand_path = cand_saved
        if (not is_auto) and _good_enough(
            sc,
            rows=list(getattr(result, "rows", None) or []),
            headers=headers,
        ):
            fixed_good = True

    table_ms = int((time.perf_counter() - t_table) * 1000)
    timing_out: Dict[str, Any] = {
        "date_ms": date_ms,
        "table_ms": table_ms,
        "ocr_ms": date_ms + table_ms,
    }

    winner_label = ""
    winner_short = ""
    if best_idx >= 0 and best_idx < len(cand_records):
        win_lab = str(cand_records[best_idx].get("label") or "")
        winner_label = _cand_display_name(win_lab)
        winner_short = _cand_short_name(win_lab)

    for rec in cand_records:
        idx = int(rec["index"])
        if rec.get("status") == "skipped":
            rec["reason"] = _explain_loser(
                (-1, -1, -1, 0.0),
                best_score,
                winner_label or "—",
                skipped=True,
                skip_reason=str(rec.get("skip_reason") or ""),
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
            logger.debug(
                "%s表格识别 表头清洗 cols=%s",
                pfx,
                len(clean_h),
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
            logger.debug(
                "%s表格识别 同伴回填 gap_filled=%s winner=%s",
                pfx,
                gap_filled_n,
                winner_label or best_idx,
            )

    if best is None:
        logger.warning(
            "%s表格识别 ms=%s 失败 — 未得到有效表头/数据行",
            pfx,
            table_ms,
        )
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

    # 履历 base_date 仍记 OCR 识别日；写库时间戳按 pi.pi_date_back_days 提前后再与表头时刻拼接
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
    logger.info(
        "%s表格识别 ms=%s 完成 winner=%s  filled=%s points=%s",
        pfx,
        table_ms,
        winner_short or winner_label or f"cand_{best_idx}",
        ocr_filled,
        len(points),
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
