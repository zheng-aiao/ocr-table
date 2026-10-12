# -*- coding: utf-8 -*-
"""加载 config.yaml。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"

_config: Optional[Dict[str, Any]] = None

DEFAULT_TAG = "GX.DEMO.OCR"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """读取 YAML 配置（可重复调用以热加载）。"""
    global _config
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
    return _config


def get_config() -> Dict[str, Any]:
    if _config is None:
        return load_config()
    return _config


def get_schedule_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    定时处置配置。
    优先 capture.schedule，兼容顶层 schedule。
    阶段开关按序截断：任一为 false 则该阶段及之后不执行。
    """
    cfg = cfg or get_config()
    raw = (cfg.get("capture") or {}).get("schedule")
    if not isinstance(raw, dict) or not raw:
        raw = cfg.get("schedule") or {}
    if not isinstance(raw, dict):
        raw = {}

    enabled = bool(raw.get("enabled", False))
    try:
        interval = int(raw.get("interval_minutes", 15) or 15)
    except (TypeError, ValueError):
        interval = 15
    interval = max(1, interval)

    capture = bool(raw.get("capture", True))
    ocr = bool(raw.get("ocr", True))
    pi = bool(raw.get("pi", True))
    # 按序截断：前一阶段关闭则后续也不执行
    do_capture = capture
    do_ocr = do_capture and ocr
    do_pi = do_ocr and pi

    return {
        "enabled": enabled,
        "interval_minutes": interval,
        "capture": capture,
        "ocr": ocr,
        "pi": pi,
        "do_capture": do_capture,
        "do_ocr": do_ocr,
        "do_pi": do_pi,
    }


def parse_crop(raw: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not raw or not str(raw).strip():
        return None
    parts = [float(x.strip()) for x in str(raw).split(",")]
    if len(parts) != 4:
        raise ValueError("crop 须为 x0,y0,x1,y1")
    return parts[0], parts[1], parts[2], parts[3]


def get_single_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """capture.single（兼容旧 capture.tag/crop、ocr.crop、pi.tag）。"""
    cfg = cfg or get_config()
    cap = cfg.get("capture") or {}
    single = dict(cap.get("single") or {})
    if not str(single.get("tag") or "").strip():
        legacy = cap.get("tag") or (cfg.get("pi") or {}).get("tag")
        if legacy:
            single["tag"] = legacy
    if single.get("crop") is None or not str(single.get("crop") or "").strip():
        legacy_crop = cap.get("crop")
        if legacy_crop is None or not str(legacy_crop).strip():
            legacy_crop = (cfg.get("ocr") or {}).get("crop")
        if legacy_crop is not None:
            single["crop"] = legacy_crop
    return single


def get_capture_tag(cfg: Optional[Dict[str, Any]] = None) -> str:
    """单屏测点前缀：capture.single.tag。"""
    raw = get_single_cfg(cfg).get("tag")
    return str(raw or DEFAULT_TAG).strip()


def get_capture_crop(
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """单屏 OCR 裁剪：capture.single.crop。"""
    return parse_crop(get_single_cfg(cfg).get("crop"))


def save_config(cfg: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Path:
    """把当前配置写回 YAML。"""
    global _config
    data = cfg if cfg is not None else get_config()
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    _config = data
    return cfg_path


def update_capture_mode(mode: str) -> Dict[str, Any]:
    """更新 capture.mode（single|dual），写回 config.yaml 并热更新内存。"""
    import re

    m = str(mode or "").strip().lower()
    if m not in ("single", "dual"):
        raise ValueError("mode 须为 single 或 dual")
    cfg = get_config()
    cfg.setdefault("capture", {})["mode"] = m

    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_capture = False
    replaced = False
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if re.match(r"^capture:\s*(#.*)?$", line.rstrip("\r\n")):
            in_capture = True
            out.append(line)
            continue
        if in_capture and indent == 0 and stripped and not stripped.startswith("#"):
            in_capture = False
        if (
            in_capture
            and not replaced
            and indent == 2
            and re.match(r"^mode:\s*", stripped)
        ):
            nl = "\n" if line.endswith("\n") else ""
            if line.endswith("\r\n"):
                nl = "\r\n"
            out.append(f"{' ' * indent}mode: {m}{nl}")
            replaced = True
            continue
        out.append(line)
    if replaced:
        DEFAULT_CONFIG_PATH.write_text("".join(out), encoding="utf-8")
    else:
        save_config(cfg)
    return {"ok": True, "mode": m}


def update_capture_tag(tag: str) -> Dict[str, Any]:
    """更新内存中的 capture.single.tag，并尽量就地改写 config.yaml（保留注释）。"""
    import re

    name = (tag or "").strip()
    if not name:
        raise ValueError("测点名不能为空")
    cfg = get_config()
    cap = cfg.setdefault("capture", {})
    single = cap.setdefault("single", {})
    single["tag"] = name
    # 清理旧扁平字段
    cap.pop("tag", None)
    if isinstance(cfg.get("pi"), dict):
        cfg["pi"].pop("tag", None)
        cfg["pi"].pop("write_mode", None)
        cfg["pi"].pop("tag_mapping", None)

    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_capture = False
    in_single = False
    replaced = False
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if re.match(r"^capture:\s*(#.*)?$", line.rstrip("\r\n")):
            in_capture = True
            in_single = False
            out.append(line)
            continue
        if in_capture and indent == 0 and stripped and not stripped.startswith("#"):
            in_capture = False
            in_single = False
        if in_capture and indent == 2 and re.match(r"^single:\s*", stripped):
            in_single = True
            out.append(line)
            continue
        if in_single and indent <= 2 and stripped and not stripped.startswith("#"):
            if not re.match(r"^tag:\s*", stripped):
                in_single = False
        # 只改 capture.single.tag（缩进 4）
        if (
            in_single
            and not replaced
            and indent == 4
            and re.match(r"^tag:\s*", stripped)
        ):
            nl = "\n" if line.endswith("\n") else ""
            if line.endswith("\r\n"):
                nl = "\r\n"
            out.append(f'{" " * indent}tag: "{name}"{nl}')
            replaced = True
            continue
        out.append(line)
    if replaced:
        DEFAULT_CONFIG_PATH.write_text("".join(out), encoding="utf-8")
    else:
        save_config(cfg)
    return {"ok": True, "tag": name}


def update_pi_tag(tag: str) -> Dict[str, Any]:
    return update_capture_tag(tag)


def resolve_ocr_workdir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """OCR 引擎目录：默认本仓库 image-table-extract/（可整体打包，无需依赖外层目录）。"""
    cfg = cfg or get_config()
    raw = (cfg.get("ocr") or {}).get("workdir", "image-table-extract")
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT_DIR / p).resolve()
    return p


DATA_DIR = ROOT_DIR / "data"
_LEGACY_DATA_TOP = frozenset({"runs", "ocr_runs", "captures", "exports"})


def resolve_project_path(path: Optional[str | Path]) -> Path:
    """
    解析项目内路径；兼容旧布局 ROOT/{captures|ocr_runs|exports|runs}/…
    相对路径若仍以旧顶层目录名开头，一律落到 data/ 下。
    绝对路径若旧位置不存在，则回退到 data/ 下同相对路径。
    """
    if path is None or path == "":
        return Path()
    p = Path(path)
    if not p.is_absolute():
        parts = p.parts
        if parts and parts[0] in _LEGACY_DATA_TOP:
            p = DATA_DIR.joinpath(*parts)
        else:
            p = ROOT_DIR / p
    try:
        p_res = p.resolve()
    except OSError:
        p_res = p
    if p_res.exists():
        return p_res
    try:
        rel = p_res.relative_to(ROOT_DIR.resolve())
    except ValueError:
        return p_res
    if rel.parts and rel.parts[0] in _LEGACY_DATA_TOP:
        # 旧绝对路径已迁走：指向 data/ 下同相对路径
        return (DATA_DIR / rel).resolve()
    return p_res


def resolve_ocr_artifacts_root(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """
    OCR 过程中间文件根目录（默认 data/ocr_runs/）。
    读 ocr.artifacts_dir；兼容旧顶层 artifacts.dir / ocr.artifacts_dir。
    """
    cfg = cfg or get_config()
    ocr = cfg.get("ocr") or {}
    legacy = cfg.get("artifacts") or {}
    raw = (
        ocr.get("artifacts_dir")
        or legacy.get("dir")
        or "data/ocr_runs"
    )
    p = resolve_project_path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_pi_date_back_days(cfg: Optional[Dict[str, Any]] = None) -> int:
    """OCR 识别日相对写 PI 日期提前的天数：1=提前一天，0=当天不提前。"""
    cfg = cfg or get_config()
    raw = (cfg.get("ocr") or {}).get("pi_date_back_days", 1)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 1
    return max(0, n)


def get_ocr_min_points(cfg: Optional[Dict[str, Any]] = None) -> int:
    """OCR 有效点数绝对下限；低于则本轮识别失败、不写 PI。"""
    cfg = cfg or get_config()
    try:
        n = int((cfg.get("ocr") or {}).get("min_points", 8))
    except (TypeError, ValueError):
        n = 8
    return max(0, n)


def get_ocr_early_stop(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """
    裁剪竞选是否允许提前结束。
    true（默认）：仅当先行候选（auto）数据行全部填满（100%）时跳过 fixed；
    任一行有空格则 auto+fixed 都 OCR 再择优。
    false：始终两边都跑。
    """
    cfg = cfg or get_config()
    return bool((cfg.get("ocr") or {}).get("early_stop", True))


def get_ocr_early_stop_min_density(cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    early_stop 密度门槛（百分数）。
    现以「每行每格填满」为准；本值保留兼容，建议 100。
    """
    cfg = cfg or get_config()
    try:
        n = int((cfg.get("ocr") or {}).get("early_stop_min_density", 100))
    except (TypeError, ValueError):
        n = 100
    return max(50, min(100, n))


def get_ocr_skip_full_when_ok(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """已废弃：不再跑整图 full；保留读取以兼容旧配置。"""
    cfg = cfg or get_config()
    return bool((cfg.get("ocr") or {}).get("skip_full_when_ok", True))


def get_ocr_table_crop(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    layout: str = "single",
    side: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """
    表格固定比例裁剪（相对当前送入图：双屏=半屏，单屏=整图）。

    优先级：
    dual → capture.dual.{side}.table_crop → ocr.table_crop → 内置默认
    single → capture.single.table_crop → ocr.table_crop → 内置默认
    """
    cfg = cfg or get_config()
    mode = "dual" if str(layout).strip().lower() == "dual" else "single"
    default = (
        (0.02, 0.52, 0.985, 0.90) if mode == "dual" else (0.10, 0.52, 0.995, 0.90)
    )
    crop: Optional[Tuple[float, float, float, float]] = None
    cap = cfg.get("capture") or {}
    if mode == "dual":
        dual = cap.get("dual") or {}
        side_key = (side or "left").strip().lower()
        if side_key not in ("left", "right"):
            side_key = "left"
        side_cfg = dual.get(side_key) or {}
        crop = parse_crop(str(side_cfg.get("table_crop") or ""))
        if crop is None:
            crop = parse_crop(str(dual.get("table_crop") or ""))
    else:
        single = get_single_cfg(cfg)
        crop = parse_crop(str(single.get("table_crop") or ""))
    if crop is None:
        crop = parse_crop(str((cfg.get("ocr") or {}).get("table_crop") or ""))
    return crop if crop is not None else default


def get_ocr_parallel_sides(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """双屏左右 OCR 是否并行（默认开；不影响识别结果）。"""
    cfg = cfg or get_config()
    return bool((cfg.get("ocr") or {}).get("parallel_sides", True))


def get_ocr_max_width(cfg: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """OCR 放大后最大宽度；未配置则用引擎默认（8000）。"""
    cfg = cfg or get_config()
    raw = (cfg.get("ocr") or {}).get("max_ocr_width", None)
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def get_ocr_scale(cfg: Optional[Dict[str, Any]] = None) -> float:
    """表格 OCR 放大倍数（默认 2.0，偏速度）。"""
    cfg = cfg or get_config()
    try:
        return float((cfg.get("ocr") or {}).get("scale", 2.0))
    except (TypeError, ValueError):
        return 2.0


def get_ocr_onnx_threads(cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    RapidOCR/ONNX 每会话 CPU 线程数（默认 2）。
    默认 -1 会占满全部核心，双屏并行时整机卡顿、浏览器打开网页极慢。
    """
    cfg = cfg or get_config()
    try:
        n = int((cfg.get("ocr") or {}).get("onnx_threads", 2))
    except (TypeError, ValueError):
        n = 2
    import os

    cpu_n = os.cpu_count() or 4
    return max(1, min(n, cpu_n))


# 日期 OCR：enabled / scale 固定写死；优先自动找「日期」标签，date_crop 仅作回退
_DATE_OCR_ENABLED = True
_DATE_OCR_SCALE = 1.0
_DATE_OCR_AUTO_LOCATE = True
_DEFAULT_DATE_CROP_SINGLE = (0.14, 0.10, 0.48, 0.26)  # 整图回退：偏高矮，躲开曲线 Y 轴
_DEFAULT_DATE_CROP_DUAL = (0.0, 0.16, 0.48, 0.26)  # 半屏回退：标题下方日期框（双屏保持原值）


def get_ocr_date_cfg(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    layout: str = "single",
    side: Optional[str] = None,
) -> Dict[str, Any]:
    """
    左上角「日期」筛选框 OCR。
    默认先 OCR 定位「日期」再裁右侧筛选框；失败才用
    capture.single.date_crop / capture.dual.{left|right}.date_crop。
    相对当前送入图（双屏=半屏；单屏=整图）。enabled/scale 代码写死。
    """
    cfg = cfg or get_config()
    mode = "dual" if str(layout).strip().lower() == "dual" else "single"
    default = (
        _DEFAULT_DATE_CROP_DUAL if mode == "dual" else _DEFAULT_DATE_CROP_SINGLE
    )
    crop: Optional[Tuple[float, float, float, float]] = None
    cap = cfg.get("capture") or {}
    if mode == "dual":
        dual = cap.get("dual") or {}
        side_key = str(side or "left").strip().lower()
        if side_key not in ("left", "right"):
            side_key = "left"
        side_cfg = dual.get(side_key) or {}
        if isinstance(side_cfg, dict):
            crop = parse_crop(str(side_cfg.get("date_crop") or ""))
        # 左右共用：capture.dual.date_crop
        if crop is None:
            crop = parse_crop(str(dual.get("date_crop") or ""))
    else:
        single = get_single_cfg(cfg)
        crop = parse_crop(str(single.get("date_crop") or ""))

    # 兼容旧 ocr.date
    legacy = (cfg.get("ocr") or {}).get("date") or {}
    if not isinstance(legacy, dict):
        legacy = {}
    if crop is None:
        crop = parse_crop(str(legacy.get("crop") or ""))
    if crop is None:
        crop = default

    auto_locate = _DATE_OCR_AUTO_LOCATE
    if "auto_locate" in legacy:
        auto_locate = bool(legacy.get("auto_locate"))
    ocr_cfg = cfg.get("ocr") or {}
    if isinstance(ocr_cfg, dict) and "date_auto_locate" in ocr_cfg:
        auto_locate = bool(ocr_cfg.get("date_auto_locate"))

    extra_crops: List[Tuple[float, float, float, float]] = []
    for item in legacy.get("crops") or []:
        if isinstance(item, str):
            p = parse_crop(item)
            if p:
                extra_crops.append(p)
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            try:
                extra_crops.append(
                    (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
                )
            except (TypeError, ValueError):
                continue
    return {
        "enabled": _DATE_OCR_ENABLED,
        "crop": crop,
        "crops": extra_crops,
        "scale": _DATE_OCR_SCALE,
        "layout": mode,
        "side": side,
        "auto_locate": auto_locate,
    }


def get_ocr_sanitize_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    OCR 同行量级清洗配置。
    返回：enabled / ratio_k / min_samples。
    仅对整数为主的电力/计划行做比值清洗；电价行不清洗。
    位数众数硬删已废弃（对电价尖峰误杀）；旧键 majority_ratio/boundary_pct 忽略。
    """
    cfg = cfg or get_config()
    raw = (cfg.get("ocr") or {}).get("sanitize") or {}
    if not isinstance(raw, dict):
        raw = {}
    enabled = bool(raw.get("enabled", True))

    def _f(key: str, default: float) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    def _i(key: str, default: int) -> int:
        try:
            return int(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "enabled": enabled,
        "ratio_k": max(1.0, _f("ratio_k", 3.0)),
        "min_samples": max(2, _i("min_samples", 4)),
    }


def table_expected_cells(
    headers: Optional[List[Any]] = None,
    rows: Optional[List[Any]] = None,
) -> int:
    """
    本屏表格预计格数 = 数据行数 × 时间刻度列数。
    列数优先取 headers；无表头时取各行之最长列宽。
    """
    row_list = list(rows or [])
    nrows = len(row_list)
    if nrows <= 0:
        return 0
    header_list = list(headers or [])
    ncols = len(header_list)
    if ncols <= 0:
        ncols = max((len(r) for r in row_list), default=0)
    return max(0, nrows * ncols)


def get_ocr_expected_cells(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    row_count: Optional[int] = None,
    headers: Optional[List[Any]] = None,
    rows: Optional[List[Any]] = None,
) -> int:
    """
    表格预计值数量（履历展示）。
    默认 = 本屏 OCR 数据行 × 时间列；若配置了 ocr.expected_cells 则用固定值覆盖。
    """
    del row_count  # 兼容旧调用，忽略
    cfg = cfg or get_config()
    ocr = cfg.get("ocr") or {}
    if ocr.get("expected_cells") is not None:
        try:
            return max(0, int(ocr.get("expected_cells")))
        except (TypeError, ValueError):
            pass
    return table_expected_cells(headers, rows)


def new_ocr_run_dir(
    mode: str = "single",
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    新建一次 OCR 会话目录（始终落盘过程中间文件）：
      data/ocr_runs/single/YYYYMMDD_HHMMSS_mmm/
      data/ocr_runs/dual/YYYYMMDD_HHMMSS_mmm/
    """
    from datetime import datetime
    import uuid

    m = "dual" if str(mode).strip().lower() == "dual" else "single"
    root = resolve_ocr_artifacts_root(cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out = root / m / stamp
    if out.exists():
        out = root / m / f"{stamp}_{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True, exist_ok=True)
    return out
