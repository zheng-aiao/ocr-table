# -*- coding: utf-8 -*-
"""加载 config.yaml。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# app/backend/config.py → 项目根
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
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
        interval = int(raw.get("interval_sec", 10) or 0)
    except (TypeError, ValueError):
        interval = 10
    interval = max(0, interval)

    capture = bool(raw.get("capture", True))
    ocr = bool(raw.get("ocr", True))
    pi = bool(raw.get("pi", True))
    # 按序截断：前一阶段关闭则后续也不执行
    do_capture = capture
    do_ocr = do_capture and ocr
    do_pi = do_ocr and pi

    return {
        "enabled": enabled,
        "interval_sec": interval,
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


def _parse_side_crop_raw(raw: Any) -> Optional[Tuple[float, float, float, float]]:
    """空 / 未配 → None；非法格式抛 ValueError。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return parse_crop(text)


def get_camera_table_search(
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    表格 auto 先验搜索窗：cameras.{left|right}.table_search（x0,y0,x1,y1 相对整图）。

    先裁该窗再在窗内 table_auto。空 / 未配置 → None，整图 auto。
    """
    cfg = cfg or get_config()
    side_key = str(side or "").strip().lower()
    if side_key not in ("left", "right"):
        return None
    cams = cfg.get("cameras") or {}
    cam = cams.get(side_key) if isinstance(cams, dict) else None
    if isinstance(cam, dict) and "table_search" in cam:
        return _parse_side_crop_raw(cam.get("table_search"))
    return None


def get_camera_modal_dim(
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    弹窗蒙板判定门槛：cameras.{left|right}.modal_dim。

    可选键 mean_max / content_max / grey_min；未配则空 dict，引擎用侧别默认。
    """
    cfg = cfg or get_config()
    side_key = str(side or "").strip().lower()
    if side_key not in ("left", "right"):
        return {}
    cams = cfg.get("cameras") or {}
    cam = cams.get(side_key) if isinstance(cams, dict) else None
    if not isinstance(cam, dict):
        return {}
    raw = cam.get("modal_dim")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for key in ("mean_max", "content_max", "grey_min"):
        if key not in raw or raw[key] is None:
            continue
        try:
            out[key] = float(raw[key])
        except (TypeError, ValueError):
            continue
    return out


def get_capture_table_search(
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """兼容旧名：等同 get_camera_table_search。"""
    return get_camera_table_search(side, cfg)

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
    """更新 capture.mode：single | dual | double，写回 config.yaml。"""
    import re

    m = str(mode or "").strip().lower()
    if m in ("double_camera", "cameras"):
        m = "double"
    if m not in ("single", "dual", "double"):
        raise ValueError("mode 须为 single、dual 或 double")
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
        # capture 下尚无 mode：插到 capture: 下一行
        out2: list[str] = []
        inserted = False
        for line in out:
            out2.append(line)
            if (
                not inserted
                and re.match(r"^capture:\s*(#.*)?$", line.rstrip("\r\n"))
            ):
                out2.append(f"  mode: {m}\n")
                inserted = True
        DEFAULT_CONFIG_PATH.write_text("".join(out2), encoding="utf-8")
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
    """OCR 引擎目录：默认 app/ocr_engine/（随 app 打包）。"""
    cfg = cfg or get_config()
    raw = (cfg.get("ocr") or {}).get("workdir", "app/ocr_engine")
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


def get_pi_value_max(cfg: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """
    PI 写入绝对值上限（pi.value_max）。
    |value| 超出则该点跳过不写库；OCR/前端展示不受影响。
    未配置或 <=0 表示不限制。
    """
    cfg = cfg or get_config()
    raw = (cfg.get("pi") or {}).get("value_max")
    if raw in (None, ""):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def get_pi_date_back_days(cfg: Optional[Dict[str, Any]] = None) -> int:
    """OCR 识别日相对写 PI 日期提前的天数：1=提前一天，0=当天不提前。"""
    cfg = cfg or get_config()
    pi = cfg.get("pi") or {}
    ocr = cfg.get("ocr") or {}
    # 优先 pi.pi_date_back_days；兼容旧配置 ocr.pi_date_back_days
    raw = pi.get("pi_date_back_days", ocr.get("pi_date_back_days", 1))
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


def get_ocr_auto_enable(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """
    兼容旧配置。本分支表格 OCR 只走自动定位，该开关不再决定是否跑固定框。
    """
    cfg = cfg or get_config()
    return bool((cfg.get("ocr") or {}).get("auto_enable", True))


def get_ocr_skip_full_when_ok(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """已废弃：不再跑整图 full；保留读取以兼容旧配置。"""
    cfg = cfg or get_config()
    return bool((cfg.get("ocr") or {}).get("skip_full_when_ok", True))


def get_ocr_table_crop(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    layout: str = "double",
    side: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """
    表格固定比例裁剪（双摄：相对各侧有效区域图）。

    优先级：capture.double.{side}.table_crop → ocr.table_crop → 内置默认
    （兼容读旧键 capture.dual）
    """
    del layout
    cfg = cfg or get_config()
    default = (0.02, 0.52, 0.985, 0.90)
    crop: Optional[Tuple[float, float, float, float]] = None
    cap = cfg.get("capture") or {}
    block = cap.get("double") or cap.get("dual") or {}
    side_key = (side or "left").strip().lower()
    if side_key not in ("left", "right"):
        side_key = "left"
    side_cfg = block.get(side_key) or {}
    crop = parse_crop(str(side_cfg.get("table_crop") or ""))
    if crop is None:
        crop = parse_crop(str(block.get("table_crop") or ""))
    if crop is None:
        crop = parse_crop(str((cfg.get("ocr") or {}).get("table_crop") or ""))
    return crop if crop is not None else default


def get_ocr_parallel_sides(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """双屏左右 OCR 是否并行（默认关：同机浏览网页更顺；不影响识别结果）。"""
    cfg = cfg or get_config()
    return bool((cfg.get("ocr") or {}).get("parallel_sides", False))


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
    """表格 OCR 参考放大倍数（默认 1.35；引擎按表宽自适应，大图少放）。"""
    cfg = cfg or get_config()
    try:
        return float((cfg.get("ocr") or {}).get("scale", 1.35))
    except (TypeError, ValueError):
        return 1.35


def get_ocr_onnx_threads(cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    RapidOCR/ONNX 每会话 CPU 线程数（默认 2）。
    默认 -1 会占满全部核心，双屏并行时整机卡顿、浏览器打开网页极慢。

    自动按逻辑核预算：识别最多用一半核，并按 parallel_sides 分摊到左右会话，
    给浏览器 / uvicorn 留出余量。
    """
    cfg = cfg or get_config()
    try:
        n = int((cfg.get("ocr") or {}).get("onnx_threads", 2))
    except (TypeError, ValueError):
        n = 2
    import os

    cpu_n = max(1, os.cpu_count() or 4)
    # 同机浏览历史页时，OCR 最多吃一半逻辑核（至少留 2；双核机留 1）
    reserve = 2 if cpu_n >= 4 else 1
    budget = max(1, cpu_n - reserve)
    sides = 2 if get_ocr_parallel_sides(cfg) else 1
    max_per_side = max(1, budget // sides)
    return max(1, min(n, max_per_side))


# 日期 OCR：scale 固定；enabled 读 ocr.date_ocr_enabled；date_crop = 搜索带（双摄）
_DATE_OCR_SCALE = 1.0
# 默认搜索带：顶栏附近窄带（相对有效区域图）
_DEFAULT_DATE_SEARCH_BAND = (0.02, 0.02, 0.55, 0.34)


def get_ocr_date_ocr_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """
    本分支不识别画面日期，始终返回 false。
    业务日使用电脑系统日；写库日仍按 pi.pi_date_back_days 提前。
    """
    del cfg
    return False


def get_ocr_date_fallback_system(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """
    仅当 date_ocr_enabled=true 且日期 OCR 失败时生效：
    true：用本机系统日（年月日）继续表格 OCR 并写 PI（时分仍取表头；写库日仍按 pi.pi_date_back_days 提前）。
    false（默认）：跳过表格 OCR、不写 PI。
    """
    cfg = cfg or get_config()
    return bool((cfg.get("ocr") or {}).get("date_fallback_system", False))


def get_ocr_date_cfg(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    layout: str = "double",
    side: Optional[str] = None,
) -> Dict[str, Any]:
    """
    双摄日期配置。本分支不跑画面日期 OCR，业务日固定为电脑系统日。
    date_crop 不再使用。
    """
    del layout
    cfg = cfg or get_config()
    crop: Optional[Tuple[float, float, float, float]] = None
    cap = cfg.get("capture") or {}
    block = cap.get("double") or cap.get("dual") or {}
    side_key = str(side or "left").strip().lower()
    if side_key not in ("left", "right"):
        side_key = "left"
    side_cfg = block.get(side_key) or {}
    if isinstance(side_cfg, dict):
        crop = parse_crop(str(side_cfg.get("date_crop") or ""))
    if crop is None:
        crop = parse_crop(str(block.get("date_crop") or ""))
    if crop is None:
        crop = _DEFAULT_DATE_SEARCH_BAND
    return {
        "enabled": get_ocr_date_ocr_enabled(cfg),
        "crop": crop,  # 搜索带
        "crops": [],
        "scale": _DATE_OCR_SCALE,
        "layout": "double",
        "side": side_key,
        "auto_locate": True,
        "fallback_system": get_ocr_date_fallback_system(cfg),
    }


def get_ocr_fix_decimal_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    OCR 小数点补点后处理配置。
    返回：enabled / ratio_k。
    false 时保留 OCR 原文（不猜补 5/6 位小数点、不做电价行 4 位 d.ddd）。
    5 位无点：双候选 + 同行/邻格选优，最优相对误差 > ratio_k 则拒补。
    配置缺失时默认 enabled=true（与历史行为一致）。
    """
    cfg = cfg or get_config()
    raw = (cfg.get("ocr") or {}).get("fix_decimal") or {}
    if isinstance(raw, bool):
        return {"enabled": raw, "ratio_k": 3.0}
    if not isinstance(raw, dict):
        raw = {}
    try:
        ratio_k = max(1.0, float(raw.get("ratio_k", 3.0)))
    except (TypeError, ValueError):
        ratio_k = 3.0
    return {
        "enabled": bool(raw.get("enabled", True)),
        "ratio_k": ratio_k,
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


def normalize_ocr_artifacts_mode(mode: str = "single") -> str:
    """OCR 中间产物顶层目录名：single | dual | double。"""
    m = str(mode or "single").strip().lower()
    if m in ("double", "double_camera", "cameras"):
        return "double"
    if m == "dual":
        return "dual"
    return "single"


def new_ocr_run_dir(
    mode: str = "single",
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    新建一次 OCR 会话目录（始终落盘过程中间文件；按日分区，与 captures 一致）：
      data/ocr_runs/YYYYMMDD/single/YYYYMMDD_HHMMSS_mmm/
      data/ocr_runs/YYYYMMDD/dual/YYYYMMDD_HHMMSS_mmm/      # 双屏拼缝
      data/ocr_runs/YYYYMMDD/double/YYYYMMDD_HHMMSS_mmm/    # 双摄；下含 left/ right/
    """
    from datetime import datetime
    import uuid

    m = normalize_ocr_artifacts_mode(mode)
    root = resolve_ocr_artifacts_root(cfg)
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out = root / day / m / stamp
    if out.exists():
        out = root / day / m / f"{stamp}_{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True, exist_ok=True)
    return out
