# -*- coding: utf-8 -*-
"""显示器画面区域自动定位（纯图像，不做 OCR）。

用途：远景四周有墙/桌面，或近景宽已贴满、上下仍有环境时，
先裁到「屏内」再做左右分缝 / 日期 / 表格比例，避免环境占比搞歪相对坐标。

策略（缩略图，目标通常 <100ms）：
1) 行/列标准差（屏 UI 纹理远高于墙/桌）→ 主路径
2) Sobel 能量投影（带间隙合并）→ 辅路径
3) 亮屏连通域 → 兜底；近满幅框一律丢弃（避免墙桌连成整图误判「满屏」）
4) 左右策略（第一原则：少破坏屏内，防误截）：
   - side_trim：单侧大留白（台卡/墙柱）→ 可较狠收
   - soft_trim：两侧都有一点环境 → 软收（有上限）+ 左右外扩更大
   - width_full：两侧几乎无环境 → x 强制 0~1，只收上下
5) structure 水平范围作安全外包络：其它候选不得比它更窄砍左右
6) 近分时优先更宽框（宁多留环境，不啃屏边）
7) 无有效候选 → 回退整图 (0,0,1,1)

失败一律保守回退，宁可不裁。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_PROBE_W = 800
# 裁后面积至少占原图这么多，否则疑似切到台卡/局部
_MIN_AREA_RATIO = 0.22
_MIN_WIDTH_RATIO = 0.45
_MIN_HEIGHT_RATIO = 0.32
# 裁后宽高都 ≥ 此值视为「假满屏」（亮屏连通常把墙桌也吃进去），直接丢弃该候选
_REJECT_NEAR_FULL = 0.94
# 真正几乎满屏才跳过：收边总和很小
_MIN_TRIM = 0.04
_PAD_RATIO = 0.015  # 上下外扩，避免切掉日期栏/底边
_PAD_X_SOFT = 0.035  # 左右软收外扩更大：宁可多留环境
# 左右策略：单侧留白 ≥ 此值且对侧很小 → 去台卡
_SIDE_TRIM_MIN = 0.12
_SIDE_TRIM_OTHER_MAX = 0.08
# 两侧都很小 → 视为宽贴满；否则软收，每侧最多收这么多
_WIDTH_FULL_MAX = 0.04
_SOFT_TRIM_MAX = 0.08
# 近分阈值：分差 ≤ 此值时优先更宽框
_NEAR_SCORE_EPS = 0.02


def detect_monitor_region(
    image_path: str | Path,
    *,
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    定位显示器内容区，返回相对整图比例裁剪框。

    返回 dict:
      ok, crop (x0,y0,x1,y1), mode (energy|bright|full|fallback|disabled),
      ms, detail
    """
    t0 = time.perf_counter()
    full = (0.0, 0.0, 1.0, 1.0)
    out: Dict[str, Any] = {
        "ok": False,
        "crop": list(full),
        "mode": "fallback",
        "ms": 0,
        "detail": {},
        "enabled": bool(enabled),
    }
    if not enabled:
        out["mode"] = "disabled"
        out["ok"] = True  # 显式关闭视为「用整图成功」
        out["ms"] = int((time.perf_counter() - t0) * 1000)
        return out

    try:
        import cv2
        import numpy as np
    except ImportError:
        out["ms"] = int((time.perf_counter() - t0) * 1000)
        out["detail"] = {"error": "cv2_unavailable"}
        return out

    path = Path(image_path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            raise ValueError("empty_image")
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            raise ValueError("decode_failed")
        h0, w0 = bgr.shape[:2]
        scale = min(1.0, float(_MAX_PROBE_W) / max(w0, 1))
        if scale < 0.999:
            bgr = cv2.resize(
                bgr,
                (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        detail: Dict[str, Any] = {
            "probe_wh": [int(w), int(h)],
            "src_wh": [int(w0), int(h0)],
        }

        structure_box = _detect_via_structure(gray)
        energy_box = _detect_via_content_energy(gray)
        bright_box = _detect_via_bright_body(gray)
        detail["structure_raw"] = structure_box
        detail["energy_raw"] = energy_box
        detail["bright_raw"] = bright_box

        # 宽贴满 / 软收 / 去台卡：先归一化左右，再打分
        structure_box, sp = _normalize_horizontal(structure_box)
        energy_box, ep = _normalize_horizontal(energy_box)
        bright_box, bp = _normalize_horizontal(bright_box)
        detail["structure"] = structure_box
        detail["energy"] = energy_box
        detail["bright"] = bright_box
        detail["width_policy"] = {
            "structure": sp,
            "energy": ep,
            "bright": bp,
        }

        # structure 水平安全外包络（含左右外扩）：其它模式不得比它更窄砍左右
        x_envelope = _horizontal_safety_envelope(structure_box, policy=sp)
        detail["x_envelope"] = list(x_envelope) if x_envelope else None

        candidates: list[Tuple[str, Tuple[float, float, float, float], float, str]] = []
        # structure 优先参与打分；同面积时略加分
        for mode, box, bonus, policy in (
            ("structure", structure_box, 0.08, sp),
            ("energy", energy_box, 0.0, ep),
            ("bright", bright_box, -0.05, bp),
        ):
            if not box:
                continue
            pad_x = _pad_x_for_policy(policy)
            scored = _validate_and_pad(box, pad_y=_PAD_RATIO, pad_x=pad_x)
            if scored is None:
                continue
            crop, score = scored
            candidates.append((mode, crop, score + bonus, policy))

        if not candidates:
            out["detail"] = detail
            out["mode"] = "fallback"
            out["ms"] = int((time.perf_counter() - t0) * 1000)
            logger.info("显示器区域未命中，回退整图")
            return out

        # 高分优先；近分时优先更宽框（防 bright 以 0.00x 分差啃掉屏右缘）
        candidates.sort(key=lambda item: item[2], reverse=True)
        best_score = candidates[0][2]
        near = [c for c in candidates if best_score - c[2] <= _NEAR_SCORE_EPS]
        mode, crop, score, policy = max(
            near,
            key=lambda c: (c[1][2] - c[1][0], c[2]),
        )
        detail["chosen_score"] = round(score, 4)
        detail["chosen_policy"] = policy
        detail["near_prefer_wider"] = len(near) > 1 and near[0][0] != mode
        detail["candidates"] = [
            {
                "mode": m,
                "crop": list(c),
                "score": round(s, 4),
                "policy": p,
            }
            for m, c, s, p in candidates
        ]

        x0, y0, x1, y1 = crop
        # 套 structure 水平外包络：只扩不缩
        if x_envelope is not None:
            ex0, ex1 = x_envelope
            nx0, nx1 = min(x0, ex0), max(x1, ex1)
            if nx0 < x0 - 1e-6 or nx1 > x1 + 1e-6:
                detail["envelope_applied"] = {
                    "before": [round(x0, 4), round(x1, 4)],
                    "after": [round(nx0, 4), round(nx1, 4)],
                }
            x0, x1 = nx0, nx1

        trim = (x0 + (1.0 - x1)) + (y0 + (1.0 - y1))
        # 仅当最佳候选几乎不收边时跳过；假满屏候选已在 validate 剔除
        if trim < _MIN_TRIM:
            out["ok"] = True
            out["crop"] = list(full)
            out["mode"] = "full"
            out["detail"] = detail
            out["ms"] = int((time.perf_counter() - t0) * 1000)
            logger.info("显示器区域已近满屏，跳过裁剪 ms=%s", out["ms"])
            return out

        out["ok"] = True
        out["crop"] = [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]
        out["mode"] = mode
        out["detail"] = detail
        out["ms"] = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "显示器区域 ok=1 mode=%s policy=%s crop=%s ms=%s",
            mode,
            policy,
            out["crop"],
            out["ms"],
        )
        return out
    except Exception as e:
        logger.warning("显示器区域定位失败，回退整图: %s", e)
        out["detail"] = {"error": str(e)}
        out["mode"] = "fallback"
        out["ms"] = int((time.perf_counter() - t0) * 1000)
        return out


def apply_monitor_crop(
    image_path: str | Path,
    out_path: str | Path,
    crop: Tuple[float, float, float, float],
) -> Path:
    """按比例裁剪整图并落盘（JPEG）。"""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("显示器裁剪需要 Pillow") from e

    src = Path(image_path)
    dst = Path(out_path)
    x0, y0, x1, y1 = (float(v) for v in crop)
    with Image.open(src) as im:
        w, h = im.size
        box = (
            max(0, int(x0 * w)),
            max(0, int(y0 * h)),
            min(w, int(x1 * w)),
            min(h, int(y1 * h)),
        )
        if box[2] <= box[0] + 8 or box[3] <= box[1] + 8:
            raise ValueError(f"无效 monitor crop: {crop} → {box}")
        cropped = im.crop(box)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            cropped.convert("RGB").save(dst, quality=95)
        else:
            cropped.save(dst)
    return dst


def prepare_monitor_work_image(
    image_path: str | Path,
    run_dir: str | Path,
    *,
    enabled: bool = True,
    out_name: str = "00_monitor.jpg",
) -> Tuple[Path, Dict[str, Any], str]:
    """
    检测并（必要时）落盘屏内图，供后续分缝/OCR 使用。

    @return (work_path, monitor_meta, monitor_path_str)
            work_path 为屏内图或原图；monitor_path_str 有裁剪时为路径，否则 ""。
    """
    src = Path(image_path)
    meta = detect_monitor_region(src, enabled=bool(enabled))
    work = src
    mon_str = ""
    if (
        bool(enabled)
        and meta.get("ok")
        and meta.get("mode") not in ("full", "disabled", "fallback")
    ):
        try:
            crop_m = meta.get("crop") or [0, 0, 1, 1]
            m_box = (
                float(crop_m[0]),
                float(crop_m[1]),
                float(crop_m[2]),
                float(crop_m[3]),
            )
            if (m_box[2] - m_box[0]) * (m_box[3] - m_box[1]) < 0.985:
                mon_out = Path(run_dir) / out_name
                apply_monitor_crop(src, mon_out, m_box)
                work = mon_out
                mon_str = str(mon_out)
                logger.info(
                    "显示器裁剪 mode=%s crop=%s → %s",
                    meta.get("mode"),
                    m_box,
                    mon_out,
                )
            else:
                meta = dict(meta)
                meta["mode"] = "full"
        except Exception as e:
            logger.exception("显示器裁剪落盘失败，回退整图")
            meta = dict(meta)
            meta["ok"] = False
            meta["mode"] = "fallback"
            meta["apply_error"] = str(e)
            work = src
            mon_str = ""
    try:
        import json

        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "monitor.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("写 monitor.json 失败")
    return work, meta, mon_str


def _pad_x_for_policy(policy: str) -> float:
    """软收/去台卡：左右外扩更大；宽贴满左右已是 0~1，pad 无所谓。"""
    if policy in ("soft_trim", "side_trim"):
        return _PAD_X_SOFT
    return _PAD_RATIO


def _horizontal_safety_envelope(
    structure_box: Optional[Tuple[float, float, float, float]],
    *,
    policy: str,
) -> Optional[Tuple[float, float]]:
    """
    structure 归一化后的水平范围 + 对应外扩，作为最终裁剪左右不得更窄的安全外包络。
    """
    if not structure_box:
        return None
    x0, _y0, x1, _y1 = (float(v) for v in structure_box)
    pad_x = _pad_x_for_policy(policy)
    x0 = max(0.0, x0 - pad_x)
    x1 = min(1.0, x1 + pad_x)
    if x1 <= x0 + 0.05:
        return None
    return (x0, x1)


def _validate_and_pad(
    box: Tuple[float, float, float, float],
    *,
    pad_y: float,
    pad_x: Optional[float] = None,
) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
    """校验比例框并外扩；返回 (crop, score) 或 None。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    pw = float(_PAD_RATIO if pad_x is None else pad_x)
    ph = float(pad_y)
    x0 = max(0.0, x0 - pw)
    y0 = max(0.0, y0 - ph)
    x1 = min(1.0, x1 + pw)
    y1 = min(1.0, y1 + ph)
    bw = x1 - x0
    bh = y1 - y0
    if bw < _MIN_WIDTH_RATIO or bh < _MIN_HEIGHT_RATIO:
        return None
    area = bw * bh
    if area < _MIN_AREA_RATIO:
        return None
    # 假满屏：亮屏/形态学把墙桌连成整图 → 直接丢弃，避免误判「已满屏跳过」
    if bw >= _REJECT_NEAR_FULL and bh >= _REJECT_NEAR_FULL:
        return None
    # 横屏显示器：裁后宽高比大致合理（含双窗拼在一起）
    ar = bw / max(bh, 1e-6)
    if ar < 0.85 or ar > 3.2:
        return None
    # 收益以「上下收边」为主；左右收边权重很低（防靠砍宽度虚高分）
    trim_y = y0 + (1.0 - y1)
    trim_x = x0 + (1.0 - x1)
    trim = trim_y * 1.0 + trim_x * 0.10 + (1.0 - area) * 0.30
    area_pen = max(0.0, 0.55 - area) * 2.0
    score = trim - area_pen
    if (1.0 - area) < _MIN_TRIM:
        score *= 0.2
    return (x0, y0, x1, y1), float(score)


def _normalize_horizontal(
    box: Optional[Tuple[float, float, float, float]],
) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
    """
    左右策略（少破坏屏内优先）：
    - side_trim：单侧明显大留白、对侧很小 → 保留检测左右（去台卡）
    - soft_trim：两侧都有一点环境 → 软收（每侧有上限），不追求收尽
    - width_full：两侧几乎无环境 → 左右强制 0~1，只留上下裁

    @return (box_or_None, policy_name)
    """
    if box is None:
        return None, "none"
    x0, y0, x1, y1 = (float(v) for v in box)
    x0 = max(0.0, min(1.0, x0))
    x1 = max(0.0, min(1.0, x1))
    y0 = max(0.0, min(1.0, y0))
    y1 = max(0.0, min(1.0, y1))
    if x1 <= x0 + 0.05:
        return (x0, y0, x1, y1), "invalid"
    left = x0
    right = 1.0 - x1
    # 单侧环境（台卡/墙柱）：一侧 ≥12% 且对侧 <8%
    one_sided = (left >= _SIDE_TRIM_MIN and right < _SIDE_TRIM_OTHER_MAX) or (
        right >= _SIDE_TRIM_MIN and left < _SIDE_TRIM_OTHER_MAX
    )
    if one_sided:
        return (x0, y0, x1, y1), "side_trim"
    # 两侧都几乎贴边 → 宽贴满
    if left <= _WIDTH_FULL_MAX and right <= _WIDTH_FULL_MAX:
        return (0.0, y0, 1.0, y1), "width_full"
    # 软收：保留检测方向，但每侧最多收 _SOFT_TRIM_MAX（宁多留环境）
    sx0 = min(x0, _SOFT_TRIM_MAX)
    sx1 = max(x1, 1.0 - _SOFT_TRIM_MAX)
    return (sx0, y0, sx1, y1), "soft_trim"


def _band_from_projection(
    proj,
    *,
    thr_ratio: float = 0.28,
    min_run_ratio: float = 0.28,
    gap_fill_ratio: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """
    一维投影 → 高能量带（可合并小间隙），返回相对 [0,1) 的 (lo, hi)。
    """
    import numpy as np

    p = np.asarray(proj, dtype=np.float32)
    n = int(p.size)
    if n < 16:
        return None
    k = max(3, n // 80)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(p, kernel, mode="same")
    peak = float(smooth.max())
    if peak <= 1e-3:
        return None
    thr = peak * float(thr_ratio)
    mask = smooth >= thr

    segs: list[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        segs.append((i, j))
        i = j
    if not segs:
        return None

    # 合并小间隙：间隙须短，且谷底不能明显低于两侧（防屏+桌粘连）
    gap_max = max(2, int(n * float(gap_fill_ratio))) if gap_fill_ratio > 0 else 0
    merged: list[Tuple[int, int]] = [segs[0]]
    for a, b in segs[1:]:
        pa, pb = merged[-1]
        gap = a - pb
        if gap_max > 0 and 0 < gap <= gap_max:
            gap_mean = float(smooth[pb:a].mean()) if a > pb else thr
            left_m = float(smooth[pa:pb].mean())
            right_m = float(smooth[a:b].mean())
            if gap_mean >= 0.75 * min(left_m, right_m) and gap_mean >= thr * 0.7:
                merged[-1] = (pa, b)
                continue
        merged.append((a, b))

    # 选「长度 × 段内均值」最大的合并段，避免桌面噪点长段抢走
    best_score = -1.0
    best_a, best_b = merged[0]
    for a, b in merged:
        length = b - a
        if length < max(8, int(n * min_run_ratio * 0.55)):
            continue
        mass = float(smooth[a:b].mean()) * float(length)
        if mass > best_score:
            best_score = mass
            best_a, best_b = a, b
    if best_score < 0:
        best_a, best_b = max(merged, key=lambda ab: ab[1] - ab[0])
    if best_b - best_a < max(8, int(n * min_run_ratio)):
        return None

    # 两端最多外扩 ~3%：禁止沿浅谷一路扩进桌面/墙
    max_expand = max(2, int(n * 0.03))
    expand_thr = thr * 0.85
    a = best_a
    for _ in range(max_expand):
        if a <= 0 or smooth[a - 1] < expand_thr:
            break
        a -= 1
    b = best_b
    for _ in range(max_expand):
        if b >= n or smooth[min(n - 1, b)] < expand_thr:
            break
        b += 1
    lo = a / float(n)
    hi = b / float(n)
    if hi - lo < min_run_ratio:
        return None
    return (lo, hi)


def _detect_via_structure(gray) -> Optional[Tuple[float, float, float, float]]:
    """
    行/列灰度标准差投影：屏内 UI 纹理高，墙/桌面低。
    对「宽贴满、上下有环境」最稳，也是本场景主路径。
    """
    import numpy as np

    h, w = gray.shape[:2]
    g = gray.astype(np.float32)
    row = g.std(axis=1)
    col = g.std(axis=0)
    # 标准差主带本身连续，不做间隙合并（桌面另有一段，合并必炸）
    y_band = _band_from_projection(
        row, thr_ratio=0.24, min_run_ratio=0.30, gap_fill_ratio=0.0
    )
    x_band = _band_from_projection(
        col, thr_ratio=0.24, min_run_ratio=0.40, gap_fill_ratio=0.0
    )
    if y_band is None:
        y_band = _band_from_projection(
            row, thr_ratio=0.18, min_run_ratio=0.28, gap_fill_ratio=0.0
        )
    if y_band is None:
        return None
    y0, y1 = y_band
    if x_band is None:
        x0, x1 = 0.0, 1.0
    else:
        x0, x1 = x_band
    # 左右是否贴边由 _normalize_horizontal 统一判定
    return (float(x0), float(y0), float(x1), float(y1))


def _detect_via_content_energy(gray) -> Optional[Tuple[float, float, float, float]]:
    """Sobel 能量行/列投影 → 屏内容包围盒。"""
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # 轻中央加权，避免完全吃掉贴边屏；权重别太大
    yy = np.linspace(0.75, 1.0, h, dtype=np.float32)
    yy = np.minimum(yy, yy[::-1])
    xx = np.linspace(0.75, 1.0, w, dtype=np.float32)
    xx = np.minimum(xx, xx[::-1])
    mag_w = mag * np.outer(yy, xx)

    row = mag_w.mean(axis=1)
    col = mag_w.mean(axis=0)
    y_band = _band_from_projection(
        row, thr_ratio=0.14, min_run_ratio=0.28, gap_fill_ratio=0.08
    )
    x_band = _band_from_projection(
        col, thr_ratio=0.18, min_run_ratio=0.40, gap_fill_ratio=0.05
    )
    if y_band is None:
        return None
    if x_band is None:
        x0, x1 = 0.0, 1.0
    else:
        x0, x1 = x_band
    y0, y1 = y_band
    return (float(x0), float(y0), float(x1), float(y1))


def _detect_via_bright_body(gray) -> Optional[Tuple[float, float, float, float]]:
    """亮屏主体：相对中位更亮的连通大块；近满幅结果由 validate 丢弃。"""
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    med = float(np.median(eq))
    # 阈值略抬高，减少把浅墙连进屏
    thr = max(med + 18.0, 105.0)
    _, bw = cv2.threshold(eq, thr, 255, cv2.THRESH_BINARY)
    kx = max(5, w // 50)
    ky = max(5, h // 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx | 1, ky | 1))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    nlab, _labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    best_i = -1
    best_area = 0
    min_area = int(w * h * 0.18)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or ww < w * 0.40 or hh < h * 0.28:
            continue
        ar = ww / max(hh, 1)
        if ar < 0.8 or ar > 3.5:
            continue
        # 宽高都接近整图 → 假满屏，跳过
        if ww >= int(w * _REJECT_NEAR_FULL) and hh >= int(h * _REJECT_NEAR_FULL):
            continue
        if area > best_area:
            best_area = area
            best_i = i
    if best_i < 0:
        return None
    x = int(stats[best_i, cv2.CC_STAT_LEFT])
    y = int(stats[best_i, cv2.CC_STAT_TOP])
    ww = int(stats[best_i, cv2.CC_STAT_WIDTH])
    hh = int(stats[best_i, cv2.CC_STAT_HEIGHT])
    return (
        x / float(w),
        y / float(h),
        (x + ww) / float(w),
        (y + hh) / float(h),
    )
