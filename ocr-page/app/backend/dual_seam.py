# -*- coding: utf-8 -*-
"""双屏窗缝自动定位（纯图像，不做 OCR）。

两种现场叠窗：
1) 右屏在上：右窗左侧深色导航栏 → 亮→暗竖带左缘即缝
2) 左屏在上：左窗右缘压住右屏 → 强竖向亮度跳变即缝

耗时目标：缩略图列扫描，通常 <50ms；失败回退配置比例。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 搜索窗缝的水平比例范围（相对整图宽）
_DEFAULT_SEARCH = (0.38, 0.72)
_DEFAULT_FALLBACK = 0.55
_DEFAULT_GAP = 0.012  # 左右裁剪各离开缝的比例，减轻串边
_MAX_PROBE_W = 640


def detect_dual_seam(
    image_path: str | Path,
    *,
    search: Tuple[float, float] = _DEFAULT_SEARCH,
    fallback: float = _DEFAULT_FALLBACK,
) -> Dict[str, Any]:
    """
    定位左右窗竖缝。

    返回 dict:
      ok, seam (0~1), mode (sidebar|edge|fallback), ms, detail
    """
    t0 = time.perf_counter()
    path = Path(image_path)
    out: Dict[str, Any] = {
        "ok": False,
        "seam": float(fallback),
        "mode": "fallback",
        "ms": 0,
        "detail": {},
    }
    try:
        import cv2
        import numpy as np
    except ImportError:
        out["ms"] = int((time.perf_counter() - t0) * 1000)
        out["detail"] = {"error": "cv2_unavailable"}
        return out

    try:
        # 缩略读图，避免大图全分辨率
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

        x0 = max(1, int(w * float(search[0])))
        x1 = min(w - 2, int(w * float(search[1])))
        if x1 <= x0 + 4:
            raise ValueError("search_range_too_narrow")

        # 取中部竖带做统计，避开顶栏/任务栏反光
        y0 = int(h * 0.12)
        y1 = int(h * 0.88)
        band = gray[y0:y1, :]
        col_mean = band.mean(axis=0).astype(np.float32)
        col_std = band.std(axis=0).astype(np.float32)

        # —— 信号 A：右屏导航栏（亮|深色竖条）——
        sidebar = _score_dark_sidebar(col_mean, col_std, x0, x1, w)
        # —— 信号 B：窗边竖向跳变 ——
        edge = _score_vertical_edge(col_mean, x0, x1)

        seam_x: Optional[int] = None
        mode = "fallback"
        detail: Dict[str, Any] = {
            "sidebar": sidebar,
            "edge": edge,
            "probe_wh": [int(w), int(h)],
            "src_wh": [int(w0), int(h0)],
        }

        # 导航栏命中优先（右压左特征更稳）
        if sidebar.get("ok") and float(sidebar.get("score") or 0) >= 18.0:
            seam_x = int(sidebar["x"])
            mode = "sidebar"
        elif edge.get("ok") and float(edge.get("score") or 0) >= 12.0:
            seam_x = int(edge["x"])
            mode = "edge"
        elif sidebar.get("ok") and float(sidebar.get("score") or 0) >= float(
            edge.get("score") or 0
        ):
            seam_x = int(sidebar["x"])
            mode = "sidebar_weak"
        elif edge.get("ok"):
            seam_x = int(edge["x"])
            mode = "edge_weak"

        if seam_x is not None:
            seam = float(seam_x) / float(w)
            # 映射回原图比例（缩略与原图同比例）
            seam = max(0.35, min(0.75, seam))
            out["ok"] = True
            out["seam"] = round(seam, 4)
            out["mode"] = mode
        else:
            out["seam"] = float(fallback)
            out["mode"] = "fallback"

        out["detail"] = detail
    except Exception as e:
        logger.warning("双屏缝定位失败，回退 fallback=%.3f: %s", fallback, e)
        out["detail"] = {"error": str(e)}
        out["seam"] = float(fallback)
        out["mode"] = "fallback"

    out["ms"] = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "双屏缝定位 ms=%s ok=%s seam=%.4f mode=%s",
        out["ms"],
        out["ok"],
        out["seam"],
        out["mode"],
    )
    return out


def seam_to_side_crops(
    seam: float,
    *,
    gap: float = _DEFAULT_GAP,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """缝 → 左右半屏 crop（留 gap 防串边）。"""
    s = max(0.30, min(0.80, float(seam)))
    g = max(0.0, min(0.04, float(gap)))
    left = (0.0, 0.0, max(0.25, s - g * 0.5), 1.0)
    right = (min(0.85, s + g * 0.5), 0.0, 1.0, 1.0)
    if left[2] <= left[0] + 0.15 or right[2] <= right[0] + 0.15:
        # 异常缝，退回对称
        left = (0.0, 0.0, 0.545, 1.0)
        right = (0.555, 0.0, 1.0, 1.0)
    return left, right


def _score_dark_sidebar(
    col_mean,
    col_std,
    x0: int,
    x1: int,
    w: int,
) -> Dict[str, Any]:
    """亮内容 | 深色导航栏：找突变后持续偏暗的左缘。"""
    import numpy as np

    best = {"ok": False, "x": 0, "score": 0.0}
    # 导航栏宽度约 4%~12% 半屏 ≈ 整图 2%~6%
    band_lo = max(4, int(w * 0.018))
    band_hi = max(band_lo + 2, int(w * 0.070))
    for x in range(x0, x1):
        left = col_mean[max(0, x - 6) : x]
        if left.size < 3:
            continue
        left_m = float(left.mean())
        # 右侧一段应显著更暗且方差不太大（实心栏）
        for bw in (band_lo, (band_lo + band_hi) // 2, band_hi):
            if x + bw >= len(col_mean):
                continue
            right = col_mean[x : x + bw]
            right_s = col_std[x : x + bw]
            right_m = float(right.mean())
            right_std = float(right_s.mean())
            # 左亮右暗
            if left_m < 95 or right_m > 85:
                continue
            drop = left_m - right_m
            if drop < 35:
                continue
            # 实心栏方差通常低于表格区
            score = drop + max(0.0, 25.0 - right_std) * 0.35
            if score > float(best["score"]):
                best = {"ok": True, "x": int(x), "score": float(score), "bw": int(bw)}
    return best


def _score_vertical_edge(col_mean, x0: int, x1: int) -> Dict[str, Any]:
    """列均值一阶差分峰值 → 窗边。"""
    import numpy as np

    best = {"ok": False, "x": 0, "score": 0.0}
    # 平滑后再差分，抑制摩尔纹
    k = 5
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(col_mean, kernel, mode="same")
    for x in range(x0, x1):
        a = float(smooth[x - 2])
        b = float(smooth[min(len(smooth) - 1, x + 2)])
        score = abs(b - a)
        if score > float(best["score"]):
            best = {"ok": True, "x": int(x), "score": float(score)}
    return best
