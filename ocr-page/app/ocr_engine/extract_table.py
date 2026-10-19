# -*- coding: utf-8 -*-
"""
图片表格识别：OCR + 列网格对齐 + 粘连数值拆分，导出 CSV / TXT。
"""

from __future__ import annotations

import csv
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 进程内复用 RapidOCR；并行双屏时每线程独立实例，避免 ONNX 会话互踩
_SHARED_OCR = None
_THREAD_OCR = threading.local()
_OCR_INIT_LOCK = threading.Lock()
# 按线程记录已绑定的 onnx_threads，变更时重建会话
_THREAD_OCR_THREADS = threading.local()


@dataclass
class OcrCell:
    """OCR 识别到的一个文本块。"""

    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


@dataclass
class TableExtractResult:
    """表格提取结果。"""

    headers: List[str]
    rows: List[List[str]]
    raw_row_count: int
    image_path: str
    # 最终清洗前可解析填格数（OCR 识别数）；清洗后采纳数由下游 points 统计
    ocr_filled: int = 0


class TableImageExtractor:
    """从图片中识别表格并导出。"""

    TIME_PATTERN = re.compile(r"^(\d{1,2})[:：.\-]?(\d{2})$")
    TIME_WITH_SEC_PATTERN = re.compile(
        r"^(\d{1,2})[:：](\d{2})[:：](\d{2})$")
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    DEFAULT_TIME_STEP = 15
    DEFAULT_END_TIME = "24:00"
    # 整格只有「-」：内部标记，最终输出为空且不再单列重识
    NODATA_MARK = "\x00"
    # 放大后宽度上限：原 2.5x 可达 ~10k 宽，整图 OCR 很慢；8000 兼顾速度与小字可读性
    MAX_OCR_WIDTH = 8000
    # 默认表区（与 _default_table_crop / auto 失败回退一致；仅下半屏）
    DEFAULT_TABLE_CROP_RATIO: Tuple[float, float, float, float] = (
        0.0,
        0.50,
        0.99,
        0.92,
    )
    # 现货表体固定：1 行时刻表头 + 恰好 3 行数据（电力/计划/电价）
    # 斜拍拆行时禁止输出第 4 条数据行
    EXPECTED_DATA_ROWS = 3
    # 纠偏门槛：低于此角度视为已正；过大更像误检
    DESKEW_MIN_ANGLE_DEG = 0.5
    DESKEW_MAX_ANGLE_DEG = 18.0
    # 蓝底倾角只在表体够宽、够实的时候才信；零星蓝点的外接矩形会给出假倾角
    DESKEW_BLUE_MIN_AREA_RATIO = 0.02
    DESKEW_BLUE_MIN_WIDTH_RATIO = 0.55
    # 蓝底角与表格线中位角差过这个值，以表格线为准，不按蓝底转
    DESKEW_BLUE_HOUGH_MAX_DISAGREE_DEG = 4.0
    # 透视四边形：超过这些更像「蓝底吃进曲线/假梯形」，宁可不拉
    DESKEW_QUAD_MAX_EDGE_ANGLE_DEG = 6.0
    DESKEW_QUAD_MAX_SIDE_SKEW = 0.12
    DESKEW_QUAD_MAX_TOP_BOT_ANGLE_DIFF_DEG = 4.0
    DESKEW_QUAD_MIN_AREA_RATIO = 0.85
    DESKEW_QUAD_MAX_AR_CHANGE = 0.12
    # auto 裁剪常含上半曲线：透视只用更靠下的蓝表带，避免左高右低假四边形
    DESKEW_QUAD_IGNORE_TOP_RATIO = 0.32

    @staticmethod
    def resolve_image_path(image_path: str, base_dir: Optional[Path] = None) -> Path:
        """
        解析并校验图片路径（支持相对/绝对路径）。

        @param image_path 用户指定的图片路径
        @param base_dir   相对路径基准目录
        @return 绝对路径
        """
        if not image_path or not str(image_path).strip():
            raise ValueError("图片路径不能为空")
        raw = Path(image_path.strip().strip('"').strip("'"))
        if not raw.is_absolute() and base_dir is not None:
            raw = base_dir / raw
        resolved = raw.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError("图片不存在: " + str(resolved))
        if resolved.suffix.lower() not in TableImageExtractor.IMAGE_SUFFIXES:
            raise ValueError("不支持的图片格式: " + resolved.suffix)
        return resolved

    @staticmethod
    def read_image_file(image_path: str) -> np.ndarray:
        """
        从指定位置读取图片（兼容 Windows 中文路径）。

        @param image_path 图片路径
        @return BGR 图像数组
        """
        path = TableImageExtractor.resolve_image_path(image_path)
        buffer = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图片: " + str(path))
        return img

    def __init__(
        self,
        scale: float = 2.5,
        min_confidence: float = 0.25,
        fix_decimal: Optional[Dict[str, Any]] = None,
        max_ocr_width: Optional[int] = None,
        onnx_threads: Optional[int] = None,
    ):
        """
        @param scale          识别前放大倍数，宽表建议 2.5
        @param min_confidence OCR 最低置信度
        @param fix_decimal    小数点补点配置（见 config.yaml ocr.fix_decimal）
        @param max_ocr_width  放大后最大宽度（像素），默认 MAX_OCR_WIDTH
        @param onnx_threads   ONNX 每会话 intra 线程数；默认 2，避免占满 CPU 拖慢浏览器
        """
        self.scale = scale
        self.min_confidence = min_confidence
        self._ocr = None
        if max_ocr_width is not None:
            try:
                mw = int(max_ocr_width)
                self.max_ocr_width = mw if mw > 0 else self.MAX_OCR_WIDTH
            except (TypeError, ValueError):
                self.max_ocr_width = self.MAX_OCR_WIDTH
        else:
            self.max_ocr_width = self.MAX_OCR_WIDTH
        try:
            nt = int(onnx_threads) if onnx_threads is not None else 1
        except (TypeError, ValueError):
            nt = 1
        # 同机浏览网页时勿吃满逻辑核：最多用一半（至少留 1～2）
        cpu_n = max(1, os.cpu_count() or 4)
        reserve = 2 if cpu_n >= 4 else 1
        self.onnx_threads = max(1, min(nt, cpu_n - reserve, cpu_n))
        if isinstance(fix_decimal, bool):
            self.fix_decimal_enabled = fix_decimal
            self.fix_decimal_ratio_k = 3.0
        elif isinstance(fix_decimal, dict):
            self.fix_decimal_enabled = bool(fix_decimal.get("enabled", True))
            try:
                self.fix_decimal_ratio_k = max(
                    1.0, float(fix_decimal.get("ratio_k", 3.0))
                )
            except (TypeError, ValueError):
                self.fix_decimal_ratio_k = 3.0
        else:
            self.fix_decimal_enabled = True
            self.fix_decimal_ratio_k = 3.0
        # 5 位双候选得分接近时的破平阈值（相对误差差）
        self.fix_decimal_tie_eps = 0.15
        # 最近一次 auto 裁剪的比例框 (x0,y0,x1,y1)，供上游竞选打分
        self.last_auto_crop_ratio: Optional[Tuple[float, float, float, float]] = None

    def _get_ocr(self):
        """懒加载 OCR 引擎（线程内单例，兼容 rapidocr_onnxruntime / rapidocr）。"""
        want = int(self.onnx_threads)
        if self._ocr is not None and getattr(_THREAD_OCR_THREADS, "n", None) == want:
            return self._ocr
        ocr = getattr(_THREAD_OCR, "ocr", None)
        bound = getattr(_THREAD_OCR_THREADS, "n", None)
        if ocr is None or bound != want:
            with _OCR_INIT_LOCK:
                ocr = getattr(_THREAD_OCR, "ocr", None)
                bound = getattr(_THREAD_OCR_THREADS, "n", None)
                if ocr is None or bound != want:
                    # 限制 OpenCV / OpenMP，避免与 ONNX 叠加占满核
                    try:
                        cv2.setNumThreads(max(1, min(want, 2)))
                    except Exception:
                        pass
                    for key in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    ):
                        os.environ.setdefault(key, str(want))
                    try:
                        from rapidocr_onnxruntime import RapidOCR
                    except ImportError:
                        from rapidocr import RapidOCR
                    # RapidOCR 默认 intra_op=-1（用满全部 CPU）；限制后浏览器才不会卡死
                    ocr = RapidOCR(
                        intra_op_num_threads=want,
                        inter_op_num_threads=1,
                    )
                    _THREAD_OCR.ocr = ocr
                    _THREAD_OCR_THREADS.n = want
                    global _SHARED_OCR
                    if _SHARED_OCR is None:
                        _SHARED_OCR = ocr
        self._ocr = ocr
        return self._ocr

    def _scale_image_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """按 scale 放大，但限制最大宽度以控制耗时。"""
        scale = float(self.scale or 1.0)
        if scale <= 0:
            scale = 1.0
        h, w = img.shape[:2]
        target_w = int(w * scale)
        max_w = float(self.max_ocr_width or self.MAX_OCR_WIDTH)
        if target_w > max_w:
            scale = max_w / float(w)
        if abs(scale - 1.0) < 1e-3:
            return img
        return cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        """四边形角点排成 TL, TR, BR, BL。"""
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(4)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _estimate_hough_median_angle(self, img: np.ndarray) -> Optional[float]:
        """
        近水平霍夫线段的长度加权中位角（度）。
        拍屏网格偏淡：高阈值 Canny 经常一条线都没有，验收就会放行假旋转。
        高低两档边缘合并，并放宽最短线长。
        OpenCV 4 返回 (N,1,4)，OpenCV 5 返回 (N,4)，统一 reshape。
        """
        h, w = img.shape[:2]
        if h < 20 or w < 40:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.bitwise_or(
            cv2.Canny(gray, 50, 150, apertureSize=3),
            cv2.Canny(gray, 25, 80, apertureSize=3),
        )
        min_len = max(int(w * 0.055), 32)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(22, w // 80),
            minLineLength=min_len,
            maxLineGap=max(10, w // 70),
        )
        if lines is None:
            return None
        # OpenCV 5+: (N,4)；旧版: (N,1,4)
        pts = np.asarray(lines).reshape(-1, 4)
        angles: List[float] = []
        weights: List[float] = []
        for x1, y1, x2, y2 in pts:
            dx = float(x2) - float(x1)
            dy = float(y2) - float(y1)
            if abs(dx) < 1e-3:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            if abs(ang) > 25.0:
                continue
            angles.append(ang)
            weights.append(math.hypot(dx, dy))
        if len(angles) < 8:
            return None
        order = np.argsort(np.asarray(angles, dtype=np.float64))
        ang_s = np.asarray(angles, dtype=np.float64)[order]
        w_s = np.asarray(weights, dtype=np.float64)[order]
        cum = np.cumsum(w_s)
        idx = int(np.searchsorted(cum, cum[-1] * 0.5))
        idx = min(max(idx, 0), len(ang_s) - 1)
        return float(ang_s[idx])

    def _table_body_blue_mask(
        self,
        img: np.ndarray,
        *,
        ignore_top: int = 0,
    ) -> np.ndarray:
        """
        浅蓝表体行 mask（对侧光/左侧阴影鲁棒）。

        现场斜拍时常左侧更暗：原 HSV 固定 V≥140 会把暗侧同色行判丢，
        左右断裂后 boundingRect 偏右、上扩又易吃进曲线。此处：
        1) V 通道 CLAHE 均衡亮度；2) 放宽 V 下限；3) 相对偏蓝兜底；
        4) 强水平闭运算把同行断开的蓝底连回一条。
        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hch, sch, vch = cv2.split(hsv)
        tile_w = max(8, min(32, w // 40))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile_w, 8))
        hsv_eq = cv2.merge([hch, sch, clahe.apply(vch)])
        # 均衡后仍放宽 V：暗侧常见均值约 110~130
        mask_hsv = cv2.inRange(hsv_eq, (85, 8, 55), (135, 120, 255))
        b, g, r = cv2.split(img.astype(np.float32))
        blue_score = b - 0.5 * (g + r)
        mask_rel = (
            (blue_score > 3.0)
            & (b > 35)
            & (hch >= 85)
            & (hch <= 135)
        ).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask_hsv, mask_rel)
        top = max(0, min(int(ignore_top), h - 1))
        if top > 0:
            mask[:top, :] = 0
        # 宽核水平 CLOSE：把左侧阴影断开的同行蓝底粘回
        kx = max(41, w // 18)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 9)),
        )
        return mask

    def _deskew_via_blue_quad(
        self,
        img: np.ndarray,
        *,
        hough_angle: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        用浅蓝表体轮廓近似四边形，透视拉成矩形。
        斜拍主因是透视梯形，不是平面旋转——即使霍夫中位角很小也要尝试。
        失败、形变太小/过大（假四边形）、或拉正后残角更差则返回 None。
        """
        del hough_angle  # 透视与中位角解耦：不因「接近水平」跳过梯形纠正
        h, w = img.shape[:2]
        if h < 40 or w < 80:
            return None
        # 裁剪后图上部常是曲线区：忽略顶部，只拿蓝表带，避免左高右低假梯形
        ignore_top = max(1, int(h * self.DESKEW_QUAD_IGNORE_TOP_RATIO))
        mask = self._table_body_blue_mask(img, ignore_top=ignore_top)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        # 忽略顶部后表带占比更小，阈值相对放宽但须覆盖有效宽度
        if area < w * h * 0.04:
            return None
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            # 凸包再压一次，提高得到四边形的概率
            hull = cv2.convexHull(cnt)
            peri2 = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04 * peri2, True)
        if len(approx) != 4:
            # 最小外接矩形四点：轮廓不是干净四边形时的兜底
            # 此时透视≈旋转，交给 blue_min_rect / hough，避免重复拧
            return None
        src = self._order_quad_points(approx)
        top_w = float(np.linalg.norm(src[1] - src[0]))
        bot_w = float(np.linalg.norm(src[2] - src[3]))
        left_h = float(np.linalg.norm(src[3] - src[0]))
        right_h = float(np.linalg.norm(src[2] - src[1]))
        if top_w < 20 or bot_w < 20 or left_h < 20 or right_h < 20:
            return None
        width_skew = abs(top_w - bot_w) / max(top_w, bot_w)
        height_skew = abs(left_h - right_h) / max(left_h, right_h)
        edge_ang = math.degrees(
            math.atan2(float(src[1][1] - src[0][1]), float(src[1][0] - src[0][0]))
        )
        bot_ang = math.degrees(
            math.atan2(float(src[2][1] - src[3][1]), float(src[2][0] - src[3][0]))
        )
        # 必须有「明显」倾斜/透视才拉；否则正图也会被假四边形拧坏
        significant = (
            width_skew >= 0.03
            or height_skew >= 0.03
            or abs(edge_ang) >= 0.8
            or abs(bot_ang) >= 0.8
        )
        if not significant:
            return None
        # 过大更像蓝底误检（吃进曲线 / 单侧断裂），拉正会把表拧毁
        if (
            width_skew > self.DESKEW_QUAD_MAX_SIDE_SKEW
            or height_skew > self.DESKEW_QUAD_MAX_SIDE_SKEW
            or abs(edge_ang) > self.DESKEW_QUAD_MAX_EDGE_ANGLE_DEG
            or abs(bot_ang) > self.DESKEW_QUAD_MAX_EDGE_ANGLE_DEG
            or abs(edge_ang - bot_ang) > self.DESKEW_QUAD_MAX_TOP_BOT_ANGLE_DIFF_DEG
        ):
            return None
        out_w = int(max(top_w, bot_w))
        out_h = int(max(left_h, right_h))
        out_w = max(out_w, 64)
        out_h = max(out_h, 48)
        # 限制放大，避免异常四边形把图撑爆
        out_w = min(out_w, int(w * 1.25))
        out_h = min(out_h, int(h * 1.25))
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        m = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            img, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        before = self._estimate_hough_median_angle(img)
        after = self._estimate_hough_median_angle(warped)
        in_area = float(h * w)
        out_area = float(out_h * out_w)
        # 面积/宽高比剧变 → 假四边形
        if (
            out_area < in_area * self.DESKEW_QUAD_MIN_AREA_RATIO
            or out_area > in_area * 1.25
        ):
            return None
        in_ar = float(w) / max(float(h), 1.0)
        out_ar = float(out_w) / max(float(out_h), 1.0)
        if abs(out_ar - in_ar) / max(in_ar, 1e-3) > self.DESKEW_QUAD_MAX_AR_CHANGE:
            return None
        # 残角验收：有霍夫基准时须改善或接近水平；纯透视（中位角本就小）看几何即可
        if before is not None and after is not None and abs(before) >= 0.8:
            if abs(after) > abs(before) + 0.15:
                return None
            improved = abs(after) <= abs(before) - 0.25 or abs(after) < 0.8
            if not improved:
                return None
        elif before is None and after is not None and abs(after) > 2.5:
            return None
        return warped

    def _rotate_image_by_angle(
        self, img: np.ndarray, angle_deg: float
    ) -> Optional[np.ndarray]:
        """
        按倾角 θ 拉平。
        角度来自 atan2(dy, dx)，y 向下，正角表示右端更低。
        OpenCV 正角为逆时针，与这个定义同向，直接用 θ 即可把行拉平。
        """
        h, w = img.shape[:2]
        if abs(angle_deg) < 1e-6:
            return img
        center = (w * 0.5, h * 0.5)
        m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        cos_a = abs(m[0, 0])
        sin_a = abs(m[0, 1])
        nw = int(h * sin_a + w * cos_a)
        nh = int(h * cos_a + w * sin_a)
        m[0, 2] += (nw - w) * 0.5
        m[1, 2] += (nh - h) * 0.5
        return cv2.warpAffine(
            img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def _blue_mask_long_edge_angle(self, img: np.ndarray) -> Optional[float]:
        """浅蓝表体最小外接矩形长边相对水平的倾角（度）。"""
        h, w = img.shape[:2]
        if h < 40 or w < 80:
            return None
        mask = self._table_body_blue_mask(img, ignore_top=max(1, int(h * 0.08)))
        coords = cv2.findNonZero(mask)
        if coords is None or len(coords) < 80:
            return None
        # 灰白表体上的零星蓝点：外接矩形很大但像素极少，倾角不可用
        if len(coords) < max(800, int(h * w * self.DESKEW_BLUE_MIN_AREA_RATIO)):
            return None
        xs = coords[:, 0, 0]
        ys = coords[:, 0, 1]
        if float(xs.max() - xs.min()) < w * self.DESKEW_BLUE_MIN_WIDTH_RATIO:
            return None
        if float(ys.max() - ys.min()) < h * 0.08:
            return None
        rect = cv2.minAreaRect(coords)
        box = self._order_quad_points(cv2.boxPoints(rect))
        e01 = float(np.linalg.norm(box[1] - box[0]))
        e12 = float(np.linalg.norm(box[2] - box[1]))
        if e01 >= e12:
            return math.degrees(
                math.atan2(
                    float(box[1][1] - box[0][1]), float(box[1][0] - box[0][0])
                )
            )
        angle = math.degrees(
            math.atan2(
                float(box[2][1] - box[1][1]), float(box[2][0] - box[1][0])
            )
        )
        if angle > 0:
            return angle - 90.0
        return angle + 90.0

    def _deskew_via_blue_min_rect(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        浅蓝表体最小外接矩形长边倾角 → 整图旋转。
        用蓝底长边 + 霍夫残角双重验收：任一明显变差则丢弃。
        """
        angle = self._blue_mask_long_edge_angle(img)
        if angle is None:
            return None
        if abs(angle) < self.DESKEW_MIN_ANGLE_DEG:
            return None
        if abs(angle) > self.DESKEW_MAX_ANGLE_DEG:
            return None
        before_hough = self._estimate_hough_median_angle(img)
        # 表格线已经接近水平，或与蓝底角明显不一致：蓝底是误检，不转
        if before_hough is not None and (
            abs(before_hough) < self.DESKEW_MIN_ANGLE_DEG
            or abs(angle - before_hough) > self.DESKEW_BLUE_HOUGH_MAX_DISAGREE_DEG
        ):
            return None
        rotated = self._rotate_image_by_angle(img, angle)
        if rotated is None:
            return None
        after = self._blue_mask_long_edge_angle(rotated)
        if after is not None and abs(after) > abs(angle) - 0.3:
            return None
        after_hough = self._estimate_hough_median_angle(rotated)
        # 蓝底倾角与文字行不一致时，旋转会把霍夫中位角拧坏
        if (
            before_hough is not None
            and after_hough is not None
            and abs(after_hough) > abs(before_hough) + 0.2
        ):
            return None
        return rotated

    def _deskew_via_hough_rotate(
        self,
        img: np.ndarray,
        *,
        angle: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        用近水平线段中位数角度做整图旋转纠偏。
        测得倾角 θ 时按 θ 拉平；纠后残角须改善，测不到残角也丢弃（宁可不转）。
        """
        if angle is None:
            angle = self._estimate_hough_median_angle(img)
        if angle is None:
            return None
        if abs(angle) < self.DESKEW_MIN_ANGLE_DEG:
            return None
        if abs(angle) > self.DESKEW_MAX_ANGLE_DEG:
            return None
        before = self._estimate_hough_median_angle(img)
        rotated = self._rotate_image_by_angle(img, float(angle))
        if rotated is None:
            return None
        after = self._estimate_hough_median_angle(rotated)
        if before is None or after is None:
            return None
        if abs(after) > abs(before) + 0.05:
            return None
        if (
            abs(after) > abs(before) - 0.15
            and abs(after) >= self.DESKEW_MIN_ANGLE_DEG
        ):
            return None
        return rotated

    def _deskew_table_image(self, img: np.ndarray) -> np.ndarray:
        """
        表格裁剪后、OCR 前纠偏（优先几何拉正，少靠事后拼行）。

        顺序：浅蓝表体透视 → 蓝底外接矩形旋转 → 霍夫整图旋转。
        透视须过严格几何门槛（假梯形宁可不拉）；旋转须蓝底+霍夫双验收。
        失败则保持原图（宁可不转，不可拧歪）。
        """
        if img is None or img.size == 0:
            return img
        angle = None
        try:
            angle = self._estimate_hough_median_angle(img)
        except Exception:
            angle = None
        # 1) 透视优先（与霍夫角解耦）
        try:
            warped = self._deskew_via_blue_quad(img, hough_angle=angle)
            if warped is not None and warped.size > 0:
                return warped
        except Exception:
            pass
        # 2) 蓝底外接矩形旋转（右屏斜拍主路径）
        try:
            rected = self._deskew_via_blue_min_rect(img)
            if rected is not None and rected.size > 0:
                return rected
        except Exception:
            pass
        # 3) 霍夫平面旋转：角过大更像误检；小角须验收改善
        if angle is not None and abs(angle) > self.DESKEW_MAX_ANGLE_DEG:
            return img
        try:
            rotated = self._deskew_via_hough_rotate(img, angle=angle)
            if rotated is not None and rotated.size > 0:
                return rotated
        except Exception:
            pass
        return img

    def extract(self, image_path: str,
                crop_ratio: Optional[Tuple[float, float, float, float]] = None) -> TableExtractResult:
        """
        从图片提取表格（整表 OCR + 列对齐 + 零行补全）。

        @param image_path 图片路径
        @param crop_ratio 裁剪区域 (x0,y0,x1,y1) 比例 0~1
        @return 提取结果
        """
        img_path = self.resolve_image_path(image_path)
        img = self.read_image_file(str(img_path))
        if crop_ratio is not None:
            h, w = img.shape[:2]
            x0, y0, x1, y1 = crop_ratio
            img = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        else:
            # 实际送入 OCR 的框会写入 last_auto_crop_ratio（含回退到默认表区）
            img = self._resolve_table_auto_crop(img)

        # fixed / auto 裁完后共用纠偏（斜拍时固定框同样必须先拉正再 OCR）
        img = self._deskew_table_image(img)
        img = self._scale_image_for_ocr(img)

        cells = self._ocr_merge_multipass(img)
        rows = self._cluster_rows(cells)
        if not cells or not rows:
            return TableExtractResult(
                headers=[],
                rows=[],
                raw_row_count=0,
                image_path=str(img_path),
            )

        img_w = int(img.shape[1]) if img is not None else 0
        header_idx, headers, col_centers = self._extract_header_info(
            rows, img_width=img_w, img=img
        )
        data_idx, data_cells = self._find_main_data_row(rows, header_idx)
        # 拍屏时同一数据行常被拆成多行，合并后再对齐
        data_cells = self._merge_fragmented_data_rows(rows, data_idx, header_idx)
        numeric_tokens = sum(
            self._count_numeric_tokens(c.text) for c in self._expand_row_cells(data_cells)
        )

        # 表头不足时：仅从全图再收集已识别时间，绝不按上午/晚间模板编造
        if len(headers) < 4:
            if numeric_tokens < 6:
                return TableExtractResult(
                    headers=[],
                    rows=[],
                    raw_row_count=len(rows),
                    image_path=str(img_path),
                )
            headers, col_centers = self._infer_headers_fallback(
                rows, data_cells, img_w, img=img
            )
            if len(headers) < 4:
                return TableExtractResult(
                    headers=[],
                    rows=[],
                    raw_row_count=len(rows),
                    image_path=str(img_path),
                )
            header_idx = -1

        # 仅格式统一；补洞/外推以 OCR 锚段为准，左右缺列按数据 X 跨度推导
        headers = self._refine_header_times(headers)
        headers, col_centers = self._sanitize_header_sequence(headers, col_centers)
        headers, col_centers = self._complete_time_headers(headers, col_centers)
        # 宁缺毋滥：不再按数据 X 跨度外推缺列（半截列硬认危害大）
        # 仅丢掉本图边缘被裁断的半列（双屏分界后不再按大半列防邻屏）
        headers, col_centers = self._drop_edge_clipped_columns(
            headers, col_centers, img.shape[1]
        )
        headers, col_centers = self._sanitize_header_sequence(headers, col_centers)
        headers, col_centers = self._drop_edge_clipped_columns(
            headers, col_centers, img.shape[1]
        )
        col_count = len(headers)

        matrix = self._build_full_matrix(
            rows, header_idx, data_idx, headers, col_centers, col_count)
        matrix = self._postprocess_rows(matrix)
        # 用合并后的主数据行按 X 对齐（不顺序压列），覆盖末行拆碎错位
        # 仅当主行确实像当前末行（或末行过稀）时覆盖，避免把第 2 行指标盖成第 3 行
        if matrix and data_cells:
            aligned = self._align_row_to_columns(
                self._expand_row_cells(data_cells), col_centers, len(headers)
            )
            if aligned and self._should_overwrite_last_with_main(matrix, aligned):
                matrix[-1] = [
                    self.NODATA_MARK
                    if v == self.NODATA_MARK
                    else (self._fix_decimal_ocr(v) if v else "")
                    for v in aligned
                ]
        # 先按 X 回填空格，空格再局部 OCR；识别结果原样保留（异常值不置空）
        matrix = self._fill_sparse_data_row(matrix, data_cells, col_centers, col_count)
        # 先清表头时刻泄漏（23:00→23.00），再判是否整表未出清
        matrix = self._scrub_header_time_leaks(headers, matrix)
        # 初看全空时仍先做一轮列补识：真实「全 0 + 电价」常被整图漏检，
        # 若直接当未出清跳过，会把可见数据整表置空。补识后再判定。
        looked_empty = self._matrix_looks_all_uncleared(matrix, col_count)
        matrix = self._refill_matrix_empties(
            img, matrix, rows, header_idx, data_idx, data_cells, col_centers
        )
        matrix = self._scrub_header_time_leaks(headers, matrix)
        if self._matrix_looks_all_uncleared(matrix, col_count):
            # 补识后仍无真实功率/电价：视为屏上未出清（全 -），禁止再粘连/电价重识
            # （否则易把表头读成 22.30）；保留表头时刻，数据置空不写 PI
            matrix = [[""] * col_count for _ in range(self.EXPECTED_DATA_ROWS)]
        else:
            # 引擎宽框粘连：不按位数猜拆，对被盖住的列做单列裁剪 OCR
            matrix = self._reocr_geometry_sticky_cells(
                img, matrix, rows, header_idx, data_idx, data_cells, col_centers
            )
            # 电价行可疑已填格（漏点/串位）按列强制重识，可覆盖错值
            matrix = self._reocr_suspicious_price_cells(
                img, matrix, rows, header_idx, data_idx, data_cells, col_centers
            )
            if looked_empty:
                # 补识前全空、补识后有值：再 scrub 一次表头泄漏
                matrix = self._scrub_header_time_leaks(headers, matrix)
        ocr_filled = self._count_numeric_cells(matrix)
        headers, matrix = self._trim_trailing_empty_and_broken_headers(
            headers, matrix
        )
        # 斜拍/碎片兜底：数据行硬封顶为 3；不足则按电价位补零行槽
        matrix = self._enforce_expected_data_rows(
            matrix, expected=self.EXPECTED_DATA_ROWS
        )
        # 电力/计划（整型）与电价（小数）混行时拆开
        matrix = self._untangle_integer_decimal_rows(matrix)
        # 表头时刻泄漏进数据格（22:30→22.30）清掉，再判未出清
        matrix = self._scrub_header_time_leaks(headers, matrix)
        # 前两行常为全 0：整图检测极易漏检；表头几何分带 + 按列小图 OCR / 墨点 / 业务先验补全
        # （不依赖左侧行名——横向滚动后行名可能不可见）
        # 整表/末段未出清时内部会跳过补 0
        matrix = self._recover_leading_zero_data_rows(
            img,
            matrix,
            col_centers,
            rows=rows,
            header_idx=header_idx,
            data_cells=data_cells,
            headers=headers,
        )
        # 漏点补救：5 位、4 位都按同行锚点双候选选优（6 位已在 _fix_decimal_ocr）
        if self.fix_decimal_enabled:
            matrix = [self._apply_decimal_row_repairs(r) for r in matrix]
        matrix = self._scrub_header_time_leaks(headers, matrix)
        matrix = self._force_uncleared_columns_empty(headers, matrix)
        matrix = self._strip_nodata_marks(matrix)
        ocr_filled = self._count_numeric_cells(matrix)

        return TableExtractResult(
            headers=headers,
            rows=matrix,
            raw_row_count=len(rows),
            image_path=str(img_path),
            ocr_filled=ocr_filled,
        )

    def export(self, result: TableExtractResult, output_path: str,
               fmt: str = "csv") -> str:
        """
        导出为 CSV 或 TXT。

        @param result      提取结果
        @param output_path 输出路径
        @param fmt         csv 或 txt
        @return 绝对路径
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fmt = fmt.lower()
        if fmt == "csv":
            self._write_csv(result, out)
        elif fmt == "txt":
            self._write_txt(result, out)
        else:
            raise ValueError("fmt 仅支持 csv 或 txt")
        return str(out.resolve())

    def _resolve_table_auto_crop(self, img: np.ndarray) -> np.ndarray:
        """
        选定 table_auto 实际裁剪，并保证 last_auto_crop_ratio 与结果一致。

        auto 过高/过矮时改用默认表区；此前若只写 auto 框、却送 fallback 图，
        竞选面积会偏小（图看着大、分却像很紧）。
        """
        auto = self._auto_crop_table(img)
        fallback_ratio = self.DEFAULT_TABLE_CROP_RATIO
        fallback = self._default_table_crop(img)
        auto_h = float(auto.shape[0]) / max(float(img.shape[0]), 1.0)
        # 表体不应超过半屏；过高说明仍吃进曲线
        if auto_h > 0.55:
            self.last_auto_crop_ratio = fallback_ratio
            return fallback
        if auto.shape[0] >= fallback.shape[0] * 0.75:
            # last_auto_crop_ratio 已由 _auto_crop_table 写入
            return auto
        self.last_auto_crop_ratio = fallback_ratio
        return fallback

    def _auto_crop_table(self, img: np.ndarray) -> np.ndarray:
        """
        自动裁剪表格区域（浅蓝底行 + 仅下半屏搜索）。

        现货 UI：上下均分后表一定在下半；蓝区搜索从 50% 高度起，上扩只留约 1 行
        白底时刻，禁止吃进上半曲线（否则 cand_auto 像整屏）。
        """
        h, w = img.shape[:2]
        fallback_box = self.DEFAULT_TABLE_CROP_RATIO
        # 只在下半屏找表体，彻底避开上半曲线/坐标轴
        ignore_top = int(h * 0.50)
        mask = self._table_body_blue_mask(img, ignore_top=ignore_top)
        # 去掉碎噪：保留较大且有一定宽度的连通域（表行接近全宽）
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        mask2 = np.zeros_like(mask)
        min_area = max(int(w * h * 0.003), 600)
        min_w = max(int(w * 0.18), 40)
        for i in range(1, nlab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            if area >= min_area and ww >= min_w:
                mask2[labels == i] = 255
        coords = cv2.findNonZero(mask2)
        if coords is None:
            self.last_auto_crop_ratio = fallback_box
            x0, y0, x1, y1 = (
                int(w * fallback_box[0]),
                int(h * fallback_box[1]),
                int(w * fallback_box[2]),
                int(h * fallback_box[3]),
            )
            return img[y0:y1, x0:x1]

        x, y, bw, bh = cv2.boundingRect(coords)
        # 用行密度再收紧上下界：ignore_top 边界易残留碎蓝，勿把白缝/曲线底算进表体
        row_nz = np.count_nonzero(mask2, axis=1)
        dense_thr = max(int(w * 0.12), 30)
        dense_rows = np.where(row_nz >= dense_thr)[0]
        if dense_rows.size >= 3:
            # 取下半屏内最长连续密行带（真正浅蓝数据行）
            breaks = np.where(np.diff(dense_rows) > 2)[0]
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [dense_rows.size - 1]))
            best_i = int(np.argmax(ends - starts))
            y = int(dense_rows[starts[best_i]])
            y_bot = int(dense_rows[ends[best_i]])
            bh = max(y_bot - y + 1, bh // 2)
            # x 范围仍用整 mask，避免暗侧漏列
            xs = np.where(mask2[y : y + bh, :].any(axis=0))[0]
            if xs.size:
                x = int(xs[0])
                bw = int(xs[-1] - xs[0] + 1)
        # 仅上扩约 1 行表头（白/灰底时刻），绝不 pad 到图顶/上半屏
        row_est = max(bh / 3.5, h * 0.028)
        pad_y_top = int(min(row_est * 1.35, h * 0.06))
        pad_y_bot = max(6, int(bh * 0.08), int(h * 0.015))
        pad_x = max(6, int(bw * 0.015), int(w * 0.004))
        # 左缘常因首列偏暗/半截漏检被啃掉一列；左侧多扩约 1 列宽
        col_est = max(bw / 8.0, w * 0.09)
        pad_x_left = max(pad_x, int(col_est * 0.95), int(w * 0.04))
        pad_x_right = pad_x
        x0 = max(0, x - pad_x_left)
        # 搜索从 50% 起；表头可略越过，但不得低于 48%（避免曲线）
        y0 = max(int(h * 0.48), y - pad_y_top)
        x1 = min(w, x + bw + pad_x_right)
        y1 = min(h, y + bh + pad_y_bot)
        # 蓝带起点仍偏右（>4%）：再向左拉一列，避免首列时刻/电价被裁断
        if x0 > int(w * 0.04):
            x0 = max(0, int(x0 - col_est * 0.9))
        # 表通常接近全宽；偏窄时左右略扩（暗侧漏检时尤其需要）
        if (x1 - x0) < w * 0.70:
            mid = 0.5 * (x0 + x1)
            half = 0.45 * w
            x0 = max(0, int(mid - half))
            x1 = min(w, int(mid + half))
        # 高度上限：表体一般不超过半屏；过高则从上方压回
        max_h = int(h * 0.48)
        if (y1 - y0) > max_h:
            y0 = max(int(h * 0.48), y1 - max_h)
        # 仍偏上或过高 → 回退默认表区
        if y0 < h * 0.45 or (y1 - y0) > h * 0.52:
            self.last_auto_crop_ratio = fallback_box
            return img[
                int(h * fallback_box[1]) : int(h * fallback_box[3]),
                int(w * fallback_box[0]) : int(w * fallback_box[2]),
            ]
        cropped = img[y0:y1, x0:x1]
        if cropped.size == 0:
            self.last_auto_crop_ratio = fallback_box
            return img[
                int(h * fallback_box[1]) : int(h * fallback_box[3]),
                int(w * fallback_box[0]) : int(w * fallback_box[2]),
            ]
        # 去掉底部任务栏/黑边：自下而上裁掉过暗条带
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape[:2]
        cut = ch
        step = max(2, ch // 80)
        for yy in range(ch - 1, max(int(ch * 0.55), 0), -step):
            if float(np.mean(gray[max(0, yy - step) : yy, :])) < 55.0:
                cut = yy - step
            else:
                break
        if cut < ch and cut > int(ch * 0.55):
            cropped = cropped[:cut, :]
            y1 = y0 + cut
        self.last_auto_crop_ratio = (
            x0 / float(w),
            y0 / float(h),
            x1 / float(w),
            y1 / float(h),
        )
        return cropped

    def _default_table_crop(self, img: np.ndarray) -> np.ndarray:
        """
        默认表格裁剪比例（全屏上图下表；略抬高以含白底时刻行）。
        """
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.DEFAULT_TABLE_CROP_RATIO
        return img[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]

    def _demoire_table_gray(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        表格轻量去摩尔纹：双边 + 频域低通，削弱拍屏网格竖纹。
        只产一路变体，避免像日期 OCR 那样多通道拖慢整表。
        """
        try:
            soft = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
            f = np.fft.fft2(soft.astype(np.float32))
            fshift = np.fft.fftshift(f)
            rows, cols = soft.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            radius = max(14, min(rows, cols) // 7)
            cv2.circle(mask, (ccol, crow), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, radius / 6.0))
            img_back = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
            low = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(low)
        except Exception:
            return None

    def _preprocess_variants(self, img: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        生成预处理图像，尽量保住小数点笔画。

        通道：原图 → CLAHE/反色 →（浅色表）轻度放大锐化 → 轻量去摩尔纹。
        返回 (图像, 相对原图的尺度)；放大通道 OCR 后须除以该尺度映射回原图坐标。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_v = float(np.mean(gray))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        if mean_v < 110:
            inv = clahe.apply(cv2.bitwise_not(gray))
            return [(img, 1.0), (cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR), 1.0)]
        soft = cv2.bilateralFilter(gray, 5, 40, 40)
        enhanced = clahe.apply(soft)
        out: List[Tuple[np.ndarray, float]] = [
            (img, 1.0),
            (cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), 1.0),
        ]
        # 浅色密表：再放大+轻锐化，提高小数点命中；大图已够清晰时跳过（省 1 遍全图 OCR）
        h, w = enhanced.shape[:2]
        if max(h, w) < 2200:
            up_scale = 1.6
            up = cv2.resize(
                enhanced, None, fx=up_scale, fy=up_scale, interpolation=cv2.INTER_CUBIC
            )
            sharp = cv2.addWeighted(
                up, 1.35, cv2.GaussianBlur(up, (0, 0), 0.8), -0.35, 0
            )
            out.append((cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), up_scale))
        # 拍屏竖纹：多一路去摩尔纹，利于细小 0 / 小数点
        demoire = self._demoire_table_gray(gray)
        if demoire is not None and demoire.size:
            out.append((cv2.cvtColor(demoire, cv2.COLOR_GRAY2BGR), 1.0))
        return out

    @staticmethod
    def _scale_ocr_cells(cells: List[OcrCell], scale: float) -> List[OcrCell]:
        """将 OCR 框坐标从变体图映射回基准图（scale=变体相对基准的放大倍数）。"""
        if abs(scale - 1.0) < 1e-6 or scale <= 0:
            return cells
        inv = 1.0 / float(scale)
        return [
            OcrCell(
                text=c.text,
                confidence=c.confidence,
                x=c.x * inv,
                y=c.y * inv,
                width=c.width * inv,
                height=c.height * inv,
            )
            for c in cells
        ]

    def _ocr_merge_multipass(self, img: np.ndarray) -> List[OcrCell]:
        """
        多通道 OCR 并合并去重。

        原图 + 增强通道都跑；同位置优先保留 OCR 原文带小数点的结果。
        放大通道的检测框映射回原图坐标后再合并，避免右侧 X 虚高触发贴边误删。
        """
        variants = self._preprocess_variants(img)
        merged: List[OcrCell] = []
        for variant, scale in variants:
            cells = self._ocr_image(variant)
            merged.extend(self._scale_ocr_cells(cells, scale))
        return self._dedupe_cells(merged)

    def _time_to_minutes(self, t: str) -> Optional[int]:
        try:
            h, m = t.split(":")
            hh, mm = int(h), int(m)
            if hh == 24 and mm == 0:
                return 24 * 60
            if mm >= 60 or hh > 24:
                return None
            return hh * 60 + mm
        except ValueError:
            return None

    def _snap_minutes_to_quarter(self, mins: int, *, max_delta: int = 2) -> int:
        """
        表头时刻对齐到 15 分钟刻度。
        OCR 常把 15:15 读成 15:16、16:45 读成 16:46；偏差 ≤max_delta 分钟则吸附。
        """
        if mins == 24 * 60:
            return mins
        slot = int(round(mins / 15.0)) * 15
        slot = max(0, min(24 * 60, slot))
        if abs(slot - mins) <= max_delta:
            return slot
        return mins

    def _minutes_to_time(self, m: int) -> str:
        if m >= 24 * 60:
            return "24:00"
        m = max(0, m)
        return "%02d:%02d" % (m // 60, m % 60)

    def _header_step_score(self, mins: List[int]) -> float:
        """评估时刻序列是否像规则 15 分钟表头（越高越好）。

        曲线 X 轴常见整点/整刻度（Δ≈60）；表头以 Δ≈15 为主。
        允许 14~16 的 OCR 近邻差（吸附前）。
        """
        if len(mins) < 3:
            return -1.0
        diffs = [b - a for a, b in zip(mins, mins[1:]) if b > a]
        if not diffs:
            return -1.0
        exact15 = sum(1 for d in diffs if d == 15)
        near15 = sum(1 for d in diffs if 14 <= d <= 16)
        step15_mult = sum(1 for d in diffs if d % 15 == 0)
        # 惩罚偏小时轴的大步（45/60）：这些更像曲线刻度
        big = sum(1 for d in diffs if d >= 45)
        mono = sum(1 for d in diffs if d > 0)
        return (
            float(exact15) * 2.0
            + float(near15 - exact15) * 1.2
            + 0.25 * float(step15_mult)
            + 0.05 * float(len(mins))
            + 0.15 * float(mono)
            - 1.5 * float(big)
        )

    @staticmethod
    def _raw_time_hour_digit_count(text: str) -> Optional[int]:
        """OCR 原文小时位数；无法解析为时刻则 None。

        界面完整列为零填充 HH:MM（如 01:15）。贴边裁断时常只剩 1:15，
        小时仅 1 位——这类残片不得当成完整列。
        """
        raw = str(text or "").strip().replace("O", "0").replace("o", "0")
        raw = raw.replace("l", "1").replace("I", "1").replace("|", "1")
        raw = raw.replace("：", ":").replace("．", ".").replace(";", ":")
        m = re.match(r"^(\d{1,2})[:：.\-](\d{2})(?:[:：]\d{2})?$", raw)
        if m:
            return len(m.group(1))
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 4:
            return 2
        if len(digits) == 3:
            # 115 → 可能是残缺的 1:15（缺分隔）
            return 1
        return None

    @staticmethod
    def _cell_touches_image_edge(
        cell: OcrCell, img_width: int, *, pad_px: Optional[float] = None
    ) -> bool:
        """文本框是否贴齐画布左/右裁切线（半列被裁的典型信号）。"""
        if img_width <= 0:
            return False
        pad = (
            float(pad_px)
            if pad_px is not None
            else max(4.0, min(float(img_width) * 0.015, 14.0))
        )
        left = float(cell.x)
        right = left + float(cell.width or 0)
        return left <= pad or right >= float(img_width) - pad

    def _glyph_run_widths(self, img: np.ndarray, cell: OcrCell) -> List[int]:
        """单元格内横向墨迹段宽度（每个可见字符一段）。"""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        x0 = max(0, int(math.floor(float(cell.x))))
        y0 = max(0, int(math.floor(float(cell.y))))
        x1 = min(w, int(math.ceil(float(cell.x) + float(cell.width or 0))))
        y1 = min(h, int(math.ceil(float(cell.y) + float(cell.height or 0))))
        if x1 - x0 < 8 or y1 - y0 < 6:
            return []
        crop = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        bg = cv2.GaussianBlur(gray, (0, 0), 2)
        ink = (bg.astype(np.int16) - gray.astype(np.int16)) > 14
        occ = ink.sum(axis=0)
        thr = max(2, int(0.18 * ink.shape[0]))
        xs = np.where(occ >= thr)[0]
        if len(xs) == 0:
            return []
        widths: List[int] = []
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x - prev > 2:
                widths.append(prev - start + 1)
                start = x
            prev = x
        widths.append(prev - start + 1)
        return widths

    def _leading_glyph_clipped(
        self,
        img: np.ndarray,
        cell: OcrCell,
        others: Sequence[OcrCell],
    ) -> bool:
        """最左时刻的首位笔画是否明显细于其余列的首位（裁掉半个 1）。

        正常列的「1」宽度接近；左缘只剩一条细缝时，OCR 仍可能读成完整 10:15。
        只跟其它列的首位笔画比，避免被 0/5 等宽字拉高中位数。
        """
        if img is None or not others:
            return False
        own = self._glyph_run_widths(img, cell)
        if not own:
            return False
        leads: List[int] = []
        for other in others:
            runs = self._glyph_run_widths(img, other)
            if runs:
                leads.append(int(runs[0]))
        if len(leads) < 3:
            return False
        med = float(np.median(leads))
        if med < 3.0:
            return False
        lead = float(own[0])
        return lead + 1.5 < med and lead < med * 0.65

    def _row_time_candidates(
        self,
        row: Sequence[OcrCell],
        *,
        img_width: int = 0,
        img: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """收集行内时刻；小时非两位、贴齐裁切线、或最左列首位被裁的残片一律丢弃。

        现货表头固定零填充 HH:MM（04:00）。裁切半列时常读成 4:00，
        若仅在「贴边」时才丢，框稍离边缘仍会被 _normalize_time 补成 04:00 误用。
        另：首位只裁一丁点时框并不贴画布，OCR 仍可能读成完整 10:15——
        仅最左列、且首位墨迹明显细于其余列时不采纳。
        """
        time_cells = [
            c
            for c in row
            if self._raw_time_hour_digit_count(c.text) is not None
            or self._normalize_time(c.text)
        ]
        leftmost = min(time_cells, key=lambda c: c.x) if time_cells else None
        drop_left = False
        if leftmost is not None and img is not None:
            drop_left = self._leading_glyph_clipped(
                img, leftmost, [c for c in time_cells if c is not leftmost]
            )
        times: List[Tuple[str, float]] = []
        for cell in row:
            hour_digits = self._raw_time_hour_digit_count(cell.text)
            touches = self._cell_touches_image_edge(cell, img_width)
            # 小时非两位（4:00 / 1:15 / 115）→ 识别不全，整列不采用
            if hour_digits is not None and hour_digits < 2:
                continue
            # 贴齐裁切线：半截仍读成两位也丢
            if touches and hour_digits is not None:
                continue
            if drop_left and cell is leftmost:
                continue
            norm = self._normalize_time(cell.text)
            if norm:
                times.append((norm, cell.x + cell.width / 2))
        return times

    def _extract_header_info(
        self,
        rows: Sequence[List[OcrCell]],
        *,
        img_width: int = 0,
        img: Optional[np.ndarray] = None,
    ) -> Tuple[int, List[str], List[float]]:
        """
        提取表头行、时间列名、列中心 X。

        优先「严格 15 分钟连续」且偏下方的行（白底时刻表头），
        避免上半曲线 X 轴刻度（常为整点/大步）污染。
        """
        best_idx = -1
        # (step_score, count+ratio, y_center, idx)
        best_key: Tuple[float, float, float, int] = (-1e9, -1.0, -1.0, -1)
        best_times: List[Tuple[str, float]] = []
        for idx, row in enumerate(rows):
            times = self._row_time_candidates(row, img_width=img_width, img=img)
            if len(times) < 2:
                continue
            ratio = len(times) / max(len(row), 1)
            if len(times) < 3 and ratio < 0.45:
                continue
            mins: List[int] = []
            for t, _x in sorted(times, key=lambda item: item[1]):
                m = self._time_to_minutes(t)
                if m is not None:
                    mins.append(m)
            if len(mins) >= 3:
                diffs = [b - a for a, b in zip(mins, mins[1:]) if b > a]
                if diffs:
                    near15 = sum(1 for d in diffs if 14 <= d <= 16)
                    if near15 == 0 and sum(1 for d in diffs if d >= 45) >= max(
                        1, len(diffs) // 2
                    ):
                        continue
            # 评分用吸附后的分钟，避免 15:16/16:46 被当非表头
            mins_snap = [self._snap_minutes_to_quarter(m) for m in mins]
            step = self._header_step_score(mins_snap)
            y_c = self._row_y_center(row)
            key = (step, float(len(times)) + 2.0 * ratio, y_c, idx)
            if key > best_key:
                best_key = key
                best_idx = idx
                best_times = times

        # 合并 Y 很近的邻行时刻：斜拍时左/右缘表头常被拆到邻行（如 09:30），
        # 旧逻辑仅在单行 <4 列时合并，主行已够长时会丢贴边列。
        # y_tol 仍卡死曲线轴（通常远在上方）。
        if best_idx >= 0 and best_times:
            base_y = self._row_y_center(rows[best_idx])
            y_tol = 28.0
            if rows[best_idx]:
                y_tol = max(
                    28.0,
                    float(np.median([c.height for c in rows[best_idx]])) * 1.6,
                )
            merged: List[Tuple[str, float]] = list(best_times)
            for idx, row in enumerate(rows):
                if idx == best_idx:
                    continue
                if abs(self._row_y_center(row) - base_y) > y_tol:
                    continue
                # 邻行若几乎全是大步时刻（曲线轴），勿并入
                neigh = self._row_time_candidates(row, img_width=img_width, img=img)
                if not neigh:
                    continue
                neigh_mins = [
                    m
                    for m in (self._time_to_minutes(t) for t, _ in neigh)
                    if m is not None
                ]
                if len(neigh_mins) >= 3:
                    diffs = [b - a for a, b in zip(sorted(neigh_mins), sorted(neigh_mins)[1:]) if b > a]
                    near15 = sum(1 for d in diffs if 14 <= d <= 16)
                    big = sum(1 for d in diffs if d >= 45)
                    if near15 == 0 and big >= max(1, len(diffs) // 2):
                        continue
                merged.extend(neigh)
            best_times = merged

        if len(best_times) < 4:
            return -1, [], []
        best_times.sort(key=lambda t: t[1])  # 严格按列 X
        # 同 X 去重：多读时优先续上 Δ=15（修 05:45↔06:45 类互斥误读）
        dedup2 = self._dedupe_header_times_prefer_delta15(best_times)
        headers = [t[0] for t in dedup2]
        col_centers = [t[1] for t in dedup2]
        headers, col_centers = self._sanitize_header_sequence(headers, col_centers)
        if len(headers) < 4:
            return -1, [], []
        return best_idx, headers, col_centers

    def _dedupe_header_times_prefer_delta15(
        self,
        times: Sequence[Tuple[str, float]],
        *,
        x_tol: float = 22.0,
    ) -> List[Tuple[str, float]]:
        """
        按 X 聚列并去重。同列多读（如 05:45 与 06:45）时：
        优先能与左侧已定列构成严格 Δ=15 的读法；否则再看右侧邻列提示。
        """
        if not times:
            return []
        ordered = sorted(times, key=lambda item: item[1])
        groups: List[List[Tuple[str, float]]] = []
        for t, x in ordered:
            if groups and abs(x - groups[-1][-1][1]) < x_tol:
                groups[-1].append((t, x))
            else:
                groups.append([(t, x)])

        resolved: List[Tuple[str, float]] = []
        for gi, group in enumerate(groups):
            # 组内按 (label -> 平均 x)
            by_lab: Dict[str, List[float]] = {}
            for t, x in group:
                by_lab.setdefault(t, []).append(float(x))
            options: List[Tuple[str, float, Optional[int]]] = []
            for lab, xs in by_lab.items():
                options.append((lab, float(np.mean(xs)), self._time_to_minutes(lab)))
            if len(options) == 1:
                resolved.append((options[0][0], options[0][1]))
                continue

            prev_m: Optional[int] = None
            if resolved:
                prev_m = self._time_to_minutes(resolved[-1][0])

            next_opts: List[Optional[int]] = []
            if gi + 1 < len(groups):
                for t2, _x2 in groups[gi + 1]:
                    next_opts.append(self._time_to_minutes(t2))

            def _score(opt: Tuple[str, float, Optional[int]]) -> Tuple[int, int, int]:
                lab, _x, m = opt
                left_fit = 0
                if prev_m is not None and m is not None and m - prev_m == 15:
                    left_fit = 2
                right_fit = 0
                if m is not None:
                    for nm in next_opts:
                        if nm is not None and nm - m == 15:
                            right_fit = 1
                            break
                # 同读次数字多者优先（弱 tie-break）
                return (left_fit, right_fit, len(by_lab.get(lab, [])))

            best_opt = max(options, key=_score)
            resolved.append((best_opt[0], best_opt[1]))
        return resolved

    def _longest_delta15_header_indices(
        self,
        headers: Sequence[str],
        col_centers: Optional[Sequence[float]] = None,
    ) -> List[int]:
        """兼容旧名：现委托 `_keep_delta15_header_indices`（可保留多段）。"""
        return self._keep_delta15_header_indices(headers, col_centers)

    def _median_header_col_gap(
        self,
        xs: Sequence[float],
        mins: Sequence[Optional[int]],
    ) -> float:
        """用已是 Δ=15 的相邻列估计列距；不足则退回全部正间距中位数。"""
        step_gaps: List[float] = []
        all_gaps: List[float] = []
        for i in range(len(xs) - 1):
            dx = float(xs[i + 1]) - float(xs[i])
            if dx <= 0:
                continue
            all_gaps.append(dx)
            mi, mj = mins[i], mins[i + 1]
            if mi is not None and mj is not None and mj - mi == 15:
                step_gaps.append(dx)
        if len(step_gaps) >= 2:
            return float(np.median(step_gaps))
        if all_gaps:
            return float(np.median(all_gaps))
        return 80.0

    def _can_bridge_header_gap(
        self,
        m_left: int,
        m_right: int,
        x_left: float,
        x_right: float,
        med_gap: float,
        *,
        max_slots: int = 4,
    ) -> bool:
        """
        两段 Δ=15 链之间是否像「中间漏了若干格 / 夹了脏列」而非无关簇。

        slots = dm/15：2 表示中间缺 1 个刻度（如 05:30→06:00）；不编造缺刻度。
        """
        dm = int(m_right) - int(m_left)
        if dm <= 15 or dm % 15 != 0:
            return False
        slots = dm // 15
        if slots < 2 or slots > max_slots:
            return False
        dx = float(x_right) - float(x_left)
        if dx <= 0:
            return False
        if med_gap <= 8:
            return True
        expected = float(slots) * float(med_gap)
        return 0.45 * expected <= dx <= 1.7 * expected

    def _keep_delta15_header_indices(
        self,
        headers: Sequence[str],
        col_centers: Optional[Sequence[float]] = None,
    ) -> List[int]:
        """
        保留可并入同一表头的严格 Δ=15 列（按 X 序）。

        - 丢掉接不上的脏列（如误读成 06:45 插在 05:30 与 06:00 之间）
        - 若中间仅因误读断开，两端链在列距像「漏格」时**都保留**，不编造中间时刻
        - 多段无法桥接时，保留列数最多的那一组（避免拼进曲线轴等无关簇）
        """
        n = len(headers)
        if n <= 0:
            return []
        order = list(range(n))
        if col_centers is not None and len(col_centers) >= n:
            order.sort(key=lambda i: float(col_centers[i]))
        xs = [
            float(col_centers[i]) if col_centers is not None and i < len(col_centers) else float(i)
            for i in order
        ]
        mins: List[Optional[int]] = [self._time_to_minutes(headers[i]) for i in order]
        med = self._median_header_col_gap(xs, mins)

        # 1) 切成极大严格 Δ=15 段；单独接不上的点当作 orphan（稍后丢弃）
        runs: List[List[int]] = []  # 存 order 内下标
        cur: List[int] = []
        for pos in range(n):
            mi = mins[pos]
            if mi is None:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
                continue
            if not cur:
                cur = [pos]
                continue
            prev = cur[-1]
            mj = mins[prev]
            if mj is not None and mi - mj == 15:
                cur.append(pos)
                continue
            # 断开：当前段落盘，本点另起
            if len(cur) >= 2:
                runs.append(cur)
            elif len(cur) == 1:
                # 单点不成链，不进入 runs（视为脏/孤立）
                pass
            cur = [pos]
        if len(cur) >= 2:
            runs.append(cur)

        if not runs:
            # 构不成段：退回全部合法刻度（与旧行为接近）
            return [order[i] for i in range(n) if mins[i] is not None]

        # 2) 按 X 序尝试桥接相邻段（中间 orphan 自然被跳过）
        groups: List[List[int]] = [list(runs[0])]
        for run in runs[1:]:
            prev = groups[-1]
            left, right = prev[-1], run[0]
            m_l, m_r = mins[left], mins[right]
            if (
                m_l is not None
                and m_r is not None
                and self._can_bridge_header_gap(m_l, m_r, xs[left], xs[right], med)
            ):
                groups[-1].extend(run)
            else:
                groups.append(list(run))

        # 3) 取列数最多的一组（同分偏更靠右/更长连续时已由桥接合并）
        best = max(groups, key=lambda g: (len(g), g[-1] - g[0] if g else 0))
        if len(best) < 2:
            return [order[i] for i in range(n) if mins[i] is not None]
        return [order[p] for p in best]

    def _sanitize_header_sequence(
        self,
        headers: List[str],
        col_centers: List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        清洗表头时刻序列（仍按 OCR 原文，不发明时刻）：

        1. 丢掉左侧邻屏漏进来的乱序头（如 03:15 出现在 00:15 之前）
        2. 丢掉与主簇 X 间距过大的左缘孤列
        3. 保留可桥接的严格 Δ=15 段（中间脏列丢弃；两端不断开只因误读时都保留）
        """
        n = min(len(headers), len(col_centers))
        if n < 2:
            return list(headers[:n]), list(col_centers[:n])

        paired: List[Tuple[float, str, Optional[int]]] = []
        for i in range(n):
            lab = headers[i]
            paired.append((col_centers[i], lab, self._time_to_minutes(lab)))
        paired.sort(key=lambda item: item[0])

        # 1) 去掉「时刻倒退」的前缀脏列（邻屏漏读）
        while len(paired) >= 2:
            m0, m1 = paired[0][2], paired[1][2]
            if m0 is not None and m1 is not None and m0 > m1:
                paired.pop(0)
                continue
            break

        # 2) 左缘孤列：首间距远大于其余列距中位数 → 丢掉
        if len(paired) >= 4:
            xs = [p[0] for p in paired]
            gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] > xs[i]]
            if gaps:
                med = float(np.median(gaps[1:] if len(gaps) > 2 else gaps))
                if med > 8 and gaps[0] > med * 2.4:
                    paired.pop(0)

        labs = [p[1] for p in paired]
        xs2 = [p[0] for p in paired]
        keep = self._keep_delta15_header_indices(labs, xs2)
        if len(keep) >= 2:
            return [labs[i] for i in keep], [xs2[i] for i in keep]
        if len(paired) < 2:
            return list(headers[:n]), list(col_centers[:n])
        return labs, xs2

    def _infer_headers_fallback(
        self,
        rows: Sequence[List[OcrCell]],
        data_cells: List[OcrCell],
        img_width: int,
        img: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], List[float]]:
        """表头行弱时：汇总偏下方、且步长像 15 分钟的时刻（不编造模板）。"""
        del data_cells
        # 优先取图像下半的行，减少曲线轴
        scored_rows: List[Tuple[float, List[Tuple[str, float]]]] = []
        max_y = 1.0
        for row in rows:
            yc = self._row_y_center(row)
            if yc > max_y:
                max_y = yc
        for row in rows:
            times = self._row_time_candidates(row, img_width=img_width, img=img)
            if len(times) < 3:
                continue
            mins = [
                m
                for m in (self._time_to_minutes(t) for t, _ in times)
                if m is not None
            ]
            step = self._header_step_score(mins)
            yc = self._row_y_center(row)
            # 下半加权
            y_boost = 0.5 if (max_y <= 0 or yc >= max_y * 0.45) else -1.0
            scored_rows.append((step + y_boost, times))
        if not scored_rows:
            times_all: List[Tuple[str, float]] = []
            for row in rows:
                times_all.extend(
                    self._row_time_candidates(row, img_width=img_width, img=img)
                )
            dedup0 = self._dedupe_header_times_prefer_delta15(times_all, x_tol=25.0)
            if len(dedup0) < 4:
                return [], []
            return self._sanitize_header_sequence(
                [t for t, _ in dedup0], [x for _, x in dedup0]
            )

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        times = scored_rows[0][1]
        times.sort(key=lambda t: t[1])
        dedup = self._dedupe_header_times_prefer_delta15(times, x_tol=25.0)
        if len(dedup) < 4:
            return [], []
        return self._sanitize_header_sequence(
            [t for t, _ in dedup], [x for _, x in dedup]
        )

    def _refine_header_times(self, headers: List[str]) -> List[str]:
        """
        整理表头时刻：格式统一 + 吸附到 15 分钟刻度（修 15:16/16:46 类 OCR 误差）。
        不做 00↔18 整体平移。
        """
        if not headers:
            return headers
        out: List[str] = []
        for t in headers:
            m = self._time_to_minutes(t)
            if m is None:
                out.append(t)
                continue
            m = self._snap_minutes_to_quarter(m)
            out.append(self._minutes_to_time(m))
        return out

    def _fix_misread_morning_headers(self, headers: List[str]) -> List[str]:
        """兼容旧名：仅格式化，不再做 18↔00 互转。"""
        return self._refine_header_times(headers)

    def _dedupe_cells(self, cells: List[OcrCell]) -> List[OcrCell]:
        """
        按位置去重：同位置优先保留带小数点、更长、置信度更高的文本。

        @param cells OCR 列表
        @return 去重后列表
        """

        def quality(c: OcrCell) -> Tuple[int, int, float]:
            has_dot = 1 if "." in c.text else 0
            return (has_dot, len(re.sub(r"\D", "", c.text)), c.confidence)

        kept: List[OcrCell] = []
        for cell in sorted(cells, key=quality, reverse=True):
            replaced = False
            for i, k in enumerate(kept):
                if abs(cell.x - k.x) < 22 and abs(cell.y - k.y) < 20:
                    same = (
                        cell.text == k.text
                        or cell.text in k.text
                        or k.text in cell.text
                        or re.sub(r"\D", "", cell.text) == re.sub(r"\D", "", k.text)
                    )
                    if same:
                        if quality(cell) > quality(k):
                            kept[i] = cell
                        replaced = True
                        break
            if not replaced:
                kept.append(cell)
        return kept

    def _ocr_image(self, img: np.ndarray) -> List[OcrCell]:
        """对图片执行 OCR。"""
        ocr = self._get_ocr()
        result, _ = ocr(img)
        cells: List[OcrCell] = []
        if not result:
            return cells
        for item in result:
            box, text, score = item[0], str(item[1]).strip(), float(item[2])
            if score < self.min_confidence or not text:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cells.append(OcrCell(
                text=text, confidence=score,
                x=min(xs), y=min(ys),
                width=max(xs) - min(xs), height=max(ys) - min(ys),
            ))
        return cells

    def _cluster_rows(
        self, cells: List[OcrCell], y_tolerance: Optional[float] = None
    ) -> List[List[OcrCell]]:
        """
        按 Y 坐标聚类为行。

        拍屏透视下同行左右格子 Y 常呈斜线漂移，且相邻指标行会在 Y 上交织。
        因此从左到右分配：用当前行已有格子拟合 y≈a·x+b，按残差归行；
        同一列已有格子则绝不再并入（防止 280 与电价叠进一行）。

        @param cells       OCR 单元格
        @param y_tolerance 同行 Y 容差；None 时按字高自适应
        @return 行列表
        """
        if not cells:
            return []
        heights = [c.height for c in cells if c.height > 0]
        med_h = float(np.median(heights)) if heights else 20.0
        # 斜线残差容差；过小会把同行拆碎，过大仍会串行
        tol = float(y_tolerance) if y_tolerance is not None else max(28.0, med_h * 1.35)
        col_tol = max(22.0, med_h * 0.9)

        def _cx(c: OcrCell) -> float:
            return float(c.x + c.width / 2)

        def _cy(c: OcrCell) -> float:
            return float(c.y + c.height / 2)

        def _predict_y(row: List[OcrCell], x: float) -> float:
            if len(row) == 1:
                return _cy(row[0])
            xs = np.array([_cx(c) for c in row], dtype=float)
            ys = np.array([_cy(c) for c in row], dtype=float)
            if float(xs.max() - xs.min()) < 1.0:
                return float(np.median(ys))
            # 两点或更多：线性拟合斜行
            a, b = np.polyfit(xs, ys, 1)
            return float(a * x + b)

        def _column_taken(row: List[OcrCell], x: float, width: float) -> bool:
            gap = max(col_tol, width * 0.45)
            return any(abs(_cx(c) - x) < gap for c in row)

        # 左→右分配，斜线预测比纯 Y 排序更抗透视
        sorted_cells = sorted(cells, key=lambda c: (_cx(c), _cy(c)))
        rows: List[List[OcrCell]] = []
        for cell in sorted_cells:
            cx = _cx(cell)
            cy = _cy(cell)
            best_i = -1
            best_d = 1e18
            for i, row in enumerate(rows):
                if _column_taken(row, cx, float(cell.width or med_h)):
                    continue
                d = abs(cy - _predict_y(row, cx))
                if d <= tol and d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                rows[best_i].append(cell)
            else:
                rows.append([cell])

        rows.sort(key=lambda r: float(np.median([_cy(c) for c in r])))
        return [sorted(r, key=lambda c: c.x) for r in rows]

    def _find_main_data_row(self, rows: Sequence[List[OcrCell]],
                            header_idx: int) -> Tuple[int, List[OcrCell]]:
        """
        定位主数据行：现货表默认取表体最后一条有效数值行（指标3）。

        @param rows      OCR 行
        @param header_idx 表头行索引，-1 表示未识别
        @return (data_idx, data_cells)
        """
        best_idx = -1
        best_key: Tuple[int, float, float] = (-1, -1.0, -1.0)
        best_cells: List[OcrCell] = []
        start = header_idx + 1 if header_idx >= 0 else 0
        for idx in range(start, len(rows)):
            row = rows[idx]
            joined = " ".join(c.text for c in row)
            if self._is_footer_text(joined):
                continue
            # 整行都是时刻表头碎片：不是数据行
            if row and all(
                self._normalize_time(c.text) or not str(c.text).strip() for c in row
            ):
                continue
            token_n = 0
            magnitude = 0.0
            for c in row:
                if self._normalize_time(c.text):
                    continue
                if self._is_footer_text(c.text):
                    continue
                for part in self._split_cell_values(c.text):
                    token_n += 1
                    try:
                        magnitude += abs(float(part))
                    except ValueError:
                        pass
            if token_n < 2:
                continue
            # 优先更靠后的行（指标3），其次 token 数、量级
            key = (idx, float(token_n), magnitude)
            if key > best_key:
                best_key = key
                best_idx = idx
                best_cells = row
        if best_idx < 0 and rows:
            # 禁止回落到页脚 / 纯时刻行——否则会拆出 133、23.00 等伪数
            for idx in range(len(rows) - 1, start - 1, -1):
                row = rows[idx]
                joined = " ".join(c.text for c in row)
                if self._is_footer_text(joined):
                    continue
                if row and all(
                    self._normalize_time(c.text) or not str(c.text).strip()
                    for c in row
                ):
                    continue
                best_idx = idx
                best_cells = list(row)
                break
            if best_idx < 0:
                return -1, []
        return best_idx, best_cells

    def _merge_fragmented_data_rows(
        self,
        rows: Sequence[List[OcrCell]],
        data_idx: int,
        header_idx: int,
    ) -> List[OcrCell]:
        """
        合并被垂直拆碎的同一数据行 OCR 块（拍屏/摩尔纹常见）。
        仅合并量级接近且 Y 很近的碎片，避免 550 行与电价行互吞。
        """
        if not rows or data_idx < 0 or data_idx >= len(rows):
            return list(rows[data_idx]) if 0 <= data_idx < len(rows) else []

        start = max(header_idx + 1, 0) if header_idx >= 0 else 0
        base = rows[data_idx]
        base_y = float(np.median([c.y + c.height / 2 for c in base])) if base else 0.0
        y_tol = max(36.0, float(np.median([c.height for c in base]) * 1.8) if base else 36.0)
        base_mag = self._row_numeric_median(base)

        merged: List[OcrCell] = []
        for idx in range(start, len(rows)):
            row = rows[idx]
            joined = " ".join(c.text for c in row)
            if self._is_footer_text(joined):
                if idx > data_idx:
                    break
                continue
            if idx != data_idx and re.search(r"指标\\s*[123]|指标1|指标2|指标3", joined):
                continue
            ys = [c.y + c.height / 2 for c in row]
            if not ys:
                continue
            row_y = float(np.median(ys))
            numeric_ratio = sum(
                1 for c in row if self._count_numeric_tokens(c.text) > 0
            ) / max(len(row), 1)
            row_mag = self._row_numeric_median(row)
            near_y = abs(row_y - base_y) <= y_tol
            med_h = float(np.median([c.height for c in base])) if base else 28.0
            # 摩尔纹双框常只差几个像素；电价行量级 < 20，不能再用 same_series 门槛
            same_line = abs(row_y - base_y) <= max(10.0, med_h * 0.45)
            dec_base = self._row_decimal_ratio(base)
            dec_row = self._row_decimal_ratio(row)
            same_kind = (dec_base >= 0.4 and dec_row >= 0.4) or (
                dec_base <= 0.15 and dec_row <= 0.15
            )
            same_series = (
                base_mag >= 20
                and row_mag >= 20
                and abs(row_mag - base_mag) / max(base_mag, 1.0) <= 0.22
            )
            if idx == data_idx or (
                numeric_ratio >= 0.5
                and ((near_y and same_series) or (same_line and same_kind))
            ):
                for c in row:
                    if self._normalize_time(c.text):
                        continue
                    if re.search(r"指标|指称", c.text):
                        continue
                    if self._count_numeric_tokens(c.text) <= 0 and not re.search(
                        r"\\d", c.text
                    ):
                        continue
                    merged.append(c)

        if not merged:
            return list(base)
        merged.sort(key=lambda c: c.x)
        dedup: List[OcrCell] = []
        for cell in merged:
            if dedup and abs(cell.x - dedup[-1].x) < 18:
                if self._better_ocr_cell(cell, dedup[-1]):
                    dedup[-1] = cell
                continue
            dedup.append(cell)
        return dedup

    def _row_numeric_median(self, row: Sequence[OcrCell]) -> float:
        """估算一行数值中位数，用于判断是否同属一条曲线。"""
        vals: List[float] = []
        for c in row:
            for part in self._split_cell_values(c.text):
                try:
                    vals.append(abs(float(part)))
                except ValueError:
                    continue
        if not vals:
            return 0.0
        return float(np.median(vals))

    def _infer_columns_from_data_row(self, data_cells: List[OcrCell],
                                     img_width: int) -> Tuple[List[str], List[float]]:
        """
        仅估算列中心；不再凭空生成时间表头（表头必须来自 OCR 识别）。

        @param data_cells 主数据行 OCR 块
        @param img_width  图像宽度
        @return (空 headers, col_centers)
        """
        _col_count, col_centers = self._estimate_column_grid(
            self._expand_row_cells(data_cells), img_width
        )
        return [], col_centers

    def _estimate_column_grid(self, cells: List[OcrCell],
                              img_width: int) -> Tuple[int, List[float]]:
        """
        根据 OCR 块 X 坐标估算列数与列中心（首列锚定、右侧外推）。

        @param cells     拆分后的 OCR 块
        @param img_width 图像宽度
        @return (列数, 列中心列表)
        """
        cells = self._filter_edge_noise_cells(cells)
        if not cells:
            return 24, self._uniform_centers([img_width * 0.05, img_width * 0.95], 24)

        centers = sorted(c.x + c.width / 2 for c in cells)
        clusters: List[float] = []
        tol = max(28.0, img_width * 0.012)
        for cx in centers:
            if clusters and cx - clusters[-1] < tol:
                clusters[-1] = (clusters[-1] + cx) / 2
            else:
                clusters.append(cx)

        n = len(clusters)
        if n < 2:
            return 24, self._uniform_centers([img_width * 0.05, img_width * 0.95], 24)

        # 碎片 OCR 会产生过多 X 簇；用单元格数回落
        cell_n = len(cells)
        if n > 28 and cell_n <= 28:
            n = max(cell_n, 4)
        elif n > 28 and n < 90:
            n = min(n, max(cell_n, 24))

        col_count = self._snap_column_count(n)
        span = clusters[-1] - clusters[0]
        step = span / max(len(clusters) - 1, 1)
        first = clusters[0]
        # 列网格铺满整行宽度，避免最后一列贴边漏识
        if col_count > 1 and img_width > first + step:
            step = (img_width * 0.97 - first) / (col_count - 1)
        col_centers = [first + step * i for i in range(col_count)]
        return col_count, col_centers

    def _filter_edge_noise_cells(self, cells: List[OcrCell]) -> List[OcrCell]:
        """
        过滤行首/行尾的 OCR 噪声（孤立短数字、贴边半截块）。
        """
        if len(cells) < 3:
            return cells
        sorted_cells = sorted(cells, key=lambda c: c.x)
        median_w = float(np.median([c.width for c in sorted_cells]) or 20.0)
        xs = [c.x + c.width / 2 for c in sorted_cells]
        span = max(xs) - min(xs) if xs else 0.0
        edge_band = max(median_w * 0.55, span * 0.02) if span > 0 else median_w * 0.55
        x_min, x_max = min(xs), max(xs)
        kept: List[OcrCell] = []
        for cell in sorted_cells:
            cx = cell.x + cell.width / 2
            near_edge = cx <= x_min + edge_band or cx >= x_max - edge_band
            narrow = float(cell.width or 0) < median_w * 0.72
            parts = self._split_cell_values(cell.text)
            if self._is_monster_numeric_token(cell.text):
                continue
            if near_edge and narrow:
                # 贴边半截：短整数 / 单 token 弱读一律丢
                if not parts or (
                    len(parts) == 1
                    and (
                        parts[0].isdigit()
                        and len(parts[0]) <= 3
                        or re.fullmatch(r"-?\d{1,2}(\.\d+)?", parts[0] or "")
                    )
                ):
                    continue
            kept.append(cell)
        return kept if kept else sorted_cells

    def _snap_column_count(self, count: int) -> int:
        """
        将估算列数对齐到常见时间表列数。

        @param count 原始估算列数
        @return 修正后的列数
        """
        if 22 <= count <= 26:
            return 24
        if 92 <= count <= 100:
            return 96
        if count < 4:
            return 24
        return count

    def _headers_for_count(self, col_count: int, morning: bool = False) -> List[str]:
        """
        按 15 分钟步长生成表头。
        - >=90 列：全天 00:15~24:00（96 点）
        - morning：从 00:15 起的窗口（Demo/上午表）
        - 否则：以 24:00 为尾的窗口（常见 24 列晚间现货表）
        """
        if col_count <= 0:
            return []
        if col_count >= 90:
            col_count = 96
            start_min = 15
        elif morning:
            start_min = self.DEFAULT_TIME_STEP
        else:
            end_min = 24 * 60
            start_min = end_min - (col_count - 1) * self.DEFAULT_TIME_STEP
        headers: List[str] = []
        cur = start_min
        while len(headers) < col_count:
            headers.append("%02d:%02d" % (cur // 60, cur % 60))
            cur += self.DEFAULT_TIME_STEP
        return headers

    def _expand_row_cells(self, row: List[OcrCell]) -> List[OcrCell]:
        """
        将粘连 OCR 块拆成多个虚拟单元格（按列宽估算 X）。

        多列检测粘连（一个框盖住多个时刻）禁止按文本猜拆，留给按列局部 OCR。
        """
        result: List[OcrCell] = []
        for cell in row:
            # 单点/无点 + 超长数字：引擎侧多格粘连，勿 DP/位宽猜拆
            if self._looks_like_multicolumn_glue_text(cell.text):
                result.append(cell)
                continue
            parts = self._split_cell_values(cell.text)
            if len(parts) <= 1:
                result.append(cell)
                continue
            n = len(parts)
            step = cell.width / max(n, 1)
            for i, val in enumerate(parts):
                result.append(OcrCell(
                    text=val, confidence=cell.confidence,
                    x=cell.x + step * i, y=cell.y,
                    width=max(step * 0.9, 8), height=cell.height,
                ))
        return result

    @staticmethod
    def _looks_like_multicolumn_glue_text(text: str) -> bool:
        """OCR 文本是否像多列粘成一串（非「原文已带多个小数点」）。"""
        t = (text or "").strip().replace("O", "0").replace("o", "0")
        t = t.replace("−", "-").replace("–", "-").replace("—", "-")
        t = re.sub(r"[^\d.\-]", "", t)
        if not t:
            return False
        digits = re.sub(r"[.\-]", "", t)
        # 已明确多个「整数≤3 位」的小数：可按文本拆（333.48、187.34）
        if len(re.findall(r"-?\d{1,3}\.\d{1,3}", t)) >= 2:
            return False
        # 已带小数点且整数部分 ≥4 位：千位以上，按粘连风险处理
        if "." in t:
            int_digits = re.sub(r"\D", "", t.split(".", 1)[0])
            if len(int_digits) >= 4:
                return True
        # 总位数 ≥7 且至多一个点：两格粘连（如 333.48+187.34→33348187.34）
        return len(digits) >= 7 and t.count(".") <= 1

    def _columns_covered_by_cell(
        self,
        cell: "OcrCell",
        col_centers: Sequence[float],
        step: float,
    ) -> List[int]:
        """检测框几何覆盖到的列中心索引（多列粘连判据）。"""
        if not col_centers:
            return []
        x0 = float(cell.x)
        x1 = float(cell.x + max(cell.width, 1))
        pad = max(step * 0.28, 8.0)
        return [
            j
            for j in range(len(col_centers))
            if (x0 - pad) <= float(col_centers[j]) <= (x1 + pad)
        ]

    def _complete_time_headers(
        self,
        headers: List[str],
        col_centers: List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        按列 X 整理表头，以 OCR 原文为准。

        - 不改写已识别时刻（不做「期望纠偏」覆盖）
        - 仅当邻列 X 间距约 2 格、且两端 OCR 时刻恰好差 2×15 分钟时，中间插 1 列
        - 全程最多插 2 个漏识格；晚间末列缺 24:00 仍可补 1 次
        """
        if len(headers) < 2:
            return list(headers), list(col_centers[: len(headers)])

        n = min(len(headers), len(col_centers))
        paired: List[Tuple[float, Optional[int], str]] = []
        for i in range(n):
            mins = self._time_to_minutes(headers[i])
            paired.append((col_centers[i], mins, headers[i]))
        paired.sort(key=lambda item: item[0])  # 按 X

        x_diffs = [b[0] - a[0] for a, b in zip(paired, paired[1:]) if b[0] > a[0]]
        med_x = float(np.median(x_diffs)) if x_diffs else 80.0
        step = self.DEFAULT_TIME_STEP
        max_inserts = 2
        inserts = 0

        out_headers: List[str] = []
        out_centers: List[float] = []
        prev_m: Optional[int] = None
        prev_x: Optional[float] = None
        for x, mins, lab in paired:
            # OCR 未解析成合法时刻：跳过该块，绝不发明替代值
            if mins is None:
                continue
            cur_m = mins
            if prev_m is not None and prev_x is not None and inserts < max_inserts:
                dx = x - prev_x
                # 仅「约 2 倍列距 + 两端时刻恰好差 2×step」才认作漏一格
                if (
                    med_x > 8
                    and 1.6 * med_x <= dx <= 2.4 * med_x
                    and cur_m - prev_m == 2 * step
                    and prev_m + step < cur_m
                ):
                    out_headers.append(self._minutes_to_time(prev_m + step))
                    out_centers.append(prev_x + 0.5 * dx)
                    inserts += 1
            # 去重：同分钟不重复（保留更靠右的 X）
            if out_headers and self._time_to_minutes(out_headers[-1]) == cur_m:
                out_centers[-1] = x
                prev_m, prev_x = cur_m, x
                continue
            out_headers.append(self._minutes_to_time(cur_m))
            out_centers.append(x)
            prev_m, prev_x = cur_m, x

        # 晚间宽表末列偶发漏识 24:00：仅在 OCR 已到 23:45 时补 1 格
        if (
            inserts < max_inserts
            and step == self.DEFAULT_TIME_STEP
            and len(out_headers) >= 20
            and out_headers
            and out_headers[0] in ("18:00", "18:15")
            and out_headers[-1] == "23:45"
            and len(out_centers) >= 2
        ):
            step_x = out_centers[-1] - out_centers[-2]
            out_headers.append("24:00")
            out_centers.append(out_centers[-1] + step_x)

        if len(out_headers) < 2:
            return list(headers), list(col_centers[: len(headers)])
        return out_headers, out_centers

    def _extend_headers_by_data_span(
        self,
        headers: List[str],
        col_centers: List[float],
        data_cells: List[OcrCell],
        img_width: int,
    ) -> Tuple[List[str], List[float]]:
        """
        以 OCR 已识别的严格 Δ=15 锚段为准，按数据行 X 跨度向左右外推缺列。

        例：锚段为 16:00~17:15（纯 OCR），数据左缘已到更左列 →
        向左推导 15:45、15:30、15:15…；右侧同理。
        不改写锚段内已有时刻；外推列仅按步长生成。

        贴边窄带不外推（避免把本图裁切半列当成缺列）。
        """
        if len(headers) < 3 or len(col_centers) < 3 or not data_cells:
            return headers, col_centers

        step_m = self.DEFAULT_TIME_STEP
        x_diffs = [
            col_centers[i + 1] - col_centers[i]
            for i in range(len(col_centers) - 1)
            if col_centers[i + 1] > col_centers[i]
        ]
        if not x_diffs:
            return headers, col_centers
        step_x = float(np.median(x_diffs))
        if step_x < 8:
            return headers, col_centers

        edge_m = self._edge_margin_px(img_width, step_x)

        # 锚段必须本身是连续 Δ=15（sanitize 后的纯 OCR/吸附结果）
        anchor_mins: List[int] = []
        for h in headers:
            m = self._time_to_minutes(h)
            if m is None:
                return headers, col_centers
            anchor_mins.append(m)
        for a, b in zip(anchor_mins, anchor_mins[1:]):
            if b - a != step_m:
                return headers, col_centers

        expanded = self._expand_row_cells(data_cells)
        strong_xs: List[float] = []
        for c in expanded:
            if self._normalize_time(c.text):
                continue
            # 贴边 OCR 残片不参与跨度
            cx = c.x + c.width / 2
            if img_width > 0 and (cx < edge_m or cx > img_width - edge_m):
                continue
            if float(c.width or 0) < step_x * 0.35 and img_width > 0:
                if c.x < edge_m * 1.5 or (c.x + c.width) > img_width - edge_m * 1.5:
                    continue
            parts = self._split_cell_values(c.text)
            if not parts:
                continue
            for part in parts:
                s = str(part).strip()
                if not s or self._normalize_time(s):
                    continue
                if self._is_monster_numeric_token(s):
                    continue
                # 仅强数值（≥3 位或带小数）参与外推，弱 1~2 位整数不触发
                if re.fullmatch(r"-?\d{1,2}", s):
                    continue
                if re.fullmatch(r"-?\d+(\.\d+)?", s):
                    strong_xs.append(cx)
        if len(strong_xs) < 3:
            return headers, col_centers

        left_x = min(strong_xs)
        right_x = max(strong_xs)
        out_h = list(headers)
        out_c = list(col_centers)

        def _has_strong_near(x: float) -> bool:
            return any(abs(sx - x) <= step_x * 0.45 for sx in strong_xs)

        # —— 向左外推：须有强数值锚，且新列中心离开贴边区 ——
        max_left = 2
        for _ in range(max_left):
            if left_x >= out_c[0] - step_x * 0.70:
                break
            first_m = self._time_to_minutes(out_h[0])
            if first_m is None or first_m < step_m:
                break
            next_x = out_c[0] - step_x
            if next_x < edge_m:
                break
            if next_x < left_x - step_x * 0.55:
                break
            if not _has_strong_near(next_x):
                break
            out_h.insert(0, self._minutes_to_time(first_m - step_m))
            out_c.insert(0, next_x)

        # —— 向右外推 ——
        max_right = 2
        last_m = self._time_to_minutes(out_h[-1])
        if last_m is None:
            return out_h, out_c
        for _ in range(max_right):
            if right_x <= out_c[-1] + step_x * 0.75:
                break
            next_x = out_c[-1] + step_x
            if img_width > 0 and next_x > img_width - edge_m:
                break
            if next_x > right_x + step_x * 0.05:
                break
            if not _has_strong_near(next_x):
                break
            last_m = last_m + step_m
            if last_m > 24 * 60:
                break
            out_h.append(self._minutes_to_time(last_m))
            out_c.append(next_x)

        return out_h, out_c

    @staticmethod
    def _edge_margin_px(img_width: int, step_x: float = 60.0) -> float:
        """本图裁切半列判定带：约 0.4 列宽（半列中心通常落在此内）。

        完整贴边列中心约在 0.5×step 处，予以保留；被裁半列中心更靠外缘则丢弃。
        """
        if img_width <= 0:
            return max(8.0, step_x * 0.35)
        return max(8.0, min(img_width * 0.04, step_x * 0.40))

    def _drop_edge_clipped_columns(
        self,
        headers: List[str],
        col_centers: List[float],
        img_width: int,
    ) -> Tuple[List[str], List[float]]:
        """丢掉本图边缘被裁断的半列。

        中心已出画布，或落入约 0.4 列宽的贴边带（半列中心典型位置）则丢弃。
        完整列即便靠外缘，中心通常仍在约 0.5×列宽处，予以保留。
        """
        n = min(len(headers), len(col_centers))
        if n < 2 or img_width <= 0:
            return list(headers[:n]), list(col_centers[:n])
        x_diffs = [
            col_centers[i + 1] - col_centers[i]
            for i in range(n - 1)
            if col_centers[i + 1] > col_centers[i]
        ]
        step_x = float(np.median(x_diffs)) if x_diffs else 60.0
        margin = self._edge_margin_px(img_width, step_x)
        w = float(img_width)
        keep_h: List[str] = []
        keep_c: List[float] = []
        for i in range(n):
            cx = float(col_centers[i])
            if cx < 0.0 or cx > w:
                continue
            if cx < margin or cx > w - margin:
                continue
            keep_h.append(headers[i])
            keep_c.append(cx)
        if len(keep_h) < 2:
            return list(headers[:n]), list(col_centers[:n])
        return keep_h, keep_c

    def _uniform_centers(self, col_centers: List[float], col_count: int) -> List[float]:
        """
        根据首尾列中心均匀插值所有列中心。

        @param col_centers 已识别列中心
        @param col_count   目标列数
        @return 均匀列中心
        """
        if col_count <= 0:
            return []
        if not col_centers:
            return [0.0] * col_count
        sorted_c = sorted(col_centers)
        if len(sorted_c) == 1:
            step = 80.0
            return [sorted_c[0] + step * i for i in range(col_count)]
        step = (sorted_c[-1] - sorted_c[0]) / max(col_count - 1, 1)
        return [sorted_c[0] + step * i for i in range(col_count)]

    def _row_y_center(self, row: Sequence[OcrCell]) -> float:
        """行中心 Y，用于判断是否同属主数据行碎片。"""
        if not row:
            return -1.0
        return float(np.median([c.y + c.height / 2 for c in row]))

    def _x_span(self, row: Sequence[OcrCell]) -> Tuple[float, float]:
        xs = [c.x + c.width / 2 for c in row]
        if not xs:
            return (0.0, 0.0)
        return (float(min(xs)), float(max(xs)))

    def _is_horizontal_complement(
        self, row: Sequence[OcrCell], ref_row: Sequence[OcrCell]
    ) -> bool:
        """左右半边互补：X 跨度几乎不重叠（拍屏斜切常见）。"""
        a0, a1 = self._x_span(row)
        b0, b1 = self._x_span(ref_row)
        if a1 <= a0 or b1 <= b0:
            return False
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        return overlap <= 0.22 * min(a1 - a0, b1 - b0)

    def _row_decimal_ratio(self, row: Sequence[OcrCell]) -> float:
        """带小数点的数值 token 占比（电价行特征）。"""
        total = 0
        dotted = 0
        for c in row:
            for part in self._split_cell_values(c.text):
                try:
                    float(part)
                except ValueError:
                    continue
                total += 1
                if "." in part:
                    dotted += 1
        if total <= 0:
            return 0.0
        return dotted / total

    def _cells_are_zeroish(self, row: Sequence[OcrCell]) -> bool:
        """电力/计划常为全 0：量级极低即视为零行。"""
        mag = self._row_numeric_median(row)
        if mag >= 5:
            return False
        nums = 0
        zeros = 0
        for c in row:
            for part in self._split_cell_values(c.text):
                try:
                    v = float(part)
                except ValueError:
                    continue
                nums += 1
                if abs(v) < 1e-6:
                    zeros += 1
        if nums <= 0:
            return mag < 1
        return zeros / nums >= 0.7

    def _is_same_indicator_fragment(
        self,
        row: Sequence[OcrCell],
        ref_row: Sequence[OcrCell],
    ) -> bool:
        """
        同一指标行的碎片：Y 几乎重叠，或左右互补。

        现货表「电力/计划」两行数值常完全相同（全 0 或同功率），
        只要不是叠在同一横线，一律当独立指标，禁止合并。

        电价行左右量级常差一个数量级（左 8/88，右 3.8），
        斜拍拆成左右残片时不能按中位数卡住。
        """
        if not row or not ref_row:
            return False
        ref_y = self._row_y_center(ref_row)
        row_y = self._row_y_center(row)
        med_h = (
            float(np.median([c.height for c in ref_row])) if ref_row else 28.0
        )
        dy = abs(row_y - ref_y)
        # 相邻指标行距 ≈ 1 个字高；只有叠在同一横线上才算碎片
        same_line = dy <= max(10.0, med_h * 0.45)
        zero_a = self._cells_are_zeroish(row)
        zero_b = self._cells_are_zeroish(ref_row)
        if zero_a and zero_b:
            return same_line

        horiz = self._is_horizontal_complement(row, ref_row)
        # 左右残片：允许略大的 Y 漂移；否则必须同线
        if horiz:
            y_tol = max(36.0, med_h * 1.8)
            if dy > y_tol * 2.4:
                return False
        elif not same_line:
            return False

        dec_a = self._row_decimal_ratio(row)
        dec_b = self._row_decimal_ratio(ref_row)
        # 一整型一小数：电力/计划 vs 电价，禁止并
        if (dec_a >= 0.5 and dec_b <= 0.15) or (dec_b >= 0.5 and dec_a <= 0.15):
            return False

        ref_mag = self._row_numeric_median(ref_row)
        row_mag = self._row_numeric_median(row)

        # 电价左右残片：两侧都偏小数 → 量级可差很多，直接并
        if horiz and dec_a >= 0.4 and dec_b >= 0.4:
            return True

        # 低量级且不在同一线：独立行（0 与 4.x 也不能并）
        # 左右残片已在上方按小数特征放行
        if ref_mag < 5 or row_mag < 5:
            return same_line
        # 非左右残片：同线才认碎片（上面已拦）
        if not horiz:
            return True
        return abs(row_mag - ref_mag) / max(ref_mag, 1.0) <= 0.35

    def _band_token_texts(self, band: Sequence[OcrCell]) -> List[str]:
        return [
            p
            for c in self._expand_row_cells(list(band))
            for p in self._split_cell_values(c.text)
        ]

    def _bands_look_like_full_duplicate_rows(
        self, a: Sequence[OcrCell], b: Sequence[OcrCell]
    ) -> bool:
        """两行都较满且数值高度相似 → 独立指标（电力=计划），不是碎片。"""
        ta = self._band_token_texts(a)
        tb = self._band_token_texts(b)
        if len(ta) < 6 or len(tb) < 6:
            return False
        n = min(len(ta), len(tb), 16)
        same = sum(1 for x, y in zip(ta[:n], tb[:n]) if x == y)
        return same / max(n, 1) >= 0.7

    def _collect_indicator_bands(
        self,
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_idx: int,
        data_cells: List[OcrCell],
    ) -> List[List[OcrCell]]:
        """
        按 Y 从上到下收集表头以下的独立数据行。
        时刻表头行及其上方（曲线刻度等）一律不进入数据带。
        """
        start = header_idx + 1 if header_idx >= 0 else 0
        header_y = (
            self._row_y_center(rows[header_idx])
            if 0 <= header_idx < len(rows)
            else -1.0
        )
        bands: List[List[OcrCell]] = []
        band_refs: List[List[OcrCell]] = []

        candidates: List[Tuple[int, List[OcrCell]]] = []
        for idx in range(start, len(rows)):
            row = rows[idx]
            joined = " ".join(c.text for c in row)
            if self._is_footer_text(joined):
                continue
            row_y = self._row_y_center(row)
            # 表头行及上方：绝不当数据行（避免时刻行变成 OCR_N）
            if header_y >= 0 and row_y <= header_y + 10:
                continue
            time_n = sum(1 for c in row if self._normalize_time(c.text))
            token_n = sum(self._count_numeric_tokens(c.text) for c in row)
            # 时刻占多数 → 表头碎片
            if time_n >= 3 and time_n >= max(token_n, 1):
                continue
            if time_n >= max(3, len(row) // 2):
                continue
            # 全 0 行常只检出 1 个「0」；旧门槛 <2 会丢掉计划行
            if token_n < 1 and idx != data_idx:
                continue
            cells = list(data_cells) if idx == data_idx and data_cells else list(row)
            if not cells:
                continue
            candidates.append((idx, cells))

        for idx, cells in candidates:
            del idx
            placed = False
            for bi, ref in enumerate(band_refs):
                if self._is_same_indicator_fragment(cells, ref):
                    merged = list(bands[bi]) + list(cells)
                    merged.sort(key=lambda c: c.x)
                    dedup: List[OcrCell] = []
                    for cell in merged:
                        if dedup and abs(cell.x - dedup[-1].x) < 18:
                            if self._better_ocr_cell(cell, dedup[-1]):
                                dedup[-1] = cell
                            continue
                        dedup.append(cell)
                    bands[bi] = dedup
                    band_refs[bi] = dedup
                    placed = True
                    break
            if not placed:
                bands.append(list(cells))
                band_refs.append(list(cells))

        bands.sort(key=lambda b: self._row_y_center(b))
        bands = self._dedupe_near_duplicate_bands(bands)
        return self._cap_indicator_bands(bands, max_n=self.EXPECTED_DATA_ROWS)

    def _band_token_count(self, band: Sequence[OcrCell]) -> int:
        return sum(self._count_numeric_tokens(c.text) for c in band)

    def _cap_indicator_bands(
        self,
        bands: List[List[OcrCell]],
        *,
        max_n: int = 3,
    ) -> List[List[OcrCell]]:
        """
        现货表体最多 3 条数据带。纠偏后仍超量时只丢最弱噪声带，不做臆测合并。
        （同线碎片合并已由 _dedupe_near_duplicate_bands 完成。）
        """
        if len(bands) <= max_n:
            return bands
        bands = sorted(bands, key=lambda b: self._row_y_center(b))
        ranked = sorted(
            range(len(bands)),
            key=lambda k: (
                self._band_token_count(bands[k]),
                self._row_decimal_ratio(bands[k]),
            ),
            reverse=True,
        )
        keep = sorted(ranked[:max_n])
        return [bands[k] for k in keep]

    def _dedupe_near_duplicate_bands(
        self, bands: List[List[OcrCell]]
    ) -> List[List[OcrCell]]:
        """
        去掉几乎重复的数据带（同一行被垂直拆两次）。
        必须 Y 同线才合并——现货表常见「电力/计划」两行数值完全相同，
        不能仅凭数值相似度丢掉独立指标行。
        """
        if len(bands) <= 1:
            return bands
        kept: List[List[OcrCell]] = []
        for band in bands:
            texts = self._band_token_texts(band)
            if not texts:
                continue
            dup = False
            band_y = self._row_y_center(band)
            band_h = float(np.median([c.height for c in band])) if band else 20.0
            for prev in kept:
                prev_texts = self._band_token_texts(prev)
                if not prev_texts:
                    continue
                prev_h = float(
                    np.median([c.height for c in prev]) if prev else band_h
                )
                dy = abs(band_y - self._row_y_center(prev))
                same_line = dy <= max(10.0, prev_h * 0.45)
                # 电力/计划：全 0 或非零同值满行 → 非同线绝不并
                if self._cells_are_zeroish(band) and self._cells_are_zeroish(prev):
                    if not same_line:
                        continue
                if self._bands_look_like_full_duplicate_rows(band, prev):
                    if not same_line:
                        continue
                horiz = self._is_horizontal_complement(band, prev)
                if not horiz and not same_line:
                    continue
                if horiz:
                    y_tol = max(36.0, prev_h * 1.5)
                    if dy > y_tol * 2.2:
                        continue
                n = min(len(texts), len(prev_texts), 12)
                same = sum(1 for a, b in zip(texts[:n], prev_texts[:n]) if a == b)
                mag_a = self._row_numeric_median(band)
                mag_b = self._row_numeric_median(prev)
                dec_a = self._row_decimal_ratio(band)
                dec_b = self._row_decimal_ratio(prev)
                # 一整型一小数：电力 vs 电价，不并
                if (dec_a >= 0.5 and dec_b <= 0.15) or (
                    dec_b >= 0.5 and dec_a <= 0.15
                ):
                    continue
                # 电价左右残片：小数特征即可并（量级常差一个数量级）
                price_horiz = horiz and dec_a >= 0.4 and dec_b >= 0.4
                mag_ok = (
                    mag_a >= 5
                    and mag_b >= 5
                    and abs(mag_a - mag_b) / max(mag_a, 1.0)
                    <= (0.35 if horiz else 0.12)
                )
                if (
                    price_horiz
                    or (horiz and mag_ok)
                    or (
                        same_line
                        and n >= 4
                        and same / n >= 0.75
                        and mag_a >= 5
                        and mag_b >= 5
                    )
                    or (
                        same_line
                        and mag_a >= 20
                        and mag_b >= 20
                        and abs(mag_a - mag_b) / max(mag_a, 1.0) <= 0.12
                        and same / max(n, 1) >= 0.5
                    )
                ):
                    if horiz:
                        merged = sorted(list(prev) + list(band), key=lambda c: c.x)
                        kept[kept.index(prev)] = merged
                    elif len(texts) > len(prev_texts):
                        kept[kept.index(prev)] = band
                    dup = True
                    break
            if not dup:
                kept.append(band)
        return kept

    def _row_filled_median(self, row: Sequence[str]) -> float:
        vals: List[float] = []
        for v in row:
            if not v or not str(v).strip():
                continue
            try:
                vals.append(float(str(v).strip()))
            except ValueError:
                continue
        return float(np.median(vals)) if vals else 0.0

    def _rows_are_complementary_fragments(
        self, a: List[str], b: List[str]
    ) -> bool:
        """
        列对齐后的左右残片：填格几乎不重叠、跨度互补、量级接近。
        典型 Excel 表现：一行只填前半列、另一行只填后半列。
        """
        n = max(len(a), len(b))
        if n < 4:
            return False
        aa = list(a) + [""] * (n - len(a))
        bb = list(b) + [""] * (n - len(b))
        ia = [i for i, v in enumerate(aa) if v and str(v).strip()]
        ib = [i for i, v in enumerate(bb) if v and str(v).strip()]
        if len(ia) < 2 or len(ib) < 2:
            return False

        overlap_idx = set(ia) & set(ib)
        conflicts = sum(1 for i in overlap_idx if aa[i] != bb[i])
        # 允许 0～1 列对齐误差；多列冲突则是两条独立指标
        if conflicts > 1:
            return False
        if len(overlap_idx) > max(1, int(0.2 * min(len(ia), len(ib)))):
            return False

        mid = (n - 1) / 2.0
        a_left = sum(1 for i in ia if i <= mid)
        a_right = len(ia) - a_left
        b_left = sum(1 for i in ib if i <= mid)
        b_right = len(ib) - b_left
        lr_comp = (a_left >= a_right and b_right > b_left) or (
            b_left >= b_right and a_right > a_left
        )

        a0, a1 = min(ia), max(ia)
        b0, b1 = min(ib), max(ib)
        span_overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
        # 跨度首尾相接或略重叠
        span_comp = span_overlap <= 2 and (a1 < b0 + 2 or b1 < a0 + 2)

        if not (lr_comp or span_comp):
            return False

        # 任一行过满（几乎整行）则不像残片
        if len(ia) >= int(n * 0.75) and len(ib) >= int(n * 0.75):
            return False

        ma = self._row_filled_median(aa)
        mb = self._row_filled_median(bb)
        # 矩阵阶段没有 Y：两行 0（电力/计划）绝不能当左右残片
        if ma < 1 or mb < 1:
            return False
        if self._row_is_mostly_zeros(aa) or self._row_is_mostly_zeros(bb):
            return False
        # 一整型一小数：电力/计划 vs 电价，禁止当左右残片合并（否则第二/三行会混）
        def _dec_ratio(row: Sequence[str]) -> float:
            filled = [str(c).strip() for c in row if c and str(c).strip()]
            if not filled:
                return 0.0
            return sum(1 for c in filled if "." in c) / len(filled)

        da, db = _dec_ratio(aa), _dec_ratio(bb)
        if (da >= 0.45 and db <= 0.2) or (db >= 0.45 and da <= 0.2):
            return False
        return abs(ma - mb) / max(ma, mb, 1.0) <= 0.38

    def _merge_two_aligned_rows(self, a: List[str], b: List[str]) -> List[str]:
        n = max(len(a), len(b))
        out: List[str] = []
        for i in range(n):
            va = a[i] if i < len(a) else ""
            vb = b[i] if i < len(b) else ""
            sa = str(va).strip() if va else ""
            sb = str(vb).strip() if vb else ""
            if sa and sb:
                # 同列双值：优先带小数、更长 token
                if ("." in sa) != ("." in sb):
                    out.append(sa if "." in sa else sb)
                else:
                    out.append(sa if len(sa) >= len(sb) else sb)
            else:
                out.append(sa or sb)
        return out

    def _merge_complementary_matrix_rows(
        self, matrix: List[List[str]]
    ) -> List[List[str]]:
        """合并左右互补残片，避免 3 行指标被拆成 OCR_4。"""
        if len(matrix) <= 1:
            return matrix
        used = [False] * len(matrix)
        out: List[List[str]] = []
        for i, row in enumerate(matrix):
            if used[i]:
                continue
            merged = list(row)
            for j in range(i + 1, len(matrix)):
                if used[j]:
                    continue
                if self._rows_are_complementary_fragments(merged, matrix[j]):
                    merged = self._merge_two_aligned_rows(merged, matrix[j])
                    used[j] = True
            used[i] = True
            out.append(merged)
        return out

    def _aligned_rows_similar(self, a: List[str], b: List[str]) -> bool:
        """
        判断两行是否像「同一行的残片补全」关系。
        两行都填得较满且内容相同 → 视为两条独立指标（如电力=计划），不算重复。
        """
        if self._row_is_mostly_zeros(a) and self._row_is_mostly_zeros(b):
            return False
        if self._rows_are_complementary_fragments(a, b):
            return True
        filled_a = sum(1 for v in a if v)
        filled_b = sum(1 for v in b if v)
        # 两行都较完整：即使数值相同也是两条测点
        if filled_a >= 6 and filled_b >= 6:
            return False
        va = [v for v in a if v]
        vb = [v for v in b if v]
        n = min(len(va), len(vb), 16)
        if n < 3:
            return False
        same = sum(1 for x, y in zip(va[:n], vb[:n]) if x == y)
        # 仅当一行明显是另一行的稀疏子集时才合并
        sparse = min(filled_a, filled_b) <= max(3, int(max(filled_a, filled_b) * 0.45))
        return sparse and same / n >= 0.7

    def _dedupe_aligned_matrix(self, matrix: List[List[str]]) -> List[List[str]]:
        """先合并左右残片，再去「残片 vs 完整行」重复；保留同值独立指标行。"""
        if len(matrix) <= 1:
            return matrix
        matrix = self._merge_complementary_matrix_rows(matrix)
        kept: List[List[str]] = []
        for row in matrix:
            dup = False
            for i, prev in enumerate(kept):
                if self._aligned_rows_similar(row, prev):
                    if self._rows_are_complementary_fragments(row, prev):
                        kept[i] = self._merge_two_aligned_rows(prev, row)
                    elif sum(1 for v in row if v) > sum(1 for v in prev if v):
                        kept[i] = row
                    dup = True
                    break
            if not dup:
                kept.append(row)
        # 再扫一遍：合并后可能产生新的互补对
        return self._merge_complementary_matrix_rows(kept)

    def _build_full_matrix(self, rows: Sequence[List[OcrCell]],
                           header_idx: int, data_idx: int,
                           headers: List[str], col_centers: List[float],
                           col_count: int) -> List[List[str]]:
        """
        构建完整数据矩阵：headers 为时刻表头，此处只输出表头以下的数据行。
        """
        del headers
        main_cells = rows[data_idx] if 0 <= data_idx < len(rows) else []
        merged_main = self._merge_fragmented_data_rows(rows, data_idx, header_idx)
        if not merged_main:
            merged_main = list(main_cells)
        bands = self._collect_indicator_bands(
            rows, header_idx, data_idx, merged_main
        )

        matrix: List[List[str]] = []
        for band in bands:
            aligned = self._align_row_to_columns(
                self._expand_row_cells(band), col_centers, col_count
            )
            if not aligned or not self._is_valid_data_row(aligned):
                continue
            if self._row_is_mostly_zeros(aligned):
                matrix.append(
                    [
                        self.NODATA_MARK if v == self.NODATA_MARK else "0"
                        for v in aligned
                    ]
                )
            else:
                matrix.append(aligned)

        # 主数据行覆盖同带；禁止再追加出第 N+1 个重复测点
        if merged_main:
            main_aligned = self._align_row_to_columns(
                self._expand_row_cells(merged_main), col_centers, col_count
            )
            if main_aligned and self._is_valid_data_row(main_aligned):
                replaced = False
                for i, band in enumerate(bands[: len(matrix)]):
                    if self._is_same_indicator_fragment(band, merged_main):
                        # 同线双框时 band 往往已经挑过更好的读数，主行更差则不覆盖
                        if self._aligned_row_strength(main_aligned) > self._aligned_row_strength(
                            matrix[i]
                        ):
                            matrix[i] = main_aligned
                        replaced = True
                        break
                if not replaced:
                    for i, row in enumerate(matrix):
                        if self._aligned_rows_similar(row, main_aligned):
                            if self._aligned_row_strength(main_aligned) > self._aligned_row_strength(
                                row
                            ):
                                matrix[i] = main_aligned
                            replaced = True
                            break
                if not replaced:
                    main_mag = self._row_numeric_median(merged_main)
                    similar = False
                    for band in bands[: len(matrix)]:
                        bmag = self._row_numeric_median(band)
                        if (
                            main_mag >= 20
                            and bmag >= 20
                            and abs(main_mag - bmag) / max(main_mag, 1.0) <= 0.18
                        ):
                            similar = True
                            break
                    # 仅当矩阵还没有足够数据行时才追加，避免表头/碎片变成 OCR_4
                    if not similar and len(matrix) == 0:
                        matrix.append(main_aligned)
                    elif not similar and len(matrix) < 3:
                        # 不足 3 行时允许补主行（常见末行电价）
                        matrix.append(main_aligned)

        if not matrix and merged_main:
            aligned = self._align_row_to_columns(
                self._expand_row_cells(merged_main), col_centers, col_count
            )
            if aligned:
                matrix = [aligned]

        return self._enforce_expected_data_rows(
            self._dedupe_aligned_matrix(matrix),
            expected=self.EXPECTED_DATA_ROWS,
        )

    def _aligned_row_strength(self, row: Sequence[str]) -> Tuple[int, int, int]:
        """行强度：填格数、小数格数、有效字符量（超量时择优保留，不拿来瞎合并）。"""
        filled = 0
        dec = 0
        chars = 0
        for v in row:
            s = str(v).strip() if v else ""
            if not s:
                continue
            filled += 1
            chars += len(s)
            if "." in s:
                dec += 1
        return (filled, dec, chars)

    def _enforce_expected_data_rows(
        self,
        matrix: List[List[str]],
        *,
        expected: int = 3,
    ) -> List[List[str]]:
        """
        硬约束：数据行恰好 expected 条（默认 3）。
        超量：丢掉最弱噪声行（保留小数电价行）。
        不足：电价沉底后在上方补全零行槽（供后续按列 OCR / 先验填 0）。
        """
        if not matrix:
            return matrix
        if len(matrix) > expected:
            strengths = [self._aligned_row_strength(r) for r in matrix]
            decimalish = [
                i
                for i, r in enumerate(matrix)
                if strengths[i][0] > 0
                and strengths[i][1] / max(strengths[i][0], 1) >= 0.35
            ]
            ranked = sorted(
                range(len(matrix)),
                key=lambda k: strengths[k],
                reverse=True,
            )
            keep_set = set(ranked[:expected])
            if decimalish and not (keep_set & set(decimalish)):
                best_dec = max(decimalish, key=lambda k: strengths[k])
                drop_cands = [k for k in keep_set if k not in decimalish]
                if not drop_cands:
                    drop_cands = list(keep_set)
                drop = min(drop_cands, key=lambda k: strengths[k])
                keep_set.discard(drop)
                keep_set.add(best_dec)
            keep = sorted(keep_set)
            matrix = [matrix[i] for i in keep]
        if len(matrix) < expected:
            matrix = self._pad_short_matrix_to_expected(matrix, expected=expected)
        return matrix

    def _row_decimal_fill_ratio(self, row: Sequence[str]) -> float:
        filled = [str(c).strip() for c in row if c and str(c).strip()]
        if not filled:
            return 0.0
        return sum(1 for c in filled if "." in c) / len(filled)

    def _row_looks_like_decimal_indicator(self, row: Sequence[str]) -> bool:
        """
        是否像独立的小数指标行（非「全 0/整型行里渗漏的几个小数」）。

        现货表两种合法形态：
        - 经典：前两行整型/全 0，末行电价小数；
        - 新画面：三行指标均可为小数（如 15.x / 18.x / 9.x）。
        """
        filled = [str(c).strip() for c in row if c and str(c).strip()]
        if len(filled) < 3:
            return False
        if self._row_is_mostly_zeros(list(row)):
            return False
        dec_n = sum(1 for c in filled if "." in c)
        if dec_n / len(filled) < 0.4:
            return False
        # 排除「多数 0 + 偶发小数噪点」
        zeroish = sum(
            1
            for c in filled
            if c in ("0", "0.0", "-0", "-0.0", "O", "o", "D", "U")
        )
        return zeroish < len(filled) * 0.5

    def _row_is_integerish_or_empty(self, row: Sequence[str]) -> bool:
        """空行 / 全 0 / 以整型功率为主 → 可走经典电力-计划拆行逻辑。"""
        filled = [str(c).strip() for c in row if c and str(c).strip()]
        if not filled:
            return True
        if self._row_is_mostly_zeros(list(row)):
            return True
        if self._row_looks_like_decimal_indicator(row):
            return False
        int_n = 0
        for s in filled:
            body = s[1:] if s.startswith(("-", "−", "–")) else s
            if s in ("0", "-0", "0.0", "-0.0", "O", "o", "D", "U"):
                int_n += 1
            elif "." not in s and re.fullmatch(r"-?\d{1,4}", s.replace("−", "-")):
                int_n += 1
            elif body in ("0", "0.0"):
                int_n += 1
        return int_n >= max(1, int(len(filled) * 0.55))

    def _matrix_decimal_indicator_count(self, matrix: Sequence[Sequence[str]]) -> int:
        return sum(1 for r in matrix if self._row_looks_like_decimal_indicator(r))

    def _matrix_is_multi_decimal_indicators(
        self, matrix: Sequence[Sequence[str]]
    ) -> bool:
        """≥2 条独立小数指标行 → 禁止「小数一律沉底/清上方」的经典电价假设。"""
        return self._matrix_decimal_indicator_count(matrix) >= 2

    def _aligned_rows_same_series(
        self, a: Sequence[str], b: Sequence[str]
    ) -> bool:
        """
        两行是否像同一指标（用于末行覆盖护栏）。

        以同列值重叠为主；仅当一行明显更稀（残片补全）时才用量级接近兜底。
        禁止仅凭「中位数都在 15–20」把两条独立小数指标判成同一行。
        """
        n = min(len(a), len(b))
        if n <= 0:
            return False
        overlap = 0
        compared = 0
        filled_a = 0
        filled_b = 0
        for i in range(max(len(a), len(b))):
            sa = str(a[i]).strip() if i < len(a) and a[i] else ""
            sb = str(b[i]).strip() if i < len(b) and b[i] else ""
            if sa:
                filled_a += 1
            if sb:
                filled_b += 1
            if i >= n or not sa or not sb:
                continue
            compared += 1
            if sa == sb:
                overlap += 1
                continue
            try:
                fa, fb = float(sa), float(sb)
            except ValueError:
                continue
            if abs(fa - fb) <= max(0.05, abs(fa) * 0.02):
                overlap += 1
        if compared >= 3 and overlap / compared >= 0.4:
            return True
        # 残片 vs 完整行：一稀一密且量级接近
        sparse = min(filled_a, filled_b) <= max(
            3, int(max(filled_a, filled_b) * 0.45)
        )
        if not sparse or max(filled_a, filled_b) < 4:
            return False
        ma = self._row_filled_median(list(a))
        mb = self._row_filled_median(list(b))
        if ma >= 5 and mb >= 5 and abs(ma - mb) / max(ma, mb, 1.0) <= 0.18:
            return True
        return False

    def _should_overwrite_last_with_main(
        self,
        matrix: Sequence[Sequence[str]],
        aligned: Sequence[str],
    ) -> bool:
        """
        主数据行是否允许覆盖 matrix[-1]。

        经典：末行稀/碎，主行可补全。
        多小数指标：主行若其实是第 1/2 行，禁止盖掉真正的末行。
        """
        if not matrix or not aligned:
            return False
        last = list(matrix[-1])
        last_n = sum(1 for v in last if v and str(v).strip())
        main_n = sum(1 for v in aligned if v and str(v).strip())
        if main_n < max(4, last_n // 2):
            return False
        # 主行更像上方某一已填指标行 → 绝不能盖/写入末行（即使末行空）
        for upper in matrix[:-1]:
            upper_n = sum(1 for v in upper if v and str(v).strip())
            if upper_n >= 3 and self._aligned_rows_same_series(upper, aligned):
                if last_n < 2 or not self._aligned_rows_same_series(last, aligned):
                    return False
        # 末行过稀且主行不像上方行：允许主行写入
        if last_n < 4:
            return True
        # 末行已是更完整/更多小数的版本：禁止用较差主行盖掉
        if self._aligned_row_strength(last) > self._aligned_row_strength(aligned):
            return False
        # 主行已像末行 → 可覆盖补全
        if self._aligned_rows_same_series(last, aligned):
            return True
        # 多小数指标且末行已较满：保持 band 顺序，不做强制覆盖
        if self._matrix_is_multi_decimal_indicators(matrix) and last_n >= 4:
            return False
        # 末行已有内容且与主行不像同一指标
        if last_n >= 4 and not self._aligned_rows_same_series(last, aligned):
            return False
        return True

    def _pad_short_matrix_to_expected(
        self,
        matrix: List[List[str]],
        *,
        expected: int = 3,
    ) -> List[List[str]]:
        """不足 expected 行时：经典模式小数行沉底；多小数指标保持原序。"""
        if not matrix or len(matrix) >= expected:
            return matrix
        ncols = max((len(r) for r in matrix), default=0)
        if ncols <= 0:
            return matrix

        def _pad(r: Sequence[str]) -> List[str]:
            out = list(r) + [""] * max(0, ncols - len(r))
            return out[:ncols]

        out = [_pad(r) for r in matrix]
        ratios = [self._row_decimal_fill_ratio(r) for r in out]
        # 仅「上方整型/全0 + 唯一小数电价」时才把小数沉底；多小数指标保 Y 序
        price_i = max(range(len(out)), key=lambda i: (ratios[i], i))
        classic_sink = (
            ratios[price_i] >= 0.25
            and price_i != len(out) - 1
            and not self._matrix_is_multi_decimal_indicators(out)
        )
        if classic_sink:
            price = out.pop(price_i)
            out.append(price)
            ratios = [self._row_decimal_fill_ratio(r) for r in out]

        empty_slot = [""] * ncols
        while len(out) < expected:
            # 经典电价在底：空槽插在末行之上；多小数/无电价特征：插顶部
            if (
                ratios
                and ratios[-1] >= 0.25
                and not self._matrix_is_multi_decimal_indicators(
                    [r for r in out if any(str(c).strip() for c in r)]
                )
            ):
                out.insert(len(out) - 1, list(empty_slot))
            else:
                out.insert(0, list(empty_slot))
            ratios = [self._row_decimal_fill_ratio(r) for r in out]
        return out[:expected]

    def _untangle_integer_decimal_rows(
        self, matrix: List[List[str]]
    ) -> List[List[str]]:
        """
        拆开混行的电力/计划（整型）与电价（小数）。

        经典形态：前两行整型/全 0，末行电价带小数；斜拍时小数渗进第 2 行。
        新画面：多行独立小数指标 → 原样保留，禁止把上方小数清成 0。
        """
        if not matrix or len(matrix) < 2:
            return matrix
        out = [list(r) for r in matrix]
        ncols = max(len(r) for r in out)
        out = [r + [""] * (ncols - len(r)) for r in out]

        def _dec_ratio(row: Sequence[str]) -> float:
            return self._row_decimal_fill_ratio(row)

        def _looks_price(cell: str) -> bool:
            s = str(cell or "").strip().replace("−", "-")
            if not s:
                return False
            if "." in s:
                return True
            body = s[1:] if s.startswith("-") else s
            return bool(re.fullmatch(r"\d{5,6}", body))

        def _looks_power_int(cell: str) -> bool:
            s = str(cell or "").strip().replace("−", "-").replace("–", "-")
            # 单字符 0 是电力/计划最常见值，必须进整型池
            if s in ("0", "-0", "0.0", "-0.0"):
                return True
            return bool(re.fullmatch(r"-?\d{1,4}", s)) and "." not in s

        # ≥2 条小数指标：保持行序与内容，不做「小数归末行」重分
        if self._matrix_is_multi_decimal_indicators(out):
            return out

        ratios = [_dec_ratio(r) for r in out]
        price_i = max(range(len(out)), key=lambda i: (ratios[i], i))
        # 小数行沉底（仅经典单电价）
        if price_i != len(out) - 1 and ratios[price_i] >= 0.25:
            out[price_i], out[-1] = out[-1], out[price_i]
            ratios[price_i], ratios[-1] = ratios[-1], ratios[price_i]

        # 逐列重分：小数归电价，整型归上方行；电价行已有小数优先保留（避免污染值盖掉真电价）
        price = list(out[-1])
        int_rows = [list(r) for r in out[:-1]]
        # 上方若已是独立小数指标，格子保留（防漏判 multi 时误清）
        upper_keep_decimal = [
            self._row_looks_like_decimal_indicator(r) for r in int_rows
        ]
        for ci in range(ncols):
            leaked_prices: List[str] = []
            int_pool: List[str] = []
            for ri, row in enumerate(int_rows):
                cur = row[ci] if ci < len(row) else ""
                s = str(cur).strip() if cur else ""
                if not s:
                    continue
                if _looks_price(s):
                    if upper_keep_decimal[ri]:
                        continue
                    leaked_prices.append(s)
                    row[ci] = ""
                elif _looks_power_int(s):
                    if s not in int_pool:
                        int_pool.append(s)

            pv = str(price[ci]).strip() if ci < len(price) and price[ci] else ""
            if _looks_price(pv):
                # 已在电价行的小数保留
                pass
            elif leaked_prices:
                # 电价格空或非小数：用渗漏小数回填（优先带点、位数接近电价）
                price[ci] = max(
                    leaked_prices,
                    key=lambda s: (
                        1 if "." in s else 0,
                        -abs(len(re.sub(r"\D", "", s)) - 5),
                        len(s),
                    ),
                )
            elif _looks_power_int(pv):
                same = any(
                    ci < len(r) and str(r[ci]).strip() == pv for r in int_rows
                )
                if same or (int_pool and pv in int_pool):
                    price[ci] = ""

            # 整型行空格：用同列整型池回填（电力≈计划时可共享）
            if int_pool:
                for ri, row in enumerate(int_rows):
                    if upper_keep_decimal[ri]:
                        continue
                    cur = row[ci] if ci < len(row) else ""
                    if cur and str(cur).strip():
                        continue
                    row[ci] = (
                        int_pool[ri] if len(int_pool) > 1 and ri < len(int_pool)
                        else int_pool[0]
                    )

        # 护栏：整型行被电价碎片掏空后不得留整行空（现货前两行常为 0）
        price_dec = _dec_ratio(price)
        for ri, row in enumerate(int_rows):
            if upper_keep_decimal[ri]:
                continue
            if any(str(c).strip() for c in row):
                continue
            if price_dec >= 0.25:
                for i in range(ncols):
                    row[i] = "0"

        out = int_rows + [price]
        return out

    def _resolve_header_anchor_y(
        self,
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_cells: Sequence[OcrCell],
        img_h: int,
        matrix: Sequence[Sequence[str]],
        col_centers: Sequence[float],
    ) -> Tuple[float, float]:
        """
        表头中心 Y 与数据行高估计。不依赖行名，仅用时刻行 / 电价行几何。
        @return (header_y, row_h)
        """
        header_y = -1.0
        header_h = 28.0
        if 0 <= header_idx < len(rows) and rows[header_idx]:
            header_y = self._row_y_center(rows[header_idx])
            hs = [c.height for c in rows[header_idx] if c.height > 0]
            if hs:
                header_h = float(np.median(hs))
        elif data_cells:
            # 无表头索引：用主数据行（电价）往上推约 3 行
            dy = self._row_y_center(list(data_cells))
            hs = [c.height for c in data_cells if c.height > 0]
            header_h = float(np.median(hs)) if hs else 28.0
            header_y = dy - 3.1 * max(header_h * 1.45, 28.0)
        else:
            header_y = float(img_h) * 0.38
            header_h = max(24.0, float(img_h) * 0.035)

        row_h = max(header_h * 1.45, 28.0)
        # 若已有电价行填格，用其 Y 反推更稳的行距
        if data_cells and header_y >= 0:
            price_y = self._row_y_center(list(data_cells))
            if price_y > header_y + 40:
                row_h = max(row_h, (price_y - header_y) / 3.15)
        del matrix, col_centers
        return float(header_y), float(row_h)

    def _geometric_data_band_centers(
        self,
        header_y: float,
        row_h: float,
        img_h: int,
    ) -> List[Tuple[float, float, float]]:
        """
        表头下方固定 3 条数据带：(y0, y1, y_center)。
        """
        bands: List[Tuple[float, float, float]] = []
        for i in range(self.EXPECTED_DATA_ROWS):
            yc = header_y + (i + 1.05) * row_h
            half = row_h * 0.48
            y0 = max(0.0, yc - half)
            y1 = min(float(img_h), yc + half)
            bands.append((y0, y1, yc))
        return bands

    def _cell_ink_looks_like_zero(
        self,
        img: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> bool:
        """亮底 + 中心少量暗墨 → 很像单字 0（整图检测常漏）。"""
        h, w = img.shape[:2]
        xa, xb = max(0, min(x0, x1)), min(w, max(x0, x1))
        ya, yb = max(0, min(y0, y1)), min(h, max(y0, y1))
        if xb - xa < 6 or yb - ya < 6:
            return False
        crop = img[ya:yb, xa:xb]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        med = float(np.median(gray))
        if med < 145:
            return False
        ink = gray < min(150.0, med - 35.0)
        ratio = float(np.mean(ink))
        if ratio < 0.003 or ratio > 0.28:
            return False
        ys, xs = np.where(ink)
        if len(xs) < 6:
            return False
        cx = float(xs.mean()) / max(gray.shape[1], 1)
        cy = float(ys.mean()) / max(gray.shape[0], 1)
        return 0.22 <= cx <= 0.78 and 0.18 <= cy <= 0.82

    def _cell_is_uncleared_token(self, raw: object) -> bool:
        """空 / 横杠 / NODATA / 仅 0 → 视为未出清占位（非真实功率/电价）。"""
        if raw is None:
            return True
        s = str(raw).strip()
        if not s or s == self.NODATA_MARK or self._is_lone_dash(s):
            return True
        if s in ("0", "0.0", "0.00", "0.000", "-0", "-0.0"):
            return True
        # OCR 把横杠/噪点读成 00
        if re.fullmatch(r"0+", s):
            return True
        return False

    def _value_looks_like_header_time(self, raw: object, header: str) -> bool:
        """数据格是否像把表头时刻读进来（22:30→22.30 / 23:00→23.00）。"""
        h = str(header or "").strip()
        s = str(raw or "").strip().replace("：", ":")
        if not h or not s or s in ("0", self.NODATA_MARK):
            return False
        hm = self._time_to_minutes(h)
        if hm is None:
            return False
        # 直接读成时刻
        if self._normalize_time(s) == self._minutes_to_time(hm):
            return True
        # 冒号被读成小数点：22.30 / 23.00（分必须两位）
        m = re.fullmatch(r"-?(\d{1,2})\.(\d{2})", s)
        if not m:
            return False
        try:
            hh, mm = int(m.group(1)), int(m.group(2))
        except ValueError:
            return False
        if mm not in (0, 15, 30, 45):
            return False
        if hh > 24 or (hh == 24 and mm != 0):
            return False
        return hh * 60 + mm == hm

    def _scrub_header_time_leaks(
        self, headers: Sequence[str], matrix: List[List[str]]
    ) -> List[List[str]]:
        """清掉数据区里误读成表头时刻的格子。"""
        if not matrix or not headers:
            return matrix
        out = [list(r) for r in matrix]
        for ri, row in enumerate(out):
            for ci in range(min(len(row), len(headers))):
                if self._value_looks_like_header_time(row[ci], headers[ci]):
                    out[ri][ci] = self.NODATA_MARK
        return out

    def _matrix_looks_all_uncleared(
        self, matrix: Sequence[Sequence[str]], ncols: int
    ) -> bool:
        """
        整表像未出清（屏上全 -）：无真实功率/电价。

        OCR 常认不出细横杠，格子为空；或已被误补成 0 / 表头泄漏小数。
        矩阵为空（未识别到任何数据行）也视为未出清，避免再列 OCR / 补 0。
        """
        if ncols <= 0:
            return False
        if not matrix:
            return True
        for row in matrix:
            padded = (list(row) + [""] * max(0, ncols - len(row)))[:ncols]
            for raw in padded:
                if self._cell_is_uncleared_token(raw):
                    continue
                # 仍可能是表头泄漏，交给 scrub 后再判；此处有其它数字则不算全未出清
                return False
        return True

    def _force_uncleared_columns_empty(
        self, headers: Sequence[str], matrix: List[List[str]]
    ) -> List[List[str]]:
        """按未出清列（含整表全 -）强制置空，禁止残留 0 / 泄漏时刻。"""
        if not matrix or not headers:
            return matrix
        ncols = len(headers)
        out = [list(r) for r in matrix]
        for r in out:
            while len(r) < ncols:
                r.append("")
            del r[ncols:]
        nodata = self._trailing_nodata_columns(out, ncols)
        if len(nodata) < ncols and self._matrix_looks_all_uncleared(out, ncols):
            nodata = set(range(ncols))
        if not nodata:
            return out
        for ri in range(len(out)):
            for ci in nodata:
                out[ri][ci] = ""
        return out

    def _trailing_nodata_columns(
        self, matrix: Sequence[Sequence[str]], ncols: int
    ) -> set:
        """
        末段「未出清」列：以电价行（末行）为准，最后一个有值时刻之后的列。

        业务：0=该时刻真实零值（写 PI）；-=尚未到点/无数据（置空、不写 PI）。
        屏上未出清时三行同为 -；电价行空而电力行被误补成 0 时，仍按电价行截断。
        电价行完全无真实值且上两行也无真实功率 → 整表未出清（全列）。
        """
        if not matrix or ncols <= 0:
            return set()
        price = (list(matrix[-1]) + [""] * max(0, ncols - len(matrix[-1])))[
            :ncols
        ]
        last_filled = -1
        for c, raw in enumerate(price):
            s = str(raw).strip() if raw is not None else ""
            if not s or s == "-" or s == self.NODATA_MARK:
                continue
            if self._cell_is_uncleared_token(s):
                continue
            last_filled = c
        if last_filled < 0:
            # 整表未出清（全 - / 全空 / 仅误补 0）：所有列置空，禁止补 0
            if self._matrix_looks_all_uncleared(matrix, ncols):
                return set(range(ncols))
            return set()
        return set(range(last_filled + 1, ncols))

    def _fill_row_by_column_zero_ocr(
        self,
        img: np.ndarray,
        row: Sequence[str],
        col_centers: Sequence[float],
        y0: float,
        y1: float,
        yc: float,
        nodata_cols: Optional[set] = None,
    ) -> List[str]:
        """
        补全电力/计划行空格为 0。

        整图检测常漏单字 0；单列 OCR 又易把竖线读成 1/16。
        保留整图已写入的非空格；空格填 0（可用墨点确认，结果相同）。
        未出清列（nodata_cols，屏上 -）保持空，禁止补 0。
        非零功率只信任整图 OCR，不在此路径臆造。
        """
        del img, y0, y1, yc  # 签名保留供几何带调用；先验填 0 无需再 OCR
        skip = nodata_cols or set()
        ncols = len(col_centers)
        out = list(row) + [""] * max(0, ncols - len(row))
        out = out[:ncols]
        for ci in range(ncols):
            if ci in skip:
                out[ci] = ""
                continue
            cur = str(out[ci]).strip() if out[ci] else ""
            if cur == self.NODATA_MARK:
                out[ci] = self.NODATA_MARK
                continue
            if cur:
                if cur in ("O", "o", "D", "U"):
                    out[ci] = "0"
                continue
            out[ci] = "0"
        return out

    def _recover_leading_zero_data_rows(
        self,
        img: np.ndarray,
        matrix: List[List[str]],
        col_centers: List[float],
        *,
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_cells: Sequence[OcrCell],
        headers: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """
        恢复前两行易漏检的全 0 数据行。

        锚点：时刻表头 Y（无行名依赖，横向滚动后仍可用）。
        电价行保持末行；上方行空/过稀则补 0，并清除多数零行里的 1/2 噪点。
        末段/整表未出清（-）列不补 0，置空以便 PI 跳过。
        """
        if not col_centers:
            return matrix
        expected = self.EXPECTED_DATA_ROWS
        matrix = self._enforce_expected_data_rows(matrix, expected=expected)
        if not matrix:
            return matrix

        ncols = len(col_centers)
        out = [list(r) + [""] * max(0, ncols - len(r)) for r in matrix]
        out = [r[:ncols] for r in out]
        if headers:
            out = self._scrub_header_time_leaks(headers, out)

        # 经典单电价：小数沉底。多小数指标行保持 band/Y 序，禁止互调
        ratios = [self._row_decimal_fill_ratio(r) for r in out]
        price_i = max(range(len(out)), key=lambda i: (ratios[i], i))
        if (
            ratios[price_i] >= 0.25
            and price_i != len(out) - 1
            and not self._matrix_is_multi_decimal_indicators(out)
        ):
            out[price_i], out[-1] = out[-1], out[price_i]

        nodata_cols = self._trailing_nodata_columns(out, ncols)
        # 整表未出清：直接置空返回，禁止前两行补 0
        if len(nodata_cols) >= ncols:
            return [[""] * ncols for _ in out]

        header_y, row_h = self._resolve_header_anchor_y(
            rows, header_idx, data_cells, img.shape[0], out, col_centers
        )
        bands = self._geometric_data_band_centers(header_y, row_h, img.shape[0])

        # 仅处理电价之上的行；独立小数指标行不补 0、不按零行降噪
        upper_n = max(0, len(out) - 1)
        for ri in range(upper_n):
            row = out[ri]
            if self._row_looks_like_decimal_indicator(row):
                continue
            filled = sum(1 for v in row if v and str(v).strip())
            empty_n = ncols - filled
            mostly_zero = self._row_is_mostly_zeros(row)
            # 已满且明显非零功率行 → 不动
            if empty_n <= 0 and not mostly_zero:
                # 仍可能是「多数 0 + 少数 1 噪点」
                out[ri] = self._coerce_zero_row_ocr_noise(
                    row, nodata_cols=nodata_cols
                )
                continue
            if empty_n <= 0 and mostly_zero:
                out[ri] = self._coerce_zero_row_ocr_noise(
                    row, nodata_cols=nodata_cols
                )
                continue
            need = empty_n >= max(1, int(ncols * 0.25)) or mostly_zero or filled == 0
            if not need:
                out[ri] = self._coerce_zero_row_ocr_noise(
                    row, nodata_cols=nodata_cols
                )
                continue
            y0, y1, yc = bands[ri] if ri < len(bands) else bands[-1]
            filled_row = self._fill_row_by_column_zero_ocr(
                img, row, col_centers, y0, y1, yc, nodata_cols=nodata_cols
            )
            out[ri] = self._coerce_zero_row_ocr_noise(
                filled_row, nodata_cols=nodata_cols
            )

        # 未出清列：上两行即使误读成 0 也强制置空（与电价行空一致，不写 PI）
        if nodata_cols:
            for ri in range(upper_n):
                if self._row_looks_like_decimal_indicator(out[ri]):
                    continue
                for ci in nodata_cols:
                    if ci < len(out[ri]):
                        out[ri][ci] = ""
            for ci in nodata_cols:
                if ci < len(out[-1]):
                    out[-1][ci] = ""

        return out

    # 拍屏摩尔纹/竖线易把 0 读成这些；在疑似全 0 行里一律压回 0
    _MOIRE_ZERO_NOISE_TOKENS = frozenset(
        {
            "1",
            "2",
            "4",
            "7",
            "11",
            "16",
            "17",
            "71",
            "111",
            "110",
            "101",
            "011",
            "01",
            "10",
            "l",
            "I",
            "|",
            "ll",
            "II",
        }
    )

    def _is_moire_zero_noise_token(self, s: str) -> bool:
        """单元格是否像「全 0 行」上的摩尔纹假读。"""
        t = str(s or "").strip()
        if not t:
            return True
        if t in ("0", "0.0", "-0", "-0.0", "O", "o", "D", "U"):
            return True
        if t in self._MOIRE_ZERO_NOISE_TOKENS:
            return True
        if "." in t:
            return False
        try:
            fv = abs(float(t))
        except ValueError:
            return True
        # 无小数且极小的整数，常见于格线误读
        return fv <= 4.0

    def _row_looks_like_moire_zero_junk(self, row: Sequence[str]) -> bool:
        """
        电力/计划行是否像「本应全 0、却被摩尔纹填成 11/4/1/111」。
        用于判断 fixed 是否够格跳过 auto，以及零行压噪。
        真·全 0 行不算垃圾（否则 fixed 永不够格，开启 auto 时总会再跑一轮）。
        """
        cells = [str(v).strip() if v else "" for v in row]
        if not cells:
            return False
        if any("." in c for c in cells if c):
            return False
        filled = [c for c in cells if c]
        if len(filled) < 3:
            return False
        zero_like = ("0", "0.0", "-0", "-0.0", "O", "o", "D", "U")
        # 已出清且全是 0：正常电力/计划行，不是摩尔纹垃圾
        if all(c in zero_like for c in filled):
            return False
        junk_n = sum(1 for c in filled if self._is_moire_zero_noise_token(c))
        if junk_n < max(3, int(len(filled) * 0.5)):
            return False
        # 真功率行数值更分散且量级通常 >10、彼此接近；垃圾行常夹杂 0 与 11/111
        try:
            vals = [abs(float(c)) for c in filled]
        except ValueError:
            return True
        med = float(np.median(vals)) if vals else 0.0
        # 中位很小，或 max 相对中位离谱（如 0,0,0,111）
        if med <= 5.0:
            return True
        mx = max(vals) if vals else 0.0
        return mx >= max(50.0, med * 8.0) and junk_n >= len(filled) * 0.4

    def _coerce_zero_row_ocr_noise(
        self, row: Sequence[str], nodata_cols: Optional[set] = None
    ) -> List[str]:
        """
        疑似全 0 的电力/计划行：把摩尔纹误读（1/4/11/111…）压回 0。
        真功率行（带小数，或非噪点数值占多数且量级正常）不改。
        未出清列（空 / -）保持空，禁止压成 0。
        """
        skip = nodata_cols or set()
        out = [str(v).strip() if v else "" for v in row]
        n = len(out)
        if n <= 0:
            return out

        def _is_zero_token(s: str) -> bool:
            return s in ("0", "0.0", "-0", "-0.0", "O", "o", "D", "U")

        zero_n = sum(1 for s in out if _is_zero_token(s) or s == "")
        junk_n = sum(1 for s in out if s and self._is_moire_zero_noise_token(s))
        looks_junk = self._row_looks_like_moire_zero_junk(out)
        # 触发：≥30% 真 0，或 ≥50% 为零/噪点，或整体像摩尔纹垃圾行
        if not looks_junk and zero_n < n * 0.3 and junk_n < n * 0.5:
            cleaned = list(out)
            for ci in skip:
                if 0 <= ci < len(cleaned):
                    cleaned[ci] = ""
            return cleaned

        cleaned: List[str] = []
        for ci, s in enumerate(out):
            if s == self.NODATA_MARK:
                cleaned.append(self.NODATA_MARK)
                continue
            if ci in skip or s == "" or s == "-":
                cleaned.append("")
                continue
            if _is_zero_token(s) or self._is_moire_zero_noise_token(s):
                cleaned.append("0")
                continue
            if "." in s:
                # 小数不该出现在电力/计划零行；整行 junk 时压掉，否则保留
                cleaned.append("0" if looks_junk else s)
                continue
            try:
                fv = abs(float(s))
            except ValueError:
                cleaned.append("0")
                continue
            # 疑似零行：连 ≥10 的 11/111 也清掉；仅保留看起来像真功率的值
            if looks_junk or fv < 20:
                cleaned.append("0")
            else:
                cleaned.append(s)
        return cleaned

    def _median_col_step(self, col_centers: List[float]) -> float:
        """相邻列中心间距的中位数。"""
        if len(col_centers) < 2:
            return 60.0
        diffs = [
            abs(col_centers[i + 1] - col_centers[i])
            for i in range(len(col_centers) - 1)
            if abs(col_centers[i + 1] - col_centers[i]) > 1e-6
        ]
        if not diffs:
            return 60.0
        return float(np.median(diffs))

    def _token_quality(self, text: str, confidence: float = 0.0) -> Tuple[int, int, float]:
        """数值 token 质量：带小数 > 多位整数 > 置信度。"""
        t = (text or "").strip()
        has_dot = 1 if "." in t else 0
        digits = len(re.sub(r"\D", "", t))
        return (has_dot, digits, float(confidence))

    def _better_ocr_cell(self, new: OcrCell, prev: OcrCell) -> bool:
        """同位置双框：带小数点、位数更多、置信度更高者优先。"""
        return self._token_quality(new.text, new.confidence) > self._token_quality(
            prev.text, prev.confidence
        )

    def _filter_overlap_noise_cells(self, row: List[OcrCell]) -> List[OcrCell]:
        """过滤同行内明显重叠/噪声的 OCR 块，保留质量更高者。"""
        if len(row) <= 1:
            return list(row)
        kept: List[OcrCell] = []
        for cell in sorted(row, key=lambda c: c.x):
            if not kept:
                kept.append(cell)
                continue
            prev = kept[-1]
            if abs(cell.x - prev.x) < 14:
                q_new = self._token_quality(cell.text, cell.confidence)
                q_old = self._token_quality(prev.text, prev.confidence)
                if q_new > q_old:
                    kept[-1] = cell
                continue
            kept.append(cell)
        return kept

    def _is_lone_dash(self, text: str) -> bool:
        """整格只有横杠、没有数字（未出清）。负数如 -2.3 不是。"""
        t = str(text or "").strip()
        if not t or t == self.NODATA_MARK:
            return t == self.NODATA_MARK
        for ch in ("−", "–", "—", "﹣", "－", "­"):
            t = t.replace(ch, "-")
        return bool(t) and set(t) <= {"-"} and not re.search(r"\d", t)

    def _strip_nodata_marks(self, matrix: List[List[str]]) -> List[List[str]]:
        """无数据标记落成空字符串，供展示与 PI 跳过。"""
        return [
            ["" if c == self.NODATA_MARK else c for c in row]
            for row in matrix
        ]

    def _align_row_to_columns(self, row: List[OcrCell],
                              col_centers: List[float],
                              col_count: int) -> List[str]:
        """
        将 OCR 行按 X 坐标对齐到列网格。

        绝不「从左到右顺序填列」：中间漏识必须留空，不能把右侧数值前移。
        粘连多值块优先按「块覆盖的列中心」落列，不再只按宽度均分。
        若检测框本身盖住 ≥2 列且文本像多格粘连：不落库，留给按列局部 OCR。
        """
        if not row or col_count <= 0 or not col_centers:
            return []

        values = [""] * col_count
        qualities = [(-1, 0, 0.0)] * col_count
        step = self._median_col_step(col_centers)
        max_dist = max(step * 0.60, 28.0)

        for cell in self._filter_overlap_noise_cells(row):
            if self._is_footer_text(cell.text):
                continue
            # 时刻表头碎片禁止落入数据格（否则 23:00→23.00）
            if self._normalize_time(cell.text):
                continue
            covered = self._columns_covered_by_cell(cell, col_centers, step)
            # 单列整格只有「-」：记为无数据，禁止后续单列重识
            if self._is_lone_dash(cell.text):
                target = covered[0] if len(covered) == 1 else None
                if target is None and cell.width <= step * 1.35:
                    cx = float(cell.x + max(cell.width, 1) / 2.0)
                    col = min(
                        range(col_count),
                        key=lambda j: abs(cx - float(col_centers[j])),
                    )
                    if abs(cx - float(col_centers[col])) <= max_dist:
                        target = col
                if (
                    target is not None
                    and not values[target]
                ):
                    values[target] = self.NODATA_MARK
                    qualities[target] = (-1, 0, 0.0)
                continue
            # 宽框多列粘连：先试均衡文本拆；成功则落入覆盖列，失败才留空给几何重识
            if len(covered) >= 2 and self._looks_like_multicolumn_glue_text(cell.text):
                parts = self._split_cell_values(cell.text)
                if parts and len(parts) >= 2 and self._sticky_pair_plausible(parts):
                    self._assign_parts_into_columns(
                        cell,
                        parts,
                        col_centers,
                        col_count,
                        values,
                        qualities,
                        step=step,
                        max_dist=max_dist,
                    )
                elif parts and len(parts) == 1:
                    # 9 位缺字只信左段：写入最左覆盖列，其余列留给单列 OCR
                    self._assign_parts_into_columns(
                        cell,
                        parts,
                        col_centers,
                        col_count,
                        values,
                        qualities,
                        step=step,
                        max_dist=max_dist,
                        force_cols=covered[:1],
                    )
                continue
            parts = self._split_cell_values(cell.text)
            if not parts:
                continue
            self._assign_parts_into_columns(
                cell,
                parts,
                col_centers,
                col_count,
                values,
                qualities,
                step=step,
                max_dist=max_dist,
            )

        return values if any(v.strip() for v in values) else []

    def _assign_parts_into_columns(
        self,
        cell: "OcrCell",
        parts: List[str],
        col_centers: List[float],
        col_count: int,
        values: List[str],
        qualities: List[Tuple[int, int, float]],
        *,
        step: float,
        max_dist: float,
        force_cols: Optional[Sequence[int]] = None,
    ) -> None:
        """把拆出的数值按列中心落入网格（粘连块按覆盖列切，而非宽度均分）。"""
        x0 = float(cell.x)
        x1 = float(cell.x + max(cell.width, 1))
        pad = step * 0.28
        covered = [
            j
            for j in range(col_count)
            if (x0 - pad) <= float(col_centers[j]) <= (x1 + pad)
        ]
        if force_cols is not None:
            covered = [int(j) for j in force_cols if 0 <= int(j) < col_count]
        n = len(parts)

        def _put(col: int, val: str) -> None:
            if col < 0 or col >= col_count:
                return
            if self._is_monster_numeric_token(val):
                return
            q = self._token_quality(val, cell.confidence)
            # 无数据标记可被真实数值覆盖
            if (
                not values[col]
                or values[col] == self.NODATA_MARK
                or q > qualities[col]
            ):
                values[col] = val
                qualities[col] = q

        if n == 1:
            if force_cols is not None and covered:
                _put(covered[0], parts[0])
                return
            cx = (x0 + x1) / 2.0
            col = min(range(col_count), key=lambda j: abs(cx - col_centers[j]))
            if abs(cx - col_centers[col]) <= max_dist:
                _put(col, parts[0])
            return

        # 块覆盖 ≥2 列：按列中心切分落位
        if len(covered) >= 2:
            if n == len(covered):
                for val, col in zip(parts, covered):
                    _put(col, val)
                return
            if n < len(covered):
                # 部分列漏拆：按比例映射到 covered
                last = max(n - 1, 1)
                for i, val in enumerate(parts):
                    idx = int(round(i * (len(covered) - 1) / last))
                    _put(covered[idx], val)
                return
            # 拆出段数多于覆盖列：用块内相对位置找最近列中心
            span = max(x1 - x0, 1.0)
            for i, val in enumerate(parts):
                cx = x0 + (i + 0.5) / n * span
                col = min(covered, key=lambda j: abs(float(col_centers[j]) - cx))
                if abs(float(col_centers[col]) - cx) <= max_dist * 1.25:
                    _put(col, val)
            return

        # 回退：宽度均分（覆盖列不足时）
        for i, val in enumerate(parts):
            seg = (x1 - x0) / n
            cx = x0 + seg * i + seg / 2
            col = min(range(col_count), key=lambda j: abs(cx - col_centers[j]))
            dist = abs(cx - col_centers[col])
            if dist > max_dist * 1.15:
                continue
            _put(col, val)

    def _refill_matrix_empties(
        self,
        img: np.ndarray,
        matrix: List[List[str]],
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_idx: int,
        data_cells: List[OcrCell],
        col_centers: List[float],
    ) -> List[List[str]]:
        """量级清理后，对空格做局部 OCR；优先末行（电价），空得多也全量补。"""
        if not matrix or not col_centers:
            return matrix
        bands = self._collect_indicator_bands(
            rows, header_idx, data_idx, data_cells or []
        )
        out = [list(r) for r in matrix]
        img_w = img.shape[1]
        ncols = len(col_centers)
        # 末段未出清（-）列：跳过单列 OCR，避免对空/横杠狂重识拖垮耗时
        nodata_cols = self._trailing_nodata_columns(out, ncols)
        # 从末行往前（电价行最易粘连漏识）
        order = list(range(len(out) - 1, -1, -1))
        for ri in order:
            row = out[ri]
            empty_n = sum(
                1
                for i, v in enumerate(row)
                if (not v) and i not in nodata_cols
            )
            if empty_n <= 0:
                continue
            band = bands[ri] if ri < len(bands) else (
                data_cells if ri == len(out) - 1 else []
            )
            if not band:
                if not data_cells:
                    continue
                y0 = self._row_y_center(data_cells)
                h_med = float(np.median([c.height for c in data_cells]) or 20.0)
                fake_y = y0 - (len(out) - 1 - ri) * h_med * 1.3
                band = [
                    OcrCell(
                        text="0",
                        confidence=0.5,
                        x=col_centers[0],
                        y=fake_y - h_med / 2,
                        width=20,
                        height=h_med,
                    )
                ]
            is_price_row = ri == len(out) - 1
            # 电力/计划行空得多：禁止局部 OCR（竖纹易读成 11/1/111），
            # 留给 _recover_leading_zero_data_rows 用先验填 0
            if (
                not is_price_row
                and empty_n >= max(2, int(len(row) * 0.35))
            ):
                continue
            # 空格很多时：末行仍全量局部重识；其它行只补尾部，避免拖垮
            if empty_n > max(8, int(len(row) * 0.45)) and not is_price_row:
                out[ri] = self._override_tail_by_cell_ocr(
                    img, row, col_centers, band, img_w, tail=4
                )
                continue
            out[ri] = self._ocr_fill_empty_cells(
                img,
                row,
                col_centers,
                band,
                img_w,
                prefer_sharp=is_price_row,
                skip_cols=nodata_cols,
            )
        return out

    def _reocr_geometry_sticky_cells(
        self,
        img: np.ndarray,
        matrix: List[List[str]],
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_idx: int,
        data_cells: List[OcrCell],
        col_centers: List[float],
    ) -> List[List[str]]:
        """
        检测框盖住 ≥2 列的引擎粘连：清空被盖列，按列裁剪后重新 OCR。

        不做 5/6 位文本切分；只依赖列几何 + 单格 OCR（与前列能认点同一路径）。
        """
        if not matrix or not col_centers or not data_cells:
            return matrix
        step = self._median_col_step(col_centers)
        sticky_cols: List[int] = []
        for cell in data_cells:
            covered = self._columns_covered_by_cell(cell, col_centers, step)
            if len(covered) < 2:
                continue
            if not self._looks_like_multicolumn_glue_text(cell.text):
                continue
            for j in covered:
                if j not in sticky_cols:
                    sticky_cols.append(j)
        if not sticky_cols:
            return matrix

        out = [list(r) for r in matrix]
        bands = self._collect_indicator_bands(
            rows, header_idx, data_idx, data_cells or []
        )
        # 主数据行（电价）优先；同 y 带也可能被宽框污染
        target_rows = [len(out) - 1]
        for ri, band in enumerate(bands):
            if ri >= len(out) or not band:
                continue
            for cell in band:
                covered = self._columns_covered_by_cell(cell, col_centers, step)
                if len(covered) >= 2 and self._looks_like_multicolumn_glue_text(
                    cell.text
                ):
                    if ri not in target_rows:
                        target_rows.append(ri)
                    break

        for ri in target_rows:
            row = out[ri]
            cleared = list(row)
            for j in sticky_cols:
                if j < len(cleared) and cleared[j] != self.NODATA_MARK:
                    cleared[j] = ""
            band = bands[ri] if ri < len(bands) and bands[ri] else data_cells
            filled = self._ocr_fill_empty_cells(
                img,
                cleared,
                col_centers,
                band,
                img.shape[1],
                prefer_sharp=True,
                trust_cell_ocr=True,
            )
            for j in sticky_cols:
                if j >= len(filled) or not filled[j]:
                    continue
                if row[j]:
                    # 已有均衡拆出的合理值时，单列 OCR 残片不得覆盖
                    if not self._reocr_better_than_existing(
                        filled[j], row[j], row
                    ):
                        continue
                    row[j] = filled[j]
                elif self._sticky_empty_fill_plausible(filled[j], row):
                    row[j] = filled[j]
            out[ri] = row
        return out

    def _sticky_empty_fill_plausible(
        self, cand: str, row: Sequence[str]
    ) -> bool:
        """空列单列 OCR 填入门禁：挡 3.1 / 5.3 等窄列残片。"""
        if self._is_monster_numeric_token(cand):
            return False
        pv = self._preview_sticky_price(cand)
        if pv is None or pv >= 400.0:
            return False
        _, small_med, _ = self._price_row_small_stats(row)
        if small_med <= 0:
            return True
        if pv > small_med * 4.0 or pv < max(small_med / 4.0, 0.5):
            return False
        return True

    def _reocr_better_than_existing(
        self, cand: str, existing: str, row: Sequence[str]
    ) -> bool:
        """
        单列重识是否明显优于已有值。

        粘连均衡拆常先得到无点 5 位（19396），几何重识若用窄列残片
        （1.939 / 5.3）仅因「带小数点」覆盖，会在补点之前毁掉正确段。
        以量级预览相对小电价簇为准，不单看有无小数点。
        """
        if not existing:
            return True
        if not cand:
            return False
        pv_new = self._preview_sticky_price(cand)
        pv_old = self._preview_sticky_price(existing)
        if pv_new is None:
            return False
        _, small_med, _ = self._price_row_small_stats(row)
        if pv_old is not None and pv_old < 400.0 and small_med > 0:
            err_new = abs(pv_new - small_med)
            err_old = abs(pv_old - small_med)
            # 量级明显更差（1.939 盖 19396→19.396）→ 拒绝
            if err_new > err_old * 1.25 + 2.0:
                return False
            # 已有漏点长数字，候选带点且量级不更差 → 允许升级认点
            if "." not in str(existing) and "." in str(cand):
                return err_new <= err_old * 1.25 + 2.0
            # 需明显更接近簇才覆盖
            return err_new < err_old * 0.85
        q_new = self._token_quality(cand)
        q_old = self._token_quality(existing)
        if q_old[0] and (not q_new[0] or q_new[1] + 1 < q_old[1]):
            return False
        return q_new > q_old

    def _reocr_suspicious_price_cells(
        self,
        img: np.ndarray,
        matrix: List[List[str]],
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_idx: int,
        data_cells: List[OcrCell],
        col_centers: List[float],
    ) -> List[List[str]]:
        """电价行可疑已填格按列强制重识（可覆盖串位/漏点错值）。"""
        if not matrix or not col_centers or len(matrix) < 1:
            return matrix
        out = [list(r) for r in matrix]
        ri = len(out) - 1
        row = out[ri]
        small_vals, small_med, med_int = self._price_row_small_stats(row)
        sus = [
            i
            for i in range(len(row))
            if row[i] != self.NODATA_MARK
            and self._is_suspicious_price_cell(row, i, small_med, med_int)
        ]
        if not sus:
            return out
        bands = self._collect_indicator_bands(
            rows, header_idx, data_idx, data_cells or []
        )
        band = bands[ri] if ri < len(bands) else (data_cells or [])
        if not band and data_cells:
            band = data_cells
        if not band:
            return out
        short = [
            i
            for i in sus
            if "." not in str(row[i])
            and re.fullmatch(r"\d{2,3}", re.sub(r"\D", "", str(row[i]).lstrip("-")))
        ]
        # 2～3 位无点：竖纹抑制后按邻格量级择优，不用通用重识（通用通道会偏向无点高置信错读）
        if short:
            row = self._reread_short_price_integers(img, row, col_centers, band)
        rest = [i for i in sus if i not in short]
        if not rest:
            out[ri] = row
            return out
        # 先置空再走局部 OCR（强制覆盖）
        cleared = list(row)
        for i in rest:
            cleared[i] = ""
        filled = self._ocr_fill_empty_cells(
            img,
            cleared,
            col_centers,
            band,
            img.shape[1],
            prefer_sharp=True,
        )
        # 重识成功则覆盖；极端离谱且重识失败 → 置空（宁缺毋滥，避免 193/969.307 落库）
        for i in rest:
            if filled[i]:
                row[i] = filled[i]
            elif self._extreme_price_outlier(str(out[ri][i]), small_med):
                row[i] = ""
        out[ri] = row
        return out

    def _suppress_vertical_stripes(self, gray: np.ndarray) -> np.ndarray:
        """削弱拍屏竖向周期条纹，尽量留下数字笔画和小数点。"""
        try:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            rows, cols = gray.shape[:2]
            crow = rows // 2
            ccol = cols // 2
            mask = np.ones((rows, cols), np.float32)
            yband = max(3, rows // 18)
            xkeep = max(2, cols // 80) * 3
            mask[crow - yband : crow + yband, :] = 0.15
            mask[crow - yband : crow + yband, ccol - xkeep : ccol + xkeep] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
            back = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
            return np.clip(np.abs(back), 0, 255).astype(np.uint8)
        except Exception:
            return gray

    def _price_dot_forms(self, text: str) -> List[str]:
        """把单格读数展开成可与邻格比较的小数形态。已有小数点则原样保留。"""
        s = str(text or "").strip().replace("−", "-").replace("–", "-").replace("—", "-")
        if not s or s in ("-", "0", "-0"):
            return []
        sign = "-" if s.startswith("-") else ""
        body = s[1:] if sign else s
        if re.fullmatch(r"\d{1,3}\.\d{1,3}", body):
            return [sign + body]
        digits = re.sub(r"\D", "", body)
        if not re.fullmatch(r"\d+", digits or ""):
            return []
        # 2 位不猜点（67→6.7 太容易伤到真实整数）；3 位给 xx.x / x.xx
        if len(digits) == 3:
            return [sign + digits[:2] + "." + digits[2:], sign + digits[0] + "." + digits[1:]]
        if len(digits) == 4:
            return [sign + digits[0] + "." + digits[1:]]
        return []

    def _reread_short_price_integers(
        self,
        img: np.ndarray,
        row: List[str],
        col_centers: List[float],
        band: Sequence[OcrCell],
    ) -> List[str]:
        """
        电价行 2～3 位无点整数：抑制竖纹后重识，只在更贴近左右邻格时覆盖。

        62.4 被读成 67、28.5 被读成 111、15.2 被读成 152 时，竖纹滤掉后常见 624 / 28.5 / 152。
        用邻格相对误差选 xx.x 或 x.xx；对不上邻格就保留原文。
        """
        if not row or not band or not col_centers:
            return row
        targets = [
            i
            for i, cell in enumerate(row)
            if "." not in str(cell)
            and re.fullmatch(r"-?\d{2,3}", str(cell).strip().replace("−", "-").replace("–", "-"))
        ]
        if not targets:
            return row
        ys = [c.y + c.height / 2 for c in band]
        hs = [c.height for c in band if c.height and c.height > 0]
        y_med = float(np.median(ys))
        h_med = float(np.median(hs)) if hs else 40.0
        y0 = int(max(0, y_med - h_med * 1.15))
        y1 = int(min(img.shape[0], y_med + h_med * 1.15))
        bounds = self._column_bounds(col_centers, img.shape[1])
        # 锚点看原始行，避免同一行多个错格互相带偏
        base = list(row)
        out = list(row)
        ocr = self._get_ocr()
        for col in targets:
            if col >= len(bounds):
                continue
            x0, x1 = bounds[col]
            span = max(8, x1 - x0)
            pad = max(2, int(span * 0.06))
            rx0 = max(0, x0 - pad)
            rx1 = min(img.shape[1], x1 + pad)
            if rx1 - rx0 < 12 or y1 - y0 < 12:
                continue
            crop = img[y0:y1, rx0:rx1]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            notch = self._suppress_vertical_stripes(gray)
            notch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(notch)
            tight0 = max(0, x0 + max(1, int(span * 0.04)))
            tight1 = min(img.shape[1], x1 - max(1, int(span * 0.04)))
            variants = [cv2.cvtColor(notch, cv2.COLOR_GRAY2BGR)]
            up = cv2.resize(notch, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            variants.append(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR))
            if tight1 - tight0 >= 12:
                tight = img[y0:y1, tight0:tight1]
                tg = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
                tn = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(
                    self._suppress_vertical_stripes(tg)
                )
                variants.append(
                    cv2.cvtColor(
                        cv2.resize(tn, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC),
                        cv2.COLOR_GRAY2BGR,
                    )
                )
            # 整图 OCR 往往已经读对各位（624/285/152），只是丢了小数点
            readings: List[str] = [str(base[col]).strip()]
            for variant in variants:
                result, _ = ocr(variant)
                if not result:
                    continue
                for item in result:
                    norm = self._normalize_cell_value(str(item[1]))
                    if norm and norm not in ("0", "0.0"):
                        readings.append(norm)
            anchors: List[float] = []
            for nb in (col - 1, col + 1):
                if 0 <= nb < len(base):
                    fv = self._parse_dotted_cell_value(str(base[nb]))
                    if fv is not None:
                        anchors.append(fv)
            if len(anchors) < 1:
                dotted = [
                    self._parse_dotted_cell_value(str(c))
                    for j, c in enumerate(base)
                    if j != col
                ]
                dotted_f = [v for v in dotted if v is not None and v < 1000.0]
                if dotted_f:
                    anchors.append(float(np.median(dotted_f)))
            if not anchors:
                continue
            orig_rel = self._rel_err_to_anchors(abs(float(base[col])), anchors)
            best_text = ""
            best_rel = orig_rel
            seen = set()
            for reading in readings:
                for form in self._price_dot_forms(reading):
                    if form in seen or "." not in form:
                        continue
                    seen.add(form)
                    try:
                        fv = abs(float(form))
                    except ValueError:
                        continue
                    rel = self._rel_err_to_anchors(fv, anchors)
                    if rel < best_rel:
                        best_rel = rel
                        best_text = form
            # 必须是小数，且明显比原文更靠近邻格；0.40 挡住 111→11.1 这种乱插点
            if best_text and best_rel <= 0.40 and best_rel + 0.05 < orig_rel:
                out[col] = best_text
        return out

    def _extreme_price_outlier(self, raw: str, small_med: float) -> bool:
        """
        相对小电价簇极端偏离（粘连误拆残骸）。

        用补点预览量级：4306→4.306、19396→19.396 不得当 4306/19396 清空。
        """
        s = str(raw or "").strip().replace("−", "-").replace("–", "-")
        if not s or s in ("-", "0", "-0"):
            return False
        pv = self._preview_sticky_price(s)
        if pv is None:
            try:
                pv = abs(float(s))
            except ValueError:
                return False
        # 电价整数最多 3 位；预览到千位（≥1000）才是粘连残骸
        return pv >= 1000.0

    def _ocr_single_cell(self, img: np.ndarray,
                         x0: int, y0: int, x1: int, y1: int,
                         fast: bool = True) -> str:
        """对单个网格单元格 OCR，返回文本（兼容旧调用）。"""
        text, _ = self._ocr_single_cell_located(img, x0, y0, x1, y1, fast=fast)
        return text

    def _ocr_single_cell_located(
        self,
        img: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        fast: bool = True,
    ) -> Tuple[str, Optional[float]]:
        """
        单元格 OCR，同时返回检出文本块中心 X（图坐标系）。
        用于判断是否误读到邻列。
        """
        h, w = img.shape[:2]
        col_w = max(1, x1 - x0)
        pad_x = max(2, int(col_w * 0.08))
        pad_y = 4
        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rx1 = min(w, x1 + pad_x)
        ry1 = min(h, y1 + pad_y)
        if rx1 - rx0 < 10 or ry1 - ry0 < 10:
            return "", None
        crop = img[ry0:ry1, rx0:rx1]
        best_text = ""
        best_key: Tuple[int, int, float] = (-1, 0, -1.0)
        best_cx: Optional[float] = None
        for variant in self._cell_ocr_variants(crop, fast=fast):
            ocr = self._get_ocr()
            result, _ = ocr(variant)
            if not result:
                continue
            saw_dash = False
            for item in result:
                box, text, score = item[0], str(item[1]).strip(), float(item[2])
                if self._is_lone_dash(text):
                    saw_dash = True
                    continue
                norm = self._normalize_cell_value(text)
                if not norm:
                    continue
                key = self._token_quality(norm, score)
                if key > best_key:
                    best_key = key
                    best_text = norm
                    xs = [p[0] for p in box]
                    # 映射回原图 X
                    scale_x = (rx1 - rx0) / max(variant.shape[1], 1)
                    best_cx = rx0 + (min(xs) + max(xs)) / 2.0 * scale_x
            if best_key[0] == 1 and best_key[2] >= 3.0:
                return best_text, best_cx
            # 本变体只认出「-」：置空并停止其余预处理，不再当漏识重试
            if saw_dash and not best_text:
                return self.NODATA_MARK, None
        return best_text, best_cx

    def _cell_ocr_variants(self, crop: np.ndarray,
                           fast: bool = True) -> List[np.ndarray]:
        """单元格预处理。fast 也做轻度放大；非 fast 再加锐化/二值，利于小数点。"""
        variants = [crop]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        up = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR))
        if fast:
            return variants
        sharp = cv2.addWeighted(
            up, 1.4, cv2.GaussianBlur(up, (0, 0), 0.9), -0.4, 0
        )
        variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
        _, bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))
        return variants

    def _ocr_fill_empty_cells(
        self,
        img: np.ndarray,
        row: List[str],
        col_centers: List[float],
        data_cells: List[OcrCell],
        img_width: int,
        *,
        prefer_sharp: bool = False,
        trust_cell_ocr: bool = False,
        skip_cols: Optional[set] = None,
    ) -> List[str]:
        """
        对数据行空单元格局部 OCR 补全。

        紧裁优先；若检出文本中心更靠近邻列中心，则丢弃（防后列前移补位）。
        prefer_sharp：电价行空格优先用非 fast（锐化/二值）利于小数点。
        trust_cell_ocr：信任单列 OCR 原文（几何粘连重识用），不做同行量级过滤。
        skip_cols：未出清列等，跳过单列 OCR。
        """
        if not row or not data_cells:
            return row
        skip = skip_cols or set()
        empty_cols = [
            i for i, v in enumerate(row) if (not v) and i not in skip
        ]
        if not empty_cols:
            return row

        ys = [c.y + c.height / 2 for c in data_cells]
        hs = [c.height for c in data_cells]
        y_med = float(np.median(ys))
        h_med = float(np.median(hs)) if hs else 40.0
        y_tight0 = int(max(0, y_med - h_med * 1.25))
        y_tight1 = int(min(img.shape[0], y_med + h_med * 1.25))
        box_h = int(max(h_med * 2.6, 40))
        y_box0 = int(max(0, y_med - box_h / 2))
        y_box1 = int(min(img.shape[0], y_med + box_h / 2))
        bounds = self._column_bounds(col_centers, img_width)
        step = self._median_col_step(col_centers)

        ref_vals: List[float] = []
        for v in row:
            if not v or v in ("0", "0.0"):
                continue
            try:
                fv = abs(float(v))
            except ValueError:
                continue
            if fv >= 1.0:
                ref_vals.append(fv)
        ref_med = float(np.median(ref_vals)) if ref_vals else 0.0

        filled = list(row)
        for col in empty_cols:
            x0, x1 = bounds[col]
            cx = col_centers[col]
            # 列内再收紧，减少邻列干扰
            inset = max(2, int((x1 - x0) * 0.10))
            pad_small = max(1, int((x1 - x0) * 0.04))
            pad_mid = max(3, int((x1 - x0) * 0.12))
            if prefer_sharp:
                attempts = (
                    (x0 + inset, y_tight0, x1 - inset, y_tight1, False),
                    (x0 + pad_small, y_tight0, x1 - pad_small, y_tight1, False),
                    (x0, y_tight0, x1, y_tight1, True),
                    (x0 - pad_mid, y_box0, x1 + pad_mid, y_box1, False),
                )
            else:
                attempts = (
                    (x0 + inset, y_tight0, x1 - inset, y_tight1, True),
                    (x0 + pad_small, y_tight0, x1 - pad_small, y_tight1, True),
                    (x0, y_tight0, x1, y_tight1, True),
                    (x0 - pad_mid, y_box0, x1 + pad_mid, y_box1, False),
                )
            best = ""
            for ax0, ay0, ax1, ay1, fast in attempts:
                if ax1 - ax0 < 8:
                    continue
                cand, found_x = self._ocr_single_cell_located(
                    img, ax0, ay0, ax1, ay1, fast=fast
                )
                if cand == self.NODATA_MARK:
                    break
                if not cand:
                    continue
                if found_x is not None and len(col_centers) > 1:
                    nearest = min(
                        range(len(col_centers)),
                        key=lambda j: abs(found_x - col_centers[j]),
                    )
                    if nearest != col:
                        if abs(found_x - col_centers[nearest]) + step * 0.15 < abs(
                            found_x - cx
                        ):
                            continue
                if trust_cell_ocr:
                    # 几何粘连：单列 OCR 原文即结果，跳过量级过滤
                    if self._is_monster_numeric_token(cand):
                        continue
                    picked = cand
                else:
                    picked = self._pick_fill_candidate([cand], ref_med)
                    if not picked:
                        continue
                stolen = False
                for nb_i in (col - 1, col + 1):
                    if 0 <= nb_i < len(filled) and self._is_neighbor_fragment(
                        picked, filled[nb_i]
                    ):
                        stolen = True
                        break
                if stolen:
                    continue
                best = picked
                break
            if best:
                # trust_cell_ocr 时保留 OCR 原文（含小数点），不做补点猜测
                if trust_cell_ocr:
                    filled[col] = best
                else:
                    filled[col] = self._fix_decimal_ocr(best) or best
        return filled

    def _is_neighbor_fragment(self, cand: str, neighbor: str) -> bool:
        """候选是否像邻列数值的残片（防止后列前移/截断补位）。"""
        if not cand or not neighbor:
            return False
        if cand == neighbor:
            return False
        try:
            a, b = abs(float(cand)), abs(float(neighbor))
        except ValueError:
            return False
        da = re.sub(r"\D", "", cand)
        db = re.sub(r"\D", "", neighbor)
        # 数值极近且候选更「残」（无小数/位数更少）才判偷读
        if b >= 1 and abs(a - b) / b <= 0.03:
            if ("." in neighbor and "." not in cand) or (
                da and db and len(da) < len(db)
            ):
                return True
        if da and db and da != db and len(da) + 1 < len(db) and (
            da in db or db.startswith(da) or db.endswith(da)
        ):
            return True
        return False

    def _pick_fill_candidate(self, cands: List[str], ref_med: float) -> str:
        """从空格补洞候选中选最佳：非零优先，贴近行量级；尝试恢复丢首位（110→410）。"""
        scored: List[Tuple[Tuple[int, int, float, float], str]] = []
        for raw in cands:
            if not raw or raw in ("0", "0.0"):
                continue
            for cand in self._magnitude_repair_candidates(raw, ref_med):
                q = self._token_quality(cand)
                try:
                    fv = abs(float(cand))
                except ValueError:
                    continue
                mag_bonus = 0
                prox = 0.0
                if ref_med >= 50:
                    ratio = fv / ref_med
                    if 0.55 <= ratio <= 1.45:
                        mag_bonus = 2
                        prox = -abs(fv - ref_med)
                    elif 0.35 <= ratio <= 1.7:
                        mag_bonus = 1
                        prox = -abs(fv - ref_med)
                    else:
                        continue
                scored.append(((mag_bonus, q[0], q[1], prox), cand))
        if not scored:
            return ""
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _magnitude_repair_candidates(
        self, raw: str, ref_med: float
    ) -> List[str]:
        """
        生成量级修复候选：拍屏常把 410 读成 110/-110、550 读成 050。
        """
        out = [raw]
        s = raw.strip().lstrip("-")
        if ref_med < 50:
            return out
        # 3 位整数丢百位：110 → 410
        if re.fullmatch(r"\d{3}", s):
            lead = int(ref_med) // 100
            if 1 <= lead <= 9:
                repaired = f"{lead}{s[1:]}"
                if repaired != s:
                    out.append(repaired)
            # 也试邻近百位
            for d in (lead - 1, lead + 1, 4, 5, 3):
                if 1 <= d <= 9:
                    cand = f"{d}{s[1:]}"
                    if cand not in out:
                        out.append(cand)
        # 2 位：55 → 550（电力行）
        if re.fullmatch(r"\d{2}", s) and ref_med >= 200:
            cand = s + "0"
            if cand not in out:
                out.append(cand)
            lead = int(ref_med) // 100
            if 1 <= lead <= 9:
                cand2 = f"{lead}{s}"
                if cand2 not in out:
                    out.append(cand2)
        return out

    def _override_tail_by_cell_ocr(self, img: np.ndarray, row: List[str],
                                   col_centers: List[float],
                                   data_cells: List[OcrCell],
                                   img_width: int,
                                   tail: int = 4) -> List[str]:
        """
        对行尾若干列强制逐格 OCR 覆盖（修正顺序填充导致的错位）。

        @param img         缩放后的表格图
        @param row         当前数据行
        @param col_centers 列中心
        @param data_cells  主数据行 OCR 块
        @param img_width   图像宽度
        @param tail        覆盖列数（含 23:30/23:45/24:00）
        @return 修正后的行
        """
        if not row or not data_cells:
            return row
        ys = [c.y for c in data_cells] + [c.y + c.height for c in data_cells]
        y0 = int(max(0, min(ys) - 10))
        y1 = int(min(img.shape[0], max(ys) + 10))
        bounds = self._column_bounds(col_centers, img_width)
        start = max(0, len(row) - tail)
        refined = list(row)
        for col in range(start, len(row)):
            # 只补空格，不覆盖已按 X 对齐的值（避免后列读数前移）
            if refined[col]:
                continue
            x0, x1 = bounds[col]
            val, found_x = self._ocr_single_cell_located(
                img, x0, y0, x1, y1, fast=False
            )
            if not val:
                continue
            if found_x is not None and len(col_centers) > 1:
                nearest = min(
                    range(len(col_centers)),
                    key=lambda j: abs(found_x - col_centers[j]),
                )
                if nearest != col:
                    continue
            refined[col] = val
        return refined

    def _refine_tail_columns(self, img: np.ndarray, row: List[str],
                             col_centers: List[float],
                             data_cells: List[OcrCell],
                             tail: int = 6,
                             img_width: int = 0) -> List[str]:
        """
        对行尾若干列逐格 OCR，修正顺序填充时的尾部缺漏。

        @param img         缩放后的表格图
        @param row         当前数据行
        @param col_centers 列中心
        @param data_cells  主数据行 OCR 块
        @param tail        重识列数
        @param img_width   图像宽度
        @return 修正后的行
        """
        if not row or not data_cells:
            return row
        ys = [c.y for c in data_cells] + [c.y + c.height for c in data_cells]
        y0 = int(max(0, min(ys) - 8))
        y1 = int(min(img.shape[0], max(ys) + 8))
        bounds = self._column_bounds(col_centers, img_width or img.shape[1])
        start = max(0, len(row) - tail)
        refined = list(row)
        for col in range(start, len(row)):
            if refined[col]:
                continue
            x0, x1 = bounds[col]
            val = self._ocr_single_cell(img, x0, y0, x1, y1)
            if val:
                refined[col] = val
        return refined

    def _column_bounds(self, col_centers: List[float],
                       img_width: int = 0) -> List[Tuple[int, int]]:
        """
        计算每列左右边界（最后一列延伸至图像右缘）。

        @param col_centers 列中心 X
        @param img_width   图像宽度，用于最后一列右边界
        @return (left, right) 列表
        """
        n = len(col_centers)
        bounds: List[Tuple[int, int]] = []
        for i in range(n):
            if i == 0:
                left = col_centers[i] - (col_centers[1] - col_centers[0]) / 2 if n > 1 else col_centers[i] - 30
            else:
                left = (col_centers[i - 1] + col_centers[i]) / 2
            if i == n - 1:
                step = (col_centers[i] - col_centers[i - 1]) if n > 1 else 60
                right = img_width - 1 if img_width > 0 else col_centers[i] + step / 2
            else:
                right = (col_centers[i] + col_centers[i + 1]) / 2
            bounds.append((int(left), int(right)))
        return bounds

    def _normalize_cell_value(self, text: str) -> str:
        """
        规范化单元格 OCR 文本。

        @param text 原始文本
        @return 数值字符串或空
        """
        if not text or not text.strip():
            return ""
        # 页脚「显示第1到第3条记录」禁止拆成 133 等伪数
        if self._is_footer_text(text):
            return ""
        t = text.strip().replace("O", "0").replace("o", "0")
        # 统一各类短横线 / 负号字符，便于后续判断
        t = (
            t.replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("﹣", "-")
            .replace("－", "-")
            .replace("­", "-")
        )
        # 仅「整格只有 -」= 无数据置空；切勿误伤 -2.3 / -0.5 等负数
        if t == "-" or (t and set(t) <= {"-"} and not re.search(r"\d", t)):
            return ""
        if t in ("0", "D", "U", "U."):
            return "0"
        # 负号可能落在首位非 ASCII 连字符后：再剥一层
        negative = False
        if t.startswith("-") and len(t) > 1 and bool(re.search(r"\d", t)):
            negative = True
        elif re.match(r"^[\u2212\uFF0D\uFE63\-]\s*\d", t):
            negative = True
            t = "-" + re.sub(r"^[\u2212\uFF0D\uFE63\-]\s*", "", t)
        digits = re.sub(r"[^\d.]", "", t)
        if digits and re.fullmatch(r"\d+\.?\d*", digits):
            if ":" not in text and "：" not in text:
                fixed = self._fix_decimal_ocr(digits)
                if negative and fixed and fixed != "0":
                    return "-" + fixed.lstrip("-")
                return fixed
            # 24:00:00 等时间格式不当数值
            if self._normalize_time(text):
                return ""
        if self._normalize_time(t):
            return ""
        parts = self._split_cell_values(t)
        return parts[0] if parts else ""

    def _fill_sparse_data_row(self, rows: List[List[str]],
                              data_cells: List[OcrCell],
                              col_centers: List[float],
                              col_count: int) -> List[List[str]]:
        """
        用主数据行 OCR 结果回填矩阵最后一行的空单元格。

        主行须像末行；若主行更像上方指标，禁止把第 2 行值填进第 3 行空格。
        """
        if not rows or not data_cells:
            return rows
        ref = self._align_row_to_columns(
            self._expand_row_cells(data_cells), col_centers, col_count)
        if not ref:
            return rows
        last = list(rows[-1]) + [""] * max(0, col_count - len(rows[-1]))
        last = last[:col_count]
        last_n = sum(1 for v in last if v and str(v).strip())
        same_as_last = last_n >= 2 and self._aligned_rows_same_series(last, ref)
        matches_upper = False
        for upper in rows[:-1]:
            upper_n = sum(1 for v in upper if v and str(v).strip())
            if upper_n >= 3 and self._aligned_rows_same_series(upper, ref):
                matches_upper = True
                break
        if matches_upper and not same_as_last:
            return rows
        if last_n >= 4 and not same_as_last:
            return rows
        for i in range(col_count):
            if not last[i] and ref[i]:
                last[i] = ref[i]
        rows[-1] = last
        return rows

    def _postprocess_rows(self, rows: List[List[str]]) -> List[List[str]]:
        """
        后处理：修正 OCR 误识别、合并孤立小数片段。

        @param rows 原始矩阵
        @return 修正后矩阵
        """
        result: List[List[str]] = []
        for row in rows:
            fixed: List[str] = []
            for val in row:
                raw = str(val).strip() if val is not None else ""
                # 整格只有「-」或已标记无数据：保留标记，跳过后续单列重识
                if raw == self.NODATA_MARK or self._is_lone_dash(raw):
                    fixed.append(self.NODATA_MARK)
                    continue
                v = self._fix_decimal_ocr(val) if val else ""
                if v in ("D", "U", "U.", "O", "o"):
                    v = "0"
                if v.replace("−", "-").replace("–", "-").replace("—", "-") == "-":
                    v = ""
                if v and v != "0":
                    v = v.replace("−", "-").replace("–", "-")
                    # 保留前导负号，只清其它杂质
                    neg = v.startswith("-")
                    body = re.sub(r"[^\d.]", "", v.lstrip("-"))
                    v = ("-" + body) if neg and body else body
                    if (
                        re.fullmatch(r"-?\d+\.?\d*", v if not neg else "-" + body)
                        and "." not in body
                        and len(body) >= 5
                    ):
                        v = self._fix_decimal_ocr(("-" if neg else "") + body)
                fixed.append(v)
            result.append(fixed)
        return [self._apply_decimal_row_repairs(r) for r in result]

    def _apply_decimal_row_repairs(self, row: List[str]) -> List[str]:
        """行级小数点补救：5 位、4 位都按同行锚点在两种插点里选优。"""
        if not self.fix_decimal_enabled:
            return row
        row = self._repair_five_digit_lost_dots(row)
        return self._repair_price_row_lost_dots(row)

    def _parse_dotted_cell_value(self, raw: str) -> Optional[float]:
        """解析已带点的数值格；失败返回 None。"""
        s = str(raw or "").strip().replace("−", "-").replace("–", "-").replace("—", "-")
        if not s or s == "-":
            return None
        if not re.fullmatch(r"-?\d{1,3}\.\d{1,3}", s):
            return None
        try:
            return abs(float(s))
        except ValueError:
            return None

    def _row_fix_decimal_anchors(
        self, row: Sequence[str], col: int
    ) -> List[float]:
        """
        5 位选优锚点：左右邻带点格 + 同行带点中位数 + 小电价簇中位数。

        邻格只用紧邻（允许真实 15 分钟跳变）；中位数提供全局稳健兜底。
        """
        anchors: List[float] = []
        n = len(row)
        if 0 <= col - 1 < n:
            left = self._parse_dotted_cell_value(str(row[col - 1]))
            if left is not None:
                anchors.append(left)
        if 0 <= col + 1 < n:
            right = self._parse_dotted_cell_value(str(row[col + 1]))
            if right is not None:
                anchors.append(right)

        dotted_all: List[float] = []
        dotted_small: List[float] = []
        for j, cell in enumerate(row):
            if j == col:
                continue
            fv = self._parse_dotted_cell_value(str(cell))
            if fv is None:
                continue
            dotted_all.append(fv)
            if fv < 1000.0:
                dotted_small.append(fv)
        if len(dotted_all) >= 1:
            anchors.append(float(np.median(dotted_all)))
        if len(dotted_small) >= 1:
            anchors.append(float(np.median(dotted_small)))
        return anchors

    def _row_frac_digit_mode(self, row: Sequence[str], skip_col: int = -1) -> int:
        """同行已带点格小数位数众数；默认 3（现货电价常见）。"""
        counts = {2: 0, 3: 0}
        for j, cell in enumerate(row):
            if j == skip_col:
                continue
            s = str(cell or "").strip()
            if "." not in s:
                continue
            if not re.fullmatch(r"-?\d+\.\d+", s.replace("−", "-").replace("–", "-")):
                continue
            frac = s.rsplit(".", 1)[-1]
            n = len(re.sub(r"\D", "", frac))
            if n in counts:
                counts[n] += 1
        if counts[2] == 0 and counts[3] == 0:
            return 3
        return 3 if counts[3] >= counts[2] else 2

    def _row_looks_like_price(self, row: Sequence[str]) -> bool:
        """小数占比高 → 更像电价行（无锚点时 5 位兜底用）。"""
        parsed = 0
        dotted = 0
        for cell in row:
            raw = str(cell or "").strip().replace("−", "-").replace("–", "-")
            if not raw or raw == "-":
                continue
            try:
                float(raw)
            except ValueError:
                continue
            parsed += 1
            if "." in raw:
                dotted += 1
        return parsed >= 3 and dotted / max(parsed, 1) >= 0.4

    @staticmethod
    def _rel_err_to_anchors(value: float, anchors: Sequence[float]) -> float:
        """相对最近锚点的误差；无锚点时为 +inf。"""
        if not anchors:
            return float("inf")
        best = float("inf")
        for a in anchors:
            denom = max(abs(a), 1e-9)
            best = min(best, abs(value - a) / denom)
        return best

    def _pick_lost_dot_candidate(
        self,
        candidates: Sequence[str],
        row: Sequence[str],
        col: int,
        sign: str = "",
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        """
        无点双候选选优。

        得分 = 相对所有锚点的 min 相对误差（抗 15 分钟跳变）。
        最优 > ratio_k → 拒补；两候选接近 → 按小数位数众数破平，仍平则拒补。
        无锚点：仅当调用方给出 fallback 且本行像电价时用 fallback，否则拒补。
        """
        anchors = self._row_fix_decimal_anchors(row, col)
        k = self.fix_decimal_ratio_k
        tie_eps = self.fix_decimal_tie_eps

        if not anchors:
            if fallback and self._row_looks_like_price(row):
                return f"{sign}{fallback}"
            return None

        scored: List[Tuple[float, str, int]] = []
        for cand in candidates:
            try:
                fv = abs(float(cand))
            except ValueError:
                continue
            frac_n = len(cand.split(".", 1)[1]) if "." in cand else 0
            scored.append((self._rel_err_to_anchors(fv, anchors), cand, frac_n))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        best_r, best_c, _ = scored[0]
        if best_r > k:
            return None
        if len(scored) >= 2 and abs(scored[0][0] - scored[1][0]) < tie_eps:
            prefer = self._row_frac_digit_mode(row, skip_col=col)
            for r, cand, frac_n in scored:
                if r <= k and frac_n == prefer:
                    return f"{sign}{cand}"
            return None
        return f"{sign}{best_c}"

    def _pick_five_digit_fix(
        self,
        digits: str,
        row: Sequence[str],
        col: int,
        sign: str = "",
    ) -> Optional[str]:
        """5 位无点双候选：xx.xxx / xxx.xx。无锚点时电价行默认 xx.xxx。"""
        if not re.fullmatch(r"\d{5}", digits):
            return None
        cand_xx = f"{digits[:2]}.{digits[2:]}"   # xx.xxx
        cand_xxx = f"{digits[:3]}.{digits[3:]}"  # xxx.xx
        return self._pick_lost_dot_candidate(
            (cand_xx, cand_xxx), row, col, sign=sign, fallback=cand_xx
        )

    def _pick_four_digit_fix(
        self,
        digits: str,
        row: Sequence[str],
        col: int,
        sign: str = "",
    ) -> Optional[str]:
        """4 位无点双候选：xx.xx / x.xxx。无锚点时不猜，保留原文。"""
        if not re.fullmatch(r"\d{4}", digits):
            return None
        cand_xx = f"{digits[:2]}.{digits[2:]}"  # xx.xx
        cand_x = f"{digits[:1]}.{digits[1:]}"   # x.xxx
        return self._pick_lost_dot_candidate(
            (cand_xx, cand_x), row, col, sign=sign, fallback=None
        )

    def _repair_five_digit_lost_dots(self, row: List[str]) -> List[str]:
        """
        5 位无点漏小数点补救：双候选 + 邻格/同行选优。

        73713 → 在同行像 73.x 时选 73.713，而非旧规则的 737.13。
        两候选都不像则保留原文。
        """
        if not self.fix_decimal_enabled or not row:
            return row
        digit5_idxs: List[int] = []
        for i, cell in enumerate(row):
            raw = str(cell or "").strip().replace("−", "-").replace("–", "-")
            if not raw or raw == "-":
                continue
            body = raw[1:] if raw.startswith("-") and len(raw) > 1 else raw
            if re.fullmatch(r"\d{5}", body):
                digit5_idxs.append(i)
        if not digit5_idxs:
            return row
        # 锚点一律看原始行，避免同批多个 5 位格互相污染
        out = list(row)
        for i in digit5_idxs:
            raw = str(row[i]).strip().replace("−", "-").replace("–", "-")
            sign = "-" if raw.startswith("-") else ""
            body = raw[1:] if sign else raw
            picked = self._pick_five_digit_fix(body, row, i, sign=sign)
            if picked:
                out[i] = picked
        return out

    def _repair_price_row_lost_dots(self, row: List[str]) -> List[str]:
        """
        4 位无点漏小数点补救：双候选 + 邻格/同行选优。

        1205 在同行像 15.x 时选 12.05，在同行像 3.x 时选 1.205。
        两候选都不像则保留原文。锚点看原始行，避免同批多个 4 位格互相污染。
        """
        if not self.fix_decimal_enabled or not row:
            return row
        digit4_idxs: List[int] = []
        for i, cell in enumerate(row):
            raw = str(cell or "").strip().replace("−", "-").replace("–", "-")
            if not raw or raw == "-":
                continue
            body = raw[1:] if raw.startswith("-") and len(raw) > 1 else raw
            if re.fullmatch(r"\d{4}", body):
                digit4_idxs.append(i)
        if not digit4_idxs:
            return row
        out = list(row)
        for i in digit4_idxs:
            raw = str(row[i]).strip().replace("−", "-").replace("–", "-")
            sign = "-" if raw.startswith("-") else ""
            body = raw[1:] if sign else raw
            picked = self._pick_four_digit_fix(body, row, i, sign=sign)
            if picked:
                out[i] = picked
        return out

    def _price_row_small_stats(
        self, row: Sequence[str]
    ) -> Tuple[List[float], float, float]:
        """电价行小量级簇：返回 (small_vals, small_med, med_int_digits)。"""
        small: List[float] = []
        int_lens: List[int] = []
        for cell in row:
            raw = str(cell or "").strip().replace("−", "-").replace("–", "-")
            if not raw or "." not in raw:
                continue
            try:
                fv = abs(float(raw))
            except ValueError:
                continue
            # 整数 1～3 位（<1000）都是正常电价；333.48 不得被排除在参照之外
            if fv >= 1000.0:
                continue
            small.append(fv)
            body = raw.lstrip("-").split(".", 1)[0]
            int_lens.append(len(re.sub(r"\D", "", body)))
        if not small:
            return [], 0.0, 3.0
        return (
            small,
            float(np.median(small)),
            float(np.median(int_lens)) if int_lens else 2.0,
        )

    def _is_suspicious_price_cell(
        self,
        row: Sequence[str],
        idx: int,
        small_med: float,
        med_int: float,
    ) -> bool:
        """电价行可疑格：整数≥4 位或总位数≥7 才重识。1～3 位整数的已识别结果保留。"""
        del small_med, med_int
        raw = str(row[idx] or "").strip().replace("−", "-").replace("–", "-")
        if not raw or raw == "-":
            return False
        body = raw[1:] if raw.startswith("-") and len(raw) > 1 else raw
        digits = re.sub(r"\D", "", body)
        # 已有小数点：187.34、333.48 这种整数 1～3 位直接保留。
        # 整数部分 ≥4 位，或连小数一共 ≥7 位，才是粘连风险，按列重识。
        if "." in body:
            int_digits = re.sub(r"\D", "", body.split(".", 1)[0])
            if len(int_digits) >= 4 or len(digits) >= 7:
                return True
            return False
        # 无点整数：
        # - ≥7 位：粘连，列重识
        # - 5/6 位：交给行级补点，不在这里清空
        # - 4 位：预览像电价则交给 xx.xx / x.xxx 选优，否则重识
        # - 1～3 位：正常整数部分（可以是 333），不因「通常只有两位」而重识
        if not re.fullmatch(r"-?\d+", raw):
            return False
        if raw in ("0", "-0"):
            return False
        if len(digits) >= 7:
            return True
        if len(digits) >= 5:
            return False
        if len(digits) == 4:
            pv = self._preview_sticky_price(raw)
            if pv is not None and 0.5 <= pv <= 999.0:
                return False
            return True
        return False

    def _blank_cross_column_bleed(self, row: List[str]) -> List[str]:
        """已停用：串位成对置空过严；按「识别到什么写什么」保留 OCR 原文。"""
        return row

    def _trim_trailing_empty_and_broken_headers(
        self,
        headers: List[str],
        rows: List[List[str]],
    ) -> Tuple[List[str], List[List[str]]]:
        """
        去掉尾部全空列，并保留可桥接的严格 Δ=15 表头段。

        旧逻辑在「首次非 +15」处截断：纠偏后若左缘误读插入乱序时刻，会只剩 1 列
        → 候选被判 invalid、有效点变 0。现跳过脏列；两端因误读断开时都保留。
        """
        if not headers:
            return headers, rows
        n = len(headers)
        rows = [list(r) + [""] * max(0, n - len(r)) for r in (rows or [])]
        rows = [r[:n] for r in rows]

        def _filled(j: int) -> int:
            return sum(
                1
                for r in rows
                if j < len(r)
                and str(r[j]).strip()
                not in ("", "-", "—", "–", "−", self.NODATA_MARK)
            )

        # 从右去掉全空列
        last = n - 1
        while last >= 0 and _filled(last) == 0:
            last -= 1
        if last < 0:
            # 整表数据空（屏上全 -）：保留时刻表头，数据保持空，供未出清逻辑使用
            return list(headers), [list(r) for r in rows]
        headers = headers[: last + 1]
        rows = [r[: last + 1] for r in rows]

        keep = self._longest_delta15_header_indices(headers)
        if len(keep) >= 2 and keep != list(range(len(headers))):
            headers = [headers[i] for i in keep]
            rows = [[r[i] for i in keep] for r in rows]
        elif len(keep) == 1:
            # 构不成链时仍尽量丢掉首列非法，避免整表被废
            headers = [headers[keep[0]]]
            rows = [[r[keep[0]]] for r in rows]

        # 再清一次尾空
        while headers and _filled(len(headers) - 1) == 0:
            headers = headers[:-1]
            rows = [r[: len(headers)] for r in rows]
        return headers, rows

    def _count_numeric_cells(self, matrix: Sequence[Sequence[str]]) -> int:
        """统计矩阵中可解析为浮点数的非空格数。"""
        n = 0
        for row in matrix:
            for cell in row:
                if not cell or not str(cell).strip():
                    continue
                try:
                    float(str(cell).strip())
                except ValueError:
                    continue
                n += 1
        return n

    def _split_cell_values(self, text: str) -> List[str]:
        """
        拆分粘连数值（支持小数、整数与负号）。

        超长「假小数」（如 28478281451265311272.539 实为多格粘连）拆失败则返回空，
        由上层留空，禁止把怪物数当成单格写入。
        """
        t = text.strip().replace("O", "0").replace("o", "0").replace(":", ".")
        t = t.replace("−", "-").replace("–", "-").replace("—", "-")
        t = re.sub(r"[^\d.\-]", "", t)
        if not t or re.fullmatch(r"[.\-]+", t):
            return []

        sticky0 = re.sub(r"[.\-]", "", t)

        def _ok_parts(parts: List[str]) -> List[str]:
            fixed = [self._fix_decimal_ocr(p) for p in parts if p]
            # 任一 token 仍是怪物 → 整段拒写
            if any(self._is_monster_numeric_token(p) for p in fixed):
                return []
            return [p for p in fixed if p]

        # 优先拆成多段「短整数.1~3位小数」（电价常见；两点齐全可安全拆）
        multi_dec = re.findall(r"-?\d{1,3}\.\d{1,3}", t)
        if len(multi_dec) >= 2:
            digits_only = re.sub(r"[.\-]", "", t)
            covered = "".join(re.sub(r"-", "", p) for p in multi_dec).replace(".", "")
            if covered == digits_only or len(covered) >= len(digits_only) - 1:
                got = _ok_parts(multi_dec)
                if got:
                    return got

        # 带符号小数：-3.979 / 334.347-3.979；拒绝「整数部过长」的假单小数
        signed_decimals = re.findall(r"-?\d+\.\d{1,3}", t)
        usable_dec = [
            p
            for p in signed_decimals
            if len(re.sub(r"[^\d]", "", p.split(".", 1)[0])) <= 3
        ]
        digits_only = re.sub(r"[.\-]", "", t)
        if usable_dec:
            covered = "".join(re.sub(r"-", "", p) for p in usable_dec).replace(".", "")
            if covered == digits_only:
                return _ok_parts(usable_dec)

        leading_neg = t.startswith("-")
        sticky = re.sub(r"[.\-]", "", t)
        # ≥7 位：粘连风险。8~11 位只试均衡 4~6|4~6，失败或 7 位都不 DP 猜拆
        if len(sticky) >= 7 and t.count(".") <= 1:
            if 8 <= len(sticky) <= 11:
                paired = self._split_sticky_two_cells(sticky)
                if paired:
                    fixed = _ok_parts(paired)
                    if fixed and self._sticky_pair_plausible(fixed):
                        if leading_neg and fixed:
                            fixed[0] = "-" + fixed[0].lstrip("-")
                        return fixed
            return []

        if len(sticky) >= 6:
            parts = self._dp_split_digits(sticky)
            if parts and not any(len(p) > 6 for p in parts):
                fixed = _ok_parts(parts)
                if fixed and (
                    len(fixed) == 1 or self._sticky_pair_plausible(fixed)
                ):
                    if leading_neg and fixed:
                        fixed[0] = "-" + fixed[0].lstrip("-")
                    return fixed
            if len(sticky) >= 8:
                return []

        parts = re.findall(r"-?\d{1,6}\.\d{1,3}|-?\d{1,6}", t)
        got = _ok_parts(parts)
        if got:
            return got
        if self._is_monster_numeric_token(t):
            return []
        return []

    def _split_sticky_two_cells(self, digits: str) -> List[str]:
        """
        8~11 位纯数字两格粘连：枚举均衡切分（每段 4~6 位）。

        典型：1939619307 → 19396|19307（再经补点 → 19.396 / 19.307）。
        拒绝 193|969307 这类 3+6。
        """
        n = len(digits)
        if n < 8 or n > 11 or not digits.isdigit():
            return []
        cands: List[Tuple[int, List[str]]] = []
        for left in range(4, 7):
            right = n - left
            if right < 4 or right > 6:
                continue
            parts = [digits[:left], digits[left:]]
            # 长度分：5 最优，其次 6/4；惩罚左右长度差
            score = 0
            for p in parts:
                score += {5: 8, 6: 6, 4: 5}.get(len(p), 0)
            score -= abs(left - right) * 3
            previews = [self._preview_sticky_price(p) for p in parts]
            if any(v is None for v in previews):
                continue
            assert previews[0] is not None and previews[1] is not None
            a, b = float(previews[0]), float(previews[1])
            # 单格电价整数最多 3 位，预览到千位视为切分失败
            if a >= 1000.0 or b >= 1000.0:
                continue
            if a < 0.05 or b < 0.05:
                continue
            # 邻格量级接近加分（19.396↔19.307）
            ratio = max(a, b) / max(min(a, b), 1e-9)
            if ratio <= 2.0:
                score += 6
            elif ratio <= 4.0:
                score += 2
            else:
                score -= 4
            cands.append((score, parts, ratio))
        if not cands:
            return []
        cands.sort(key=lambda x: x[0], reverse=True)
        _score, best_parts, best_ratio = cands[0]
        # 9 位缺字（19.396+19.307→193969307）：右段常丢首位 1
        if (
            best_ratio > 2.0
            and n == 9
            and len(best_parts[0]) == 5
            and len(best_parts[1]) == 4
        ):
            left, right = best_parts
            pv_l = self._preview_sticky_price(left)
            trial = "1" + right
            pv_t = self._preview_sticky_price(trial)
            if (
                pv_l is not None
                and pv_t is not None
                and abs(pv_t - pv_l) / max(pv_l, 1e-9) <= 0.35
            ):
                return [left, trial]
            return [left]
        if best_ratio > 2.0:
            if len(best_parts[0]) >= len(best_parts[1]):
                return [best_parts[0]]
            return [best_parts[1]]
        return best_parts

    def _preview_sticky_price(self, tok: str) -> Optional[float]:
        """粘连段量级预览（含漏点常见形态），用于切分选优。"""
        s = self._fix_decimal_ocr(tok) or str(tok).strip()
        if not s:
            return None
        if "." in s:
            try:
                return abs(float(s))
            except ValueError:
                return None
        d = re.sub(r"\D", "", s)
        try:
            if len(d) == 6:
                return abs(float(d[:3] + "." + d[3:]))
            if len(d) == 5:
                return abs(float(d[:2] + "." + d[2:]))  # xx.xxx
            if len(d) == 4:
                return abs(float(d[0] + "." + d[1:]))  # d.ddd
            if 1 <= len(d) <= 3:
                return abs(float(d))
        except ValueError:
            return None
        return None

    def _sticky_pair_plausible(self, parts: Sequence[str]) -> bool:
        """粘连拆出的多段是否像正常表体（挡 193 + 969.307）。"""
        if len(parts) < 2:
            return True
        vals: List[float] = []
        for p in parts:
            pv = self._preview_sticky_price(str(p))
            if pv is None:
                return False
            if pv >= 1000.0:
                return False
            vals.append(pv)
        # 一段是无点短整数、另一段预览已很大 → 典型误拆
        for i, p in enumerate(parts):
            raw = str(p).strip()
            if "." not in raw and re.fullmatch(r"-?\d{1,3}", raw):
                others = [vals[j] for j in range(len(vals)) if j != i]
                if others and max(others) >= 200.0:
                    return False
        return True

    @staticmethod
    def _is_monster_numeric_token(text: str) -> bool:
        """明显粘连/误读的超长数值：不应落库。"""
        s = str(text).strip().replace(",", "")
        if not s:
            return False
        body = s[1:] if s.startswith("-") else s
        digits = re.sub(r"\D", "", body)
        if len(digits) >= 10:
            return True
        if "." in body:
            int_part = body.split(".", 1)[0]
            if len(re.sub(r"\D", "", int_part)) >= 7:
                return True
        try:
            v = abs(float(body))
        except ValueError:
            return len(digits) >= 8
        # 现货功率/电价量级；极端尖峰也不该到千万
        return v >= 1_000_000.0

    def _dp_split_digits(self, digits: str) -> List[str]:
        """
        动态规划拆分纯数字粘连串。
        优先 6/5/4/3 位段，尽量避免拆出孤立个位数（拍屏漏点后常见）。

        @param digits 纯数字串
        @return 拆分结果
        """

        def valid_token(tok: str) -> bool:
            if "." in tok:
                return bool(re.fullmatch(r"\d+\.\d{1,3}", tok))
            return 1 <= len(tok) <= 6

        def token_score(tok: str) -> int:
            n = len(tok)
            # 现货数值多为 3~6 位（含小数修复前），惩罚 1 位
            if n == 6:
                return 8
            if n == 5:
                return 7
            if n == 4:
                return 5
            if n == 3:
                return 4
            if n == 2:
                return 2
            return -3

        n = len(digits)
        best: List[Optional[List[str]]] = [None] * (n + 1)
        best_score = [-10 ** 9] * (n + 1)
        best[0] = []
        best_score[0] = 0
        for i in range(n):
            if best[i] is None:
                continue
            for length in (6, 5, 4, 3, 2, 1):
                if i + length > n:
                    continue
                tok = digits[i:i + length]
                if not valid_token(tok):
                    continue
                cand = best[i] + [tok]
                score = best_score[i] + token_score(tok)
                j = i + length
                if best[j] is None or score > best_score[j] or (
                    score == best_score[j] and len(cand) < len(best[j] or [])
                ):
                    best[j] = cand
                    best_score[j] = score
        return best[n] if best[n] else []

    def _count_numeric_tokens(self, text: str) -> int:
        """
        统计文本中数值 token 数量。

        @param text OCR 文本
        @return 数量
        """
        return len(self._split_cell_values(text))

    def _drop_singleton_noise(self, tokens: List[str]) -> List[str]:
        """
        在多数为多位数的序列中，去掉孤立个位噪声（粘连拆分副作用）。

        @param tokens 数值字符串列表
        @return 过滤后列表
        """
        if len(tokens) < 4:
            return tokens
        vals: List[float] = []
        for t in tokens:
            try:
                vals.append(abs(float(t)))
            except ValueError:
                vals.append(0.0)
        median = float(np.median(vals)) if vals else 0.0
        if median < 8:
            return tokens
        kept: List[str] = []
        for t, v in zip(tokens, vals):
            # 个位数且远小于行中位数 → 视为拆分噪声
            if re.fullmatch(r"\d", t) and v < median * 0.08:
                continue
            kept.append(t)
        return kept if kept else tokens

    def _fix_decimal_ocr(self, num: str) -> str:
        """修正丢失小数点的数字（保留前导负号）。

        仅对「位数齐全、整段无点」做单格可判定的格式恢复：
        - 6 位 → xxx.xxx（整数部分 3 位）
        - ≥7 位不删字猜点（粘连风险，交给列重识）
        - 5 位不在此猜点（交给 _repair_five_digit_lost_dots 同行选优）
        - 4 位及以下保持原文（4 位由 _repair_price_row_lost_dots 在 xx.xx / x.xxx 间选优）
        fix_decimal.enabled=false 时只做符号规范化，不猜补点。
        """
        if not num:
            return ""
        num = str(num).strip().replace("−", "-").replace("–", "-").replace("—", "-")
        sign = ""
        if num.startswith("-"):
            sign = "-"
            num = num[1:]
        num = num.rstrip("-")
        if not num:
            return ""
        if not re.fullmatch(r"\d+(\.\d+)?", num):
            return ""
        if "." in num:
            return sign + num
        if num == "0":
            return sign + "0" if sign else "0"
        # 关闭补点：保留 OCR 无点原文，便于评估引擎认点能力
        if not self.fix_decimal_enabled:
            return sign + num
        # 6 位：xxx.xxx（281451 → 281.451），整数部分仍是 3 位
        if re.fullmatch(r"\d{6}", num):
            return sign + num[:-3] + "." + num[-3:]
        # ≥7 位不删字猜点，留给粘连列重识
        # 5 位及以下：单格不猜；5 位行级选优，4 位电价行补点
        return sign + num

    def _row_is_mostly_zeros(self, row: List[str]) -> bool:
        """
        判断一行是否几乎全为 0。

        @param row 行数据
        @return 是否零行
        """
        if not row:
            return True
        zeros = sum(
            1 for v in row if v in ("", "0", self.NODATA_MARK)
        )
        return zeros >= len(row) * 0.85

    def _is_footer_text(self, text: str) -> bool:
        """判断页脚文字。"""
        keywords = (
            "条记录", "显示第", "总共", "认证", "AOC", "2026-", "CHS", "现货",
            "一键全流程", "截图", "写入 PI", "Excel", "JSON", "对照", "pi_code",
            "time_label", "timestamp", "points",
        )
        return any(k in text for k in keywords)

    def _is_valid_data_row(self, row: List[str]) -> bool:
        """判断是否有效数据行。"""
        numeric = sum(1 for v in row if v and re.fullmatch(r"-?\d+(\.\d+)?", v))
        return numeric >= 1

    def _normalize_time(self, text: str) -> Optional[str]:
        """
        规范化时间为 HH:MM（支持 HH:mm:ss；纯 3 位整数不算时间）。

        @param text 原始 OCR 文本
        @return 时间字符串或 None
        """
        raw = text.strip().replace("O", "0").replace("o", "0")
        raw = raw.replace("l", "1").replace("I", "1").replace("|", "1")
        has_sep = ":" in raw or "：" in raw
        t = raw.replace("：", ":").replace("．", ".").replace(";", ":")
        digits_only = re.sub(r"\D", "", t)

        # 400/410/395 等 3 位整数是表体数值，不是 04:00
        if not has_sep and len(digits_only) == 3 and re.fullmatch(r"\d{3}", digits_only):
            return None

        def _quarter(hour: int, minute: int) -> Optional[str]:
            if minute >= 60 or hour > 24:
                return None
            if hour == 24 and minute == 0:
                return "24:00"
            # OCR 常把 15:15→15:16、16:45→16:46：吸附到最近 15 分钟刻度
            total = hour * 60 + minute
            snapped = self._snap_minutes_to_quarter(total, max_delta=2)
            sh, sm = snapped // 60, snapped % 60
            if sm not in (0, 15, 30, 45) and not (sh == 24 and sm == 0):
                return None
            if sh > 24 or (sh == 24 and sm != 0):
                return None
            return "%02d:%02d" % (sh, sm)

        # 24:00:00 / 23:45:00 等带秒格式
        sec_m = self.TIME_WITH_SEC_PATTERN.match(t)
        if sec_m:
            hour, minute, second = int(sec_m.group(1)), int(sec_m.group(2)), int(sec_m.group(3))
            if second < 60:
                got = _quarter(hour, minute)
                if got:
                    return got

        # 240000 → 24:00
        if not has_sep and len(digits_only) == 6:
            hour = int(digits_only[:2])
            minute = int(digits_only[2:4])
            second = int(digits_only[4:6])
            if second < 60:
                got = _quarter(hour, minute)
                if got:
                    return got

        m = self.TIME_PATTERN.match(t)
        if not m:
            if len(digits_only) == 4:
                return _quarter(int(digits_only[:2]), int(digits_only[2:]))
            return None
        if not has_sep and len(digits_only) != 4:
            return None
        return _quarter(int(m.group(1)), int(m.group(2)))

    def _write_csv(self, result: TableExtractResult, out: Path) -> None:
        """写入 CSV。"""
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result.headers)
            for row in result.rows:
                padded = row + [""] * (len(result.headers) - len(row))
                writer.writerow(padded[: len(result.headers)])

    def _write_txt(self, result: TableExtractResult, out: Path) -> None:
        """写入 TXT。"""
        lines = ["\t".join(result.headers)]
        for row in result.rows:
            padded = row + [""] * (len(result.headers) - len(row))
            lines.append("\t".join(padded[: len(result.headers)]))
        out.write_text("\n".join(lines), encoding="utf-8")
