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
        sanitize: Optional[Dict[str, Any]] = None,
        max_ocr_width: Optional[int] = None,
        onnx_threads: Optional[int] = None,
    ):
        """
        @param scale          识别前放大倍数，宽表建议 2.5
        @param min_confidence OCR 最低置信度
        @param sanitize       同行量级清洗配置（见 config.yaml ocr.sanitize）
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
            nt = int(onnx_threads) if onnx_threads is not None else 2
        except (TypeError, ValueError):
            nt = 2
        self.onnx_threads = max(1, min(nt, os.cpu_count() or 4))
        raw = sanitize if isinstance(sanitize, dict) else {}
        self.sanitize_enabled = bool(raw.get("enabled", True))
        try:
            self.sanitize_ratio_k = max(1.0, float(raw.get("ratio_k", 3.0)))
        except (TypeError, ValueError):
            self.sanitize_ratio_k = 3.0
        try:
            self.sanitize_min_samples = max(2, int(raw.get("min_samples", 4)))
        except (TypeError, ValueError):
            self.sanitize_min_samples = 4
        # 位数众数相关配置已废弃，保留属性以免旧调用报错
        self.sanitize_majority_ratio = 0.5
        self.sanitize_boundary_pct = 0.15
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
        近水平霍夫线段中位数角度（度）。
        OpenCV 4 返回 (N,1,4)，OpenCV 5 返回 (N,4)，统一 reshape。
        """
        h, w = img.shape[:2]
        if h < 20 or w < 40:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_len = max(int(w * 0.12), 40)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(40, w // 40),
            minLineLength=min_len,
            maxLineGap=max(8, w // 80),
        )
        if lines is None:
            return None
        # OpenCV 5+: (N,4)；旧版: (N,1,4)
        pts = np.asarray(lines).reshape(-1, 4)
        angles: List[float] = []
        for x1, y1, x2, y2 in pts:
            dx = float(x2) - float(x1)
            dy = float(y2) - float(y1)
            if abs(dx) < 1e-3:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            if abs(ang) > 25.0:
                continue
            angles.append(ang)
        if len(angles) < 5:
            return None
        return float(np.median(angles))

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
        """按倾角 θ 反向旋转拉平（OpenCV 正角为逆时针）。"""
        h, w = img.shape[:2]
        if abs(angle_deg) < 1e-6:
            return img
        center = (w * 0.5, h * 0.5)
        m = cv2.getRotationMatrix2D(center, -float(angle_deg), 1.0)
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
        测得倾角 θ 时旋转 -θ 拉平；纠后残角须改善，否则丢弃（宁可不转）。
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
        if before is not None and after is not None:
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

        header_idx, headers, col_centers = self._extract_header_info(rows)
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
                rows, data_cells, img.shape[1]
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
        if matrix and data_cells:
            aligned = self._align_row_to_columns(
                self._expand_row_cells(data_cells), col_centers, len(headers)
            )
            if aligned and sum(1 for v in aligned if v) >= max(
                4, sum(1 for v in matrix[-1] if v) // 2
            ):
                matrix[-1] = [
                    self._fix_decimal_ocr(v) if v else "" for v in aligned
                ]
        # 先按 X 回填空格，再量级清理；清理掉的噪声格再局部 OCR（避免 410→110 被清后永远空着）
        matrix = self._fill_sparse_data_row(matrix, data_cells, col_centers, col_count)
        matrix = [self._sanitize_data_row(r) for r in matrix]
        matrix = self._refill_matrix_empties(
            img, matrix, rows, header_idx, data_idx, data_cells, col_centers
        )
        # 电价行可疑已填格（漏点/串位）按列强制重识，可覆盖错值
        matrix = self._reocr_suspicious_price_cells(
            img, matrix, rows, header_idx, data_idx, data_cells, col_centers
        )
        ocr_filled = self._count_numeric_cells(matrix)
        matrix = [self._sanitize_data_row(r) for r in matrix]
        headers, matrix = self._trim_trailing_empty_and_broken_headers(
            headers, matrix
        )
        # 斜拍/碎片兜底：数据行硬封顶为 3（表头时刻行已剥离）
        matrix = self._enforce_expected_data_rows(
            matrix, expected=self.EXPECTED_DATA_ROWS
        )
        # 电力/计划（整型）与电价（小数）混行时拆开
        matrix = self._untangle_integer_decimal_rows(matrix)
        # 电价行漏点补救（5/6 位已在 _fix_decimal_ocr；此处补 4 位 d.ddd）
        matrix = [self._repair_price_row_lost_dots(r) for r in matrix]
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
        x0 = max(0, x - pad_x)
        # 搜索从 50% 起；表头可略越过，但不得低于 48%（避免曲线）
        y0 = max(int(h * 0.48), y - pad_y_top)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y_bot)
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

    def _preprocess_variants(self, img: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        生成预处理图像，尽量保住小数点笔画。

        通道：原图 → CLAHE/反色 →（浅色表）轻度放大锐化。
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

    def _row_time_candidates(
        self, row: Sequence[OcrCell]
    ) -> List[Tuple[str, float]]:
        times: List[Tuple[str, float]] = []
        for cell in row:
            norm = self._normalize_time(cell.text)
            if norm:
                times.append((norm, cell.x + cell.width / 2))
        return times

    def _extract_header_info(self, rows: Sequence[List[OcrCell]]) -> Tuple[int, List[str], List[float]]:
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
            times = self._row_time_candidates(row)
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

        # 单行不足时：只合并 Y 很近的邻行（避免把曲线轴时刻并进表头）
        if 1 <= len(best_times) < 4 and best_idx >= 0:
            base_y = self._row_y_center(rows[best_idx])
            y_tol = 28.0
            if rows[best_idx]:
                y_tol = max(
                    28.0,
                    float(np.median([c.height for c in rows[best_idx]])) * 1.6,
                )
            merged: List[Tuple[str, float]] = list(best_times)
            for idx in range(max(0, best_idx - 1), min(len(rows), best_idx + 2)):
                if idx == best_idx:
                    continue
                if abs(self._row_y_center(rows[idx]) - base_y) > y_tol:
                    continue
                merged.extend(self._row_time_candidates(rows[idx]))
            merged.sort(key=lambda t: t[1])
            dedup: List[Tuple[str, float]] = []
            for t, x in merged:
                if dedup and abs(x - dedup[-1][1]) < 25:
                    continue
                dedup.append((t, x))
            if len(dedup) > len(best_times):
                best_times = dedup

        if len(best_times) < 4:
            return -1, [], []
        best_times.sort(key=lambda t: t[1])  # 严格按列 X
        # 同 X 去重
        dedup2: List[Tuple[str, float]] = []
        for t, x in best_times:
            if dedup2 and abs(x - dedup2[-1][1]) < 22:
                continue
            dedup2.append((t, x))
        headers = [t[0] for t in dedup2]
        col_centers = [t[1] for t in dedup2]
        headers, col_centers = self._sanitize_header_sequence(headers, col_centers)
        if len(headers) < 4:
            return -1, [], []
        return best_idx, headers, col_centers

    def _longest_delta15_header_indices(
        self,
        headers: Sequence[str],
        col_centers: Optional[Sequence[float]] = None,
    ) -> List[int]:
        """
        在按 X 排列的时刻列中，找最长的严格相邻 Δ=15 子序列下标。

        纠偏后偶发「左缘误读 16:30 插在 15:15 与 16:00 之间」：若只按前缀截断
        会只剩 1 列；同分钟去重若保留左侧误读会挤掉右侧真列。子序列可跳过脏列。
        """
        n = len(headers)
        if n <= 0:
            return []
        order = list(range(n))
        if col_centers is not None and len(col_centers) >= n:
            order.sort(key=lambda i: float(col_centers[i]))
        mins: List[Optional[int]] = [self._time_to_minutes(headers[i]) for i in order]
        # DP：chain[i] = 以 order[i] 结尾的最长 Δ=15 链
        length = [1] * n
        prev = [-1] * n
        for i in range(n):
            mi = mins[i]
            if mi is None:
                continue
            for j in range(i):
                mj = mins[j]
                if mj is None:
                    continue
                if mi - mj != 15:
                    continue
                if length[j] + 1 > length[i]:
                    length[i] = length[j] + 1
                    prev[i] = j
        best = max(range(n), key=lambda i: (length[i], i))
        if length[best] < 2:
            # 构不成链：退回全部合法刻度下标（保持原序）
            return [order[i] for i in range(n) if mins[i] is not None]
        chain_pos: List[int] = []
        cur = best
        while cur >= 0:
            chain_pos.append(cur)
            cur = prev[cur]
        chain_pos.reverse()
        return [order[p] for p in chain_pos]

    def _sanitize_header_sequence(
        self,
        headers: List[str],
        col_centers: List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        清洗表头时刻序列（仍按 OCR 原文，不发明时刻）：

        1. 丢掉左侧邻屏漏进来的乱序头（如 03:15 出现在 00:15 之前）
        2. 丢掉与主簇 X 间距过大的左缘孤列
        3. 取最长严格 Δ=15 子序列（跳过误读/乱序列；同分钟保留链上那一次）
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
        keep = self._longest_delta15_header_indices(labs, xs2)
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
    ) -> Tuple[List[str], List[float]]:
        """表头行弱时：汇总偏下方、且步长像 15 分钟的时刻（不编造模板）。"""
        del data_cells, img_width
        # 优先取图像下半的行，减少曲线轴
        scored_rows: List[Tuple[float, List[Tuple[str, float]]]] = []
        max_y = 1.0
        for row in rows:
            yc = self._row_y_center(row)
            if yc > max_y:
                max_y = yc
        for row in rows:
            times = self._row_time_candidates(row)
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
                times_all.extend(self._row_time_candidates(row))
            times_all.sort(key=lambda t: t[1])
            dedup0: List[Tuple[str, float]] = []
            for t, x in times_all:
                if dedup0 and abs(x - dedup0[-1][1]) < 25:
                    continue
                dedup0.append((t, x))
            if len(dedup0) < 4:
                return [], []
            return self._sanitize_header_sequence(
                [t for t, _ in dedup0], [x for _, x in dedup0]
            )

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        times = scored_rows[0][1]
        times.sort(key=lambda t: t[1])
        dedup: List[Tuple[str, float]] = []
        for t, x in times:
            if dedup and abs(x - dedup[-1][1]) < 25:
                continue
            dedup.append((t, x))
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
            token_n = 0
            magnitude = 0.0
            for c in row:
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
            best_idx = len(rows) - 1
            best_cells = rows[-1]
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
            same_series = (
                base_mag >= 20
                and row_mag >= 20
                and abs(row_mag - base_mag) / max(base_mag, 1.0) <= 0.22
            )
            if idx == data_idx or (near_y and numeric_ratio >= 0.5 and same_series):
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
                prev = dedup[-1]
                prefer_new = ("." in cell.text and "." not in prev.text) or (
                    len(cell.text) > len(prev.text)
                    and cell.confidence >= prev.confidence * 0.9
                )
                if prefer_new:
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

        @param row 原始行 OCR 块
        @return 拆分后的块列表
        """
        result: List[OcrCell] = []
        for cell in row:
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
        """本图裁切半列窄带：仅贴齐画布边缘（双屏已切开，不再用大半列防邻屏）。"""
        if img_width <= 0:
            return max(8.0, step_x * 0.15)
        return max(8.0, min(img_width * 0.01, step_x * 0.18))

    def _drop_edge_clipped_columns(
        self,
        headers: List[str],
        col_centers: List[float],
        img_width: int,
    ) -> Tuple[List[str], List[float]]:
        """丢掉本图边缘被裁断的半列。

        只丢：中心已出画布，或落入极窄贴边带（表头几乎贴齐裁切线）。
        靠外缘但中心仍在图内的完整列保留——不再用「半列理论越界」砍合法末列。
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
            if token_n < 2 and idx != data_idx:
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
                            prev = dedup[-1]
                            if ("." in cell.text and "." not in prev.text) or len(
                                cell.text
                            ) > len(prev.text):
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
                matrix.append(["0"] * col_count)
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
                        matrix[i] = main_aligned
                        replaced = True
                        break
                if not replaced:
                    for i, row in enumerate(matrix):
                        if self._aligned_rows_similar(row, main_aligned):
                            if sum(1 for v in main_aligned if v) >= sum(
                                1 for v in row if v
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
        硬约束：数据行至多 expected 条（默认 3）。
        只做已有的安全互补去重 + 丢掉最弱噪声行；禁止臆测拼行。
        斜拍拆行应靠 OCR 前纠偏解决，而不是事后合并。
        """
        if len(matrix) <= expected:
            return matrix
        # 仅保留 _dedupe 已认可的互补残片合并结果；此处不再二次激进合并
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
        return [matrix[i] for i in keep]

    def _untangle_integer_decimal_rows(
        self, matrix: List[List[str]]
    ) -> List[List[str]]:
        """
        拆开混行的电力/计划（整型）与电价（小数）。

        业务固定 3 行：前两行应为整数功率/计划，末行电价带小数。
        斜拍/粘连时小数会渗进第 2 行、整数渗进第 3 行。
        """
        if not matrix or len(matrix) < 2:
            return matrix
        out = [list(r) for r in matrix]
        ncols = max(len(r) for r in out)
        out = [r + [""] * (ncols - len(r)) for r in out]

        def _dec_ratio(row: Sequence[str]) -> float:
            filled = [str(c).strip() for c in row if c and str(c).strip()]
            if not filled:
                return 0.0
            dec = sum(1 for c in filled if "." in c)
            return dec / len(filled)

        def _looks_price(cell: str) -> bool:
            s = str(cell or "").strip().replace("−", "-")
            if not s:
                return False
            if "." in s:
                return True
            body = s[1:] if s.startswith("-") else s
            return bool(re.fullmatch(r"\d{5,6}", body))

        def _looks_power_int(cell: str) -> bool:
            s = str(cell or "").strip()
            return bool(re.fullmatch(r"-?\d{2,4}", s)) and "." not in s

        ratios = [_dec_ratio(r) for r in out]
        price_i = max(range(len(out)), key=lambda i: (ratios[i], i))
        # 小数行沉底
        if price_i != len(out) - 1 and ratios[price_i] >= 0.25:
            out[price_i], out[-1] = out[-1], out[price_i]
            ratios[price_i], ratios[-1] = ratios[-1], ratios[price_i]

        # 逐列重分：小数归电价，整型归上方行；电价行已有小数优先保留（避免污染值盖掉真电价）
        price = list(out[-1])
        int_rows = [list(r) for r in out[:-1]]
        for ci in range(ncols):
            leaked_prices: List[str] = []
            int_pool: List[str] = []
            for row in int_rows:
                cur = row[ci] if ci < len(row) else ""
                s = str(cur).strip() if cur else ""
                if not s:
                    continue
                if _looks_price(s):
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
                    cur = row[ci] if ci < len(row) else ""
                    if cur and str(cur).strip():
                        continue
                    row[ci] = (
                        int_pool[ri] if len(int_pool) > 1 and ri < len(int_pool)
                        else int_pool[0]
                    )

        out = int_rows + [price]
        return out

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

    def _align_row_to_columns(self, row: List[OcrCell],
                              col_centers: List[float],
                              col_count: int) -> List[str]:
        """
        将 OCR 行按 X 坐标对齐到列网格。

        绝不「从左到右顺序填列」：中间漏识必须留空，不能把右侧数值前移。
        粘连多值块优先按「块覆盖的列中心」落列，不再只按宽度均分。
        """
        if not row or col_count <= 0 or not col_centers:
            return []

        values = [""] * col_count
        qualities = [(-1, 0, 0.0)] * col_count
        step = self._median_col_step(col_centers)
        max_dist = max(step * 0.60, 28.0)

        for cell in self._filter_overlap_noise_cells(row):
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
        n = len(parts)

        def _put(col: int, val: str) -> None:
            if col < 0 or col >= col_count:
                return
            if self._is_monster_numeric_token(val):
                return
            q = self._token_quality(val, cell.confidence)
            if not values[col] or q > qualities[col]:
                values[col] = val
                qualities[col] = q

        if n == 1:
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
        # 从末行往前（电价行最易粘连漏识）
        order = list(range(len(out) - 1, -1, -1))
        for ri in order:
            row = out[ri]
            empty_n = sum(1 for v in row if not v)
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
            )
        return out

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
            if self._is_suspicious_price_cell(row, i, small_med, med_int)
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
        # 先置空再走局部 OCR（强制覆盖）
        cleared = list(row)
        for i in sus:
            cleared[i] = ""
        filled = self._ocr_fill_empty_cells(
            img,
            cleared,
            col_centers,
            band,
            img.shape[1],
            prefer_sharp=True,
        )
        # 重识失败则保留原值
        for i in sus:
            if filled[i]:
                row[i] = filled[i]
        out[ri] = row
        return out

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
            for item in result:
                box, text, score = item[0], str(item[1]).strip(), float(item[2])
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
    ) -> List[str]:
        """
        对数据行空单元格局部 OCR 补全。

        紧裁优先；若检出文本中心更靠近邻列中心，则丢弃（防后列前移补位）。
        prefer_sharp：电价行空格优先用非 fast（锐化/二值）利于小数点。
        """
        if not row or not data_cells:
            return row
        empty_cols = [i for i, v in enumerate(row) if not v]
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

        @param rows        当前矩阵
        @param data_cells  主数据行 OCR
        @param col_centers 列中心
        @param col_count   列数
        @return 补全后的矩阵
        """
        if not rows or not data_cells:
            return rows
        ref = self._align_row_to_columns(
            self._expand_row_cells(data_cells), col_centers, col_count)
        if not ref:
            return rows
        target = rows[-1]
        for i in range(col_count):
            if not target[i] and ref[i]:
                target[i] = ref[i]
        rows[-1] = target
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
                # 仅整格横杠=无数据；-2.3 等负数保留
                if raw.replace("−", "-").replace("–", "-").replace("—", "-") == "-":
                    fixed.append("")
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
        return [
            self._repair_price_row_lost_dots(self._sanitize_data_row(r))
            for r in result
        ]

    def _repair_price_row_lost_dots(self, row: List[str]) -> List[str]:
        """
        电价行漏小数点补救。

        4 位无点整数 → d.ddd（3262→3.262、5308→5.308）。
        量级校验优先用「小电价簇」(abs<80)，避免同行大电价把中位拉高导致小值补点失败。
        """
        if not row or len(row) < 3:
            return row
        dotted_all: List[float] = []
        dotted_small: List[float] = []
        digit4_idxs: List[int] = []
        for i, cell in enumerate(row):
            raw = str(cell or "").strip().replace("−", "-").replace("–", "-")
            if not raw or raw == "-":
                continue
            if re.fullmatch(r"-?\d{1,3}\.\d{2,3}", raw):
                try:
                    fv = abs(float(raw))
                except ValueError:
                    continue
                dotted_all.append(fv)
                if fv < 80.0:
                    dotted_small.append(fv)
            elif re.fullmatch(r"-?\d{4}", raw):
                digit4_idxs.append(i)
        if not digit4_idxs:
            return row
        # 小簇至少 1 个、或全体带点至少 2 个
        if len(dotted_small) < 1 and len(dotted_all) < 2:
            return row
        out = list(row)
        for i in digit4_idxs:
            raw = str(out[i]).strip().replace("−", "-").replace("–", "-")
            sign = "-" if raw.startswith("-") else ""
            body = raw[1:] if sign else raw
            fixed = f"{sign}{body[0]}.{body[1:]}"
            try:
                fv = abs(float(fixed))
            except ValueError:
                continue
            ok = False
            if dotted_small and fv < 80.0:
                med = float(np.median(dotted_small))
                if med > 0 and fv <= max(50.0, med * 10) and fv >= min(0.05, med / 10):
                    ok = True
            if not ok and len(dotted_all) >= 2:
                med = float(np.median(dotted_all))
                if med > 0 and not (fv > med * 8 or fv < med / 8):
                    ok = True
                elif 0.05 <= fv <= 50 and 0.05 <= med <= 50:
                    ok = True
            # 小电价行常见：有任一小簇锚点且补点结果本身像电价
            if not ok and dotted_small and 0.05 <= fv <= 50:
                ok = True
            if ok:
                out[i] = fixed
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
            if fv >= 80.0:
                continue
            small.append(fv)
            body = raw.lstrip("-").split(".", 1)[0]
            int_lens.append(len(re.sub(r"\D", "", body)))
        if not small:
            return [], 0.0, 2.0
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
        """电价行可疑格：漏点 4 位、或整数部过长疑似吃了邻列。"""
        raw = str(row[idx] or "").strip().replace("−", "-").replace("–", "-")
        if not raw or raw == "-":
            return False
        if re.fullmatch(r"-?\d{4}", raw):
            return True
        if "." not in raw:
            return False
        try:
            fv = abs(float(raw))
        except ValueError:
            return False
        body = raw.lstrip("-")
        int_part, _, frac = body.partition(".")
        int_digits = re.sub(r"\D", "", int_part)
        if len(int_digits) < 3 or fv < 80.0:
            return False
        # 去掉首位后更接近小电价簇 → 像串位多吃一位（758.378→58.378）
        if small_med > 0 and len(int_digits) >= 3:
            try:
                dropped = float(int_digits[1:] + (("." + frac) if frac else ""))
            except ValueError:
                dropped = -1.0
            if dropped > 0 and abs(dropped - small_med) * 4 < abs(fv - small_med):
                if fv > small_med * 8:
                    return True
        if med_int <= 2 and len(int_digits) >= med_int + 1 and fv > 100:
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
        去掉尾部全空列，并只保留最长的严格 Δ=15 表头子序列。

        旧逻辑在「首次非 +15」处截断：纠偏后若左缘误读插入乱序时刻，会只剩 1 列
        → 候选被判 invalid、有效点变 0。改为跳过脏列取最长合法链。
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
                if j < len(r) and str(r[j]).strip() not in ("", "-", "—", "–", "−")
            )

        # 从右去掉全空列
        last = n - 1
        while last >= 0 and _filled(last) == 0:
            last -= 1
        if last < 0:
            return [], []
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

    @staticmethod
    def _integer_digits(v: float) -> int:
        """绝对值的整数位数（0→1；0.x→1）。"""
        x = abs(float(v))
        if x < 1.0:
            return 1
        return int(math.floor(math.log10(x))) + 1

    def _near_power10_boundary(self, v: float) -> bool:
        """是否靠近 10^k（位数边界，如 99↔100），交给辅规则而非位数硬删。"""
        x = abs(float(v))
        if x <= 0:
            return False
        k = int(round(math.log10(x)))
        boundary = 10.0 ** k
        if boundary <= 0:
            return False
        return abs(x - boundary) / boundary <= self.sanitize_boundary_pct

    def _sanitize_data_row(self, row: List[str]) -> List[str]:
        """开启时对整数行为比值清洗；电价行跳过。异常格只置空，不猜修。"""
        cleaned = [
            ("" if (cell and self._is_monster_numeric_token(cell)) else cell)
            for cell in row
        ]
        if not self.sanitize_enabled:
            return cleaned
        return self._sanitize_integer_peer_row(cleaned)

    def _sanitize_integer_peer_row(self, row: List[str]) -> List[str]:
        """
        仅清洗「整数为主」的电力/计划行：相对同行中位数超出 ratio_k → 置空。

        处置说明：
        - 已去掉「位数众数」硬删：该规则假定同行同整数位数，电价尖峰
          （3.x↔88.x）必然误杀，规则本身不合理。
        - 电力行漏位/多位（28 / 2800）用比值即可覆盖，不必再按位数众数删。
        - 小数为主的电价行跨数量级是业务正常，整行跳过清洗。
        - OCR 明确负号不做量级置空。
        """
        parsed: List[Tuple[int, float, bool]] = []
        dotted = 0
        for i, cell in enumerate(row):
            if not cell or not str(cell).strip():
                continue
            raw = str(cell).strip().replace("−", "-").replace("–", "-")
            try:
                keep_signed = raw.startswith("-") and len(raw) > 1
                parsed.append((i, abs(float(raw)), keep_signed))
                if "." in raw:
                    dotted += 1
            except ValueError:
                continue
        if len(parsed) < self.sanitize_min_samples:
            return row
        # 电价行：小数占比高 → 不清洗
        if dotted / max(len(parsed), 1) >= 0.4:
            return row

        peers = [v for _, v, _ in parsed]
        med = float(np.median(peers))
        if med <= 0:
            return row

        k = self.sanitize_ratio_k
        out = list(row)
        for i, v, keep_signed in parsed:
            if keep_signed:
                continue
            if v > med * k or v < med / k:
                out[i] = ""
        return out

    # 兼容旧名
    def _sanitize_digit_mode_row(self, row: List[str]) -> List[str]:
        return self._sanitize_integer_peer_row(row)

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
        # 单小数点 + 超长整数部：多格粘连 → 先 DP 拆，拆出多段则保留（再补点）
        force_dp = False
        if t.count(".") == 1 and len(sticky0) >= 10:
            int_part = t.split(".", 1)[0].lstrip("-")
            if len(re.sub(r"\D", "", int_part)) >= 8:
                force_dp = True

        def _ok_parts(parts: List[str]) -> List[str]:
            fixed = [self._fix_decimal_ocr(p) for p in parts if p]
            # 任一 token 仍是怪物 → 整段拒写
            if any(self._is_monster_numeric_token(p) for p in fixed):
                return []
            return [p for p in fixed if p]

        if force_dp:
            leading_neg = t.startswith("-")
            parts = self._dp_split_digits(sticky0)
            if parts and len(parts) >= 2 and not any(len(p) > 6 for p in parts):
                fixed = _ok_parts(parts)
                if fixed:
                    if leading_neg and fixed:
                        fixed[0] = "-" + fixed[0].lstrip("-")
                    return fixed
            # 拆不出合理多段：整段拒写（避免怪物单值）
            return []

        # 优先拆成多段「短整数.1~3位小数」（电价常见）
        multi_dec = re.findall(r"-?\d{1,4}\.\d{1,3}", t)
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
            if len(re.sub(r"[^\d]", "", p.split(".", 1)[0])) <= 4
        ]
        digits_only = re.sub(r"[.\-]", "", t)
        if usable_dec:
            covered = "".join(re.sub(r"-", "", p) for p in usable_dec).replace(".", "")
            if covered == digits_only:
                return _ok_parts(usable_dec)

        leading_neg = t.startswith("-")
        sticky = re.sub(r"[.\-]", "", t)
        if len(sticky) >= 6:
            parts = self._dp_split_digits(sticky)
            if parts and not any(len(p) > 6 for p in parts):
                fixed = _ok_parts(parts)
                if fixed:
                    if leading_neg and fixed:
                        fixed[0] = "-" + fixed[0].lstrip("-")
                    return fixed
            # 粘连过长且拆不好 → 拒写（避免整坨进一格）
            if len(sticky) >= 10:
                return []

        parts = re.findall(r"-?\d{1,6}\.\d{1,3}|-?\d{1,6}", t)
        got = _ok_parts(parts)
        if got:
            return got
        # 仍只得到怪物单值
        if self._is_monster_numeric_token(t):
            return []
        return []

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

        仅对「位数齐全、整段无点」做格式恢复（6→xxx.xxx、5 位按量级），
        不把残缺串硬凑成小数。
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
        # 6 位：xxx.xxx（281451 → 281.451）
        if re.fullmatch(r"\d{6}", num):
            return sign + num[:-3] + "." + num[-3:]
        # 7 位：尝试去首/尾后按 6 位
        if re.fullmatch(r"\d{7}", num):
            for cand in (num[1:], num[:-1]):
                fixed = cand[:3] + "." + cand[3:]
                try:
                    fv = float(fixed)
                except ValueError:
                    continue
                if 50 <= fv <= 999:
                    return sign + fixed
        # 5 位：百位整数优先 xxx.xx，否则 xx.xxx
        if re.fullmatch(r"\d{5}", num):
            head2 = int(num[:2])
            head3 = int(num[:3])
            if head3 >= 200:
                return sign + num[:3] + "." + num[3:]
            if head2 <= 25:
                return sign + num[:2] + "." + num[2:]
            return sign + num[:3] + "." + num[3:]
        # 4 位及以下：保持原文，不猜点
        return sign + num

    def _row_is_mostly_zeros(self, row: List[str]) -> bool:
        """
        判断一行是否几乎全为 0。

        @param row 行数据
        @return 是否零行
        """
        if not row:
            return True
        zeros = sum(1 for v in row if v in ("", "0"))
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
