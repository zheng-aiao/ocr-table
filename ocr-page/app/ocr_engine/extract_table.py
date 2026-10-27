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
import time
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
    # 非空表示该块来自业务层粘连处理（拆分/待重识），供落格高亮
    sticky: str = ""


@dataclass
class TableExtractResult:
    """表格提取结果。"""

    headers: List[str]
    rows: List[List[str]]
    raw_row_count: int
    image_path: str
    # 最终清洗前可解析填格数（OCR 识别数）；清洗后采纳数由下游 points 统计
    ocr_filled: int = 0
    # 与 rows 同形：格内为粘连处理原因（粘连拆分 / 粘连重识），未处理为空串
    sticky_map: Optional[List[List[str]]] = None
    sticky_filled: int = 0


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
    # 蓝带失败时回退整图（相对 table_auto 输入：整帧或搜索窗）
    DEFAULT_TABLE_CROP_RATIO: Tuple[float, float, float, float] = (
        0.0,
        0.0,
        1.0,
        1.0,
    )
    # 回退原因文案（联调/详情页提示用）
    CROP_FALLBACK_REASONS: Dict[str, str] = {
        "no_blue": "未检测到浅蓝表体，已回退原图",
        "too_short": "自动框过矮，已回退原图",
        "clipped_top": "蓝带贴搜索窗顶（表被先验窗裁断），已回退原图",
        "low_blue_quality": "蓝带不像有效表体（空表/页脚干扰），已回退原图",
        "ocr_invalid_auto_crop": "自动定位未识别到有效时刻表（本轮结束，请下一轮重试）",
        # 弹窗蒙板压暗：不提亮硬识，本轮直接失败，等关弹窗后再抓
        "dimmed_modal": "画面被弹窗蒙板压暗，本轮 auto 失败（请确认弹窗后重试）",
        # 显示器黑框 OSD（如信号超出范围）：遮挡表体，本轮不识
        "black_osd": "画面存在显示器黑框 OSD（如信号超出范围），本轮不识别",
    }
    # 本轮直接空结果、不跑纠偏/OCR 的裁剪失败码（与弹窗同类）
    CROP_ABORT_REASONS = frozenset({"dimmed_modal", "black_osd"})
    # 出清弹窗：有弹窗必有灰蒙板 → auto 快速失败（左右同一套规则）。
    # content/mean 门槛按侧区分：右表白底很亮，左 search 正常已偏暗，不能共用一个 content。
    # - 右正常 content≈210、灰≈0.06；右蒙板≈110、灰≈0.50
    # - 左正常 search content≈145~151、灰≈0.24、无弹窗白纸；左蒙板≈130~135、灰≈0.40
    # （白弹窗抬高 p75，故不用 p75 判暗）
    MODAL_POPUP_WHITE_MIN = 0.015
    MODAL_POPUP_WHITE_MAX = 0.12
    MODAL_POPUP_CONTENT_V_MAX = 165.0
    # side → mean_max / content_max / grey_min（可被 cameras.*.modal_dim 覆盖）
    MODAL_DIM_BY_SIDE: Dict[str, Dict[str, float]] = {
        "left": {"mean_max": 140.0, "content_max": 140.0, "grey_min": 0.33},
        "right": {"mean_max": 140.0, "content_max": 160.0, "grey_min": 0.33},
        # 未传 side 时偏保守（接近右：正常很亮才不误杀左正常）
        "": {"mean_max": 140.0, "content_max": 140.0, "grey_min": 0.33},
    }
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
    # 透视估计略去顶部细边（白底表头），避免把表头当蓝带四边形
    DESKEW_QUAD_IGNORE_TOP_RATIO = 0.08
    # 角点贴图边：蓝带铺满裁剪框时 approx 会粘在右/底边，拉正只会拧歪
    DESKEW_QUAD_BORDER_TOL_PX = 4
    DESKEW_QUAD_MAX_BORDER_CORNERS = 1
    # 拍屏摩尔纹常产生大量真水平短线，霍夫中位角被钉在 0°；剔掉后再取中位
    DESKEW_HOUGH_MOIRE_ABS_DEG = 0.35
    DESKEW_HOUGH_MOIRE_MIN_LINES = 12
    DESKEW_HOUGH_MOIRE_MIN_WEIGHT_RATIO = 0.15

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
        @param scale          识别前参考放大倍数（相对约 1600px 宽表）；大图会少放
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
        # 表格区域定位失败并回退整图时置位（供联调/详情页提示）
        self.last_crop_fallback: bool = False
        self.last_crop_fallback_reason: str = ""
        # 摄像头侧 left|right：蒙板 content 门槛按侧；可由上游写入
        self.camera_side: str = ""
        # 可选覆盖 MODAL_DIM_BY_SIDE 单项：mean_max / content_max / grey_min
        self.modal_dim_cfg: Dict[str, float] = {}
        # 最近一次 extract 粗粒度耗时（毫秒）：region/data + 重点子步骤
        # region: crop, deskew；data: ocr, cell_fix, recover
        self.last_step_ms: Dict[str, int] = {}

    def _mark_ms(self, name: str, t0: float) -> float:
        """记一步耗时到 last_step_ms，返回新计时起点（不打日志）。"""
        self.last_step_ms[str(name)] = int((time.perf_counter() - t0) * 1000)
        return time.perf_counter()

    def _modal_dim_params(self) -> Dict[str, float]:
        """当前侧的蒙板判定门槛（默认表 ∪ cameras.*.modal_dim）。"""
        side = str(getattr(self, "camera_side", "") or "").strip().lower()
        if side not in ("left", "right"):
            side = ""
        base = dict(self.MODAL_DIM_BY_SIDE.get(side) or self.MODAL_DIM_BY_SIDE[""])
        cfg = getattr(self, "modal_dim_cfg", None) or {}
        if isinstance(cfg, dict):
            for key in ("mean_max", "content_max", "grey_min"):
                if key not in cfg or cfg[key] is None:
                    continue
                try:
                    base[key] = float(cfg[key])
                except (TypeError, ValueError):
                    pass
        return base

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
        """
        识别前缩放：以约 1600px 宽表为参考，按 scale 得到期望宽度，再受 max_ocr_width 封顶。

        已够宽（如 2K 裁表）则少放或不放——盲目 ×2 + CUBIC 会放大拍屏摩尔纹，
        且多通道全图 OCR 耗时近似随像素线性恶化。
        """
        scale = float(self.scale or 1.0)
        if scale <= 0:
            scale = 1.0
        h, w = img.shape[:2]
        if w <= 0:
            return img
        max_w = float(self.max_ocr_width or self.MAX_OCR_WIDTH)
        # 中等表参考宽；期望宽 = min(封顶, 参考×scale)
        ref_w = 1600.0
        desired_w = min(max_w, ref_w * scale)
        # 已接近期望宽：不放。避免 2113→2160 这类无意义微放大再拧纹
        if float(w) >= desired_w * 0.95:
            return img
        actual = min(scale, desired_w / float(w), max_w / float(w))
        if actual <= 1.05:
            return img
        # LINEAR 轻放大；CUBIC 更易把竖纹/摩尔纹拧得更尖
        return cv2.resize(
            img, None, fx=actual, fy=actual, interpolation=cv2.INTER_LINEAR
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

    @staticmethod
    def _nonzero_xy(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        cv2.findNonZero 兼容：OpenCV 4 多为 (N,1,2)，部分环境是 (N,2)。
        """
        arr = np.asarray(coords)
        if arr.ndim == 3:
            return arr[:, 0, 0], arr[:, 0, 1]
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 0], arr[:, 1]
        raise ValueError(f"unexpected findNonZero shape: {arr.shape}")

    def _quad_border_corner_count(
        self, src: np.ndarray, w: int, h: int
    ) -> int:
        """统计四边形有多少角点贴在图像边界上。"""
        tol = float(self.DESKEW_QUAD_BORDER_TOL_PX)
        n = 0
        for p in np.asarray(src, dtype=np.float32).reshape(4, 2):
            if (
                p[0] <= tol
                or p[0] >= float(w - 1) - tol
                or p[1] <= tol
                or p[1] >= float(h - 1) - tol
            ):
                n += 1
        return n

    @staticmethod
    def _weighted_median_angle(
        angles: Sequence[float], weights: Sequence[float]
    ) -> Optional[float]:
        if len(angles) < 8:
            return None
        order = np.argsort(np.asarray(angles, dtype=np.float64))
        ang_s = np.asarray(angles, dtype=np.float64)[order]
        w_s = np.asarray(weights, dtype=np.float64)[order]
        cum = np.cumsum(w_s)
        if cum[-1] <= 1e-6:
            return None
        idx = int(np.searchsorted(cum, cum[-1] * 0.5))
        idx = min(max(idx, 0), len(ang_s) - 1)
        return float(ang_s[idx])

    def _estimate_hough_median_angle(self, img: np.ndarray) -> Optional[float]:
        """
        近水平霍夫线段的长度加权中位角（度）。
        拍屏网格偏淡：高阈值 Canny 经常一条线都没有，验收就会放行假旋转。
        高低两档边缘合并，并放宽最短线长。
        摩尔纹会产生大量真水平线把中位角钉在 0°：若剔掉近零线后仍有足够证据，改用剔后中位。
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
        full = self._weighted_median_angle(angles, weights)
        if full is None:
            return None
        # 摩尔纹：近零线占主导时，用 |θ|≥阈值 的子集
        moire = float(self.DESKEW_HOUGH_MOIRE_ABS_DEG)
        filt_a = [a for a in angles if abs(a) >= moire]
        filt_w = [w for a, w in zip(angles, weights) if abs(a) >= moire]
        if (
            len(filt_a) >= self.DESKEW_HOUGH_MOIRE_MIN_LINES
            and abs(full) < self.DESKEW_MIN_ANGLE_DEG
        ):
            w_all = float(sum(weights)) + 1e-6
            w_filt = float(sum(filt_w))
            if w_filt / w_all >= self.DESKEW_HOUGH_MOIRE_MIN_WEIGHT_RATIO:
                filtered = self._weighted_median_angle(filt_a, filt_w)
                if (
                    filtered is not None
                    and abs(filtered) >= self.DESKEW_MIN_ANGLE_DEG
                ):
                    return filtered
        return full

    def _table_illumination_stats(
        self, img: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        表格区曝光统计：mean / p75 / 非纯黑区 mean。

        显示器 OSD、边框等大块暗区会拉低全局 mean，但表体本身可能已够亮；
        此时若仍按 mean 做全局增益，会把白底表打爆（V→255、S→0），蓝带丢光。
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mean_v = float(np.mean(v))
        p75_v = float(np.percentile(v, 75))
        bright = v >= 50
        if int(np.count_nonzero(bright)) >= max(100, int(v.size * 0.08)):
            content_v = float(np.mean(v[bright]))
        else:
            content_v = mean_v
        return mean_v, p75_v, content_v

    def _modal_mask_features(
        self, img: np.ndarray
    ) -> Tuple[float, float]:
        """
        蒙板特征：灰蒙占比、最大低饱和亮白连通域占比（出清弹窗白纸）。

        正常亮表白底连通域常 >0.3；弹窗白纸约 0.02~0.08；无弹窗 ≈0。
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        grey_frac = float(np.mean((v >= 70) & (v < 155)))
        white = ((v >= 200) & (s <= 40)).astype(np.uint8) * 255
        nlab, _labels, stats, _ = cv2.connectedComponentsWithStats(
            white, connectivity=8
        )
        max_white = 0.0
        if nlab > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_white = float(np.max(areas)) / float(max(int(v.size), 1))
        return grey_frac, max_white

    def _is_modal_dimmed(self, img: np.ndarray) -> bool:
        """
        出清弹窗蒙板：有弹窗必有蒙板 → 表区发灰，auto 不可靠。

        规则左右一致；content/mean 门槛按 camera_side 区分，避免误杀偏暗的左正常图。
        1) 中块白纸弹窗 + 大面积灰蒙（主信号）
        2) 整页压暗（弹窗落在搜索窗外时的兜底，门槛按侧）
        """
        if img is None or getattr(img, "size", 0) <= 0:
            return False
        params = self._modal_dim_params()
        mean_max = float(params.get("mean_max", 140.0))
        content_max = float(params.get("content_max", 140.0))
        grey_min = float(params.get("grey_min", 0.33))
        mean_v, _p75_v, content_v = self._table_illumination_stats(img)
        grey_frac, max_white = self._modal_mask_features(img)
        # 弹窗白纸 + 灰蒙板（白块抬高 content 时仍能抓到）
        if (
            float(self.MODAL_POPUP_WHITE_MIN)
            <= max_white
            <= float(self.MODAL_POPUP_WHITE_MAX)
            and grey_frac >= grey_min
            and content_v < float(self.MODAL_POPUP_CONTENT_V_MAX)
        ):
            return True
        # 整页压暗、无明显亮表白底（左右 content 门槛不同）
        return mean_v < mean_max and content_v < content_max

    def _has_black_osd_box(self, img: np.ndarray) -> bool:
        """
        显示器黑框 OSD（如「输入信号超出范围」）覆在表上：本轮应跳过。

        与出清白纸弹窗不同：OSD 是实心近黑矩形，周围（尤其右侧）仍是亮表体，
        不会触发整页蒙板 dimmed_modal。切掉边框/底栏后再找高填充暗块，
        要求右侧明显更亮，避免把阴影或机位暗角当 OSD。
        """
        if img is None or getattr(img, "size", 0) <= 0:
            return False
        h, w = img.shape[:2]
        if h < 80 or w < 80:
            return False
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        # 切掉显示器边框/任务栏，避免暗块与边框粘连
        mx0 = int(w * 0.08)
        mx1 = int(w * 0.96)
        my0 = int(h * 0.06)
        my1 = int(h * 0.78)
        if mx1 - mx0 < 40 or my1 - my0 < 40:
            return False
        roi_v = v[my0:my1, mx0:mx1]
        roi_s = s[my0:my1, mx0:mx1]
        dark = ((roi_v <= 50) & (roi_s <= 85)).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close, iterations=2)
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_open)
        _n, _lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        area_img = float(max(h * w, 1))
        for i in range(1, int(stats.shape[0])):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            ar = float(area) / area_img
            if ar < 0.035 or ar > 0.30:
                continue
            fill = float(area) / float(max(ww * hh, 1))
            if fill < 0.72:
                continue
            asp = float(ww) / float(max(hh, 1))
            if asp < 1.2 or asp > 4.5:
                continue
            if hh < h * 0.07 or ww < w * 0.16:
                continue
            bx0 = x + mx0
            by0 = y + my0
            cx = bx0 + ww * 0.5
            cy = by0 + hh * 0.5
            if cx < w * 0.12 or cx > w * 0.75:
                continue
            if cy < h * 0.08 or cy > h * 0.70:
                continue
            box = v[by0 : by0 + hh, bx0 : bx0 + ww]
            inv = float(np.mean(box))
            if inv > 48.0:
                continue
            if float(np.std(box)) > 32.0:
                continue
            pad = max(14, int(min(ww, hh) * 0.14))
            rx0 = min(w, bx0 + ww)
            rx1 = min(w, bx0 + ww + pad)
            if rx1 - rx0 < 8:
                continue
            right_roi = v[by0 : by0 + hh, rx0:rx1]
            right = float(np.mean(right_roi))
            bright_frac = float(np.mean(right_roi >= 130))
            if right < 120.0 or (right - inv) < 70.0 or bright_frac < 0.35:
                continue
            below_y0 = min(h, by0 + hh)
            below_y1 = min(h, by0 + hh + pad)
            above_y0 = max(0, by0 - pad)
            above_y1 = by0
            below = (
                float(np.mean(v[below_y0:below_y1, bx0 : bx0 + ww]))
                if below_y1 > below_y0
                else 0.0
            )
            above = (
                float(np.mean(v[above_y0:above_y1, bx0 : bx0 + ww]))
                if above_y1 > above_y0
                else 0.0
            )
            vert = max(below - inv, above - inv, 0.0)
            if vert < 40.0 and (right - inv) < 100.0:
                continue
            return True
        return False

    def _normalize_table_illumination(self, img: np.ndarray) -> np.ndarray:
        """
        轻度偏暗时提亮，供蓝带检测（整页蒙板已在上游 fail-fast，不走这里）。

        - 正常亮图 V 约 180~200
        - 局部黑块（分辨率 OSD）拉低 mean、但 p75 已亮 → 禁止全局增益，
          否则表白底过曝、饱和度归零，真表蓝带消失、曲线区假蓝胜出
        - 确需提亮时用高光保护：已亮像素少增益，只抬暗部
        """
        if img is None or getattr(img, "size", 0) <= 0:
            return img
        mean_v, p75_v, content_v = self._table_illumination_stats(img)
        # 亮区已够亮（仅局部黑块拉低 mean）→ 保持原图
        if p75_v >= 155.0 or content_v >= 145.0 or mean_v >= 145.0:
            return img
        gain = min(2.0, max(1.05, 175.0 / max(content_v, 1.0)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_f = hsv[:, :, 2].astype(np.float32)
        # V>=165 权重 0；V<=75 权重 1
        w = np.clip((165.0 - v_f) / 90.0, 0.0, 1.0)
        out = img.astype(np.float32)
        out = out * (1.0 + (gain - 1.0) * w[:, :, None])
        return np.clip(out, 0, 255).astype(np.uint8)

    def _table_body_blue_mask(
        self,
        img: np.ndarray,
        *,
        ignore_top: int = 0,
        s_min: Optional[int] = None,
    ) -> np.ndarray:
        """
        浅蓝/青绿表体行 mask（侧光阴影 + 弹窗蒙板 + 行选中高亮 + 底栏 OSD 鲁棒）。

        - 亮图：表体蓝行很淡（S≈6~10），s_min 宜低
        - 蒙板暗图：灰蒙板也是低 S，须提高 s_min，并靠提亮前的判定传入
        - 行选中高亮：青绿/石板蓝 S 可达 40~90，S 上限须高于旧 140
        - 主通路 V≥135：丢掉暗蓝 OSD；亮表体/选中行多 ≥180
        - 暗侧光/拍屏：真表体常 V≈90~130 且 S≈20+；另开「有色度暗青」通路，
          否则只剩表下空白区摩尔纹假蓝 → 裁切从腰线切开、表头进不了 OCR
        - CLOSE 纵向约一行高：粘合斑马纹白缝，但不跨到更远的底栏
        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hch, sch, vch = cv2.split(hsv)
        if s_min is None:
            # 调用方未指定时：已提亮图偏亮 → 按亮图淡蓝；仍暗 → 挡蒙板
            s_min = 6 if float(np.mean(vch)) >= 160.0 else 14
        s_min = int(max(4, min(30, s_min)))
        tile_w = max(8, min(32, w // 40))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile_w, 8))
        hsv_eq = cv2.merge([hch, sch, clahe.apply(vch)])
        # H 80~140：青→蓝；S 上限 180：兼容选中行较饱和青绿
        mask_hsv = cv2.inRange(hsv_eq, (80, s_min, 50), (140, 180, 255))
        # 暗色 OSD / 任务栏 V 常 <130；亮表体/选中行多 ≥180
        bright = (vch >= 135).astype(np.uint8) * 255
        mask_hsv = cv2.bitwise_and(mask_hsv, bright)
        b, g, r = cv2.split(img.astype(np.float32))
        blue_score = b - 0.5 * (g + r)
        # 青绿选中：G 偏高会压低 blue_score，另开 B 主导通路
        teal_score = (b - r) - 0.25 * np.maximum(g - b, 0.0)
        chroma_ok = (blue_score > 4.0) | (teal_score > 8.0)
        hue_ok = (hch >= 80) & (hch <= 140)
        mask_rel = (
            (
                chroma_ok
                & (b > 40)
                & (sch >= s_min)
                & (vch >= 135)
                & hue_ok
            )
        ).astype(np.uint8) * 255
        # 有色度暗青：侧光/拍屏时真表体 V 常 <135，但 S 明显高于空白摩尔纹
        dim_s = int(max(18, s_min + 4))
        mask_dim = (
            (
                chroma_ok
                & (b > 40)
                & (sch >= dim_s)
                & (vch >= 85)
                & (vch < 135)
                & hue_ok
            )
        ).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask_hsv, mask_rel)
        mask = cv2.bitwise_or(mask, mask_dim)
        top = max(0, min(int(ignore_top), h - 1))
        if top > 0:
            mask[:top, :] = 0
        # 整行偏暗且低饱和（OSD/任务栏亮字点）清掉；有色度暗青表体保留
        row_v = np.median(vch.astype(np.float64), axis=1)
        row_s = np.median(sch.astype(np.float64), axis=1)
        dark_rows = np.where((row_v < 140.0) & (row_s < float(dim_s)))[0]
        if dark_rows.size:
            mask[dark_rows, :] = 0
        # 水平粘断列 + 纵向约一行：合并斑马纹/选中块，不桥接表↔底栏大缝
        kx = max(41, w // 18)
        ky = max(9, min(21, h // 28))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
        )
        return mask

    def _lift_quad_above_header(self, src: np.ndarray, img_h: int) -> np.ndarray:
        """
        沿左右边把蓝带四边形上沿抬高，把白/灰时刻行留在透视结果里。

        拟合蓝带时会略去表头，上沿又常一边高一边低，直接拉成矩形会把
        较低一侧的时刻行裁掉。
        """
        tl, tr, br, bl = [np.asarray(p, dtype=np.float32) for p in src]
        left_v = tl - bl
        right_v = tr - br
        left_n = float(np.linalg.norm(left_v))
        right_n = float(np.linalg.norm(right_v))
        span = max(left_n, right_n, 1.0)
        top_drop = abs(float(tr[1] - tl[1]))
        pad = max(top_drop * 0.55 + span * 0.12, float(img_h) * 0.045, 16.0)
        pad = min(pad, span * 0.30, float(img_h) * 0.16)

        def lift(top: np.ndarray, length: float, vec: np.ndarray) -> np.ndarray:
            if length < 1.0:
                return top + np.array([0.0, -pad], dtype=np.float32)
            return top + (vec / length) * pad

        lifted = np.array(
            [lift(tl, left_n, left_v), lift(tr, right_n, right_v), br, bl],
            dtype=np.float32,
        )
        return lifted

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
        # 蓝带铺满裁剪框时，右/底角会粘在图像边上，approx 不是真实表框
        # 17-12_right：3/4 角贴边，拉正后左边框从 ~1° 拧到 ~40°
        if self._quad_border_corner_count(src, w, h) > self.DESKEW_QUAD_MAX_BORDER_CORNERS:
            return None
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
        # 蓝带拟合故意略去白/灰表头，上沿会斜着切过时刻行；拉正前先把上沿抬回去
        src = self._lift_quad_above_header(src, h)
        # 抬表头后角点可能出画，再计一次贴边（抬高不算贴边失败）
        # 但右/底若仍贴边说明几何仍不可信
        border_after = self._quad_border_corner_count(src, w, h)
        right_or_bot = 0
        for p in src.reshape(4, 2):
            if p[0] >= float(w - 1) - float(self.DESKEW_QUAD_BORDER_TOL_PX):
                right_or_bot += 1
            if p[1] >= float(h - 1) - float(self.DESKEW_QUAD_BORDER_TOL_PX):
                right_or_bot += 1
        if right_or_bot >= 2 or border_after > self.DESKEW_QUAD_MAX_BORDER_CORNERS + 1:
            return None
        top_w = float(np.linalg.norm(src[1] - src[0]))
        bot_w = float(np.linalg.norm(src[2] - src[3]))
        left_h = float(np.linalg.norm(src[3] - src[0]))
        right_h = float(np.linalg.norm(src[2] - src[1]))
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
        xs, ys = self._nonzero_xy(coords)
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

    def _estimate_left_content_edge_angle(self, img: np.ndarray) -> Optional[float]:
        """
        显示器左边框（暗→亮）相对竖直的倾角，转为整图旋转角（度）。
        斜拍时蓝带铺满裁剪框、霍夫被摩尔纹钉在 0°，左边框仍是可靠几何信号。
        返回值可直接交给 _rotate_image_by_angle。
        """
        h, w = img.shape[:2]
        if h < 40 or w < 80:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts: List[List[float]] = []
        y0 = max(2, int(h * 0.04))
        y1 = min(h - 2, int(h * 0.72))
        x_lim = max(12, int(w * 0.38))
        min_dark = max(4, int(w * 0.008))
        for yy in range(y0, y1, 2):
            row = gray[yy]
            dark_run = 0
            for xx in range(0, x_lim):
                val = float(row[xx])
                if val < 42.0:
                    dark_run += 1
                    continue
                if val > 70.0 and dark_run >= min_dark:
                    pts.append([float(xx), float(yy)])
                    break
                dark_run = 0
        if len(pts) < 24:
            return None
        arr = np.asarray(pts, dtype=np.float32)
        # 左边框应大致竖直且 x 变化不大；散得太开说明不是完整边框
        if float(arr[:, 0].max() - arr[:, 0].min()) > w * 0.12:
            return None
        vx, vy, _, _ = cv2.fitLine(
            arr.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
        ).ravel()
        # 方向取朝下，与图像 y 增大同向
        if float(vy) < 0.0:
            vx, vy = -float(vx), -float(vy)
        else:
            vx, vy = float(vx), float(vy)
        ang_down = math.degrees(math.atan2(vy, vx))
        # 竖直朝下为 +90°。图像逆时针转 θ 后，测得边角同样 +θ；
        # 要让 ang_down→90，应转 (ang_down-90)。符号反了会越转越歪。
        rot = ang_down - 90.0
        if abs(rot) < self.DESKEW_MIN_ANGLE_DEG:
            return None
        if abs(rot) > self.DESKEW_MAX_ANGLE_DEG:
            return None
        return float(rot)

    def _estimate_blue_band_top_angle(self, img: np.ndarray) -> Optional[float]:
        """
        浅蓝表体上沿（画面中段）相对水平的倾角（度）。
        用于验收左边框旋转：透视下竖边与横边倾角常不一致，
        只按左边框转会把表头拧得更斜。
        """
        h, w = img.shape[:2]
        if h < 40 or w < 80:
            return None
        mask = self._table_body_blue_mask(img, ignore_top=max(1, int(h * 0.06)))
        tops: List[List[float]] = []
        step = max(1, w // 200)
        for xx in range(int(w * 0.18), int(w * 0.82), step):
            nz = np.where(mask[:, xx] > 0)[0]
            if nz.size < 5:
                continue
            tops.append([float(xx), float(nz[0])])
        if len(tops) < 20:
            return None
        pts = np.asarray(tops, dtype=np.float32)
        best: Optional[np.ndarray] = None
        best_n = 0
        rng = np.random.default_rng(0)
        tol = max(3.0, float(h) * 0.01)
        for _ in range(64):
            i = rng.choice(len(pts), 2, replace=False)
            p1, p2 = pts[i]
            if abs(float(p2[0] - p1[0])) < 8.0:
                continue
            a = (float(p2[1]) - float(p1[1])) / (float(p2[0]) - float(p1[0]))
            b = float(p1[1]) - a * float(p1[0])
            err = np.abs(pts[:, 1] - (a * pts[:, 0] + b))
            inl = err < tol
            n_inl = int(inl.sum())
            if n_inl > best_n:
                best_n = n_inl
                best = pts[inl]
        if best is None or best_n < 15:
            return None
        vx, vy, _, _ = cv2.fitLine(
            best.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
        ).ravel()
        return math.degrees(math.atan2(float(vy), float(vx)))

    def _deskew_via_content_edge_rotate(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        按左边框拉竖直做平面旋转。
        验收：左边框残角变小，且蓝带上沿（横线）残角不得变差。
        透视斜拍时竖边/横边倾角常相反，只拧边框会把表头拧歪——此时返回 None。
        """
        angle = self._estimate_left_content_edge_angle(img)
        if angle is None:
            return None
        before_edge = abs(float(angle))
        before_top = self._estimate_blue_band_top_angle(img)
        before_hough = self._estimate_hough_median_angle(img)
        rotated = self._rotate_image_by_angle(img, float(angle))
        if rotated is None:
            return None
        after_edge = self._estimate_left_content_edge_angle(rotated)
        # 拉正后左边框应接近竖直（测不到角或残角明显更小）
        if after_edge is not None and abs(after_edge) > before_edge - 0.25:
            return None
        after_top = self._estimate_blue_band_top_angle(rotated)
        # 横向上沿被拧得更斜 → 透视不一致，放弃边框旋转
        if (
            before_top is not None
            and after_top is not None
            and abs(after_top) > abs(before_top) + 0.25
        ):
            return None
        after_hough = self._estimate_hough_median_angle(rotated)
        # 仅当霍夫与左边框方向大致一致时才用霍夫验收；
        # 摩尔纹中位角常与边框反向，不能单独用来否决。
        if (
            before_hough is not None
            and after_hough is not None
            and abs(before_hough) >= self.DESKEW_MIN_ANGLE_DEG
            and abs(float(angle) - float(before_hough))
            <= self.DESKEW_BLUE_HOUGH_MAX_DISAGREE_DEG
            and abs(after_hough) > abs(before_hough) + 0.35
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

        顺序：浅蓝表体透视 → 左边框旋转 → 蓝底外接矩形旋转 → 霍夫整图旋转 → 裁左黑边。
        透视须过严格几何门槛（粘边假梯形宁可不拉）；旋转须残角验收。
        近水平（霍夫/蓝带上沿均很小）不做平面旋转：小角拧图易裁掉右缘列（15:30+）。
        失败则保持原图（宁可不转，不可拧歪）。
        """
        if img is None or img.size == 0:
            return img
        angle = None
        try:
            angle = self._estimate_hough_median_angle(img)
        except Exception:
            angle = None
        top_ang = None
        try:
            top_ang = self._estimate_blue_band_top_angle(img)
        except Exception:
            top_ang = None
        out = img
        # 1) 透视优先（与霍夫角解耦；粘边四边形会在内部被拒）
        try:
            warped = self._deskew_via_blue_quad(img, hough_angle=angle)
            if warped is not None and warped.size > 0:
                out = warped
                return self._trim_left_bezel(out)
        except Exception:
            pass
        # 近水平：跳过平面旋转，留给列几何 OCR；强拧只会丢右缘
        near_level = (
            (angle is None or abs(float(angle)) < 1.2)
            and (top_ang is None or abs(float(top_ang)) < 1.2)
        )
        if near_level:
            try:
                return self._trim_left_bezel(out)
            except Exception:
                return out
        # 2) 左边框拉竖直（右屏斜拍 + 摩尔纹钉死霍夫时的主路径）
        try:
            edged = self._deskew_via_content_edge_rotate(img)
            if edged is not None and edged.size > 0:
                out = edged
                return self._trim_left_bezel(out)
        except Exception:
            pass
        # 3) 蓝底外接矩形旋转
        try:
            rected = self._deskew_via_blue_min_rect(img)
            if rected is not None and rected.size > 0:
                out = rected
                return self._trim_left_bezel(out)
        except Exception:
            pass
        # 4) 霍夫平面旋转：角过大更像误检；小角须验收改善
        if not (angle is not None and abs(angle) > self.DESKEW_MAX_ANGLE_DEG):
            try:
                rotated = self._deskew_via_hough_rotate(img, angle=angle)
                if rotated is not None and rotated.size > 0:
                    out = rotated
            except Exception:
                pass
        try:
            return self._trim_left_bezel(out)
        except Exception:
            return out

    def extract(self, image_path: str,
                crop_ratio: Optional[Tuple[float, float, float, float]] = None) -> TableExtractResult:
        """
        从图片提取表格（整表 OCR + 列对齐 + 零行补全）。

        @param image_path 图片路径
        @param crop_ratio 裁剪区域 (x0,y0,x1,y1) 比例 0~1
        @return 提取结果
        """
        self.last_step_ms = {}
        t_region = time.perf_counter()

        img_path = self.resolve_image_path(image_path)
        img = self.read_image_file(str(img_path))
        t = time.perf_counter()
        if crop_ratio is not None:
            h, w = img.shape[:2]
            x0, y0, x1, y1 = crop_ratio
            img = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
            # 固定框（含回退整图重识）也写入，便于预览与 last_auto_crop_ratio 一致
            self.last_auto_crop_ratio = (
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            )
            t = self._mark_ms("crop", t)
        else:
            # 实际送入 OCR 的框会写入 last_auto_crop_ratio（含回退整图）
            img = self._resolve_table_auto_crop(img)
            # 蒙板/黑框 OSD：不跑纠偏/OCR，本轮直接空结果（bridge 判 auto 失败）
            if self.last_crop_fallback_reason in self.CROP_ABORT_REASONS:
                self._mark_ms("crop", t)
                self.last_step_ms["region"] = int(
                    (time.perf_counter() - t_region) * 1000
                )
                self.last_step_ms["data"] = 0
                return TableExtractResult(
                    headers=[],
                    rows=[],
                    raw_row_count=0,
                    image_path=str(img_path),
                )
            t = self._mark_ms("crop", t)

        # fixed / auto 裁完后共用纠偏（斜拍时固定框同样必须先拉正再 OCR）
        img = self._deskew_table_image(img)
        self._mark_ms("deskew", t)
        self.last_step_ms["region"] = int((time.perf_counter() - t_region) * 1000)

        t_data = time.perf_counter()
        img = self._scale_image_for_ocr(img)

        t = time.perf_counter()
        cells = self._ocr_merge_multipass(img)
        t = self._mark_ms("ocr", t)
        rows = self._cluster_rows(cells)
        if not cells or not rows:
            self.last_step_ms["data"] = int((time.perf_counter() - t_data) * 1000)
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
                self.last_step_ms["data"] = int((time.perf_counter() - t_data) * 1000)
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
                self.last_step_ms["data"] = int((time.perf_counter() - t_data) * 1000)
                return TableExtractResult(
                    headers=[],
                    rows=[],
                    raw_row_count=len(rows),
                    image_path=str(img_path),
                )
            header_idx = -1

        # 仅格式统一；补洞/外推以 OCR 锚段为准，左右缺列按数据 X 跨度推导
        headers = self._refine_header_times(headers)
        # 5 被读成 6：15:15→16:15。先改回再清洗，避免乱序段被整列丢掉
        headers, col_centers = self._repair_plus60_header_hours(headers, col_centers)
        headers, col_centers = self._sanitize_header_sequence(headers, col_centers)
        headers, col_centers = self._complete_time_headers(headers, col_centers)
        # 宁缺毋滥：不再按数据 X 跨度外推缺列（半截列硬认危害大）
        # 丢掉本图边缘半列；表头完整但数据被横向滚动裁断时一并丢掉
        edge_data_cells = self._collect_edge_clip_data_cells(
            rows, header_idx, data_cells
        )
        headers_before_edge = list(headers)
        headers, col_centers = self._drop_edge_clipped_columns(
            headers, col_centers, img.shape[1], data_cells=edge_data_cells
        )
        headers, col_centers = self._sanitize_header_sequence(headers, col_centers)
        # sanitize 后再做几何贴边清理；不再二次跑数据形态 drop，避免末列级联误杀
        headers, col_centers = self._drop_edge_clipped_columns(
            headers, col_centers, img.shape[1], data_cells=None
        )
        col_count = len(headers)
        # 因数据裁断丢掉首/末列后：新边缘列左侧/右侧常残留半截墨迹，
        # 单列补洞易读成 298.358 之类脏值，禁止再局部 OCR 填这些列的空格
        skip_edge_reocr_cols = self._edge_reocr_skip_cols(
            headers_before_edge, headers
        )
        # 丢掉边缘列后，剔除伸入已裁区域的数据框，避免残片粘到新首/末列
        rows, data_cells = self._filter_cells_outside_header_span(
            rows, header_idx, data_cells, col_centers
        )

        matrix, sticky_map = self._build_full_matrix(
            rows, header_idx, data_idx, headers, col_centers, col_count)
        matrix = self._postprocess_rows(matrix)
        sticky_map = self._fit_sticky_map(sticky_map, matrix)
        # 用合并后的主数据行按 X 对齐（不顺序压列），覆盖末行拆碎错位
        # 仅当主行确实像当前末行（或末行过稀）时覆盖，避免把第 2 行指标盖成第 3 行
        if matrix and data_cells:
            aligned, aligned_sticky = self._align_row_to_columns(
                self._expand_row_cells(data_cells, col_centers),
                col_centers,
                len(headers),
            )
            if aligned and self._should_overwrite_last_with_main(matrix, aligned):
                matrix[-1] = [
                    self.NODATA_MARK
                    if v == self.NODATA_MARK
                    else (self._fix_decimal_ocr(v) if v else "")
                    for v in aligned
                ]
                sticky_map = self._fit_sticky_map(sticky_map, matrix)
                sticky_map[-1] = list(aligned_sticky[: len(matrix[-1])]) + [""] * max(
                    0, len(matrix[-1]) - len(aligned_sticky)
                )
        # 先按 X 回填空格，空格再局部 OCR；识别结果原样保留（异常值不置空）
        matrix, sticky_map = self._fill_sparse_data_row(
            matrix, data_cells, col_centers, col_count, sticky_map=sticky_map
        )
        # 先清表头时刻泄漏（23:00→23.00），再判是否整表未出清
        matrix = self._scrub_header_time_leaks(headers, matrix)
        sticky_map = self._clear_sticky_where_empty(sticky_map, matrix)
        # 末行若被误填成全 0（电价漏检占位），先置空再走列补识
        matrix = self._blank_fake_zero_price_row(matrix)
        sticky_map = self._clear_sticky_where_empty(
            self._fit_sticky_map(sticky_map, matrix), matrix
        )
        # 主路径：表头锚点下固定 3 带按列 OCR（先电价、再上两行 0）
        # 不依赖整图是否碰巧框到白底第三行，避免用 0 补洞冒充电价
        matrix = self._ensure_geometric_three_data_rows(
            img,
            matrix,
            col_centers,
            rows=rows,
            header_idx=header_idx,
            data_cells=data_cells,
            headers=headers,
        )
        matrix = self._scrub_price_row_no_crossfill(matrix)
        sticky_map = self._clear_sticky_where_empty(
            self._fit_sticky_map(sticky_map, matrix), matrix
        )
        # 初看全空时仍先做一轮列补识：真实「全 0 + 电价」常被整图漏检，
        # 若直接当未出清跳过，会把可见数据整表置空。补识后再判定。
        looked_empty = self._matrix_looks_all_uncleared(matrix, col_count)
        t = time.perf_counter()
        matrix = self._refill_matrix_empties(
            img,
            matrix,
            rows,
            header_idx,
            data_idx,
            data_cells,
            col_centers,
            skip_cols=skip_edge_reocr_cols,
        )
        matrix = self._scrub_header_time_leaks(headers, matrix)
        sticky_map = self._clear_sticky_where_empty(sticky_map, matrix)
        sticky_price_hint = self._rows_have_sticky_price_evidence(
            rows, header_idx, data_cells
        )
        # 即便矩阵像全未出清，也先走粘连/电价可疑重识 + 后续几何恢复；
        # 过早整表置空会让 _recover_* 因 nodata=全列直接退出（拍屏回退整图时常见）。
        matrix, sticky_map = self._reocr_geometry_sticky_cells(
            img,
            matrix,
            rows,
            header_idx,
            data_idx,
            data_cells,
            col_centers,
            skip_cols=skip_edge_reocr_cols,
            sticky_map=sticky_map,
        )
        # 粘连重识可能把电价写入上方行：先清渗漏，再走后续补 0
        matrix, sticky_map = self._scrub_price_echo_from_upper_rows(
            matrix, sticky_map
        )
        # 先丢掉右缘短小数 / 左缘缺高位列，再可疑重识
        headers, matrix, sticky_map = self._drop_right_edge_short_frac_column(
            headers, matrix, sticky_map=sticky_map
        )
        headers, matrix, sticky_map = self._drop_left_edge_short_int_column(
            headers, matrix, sticky_map=sticky_map
        )
        col_count = len(headers)
        if len(col_centers) > col_count:
            col_centers = list(col_centers[:col_count])
        # 电价行可疑已填格（漏点/串位）按列强制重识，可覆盖错值
        matrix = self._reocr_suspicious_price_cells(
            img,
            matrix,
            rows,
            header_idx,
            data_idx,
            data_cells,
            col_centers,
            skip_cols=skip_edge_reocr_cols,
        )
        if looked_empty or sticky_price_hint:
            # 补识前全空、补识后有值：再 scrub 一次表头泄漏
            matrix = self._scrub_header_time_leaks(headers, matrix)
        ocr_filled = self._count_numeric_cells(matrix)
        headers, matrix, sticky_map = self._trim_trailing_empty_and_broken_headers(
            headers, matrix, sticky_map=sticky_map
        )
        # 右缘短小数 / 左缘缺高位 → 整列丢（表头常被补全成完整 HH:MM）
        headers, matrix, sticky_map = self._drop_right_edge_short_frac_column(
            headers, matrix, sticky_map=sticky_map
        )
        headers, matrix, sticky_map = self._drop_left_edge_short_int_column(
            headers, matrix, sticky_map=sticky_map
        )
        # 斜拍/碎片兜底：数据行硬封顶为 3；不足则按电价位补零行槽
        matrix, sticky_map = self._enforce_expected_data_rows(
            matrix, sticky_map=sticky_map, expected=self.EXPECTED_DATA_ROWS
        )
        # 电力/计划（整型）与电价（小数）混行时拆开
        matrix = self._untangle_integer_decimal_rows(matrix)
        sticky_map = self._fit_sticky_map(sticky_map, matrix)
        # 表头时刻泄漏进数据格（22:30→22.30）清掉，再判未出清
        matrix = self._scrub_header_time_leaks(headers, matrix)
        sticky_map = self._clear_sticky_where_empty(sticky_map, matrix)
        t = self._mark_ms("cell_fix", t)
        # 后置兜底：早期几何三带若仍缺电价/零行，再救一轮
        # （_trailing_nodata 以电价为准；先有电价再补 0）
        matrix = self._ensure_geometric_three_data_rows(
            img,
            matrix,
            col_centers,
            rows=rows,
            header_idx=header_idx,
            data_cells=data_cells,
            headers=headers,
        )
        sticky_map = self._clear_sticky_where_empty(
            self._fit_sticky_map(sticky_map, matrix), matrix
        )
        matrix = self._recover_missing_price_row(
            img,
            matrix,
            col_centers,
            rows=rows,
            header_idx=header_idx,
            data_cells=data_cells,
            headers=headers,
        )
        matrix = self._scrub_price_row_no_crossfill(matrix)
        sticky_map = self._clear_sticky_where_empty(
            self._fit_sticky_map(sticky_map, matrix), matrix
        )
        # 电价恢复后再清一次上方渗漏（末行补全后才能判定 echo）
        matrix, sticky_map = self._scrub_price_echo_from_upper_rows(
            matrix, sticky_map
        )
        # 前两行常为全 0：整图检测极易漏检；表头几何分带 + 按列补 0
        # （不依赖左侧行名——横向滚动后行名可能不可见）
        matrix = self._recover_leading_zero_data_rows(
            img,
            matrix,
            col_centers,
            rows=rows,
            header_idx=header_idx,
            data_cells=data_cells,
            headers=headers,
        )
        sticky_map = self._clear_sticky_where_empty(
            self._fit_sticky_map(sticky_map, matrix), matrix
        )
        # 漏点补救：5 位、4 位都按同行锚点双候选选优（6 位已在 _fix_decimal_ocr）
        if self.fix_decimal_enabled:
            matrix = [self._apply_decimal_row_repairs(r) for r in matrix]
        # 补点后若出现极端跳变（如 312.443→2.443），再强制扫一遍电价可疑格
        if matrix and col_centers:
            matrix = self._reocr_suspicious_price_cells(
                img,
                matrix,
                rows,
                header_idx,
                data_idx,
                data_cells,
                col_centers,
                skip_cols=skip_edge_reocr_cols,
            )
            if self.fix_decimal_enabled:
                matrix = [self._apply_decimal_row_repairs(r) for r in matrix]
        matrix = self._scrub_header_time_leaks(headers, matrix)
        # 末行：无带点小数前禁止残留 0/上行串值；认不到保持空（不写 PI）
        matrix = self._scrub_price_row_no_crossfill(matrix)
        matrix = self._force_uncleared_columns_empty(headers, matrix)
        matrix = self._strip_nodata_marks(matrix)
        matrix = self._scrub_price_row_no_crossfill(matrix)
        # 电价行摩尔纹全 1：短整数置空；带点全 1 仅量级离谱才置空（真·11.111 保留）
        matrix = self._blank_moire_short_tokens(matrix)
        matrix = self._blank_short_frac_price_cells(matrix)
        sticky_map = self._clear_sticky_where_empty(
            self._fit_sticky_map(sticky_map, matrix), matrix
        )
        # 首列时刻前导「1」边框伪影 → 整列舍弃
        headers, matrix, sticky_map = self._drop_left_border_hour_one_column(
            headers, matrix, sticky_map=sticky_map
        )
        # 左缘电价缺高位（54.082 vs 355.xxx）→ 整列舍弃（表头常被 OCR 补成完整 06:45）
        headers, matrix, sticky_map = self._drop_left_edge_short_int_column(
            headers, matrix, sticky_map=sticky_map
        )
        # 补点/重识后再扫一次右缘短小数列（恢复步骤可能重新填出残片）
        headers, matrix, sticky_map = self._drop_right_edge_short_frac_column(
            headers, matrix, sticky_map=sticky_map
        )
        sticky_map = self._clear_sticky_where_empty(
            self._fit_sticky_map(sticky_map, matrix), matrix
        )
        self._mark_ms("recover", t)
        self.last_step_ms["data"] = int((time.perf_counter() - t_data) * 1000)
        ocr_filled = self._count_numeric_cells(matrix)
        sticky_filled = sum(
            1 for row in sticky_map for c in row if c and str(c).strip()
        )

        return TableExtractResult(
            headers=headers,
            rows=matrix,
            raw_row_count=len(rows),
            image_path=str(img_path),
            ocr_filled=ocr_filled,
            sticky_map=sticky_map,
            sticky_filled=sticky_filled,
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

    def _set_crop_fallback(self, reason: str) -> None:
        """标记表格区域定位失败并回退整图。"""
        self.last_crop_fallback = True
        self.last_crop_fallback_reason = str(reason or "fallback")
        self.last_auto_crop_ratio = self.DEFAULT_TABLE_CROP_RATIO

    def _clear_crop_fallback(self) -> None:
        self.last_crop_fallback = False
        self.last_crop_fallback_reason = ""

    def crop_fallback_message(self, reason: Optional[str] = None) -> str:
        """回退原因的中文说明（联调/详情页）。"""
        key = str(reason if reason is not None else self.last_crop_fallback_reason or "")
        return self.CROP_FALLBACK_REASONS.get(key, "表格区域定位失败，已回退原图")

    def _return_full_image_crop(
        self, img: np.ndarray, *, reason: str
    ) -> np.ndarray:
        """回退整图并记录原因。"""
        self._set_crop_fallback(reason)
        return img

    def _blue_mask_quality_ok(
        self,
        mask2: np.ndarray,
        *,
        dense_rows: np.ndarray,
        dense_thr: int,
        h: int,
        w: int,
    ) -> Tuple[bool, str]:
        """
        蓝带质量：现货表多为 2～3 段浅蓝行带；大块连续「假蓝」（空表底色/页脚）拒收。
        """
        del w  # 宽度阈值已体现在 dense_thr
        if dense_rows.size < 3:
            return False, "too_short"
        gap_tol = max(8, int(h * 0.025))
        breaks = np.where(np.diff(dense_rows) > gap_tol)[0]
        n_bands = int(breaks.size) + 1
        span = int(dense_rows[-1] - dense_rows[0] + 1)
        if n_bands >= 2:
            return True, ""
        # 单段过矮
        if span < max(24, int(h * 0.04)):
            return False, "too_short"
        # 单段过高且几乎行行都密：多半是整片染色空表，不是表体行带。
        # 行选中高亮常把 2～3 行粘成一段（约 8%～28% 高），勿当空表拒收。
        if span > int(h * 0.42):
            dens_in = np.count_nonzero(mask2, axis=1)[
                int(dense_rows[0]) : int(dense_rows[-1]) + 1
            ]
            if dens_in.size and float(np.mean(dens_in >= dense_thr)) > 0.88:
                return False, "low_blue_quality"
        return True, ""

    def _score_table_blue_band(
        self,
        y0: int,
        y1: int,
        h: int,
        *,
        dens: float = 1.0,
        bright: float = 1.0,
        chroma: float = 1.0,
    ) -> float:
        """
        多段假蓝时给「更像三行表体」的带打分。

        真实表体高度约 8%～32%；右摄表常偏搜索窗上半，勿只奖「偏下」。
        bright：带内相对亮度 0~1（亮表行高，OSD 低）。
        chroma：带内饱和度 0~1（真青绿高；空白区摩尔纹假蓝 S≈6~12 应压分）。
        """
        hh = max(int(h), 1)
        span = (int(y1) - int(y0) + 1) / float(hh)
        mid = (0.5 * (int(y0) + int(y1))) / float(hh)
        if span < 0.06:
            height_score = 0.20 * (span / 0.06)
        elif span <= 0.32:
            height_score = 1.0
        elif span <= 0.45:
            height_score = 0.35
        else:
            height_score = 0.08
        # 中部最好（约 0.30~0.55）；偏上仍可用；贴底 OSD/空白假蓝压分
        pos_score = 0.55 + 0.45 * (1.0 - min(1.0, abs(mid - 0.42) / 0.55))
        if mid > 0.88 or (y1 >= int(hh * 0.95) and mid > 0.80):
            pos_score *= 0.20
        dens_score = 0.35 + 0.65 * max(0.0, min(float(dens), 1.0))
        bright_score = 0.25 + 0.75 * max(0.0, min(float(bright), 1.0))
        chroma_score = 0.20 + 0.80 * max(0.0, min(float(chroma), 1.0))
        return float(
            height_score * pos_score * dens_score * bright_score * chroma_score
        )

    def _tighten_tall_blue_band(
        self,
        row_nz: np.ndarray,
        y0: int,
        y1: int,
        *,
        h: int,
        dense_thr: int,
    ) -> Tuple[int, int]:
        """
        过高蓝带（曲线碎蓝 + 表体 / 行选中青绿大块）时，收成表体高覆盖核心。

        - 斑马纹淡蓝：nz 峰在真表行，曲线网格更碎
        - 行选中青绿：常把表体+周围浅青粘成整窗大带；须抬高峰阈值收核
        - 多段时优先「像三行表」的高度（约 8%～32%），不再死取最靠下一段
          （右摄像位低、表在搜索窗偏上时，最下段可能是桌面反光假蓝）
        """
        hh = max(int(h), 1)
        y0 = max(0, int(y0))
        y1 = min(hh - 1, int(y1))
        if y1 <= y0:
            return y0, y1
        span = (y1 - y0 + 1) / float(hh)
        if span <= 0.28:
            return y0, y1
        seg = np.asarray(row_nz[y0 : y1 + 1], dtype=np.float64)
        if seg.size < 8:
            return y0, y1
        k = max(5, hh // 180)
        smooth = np.convolve(seg, np.ones(k, dtype=np.float64) / float(k), mode="same")
        peak = float(np.max(smooth)) if smooth.size else 0.0
        if peak < 1.0:
            return y0, y1
        # 带越高，峰阈值越严：选中高亮整窗假蓝须收到 ~0.9 peak 才分得出表核
        if span > 0.55:
            fracs = (0.88, 0.80, 0.70, 0.55)
        elif span > 0.40:
            fracs = (0.80, 0.70, 0.55)
        else:
            fracs = (0.70, 0.55)
        gap = max(6, int(hh * 0.012))
        min_core = max(24, int(hh * 0.04))
        best: Optional[Tuple[float, int, int]] = None
        for frac in fracs:
            hi = max(float(dense_thr), float(frac) * peak)
            idxs = np.where(smooth >= hi)[0]
            if idxs.size < 3:
                continue
            breaks = np.where(np.diff(idxs) > gap)[0]
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [idxs.size - 1]))
            for si, ei in zip(starts, ends):
                a = y0 + int(idxs[int(si)])
                b = y0 + int(idxs[int(ei)])
                core_h = b - a + 1
                if core_h < min_core:
                    continue
                core_span = core_h / float(hh)
                dens = float(np.mean(seg[a - y0 : b - y0 + 1] >= float(dense_thr)))
                # 表体高度像真度：8%～32% 满分；过高/过矮降权
                if core_span < 0.06:
                    h_score = 0.25 * (core_span / 0.06)
                elif core_span <= 0.32:
                    h_score = 1.0
                elif core_span <= 0.45:
                    h_score = 0.40
                else:
                    h_score = 0.10
                mid = 0.5 * (a + b) / float(hh)
                # 略偏中上/中下均可；贴底假蓝压分
                pos = 0.55 + 0.45 * (1.0 - abs(mid - 0.45) / 0.55)
                if mid > 0.88:
                    pos *= 0.25
                score = h_score * dens * pos
                cand = (score, a, b)
                if best is None or cand[0] > best[0]:
                    best = cand
            # 已收到合理表高则停，避免再降阈值又粘回大块。
            # 过矮且贴底的短核（分页/底栏假蓝）勿停，继续降阈值找中部真表。
            if best is not None:
                _s, ba, bb = best
                core_span = (bb - ba + 1) / float(hh)
                mid = 0.5 * (ba + bb) / float(hh)
                if 0.07 <= core_span <= 0.40:
                    return int(ba), int(bb)
                if core_span < 0.07 and 0.25 <= mid <= 0.78:
                    return int(ba), int(bb)
        if best is None:
            return y0, y1
        return int(best[1]), int(best[2])

    def _expand_blue_band_to_edge_rows(
        self,
        row_nz: np.ndarray,
        y: int,
        y_bot: int,
        *,
        h: int,
        dense_thr: int,
    ) -> Tuple[int, int, int]:
        """
        收核后再沿密蓝向下有限外扩，捞回峰阈值切掉的电价行。

        峰值收核偏好高覆盖核心，白底电价行常刚好落在核下沿外。
        只向下扩约 2.2 行高；向上仍靠 pad_y_top（上扩易粘回曲线摩尔纹）。
        """
        hh = max(int(h), 1)
        y = max(0, min(int(y), hh - 1))
        y_bot = max(y, min(int(y_bot), hh - 1))
        bh = max(y_bot - y + 1, 1)
        # 核过矮时（只收到 1～2 行）行高估偏小，下限按画面比例兜住
        row_h = max(int(bh / 2.5), int(hh * 0.04), 14)
        lim = int(row_h * 2.2)
        # 电价行：核下沿仍密蓝时向下扩；阈值略低于 dense_thr，兼容白底淡蓝
        thr_dn = max(8, int(dense_thr * 0.50))
        yy = y_bot
        max_dn = min(hh - 1, y_bot + lim)
        while yy < max_dn and int(row_nz[yy + 1]) >= thr_dn:
            yy += 1
        y_bot = int(yy)
        bh = max(y_bot - y + 1, 1)
        return y, y_bot, bh

    def _band_dense_ratio(
        self, row_nz: np.ndarray, y0: int, y1: int, dense_thr: int
    ) -> float:
        seg = row_nz[max(0, int(y0)) : int(y1) + 1]
        if seg.size <= 0:
            return 0.0
        return float(np.mean(seg >= int(dense_thr)))

    def _pick_table_blue_band(
        self,
        dense_rows: np.ndarray,
        *,
        h: int,
        gap_tol: int,
        row_nz: Optional[np.ndarray] = None,
        dense_thr: int = 0,
        row_v: Optional[np.ndarray] = None,
        row_s: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """
        从稠密蓝行里选出表体 y 范围（含端点）。

        - 丢掉低密度贴底页脚假蓝
        - 丢掉暗色底栏 OSD（行亮度明显低于亮表体）
        - 丢掉近白低饱和带（表下空白 + 摩尔纹假蓝，常压过偏暗真表体）
        - 过高带先收成高覆盖核心（去掉曲线网格）
        - 多段按表体像真度选，避免 longest-wins / 贴底 OSD
        """
        breaks = np.where(np.diff(dense_rows) > gap_tol)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [dense_rows.size - 1]))
        candidates: List[Tuple[float, int, int]] = []
        for si, ei in zip(starts, ends):
            y0 = int(dense_rows[si])
            y1 = int(dense_rows[ei])
            dens = 1.0
            if row_nz is not None and dense_thr > 0:
                dens = self._band_dense_ratio(row_nz, y0, y1, dense_thr)
                # 页脚/任务栏：行密度稀疏，勿当表体
                if dens < 0.55:
                    continue
                y0, y1 = self._tighten_tall_blue_band(
                    row_nz, y0, y1, h=h, dense_thr=dense_thr
                )
                dens = self._band_dense_ratio(row_nz, y0, y1, dense_thr)
            span = (y1 - y0 + 1) / float(max(int(h), 1))
            if span < 0.045:
                continue
            # 贴底短带（分页栏 / OSD）丢掉
            if y1 >= int(h * 0.96) and span < 0.18:
                continue
            bright = 1.0
            chroma = 1.0
            v_med = 200.0
            if row_v is not None and row_v.size > y1:
                seg_v = np.asarray(row_v[y0 : y1 + 1], dtype=np.float64)
                v_med = float(np.median(seg_v)) if seg_v.size else 0.0
            s_med = 40.0
            if row_s is not None and row_s.size > y1:
                seg_s = np.asarray(row_s[y0 : y1 + 1], dtype=np.float64)
                s_med = float(np.median(seg_s)) if seg_s.size else 0.0
                # 近白空白区：V 高但 S 极低，摩尔纹易进蓝 mask
                if s_med < 12.0 and v_med >= 160.0:
                    continue
                chroma = max(0.0, min(1.0, (s_med - 8.0) / 28.0))
            # 暗青表体（侧光/拍屏）V 可 ≈90；无色度暗行才当 OSD
            if v_med < 85.0:
                continue
            if v_med < 155.0 and s_med < 16.0:
                continue
            bright = max(0.0, min(1.0, (v_med - 85.0) / 115.0))
            score = self._score_table_blue_band(
                y0, y1, h, dens=dens, bright=bright, chroma=chroma
            )
            candidates.append((score, y0, y1))
        if not candidates:
            # 全部被硬滤时：勿简单 longest（空白假蓝常更长），改选色度更高的收核带
            fallback: List[Tuple[float, int, int]] = []
            for si, ei in zip(starts, ends):
                a0 = int(dense_rows[si])
                b0 = int(dense_rows[ei])
                if row_nz is not None and dense_thr > 0:
                    a0, b0 = self._tighten_tall_blue_band(
                        row_nz, a0, b0, h=h, dense_thr=dense_thr
                    )
                s_fb = 0.0
                if row_s is not None and row_s.size > b0:
                    s_fb = float(np.median(row_s[a0 : b0 + 1]))
                fallback.append((s_fb, a0, b0))
            if fallback:
                fallback.sort(key=lambda t: (t[0], -(t[2] - t[1])))
                _s, y0, y1 = fallback[-1]
                return int(y0), int(y1)
            best_i = int(np.argmax(ends - starts))
            return int(dense_rows[starts[best_i]]), int(dense_rows[ends[best_i]])
        candidates.sort(key=lambda t: (t[0], t[1]))
        _score, y0, y1 = candidates[-1]
        return int(y0), int(y1)

    def _resolve_table_auto_crop(self, img: np.ndarray) -> np.ndarray:
        """
        选定 table_auto 实际裁剪，并保证 last_auto_crop_ratio 与结果一致。

        摄像头贴近、画面基本就是表格时，自动框可以占满有效区域。
        弹窗蒙板整页压暗：直接 dimmed_modal 失败（不提亮硬识）。
        显示器黑框 OSD：直接 black_osd 失败（不硬识被挡表）。
        蓝带失败 / 过矮 / 质量不足时回退整图，并置 last_crop_fallback。
        """
        self._clear_crop_fallback()
        # 蒙板压暗：准确率有限，快速失败等下一抓拍（关弹窗后亮度恢复）
        if self._is_modal_dimmed(img):
            return self._return_full_image_crop(img, reason="dimmed_modal")
        # 黑框 OSD（信号超范围等）：表被挡，本轮跳过
        if self._has_black_osd_box(img):
            return self._return_full_image_crop(img, reason="black_osd")
        # 提亮前判暗/亮：轻度偏暗用稍高 S；局部 OSD 拉低 mean 但表已亮 → 淡蓝低 S
        mean_v, p75_v, content_v = self._table_illumination_stats(img)
        if p75_v >= 155.0 or content_v >= 150.0:
            s_min = 6
        elif mean_v < 145.0:
            s_min = 14
        else:
            s_min = 6
        work = self._normalize_table_illumination(img)
        auto = self._auto_crop_table(work, s_min=s_min)
        if self.last_crop_fallback:
            return auto
        h = max(int(work.shape[0]), 1)
        auto_h = float(auto.shape[0]) / float(h)
        if auto_h < 0.12 or auto.shape[0] < 48:
            return self._return_full_image_crop(work, reason="too_short")
        return auto

    def _auto_crop_table(
        self, img: np.ndarray, *, s_min: Optional[int] = None
    ) -> np.ndarray:
        """
        自动裁剪表格区域（整幅画面里的浅蓝数据行）。

        近距离拍摄时表头可以贴在画面顶部。在整幅图里找浅蓝行，向上多留
        约一行白/灰底时刻；行间白缝并进同一条表带，不再只搜下半屏。
        定位失败时回退整图（last_crop_fallback=True）。
        """
        h, w = img.shape[:2]
        # 只丢掉顶上一条细边，表头允许从画面顶部开始
        ignore_top = int(h * 0.02)
        mask = self._table_body_blue_mask(
            img, ignore_top=ignore_top, s_min=s_min
        )
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
            return self._return_full_image_crop(img, reason="no_blue")

        x, y, bw, bh = cv2.boundingRect(coords)
        # 用行密度再收紧上下界：ignore_top 边界易残留碎蓝，勿把白缝/曲线底算进表体
        row_nz = np.count_nonzero(mask2, axis=1)
        dense_thr = max(int(w * 0.12), 30)
        dense_rows = np.where(row_nz >= dense_thr)[0]
        ok_q, bad_reason = self._blue_mask_quality_ok(
            mask2, dense_rows=dense_rows, dense_thr=dense_thr, h=h, w=w
        )
        # too_short / 无密行：直接回退。low_blue_quality（单段过高假蓝）先收紧选带，
        # 拍屏摩尔纹时常把曲线+表连成大块，直接回退整图会导致表头有、数据全空。
        if not ok_q and bad_reason != "low_blue_quality":
            return self._return_full_image_crop(img, reason=bad_reason)
        if dense_rows.size >= 3:
            # 行间白缝仍算同一张表；只有明显大缝才拆成两段
            # 多段时按表体像真度选，避开曲线假蓝 / 底栏暗蓝 OSD
            gap_tol = max(8, int(h * 0.025))
            hsv_rows = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            row_v = np.median(hsv_rows[:, :, 2], axis=1).astype(np.float64)
            row_s = np.median(hsv_rows[:, :, 1], axis=1).astype(np.float64)
            y, y_bot = self._pick_table_blue_band(
                dense_rows,
                h=h,
                gap_tol=gap_tol,
                row_nz=row_nz,
                dense_thr=dense_thr,
                row_v=row_v,
                row_s=row_s,
            )
            # 只用本段高度；勿再 max(整图 bbox//2)，否则假蓝大框会把裁剪重新撑满半屏
            bh = max(y_bot - y + 1, 1)
            # x 范围仍用整 mask，避免暗侧漏列
            xs = np.where(mask2[y : y + bh, :].any(axis=0))[0]
            if xs.size:
                x = int(xs[0])
                bw = int(xs[-1] - xs[0] + 1)
            # 收紧后仍过高且几乎行行都密 → 确是整片染色，再回退整图
            if not ok_q and bad_reason == "low_blue_quality":
                span_after = bh / float(max(h, 1))
                dens_after = self._band_dense_ratio(row_nz, y, y + bh - 1, dense_thr)
                if span_after > 0.45 and dens_after > 0.88:
                    return self._return_full_image_crop(img, reason="low_blue_quality")
                # 假蓝大片收核过碎：近白低 S 的摩尔纹核才回退；
                # 偏亮有色短核（淡蓝表体只检出一行高）交给后续 short_core 外扩
                if span_after < 0.08:
                    y_hi = min(h - 1, y + bh - 1)
                    seg_s = (
                        float(np.median(row_s[y : y_hi + 1]))
                        if row_s.size > y_hi
                        else 0.0
                    )
                    seg_v = (
                        float(np.median(row_v[y : y_hi + 1]))
                        if row_v.size > y_hi
                        else 0.0
                    )
                    # 亮淡蓝 或 偏暗高饱和青绿（选中行/侧光）都算有效短核
                    good_short_core = y > int(h * 0.18) and (
                        (seg_v >= 140.0 and seg_s >= 14.0)
                        or (seg_v >= 100.0 and seg_s >= 22.0)
                    )
                    if not good_short_core:
                        return self._return_full_image_crop(
                            img, reason="low_blue_quality"
                        )
        elif not ok_q:
            return self._return_full_image_crop(img, reason=bad_reason)
        # 蓝带贴输入图顶：可能是先验窗裁断了表，也可能是 OSD/曲线区贴顶假蓝。
        # 若下方还有密蓝带，改选下方（真表常在中下部）；都贴顶才回退整图。
        top_clip_y = ignore_top + max(4, int(h * 0.025))
        if y <= top_clip_y or (
            bh < max(28, int(h * 0.055)) and y < int(h * 0.12)
        ):
            recovered = False
            if dense_rows.size >= 3:
                below = dense_rows[dense_rows > int(top_clip_y)]
                if below.size >= 3:
                    gap_tol = max(8, int(h * 0.025))
                    hsv_rows = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    row_v = np.median(hsv_rows[:, :, 2], axis=1).astype(
                        np.float64
                    )
                    row_s = np.median(hsv_rows[:, :, 1], axis=1).astype(
                        np.float64
                    )
                    y2, y_bot2 = self._pick_table_blue_band(
                        below,
                        h=h,
                        gap_tol=gap_tol,
                        row_nz=row_nz,
                        dense_thr=dense_thr,
                        row_v=row_v,
                        row_s=row_s,
                    )
                    if y2 > top_clip_y:
                        y, y_bot = int(y2), int(y_bot2)
                        bh = max(y_bot - y + 1, 1)
                        xs = np.where(mask2[y : y + bh, :].any(axis=0))[0]
                        if xs.size:
                            x = int(xs[0])
                            bw = int(xs[-1] - xs[0] + 1)
                        recovered = True
            if not recovered:
                return self._return_full_image_crop(img, reason="clipped_top")
        # 峰值收核常停在「上一行 0」腰线：白底电价 / 灰表头仍密蓝但低于 peak。
        # 沿密蓝有限外扩（约 1.6 行），避免只靠 pad 半截数字；贴顶重选后也走这里。
        if dense_rows.size >= 3:
            y, y_bot, bh = self._expand_blue_band_to_edge_rows(
                row_nz, y, y_bot, h=h, dense_thr=dense_thr
            )
            xs = np.where(mask2[y : y + bh, :].any(axis=0))[0]
            if xs.size:
                x = int(xs[0])
                bw = int(xs[-1] - xs[0] + 1)
        # 摩尔纹大假蓝经 tighten 常只剩单行核（~5%高），且核可能落在电价行而非顶行。
        # 上下对称外扩，覆盖「表头 + 3 数据行」，避免只向下扩时表头被切光。
        short_core = False
        min_body_h = max(40, int(h * 0.09))
        if bh < min_body_h:
            short_core = True
            row_h = max(int(bh), int(h * 0.028), 14)
            y = max(0, int(y - row_h * 3.0))
            y_bot = min(h - 1, int(y + row_h * 4.5) - 1)
            # 若上扩后 y 被顶住，仍保证总跨度
            if y_bot - y + 1 < int(row_h * 4.2):
                y_bot = min(h - 1, y + int(row_h * 4.5) - 1)
            bh = max(y_bot - y + 1, 1)
        # 上扩约 1～2 行白/灰底时刻；表可以贴在画面顶部
        # 近景特写时灰底表头不进蓝 mask，旧上限 12% 常把 21:30 等时刻切掉
        # 下扩捞电价后 bh≈3～4 行，行高按 3.5 估（勿再用 2.8，否则顶 pad 吃进曲线）
        # 短核已含表头区：row_est 用 bh/4.2，顶 pad 只需小余量
        row_est = max(bh / (4.2 if short_core else 3.5), h * 0.04)
        pad_y_top = int(min(max(row_est * 1.5, h * 0.05), h * 0.20))
        # 表体贴在画面中下部时：灰表头在蓝带上方，约再留 1～1.5 行
        if y > int(h * 0.40) or short_core:
            pad_y_top = int(
                min(
                    max(
                        pad_y_top,
                        int(row_est * (1.1 if short_core else 1.7)),
                        int(h * (0.05 if short_core else 0.07)),
                    ),
                    h * 0.22,
                )
            )
        # 底行（电价）易贴蓝带下沿；2K 拍屏时仅 bh*0.18 常只够半个数字高
        # 一律至少留约 0.9 行余量（下扩已含电价行本体），中下部/短核再加到 ~1.2 行
        pad_y_bot = max(
            12,
            int(bh * 0.16),
            int(h * 0.035),
            int(row_est * 0.9),
        )
        if y > int(h * 0.40) or short_core:
            pad_y_bot = max(
                pad_y_bot,
                int(row_est * (1.15 if short_core else 1.25)),
                int(h * 0.05),
            )
        pad_x = max(6, int(bw * 0.015), int(w * 0.004))
        # 左缘常因首列偏暗/半截漏检被啃掉一列；左侧多扩约 1 列宽
        col_est = max(bw / 8.0, w * 0.09)
        pad_x_left = max(pad_x, int(col_est * 0.95), int(w * 0.04))
        # 右缘同理：蓝 mask 常盖不住最右白/灰列（12:45/13:00），右侧也至少扩约 1 列
        pad_x_right = max(pad_x, int(col_est * 0.90), int(w * 0.035))
        x0 = max(0, x - pad_x_left)
        y0 = max(0, y - pad_y_top)
        x1 = min(w, x + bw + pad_x_right)
        y1 = min(h, y + bh + pad_y_bot)
        # 蓝带靠上且几乎铺满宽：时刻行多半贴顶，上沿贴到画面顶，避免表头被裁
        if y <= int(h * 0.35) and bw >= int(w * 0.50):
            y0 = min(y0, max(0, int(h * 0.005)))
        # 蓝带起点仍偏右（>4%）：再向左拉一列，避免首列时刻/电价被裁断
        if x0 > int(w * 0.04):
            x0 = max(0, int(x0 - col_est * 0.9))
        # 浅蓝行常常盖不住最右一两列（白底电价/反光）。右侧若是显示器黑边，扩到黑边前；
        # 找不到黑边时再至少保证约 1 列余量，避免末列贴齐裁切线被丢。
        x1 = self._extend_right_before_bezel(
            img, x0, y, x1, min(h, y + max(bh, 1)), col_est=col_est
        )
        # 表通常接近全宽；偏窄时左右略扩（暗侧漏检时尤其需要）
        if (x1 - x0) < w * 0.70:
            mid = 0.5 * (x0 + x1)
            half = 0.45 * w
            x0 = max(0, int(mid - half))
            x1 = min(w, int(mid + half))
        # 蓝带过矮说明没形成表，回退整图
        if (y1 - y0) < max(36, int(h * 0.08)):
            return self._return_full_image_crop(img, reason="too_short")
        cropped = img[y0:y1, x0:x1]
        if cropped.size == 0:
            return self._return_full_image_crop(img, reason="too_short")
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
        self._clear_crop_fallback()
        self.last_auto_crop_ratio = (
            x0 / float(w),
            y0 / float(h),
            x1 / float(w),
            y1 / float(h),
        )
        return cropped

    def _trim_left_bezel(self, img: np.ndarray) -> np.ndarray:
        """
        纠偏后再裁掉左侧显示器黑边。
        黑边要先留给左边框测角；拉正后再去掉，避免预览里斜黑边造成「严重倾斜」观感。
        """
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        if h < 40 or w < 80:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xs: List[int] = []
        step = max(2, h // 80)
        x_lim = max(12, int(w * 0.30))
        min_dark = max(4, int(w * 0.008))
        for yy in range(int(h * 0.04), int(h * 0.85), step):
            row = gray[yy]
            dark_run = 0
            for xx in range(0, x_lim):
                val = float(row[xx])
                if val < 42.0:
                    dark_run += 1
                    continue
                if val > 70.0 and dark_run >= min_dark:
                    xs.append(xx)
                    break
                dark_run = 0
        if len(xs) < 8:
            return img
        edge = int(np.median(np.asarray(xs, dtype=np.float64)))
        if edge <= 2 or edge > int(w * 0.18):
            return img
        cut = max(0, edge - 2)
        if cut < 3:
            return img
        return img[:, cut:].copy()

    def _extend_right_before_bezel(
        self,
        img: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        col_est: float = 0.0,
    ) -> int:
        """
        把表区右缘扩到右侧黑边之前。

        近景双屏时，本屏表格右边常还有一截白/灰列，蓝底掩膜到不了，
        再往右才是两块屏幕之间的暗边。停在暗边前，避免吃进邻屏。
        找不到黑边时仍向右扩约 1 列（不超过画布），保住 12:45/13:00 一类末列。
        """
        h, w = img.shape[:2]
        y0 = max(0, min(int(y0), h - 1))
        y1 = max(y0 + 1, min(int(y1), h))
        x0 = max(0, min(int(x0), w - 1))
        x1 = max(x0 + 1, min(int(x1), w))
        if y1 - y0 < 12 or x1 >= w - 4:
            return x1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col = gray[y0:y1, :].mean(axis=0).astype(np.float32)
        body = col[x0:x1]
        body = body[body > 35]
        if body.size < 8:
            return x1
        body_med = float(np.median(body))
        dark_thr = min(body_med * 0.62, 96.0)
        min_run = max(12, int(w * 0.012))
        run = 0
        start = -1
        for x in range(x1, w):
            if float(col[x]) < dark_thr:
                if run == 0:
                    start = x
                run += 1
                if run >= min_run:
                    return max(x1, start - 3)
            else:
                run = 0
                start = -1
        # 无黑边（单屏/手机近景）：再扩约 1 列，避免末列贴裁切线
        extra = int(col_est) if col_est > 0 else max(int((x1 - x0) / 8.0), int(w * 0.06))
        return min(w, x1 + max(extra, int(w * 0.03)))

    def _default_table_crop(self, img: np.ndarray) -> np.ndarray:
        """
        蓝带失败时的默认裁剪：保留几乎整幅有效画面（含顶部表头）。
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

        通道策略（按长边自适应，避免大图多遍 OCR + CUBIC 放大摩尔纹）：
        - 暗表：原图 + 反色 CLAHE（2 路）
        - 浅色小图（长边 <1600）：原图 + CLAHE + 1.5×锐化 + 去摩尔纹
        - 浅色中图（<2000）：原图 + CLAHE + 去摩尔纹（最多 3 路）
        - 浅色大图：原图 + 去摩尔纹替换 CLAHE（固定 2 路，竖纹靠 demoiré 而不是再放大）
        返回 (图像, 相对原图的尺度)；放大通道 OCR 后须除以该尺度映射回原图坐标。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_v = float(np.mean(gray))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        h, w = gray.shape[:2]
        long_side = max(h, w)
        if mean_v < 110:
            inv = clahe.apply(cv2.bitwise_not(gray))
            return [(img, 1.0), (cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR), 1.0)]
        soft = cv2.bilateralFilter(gray, 5, 40, 40)
        enhanced = clahe.apply(soft)
        out: List[Tuple[np.ndarray, float]] = [
            (img, 1.0),
            (cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), 1.0),
        ]
        # 仅真正偏小的表再轻度放大锐化；中大图 CUBIC 放大只会拧尖摩尔纹
        if long_side < 1600:
            up_scale = 1.5
            up = cv2.resize(
                enhanced, None, fx=up_scale, fy=up_scale, interpolation=cv2.INTER_LINEAR
            )
            sharp = cv2.addWeighted(
                up, 1.25, cv2.GaussianBlur(up, (0, 0), 0.8), -0.25, 0
            )
            out.append((cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), up_scale))
        demoire = self._demoire_table_gray(gray)
        if demoire is not None and demoire.size:
            demoire_bgr = cv2.cvtColor(demoire, cv2.COLOR_GRAY2BGR)
            if long_side < 2000:
                out.append((demoire_bgr, 1.0))
            else:
                # 大图：demoiré 替换 CLAHE 路，仍只 2 遍全图 OCR
                out[1] = (demoire_bgr, 1.0)
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
    def _raw_time_hhmm_digit_counts(text: str) -> Optional[Tuple[int, int]]:
        """OCR 原文 (小时位数, 分钟位数)；无法解析为时刻则 None。

        界面完整列为零填充 HH:MM（如 01:15 / 12:30）。
        左缘裁断常剩 4:00（小时 1 位），右缘裁断常剩 12:3（分钟 1 位）——
        均不得当成完整列，禁止交给 _normalize_time 补位后误用。
        """
        raw = str(text or "").strip().replace("O", "0").replace("o", "0")
        raw = raw.replace("l", "1").replace("I", "1").replace("|", "1")
        raw = raw.replace("：", ":").replace("．", ".").replace(";", ":")
        # 允许分钟 1～2 位，以便检出右缘残片 12:3（旧式要求 \d{2} 会漏判）
        m = re.match(r"^(\d{1,2})[:：.\-](\d{1,2})(?:[:：]\d{2})?$", raw)
        if m:
            return len(m.group(1)), len(m.group(2))
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 4:
            return 2, 2
        if len(digits) == 3:
            # 115 / 123 → 缺分隔的残片，视为不完整
            return 1, 1
        return None

    @classmethod
    def _raw_time_is_complete_hhmm(cls, text: str) -> bool:
        """原文是否已是完整零填充 HH:MM（两位小时 + 两位分钟）。"""
        parts = cls._raw_time_hhmm_digit_counts(text)
        return parts is not None and parts[0] >= 2 and parts[1] >= 2

    @classmethod
    def _raw_time_hour_digit_count(cls, text: str) -> Optional[int]:
        """OCR 原文小时位数；无法解析为时刻则 None（兼容旧调用）。"""
        parts = cls._raw_time_hhmm_digit_counts(text)
        return parts[0] if parts else None

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

    def _trailing_glyph_clipped(
        self,
        img: np.ndarray,
        cell: OcrCell,
        others: Sequence[OcrCell],
    ) -> bool:
        """最右时刻的末位笔画是否明显细于其余列的末位（裁掉半个 0）。

        右缘常见 01:30 → 只剩 01:3 + 半个 0，OCR 仍可能补全成完整 HH:MM。
        只跟其它列末位笔画比；摩尔纹下测不到墨迹段则返回 False，交给矩阵侧丢列。
        """
        if img is None or not others:
            return False
        own = self._glyph_run_widths(img, cell)
        if not own:
            return False
        trails: List[int] = []
        for other in others:
            runs = self._glyph_run_widths(img, other)
            if runs:
                trails.append(int(runs[-1]))
        if len(trails) < 3:
            return False
        med = float(np.median(trails))
        if med < 3.0:
            return False
        trail = float(own[-1])
        return trail + 1.5 < med and trail < med * 0.65

    def _left_time_leading_one_is_border_artifact(
        self,
        img: Optional[np.ndarray],
        cell: OcrCell,
        others: Sequence[OcrCell],
        *,
        img_width: int = 0,
    ) -> bool:
        """最左时刻小时前导「1」是否像左边框竖线误读（如 3:00→13:00）。

        拍屏时表左边框/格线贴着个位小时，OCR 常多读出一位 1，整列应舍弃。
        注意：真·15:15 / 16:00 也以「1」开头，「1」笔画天然细于「5/6」，
        禁止用「本格首位 vs 后续笔画」误杀完整首列；只认贴边或相对邻列首位被裁。
        """
        raw = str(cell.text or "").strip().replace("O", "0").replace("o", "0")
        raw = raw.replace("l", "1").replace("I", "1").replace("|", "1")
        raw = raw.replace("：", ":").replace("．", ".").replace(";", ":")
        if not re.match(r"^1\d[:：.\-]", raw):
            return False
        # 检测框贴齐画布左缘：边框极易并入首位
        if img_width > 0 and float(cell.x) <= max(
            4.0, min(float(img_width) * 0.015, 14.0)
        ):
            return True
        if img is None:
            return False
        # 仅与其它时刻列的首位比：半个 1 被裁才会明显细于邻列的 1
        return self._leading_glyph_clipped(img, cell, others)

    def _row_time_candidates(
        self,
        row: Sequence[OcrCell],
        *,
        img_width: int = 0,
        img: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """收集行内时刻；两侧非完整 HH:MM、贴齐裁切线、或最左/最右列笔画被裁一律丢弃。

        现货表头固定零填充 HH:MM。左缘残片常读成 4:00，右缘残片常读成 12:3；
        若放行会被 _normalize_time 补成 04:00 / 误吸附，故收候选时先要求
        原文已是两位小时 + 两位分钟，再进入序列整理与边缘列判定。
        另：首位只裁一丁点时框并不贴画布，OCR 仍可能读成完整 10:15——
        仅最左列、且首位墨迹明显细于其余列时不采纳。
        右缘对称：末位只剩半个 0 时 OCR 仍可能补成 01:30——最右列末位墨迹
        明显细于其余列时不采纳。
        左边框竖线误读成小时前导「1」（3:00→13:00）同样不采纳。
        """
        time_cells = [
            c
            for c in row
            if self._raw_time_hhmm_digit_counts(c.text) is not None
            or self._normalize_time(c.text)
        ]
        leftmost = min(time_cells, key=lambda c: c.x) if time_cells else None
        rightmost = (
            max(time_cells, key=lambda c: float(c.x) + float(c.width or 0))
            if time_cells
            else None
        )
        edge_pad = (
            max(4.0, min(float(img_width) * 0.015, 14.0)) if img_width > 0 else 4.0
        )
        drop_left = False
        drop_right = False
        # 完整 HH:MM 且不贴边：禁止用笔画宽误杀（摩尔纹下 15:15 的「1」常被判细于邻列）
        # 仅贴左/右裁切线时才做边框伪影 / 首末位裁断检测。
        if leftmost is not None and img_width > 0:
            touches_left = float(leftmost.x) <= edge_pad
            if touches_left:
                drop_left = self._left_time_leading_one_is_border_artifact(
                    img,
                    leftmost,
                    [c for c in time_cells if c is not leftmost],
                    img_width=img_width,
                )
                if not drop_left and img is not None:
                    drop_left = self._leading_glyph_clipped(
                        img, leftmost, [c for c in time_cells if c is not leftmost]
                    )
        if rightmost is not None and img is not None and img_width > 0:
            right_x = float(rightmost.x) + float(rightmost.width or 0)
            touches_right = right_x >= float(img_width) - edge_pad
            if touches_right:
                drop_right = self._trailing_glyph_clipped(
                    img, rightmost, [c for c in time_cells if c is not rightmost]
                )
        times: List[Tuple[str, float]] = []
        for cell in row:
            # 两侧统一：原文必须完整 HH:MM，禁止补位后再当合法列
            if not self._raw_time_is_complete_hhmm(cell.text):
                continue
            touches = self._cell_touches_image_edge(cell, img_width)
            # 贴齐裁切线：半截仍读成完整 HH:MM 也丢。
            # 左右缘例外：框宽已接近正常时刻宽度时，更像完整列贴边，予以保留
            # （否则 15:15/17:00 等完整靠外列常因贴裁切线被整列丢掉）。
            # 但首/末位笔画已明显被裁时仍丢——避免半截被补全成完整 HH:MM。
            if touches:
                pad = max(4.0, min(float(img_width) * 0.015, 14.0)) if img_width > 0 else 4.0
                left = float(cell.x)
                right = left + float(cell.width or 0)
                near_left = img_width > 0 and left <= pad
                near_right = img_width > 0 and right >= float(img_width) - pad
                wide_enough = float(cell.width or 0) >= max(28.0, float(img_width) * 0.035)
                peers = [c for c in time_cells if c is not cell]
                lead_clipped = (
                    near_left
                    and img is not None
                    and self._leading_glyph_clipped(img, cell, peers)
                )
                trail_clipped = (
                    near_right
                    and img is not None
                    and self._trailing_glyph_clipped(img, cell, peers)
                )
                keep_left = near_left and wide_enough and not lead_clipped
                keep_right = near_right and wide_enough and not trail_clipped
                if not (keep_left or keep_right):
                    continue
            if drop_left and cell is leftmost:
                continue
            if drop_right and cell is rightmost:
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
                    float(np.median([c.height for c in rows[best_idx]])) * 3.0,
                )
            if img is not None and getattr(img, "shape", None) is not None:
                # 斜拍时同一表头左右会被拆到相邻行，间距大于一字高
                y_tol = max(y_tol, float(img.shape[0]) * 0.07)
            merged: List[Tuple[str, float]] = list(best_times)
            for idx, row in enumerate(rows):
                if idx == best_idx:
                    continue
                if abs(self._row_y_center(row) - base_y) > y_tol:
                    continue
                neigh = self._row_time_candidates(row, img_width=img_width, img=img)
                if not neigh:
                    continue
                # 聚类把同一行表头拆成两簇时 Y 几乎重合，必须并入左右缘时刻
                same_line = abs(self._row_y_center(row) - base_y) <= max(
                    12.0, float(np.median([c.height for c in rows[best_idx]] or [20])) * 0.45
                )
                # 邻行若几乎全是大步时刻（曲线轴），勿并入
                neigh_mins = [
                    m
                    for m in (self._time_to_minutes(t) for t, _ in neigh)
                    if m is not None
                ]
                if not same_line and len(neigh_mins) >= 3:
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
        保留 OCR 原文里能组成 Δ=15 的表头列（按 X 序）。不改写、不补造时刻。

        - 丢掉接不上的脏列（如误读成 06:45 插在 05:30 与 06:00 之间）
        - 中间跳开时，两端各自已经识别出的 Δ=15 段都保留，不编造中间缺格
        - 列距像漏格时把两端并成一组；并不上也不丢较短的那一端
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

        # 3) 每一段 Δ=15 都留下（含中间跳开后的两端），不单留最长的一段
        kept: List[int] = []
        for group in groups:
            if len(group) >= 2:
                kept.extend(group)
        if len(kept) < 2:
            return [order[i] for i in range(n) if mins[i] is not None]
        kept.sort(key=lambda p: xs[p])
        return [order[p] for p in kept]

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

    def _repair_plus60_header_hours(
        self,
        headers: List[str],
        col_centers: List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        小时位 OCR 把 5 读成 6 时改回（15:15→16:15），列位置不动。

        只在已经有一段连续 +15 的锚上改：当前格比「上一格 +15」刚好多 60 分钟，
        且后面还有一格时刻落在二者之间（证明这一格的小时偏大，不是真跳时）。
        """
        n = min(len(headers), len(col_centers))
        if n < 4:
            return list(headers), list(col_centers[: len(headers)])
        mins: List[Optional[int]] = [self._time_to_minutes(headers[i]) for i in range(n)]
        solid = 0
        changed = False
        for i in range(1, n):
            prev, cur = mins[i - 1], mins[i]
            if prev is None or cur is None:
                solid = 0
                continue
            if cur - prev == 15:
                solid += 1
                continue
            expected = prev + 15
            if (
                solid < 2
                or cur - expected != 60
                or expected > 24 * 60
                or (cur % 60) != (expected % 60)
            ):
                solid = 0
                continue
            later = [m for m in mins[i + 1 :] if m is not None]
            if not any(expected < m < cur for m in later):
                solid = 0
                continue
            mins[i] = expected
            changed = True
            solid += 1
        if not changed:
            return list(headers), list(col_centers[:n])
        out = [
            self._minutes_to_time(mins[i]) if mins[i] is not None else headers[i]
            for i in range(n)
        ]
        return out, list(col_centers[:n])

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

    @staticmethod
    def _fit_sticky_map(
        sticky_map: Optional[List[List[str]]],
        matrix: Sequence[Sequence[str]],
    ) -> List[List[str]]:
        """把 sticky_map 裁成与 matrix 同形（缺行/列补空串）。"""
        nrows = len(matrix)
        ncols = max((len(r) for r in matrix), default=0) if matrix else 0
        src = sticky_map or []
        out: List[List[str]] = []
        for i in range(nrows):
            row = list(src[i]) if i < len(src) else []
            if len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            else:
                row = row[:ncols]
            out.append(row)
        return out

    @staticmethod
    def _clear_sticky_where_empty(
        sticky_map: List[List[str]],
        matrix: Sequence[Sequence[str]],
    ) -> List[List[str]]:
        """空值格去掉粘连标记（表头泄漏清洗等之后对齐）。"""
        out = TableImageExtractor._fit_sticky_map(sticky_map, matrix)
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                if not v or not str(v).strip() or v == TableImageExtractor.NODATA_MARK:
                    if j < len(out[i]):
                        out[i][j] = ""
        return out

    def _expand_row_cells(
        self,
        row: List[OcrCell],
        col_centers: Optional[Sequence[float]] = None,
    ) -> List[OcrCell]:
        """
        将粘连 OCR 块拆成多个虚拟单元格（按列宽估算 X）。

        宽框盖住 ≥2 列：优先已点对电价链拆成虚拟格；切不开则原样保留，
        交给落格阶段的可信拆分 / 按列几何重识。
        单列内粘连仍做点对齐链 / 文本拆。
        """
        result: List[OcrCell] = []
        step = (
            self._median_col_step(list(col_centers))
            if col_centers and len(col_centers) >= 2
            else 0.0
        )
        for cell in row:
            wide = (
                bool(col_centers)
                and step > 0
                and self._is_wide_multicolumn_sticky_cell(cell, col_centers, step)
            )
            # 宽框 / 单列：已点对电价链按字符位置落框（拆分优先于几何重识）
            placed = self._cells_from_price_chain(cell)
            if placed:
                result.extend(placed)
                continue
            if wide:
                # 点对链切不开：原样保留，落格再试 N 列切 / 待重识
                result.append(cell)
                continue
            # 单点/无点超长：单列内仍可能粘两格，交给文本拆；多列 glue 原样保留
            if self._looks_like_multicolumn_glue_text(cell.text):
                result.append(cell)
                continue
            parts = self._split_cell_values(cell.text)
            if len(parts) <= 1:
                result.append(cell)
                continue
            n = len(parts)
            seg = cell.width / max(n, 1)
            for i, val in enumerate(parts):
                result.append(OcrCell(
                    text=val, confidence=cell.confidence,
                    x=cell.x + seg * i, y=cell.y,
                    width=max(seg * 0.9, 8), height=cell.height,
                    sticky="粘连拆分",
                ))
        return result

    def _try_wide_sticky_reliable_parts(
        self,
        cell: "OcrCell",
        covered: Sequence[int],
    ) -> List[str]:
        """
        宽框可信文本拆分：点对链优先，其次按覆盖列数 N 切纯数字。

        段数必须等于覆盖列数，且通过量级门禁；否则返回空（交给几何重识）。
        """
        cols = [int(j) for j in covered]
        n = len(cols)
        if n < 2:
            return []
        text = self._normalize_numeric_text(cell.text)
        parts = self._extract_dotted_price_chain(text)
        if parts:
            dotted = [p for p in parts if "." in str(p)]
            if len(dotted) == n and self._sticky_pair_plausible(dotted):
                return dotted
            if len(parts) == n and self._sticky_pair_plausible(parts):
                return list(parts)
        digits = re.sub(r"\D", "", text)
        nparts = self._split_digit_run_into_n_prices(digits, n)
        if len(nparts) == n and self._sticky_pair_plausible(nparts):
            return nparts
        return []

    def _normalize_numeric_text(self, text: str) -> str:
        """只保留数字、小数点、负号，供粘连电价定位。"""
        t = str(text or "").strip().replace("O", "0").replace("o", "0").replace(":", ".")
        t = t.replace("−", "-").replace("–", "-").replace("—", "-")
        return re.sub(r"[^\d.\-]", "", t)

    def _price_chain_spans(
        self, src: str, parts: Sequence[str]
    ) -> Optional[List[Tuple[int, int]]]:
        """每个电价在原串中的字符区间；漏点的 6 位段按数字对齐。"""
        spans: List[Tuple[int, int]] = []
        pos = 0
        for part in parts:
            idx = src.find(part, pos)
            if idx >= 0:
                spans.append((idx, idx + len(part)))
                pos = idx + len(part)
                continue
            digits = part.replace(".", "")
            if not digits or not digits.isdigit():
                return None
            start: Optional[int] = None
            got = 0
            end = pos
            for i in range(pos, len(src)):
                ch = src[i]
                if ch.isdigit():
                    if start is None:
                        start = i
                    if got >= len(digits) or ch != digits[got]:
                        return None
                    got += 1
                    end = i + 1
                    if got == len(digits):
                        break
                elif ch == ".":
                    end = i + 1
                else:
                    return None
            if got != len(digits) or start is None:
                return None
            spans.append((start, end))
            pos = end
        return spans

    def _cells_from_price_chain(self, cell: "OcrCell") -> List[OcrCell]:
        """
        粘连电价按各自在检测框里的字符位置生成虚拟格。

        原文与拆分结果一致时返回空，沿用原框。
        """
        src = self._normalize_numeric_text(cell.text)
        parts = self._extract_dotted_price_chain(src)
        if not parts:
            return []
        if len(parts) == 1 and parts[0] == src:
            return []
        if len(parts) >= 2 and not self._sticky_pair_plausible(parts):
            return []
        spans = self._price_chain_spans(src, parts)
        if not spans or len(spans) != len(parts):
            return []
        n = max(len(src), 1)
        width = max(float(cell.width or 0), 8.0)
        out: List[OcrCell] = []
        for part, (a, b) in zip(parts, spans):
            span = max(b - a, 1)
            out.append(
                OcrCell(
                    text=part,
                    confidence=cell.confidence,
                    x=float(cell.x) + width * (float(a) / float(n)),
                    y=cell.y,
                    width=max(width * (float(span) / float(n)), 8.0),
                    height=cell.height,
                    sticky="粘连拆分",
                )
            )
        return out

    def _cell_text_looks_numeric_value(self, text: str) -> bool:
        """是否像表体数值（非时刻/页脚），供宽框粘连判定。"""
        raw = str(text or "").strip()
        if not raw:
            return False
        if self._is_footer_text(raw):
            return False
        # 时刻表头碎片
        if re.fullmatch(r"\d{1,2}:\d{2}", raw) or self._normalize_time(raw):
            return False
        t = raw.replace("O", "0").replace("o", "0")
        t = t.replace("−", "-").replace("–", "-").replace("—", "-")
        t = re.sub(r"[^\d.\-]", "", t)
        digits = re.sub(r"[.\-]", "", t)
        return len(digits) >= 1

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

    def _is_wide_multicolumn_sticky_cell(
        self,
        cell: "OcrCell",
        col_centers: Sequence[float],
        step: float,
    ) -> bool:
        """
        检测框盖住 ≥2 列且像表体数值 → 宽框粘连。

        优先可信文本拆（点对链 / N 列切）；切不开再按列几何重识。
        禁止无约束的均衡瞎切。
        """
        if not self._cell_text_looks_numeric_value(cell.text):
            return False
        if self._is_lone_dash(cell.text):
            return False
        covered = self._columns_covered_by_cell(cell, col_centers, step)
        return len(covered) >= 2

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
    def _edge_margin_px(
        img_width: int, step_x: float = 60.0, *, side: str = "left"
    ) -> float:
        """本图裁切半列判定带。

        只拦「半列中心」典型落点；完整靠外列中心约在 0.5×列宽，须保留。
        左缘约 0.28 列宽（旧 0.40 易误杀紧贴裁切线的完整首列）。
        右缘约 0.20 列宽：完整末列中心常落在 0.3～0.5×step。
        """
        if img_width <= 0:
            return max(8.0, step_x * (0.20 if side == "right" else 0.28))
        if side == "right":
            return max(6.0, min(img_width * 0.02, step_x * 0.20))
        return max(8.0, min(img_width * 0.03, step_x * 0.28))

    def _drop_edge_clipped_columns(
        self,
        headers: List[str],
        col_centers: List[float],
        img_width: int,
        data_cells: Optional[Sequence[OcrCell]] = None,
    ) -> Tuple[List[str], List[float]]:
        """丢掉本图边缘被裁断的半列。

        1) 中心已出画布，或落入贴边带（半列中心典型位置）则丢弃。
           完整列即便靠外缘，中心通常仍在约 0.5×列宽处，予以保留。
        2) 表头完整但横向滚动裁断数据时：须有数值形态残片证据才丢首/末列；
           不得仅因贴边+略窄误杀完整靠外列（如 15:15 / 17:00）。
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
        margin_l = self._edge_margin_px(img_width, step_x, side="left")
        margin_r = self._edge_margin_px(img_width, step_x, side="right")
        w = float(img_width)
        keep_h: List[str] = []
        keep_c: List[float] = []
        for i in range(n):
            cx = float(col_centers[i])
            if cx < 0.0 or cx > w:
                continue
            if cx < margin_l or cx > w - margin_r:
                continue
            keep_h.append(headers[i])
            keep_c.append(cx)
        if len(keep_h) < 2:
            return list(headers[:n]), list(col_centers[:n])
        if data_cells:
            keep_h, keep_c = self._drop_data_edge_clipped_columns(
                keep_h, keep_c, data_cells, img_width=img_width
            )
        if len(keep_h) < 2:
            return list(headers[:n]), list(col_centers[:n])
        return keep_h, keep_c

    def _cells_for_column(
        self,
        cells: Sequence[OcrCell],
        col_centers: Sequence[float],
        col_idx: int,
        *,
        step_x: float,
    ) -> List[OcrCell]:
        """落在指定列中心附近的数值 OCR 块（不含时刻）。"""
        n = len(col_centers)
        if n <= 0 or col_idx < 0 or col_idx >= n:
            return []
        cx = float(col_centers[col_idx])
        tol = max(step_x * 0.55, 28.0)
        out: List[OcrCell] = []
        for cell in cells:
            if self._normalize_time(cell.text) or self._is_footer_text(cell.text):
                continue
            # 宽框多列粘连不参与单列/边缘裁断判定（否则会误杀末列）
            if self._is_wide_multicolumn_sticky_cell(cell, col_centers, step_x):
                continue
            parts = self._split_cell_values(cell.text)
            if not parts and not re.search(r"\d", str(cell.text or "")):
                continue
            cell_cx = float(cell.x) + float(cell.width or 0) * 0.5
            if abs(cell_cx - cx) > tol:
                continue
            nearest = min(
                range(n),
                key=lambda j: abs(cell_cx - float(col_centers[j])),
            )
            if nearest != col_idx:
                continue
            out.append(cell)
        return out

    def _cells_for_interior_columns(
        self,
        cells: Sequence[OcrCell],
        col_centers: Sequence[float],
        *,
        step_x: float,
    ) -> List[OcrCell]:
        """内列（去掉首末）数值块，作边缘裁断的形态参照。"""
        n = len(col_centers)
        if n < 3:
            return []
        out: List[OcrCell] = []
        seen = set()
        for j in range(1, n - 1):
            for c in self._cells_for_column(cells, col_centers, j, step_x=step_x):
                key = (round(float(c.x), 1), round(float(c.y), 1), str(c.text))
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
        return out

    @staticmethod
    def _token_core_digits(text: str) -> str:
        """数值 token 的核心数字串（去符号/杂字符）。"""
        t = str(text or "").strip().replace("O", "0").replace("o", "0")
        t = t.replace("#", "").replace(",", "")
        m = re.search(r"-?\d+(?:\.\d+)?", t)
        return m.group(0).lstrip("-") if m else ""

    @staticmethod
    def _token_int_frac_lens(tok: str) -> Optional[Tuple[int, int]]:
        """数值 token → (整数位数, 小数位数)；纯整数 frac=0；无法解析则 None。"""
        raw = str(tok or "").strip().lstrip("#").lstrip("-")
        if not raw or not re.search(r"\d", raw):
            return None
        if raw.startswith("."):
            return 0, len(re.sub(r"\D", "", raw) or "")
        if "." in raw:
            a, _, b = raw.partition(".")
            return (
                len(re.sub(r"\D", "", a) or ""),
                len(re.sub(r"\D", "", b) or ""),
            )
        digits = re.sub(r"\D", "", raw)
        return (len(digits), 0) if digits else None

    def _column_outer_tokens(
        self, cells: Sequence[OcrCell], *, side: str
    ) -> List[str]:
        """列内数值 token；边缘列只取最靠外缘的一段（粘连时）。"""
        toks: List[str] = []
        for c in cells:
            parts = self._split_cell_values(c.text)
            if parts:
                toks.append(parts[0] if side == "left" else parts[-1])
                continue
            core = self._token_core_digits(c.text)
            if core:
                toks.append(core)
        return toks

    def _column_all_tokens(self, cells: Sequence[OcrCell]) -> List[str]:
        """列内全部数值 token（内列参照用）。"""
        toks: List[str] = []
        for c in cells:
            parts = self._split_cell_values(c.text)
            if parts:
                toks.extend(parts)
                continue
            core = self._token_core_digits(c.text)
            if core:
                toks.append(core)
        return toks

    @staticmethod
    def _median_or_none(vals: Sequence[float]) -> Optional[float]:
        if not vals:
            return None
        return float(np.median(np.asarray(vals, dtype=np.float64)))

    @staticmethod
    def _cell_y_center(cell: OcrCell) -> float:
        return float(cell.y) + float(cell.height or 0) * 0.5

    def _same_row_interior_shapes(
        self,
        edge_cell: OcrCell,
        interior_cells: Sequence[OcrCell],
        *,
        y_tol: float,
    ) -> List[Tuple[int, int]]:
        """与边缘格同行的内列数值形态 (int_len, frac_len)。"""
        ey = self._cell_y_center(edge_cell)
        shapes: List[Tuple[int, int]] = []
        for c in interior_cells:
            if abs(self._cell_y_center(c) - ey) > y_tol:
                continue
            for tok in self._column_all_tokens([c]):
                s = self._token_int_frac_lens(tok)
                if s:
                    shapes.append(s)
        return shapes

    @staticmethod
    def _is_strong_left_fragment(
        edge_tok: str,
        edge_shape: Tuple[int, int],
        row_shapes: Sequence[Tuple[int, int]],
    ) -> bool:
        """左缘强残片：相对同行一眼缺高位，整列应丢、不可采纳。

        例：同行 355.862，边缘却是 .45 / 0.33 / 3.187 / 54.082 / 444
        （半截电价缺高位；与右缘「小数尾被裁」对称）。
        同行本身也是 0.xx 小数量级时不判强残片。
        完整列同量级（264.861 vs 293.187）整数位相同，不会误杀。
        """
        if not row_shapes:
            return False
        e_int, e_frac = edge_shape
        raw = str(edge_tok or "").strip().lstrip("-")
        row_dec = [s for s in row_shapes if s[1] > 0]
        row_pure = [s for s in row_shapes if s[1] == 0 and s[0] > 0]
        # 同行全是带点小数，左缘却是纯整数：269.444→444 / 333.432→333
        # 须在 med_int 门槛之前判定（残片整数位可能与同行整数位同长）
        if e_frac == 0 and e_int > 0 and row_dec and not row_pure:
            med_frac = float(np.median([s[1] for s in row_dec]))
            if med_frac >= 2:
                return True
        peers = (
            row_dec
            if (e_frac > 0 or e_int == 0) and row_dec
            else row_pure or list(row_shapes)
        )
        med_int = float(np.median([s[0] for s in peers]))
        if med_int < 2:
            return False
        # .45 / .187
        if raw.startswith("."):
            return True
        # 0.33 / 0.45：OCR 常把裁掉高位后的小数补成 0.xx
        if re.match(r"^0\.\d{1,3}$", raw):
            return True
        # 3.187 / 1.14：整数位只剩 1 位，相对同行 ≥2 位
        if re.match(r"^\d\.\d{1,3}$", raw) and e_int <= 1:
            return True
        # 54.082 vs 355.xxx：仍带小数但整数位少 ≥1（高位被裁）
        if e_frac > 0 and e_int + 0.5 < med_int:
            return True
        return False

    def _edge_row_shape_looks_clipped(
        self,
        edge_tok: str,
        edge_shape: Tuple[int, int],
        row_shapes: Sequence[Tuple[int, int]],
        *,
        side: str,
    ) -> bool:
        """单行：边缘 token 相对同行内列是否像被横向裁断。"""
        if not row_shapes:
            return False
        e_int, e_frac = edge_shape
        raw = str(edge_tok or "").lstrip("-")

        if side == "left":
            if self._is_strong_left_fragment(edge_tok, edge_shape, row_shapes):
                return True
            row_dec = [s for s in row_shapes if s[1] > 0]
            row_pure = [s for s in row_shapes if s[1] == 0 and s[0] > 0]
            # 与右缘对称：同行全是小数，左缘却成纯整数 → 高位/点被裁
            if e_frac == 0 and e_int > 0 and row_dec and not row_pure:
                med_frac = float(np.median([s[1] for s in row_dec]))
                if med_frac >= 2:
                    return True
            peers = row_dec if (e_frac > 0 or e_int == 0) and row_dec else (
                row_pure or list(row_shapes)
            )
            med_int = float(np.median([s[0] for s in peers]))
            if med_int >= 2 and e_int + 0.5 < med_int:
                return True
            return False

        # 右缘：同行比小数位（功率整数行 vs 电价小数行互不串行）
        row_dec = [s for s in row_shapes if s[1] > 0]
        row_pure = [s for s in row_shapes if s[1] == 0 and s[0] > 0]
        # 右缘偶发 .xx（粘连拆段）：相对同行多位小数也视为残片
        if raw.startswith(".") and row_dec:
            med_frac = float(np.median([s[1] for s in row_dec]))
            if med_frac >= 2:
                return True
        if e_frac > 0 or e_int == 0:
            peers = row_dec or list(row_shapes)
            med_frac = float(np.median([s[1] for s in peers]))
            if med_frac >= 2 and e_frac + 0.5 < med_frac:
                return True
            # 同行整数位明显更长，边缘却是 0.xx / d.xxx
            med_int = float(np.median([s[0] for s in peers]))
            if med_int >= 2 and e_int <= 1 and e_frac <= 3:
                if raw.startswith("0.") or re.match(r"^\d\.\d{1,3}$", raw):
                    return True
            return False
        # 边缘是纯整数
        if row_dec and not row_pure:
            # 同行全是小数，边缘却成整数 → 小数尾被裁（333.432→333）
            med_frac = float(np.median([s[1] for s in row_dec]))
            return med_frac >= 2
        if row_pure:
            med_len = float(np.median([s[0] for s in row_pure]))
            if med_len >= 2 and e_int + 0.5 < med_len:
                return True
        return False

    def _edge_column_shape_clipped_vs_interior(
        self,
        edge_cells: Sequence[OcrCell],
        interior_cells: Sequence[OcrCell],
        *,
        side: str,
        step_x: float,
        img_width: int = 0,
    ) -> bool:
        """相对内列形态/框宽判断边缘列是否被横向裁断。

        按行对齐比较：边缘格只与同行内列比整数位/小数位，避免功率整数行
        与电价小数行跨行混算（如末列 550/550/322.249 被误杀）。
        左缘强残片（.xx / 0.33 / 3.187 / 同行全小数却读成 444）一票丢列、不采纳；
        其余看整数位/小数位系统性变短。

        完整靠外列常贴裁切线、框略窄：不得仅凭 touches+narrow 误杀；
        须有形态残片票，或「明显窄框 + 贴边 + 至少一票形态异常」。
        """
        if not edge_cells or not interior_cells:
            return False

        def _widths(cells: Sequence[OcrCell]) -> List[float]:
            ws: List[float] = []
            for c in cells:
                w = float(c.width or 0)
                if 8.0 <= w <= step_x * 0.95:
                    ws.append(w)
            return ws

        edge_w = self._median_or_none(_widths(edge_cells))
        int_w = self._median_or_none(_widths(interior_cells))
        # 旧 0.88 过敏：完整贴边列框宽常略窄于内列
        narrow = (
            edge_w is not None
            and int_w is not None
            and int_w >= 20.0
            and edge_w < int_w * 0.72
        )
        touch_n = sum(
            1
            for c in edge_cells
            if self._cell_touches_image_edge(c, img_width)
        )
        touches = img_width > 0 and touch_n >= max(1, (len(edge_cells) + 1) // 2)

        heights = [float(c.height or 0) for c in edge_cells if float(c.height or 0) > 0]
        y_tol = max(28.0, float(np.median(heights)) * 1.35 if heights else 28.0)

        comparable = 0
        clipped_votes = 0
        strong_frag = 0
        for cell in edge_cells:
            toks = self._column_outer_tokens([cell], side=side)
            # 原始文本：_split_cell_values 会把 ".45" 收成 "45"，强残片要用原文
            raw_cell = str(cell.text or "").strip().lstrip("#").lstrip("-")
            lead_dot = bool(re.match(r"^\.\d+", raw_cell))
            frag_tok = raw_cell if lead_dot else (toks[0] if toks else raw_cell)
            if not toks and not frag_tok:
                continue
            tok = toks[0] if toks else frag_tok
            shape = self._token_int_frac_lens(frag_tok if lead_dot else tok)
            if not shape:
                continue
            row_shapes = self._same_row_interior_shapes(
                cell, interior_cells, y_tol=y_tol
            )
            if not row_shapes:
                continue
            comparable += 1
            if side == "left" and self._is_strong_left_fragment(
                frag_tok, shape, row_shapes
            ):
                strong_frag += 1
                clipped_votes += 1
                continue
            if self._edge_row_shape_looks_clipped(
                frag_tok if lead_dot else tok, shape, row_shapes, side=side
            ):
                clipped_votes += 1
                raw = str(frag_tok if lead_dot else tok).lstrip("-")
                # 右缘 .xx / 0.xx 相对多位小数，也按强残片一票丢
                if side == "right" and (
                    raw.startswith(".") or re.match(r"^0\.\d{1,3}$", raw)
                ):
                    row_dec = [s for s in row_shapes if s[1] > 0]
                    if row_dec and float(np.median([s[0] for s in row_dec])) >= 2:
                        strong_frag += 1

        if comparable <= 0:
            # 无形态可比时：仅「明显窄 + 贴边」仍不够，避免空列/漏检误杀
            return False

        # 强残片：.45 / 0.33 等一眼有问题 → 整列丢弃，不采纳
        if strong_frag >= 1:
            return True
        # 多数同行形态异常才丢；弱信号需贴边/窄框，且至少 2 票
        majority = clipped_votes >= max(1, (comparable + 1) // 2)
        if majority and clipped_votes >= 2:
            return True
        if majority and clipped_votes >= 1 and narrow and touches:
            return True
        return False

    def _edge_column_data_looks_clipped(
        self,
        col_centers: Sequence[float],
        col_idx: int,
        data_cells: Sequence[OcrCell],
        *,
        side: str,
        img_width: int = 0,
    ) -> bool:
        """首/末列数据是否像横向滚动裁断（表头仍可能完整）。

        参照内列（非邻列具体数值）的整数位/小数位分布与框宽：
        左缘缺高位、右缘缺小数尾；不要求边缘值与邻列相等或互为前后缀。
        """
        n = len(col_centers)
        if n < 3 or col_idx < 0 or col_idx >= n:
            return False
        if side == "left" and col_idx != 0:
            return False
        if side == "right" and col_idx != n - 1:
            return False
        x_diffs = [
            float(col_centers[i + 1]) - float(col_centers[i])
            for i in range(n - 1)
            if float(col_centers[i + 1]) > float(col_centers[i])
        ]
        step_x = float(np.median(x_diffs)) if x_diffs else 60.0
        edge_cells = self._cells_for_column(
            data_cells, col_centers, col_idx, step_x=step_x
        )
        interior_cells = self._cells_for_interior_columns(
            data_cells, col_centers, step_x=step_x
        )
        if not edge_cells or not interior_cells:
            return False
        return self._edge_column_shape_clipped_vs_interior(
            edge_cells,
            interior_cells,
            side=side,
            step_x=step_x,
            img_width=img_width,
        )

    def _collect_edge_clip_data_cells(
        self,
        rows: Sequence[Sequence[OcrCell]],
        header_idx: int,
        data_cells: Sequence[OcrCell],
    ) -> List[OcrCell]:
        """汇总用于边缘裁断判定的数据 OCR 块（不含表头时刻行）。"""
        out: List[OcrCell] = []
        seen = set()
        for c in self._expand_row_cells(list(data_cells or [])):
            key = (round(float(c.x), 1), round(float(c.y), 1), str(c.text))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        for i, row in enumerate(rows):
            if header_idx >= 0 and i == header_idx:
                continue
            for c in row:
                if self._normalize_time(c.text):
                    continue
                key = (round(float(c.x), 1), round(float(c.y), 1), str(c.text))
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
        return out

    def _drop_data_edge_clipped_columns(
        self,
        headers: List[str],
        col_centers: List[float],
        data_cells: Sequence[OcrCell],
        *,
        img_width: int = 0,
    ) -> Tuple[List[str], List[float]]:
        """按数据侧证据丢掉首/末被横向裁断的列（每侧最多一列，防级联误杀）。"""
        keep_h = list(headers)
        keep_c = list(col_centers)
        expanded = list(data_cells)
        if len(keep_h) >= 3 and self._edge_column_data_looks_clipped(
            keep_c, 0, expanded, side="left", img_width=img_width
        ):
            keep_h = keep_h[1:]
            keep_c = keep_c[1:]
        if len(keep_h) >= 3 and self._edge_column_data_looks_clipped(
            keep_c,
            len(keep_c) - 1,
            expanded,
            side="right",
            img_width=img_width,
        ):
            keep_h = keep_h[:-1]
            keep_c = keep_c[:-1]
        return keep_h, keep_c

    def _filter_cells_outside_header_span(
        self,
        rows: Sequence[Sequence[OcrCell]],
        header_idx: int,
        data_cells: Sequence[OcrCell],
        col_centers: Sequence[float],
    ) -> Tuple[List[List[OcrCell]], List[OcrCell]]:
        """去掉中心落在当前表头列跨度之外的数据块（含伸入已丢边缘列的粘连框）。"""
        n = len(col_centers)
        if n < 2:
            return [list(r) for r in rows], list(data_cells or [])
        x_diffs = [
            float(col_centers[i + 1]) - float(col_centers[i])
            for i in range(n - 1)
            if float(col_centers[i + 1]) > float(col_centers[i])
        ]
        step = float(np.median(x_diffs)) if x_diffs else 60.0
        # 左边界略严：框起点伸进已裁列半格以上则丢（残片粘连）
        x_min = float(col_centers[0]) - step * 0.48
        x_max = float(col_centers[-1]) + step * 0.55

        def _project_cell(cell: OcrCell) -> List[OcrCell]:
            """保留落在表头跨度内的块；跨已裁边界的电价粘连只留下落在内的段。"""
            if self._normalize_time(cell.text) or self._is_footer_text(cell.text):
                return [cell]
            right = float(cell.x) + float(cell.width or 0)
            if right < x_min or float(cell.x) > x_max:
                return []
            starts_left = float(cell.x) < float(col_centers[0]) - step * 0.55
            ends_right = right > float(col_centers[-1]) + step * 0.70
            if starts_left or ends_right:
                # 不再文本猜拆：与表头跨度有交集则整框保留，交给按列几何重识
                if right >= x_min and float(cell.x) <= x_max:
                    return [cell]
                return []
            return [cell]

        new_rows: List[List[OcrCell]] = []
        for i, row in enumerate(rows):
            if header_idx >= 0 and i == header_idx:
                new_rows.append(list(row))
                continue
            projected: List[OcrCell] = []
            for c in row:
                projected.extend(_project_cell(c))
            new_rows.append(projected)
        new_data: List[OcrCell] = []
        for c in data_cells or []:
            new_data.extend(_project_cell(c))
        return new_rows, new_data

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
        即使 Y 很近也是独立指标；全 0 双行只并左右残片，禁止上下并带。

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
        horiz = self._is_horizontal_complement(row, ref_row)
        # 电力/计划双行全 0 且行距常 < 0.45 字高：绝不能当「同线碎片」合并，
        # 否则只剩 1 条 0 带 → 中间空槽被粘连用电价带填上（第 3 行串进第 2 行）。
        # 仅左右残片（水平互补）才并。
        if zero_a and zero_b:
            return bool(horiz)
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
                # 电力/计划：双行全 0 即使 Y 很近也是独立指标，只并左右残片
                zero_pair = self._cells_are_zeroish(band) and self._cells_are_zeroish(
                    prev
                )
                horiz = self._is_horizontal_complement(band, prev)
                if zero_pair and not horiz:
                    continue
                if self._bands_look_like_full_duplicate_rows(band, prev):
                    if not same_line:
                        continue
                    # 满行同值（含双行全 0）：非左右残片不并
                    if not horiz:
                        continue
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
                    or (horiz and (mag_ok or zero_pair))
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
                           col_count: int) -> Tuple[List[List[str]], List[List[str]]]:
        """
        构建完整数据矩阵：headers 为时刻表头，此处只输出表头以下的数据行。
        同时返回与矩阵同形的 sticky_map（粘连业务处理标记）。
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
        sticky_map: List[List[str]] = []
        for band in bands:
            aligned, sticky_row = self._align_row_to_columns(
                self._expand_row_cells(band, col_centers), col_centers, col_count
            )
            if not aligned:
                continue
            pending_sticky = any(str(m).strip() for m in (sticky_row or []))
            if not self._is_valid_data_row(aligned) and not pending_sticky:
                continue
            if self._row_is_mostly_zeros(aligned) and not pending_sticky:
                # 真·读到过「0」才整行填 0（电力/计划常见）。
                # 若几乎全是空串：整行漏检（常见于电价），保留空留给列补识；
                # 切勿先填「0」——会挡住 _refill / 可疑电价重识。
                if self._row_has_actual_zero_token(aligned):
                    matrix.append(
                        [
                            self.NODATA_MARK if v == self.NODATA_MARK else "0"
                            for v in aligned
                        ]
                    )
                    sticky_map.append([""] * len(aligned))
                else:
                    matrix.append(aligned)
                    sticky_map.append(sticky_row)
            else:
                matrix.append(aligned)
                sticky_map.append(sticky_row)

        # 主数据行覆盖同带；禁止再追加出第 N+1 个重复测点
        if merged_main:
            main_aligned, main_sticky = self._align_row_to_columns(
                self._expand_row_cells(merged_main, col_centers),
                col_centers,
                col_count,
            )
            main_pending = any(str(m).strip() for m in (main_sticky or []))
            if main_aligned and (
                self._is_valid_data_row(main_aligned) or main_pending
            ):
                replaced = False
                for i, band in enumerate(bands[: len(matrix)]):
                    if self._is_same_indicator_fragment(band, merged_main):
                        # 同线双框时 band 往往已经挑过更好的读数，主行更差则不覆盖
                        # 粘连待重识空壳优先写入 sticky，供后续按列 OCR
                        if main_pending or self._aligned_row_strength(
                            main_aligned
                        ) > self._aligned_row_strength(matrix[i]):
                            matrix[i] = main_aligned
                            sticky_map[i] = main_sticky
                        replaced = True
                        break
                if not replaced:
                    for i, row in enumerate(matrix):
                        if self._aligned_rows_similar(row, main_aligned):
                            if main_pending or self._aligned_row_strength(
                                main_aligned
                            ) > self._aligned_row_strength(row):
                                matrix[i] = main_aligned
                                sticky_map[i] = main_sticky
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
                        sticky_map.append(main_sticky)
                    elif not similar and len(matrix) < 3:
                        # 不足 3 行时允许补主行（常见末行电价）
                        matrix.append(main_aligned)
                        sticky_map.append(main_sticky)

        if not matrix and merged_main:
            aligned, sticky_row = self._align_row_to_columns(
                self._expand_row_cells(merged_main, col_centers),
                col_centers,
                col_count,
            )
            if aligned:
                matrix = [aligned]
                sticky_map = [sticky_row]

        matrix = self._dedupe_aligned_matrix(matrix)
        sticky_map = self._fit_sticky_map(sticky_map, matrix)
        return self._enforce_expected_data_rows(
            matrix, sticky_map=sticky_map, expected=self.EXPECTED_DATA_ROWS
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
        sticky_map: Optional[List[List[str]]] = None,
        expected: int = 3,
    ):
        """
        硬约束：数据行恰好 expected 条（默认 3）。
        超量：丢掉最弱噪声行（保留小数电价行）。
        不足：电价沉底后在上方补全零行槽（供后续按列 OCR / 先验填 0）。

        传入 sticky_map 时返回 (matrix, sticky_map)；否则只返回 matrix（兼容旧调用）。
        """
        want_sticky = sticky_map is not None
        sticky = self._fit_sticky_map(sticky_map, matrix) if want_sticky else None
        if not matrix:
            return (matrix, sticky or []) if want_sticky else matrix
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
            if sticky is not None:
                sticky = [sticky[i] for i in keep]
        if len(matrix) < expected:
            matrix = self._pad_short_matrix_to_expected(matrix, expected=expected)
            if sticky is not None:
                sticky = self._fit_sticky_map(sticky, matrix)
        return (matrix, sticky or []) if want_sticky else matrix

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

    def _decimal_row_echoes_price(
        self, upper: Sequence[str], price: Sequence[str]
    ) -> bool:
        """
        上方小数行是否只是末行电价的渗漏副本。

        粘连电价拆片/几何重识常把电价同时写入第 2 行与末行；同列相同则清上方，留给补 0。
        """
        matched = 0
        compared = 0
        for i in range(min(len(upper), len(price))):
            a = str(upper[i] or "").strip().replace("−", "-").replace("–", "-")
            b = str(price[i] or "").strip().replace("−", "-").replace("–", "-")
            if not a or "." not in a:
                continue
            compared += 1
            if not b or "." not in b:
                continue
            try:
                fa, fb = float(a), float(b)
            except ValueError:
                continue
            scale = max(abs(fa), abs(fb), 1.0)
            if abs(fa - fb) <= max(0.05, scale * 0.02):
                matched += 1
        if compared < 2:
            return False
        return matched >= max(2, int(math.ceil(compared * 0.6)))

    def _scrub_price_tokens_from_upper_cells(
        self, matrix: List[List[str]]
    ) -> List[List[str]]:
        """
        单元格级防串行：整型电力/计划行里的电价样小数 → 清出上两行。

        典型：11:30 第二行空被下扫读成 273.25，第三行反而空。
        末行该列空时把值归还给电价行；末行已有值则只清上方。
        多小数独立指标表不清洗。禁止第三行补第二行。
        """
        if not matrix or len(matrix) < 2:
            return matrix
        if self._matrix_is_multi_decimal_indicators(matrix):
            return matrix
        out = [list(r) for r in matrix]
        price = out[-1]
        for ri in range(len(out) - 1):
            row = out[ri]
            if self._row_looks_like_decimal_indicator(row):
                continue
            if not self._row_is_integerish_or_empty(row):
                continue
            for ci in range(len(row)):
                u = str(row[ci] or "").strip()
                if not u or not self._looks_like_price_decimal_token(u):
                    continue
                out[ri][ci] = ""
                if ci >= len(price):
                    continue
                pv = str(price[ci] or "").strip()
                if not pv or self._is_price_placeholder_token(pv):
                    # 归还到电价行（纠正错位，不是用第三行补第二行）
                    price[ci] = u
        out[-1] = price
        return out

    def _scrub_price_echo_from_upper_rows(
        self,
        matrix: List[List[str]],
        sticky_map: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        经典三行（电力/计划/电价）：上方行若与末行电价高度同列重合 → 粘连渗漏，整行清空。

        留给后续 `_recover_leading_zero_data_rows` 按几何带补 0。
        多小数独立指标（≥2）不清洗，避免误删真小数行。
        另：逐格清掉整型行里的电价样小数（单列串行）。
        """
        sticky_out = self._fit_sticky_map(sticky_map, matrix)
        if not matrix or len(matrix) < 2:
            return matrix, sticky_out
        out = self._scrub_price_tokens_from_upper_cells([list(r) for r in matrix])
        if self._matrix_is_multi_decimal_indicators(out):
            return out, sticky_out
        price = out[-1]
        if not self._row_looks_like_decimal_indicator(price):
            return out, sticky_out
        for ri in range(len(out) - 1):
            row = out[ri]
            if not self._decimal_row_echoes_price(row, price):
                continue
            ncols = len(row)
            out[ri] = [""] * ncols
            if ri < len(sticky_out):
                sticky_out[ri] = [""] * max(ncols, len(sticky_out[ri]))
                sticky_out[ri] = sticky_out[ri][:ncols]
        return out, sticky_out

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
        rows = [list(r) for r in matrix if r is not None]
        if not rows:
            return 0
        # 最像电价的一行作锚；与其同列相呼应的渗漏行不计独立指标
        ratios = [self._row_decimal_fill_ratio(r) for r in rows]
        price_i = max(range(len(rows)), key=lambda i: (ratios[i], i))
        price = rows[price_i]
        n = 0
        for i, r in enumerate(rows):
            if not self._row_looks_like_decimal_indicator(r):
                continue
            if i != price_i and self._decimal_row_echoes_price(r, price):
                continue
            n += 1
        return n

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
        # 主行无电价特征（无小数且非可补点长整数）→ 禁止盖末行（防功率/0 串行）
        main_price_like = any(
            ("." in str(v or ""))
            or bool(re.fullmatch(r"-?\d{4,}", str(v or "").strip()))
            for v in aligned
            if v and str(v).strip() and not self._is_price_placeholder_token(v)
        )
        if not main_price_like:
            return False
        # 主行更像上方某一已填指标行 → 绝不能盖/写入末行（即使末行空）
        for upper in matrix[:-1]:
            upper_n = sum(1 for v in upper if v and str(v).strip())
            if upper_n >= 3 and self._aligned_rows_same_series(upper, aligned):
                if last_n < 2 or not self._aligned_rows_same_series(last, aligned):
                    return False
        # 末行过稀且主行不像上方已有指标：允许按主数据带写入（主带由 Y 选出）
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
        # 不按「小数=电价」沉底；保持 Y 分带已定行序，只补空槽到 3 行
        empty_slot = [""] * ncols
        while len(out) < expected:
            if len(out) == 2 and self._pad_two_rows_same_band(out[0], out[1]):
                # 两行同系列（常见电力≈计划）→ 缺的是底行，后补空
                out.append(list(empty_slot))
            elif len(out) == 2:
                # 两行不同系列 → 更像漏了中间行，插在中间
                out.insert(1, list(empty_slot))
            else:
                out.append(list(empty_slot))
        return out[:expected]

    def _pad_two_rows_same_band(
        self, a: Sequence[str], b: Sequence[str]
    ) -> bool:
        """补行用：两行是否同一指标带（不要求 ≥3 列，允许 420.4 等小数）。"""
        if self._aligned_rows_same_series(a, b):
            return True
        n = min(len(a), len(b))
        match = 0
        compared = 0
        for i in range(n):
            sa = str(a[i]).strip() if i < len(a) and a[i] else ""
            sb = str(b[i]).strip() if i < len(b) and b[i] else ""
            if not sa or not sb:
                continue
            compared += 1
            if sa == sb:
                match += 1
                continue
            try:
                fa, fb = float(sa), float(sb)
            except ValueError:
                continue
            if abs(fa - fb) <= max(0.05, abs(fa) * 0.02):
                match += 1
        if compared >= 2 and match / compared >= 0.5:
            return True
        ma = abs(self._row_filled_median(list(a)))
        mb = abs(self._row_filled_median(list(b)))
        if ma >= 5 and mb >= 5 and abs(ma - mb) / max(ma, mb, 1.0) <= 0.18:
            return True
        return False

    def _untangle_integer_decimal_rows(
        self, matrix: List[List[str]]
    ) -> List[List[str]]:
        """
        历史：曾按「带小数点 → 电价」把格在行间重分，纠正斜拍渗漏。

        已废止。电力/计划/电价都可能出现 420.4 这类小数；按小数串行会把
        前两行合法值抽到第 3 行。行归属只信任 Y 分带 + 列网格对齐；
        斜拍靠 deskew / 行 X 偏移，不在矩阵上搬数。
        """
        return matrix

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

    def _rows_have_sticky_price_evidence(
        self,
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_cells: Sequence[OcrCell],
    ) -> bool:
        """
        OCR 行里是否仍有「多格粘连电价」原文（尚未落入矩阵）。

        用于挡住：主行误选成全 0、粘连电价未拆 → 被当成未出清整表清空。
        """
        start = header_idx + 1 if header_idx >= 0 else 0
        pool: List[OcrCell] = list(data_cells or [])
        for idx in range(start, len(rows or [])):
            pool.extend(rows[idx])
        for cell in pool:
            text = str(getattr(cell, "text", "") or "")
            if not text or self._normalize_time(text):
                continue
            if self._looks_like_multicolumn_glue_text(text):
                return True
            parts = self._split_cell_values(text)
            if len(parts) >= 2 and self._sticky_pair_plausible(parts):
                # 拆出多段小数电价，说明屏上有真实电价行
                dec_n = sum(1 for p in parts if "." in str(p))
                if dec_n >= 2:
                    return True
        return False

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

    def _row_has_nonzero_power_values(self, row: Sequence[str]) -> bool:
        """行内是否已有非零功率/计划读数（禁止再对空格臆造 0）。"""
        for v in row:
            s = str(v or "").strip()
            if not s or s == self.NODATA_MARK or self._is_lone_dash(s):
                continue
            if s in ("0", "0.0", "0.00", "0.000", "-0", "-0.0", "O", "o", "D", "U"):
                continue
            try:
                if abs(float(s)) > 1e-9:
                    return True
            except ValueError:
                continue
        return False

    def _looks_like_price_decimal_token(self, raw: object) -> bool:
        """
        两位及以上小数 → 电价样，禁止写入电力/计划行（防串行）。

        电价常见 xx.xx / xxx.xx / xxx.xxx；仅拦三位会漏掉 273.25 这类
        从第三行扫进第二行的值。一位小数（如 420.4）仍允许留在上两行。
        """
        s = str(raw or "").strip()
        if not s or "." not in s:
            return False
        body = s[1:] if s.startswith("-") else s
        if not re.fullmatch(r"\d+\.\d+", body):
            return False
        frac = body.split(".", 1)[1]
        return len(frac) >= 2

    def _accept_upper_ocr_token(
        self,
        got: str,
        *,
        price_row: Optional[Sequence[str]],
        col: int,
        row: Optional[Sequence[str]] = None,
    ) -> bool:
        """上两行新 OCR 值是否可采纳（拒电价样 / 与末行串值 / 整型行混入小数）。"""
        if not got or self._looks_like_price_decimal_token(got):
            return False
        # 同行已是整型功率/计划时，任何带点小数都视为串行（认不到留空）
        if row is not None and "." in got and self._row_is_integerish_or_empty(row):
            return False
        if price_row is None or col >= len(price_row):
            return True
        pv = str(price_row[col] or "").strip()
        if not pv:
            return True
        if got == pv:
            return False
        try:
            if abs(float(got) - float(pv)) <= 1e-6:
                return False
        except ValueError:
            pass
        return True

    def _fill_upper_row_empties_at_band(
        self,
        img: np.ndarray,
        row: Sequence[str],
        col_centers: Sequence[float],
        y0: float,
        y1: float,
        yc: float,
        nodata_cols: Optional[set] = None,
        *,
        price_row: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        上两行空格：有非零证据 → 多档 Y 按列 OCR；认不到就留空。
        禁止抄另一行（电力≠可编造的计划）。真·全 0/空行才允许先验补 0。
        已有非空格绝不因 nodata skip 清掉（未出清最终由 force_uncleared 统一处理）。
        Y 只在本带中心与略偏上试，禁止下扫进电价行（宁空勿串）。
        """
        skip = nodata_cols or set()
        ncols = len(col_centers)
        out = list(row) + [""] * max(0, ncols - len(row))
        out = out[:ncols]
        if self._row_has_nonzero_power_values(out):
            empty_before = [
                i
                for i, v in enumerate(out)
                if not str(v or "").strip() and i not in skip
            ]
            if not empty_before:
                return out
            h_med = max(12.0, abs(y1 - y0) * 0.9, 16.0)
            # 不下扫：yc+ 会读到第三行电价（如 273.25 补进第二行）
            y_tries = [yc, yc - h_med * 0.55, yc - h_med * 1.1]
            result = list(out)
            still_empty = set(empty_before)
            for try_yc in y_tries:
                if not still_empty:
                    break
                try_yc = min(
                    max(float(try_yc), h_med), float(img.shape[0]) - h_med * 0.6
                )
                band = self._fake_band_at_y(col_centers, try_yc, h_med)
                trial = list(result)
                for i in still_empty:
                    trial[i] = ""
                filled = self._ocr_fill_empty_cells(
                    img,
                    trial,
                    list(col_centers),
                    band,
                    img.shape[1],
                    prefer_sharp=False,
                    trust_cell_ocr=True,
                    skip_cols=skip | (set(range(ncols)) - still_empty),
                )
                for i in list(still_empty):
                    raw = filled[i] if i < len(filled) else ""
                    got = str(raw or "").strip()
                    if not self._accept_upper_ocr_token(
                        got, price_row=price_row, col=i, row=result
                    ):
                        continue
                    result[i] = got
                    still_empty.discard(i)
            return result[:ncols]
        return self._fill_row_by_column_zero_ocr(
            img, out, col_centers, y0, y1, yc, nodata_cols=skip
        )

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
        补全电力/计划行空格为 0（仅真·零行；禁止用于已有非零读数的行 / 电价末行）。

        整图检测常漏单字 0；单列 OCR 又易把竖线读成 1/16。
        未出清列（nodata_cols，屏上 -）保持空，禁止补 0。
        """
        del img, y0, y1, yc  # 签名保留供几何带调用；先验填 0 无需再 OCR
        # 行内已有非零 → 调用方应走 _fill_upper_row_empties_at_band；此处拒补
        if self._row_has_nonzero_power_values(row):
            ncols = len(col_centers)
            out = list(row) + [""] * max(0, ncols - len(row))
            return out[:ncols]
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

        # 行序保持 Y 分带结果，不再按小数占比把某行沉成「电价行」
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
            # 有非零 → 几何重识空格（认不到留空）；真零行才补 0；不抄另一行
            filled_row = self._fill_upper_row_empties_at_band(
                img,
                row,
                col_centers,
                y0,
                y1,
                yc,
                nodata_cols=nodata_cols,
                price_row=out[-1],
            )
            if self._row_has_nonzero_power_values(filled_row):
                out[ri] = filled_row
            else:
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

    def _row_has_actual_zero_token(self, row: Sequence[str]) -> bool:
        """行内是否出现过字面「0」（非空串占位）。"""
        return any(str(v).strip() == "0" for v in row if v is not None)

    def _price_row_looks_like_fake_zeros(
        self, matrix: Sequence[Sequence[str]]
    ) -> bool:
        """
        末行像「电价漏检后的假全 0」：无小数、几乎全 0/空。

        典型：上方有非零功率；或整表仅 0/空（功率真为 0 时电价仍应有小数，
        若末行也全 0 则多为漏检，须强制电价带重识，否则会被当成未出清整表置空）。
        """
        if not matrix or len(matrix) < 2:
            return False
        last = list(matrix[-1])
        if any("." in str(v or "") for v in last):
            return False
        if not self._row_is_mostly_zeros(last):
            return False
        ncols = max(len(last), max((len(r) for r in matrix), default=0))
        # 功率/计划真为全 0 + 电价漏检成全 0：与未出清外观相同，须先尝试救电价
        if ncols > 0 and self._matrix_looks_all_uncleared(matrix, ncols):
            return True
        # 上方至少一行像真实功率/计划（非全空/全 0）
        for upper in matrix[:-1]:
            if self._row_looks_like_decimal_indicator(upper):
                return True
            filled = [str(v).strip() for v in upper if v and str(v).strip()]
            nonzero = [
                v
                for v in filled
                if v not in ("0", "0.0", "-0", self.NODATA_MARK, "-", "—", "–")
            ]
            if len(nonzero) >= max(2, int(len(upper) * 0.35)):
                return True
        return False

    def _price_row_looks_like_short_int_frags(self, row: Sequence[str]) -> bool:
        """
        电价行像粘连/裁缘残片：有非零填格，但全是无点 1～3 位短整数（如 1/24/79），
        且无一格带小数。此时同行无可靠中位，须整行强制按列重识。
        """
        shorts = 0
        for v in row:
            s = str(v or "").strip().replace("−", "-").replace("–", "-")
            if not s or s in ("-", "—", "–", self.NODATA_MARK):
                continue
            if "." in s:
                return False
            body = s[1:] if s.startswith("-") and len(s) > 1 else s
            if not re.fullmatch(r"\d{1,3}", body):
                return False
            if body not in ("0", "00", "000"):
                shorts += 1
        return shorts >= 2

    def _blank_fake_zero_price_row(
        self, matrix: List[List[str]]
    ) -> List[List[str]]:
        """假全 0 电价行置空，便于后续列补识。"""
        if not self._price_row_looks_like_fake_zeros(matrix):
            return matrix
        out = [list(r) for r in matrix]
        out[-1] = [
            self.NODATA_MARK if v == self.NODATA_MARK else ""
            for v in out[-1]
        ]
        return out

    def _price_row_echoes_upper(
        self, matrix: Sequence[Sequence[str]]
    ) -> bool:
        """末行是否只是上方功率/计划行的整型副本（补识 Y 偏高时常见）。"""
        if not matrix or len(matrix) < 2:
            return False
        last = list(matrix[-1])
        if any("." in str(v or "") for v in last):
            return False
        last_n = sum(1 for v in last if str(v or "").strip())
        if last_n < max(3, int(len(last) * 0.4)):
            return False
        for upper in matrix[:-1]:
            if self._aligned_rows_same_series(upper, last):
                return True
        return False

    def _infer_price_row_y_band(
        self,
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_cells: Sequence[OcrCell],
        img_h: int,
        matrix: Sequence[Sequence[str]],
        col_centers: Sequence[float],
    ) -> Tuple[float, float, float]:
        """
        推断电价行 (y0, y1, yc)。

        优先用已检出的数据行 Y 外推（白底电价常被整图漏检，几何表头行高易偏小，
        会落在第 2 行功率上读成 460）。
        """
        header_y = -1.0
        if 0 <= header_idx < len(rows) and rows[header_idx]:
            header_y = self._row_y_center(rows[header_idx])

        data_ys: List[float] = []
        hs: List[float] = []
        for i, row in enumerate(rows):
            if i == header_idx or not row:
                continue
            # 跳过页脚/非数字带
            texts = [str(c.text or "").strip() for c in row]
            if any(self._is_footer_text(t) for t in texts if t):
                continue
            nums = [
                t
                for t in texts
                if t and re.fullmatch(r"-?\d+(\.\d+)?", t)
            ]
            if len(nums) < 2:
                continue
            yc = self._row_y_center(row)
            if header_y >= 0 and yc < header_y + 20:
                continue
            data_ys.append(yc)
            hs.extend(float(c.height) for c in row if c.height > 0)

        data_ys = sorted(set(round(y, 1) for y in data_ys))
        # 合并过近的 Y（同带双检）
        merged: List[float] = []
        for y in data_ys:
            if not merged or abs(y - merged[-1]) > 40:
                merged.append(float(y))
            else:
                merged[-1] = (merged[-1] + float(y)) / 2.0
        data_ys = merged

        h_med = float(np.median(hs)) if hs else 28.0
        h_med = max(h_med, 16.0)

        if len(data_ys) >= 2:
            gaps = [data_ys[i] - data_ys[i - 1] for i in range(1, len(data_ys))]
            gap = float(np.median(gaps))
            gap = max(gap, h_med * 1.1)
            yc = data_ys[-1] + gap
        elif len(data_ys) == 1:
            if header_y >= 0 and data_ys[0] > header_y + 30:
                gap = max(data_ys[0] - header_y, h_med * 1.2)
            else:
                gap = h_med * 1.35
            yc = data_ys[0] + gap
        else:
            hy, row_h = self._resolve_header_anchor_y(
                rows, header_idx, data_cells, img_h, matrix, col_centers
            )
            _y0, _y1, yc = self._geometric_data_band_centers(hy, row_h, img_h)[-1]
            h_med = max(h_med, row_h * 0.55)

        # 不超过图高；略留底边距（滚动条/页脚）
        yc = min(max(yc, h_med), float(img_h) - h_med * 0.8)
        half = max(h_med * 0.55, 14.0)
        y0 = max(0.0, yc - half)
        y1 = min(float(img_h), yc + half)
        return y0, y1, yc

    def _fake_band_at_y(
        self, col_centers: Sequence[float], yc: float, h_med: float
    ) -> List[OcrCell]:
        """供列 OCR 使用的单格锚点（只提供行 Y）。"""
        return [
            OcrCell(
                text="",
                confidence=0.5,
                x=float(col_centers[0]) if col_centers else 0.0,
                y=float(yc - h_med / 2),
                width=20.0,
                height=float(max(h_med, 12.0)),
            )
        ]

    def _ocr_price_row_at_band(
        self,
        img: np.ndarray,
        row: Sequence[str],
        col_centers: Sequence[float],
        band: Sequence[OcrCell],
        nodata_cols: set,
    ) -> List[str]:
        """在指定 Y 带对电价行空格做锐化列 OCR；裸 0/无点整数视为未识别→空。"""
        cleared = [
            self.NODATA_MARK if (i in nodata_cols or v == self.NODATA_MARK) else ""
            for i, v in enumerate(row)
        ]
        filled = self._ocr_fill_empty_cells(
            img,
            cleared,
            list(col_centers),
            list(band),
            img.shape[1],
            prefer_sharp=True,
            trust_cell_ocr=True,
            skip_cols=nodata_cols,
        )
        return [
            ""
            if (i in nodata_cols or self._is_price_placeholder_token(v))
            else str(v or "").strip()
            for i, v in enumerate(filled)
        ]

    def _price_fill_looks_valid(
        self, filled: Sequence[str], matrix: Sequence[Sequence[str]]
    ) -> bool:
        """重识结果是否像真实电价（须有带点小数或可补点长整数；非全 0、非上行副本）。"""
        vals = [
            str(v or "").strip()
            for v in filled
            if str(v or "").strip() and not self._is_price_placeholder_token(v)
        ]
        if not vals:
            return False
        # 带点小数，或 ≥4 位无点（后续补点）——禁止裸 0/短整型冒充
        has_price_like = any(
            ("." in v)
            or bool(re.fullmatch(r"-?\d{4,}", v))
            for v in vals
        )
        if not has_price_like:
            return False
        trial = [list(r) for r in matrix]
        if trial:
            trial[-1] = list(filled)
        if self._price_row_echoes_upper(trial):
            return False
        return True

    def _recover_missing_price_row(
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
        末行仍全空/假全 0/抄成上方功率时，按已检出数据行 Y 外推电价带强制列 OCR。

        整图检测常漏浅蓝下的白底电价行；表头几何行高偏小时会误扫第 2 行。
        """
        if not matrix or not col_centers or img is None or img.size == 0:
            return matrix
        out = [list(r) for r in matrix]
        out = self._enforce_expected_data_rows(out, expected=self.EXPECTED_DATA_ROWS)
        ncols = len(col_centers)
        out = [list(r) + [""] * max(0, ncols - len(r)) for r in out]
        out = [r[:ncols] for r in out]
        if headers:
            out = self._scrub_header_time_leaks(headers, out)

        last = out[-1]
        empty_n = sum(1 for v in last if not str(v or "").strip())
        need = (
            self._price_row_looks_like_fake_zeros(out)
            or self._price_row_echoes_upper(out)
        )
        if not need:
            upper_filled = any(
                sum(1 for v in r if str(v or "").strip()) >= 2 for r in out[:-1]
            )
            # 末行大半空：即使上方也空（整图漏检），仍尝试几何带电价 OCR
            if not (
                empty_n >= max(2, int(ncols * 0.5))
                and (upper_filled or self._matrix_looks_all_uncleared(out, ncols))
            ):
                return out

        nodata_cols = self._trailing_nodata_columns(out, ncols)
        # 电价行本身空 / 假全 0 导致「全列 nodata」时，仍用空 skip 做恢复；
        # 否则「功率真 0 + 电价漏检成 0」会因 empty_n 低直接退出，随后整表置空。
        if len(nodata_cols) >= ncols:
            fake_or_echo = (
                self._price_row_looks_like_fake_zeros(out)
                or self._price_row_echoes_upper(out)
            )
            if empty_n < max(2, int(ncols * 0.5)) and not fake_or_echo:
                return out
            nodata_cols = set()

        y0, y1, yc = self._infer_price_row_y_band(
            rows, header_idx, data_cells, img.shape[0], out, col_centers
        )
        h_med = max(12.0, (y1 - y0) * 0.9)
        candidates = [yc]
        # 外推后再下探一档，防止行距估短仍压在功率行上
        candidates.append(yc + h_med * 1.15)
        candidates.append(yc + h_med * 2.1)

        best: Optional[List[str]] = None
        best_score = -1
        for try_yc in candidates:
            try_yc = min(max(try_yc, h_med), float(img.shape[0]) - h_med * 0.6)
            band = self._fake_band_at_y(col_centers, try_yc, h_med)
            filled = self._ocr_price_row_at_band(
                img, last, col_centers, band, nodata_cols
            )
            if not self._price_fill_looks_valid(filled, out):
                continue
            dec_n = sum(1 for v in filled if "." in str(v or ""))
            nz = sum(
                1
                for v in filled
                if str(v or "").strip()
                not in ("", "0", "0.0", "-0", self.NODATA_MARK)
            )
            score = dec_n * 10 + nz
            if score > best_score:
                best_score = score
                best = filled
            if dec_n >= max(2, int(ncols * 0.35)):
                break
        if best is not None:
            out[-1] = best
        return out

    def _price_row_has_decimal_values(self, row: Sequence[str]) -> bool:
        """末行是否已有带小数的真实电价值（非占位 0）。"""
        for v in row:
            s = str(v or "").strip()
            if not s or s == self.NODATA_MARK or self._is_lone_dash(s):
                continue
            if "." in s and s not in ("0.0", "0.00", "0.000", "-0.0"):
                return True
        return False

    def _is_price_placeholder_token(self, raw: object) -> bool:
        """
        电价格占位/假值：空、横杠、裸 0、无点短整数（≤3 位）。

        原则：末行未读到带点小数前禁止用 0/上行值填充；认不到就留空（不写 PI）。
        4 位及以上无点串留给 _fix_decimal_ocr 补点，不在此当作占位丢掉。
        """
        if raw is None:
            return True
        s = str(raw).strip()
        if not s or s == self.NODATA_MARK or self._is_lone_dash(s):
            return True
        if s in ("0", "0.0", "0.00", "0.000", "-0", "-0.0", "O", "o", "D", "U"):
            return True
        if re.fullmatch(r"0+", s):
            return True
        # ≤3 位无点：抄上行功率/竖纹假读，不是漏点电价
        if re.fullmatch(r"-?\d{1,3}", s):
            return True
        return False

    def _scrub_price_row_no_crossfill(
        self, matrix: List[List[str]]
    ) -> List[List[str]]:
        """
        硬约束：电价末行禁止占位 0；与上方同列串值时按形态决定清哪边。

        - 裸 0 / 无点整数 → 末行置空（不入库）
        - 同列同值且像电价小数 → 清上方、保留末行（禁止第三行补第二行）
        - 同列同值且像整型功率 → 清末行、保留上方
        """
        if not matrix or len(matrix) < 1:
            return matrix
        out = self._scrub_price_tokens_from_upper_cells([list(r) for r in matrix])
        last = out[-1]
        uppers = out[:-1]
        for ci in range(len(last)):
            cur = last[ci]
            s = str(cur or "").strip()
            if not s:
                continue
            if self._is_price_placeholder_token(s):
                last[ci] = ""
                continue
            for ui, upper in enumerate(uppers):
                if ci >= len(upper):
                    continue
                u = str(upper[ci] or "").strip()
                if not u:
                    continue
                same = s == u
                if not same:
                    try:
                        same = abs(float(s) - float(u)) <= 1e-9
                    except ValueError:
                        same = False
                if not same:
                    continue
                # 电价样：清上方；整型样：清末行。宁空勿串。
                if self._looks_like_price_decimal_token(s) or self._looks_like_price_decimal_token(
                    u
                ):
                    out[ui][ci] = ""
                else:
                    last[ci] = ""
                break
        out[-1] = last
        return out

    def _ensure_geometric_three_data_rows(
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
        表头锚点下固定 3 条数据带按列 OCR，保证「表头 + 三行」结构可扫满。

        主路径不再依赖整图是否碰巧框到白底电价行：
        1) 末行无小数 → 几何电价带强制锐化列 OCR（多档 Y）
        2) 电价落地后 → 上两行空位先验补 0（真零功率合法）
        禁止在电价未落地前用 0 冒充第三行。
        """
        if not col_centers or img is None or img.size == 0:
            return matrix
        out = [list(r) for r in (matrix or [])]
        out = self._enforce_expected_data_rows(
            out, expected=self.EXPECTED_DATA_ROWS
        )
        ncols = len(col_centers)
        out = [list(r) + [""] * max(0, ncols - len(r)) for r in out]
        out = [r[:ncols] for r in out]
        while len(out) < self.EXPECTED_DATA_ROWS:
            out.append([""] * ncols)
        if headers:
            out = self._scrub_header_time_leaks(headers, out)

        # 末行假全 0 / 空：先清空再几何重识，避免 0 占位挡住列 OCR
        if not self._price_row_has_decimal_values(out[-1]):
            if self._price_row_looks_like_fake_zeros(out) or self._row_is_mostly_zeros(
                out[-1]
            ):
                out[-1] = [
                    self.NODATA_MARK if v == self.NODATA_MARK else ""
                    for v in out[-1]
                ]

        header_y, row_h = self._resolve_header_anchor_y(
            rows, header_idx, data_cells, img.shape[0], out, col_centers
        )
        geo_bands = self._geometric_data_band_centers(
            header_y, row_h, img.shape[0]
        )
        h_med = max(12.0, row_h * 0.55)

        # —— 1) 电价带强制 OCR ——
        price_yc: Optional[float] = None
        if not self._price_row_has_decimal_values(out[-1]):
            nodata_cols = set()
            y0, y1, yc = geo_bands[-1]
            iy0, iy1, iyc = self._infer_price_row_y_band(
                rows, header_idx, data_cells, img.shape[0], out, col_centers
            )
            candidates = [yc, iyc, yc + h_med * 1.15, iyc + h_med * 1.15, yc + h_med * 2.1]
            best: Optional[List[str]] = None
            best_score = -1
            best_yc: Optional[float] = None
            seen_yc: set = set()
            for try_yc in candidates:
                try_yc = float(try_yc)
                key = round(try_yc, 1)
                if key in seen_yc:
                    continue
                seen_yc.add(key)
                try_yc = min(
                    max(try_yc, h_med), float(img.shape[0]) - h_med * 0.6
                )
                band = self._fake_band_at_y(col_centers, try_yc, h_med)
                filled = self._ocr_price_row_at_band(
                    img, out[-1], col_centers, band, nodata_cols
                )
                if not self._price_fill_looks_valid(filled, out):
                    continue
                dec_n = sum(1 for v in filled if "." in str(v or ""))
                nz = sum(
                    1
                    for v in filled
                    if str(v or "").strip()
                    not in ("", "0", "0.0", "-0", self.NODATA_MARK)
                )
                score = dec_n * 10 + nz
                if score > best_score:
                    best_score = score
                    best = filled
                    best_yc = try_yc
                if dec_n >= max(2, int(ncols * 0.35)):
                    break
            if best is not None:
                out[-1] = best
                price_yc = best_yc
            out = self._scrub_price_row_no_crossfill(out)

        # —— 2) 上两行：用电价 Y 反推行带，避免扫进电价；认不到留空 ——
        price_ok = self._price_row_has_decimal_values(out[-1])
        if price_ok:
            nodata_cols = self._trailing_nodata_columns(out, ncols)
            if len(nodata_cols) >= ncols:
                nodata_cols = set()
            if price_yc is None:
                _a, _b, price_yc = self._infer_price_row_y_band(
                    rows,
                    header_idx,
                    data_cells,
                    img.shape[0],
                    out,
                    col_centers,
                )
            gap = max(h_med * 1.15, row_h * 0.95)
            upper_n = max(0, len(out) - 1)
            for ri in range(upper_n):
                if self._row_looks_like_decimal_indicator(out[ri]):
                    continue
                steps = upper_n - ri
                yc = float(price_yc) - steps * gap
                yc = min(max(yc, h_med), float(img.shape[0]) - h_med * 0.6)
                half = max(h_med * 0.55, 14.0)
                y0, y1 = max(0.0, yc - half), min(float(img.shape[0]), yc + half)
                filled_row = self._fill_upper_row_empties_at_band(
                    img,
                    out[ri],
                    col_centers,
                    y0,
                    y1,
                    yc,
                    nodata_cols=nodata_cols,
                    price_row=out[-1],
                )
                if self._row_has_nonzero_power_values(filled_row):
                    out[ri] = filled_row
                else:
                    out[ri] = self._coerce_zero_row_ocr_noise(
                        filled_row, nodata_cols=nodata_cols
                    )
            out, _ = self._scrub_price_echo_from_upper_rows(out, None)
        out = self._scrub_price_row_no_crossfill(out)
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
        # 数字位全是 1（含 1.11 / 1111）：竖纹假读，零行上压回 0
        if self._is_moire_all_ones_token(t):
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

    def _estimate_row_x_offset(
        self,
        row: Sequence[OcrCell],
        col_centers: Sequence[float],
        *,
        max_dist: float,
    ) -> float:
        """
        估计整行相对列中心的平移（斜拍/底行透视常见）。

        用各格中心到最近列中心的中位数偏差；样本不足或偏差过小则 0。
        """
        if not row or not col_centers:
            return 0.0
        deltas: List[float] = []
        for cell in row:
            if self._is_footer_text(cell.text) or self._normalize_time(cell.text):
                continue
            if not self._count_numeric_tokens(cell.text):
                continue
            cx = float(cell.x + max(cell.width, 1) / 2.0)
            nearest = min(
                (float(c) for c in col_centers),
                key=lambda c: abs(cx - c),
            )
            d = cx - nearest
            if abs(d) <= max_dist * 2.2:
                deltas.append(d)
        if len(deltas) < 2:
            return 0.0
        off = float(np.median(deltas))
        # 过小噪声忽略；过大更像错行/乱框
        if abs(off) < max(6.0, max_dist * 0.12):
            return 0.0
        if abs(off) > max_dist * 1.6:
            return 0.0
        return off

    def _align_row_to_columns(self, row: List[OcrCell],
                              col_centers: List[float],
                              col_count: int) -> Tuple[List[str], List[str]]:
        """
        将 OCR 行按 X 坐标对齐到列网格。

        绝不「从左到右顺序填列」：中间漏识必须留空，不能把右侧数值前移。
        粘连多值块优先按「块覆盖的列中心」落列，不再只按宽度均分。
        检测框盖住 ≥2 列：先试可信点对链 / N 列切；切不开才标待重识。
        底行/斜拍若整行相对表头平移，先估计 X 偏移再落格（减轻第三行前段空）。

        @return (values, sticky_marks) 同形；sticky 非空表示业务层粘连处理
        """
        empty: Tuple[List[str], List[str]] = ([], [])
        if not row or col_count <= 0 or not col_centers:
            return empty

        values = [""] * col_count
        sticky_marks = [""] * col_count
        qualities = [(-1, 0, 0.0)] * col_count
        step = self._median_col_step(col_centers)
        max_dist = max(step * 0.60, 28.0)
        x_off = self._estimate_row_x_offset(row, col_centers, max_dist=max_dist)
        # 对齐时把格子平移回列网格；覆盖列判定同步使用偏移后的框
        align_centers = [float(c) + x_off for c in col_centers] if x_off else list(
            map(float, col_centers)
        )

        for cell in self._filter_overlap_noise_cells(row):
            if self._is_footer_text(cell.text):
                continue
            # 时刻表头碎片禁止落入数据格（否则 23:00→23.00）
            if self._normalize_time(cell.text):
                continue
            covered = self._columns_covered_by_cell(cell, align_centers, step)
            # 单列整格只有「-」：记为无数据，禁止后续单列重识
            if self._is_lone_dash(cell.text):
                target = covered[0] if len(covered) == 1 else None
                if target is None and cell.width <= step * 1.35:
                    cx = float(cell.x + max(cell.width, 1) / 2.0)
                    col = min(
                        range(col_count),
                        key=lambda j: abs(cx - float(align_centers[j])),
                    )
                    if abs(cx - float(align_centers[col])) <= max_dist:
                        target = col
                if (
                    target is not None
                    and not values[target]
                ):
                    values[target] = self.NODATA_MARK
                    qualities[target] = (-1, 0, 0.0)
                continue
            # 宽框多列：可信拆分优先落库；切不开才待几何重识
            if len(covered) >= 2 and self._cell_text_looks_numeric_value(cell.text):
                reliable = self._try_wide_sticky_reliable_parts(cell, covered)
                if reliable:
                    self._assign_parts_into_columns(
                        cell,
                        reliable,
                        align_centers,
                        col_count,
                        values,
                        qualities,
                        step=step,
                        max_dist=max_dist,
                        force_cols=covered,
                        sticky_marks=sticky_marks,
                        sticky_label="粘连拆分",
                    )
                    continue
                for j in covered:
                    if 0 <= j < col_count and not sticky_marks[j]:
                        sticky_marks[j] = "粘连待重识"
                continue
            parts = self._split_cell_values(cell.text)
            if not parts:
                continue
            label = str(getattr(cell, "sticky", "") or "") or (
                "粘连拆分" if len(parts) >= 2 else ""
            )
            self._assign_parts_into_columns(
                cell,
                parts,
                align_centers,
                col_count,
                values,
                qualities,
                step=step,
                max_dist=max_dist,
                sticky_marks=sticky_marks,
                sticky_label=label,
            )

        if not any(v.strip() for v in values):
            # 宽框粘连整行待重识：保留空值壳 + sticky 标记，供几何按列 OCR
            if any(str(m).strip() for m in sticky_marks):
                return values, sticky_marks
            return empty
        return values, sticky_marks

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
        sticky_marks: Optional[List[str]] = None,
        sticky_label: str = "",
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
        label = str(sticky_label or "").strip()

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
                if sticky_marks is not None and 0 <= col < len(sticky_marks):
                    sticky_marks[col] = label if label else sticky_marks[col]

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
                gap = len(covered) - n
                # 缺 1 格：右对齐（左缘裁断/漏识更常见），避免比例映射跳空中列
                if gap == 1:
                    for i, val in enumerate(parts):
                        _put(covered[i + 1], val)
                    return
                # 缺口更大：按比例映射到 covered
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

    @staticmethod
    def _edge_reocr_skip_cols(
        headers_before: Sequence[str], headers_after: Sequence[str]
    ) -> set:
        """数据裁断丢列后，新首/末列禁止单列补洞（裁缘残墨易污读）。"""
        skip = set()
        if not headers_before or not headers_after:
            return skip
        if headers_after[0] != headers_before[0]:
            skip.add(0)
        if headers_after[-1] != headers_before[-1]:
            skip.add(len(headers_after) - 1)
        return skip

    def _refill_matrix_empties(
        self,
        img: np.ndarray,
        matrix: List[List[str]],
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_idx: int,
        data_cells: List[OcrCell],
        col_centers: List[float],
        skip_cols: Optional[set] = None,
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
        nodata_cols = self._trailing_nodata_columns(out, ncols) | set(skip_cols or ())
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
            is_price_row = ri == len(out) - 1
            band = bands[ri] if ri < len(bands) else (
                data_cells if is_price_row else []
            )
            # 电价行空得多 / 无独立指示带：按已检出数据行 Y 外推，避免扫到功率行
            if is_price_row and (
                not band or empty_n >= max(2, int(len(row) * 0.5))
            ):
                _y0, _y1, yc = self._infer_price_row_y_band(
                    rows,
                    header_idx,
                    data_cells or [],
                    img.shape[0],
                    out,
                    col_centers,
                )
                h_med = max(12.0, (_y1 - _y0) * 0.9)
                band = self._fake_band_at_y(col_centers, yc, h_med)
            elif not band:
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
            filled = self._ocr_fill_empty_cells(
                img,
                row,
                col_centers,
                band,
                img_w,
                prefer_sharp=is_price_row,
                trust_cell_ocr=is_price_row and empty_n >= max(2, int(len(row) * 0.5)),
                skip_cols=nodata_cols,
            )
            # 上两行：空格补识若扫到电价样小数 → 拒收留空（宁空勿串行）
            if not is_price_row and self._row_is_integerish_or_empty(row):
                price_row = out[-1] if out else None
                for ci, (before, after) in enumerate(zip(row, filled)):
                    if str(before or "").strip() or not str(after or "").strip():
                        continue
                    if not self._accept_upper_ocr_token(
                        str(after).strip(),
                        price_row=price_row,
                        col=ci,
                        row=row,
                    ):
                        filled[ci] = ""
            out[ri] = filled
            # 电价补识若抄成上方功率行，置空留给后续外推重试
            if is_price_row and self._price_row_echoes_upper(out):
                out[ri] = [
                    self.NODATA_MARK if v == self.NODATA_MARK else ""
                    for v in out[ri]
                ]
        return out

    def _bind_bands_to_matrix_indices(
        self,
        bands: Sequence[Sequence[OcrCell]],
        matrix: Sequence[Sequence[str]],
    ) -> Dict[int, List[OcrCell]]:
        """
        OCR 指示带 → 矩阵行下标。

        带数 == 行数：按下标一一对应。
        带数更少（常见：双行全 0 被漏/并后只剩 2 带，pad 出中间空行）：
        按 Y 排序后「末带→末行、其余从上往下」，跳过未绑定的空槽，
        避免 bands[1]=电价 写进 matrix[1]=计划空行。
        """
        n = len(matrix)
        if not bands or n <= 0:
            return {}
        live = [list(b) for b in bands if b]
        if not live:
            return {}
        if len(live) == n:
            return {i: live[i] for i in range(n)}
        ordered = sorted(
            range(len(live)),
            key=lambda i: self._row_y_center(live[i]),
        )
        bound: Dict[int, List[OcrCell]] = {}
        # 最底带 → 末行（电价）
        bound[n - 1] = list(live[ordered[-1]])
        mi = 0
        for bi in ordered[:-1]:
            while mi < n - 1 and mi in bound:
                mi += 1
            if mi >= n - 1:
                break
            bound[mi] = list(live[bi])
            mi += 1
        return bound

    def _reocr_geometry_sticky_cells(
        self,
        img: np.ndarray,
        matrix: List[List[str]],
        rows: Sequence[List[OcrCell]],
        header_idx: int,
        data_idx: int,
        data_cells: List[OcrCell],
        col_centers: List[float],
        skip_cols: Optional[set] = None,
        sticky_map: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        宽框粘连处理：拆分优先，几何重识只补空。

        1) 点对链 / N 列切先填空格（不猜无约束切点）
        2) 仅对仍空列按列中点裁剪重识（不清空已有粘连拆分）
        3) 几何结果若弱于可信拆分（截断/跳变/残片）则升级回拆分
        """
        sticky_out = self._fit_sticky_map(sticky_map, matrix)
        if not matrix or not col_centers:
            return matrix, sticky_out
        skip = set(skip_cols or ())
        step = self._median_col_step(col_centers)
        bands = self._collect_indicator_bands(
            rows, header_idx, data_idx, data_cells or []
        )
        # 带数常少于矩阵行（漏检后 pad 出空槽）：按 Y 挂带，禁止 enumerate(bands)
        # 把电价带写进中间空行（典型：2 带 + 中间 pad → bands[1]=电价 污染 matrix[1]）
        band_by_row = self._bind_bands_to_matrix_indices(bands, matrix)
        if data_cells and (len(matrix) - 1) not in band_by_row:
            band_by_row[len(matrix) - 1] = list(data_cells)

        # 按行收集宽框覆盖列（含已带多小数点的粘连串）
        row_sticky_cols: Dict[int, List[int]] = {}
        for ri, band in band_by_row.items():
            if ri < 0 or ri >= len(matrix) or not band:
                continue
            for cell in band:
                if not self._is_wide_multicolumn_sticky_cell(
                    cell, col_centers, step
                ):
                    continue
                covered = self._columns_covered_by_cell(cell, col_centers, step)
                bucket = row_sticky_cols.setdefault(ri, [])
                for j in covered:
                    if j in skip:
                        continue
                    if j not in bucket:
                        bucket.append(j)
        # 落格阶段标记的粘连格也纳入（含已拆分 / 待重识）
        for ri, srow in enumerate(sticky_out):
            for j, mark in enumerate(srow):
                if mark and j not in skip:
                    bucket = row_sticky_cols.setdefault(ri, [])
                    if j not in bucket:
                        bucket.append(j)
        if not row_sticky_cols:
            return matrix, sticky_out

        out = [list(r) for r in matrix]
        for ri, sticky_cols in row_sticky_cols.items():
            if ri >= len(out) or not sticky_cols:
                continue
            sticky_cols = sorted(int(j) for j in sticky_cols)
            row = out[ri]
            # 只用本行绑定的 OCR 带；无带则跳过（空槽留给补 0，不借电价带）
            band = band_by_row.get(ri) or []
            if not band:
                continue
            marks = sticky_out[ri] if ri < len(sticky_out) else None
            # ① 拆分优先：先用宽框原文填空
            self._fill_sticky_empties_from_dotted_chain(
                row, sticky_cols, band, col_centers, step, marks
            )
            self._fill_sticky_empties_from_n_digit_split(
                row, sticky_cols, band, col_centers, step, marks
            )
            # ② 几何重识只补仍空列；已有「粘连拆分」绝不清空
            need_reocr = [
                j
                for j in sticky_cols
                if 0 <= j < len(row)
                and row[j] != self.NODATA_MARK
                and not str(row[j] or "").strip()
            ]
            if need_reocr:
                filled = self._ocr_fill_sticky_columns(
                    img,
                    list(row),
                    need_reocr,
                    col_centers,
                    band,
                    img.shape[1],
                    skip_cols=skip,
                )
                for j in need_reocr:
                    if j >= len(filled) or not filled[j]:
                        continue
                    cand = filled[j]
                    if not self._sticky_empty_fill_plausible(cand, row):
                        continue
                    row[j] = cand
                    if marks is not None and 0 <= j < len(marks):
                        marks[j] = "粘连重识"
            # ③ 几何弱于可信拆分 → 升级回拆分（截断 / 跳变 / 残片）
            self._upgrade_sticky_truncated_from_dotted_chain(
                row, sticky_cols, band, col_centers, step, marks
            )
            self._upgrade_sticky_truncated_from_n_digit_split(
                row, sticky_cols, band, col_centers, step, marks
            )
            out[ri] = row
        return out, sticky_out

    def _split_digit_run_into_n_prices(self, digits: str, n: int) -> List[str]:
        """
        纯数字按恰好 n 段切成电价（每段 4～6 位），用邻格量级 continuity 选优。

        列数由检测框覆盖几何给出，避免「第一格到底 5 还是 6」的无约束瞎猜。
        """
        d = re.sub(r"\D", "", str(digits or ""))
        if n < 2 or len(d) < n * 4 or len(d) > n * 6:
            return []

        best: List[str] = []
        best_score = -1e18

        def _walk(start: int, left: int, acc: List[str]) -> None:
            nonlocal best, best_score
            if left == 1:
                chunk = d[start:]
                if not (4 <= len(chunk) <= 6):
                    return
                parts = acc + [chunk]
                dotted = [self._format_price_digit_chunk(p) for p in parts]
                if any(x is None for x in dotted):
                    return
                assert all(isinstance(x, str) for x in dotted)
                vals = [str(x) for x in dotted]
                if not self._sticky_pair_plausible(vals):
                    # 仍保留较平滑候选，勿整段丢弃
                    score = -50.0
                else:
                    score = 20.0 * len(vals)
                for a, b in zip(vals, vals[1:]):
                    try:
                        fa, fb = abs(float(a)), abs(float(b))
                    except ValueError:
                        return
                    ratio = max(fa, fb) / max(min(fa, fb), 1e-9)
                    if ratio <= 1.25:
                        score += 14.0
                    elif ratio <= 1.6:
                        score += 7.0
                    elif ratio <= 2.2:
                        score -= 2.0
                    else:
                        score -= 22.0
                if score > best_score:
                    best_score = score
                    best = vals
                return
            remain_min = (left - 1) * 4
            remain_max = (left - 1) * 6
            for length in (6, 5, 4):
                end = start + length
                tail = len(d) - end
                if tail < remain_min or tail > remain_max:
                    continue
                _walk(end, left - 1, acc + [d[start:end]])

        _walk(0, n, [])
        return best

    def _fill_sticky_empties_from_dotted_chain(
        self,
        row: List[str],
        sticky_cols: Sequence[int],
        band: Sequence[OcrCell],
        col_centers: Sequence[float],
        step: float,
        sticky_marks: Optional[List[str]],
    ) -> None:
        """按列重识后仍空：用宽框原文已点对电价链补空。"""
        empty_cols = [
            int(j)
            for j in sticky_cols
            if 0 <= int(j) < len(row) and not str(row[int(j)] or "").strip()
        ]
        if not empty_cols or not band:
            return
        for cell in band:
            if not self._is_wide_multicolumn_sticky_cell(cell, col_centers, step):
                continue
            parts = self._extract_dotted_price_chain(
                self._normalize_numeric_text(cell.text)
            )
            if len(parts) < 2 or not self._sticky_pair_plausible(parts):
                continue
            dotted = [p for p in parts if "." in str(p)]
            if len(dotted) < 2:
                continue
            covered = self._columns_covered_by_cell(cell, col_centers, step)
            if len(dotted) != len(covered):
                continue
            for val, col in zip(dotted, covered):
                if col not in empty_cols:
                    continue
                if not self._sticky_empty_fill_plausible(val, row):
                    continue
                row[col] = val
                if sticky_marks is not None and 0 <= col < len(sticky_marks):
                    sticky_marks[col] = "粘连拆分"
            empty_cols = [
                int(j)
                for j in sticky_cols
                if 0 <= int(j) < len(row) and not str(row[int(j)] or "").strip()
            ]
            if not empty_cols:
                return

    @staticmethod
    def _sticky_geom_weaker_than_split(existing: str, split_val: str) -> bool:
        """
        几何重识是否弱于可信拆分，应升级回拆分。

        含：截断残片（355.28 vs 355.289）、短残片（311 vs 357.605）、
        首位错读（156.289 vs 356.289）、相对量级跳变。
        """
        if not existing:
            return True
        if not split_val or existing == split_val:
            return False
        de = re.sub(r"\D", "", str(existing))
        ds = re.sub(r"\D", "", str(split_val))
        if de and ds and len(ds) > len(de) and ds.startswith(de):
            return True
        # 拆分更长且共享后缀 / 同长度首位错读（156289 vs 356289）
        if de and ds and len(de) >= 3:
            if ds.endswith(de) and len(ds) > len(de):
                return True
            if len(de) == len(ds) and len(de) >= 5 and de[1:] == ds[1:] and de[0] != ds[0]:
                return True
            if len(de) == len(ds) and len(de) >= 5 and de[-4:] == ds[-4:] and de != ds:
                return True
        try:
            fe, fs = abs(float(existing)), abs(float(split_val))
        except ValueError:
            return False
        if fs <= 0:
            return False
        if abs(fe - fs) / fs <= 0.015 and len(de) < len(ds):
            return True
        # 短整数残片 vs 完整电价
        if "." not in str(existing) and len(de) <= 3 and "." in str(split_val) and fs >= 50:
            return True
        # 相对拆分量级跳变（156 vs 356）
        ratio = max(fe, fs) / max(min(fe, fs), 1e-9)
        if ratio >= 1.45 and abs(fe - fs) >= 40.0:
            return True
        return False

    def _iter_wide_sticky_n_splits(
        self,
        band: Sequence[OcrCell],
        col_centers: Sequence[float],
        step: float,
    ):
        """产出 (covered_cols, price_parts) 的宽框 N 列切分。"""
        for cell in band:
            if not self._is_wide_multicolumn_sticky_cell(cell, col_centers, step):
                continue
            covered = self._columns_covered_by_cell(cell, col_centers, step)
            if len(covered) < 2:
                continue
            digits = re.sub(
                r"\D", "", self._normalize_numeric_text(cell.text)
            )
            parts = self._split_digit_run_into_n_prices(digits, len(covered))
            if len(parts) != len(covered):
                continue
            if not self._sticky_pair_plausible(parts):
                continue
            yield covered, parts

    def _fill_sticky_empties_from_n_digit_split(
        self,
        row: List[str],
        sticky_cols: Sequence[int],
        band: Sequence[OcrCell],
        col_centers: Sequence[float],
        step: float,
        sticky_marks: Optional[List[str]],
    ) -> None:
        """
        按列重识后仍空：用宽框数字串按覆盖列数 N 切分补空。

        例：5 列 + 29 位 → 只能是 5+6+6+6+6，不再对单格盲猜。
        """
        empty_cols = [
            int(j)
            for j in sticky_cols
            if 0 <= int(j) < len(row) and not str(row[int(j)] or "").strip()
        ]
        if not empty_cols or not band:
            return
        for covered, parts in self._iter_wide_sticky_n_splits(
            band, col_centers, step
        ):
            for val, col in zip(parts, covered):
                if col not in empty_cols:
                    continue
                if not self._sticky_empty_fill_plausible(val, row):
                    continue
                row[col] = val
                if sticky_marks is not None and 0 <= col < len(sticky_marks):
                    sticky_marks[col] = "粘连拆分"
            empty_cols = [
                int(j)
                for j in sticky_cols
                if 0 <= int(j) < len(row) and not str(row[int(j)] or "").strip()
            ]
            if not empty_cols:
                return

    def _upgrade_sticky_row_from_parts(
        self,
        row: List[str],
        sticky_cols: Sequence[int],
        covered: Sequence[int],
        parts: Sequence[str],
        sticky_marks: Optional[List[str]],
    ) -> bool:
        """
        若覆盖列中有空格或几何结果弱于 parts，则整段升级为拆分值。

        @return 是否发生升级
        """
        sticky_set = set(int(j) for j in sticky_cols)
        cols = [int(c) for c in covered]
        if not cols or any(c not in sticky_set for c in cols):
            return False
        if len(parts) != len(cols):
            return False
        need = False
        for val, col in zip(parts, cols):
            cur = str(row[col] or "").strip() if col < len(row) else ""
            if not cur or self._sticky_geom_weaker_than_split(cur, val):
                need = True
                break
            try:
                if abs(float(cur) - float(val)) / max(
                    abs(float(val)), 1e-9
                ) > 0.002:
                    # 末位不同（357.603 vs 357.605）也升级
                    if re.sub(r"\D", "", cur)[:5] == re.sub(
                        r"\D", "", val
                    )[:5]:
                        need = True
                        break
            except ValueError:
                need = True
                break
        if not need:
            return False
        for val, col in zip(parts, cols):
            if col >= len(row):
                continue
            row[col] = val
            if sticky_marks is not None and 0 <= col < len(sticky_marks):
                sticky_marks[col] = "粘连拆分"
        return True

    def _upgrade_sticky_truncated_from_dotted_chain(
        self,
        row: List[str],
        sticky_cols: Sequence[int],
        band: Sequence[OcrCell],
        col_centers: Sequence[float],
        step: float,
        sticky_marks: Optional[List[str]],
    ) -> None:
        """宽框已点对电价链完整时，覆盖几何截断/跳变/残片。"""
        if not band:
            return
        for cell in band:
            if not self._is_wide_multicolumn_sticky_cell(cell, col_centers, step):
                continue
            covered = self._columns_covered_by_cell(cell, col_centers, step)
            parts = self._try_wide_sticky_reliable_parts(cell, covered)
            if not parts:
                continue
            # 优先用带点段（与补空逻辑一致）
            dotted = [p for p in parts if "." in str(p)]
            use = dotted if len(dotted) == len(covered) else parts
            if len(use) != len(covered):
                continue
            self._upgrade_sticky_row_from_parts(
                row, sticky_cols, covered, use, sticky_marks
            )

    def _upgrade_sticky_truncated_from_n_digit_split(
        self,
        row: List[str],
        sticky_cols: Sequence[int],
        band: Sequence[OcrCell],
        col_centers: Sequence[float],
        step: float,
        sticky_marks: Optional[List[str]],
    ) -> None:
        """
        宽框数字串在列数 N 约束下切分完整时，覆盖几何截断/跳变/残片。

        拆分优先：几何弱于 N 切分时整段升级回拆分。
        """
        if not band:
            return
        for covered, parts in self._iter_wide_sticky_n_splits(
            band, col_centers, step
        ):
            self._upgrade_sticky_row_from_parts(
                row, sticky_cols, covered, parts, sticky_marks
            )

    def _sticky_empty_fill_plausible(
        self, cand: str, row: Sequence[str]
    ) -> bool:
        """空列单列 OCR 填入门禁：挡 3.1 / 5.3 / 358 / 552 等窄列残片。"""
        if self._is_monster_numeric_token(cand):
            return False
        raw = str(cand or "").strip()
        pv = self._preview_sticky_price(cand)
        # 与电价串位上限一致：≥1000 才视为粘连残骸
        if pv is None or pv >= 1000.0:
            return False
        # 粘连按列重识：无点短整数/漏点长串须先能补成带点电价
        if "." not in raw:
            if re.fullmatch(r"-?\d{1,3}", raw):
                return False
            digs = re.sub(r"\D", "", raw)
            dotted = self._format_price_digit_chunk(digs)
            if not dotted:
                return False
            raw = dotted
            cand = dotted
            pv = self._preview_sticky_price(cand)
            if pv is None or pv >= 1000.0:
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
        已有贴近电价簇的带点拆分时，重识不得用跳变错读覆盖（156 盖 356）。
        """
        if not existing:
            return True
        if not cand:
            return False
        # 几何结果本身弱于已有拆分 → 拒绝覆盖
        if self._sticky_geom_weaker_than_split(cand, existing):
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
            # 已有带点且贴近簇：重识须明显更近才覆盖（拆分优先）
            if "." in str(existing) and err_old <= max(small_med * 0.20, 25.0):
                return err_new < err_old * 0.75
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
        skip_cols: Optional[set] = None,
    ) -> List[List[str]]:
        """电价行可疑已填格按列强制重识（可覆盖串位/漏点错值）。"""
        if not matrix or not col_centers or len(matrix) < 1:
            return matrix
        skip = set(skip_cols or ())
        out = [list(r) for r in matrix]
        ri = len(out) - 1
        row = out[ri]
        # 末行全 0 且上方已有真实功率：整行当可疑（「0」默认不进可疑列表）
        force_all_zero = self._price_row_looks_like_fake_zeros(out)
        # 整行只有无点短整数残片（1/24/79）：同行无小数中位可参照，须整行强制重识
        force_short_frags = self._price_row_looks_like_short_int_frags(row)
        small_vals, small_med, med_int = self._price_row_small_stats(row)
        sus = [
            i
            for i in range(len(row))
            if i not in skip
            and row[i] != self.NODATA_MARK
            and (
                force_all_zero
                or force_short_frags
                or self._is_suspicious_price_cell(row, i, small_med, med_int)
            )
        ]
        if not sus:
            return out
        # 电价行重识一律用外推 Y（整图常无电价指示带，旧 band 易落在功率行）
        _y0, _y1, yc = self._infer_price_row_y_band(
            rows,
            header_idx,
            data_cells or [],
            img.shape[0],
            out,
            col_centers,
        )
        h_med = max(12.0, (_y1 - _y0) * 0.9)
        band = self._fake_band_at_y(col_centers, yc, h_med)
        if not band:
            return out
        short = [
            i
            for i in sus
            if "." not in str(row[i])
            and re.fullmatch(r"\d{2,3}", re.sub(r"\D", "", str(row[i]).lstrip("-")))
        ]
        # 2～3 位无点：竖纹抑制后按邻格量级择优，不用通用重识（通用通道会偏向无点高置信错读）
        # 整行残片（force_short_frags）无可靠邻格量级 → 与假全 0 一样清空后按列重识
        force_clear = force_all_zero or force_short_frags
        if short and not force_clear:
            row = self._reread_short_price_integers(img, row, col_centers, band)
        rest = [i for i in sus if i not in short] if not force_clear else list(sus)
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
            trust_cell_ocr=force_clear,
            skip_cols=skip,
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
        *,
        pad_ratio: float = 0.08,
        x_clamp: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, Optional[float]]:
        """
        单元格 OCR，同时返回检出文本块中心 X（图坐标系）。
        用于判断是否误读到邻列。

        pad_ratio：相对裁宽外扩；粘连按列重识宜小（约 0.02），避免吞邻列。
        x_clamp：(left, right) 硬边界，外扩不得越出列中点。
        """
        h, w = img.shape[:2]
        col_w = max(1, x1 - x0)
        pad_x = max(1, int(col_w * max(0.0, float(pad_ratio))))
        pad_y = 4
        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rx1 = min(w, x1 + pad_x)
        ry1 = min(h, y1 + pad_y)
        if x_clamp is not None:
            clamp_l, clamp_r = int(x_clamp[0]), int(x_clamp[1])
            if clamp_r > clamp_l:
                rx0 = max(rx0, clamp_l)
                rx1 = min(rx1, clamp_r)
        if rx1 - rx0 < 10 or ry1 - ry0 < 10:
            return "", None
        crop = img[ry0:ry1, rx0:rx1]
        # 近黑实心块（遮挡/涂黑/手指）：CLAHE 会把噪声吹成假数字（如 169.172）
        if self._cell_crop_looks_blank_or_blocked(crop):
            return "", None
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

    @staticmethod
    def _cell_crop_looks_blank_or_blocked(crop: np.ndarray) -> bool:
        """
        单格裁剪是否近黑实心/无有效笔画（遮挡涂黑、手指挡住）。

        这类块上跑 OCR+CLAHE 易臆造带小数假值；应直接当空格。
        """
        if crop is None or crop.size == 0:
            return True
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        dark_ratio = float(np.mean(gray < 45.0))
        # 实心近黑：均值低、对比弱、绝大多数像素够暗
        if mean < 40.0 and std < 28.0 and dark_ratio >= 0.78:
            return True
        if dark_ratio >= 0.92 and mean < 55.0:
            return True
        return False

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

    def _ocr_fill_sticky_columns(
        self,
        img: np.ndarray,
        row: List[str],
        sticky_cols: Sequence[int],
        col_centers: List[float],
        data_cells: Sequence[OcrCell],
        img_width: int,
        *,
        skip_cols: Optional[set] = None,
    ) -> List[str]:
        """
        宽框粘连列：按列中点边界裁剪后 OCR。

        列宽策略：
        - 左右硬边界 = 相邻列中点（_column_bounds），绝不跨列
        - 内缩约 6%～8% 列宽避开竖线，保留大部分数字区（不过窄丢小数点）
        - 外扩 pad 极小且仍夹在中点内（不过宽吞邻列）
        """
        if not row or not sticky_cols or not col_centers:
            return row
        skip = set(skip_cols or ())
        cols = [
            int(j)
            for j in sticky_cols
            if 0 <= int(j) < len(row) and int(j) not in skip and not row[int(j)]
        ]
        if not cols:
            return row

        if data_cells:
            ys = [c.y + c.height / 2 for c in data_cells]
            hs = [c.height for c in data_cells]
            y_med = float(np.median(ys))
            h_med = float(np.median(hs)) if hs else 40.0
        else:
            y_med = float(img.shape[0]) * 0.5
            h_med = 40.0
        y0 = int(max(0, y_med - h_med * 1.15))
        y1 = int(min(img.shape[0], y_med + h_med * 1.15))
        bounds = self._column_bounds(col_centers, img_width)
        step = self._median_col_step(col_centers)
        filled = list(row)

        for col in cols:
            b0, b1 = bounds[col]
            cx_i = float(col_centers[col])
            # 首/末列 bounds 可能伸到图缘：用列距夹紧，避免裁过宽吞邻格/空白
            half = step * 0.50
            x0 = max(int(b0), int(cx_i - half))
            x1 = min(int(b1), int(cx_i + half))
            if x1 - x0 < 12:
                x0, x1 = int(b0), int(b1)
            col_w = max(x1 - x0, 1)
            # 先略紧（挡邻列渗墨），再略松（补吃小数点）；均夹在列中点内
            inset_tight = max(3, int(col_w * 0.10))
            inset_mid = max(2, int(col_w * 0.07))
            inset_loose = max(2, int(col_w * 0.04))
            attempts = (
                (x0 + inset_tight, x1 - inset_tight, 0.0),
                (x0 + inset_mid, x1 - inset_mid, 0.01),
                (x0 + inset_loose, x1 - inset_loose, 0.01),
            )
            cands: List[Tuple[Tuple[int, int, float], str]] = []
            cx = float(col_centers[col])
            for ax0, ax1, pad_r in attempts:
                if ax1 - ax0 < 10:
                    continue
                cand, found_x = self._ocr_single_cell_located(
                    img,
                    int(ax0),
                    y0,
                    int(ax1),
                    y1,
                    fast=False,
                    pad_ratio=pad_r,
                    x_clamp=(int(b0), int(b1)),
                )
                if cand == self.NODATA_MARK:
                    break
                if not cand or self._is_monster_numeric_token(cand):
                    continue
                # 无点短整数：窄裁残片，继续试更宽一档
                if "." not in cand and re.fullmatch(r"-?\d{1,3}", cand.strip()):
                    continue
                # 无点 4～6 位：先补点再参与选优（35674→356.74）
                if "." not in cand:
                    digs = re.sub(r"\D", "", cand)
                    dotted = self._format_price_digit_chunk(digs)
                    if not dotted:
                        continue
                    cand = dotted
                if found_x is not None and len(col_centers) > 1:
                    nearest = min(
                        range(len(col_centers)),
                        key=lambda j: abs(found_x - col_centers[j]),
                    )
                    if nearest != col:
                        if abs(found_x - col_centers[nearest]) + step * 0.12 < abs(
                            found_x - cx
                        ):
                            continue
                stolen = False
                for nb_i in (col - 1, col + 1):
                    if 0 <= nb_i < len(filled) and self._is_neighbor_fragment(
                        cand, filled[nb_i]
                    ):
                        stolen = True
                        break
                if stolen:
                    continue
                q = self._token_quality(cand)
                cands.append((q, cand))
                # 已有带点且位数够的电价，可提前停
                if q[0] == 1 and q[1] >= 5:
                    break
            if cands:
                cands.sort(key=lambda item: item[0], reverse=True)
                filled[col] = cands[0][1]
        return filled

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
        n_cols = len(col_centers)
        for col in empty_cols:
            x0, x1 = bounds[col]
            cx = col_centers[col]
            # 列内再收紧，减少邻列干扰；首/末列少收紧、略外扩，避免电价高位/小数尾被裁
            edge = col == 0 or col == n_cols - 1
            inset = max(1, int((x1 - x0) * (0.04 if edge else 0.10)))
            pad_small = max(1, int((x1 - x0) * (0.02 if edge else 0.04)))
            pad_mid = max(3, int((x1 - x0) * (0.18 if edge else 0.12)))
            if prefer_sharp:
                attempts = (
                    (x0 + inset, y_tight0, x1 - inset, y_tight1, False),
                    (x0 + pad_small, y_tight0, x1 - pad_small, y_tight1, False),
                    (x0, y_tight0, x1, y_tight1, False),
                    (x0 - pad_mid, y_box0, x1 + pad_mid, y_box1, False),
                )
                if edge:
                    pad_wide = max(pad_mid, int((x1 - x0) * 0.28))
                    attempts = attempts + (
                        (x0 - pad_wide, y_tight0, x1 + pad_wide, y_tight1, False),
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

    def _fill_sparse_data_row(
        self,
        rows: List[List[str]],
        data_cells: List[OcrCell],
        col_centers: List[float],
        col_count: int,
        sticky_map: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        用主数据行 OCR 结果回填矩阵最后一行的空单元格。

        主行须像末行；若主行更像上方指标，禁止把第 2 行值填进第 3 行空格。
        """
        sticky_out = self._fit_sticky_map(sticky_map, rows)
        if not rows or not data_cells:
            return rows, sticky_out
        ref, ref_sticky = self._align_row_to_columns(
            self._expand_row_cells(data_cells, col_centers),
            col_centers,
            col_count,
        )
        if not ref:
            return rows, sticky_out
        last = list(rows[-1]) + [""] * max(0, col_count - len(rows[-1]))
        last = last[:col_count]
        last_sticky = list(sticky_out[-1]) + [""] * max(
            0, col_count - len(sticky_out[-1])
        )
        last_sticky = last_sticky[:col_count]
        last_n = sum(1 for v in last if v and str(v).strip())
        same_as_last = last_n >= 2 and self._aligned_rows_same_series(last, ref)
        matches_upper = False
        for upper in rows[:-1]:
            upper_n = sum(1 for v in upper if v and str(v).strip())
            if upper_n >= 3 and self._aligned_rows_same_series(upper, ref):
                matches_upper = True
                break
        if matches_upper and not same_as_last:
            return rows, sticky_out
        if last_n >= 4 and not same_as_last:
            return rows, sticky_out
        # 禁止把全 0 / 上行整型灌进电价末行：认不到就留空，不入库。
        # 仅回填「带点小数」或「≥4 位可补点」格，杜绝串行。
        def _ref_ok_for_price(v: object) -> bool:
            s = str(v or "").strip()
            if not s or self._is_price_placeholder_token(s):
                return False
            return ("." in s) or bool(re.fullmatch(r"-?\d{4,}", s))

        if not any(_ref_ok_for_price(v) for v in ref):
            return rows, sticky_out
        for i in range(col_count):
            if last[i] or i >= len(ref) or not ref[i]:
                continue
            if not _ref_ok_for_price(ref[i]):
                continue
            last[i] = ref[i]
            if i < len(ref_sticky) and ref_sticky[i]:
                last_sticky[i] = ref_sticky[i]
        rows[-1] = last
        sticky_out[-1] = last_sticky
        return rows, sticky_out

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
        """电价行可疑格：粘连长数、或相对同行中位数明显跳变 → 按列重识。"""
        del med_int
        raw = str(row[idx] or "").strip().replace("−", "-").replace("–", "-")
        if not raw or raw == "-":
            return False
        # 摩尔纹全 1（11 / 1.11 / 11.111…）先强制重识，仍全 1 再由 blank 置空
        if self._is_moire_all_ones_token(raw):
            return True
        body = raw[1:] if raw.startswith("-") and len(raw) > 1 else raw
        digits = re.sub(r"\D", "", body)
        # 已有小数点：整数 ≥4 位 / 总位 ≥7 → 粘连风险
        # 形如 834.753 但同行中位 ~380：固定 6 切串位残留，也强制重识
        if "." in body:
            int_digits = re.sub(r"\D", "", body.split(".", 1)[0])
            if len(int_digits) >= 4 or len(digits) >= 7:
                return True
            frac_n = len(re.sub(r"\D", "", body.split(".", 1)[1]))
            prefer = self._row_frac_digit_mode(row, skip_col=idx)
            dotted_peers = sum(
                1
                for j, c in enumerate(row)
                if j != idx and "." in str(c or "")
            )
            # 同行众数 3 位小数，本格少 ≥1 位（356.74 vs 356.544）→ 漏位强制重识
            # 末列交给 _drop_right_edge_short_frac_column，避免重识把 355.9「补全」成假值
            if (
                prefer >= 2
                and frac_n < prefer
                and dotted_peers >= 2
                and idx < len(row) - 1
            ):
                return True
            try:
                v = abs(float(body))
            except ValueError:
                return False
            if v <= 0:
                return False

            def _jump(ref: float) -> bool:
                if ref <= 0:
                    return False
                ratio = max(v, ref) / max(min(v, ref), 1e-9)
                # 156 vs 355：粘连重识首位错读，须进强制重识
                return ratio >= 1.50 and abs(v - ref) >= 50.0

            # 相对同行中位数跳变
            if small_med and _jump(float(small_med)):
                return True
            # 相对左右邻格跳变（串位后中位数也可能被污染）
            for j in (idx - 1, idx + 1):
                if j < 0 or j >= len(row):
                    continue
                nb = str(row[j] or "").strip().replace("−", "-").replace("–", "-")
                if not nb or "." not in nb:
                    continue
                try:
                    nv = abs(float(nb.lstrip("-") if nb.startswith("-") else nb))
                except ValueError:
                    continue
                if _jump(nv):
                    return True
            return False
        # 无点整数：
        # - ≥7 位：粘连，列重识
        # - 5/6 位：交给行级补点，不在这里清空
        # - 4 位：预览像电价则交给 xx.xx / x.xxx 选优，否则重识
        # - 1～3 位：同行已是带点电价簇时视为裁缘残片（358/552），强制重识
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
        if 1 <= len(digits) <= 3:
            if small_med and small_med >= 50:
                return True
            if any("." in str(c or "") for c in row):
                return True
        return False

    def _blank_cross_column_bleed(self, row: List[str]) -> List[str]:
        """已停用：串位成对置空过严；按「识别到什么写什么」保留 OCR 原文。"""
        return row

    # 电价行上几乎不可能的短整数全 1；直接置空（不必看量级）
    _MOIRE_PRICE_REJECT_TOKENS = frozenset({"1", "11", "111", "1111", "-1", "-11", "-111", "-1111"})

    @staticmethod
    def _is_moire_all_ones_token(text: str) -> bool:
        """数字位是否全是 1（可带符号/小数点）——摩尔纹竖线的强嫌疑形态。

        例：1 / 11 / 111 / 1.1 / 1.11 / 11.111 / 111.111
        含非 1 数字（如 348.115、11.0）不判。
        带点全 1（如真·11.111）不在此函数里直接置空，见 `_should_blank_moire_ones`。
        """
        t = str(text or "").strip().replace("−", "-").replace("–", "-").replace(",", "")
        if not t:
            return False
        if t.startswith("-"):
            t = t[1:]
        if not t or not re.fullmatch(r"\d+(\.\d+)?", t):
            return False
        digits = re.sub(r"\D", "", t)
        return len(digits) >= 1 and set(digits) == {"1"}

    def _should_blank_moire_ones(
        self, text: str, row: Sequence[str], col: int
    ) -> bool:
        """全 1 格是否置空：短整数必清；带点全 1 仅当相对同行电价簇离谱。

        流程上全 1 已先走强制重识。真·11.111 且同行也是 ~11 量级 → 保留；
        同行 ~350 却读成 11.111 → 仍当竖纹残片置空。
        """
        t = str(text or "").strip().replace("−", "-").replace("–", "-")
        if not t:
            return False
        if t in self._MOIRE_PRICE_REJECT_TOKENS:
            return True
        if not self._is_moire_all_ones_token(t):
            return False
        body = t[1:] if t.startswith("-") else t
        # 无点全 1（含更长 11111）：电价行上直接置空
        if "." not in body:
            return True
        try:
            v = abs(float(body))
        except ValueError:
            return True
        peers: List[float] = []
        for j, cell in enumerate(row):
            if j == col:
                continue
            s = str(cell or "").strip().replace("−", "-").replace("–", "-")
            if not s or "." not in s or self._is_moire_all_ones_token(s):
                continue
            try:
                fv = abs(float(s.lstrip("-") if s.startswith("-") else s))
            except ValueError:
                continue
            if 0.5 <= fv < 1000.0:
                peers.append(fv)
        if len(peers) < 2:
            # 无可靠参照：重识后仍带点全 1，宁留不杀（避免误杀真·11.111）
            return False
        med = float(np.median(peers))
        if med <= 0:
            return False
        ratio = max(v, med) / max(min(v, med), 1e-9)
        # 与电价可疑跳变同量级：11 vs 350
        return ratio >= 1.8 and abs(v - med) >= 20.0

    def _blank_moire_short_tokens(
        self, matrix: Sequence[Sequence[str]]
    ) -> List[List[str]]:
        """
        含小数的数据行上，清理摩尔纹「全 1」误读。

        优先依赖前面的强制重识；此处只做收尾：
        - 短无点 1/11/111… → 直接置空
        - 带点全 1（11.111）→ 仅当相对同行电价簇离谱才置空（保真·11.111）
        电力/计划零行由 `_coerce_zero_row_ocr_noise` 处理，本函数不动。
        """
        out: List[List[str]] = [list(r) for r in matrix]
        for ri, row in enumerate(out):
            if not any("." in str(c or "") for c in row):
                continue
            for ci, cell in enumerate(row):
                t = str(cell or "").strip()
                if self._should_blank_moire_ones(t, row, ci):
                    out[ri][ci] = ""
        return out

    def _blank_short_frac_price_cells(
        self, matrix: Sequence[Sequence[str]]
    ) -> List[List[str]]:
        """
        电价行小数位数严重短于同行众数的格置空（宁缺毋滥）。

        典型：同行 xxx.xxx，某格只剩 355.9 / 35.9（短缺 ≥2 位）。
        仅差 1 位（305.09 vs 众数 3）多为漏读，留给重识，不在此置空整格。
        末列严重残片由 `_drop_right_edge_short_frac_column` 整列丢掉。
        """
        out: List[List[str]] = [list(r) for r in matrix]
        if not out:
            return out
        pi = self._price_row_index_for_guard(out)
        if pi < 0 or pi >= len(out):
            return out
        row = out[pi]
        mode_peers = [
            self._token_frac_digit_count(str(c or "")) for c in row
        ]
        if sum(1 for n in mode_peers if n is not None) < 3:
            return out
        for i, cell in enumerate(row):
            frac = self._token_frac_digit_count(str(cell or ""))
            if frac is None:
                continue
            mode = self._row_frac_digit_mode(row, skip_col=i)
            if mode < 2 or frac >= mode:
                continue
            # 与右缘整列丢弃一致：短缺不足 2 位且整数位完整 → 保留
            shape = self._token_int_frac_lens(str(cell or ""))
            last_int = int(shape[0]) if shape else 0
            if (mode - frac) < 2 and last_int >= 2:
                continue
            peers_at_mode = sum(
                1
                for j, n in enumerate(mode_peers)
                if j != i and n == mode
            )
            if peers_at_mode >= 2:
                out[pi][i] = ""
        return out

    def _price_row_index_for_guard(self, rows: Sequence[Sequence[str]]) -> int:
        """选小数格最多的一行作电价行；没有小数则取末行。"""
        if not rows:
            return -1
        best_i, best_n = len(rows) - 1, -1
        for i, row in enumerate(rows):
            n = sum(1 for c in row if "." in str(c or ""))
            if n > best_n:
                best_n, best_i = n, i
        return best_i

    @staticmethod
    def _token_frac_digit_count(text: str) -> Optional[int]:
        """带点数值的小数位数；非带点小数则 None。"""
        s = str(text or "").strip().replace("−", "-").replace("–", "-").replace(",", "")
        if not s or "." not in s:
            return None
        if not re.fullmatch(r"-?\d+\.\d+", s):
            return None
        return len(re.sub(r"\D", "", s.rsplit(".", 1)[-1]))

    def _drop_left_edge_short_int_column(
        self,
        headers: List[str],
        rows: List[List[str]],
        sticky_map: Optional[List[List[str]]] = None,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        首列电价整数位明显短于同行众数时整列舍弃（左缘裁断缺高位）。

        典型：表头被 OCR 补成完整 06:45，但电价只剩 54.082，而内列均为 355.xxx。
        功率行常仍是完整 460，早期「多数票」不够；矩阵侧按电价行兜底。
        完整同量级首列（264.861 vs 293.187）整数位相同，不会误杀。
        """
        sticky = self._fit_sticky_map(sticky_map, rows) if rows else []
        if len(headers) < 3 or not rows:
            return list(headers), [list(r) for r in rows], sticky

        n = len(headers)
        rows = [list(r) + [""] * max(0, n - len(r)) for r in rows]
        rows = [r[:n] for r in rows]
        sticky = self._fit_sticky_map(sticky, rows)

        pi = self._price_row_index_for_guard(rows)
        if pi < 0 or pi >= len(rows):
            return headers, rows, sticky
        row = rows[pi]
        first = str(row[0] if row else "").strip()
        shape0 = self._token_int_frac_lens(first)
        if not shape0 or shape0[1] <= 0:
            return headers, rows, sticky
        first_int = int(shape0[0])

        interior_shapes: List[Tuple[int, int]] = []
        for c in row[1:]:
            s = self._token_int_frac_lens(str(c or "").strip())
            if s and s[1] > 0 and s[0] > 0:
                interior_shapes.append(s)
        if len(interior_shapes) < 2:
            return headers, rows, sticky

        med_int = float(np.median([s[0] for s in interior_shapes]))
        # 首列整数位少 ≥1（54.082 vs 355）→ 左缘缺高位
        if med_int < 2 or first_int + 0.5 >= med_int:
            return headers, rows, sticky

        headers = headers[1:]
        rows = [r[1:] for r in rows]
        sticky = [s[1:] if s else [] for s in sticky]
        sticky = self._fit_sticky_map(sticky, rows)
        return headers, rows, sticky

    def _drop_right_edge_short_frac_column(
        self,
        headers: List[str],
        rows: List[List[str]],
        sticky_map: Optional[List[List[str]]] = None,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        末列电价小数位数严重短于同行众数时整列舍弃（右缘裁断）。

        典型：表头被 OCR 补全成 01:30，但电价只剩 355.9 / 35.9，
        而内列均为 xxx.xxx。粘连宽框会跳过早期几何边缘判定，须在矩阵侧兜底。

        仅差 1 位（如 305.09 vs 众数 3）多为漏读一位，不当整列丢掉；
        须短缺 ≥2 位（355.9 / .9），或整数位也残缺（≤1 位且小数短）。
        """
        sticky = self._fit_sticky_map(sticky_map, rows) if rows else []
        if len(headers) < 3 or not rows:
            return list(headers), [list(r) for r in rows], sticky

        n = len(headers)
        rows = [list(r) + [""] * max(0, n - len(r)) for r in rows]
        rows = [r[:n] for r in rows]
        sticky = self._fit_sticky_map(sticky, rows)

        pi = self._price_row_index_for_guard(rows)
        if pi < 0 or pi >= len(rows):
            return headers, rows, sticky
        row = rows[pi]
        last = str(row[-1] if row else "").strip()
        last_frac = self._token_frac_digit_count(last)
        if last_frac is None:
            return headers, rows, sticky

        interior_dotted = sum(
            1 for c in row[:-1] if self._token_frac_digit_count(str(c or "")) is not None
        )
        if interior_dotted < 2:
            return headers, rows, sticky

        mode = self._row_frac_digit_mode(row, skip_col=len(row) - 1)
        if mode < 2 or last_frac >= mode:
            return headers, rows, sticky

        shape = self._token_int_frac_lens(last)
        last_int = int(shape[0]) if shape else 0
        # 短缺不足 2 位且整数位完整（≥2）→ 保留（防 305.09 误杀完整列）
        short_by = mode - last_frac
        if short_by < 2 and last_int >= 2:
            return headers, rows, sticky

        headers = headers[:-1]
        rows = [r[:-1] for r in rows]
        sticky = [s[:-1] if s else [] for s in sticky]
        sticky = self._fit_sticky_map(sticky, rows)
        return headers, rows, sticky

    def _drop_left_border_hour_one_column(
        self,
        headers: List[str],
        rows: List[List[str]],
        sticky_map: Optional[List[List[str]]] = None,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        首列时刻小时前导「1」为左边框伪影时整列舍弃（如 3:00→13:00）。

        表头收候选阶段已尽量按墨迹丢掉；摩尔纹下墨迹不可靠时，用该列电价
        相对内列是否离谱作旁证（贴边列数据往往一并脏读）。真·13:00 且电价
        正常的列不会被误杀。
        """
        sticky = self._fit_sticky_map(sticky_map, rows) if rows else []
        if len(headers) < 3 or not rows:
            return list(headers), [list(r) for r in rows], sticky

        n = len(headers)
        rows = [list(r) + [""] * max(0, n - len(r)) for r in rows]
        rows = [r[:n] for r in rows]
        sticky = self._fit_sticky_map(sticky, rows)

        h0 = str(headers[0] or "").strip()
        # 只处理小时以 1 开头的时刻（10:xx～19:xx），对应边框竖线误读成「1」
        if not re.match(r"^1\d:\d{2}$", h0):
            return headers, rows, sticky

        # 去掉前导 1 后须仍能成合法时刻，且比保留前导 1 更贴邻列 Δ15
        # （13:00→3:00 贴 3:15；真·15:15→5:15 反而远离 15:30，不杀）
        alt = self._normalize_time(h0[1:])  # "15:15"→"5:15"→"05:15"
        if not alt or len(headers) < 2:
            return headers, rows, sticky
        m0 = self._time_to_minutes(h0)
        m_alt = self._time_to_minutes(alt)
        m1 = self._time_to_minutes(headers[1])
        if m0 is None or m_alt is None or m1 is None:
            return headers, rows, sticky
        err0 = abs((m1 - m0) - self.DEFAULT_TIME_STEP)
        err_alt = abs((m1 - m_alt) - self.DEFAULT_TIME_STEP)
        if err_alt >= err0:
            return headers, rows, sticky

        pi = self._price_row_index_for_guard(rows)
        if pi < 0 or pi >= len(rows):
            return headers, rows, sticky
        row = rows[pi]
        raw0 = str(row[0] if row else "").strip()
        if not raw0:
            return headers, rows, sticky
        try:
            v0 = abs(float(raw0.replace(",", "")))
        except ValueError:
            return headers, rows, sticky

        interior: List[float] = []
        for c in row[1:]:
            t = str(c or "").strip()
            if not t or "." not in t:
                continue
            try:
                fv = abs(float(t.replace(",", "")))
            except ValueError:
                continue
            if fv >= 20.0:
                interior.append(fv)
        if len(interior) < 2:
            return headers, rows, sticky
        med = float(np.median(interior))
        if med < 20.0:
            return headers, rows, sticky
        # 时刻序列已偏向「去 1」+ 电价相对内列偏离 ≥18%：旁证边框伪影
        if abs(v0 - med) / med < 0.18:
            return headers, rows, sticky

        headers = headers[1:]
        rows = [r[1:] for r in rows]
        sticky = [s[1:] if len(s) > 1 else [] for s in sticky]
        sticky = self._fit_sticky_map(sticky, rows)
        return headers, rows, sticky

    def _trim_trailing_empty_and_broken_headers(
        self,
        headers: List[str],
        rows: List[List[str]],
        sticky_map: Optional[List[List[str]]] = None,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        去掉尾部全空列，并保留可桥接的严格 Δ=15 表头段。

        旧逻辑在「首次非 +15」处截断：纠偏后若左缘误读插入乱序时刻，会只剩 1 列
        → 候选被判 invalid、有效点变 0。现跳过脏列；两端因误读断开时都保留。
        """
        sticky = self._fit_sticky_map(sticky_map, rows) if rows else []
        if not headers:
            return headers, rows, sticky
        n = len(headers)
        rows = [list(r) + [""] * max(0, n - len(r)) for r in (rows or [])]
        rows = [r[:n] for r in rows]
        sticky = self._fit_sticky_map(sticky, rows)

        def _filled(j: int) -> int:
            return sum(
                1
                for r in rows
                if j < len(r)
                and str(r[j]).strip()
                not in ("", "-", "—", "–", "−", self.NODATA_MARK)
            )

        def _project(keep_idx: Sequence[int]) -> None:
            nonlocal headers, rows, sticky
            headers = [headers[i] for i in keep_idx]
            rows = [[r[i] for i in keep_idx] for r in rows]
            sticky = [[s[i] if i < len(s) else "" for i in keep_idx] for s in sticky]

        # 从右去掉全空列
        last = n - 1
        while last >= 0 and _filled(last) == 0:
            last -= 1
        if last < 0:
            # 整表数据空（屏上全 -）：保留时刻表头，数据保持空，供未出清逻辑使用
            return list(headers), [list(r) for r in rows], sticky
        headers = headers[: last + 1]
        rows = [r[: last + 1] for r in rows]
        sticky = [s[: last + 1] for s in sticky]

        keep = self._longest_delta15_header_indices(headers)
        if len(keep) >= 2 and keep != list(range(len(headers))):
            _project(keep)
        elif len(keep) == 1:
            # 构不成链时仍尽量丢掉首列非法，避免整表被废
            _project([keep[0]])

        # 再清一次尾空
        while headers and _filled(len(headers) - 1) == 0:
            headers = headers[:-1]
            rows = [r[: len(headers)] for r in rows]
            sticky = [s[: len(headers)] for s in sticky]
        return headers, rows, sticky

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

        # 已成形的 xxx.xxx：丢掉左右裁断残片，并接上紧跟的 6 位漏点电价
        # 549131.886191.917 → 131.886 / 191.917；327.304299736 → 327.304 / 299.736
        # 198.87219 → 198.872；6191.917 → 191.917
        anchored = self._extract_dotted_price_chain(t)
        if anchored:
            got = _ok_parts(anchored)
            if got and (len(got) == 1 or self._sticky_pair_plausible(got)):
                return got

        # 右缘短小数粘连：357314363.1 → 357.314/363.1（勿切成 357.31/436.31）
        # 375242384.32 → 375.242/384.32
        peeled = self._split_trailing_short_decimal(t)
        if peeled:
            got = _ok_parts(peeled)
            if got and (len(got) == 1 or self._sticky_pair_plausible(got)):
                return got

        # 电价粘连：已有若干「xxx.xxx」，中间又掉了小数点（325.508324.723311295317.19）
        if t.count(".") >= 2 and re.search(r"\d{1,3}\.\d{3}", t):
            price_parts = self._split_price_sticky_keeping_dots(t)
            if price_parts:
                got = _ok_parts(price_parts)
                if got and (len(got) == 1 or self._sticky_pair_plausible(got)):
                    return got

        # 优先拆成多段「短整数.1~3位小数」（电价常见；两点齐全可安全拆）
        # 注意：findall 会吃下「164.42.429.59.062.91」这类错一位串，必须过 plausibility
        multi_dec = re.findall(r"-?\d{1,3}\.\d{1,3}", t)
        if len(multi_dec) >= 2:
            digits_only = re.sub(r"[.\-]", "", t)
            covered = "".join(re.sub(r"-", "", p) for p in multi_dec).replace(".", "")
            if covered == digits_only or len(covered) >= len(digits_only) - 1:
                got = _ok_parts(multi_dec)
                if got and (len(got) == 1 or self._sticky_pair_plausible(got)):
                    return got
                # 点位不可信时：丢掉点按纯数字电价重切
                if len(digits_only) >= 8:
                    retry = self._split_digit_run_as_prices(
                        digits_only, allow_short=True
                    )
                    if retry:
                        fixed = _ok_parts(retry)
                        if fixed and (
                            len(fixed) == 1 or self._sticky_pair_plausible(fixed)
                        ):
                            return fixed

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
                got = _ok_parts(usable_dec)
                if got and (len(got) == 1 or self._sticky_pair_plausible(got)):
                    return got

        leading_neg = t.startswith("-")
        sticky = re.sub(r"[.\-]", "", t)
        # ≥7 位：粘连风险。8~11 位只试均衡 4~6|4~6；≥12 按电价 3+3 切
        # （319793306531206014560.816 → 319.793/306.531/206.014/560.816）
        # 右缘裁断常见非 6 倍数（…363.1 → 22 位），允许末段 4/5 位短尾
        if len(sticky) >= 7 and t.count(".") <= 1:
            if 8 <= len(sticky) <= 11:
                paired = self._split_sticky_two_cells(sticky)
                if paired:
                    fixed = _ok_parts(paired)
                    if fixed and self._sticky_pair_plausible(fixed):
                        if leading_neg and fixed:
                            fixed[0] = "-" + fixed[0].lstrip("-")
                        return fixed
            if len(sticky) >= 12:
                # 非 6 倍数必须允许末段 4 位，否则 16 位会被 6+5+5 切成 357.31/436.31
                if len(sticky) % 6 == 0:
                    price_parts = self._split_digit_run_as_prices(sticky)
                    if not price_parts:
                        price_parts = self._split_digit_run_as_prices(
                            sticky, allow_short=True
                        )
                else:
                    price_parts = self._split_digit_run_as_prices(
                        sticky, allow_short=True
                    )
                    if not price_parts:
                        price_parts = self._split_digit_run_as_prices(sticky)
                if price_parts:
                    got = _ok_parts(price_parts)
                    if got and (
                        len(got) == 1 or self._sticky_pair_plausible(got)
                    ):
                        if leading_neg and got:
                            got[0] = "-" + got[0].lstrip("-")
                        return got
            # 7 位或拆失败：不 DP 瞎切，留给几何重识
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

    def _pick_dotted_price_at(self, src: str, start: int) -> Optional[re.Match]:
        """
        在 start 处选下一个已点电价。

        默认吃足 3 位小数；若缩短为 2 位小数后，下一段恰好是「3 位整数.小数」
        （190.14298.016 → 190.14 / 298.016），则不偷下格首位。
        """
        m3 = re.match(r"(\d{1,3})\.(\d{3})", src[start:])
        if not m3:
            m2 = re.match(r"(\d{1,3})\.(\d{2})", src[start:])
            return m2
        # 缩短 1 位小数后，下一段是否更像完整电价
        short_end = start + len(m3.group(1)) + 1 + 2
        rest_short = src[short_end:]
        rest_full = src[start + m3.end() :]
        if re.match(r"\d{3}\.\d", rest_short) and re.match(r"\d{1,2}\.\d", rest_full):
            return re.match(r"(\d{1,3})\.(\d{2})", src[start:])
        return m3

    def _extract_dotted_price_chain(self, text: str) -> List[str]:
        """
        从粘连串抽出已经点对的电价（整数 ≤3 位、小数 2～3 位）。

        左侧漏点整格（如 35674+356.544→356.74）先补点收回；
        仅 1～3 位真·左缘残片在整数部已满 3 位时丢掉；
        右缘 1～2 位残片丢掉；紧跟的 6 位纯数字视为下一格漏点电价。
        盖不住则返回空，交给后面的拆法。
        """
        src = str(text or "")
        prices: List[str] = []
        pos = 0
        while True:
            # 找到下一处「数字.数字」起点
            m_find = re.search(r"\d{1,3}\.\d{2,3}", src[pos:])
            if not m_find:
                break
            abs_start = pos + m_find.start()
            m = self._pick_dotted_price_at(src, abs_start)
            if not m:
                break
            prefix = re.sub(r"\D", "", src[pos:abs_start])
            int_len = len(m.group(1))
            nxt = m.group(1) + "." + m.group(2)
            if prefix:
                # 左侧整段漏点电价：319793306531206014 + 560.816
                lead: List[str] = []
                if (
                    not prices
                    and len(prefix) >= 6
                    and len(prefix) % 6 == 0
                ):
                    lead = self._split_digit_run_as_prices(prefix)
                # 4/5 位漏点整格：35674+356.544 → 356.74，勿当左缘残片丢掉
                if (
                    not lead
                    and not prices
                    and 4 <= len(prefix) <= 5
                ):
                    one = self._format_price_digit_chunk(prefix)
                    if one and self._sticky_pair_plausible([one, nxt]):
                        lead = [one]
                if lead:
                    prices.extend(lead)
                elif prices or int_len < 3 or not (1 <= len(prefix) <= 3):
                    # 1～3 位才视为裁断残片；更长且补点失败 → 整链不可信
                    return []
            prices.append(nxt)
            pos = abs_start + m.end()
        if not prices:
            return []
        if not self._sticky_pair_plausible(prices) and len(prices) >= 2:
            return []
        tail = re.sub(r"\D", "", src[pos:])
        if not tail:
            return prices
        if len(tail) in (6, 12) and len(tail) % 6 == 0:
            for i in range(0, len(tail), 6):
                prices.append(tail[i : i + 3] + "." + tail[i + 3 : i + 6])
            if len(prices) >= 2 and not self._sticky_pair_plausible(prices):
                return []
            return prices
        if re.fullmatch(r"\d{1,2}", tail):
            return prices
        return []

    def _split_price_sticky_keeping_dots(self, text: str) -> List[str]:
        """
        保留字符串里已经点对的电价，并把中间纯数字按 3+3 位补点。

        例：325.508324.723311295317.19 → 325.508 / 324.723 / 311.295 / 317.19。
        盖不住全部数字则放弃，交给后面的拆法。
        """
        pieces: List[str] = []
        pos = 0
        for m in re.finditer(r"\d{1,3}\.\d{2,3}", text):
            gap = re.sub(r"\D", "", text[pos : m.start()])
            gap_parts = self._split_digit_run_as_prices(gap)
            if gap and not gap_parts:
                return []
            pieces.extend(gap_parts)
            pieces.append(m.group())
            pos = m.end()
        tail = re.sub(r"\D", "", text[pos:])
        tail_parts = self._split_digit_run_as_prices(tail, allow_short=True)
        if tail and not tail_parts:
            return []
        pieces.extend(tail_parts)
        digits_only = re.sub(r"[.\-]", "", text)
        covered = "".join(re.sub(r"\D", "", p) for p in pieces)
        if not pieces or covered != digits_only:
            return []
        return pieces

    def _format_price_digit_chunk(self, chunk: str) -> Optional[str]:
        """把 4/5/6 位纯数字格式化为电价小数；不合理则返回 None。"""
        d = re.sub(r"\D", "", str(chunk or ""))
        if len(d) == 6:
            return d[:3] + "." + d[3:]
        if len(d) == 5:
            # 19.396 类：前两位偏小 → xx.xxx；384.32 类 → xxx.xx
            if int(d[:2]) <= 25:
                return d[:2] + "." + d[2:]
            return d[:3] + "." + d[3:]
        if len(d) == 4:
            # 右缘半截：363.1；或 19.39
            if int(d[:2]) <= 25:
                return d[0] + "." + d[1:]
            return d[:3] + "." + d[3:]
        return None

    def _dp_split_price_digits(
        self, digits: str, *, allow_short: bool = False
    ) -> Tuple[List[str], float]:
        """
        动态规划拆纯数字电价串：允许中段混用 6 位(xxx.xxx)与 5 位(xxx.xx)。

        旧逻辑一律按 6 位切，遇到 384.32 这类 5 位会把后续全部切错：
        37524238432383475396897
          → 错：375.242 / 384.323 / 834.753 / 968.97
          → 对：375.242 / 384.32 / 383.475 / 396.897
        """
        d = re.sub(r"\D", "", str(digits or ""))
        n = len(d)
        if n < 4:
            return [], -1e18
        neg = -1e18
        best_score = [neg] * (n + 1)
        best_parts: List[Optional[List[str]]] = [None] * (n + 1)
        best_score[0] = 0.0
        best_parts[0] = []
        # 整串刚好 6 的倍数：只切 6（中段 5 会破坏真·全 xxx.xxx）
        # 余数非 0：允许中段 5（384.32）；allow_short 时末段还可 4
        if n % 6 == 0:
            lengths = (6,)
        elif allow_short:
            lengths = (6, 5, 4)
        else:
            lengths = (6, 5)
        for i in range(n):
            if best_parts[i] is None:
                continue
            for length in lengths:
                if i + length > n:
                    continue
                # 4 位只允许落在末段（右缘裁断）
                if length == 4 and i + length != n:
                    continue
                dotted = self._format_price_digit_chunk(d[i : i + length])
                if not dotted:
                    continue
                try:
                    pv = abs(float(dotted))
                except ValueError:
                    continue
                if pv >= 1000.0 or pv < 0.05:
                    continue
                # 6 位优先于 5 位：否则 3573143631 会误切 5|5→357.31/436.31，
                # 而正确是 6|4→357.314/363.1
                score = best_score[i] + {6: 12.0, 5: 8.0, 4: 6.0}[length]
                prev = best_parts[i] or []
                if prev:
                    try:
                        prev_v = abs(float(prev[-1]))
                    except ValueError:
                        prev_v = 0.0
                    ratio = max(pv, prev_v) / max(min(pv, prev_v), 1e-9)
                    # 邻格量级连续加分；突变重罚（固定 6 切串位典型 384→834）
                    if ratio <= 1.25:
                        score += 14.0
                    elif ratio <= 1.6:
                        score += 7.0
                    elif ratio <= 2.2:
                        score -= 2.0
                    else:
                        score -= 22.0
                    # 右缘 6|+4 短尾比双 5 更像裁断电价
                    if length == 4 and "." in prev[-1]:
                        prev_digits = re.sub(r"\D", "", prev[-1].split(".", 1)[0])
                        if len(prev_digits) == 3:
                            score += 3.0
                j = i + length
                if score > best_score[j]:
                    best_score[j] = score
                    best_parts[j] = prev + [dotted]
        if best_parts[n] is None:
            return [], neg
        return best_parts[n] or [], best_score[n]

    def _split_digit_run_as_prices(self, digits: str, *, allow_short: bool = False) -> List[str]:
        """
        纯数字粘连 → 电价列表。

        优先 DP 混切 5/6 位；并尝试去掉左右各 1 位残片（裁断/串位多 1 位时，
        24 位错串 375242…968973 去掉末位即可回到正确 23 位切分）。
        """
        d = re.sub(r"\D", "", str(digits or ""))
        if not d:
            return []
        if len(d) < 4:
            return []
        if len(d) <= 6:
            one = self._format_price_digit_chunk(d)
            return [one] if one else []

        # 整串长度 %6==0：先纯 6 切；量级正常则直接采用
        # （避免 allow_short 裁掉 1～2 位后再用 5 位误切）
        if len(d) % 6 == 0:
            pure6, _sc6 = self._dp_split_price_digits(d, allow_short=False)
            if pure6 and self._sticky_pair_plausible(pure6):
                return pure6

        cands: List[Tuple[float, List[str]]] = []
        trim_opts = [(0, 0), (0, 1), (1, 0)]
        if allow_short:
            trim_opts.extend([(0, 2), (1, 1)])
        seen: set = set()
        for trim_l, trim_r in trim_opts:
            if trim_l + trim_r >= len(d):
                continue
            sub = d[trim_l : len(d) - trim_r if trim_r else len(d)]
            if len(sub) < 4 or sub in seen:
                continue
            seen.add(sub)
            parts, score = self._dp_split_price_digits(sub, allow_short=allow_short)
            if not parts:
                continue
            score -= float(trim_l + trim_r) * 4.0
            score += float(len(parts)) * 0.5
            cands.append((score, parts))
        if not cands:
            return []
        cands.sort(key=lambda x: -x[0])
        best_parts = cands[0][1]
        # 安全网：最优切分量级像串位双峰时，优先换成更平滑的候选；
        # 若没有更好候选仍返回最优（电价日内可大幅波动，勿整段丢弃）
        if len(best_parts) >= 3 and not self._sticky_pair_plausible(best_parts):
            for _sc, parts in cands[1:]:
                if self._sticky_pair_plausible(parts):
                    return parts
        return best_parts

    def _split_trailing_short_decimal(self, text: str) -> List[str]:
        """
        以右缘短小数为锚拆粘连：整数段末 3 位 + 小数 → 右格，左侧按电价切。

        357314363.1 → 357.314 / 363.1
        375242384.32 → 375.242 / 384.32
        383.475396897357314363.1 → 383.475 / 396.897 / 357.314 / 363.1
        """
        t = str(text or "").strip()
        m = re.fullmatch(r"(\d{6,})\.(\d{1,2})", t)
        if m:
            body, frac = m.group(1), m.group(2)
            right = body[-3:] + "." + frac
            left_digits = body[:-3]
            if not left_digits:
                return [right]
            left_parts = self._split_digit_run_as_prices(left_digits)
            if not left_parts:
                left_parts = self._split_digit_run_as_prices(
                    left_digits, allow_short=True
                )
            if not left_parts and 4 <= len(left_digits) <= 6:
                left_parts = [self._dot_sticky_digit_token(left_digits)]
            if not left_parts:
                return []
            got = left_parts + [right]
            return got if self._sticky_pair_plausible(got) else []

        # 前段已带 xxx.xxx、末尾又是短小数（右缘裁断常见）
        m2 = re.search(r"(\d{3})\.(\d{1,2})$", t)
        if not m2:
            return []
        left = t[: m2.start()]
        if not re.search(r"\d{3}", left):
            return []
        right = m2.group(1) + "." + m2.group(2)
        left_parts = self._split_price_sticky_keeping_dots(left)
        if not left_parts:
            sticky = re.sub(r"[.\-]", "", left)
            left_parts = self._split_digit_run_as_prices(sticky)
            if not left_parts:
                left_parts = self._split_digit_run_as_prices(
                    sticky, allow_short=True
                )
        if not left_parts:
            return []
        got = left_parts + [right]
        return got if self._sticky_pair_plausible(got) else []

    def _dot_sticky_digit_token(self, digits: str) -> str:
        """按电价常见形态给粘连段补点（与 _preview_sticky_price 规则一致）。"""
        d = re.sub(r"\D", "", str(digits or ""))
        if not d:
            return str(digits or "")
        if len(d) == 6:
            return d[:3] + "." + d[3:]
        if len(d) == 5:
            # head2≤25 → xx.xxx（19.396）；否则 xxx.xx（384.32）
            if int(d[:2]) <= 25:
                return d[:2] + "." + d[2:]
            return d[:3] + "." + d[3:]
        if len(d) == 4:
            # 与 _format_price_digit_chunk 一致：363.1 / 1.205
            if int(d[:2]) <= 25:
                return d[0] + "." + d[1:]
            return d[:3] + "." + d[3:]
        return d

    def _split_sticky_two_cells(self, digits: str) -> List[str]:
        """
        8~11 位纯数字两格粘连：枚举均衡切分（每段 4~6 位）。

        典型：1939619307 → 19396|19307（再经补点 → 19.396 / 19.307）；
        37524238432 → 375242|38432 → 375.242 / 384.32（勿切成 238432）。
        拒绝 193|969307 这类 3+6。
        """
        n = len(digits)
        if n < 8 or n > 11 or not digits.isdigit():
            return []
        cands: List[Tuple[int, List[str], float]] = []
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
            # 邻格量级接近加分（19.396↔19.307；375.242↔384.32）
            ratio = max(a, b) / max(min(a, b), 1e-9)
            if ratio <= 2.0:
                score += 6
            elif ratio <= 4.0:
                score += 2
            else:
                score -= 4
            # 10 位右缘裁断：6|4→357.314/363.1 应优于 5|5→357.31/436.31
            if left == 6 and right == 4 and ratio <= 1.5:
                score += 12
            cands.append((score, parts, ratio))
        if not cands:
            return []
        # 同分时取量级更接近的切分（11 位：6|5 优于误切 5|6→238432）
        cands.sort(key=lambda x: (-x[0], x[2]))
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
                return [
                    self._dot_sticky_digit_token(left),
                    self._dot_sticky_digit_token(trial),
                ]
            return [self._dot_sticky_digit_token(left)]
        if best_ratio > 2.0:
            keep = (
                best_parts[0]
                if len(best_parts[0]) >= len(best_parts[1])
                else best_parts[1]
            )
            return [self._dot_sticky_digit_token(keep)]
        return [self._dot_sticky_digit_token(p) for p in best_parts]

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
                # 19.396 类：前两位偏小 → xx.xxx；384.32 类 → xxx.xx
                # （旧逻辑一律 xx.xxx，会把 375242|38432 错评成 375/38，再误切 238432）
                if int(d[:2]) <= 25:
                    return abs(float(d[:2] + "." + d[2:]))
                return abs(float(d[:3] + "." + d[3:]))
            if len(d) == 4:
                # 与 _format_price_digit_chunk 一致：363.1；勿用 36.31 导致 6|4 落败于 5|5
                if int(d[:2]) <= 25:
                    return abs(float(d[0] + "." + d[1:]))
                return abs(float(d[:3] + "." + d[3:]))
            if 1 <= len(d) <= 3:
                return abs(float(d))
        except ValueError:
            return None
        return None

    def _sticky_pair_plausible(self, parts: Sequence[str]) -> bool:
        """粘连拆出的多段是否像正常表体（挡 193+969.307、以及 384→834 串位）。"""
        if len(parts) < 2:
            return True
        vals: List[float] = []
        for p in parts:
            raw = str(p).strip().replace("−", "-").replace("–", "-").replace("—", "-")
            body = raw[1:] if raw.startswith("-") and len(raw) > 1 else raw
            # 前导零整数部（062.91）：典型「xxx.xx 错一位」串位，不是真电价
            if "." in body:
                int_part = body.split(".", 1)[0]
                if re.fullmatch(r"0\d+", int_part):
                    return False
            pv = self._preview_sticky_price(str(p))
            if pv is None:
                return False
            if pv >= 1000.0:
                return False
            vals.append(pv)
        # 全是无点短整数且都偏小（24+79）：电价粘连拆残，不是真功率对
        if all(
            ("." not in str(p).strip())
            and re.fullmatch(r"-?\d{1,3}", str(p).strip())
            and abs(vals[i]) < 100.0
            for i, p in enumerate(parts)
        ):
            return False
        # 一段是无点短整数、另一段预览已很大 → 典型误拆
        for i, p in enumerate(parts):
            raw = str(p).strip()
            if "." not in raw and re.fullmatch(r"-?\d{1,3}", raw):
                others = [vals[j] for j in range(len(vals)) if j != i]
                if others and max(others) >= 200.0:
                    return False
        # 邻段跳变：164.42→429.59→062.91 这类错一位串，半段中位数可能仍「看起来还行」
        if len(vals) >= 3:
            jump_n = 0
            for a, b in zip(vals, vals[1:]):
                lo, hi = min(a, b), max(a, b)
                if lo > 0 and hi / lo >= 1.85 and (hi - lo) >= 80.0:
                    jump_n += 1
            if jump_n >= max(2, len(vals) // 2):
                return False
        # 固定 6 切串位常呈「前半一簇、后半另一簇」双峰
        # （375/384 后突然 834/968）；正常电价波动多为单峰或缓变，不拦
        if len(vals) >= 4:
            mid = len(vals) // 2
            left_m = float(np.median(vals[:mid]))
            right_m = float(np.median(vals[mid:]))
            lo, hi = min(left_m, right_m), max(left_m, right_m)
            if lo > 0 and hi / lo >= 1.8 and (hi - lo) >= 150.0:
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
