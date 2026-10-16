# -*- coding: utf-8 -*-
"""海康机器人 GigE 工业相机抓图（MVS SDK / MvImport）。

依赖本机已安装 MVS，且环境变量 MVCAM_COMMON_RUNENV 指向 Development 目录。
参考官方示例：
  General/ImageSave/ImageSave.py
  General/ConnectSpecCamera/ConnectSpecCamera.py
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import time
from ctypes import byref, cast, memset, sizeof, POINTER
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .camera_common import HikvisionCaptureError, unique_time_path as _unique_time_path

logger = logging.getLogger(__name__)

_sdk_lock = threading.Lock()
_sdk_inited = False
_MvCamera = None  # type: ignore
_consts: Dict[str, Any] = {}


def _mvs_development_root() -> Path:
    env = os.environ.get("MVCAM_COMMON_RUNENV", "").strip()
    if env:
        return Path(env)
    # 常见默认安装路径
    for cand in (
        Path(r"C:\Program Files (x86)\MVS\Development"),
        Path(r"C:\Program Files\MVS\Development"),
    ):
        if cand.is_dir():
            return cand
    raise HikvisionCaptureError(
        "未找到 MVS Development 目录：请确认已安装完整 MVS，"
        "且环境变量 MVCAM_COMMON_RUNENV 已设置。"
    )


def _ensure_runtime_dll_path() -> None:
    """把 MVS Runtime 加入 PATH / add_dll_directory，供加载 MvCameraControl.dll。"""
    roots = [
        Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime"),
        Path(r"C:\Program Files\Common Files\MVS\Runtime"),
        _mvs_development_root().parent / "Runtime",
    ]
    sub = "Win64_x64" if sys.maxsize > 2**32 else "Win32_i86"
    for root in roots:
        dll_dir = root / sub
        if not dll_dir.is_dir():
            continue
        dll_str = str(dll_dir)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_str)
            except (OSError, FileNotFoundError):
                pass
        path = os.environ.get("PATH", "")
        if dll_str.lower() not in path.lower():
            os.environ["PATH"] = dll_str + os.pathsep + path
        return


def _ensure_mvimport() -> None:
    """将官方 MvImport 加入 sys.path 并 import SDK 符号。"""
    global _MvCamera, _consts
    if _MvCamera is not None:
        return

    dev = _mvs_development_root()
    mvimport = dev / "Samples" / "Python" / "MvImport"
    if not mvimport.is_dir():
        raise HikvisionCaptureError(f"未找到 MvImport: {mvimport}")

    _ensure_runtime_dll_path()
    mv_str = str(mvimport)
    if mv_str not in sys.path:
        sys.path.insert(0, mv_str)

    # MvImport 内部用相对 import，需在该目录上下文加载
    from MvCameraControl_class import (  # type: ignore
        MV_ACCESS_Exclusive,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_CC_IMAGE,
        MV_CC_SAVE_IMAGE_PARAM,
        MV_FRAME_OUT,
        MV_GIGE_DEVICE,
        MV_GIGE_DEVICE_INFO,
        MV_Image_Jpeg,
        MV_TRIGGER_MODE_OFF,
        MvCamera,
    )

    _MvCamera = MvCamera
    _consts = {
        "MV_ACCESS_Exclusive": MV_ACCESS_Exclusive,
        "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
        "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
        "MV_CC_IMAGE": MV_CC_IMAGE,
        "MV_CC_SAVE_IMAGE_PARAM": MV_CC_SAVE_IMAGE_PARAM,
        "MV_FRAME_OUT": MV_FRAME_OUT,
        "MV_GIGE_DEVICE": MV_GIGE_DEVICE,
        "MV_GIGE_DEVICE_INFO": MV_GIGE_DEVICE_INFO,
        "MV_Image_Jpeg": MV_Image_Jpeg,
        "MV_TRIGGER_MODE_OFF": MV_TRIGGER_MODE_OFF,
    }


def _ensure_sdk() -> None:
    global _sdk_inited
    _ensure_mvimport()
    with _sdk_lock:
        if _sdk_inited:
            return
        ret = _MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise HikvisionCaptureError(f"MV_CC_Initialize 失败 ret=0x{ret:x}")
        _sdk_inited = True

        def _finalize() -> None:
            global _sdk_inited
            try:
                if _sdk_inited and _MvCamera is not None:
                    _MvCamera.MV_CC_Finalize()
            except Exception:
                pass
            _sdk_inited = False

        atexit.register(_finalize)


def _parse_ipv4(ip: str) -> int:
    parts = ip.strip().split(".")
    if len(parts) != 4:
        raise HikvisionCaptureError(f"无效 IPv4: {ip!r}")
    try:
        a, b, c, d = (int(x) for x in parts)
    except ValueError as e:
        raise HikvisionCaptureError(f"无效 IPv4: {ip!r}") from e
    if not all(0 <= x <= 255 for x in (a, b, c, d)):
        raise HikvisionCaptureError(f"无效 IPv4: {ip!r}")
    return (a << 24) | (b << 16) | (c << 8) | d


def _ipv4_str(n: int) -> str:
    return (
        f"{(n >> 24) & 0xFF}."
        f"{(n >> 16) & 0xFF}."
        f"{(n >> 8) & 0xFF}."
        f"{n & 0xFF}"
    )


def _gige_current_ip(dev_info: Any) -> Optional[str]:
    try:
        n = int(dev_info.SpecialInfo.stGigEInfo.nCurrentIp)
    except Exception:
        return None
    if n == 0:
        return None
    return _ipv4_str(n)


class HikrobotMvsClient:
    """按 IP 打开 GigE 相机，软连续取流抓一帧并保存 JPEG。"""

    def __init__(
        self,
        host: str,
        timeout_sec: float = 15,
        capture_dir: str = "data/captures",
        net_export: str = "",
        jpeg_quality: int = 80,
        *,
        name_label: str = "",
        side: str = "",
    ):
        self.host = host.strip()
        self.timeout_sec = float(timeout_sec)
        self.capture_dir = Path(capture_dir)
        self.net_export = (net_export or "").strip()
        self.jpeg_quality = max(50, min(99, int(jpeg_quality)))
        self.name_label = str(name_label or side or "").strip()
        self.side = str(side or "").strip()

    @property
    def picture_url(self) -> str:
        return f"gige://{self.host}"

    def _create_handle(self, cam: Any) -> Any:
        """优先枚举匹配 host；失败则用 ConnectSpecCamera 方式按 IP 建句柄。"""
        C = _consts
        device_list = C["MV_CC_DEVICE_INFO_LIST"]()
        ret = _MvCamera.MV_CC_EnumDevices(C["MV_GIGE_DEVICE"], device_list)
        if ret != 0:
            raise HikvisionCaptureError(f"枚举 GigE 设备失败 ret=0x{ret:x}")

        target = self.host
        matched = None
        for i in range(int(device_list.nDeviceNum)):
            info = cast(
                device_list.pDeviceInfo[i], POINTER(C["MV_CC_DEVICE_INFO"])
            ).contents
            cur_ip = _gige_current_ip(info)
            if cur_ip == target:
                matched = info
                break

        if matched is not None:
            ret = cam.MV_CC_CreateHandle(matched)
            if ret != 0:
                raise HikvisionCaptureError(f"CreateHandle 失败 ret=0x{ret:x}")
            return matched

        if int(device_list.nDeviceNum) == 0 and not self.net_export:
            raise HikvisionCaptureError(
                f"未发现 GigE 相机，且未配置 cameras.*.net_export；"
                f"目标 IP={target}。请先用 MVS 客户端确认相机在线。"
            )

        # 按 IP 直接连接（需本机网卡 IP）
        if not self.net_export:
            # 仅一台设备时退回索引 0
            if int(device_list.nDeviceNum) == 1:
                info = cast(
                    device_list.pDeviceInfo[0], POINTER(C["MV_CC_DEVICE_INFO"])
                ).contents
                logger.warning(
                    "未匹配到 IP=%s，使用唯一在线设备 %s",
                    target,
                    _gige_current_ip(info),
                )
                ret = cam.MV_CC_CreateHandle(info)
                if ret != 0:
                    raise HikvisionCaptureError(f"CreateHandle 失败 ret=0x{ret:x}")
                return info
            raise HikvisionCaptureError(
                f"未找到 IP={target} 的相机（在线 {device_list.nDeviceNum} 台）。"
                f"可在 config.yaml 设置 cameras.*.net_export 为本机网卡 IP 后按 IP 直连。"
            )

        st_dev = C["MV_CC_DEVICE_INFO"]()
        st_gige = C["MV_GIGE_DEVICE_INFO"]()
        st_gige.nCurrentIp = _parse_ipv4(target)
        st_gige.nNetExport = _parse_ipv4(self.net_export)
        st_dev.nTLayerType = C["MV_GIGE_DEVICE"]
        st_dev.SpecialInfo.stGigEInfo = st_gige
        ret = cam.MV_CC_CreateHandle(st_dev)
        if ret != 0:
            raise HikvisionCaptureError(
                f"按 IP 创建句柄失败 ret=0x{ret:x} "
                f"(camera={target}, net_export={self.net_export})"
            )
        return st_dev

    def _save_jpeg(self, cam: Any, frame: Any, out_path: Path) -> None:
        C = _consts
        # SDK 要求 ascii 路径；项目路径一般为英文盘符
        path_str = str(out_path.resolve())
        try:
            path_str.encode("ascii")
        except UnicodeEncodeError:
            # 回退到系统临时目录再拷贝
            import shutil
            import tempfile

            fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="mvs_")
            os.close(fd)
            tmp_path = Path(tmp)
            try:
                self._save_jpeg(cam, frame, tmp_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(out_path))
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            return

        st_image = C["MV_CC_IMAGE"]()
        memset(byref(st_image), 0, sizeof(st_image))
        info = frame.stFrameInfo
        # 新 SDK 优先 Extend 字段
        width = getattr(info, "nExtendWidth", 0) or info.nWidth
        height = getattr(info, "nExtendHeight", 0) or info.nHeight
        frame_len = getattr(info, "nFrameLenEx", 0) or info.nFrameLen
        st_image.nWidth = width
        st_image.nHeight = height
        st_image.enPixelType = info.enPixelType
        st_image.pImageBuf = frame.pBufAddr
        st_image.nImageBufSize = frame_len
        st_image.nImageLen = frame_len

        st_save = C["MV_CC_SAVE_IMAGE_PARAM"]()
        memset(byref(st_save), 0, sizeof(st_save))
        st_save.enImageType = C["MV_Image_Jpeg"]
        st_save.iMethodValue = 1
        st_save.nQuality = self.jpeg_quality
        st_save.nEndian = 0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ret = cam.MV_CC_SaveImageToFileEx2(st_image, st_save, path_str)
        if ret != 0:
            raise HikvisionCaptureError(f"保存 JPEG 失败 ret=0x{ret:x} -> {path_str}")

    def capture(self) -> Path:
        """打开相机抓一帧 JPEG 并落盘，返回文件路径。"""
        _ensure_sdk()
        C = _consts
        cam = _MvCamera()
        opened = False
        grabbing = False
        handle_ok = False
        frame = C["MV_FRAME_OUT"]()
        memset(byref(frame), 0, sizeof(frame))
        got_frame = False
        t0 = time.perf_counter()

        try:
            st_dev = self._create_handle(cam)
            handle_ok = True

            ret = cam.MV_CC_OpenDevice(C["MV_ACCESS_Exclusive"], 0)
            if ret != 0:
                raise HikvisionCaptureError(
                    f"打开相机失败 ret=0x{ret:x}（请确认 MVS 客户端未占用设备）"
                )
            opened = True

            # GigE 最优包大小
            try:
                n_layer = int(getattr(st_dev, "nTLayerType", C["MV_GIGE_DEVICE"]))
            except Exception:
                n_layer = C["MV_GIGE_DEVICE"]
            if n_layer == C["MV_GIGE_DEVICE"]:
                n_packet = cam.MV_CC_GetOptimalPacketSize()
                if int(n_packet) > 0:
                    ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(n_packet))
                    if ret != 0:
                        logger.warning("设置 GevSCPSPacketSize 失败 ret=0x%x", ret)
                else:
                    logger.warning("GetOptimalPacketSize 失败 ret=0x%x", int(n_packet))

            ret = cam.MV_CC_SetEnumValue("TriggerMode", C["MV_TRIGGER_MODE_OFF"])
            if ret != 0:
                raise HikvisionCaptureError(f"设置 TriggerMode=Off 失败 ret=0x{ret:x}")

            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise HikvisionCaptureError(f"StartGrabbing 失败 ret=0x{ret:x}")
            grabbing = True

            timeout_ms = max(1000, int(self.timeout_sec * 1000))
            ret = cam.MV_CC_GetImageBuffer(frame, timeout_ms)
            if frame.pBufAddr is None or ret != 0:
                raise HikvisionCaptureError(
                    f"取流超时/失败 ret=0x{ret:x}（timeout={timeout_ms}ms, host={self.host}）"
                )
            got_frame = True

            now = datetime.now()
            out_dir = self.capture_dir / now.strftime("%Y%m%d")
            out_path = _unique_time_path(out_dir, now, label=self.name_label)
            self._save_jpeg(cam, frame, out_path)

            size = out_path.stat().st_size if out_path.is_file() else 0
            from .camera_common import cam_log_prefix

            ms = max(0, int(round((time.perf_counter() - t0) * 1000)))
            logger.info(
                "%s抓图完成 ms=%s type=mvs host=%s bytes=%d path=%s",
                cam_log_prefix(self.side or self.name_label),
                ms,
                self.host,
                size,
                out_path,
            )
            return out_path
        finally:
            if got_frame:
                try:
                    cam.MV_CC_FreeImageBuffer(frame)
                except Exception:
                    pass
            if grabbing:
                try:
                    cam.MV_CC_StopGrabbing()
                except Exception:
                    pass
            if opened:
                try:
                    cam.MV_CC_CloseDevice()
                except Exception:
                    pass
            if handle_ok:
                try:
                    cam.MV_CC_DestroyHandle()
                except Exception:
                    pass
