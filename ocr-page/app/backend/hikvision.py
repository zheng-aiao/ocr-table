# -*- coding: utf-8 -*-
"""海康 ISAPI 抓图（Digest / Basic 自动尝试）。

通道说明：
  设备「通道个数=1」表示物理通道数；ISAPI 流通道号通常是：
    主码流 = 物理通道*100 + 1  →  通道1 主码流 = 101
  config 里可写 1 或 101，客户端会规范成 101。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from .camera_common import (
    HikvisionCaptureError,
    format_capture_stem,
    unique_time_path as _unique_time_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HikvisionCaptureError",
    "HikvisionIsapiClient",
    "build_camera_client",
    "build_camera_client_for_side",
    "format_capture_stem",
    "normalize_stream_channel",
]


def normalize_stream_channel(channel: str | int) -> str:
    """
    将配置中的通道规范为 ISAPI Streaming 通道号。

    - 1 / "1" → "101"（物理通道1主码流）
    - 101 / "101" → "101"
    - 2 → "201"
    """
    raw = str(channel).strip()
    if not raw.isdigit():
        return raw
    n = int(raw)
    if 1 <= n <= 64:
        return f"{n}01"
    return raw


class HikvisionIsapiClient:
    """GET /ISAPI/Streaming/channels/{channel}/picture"""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        port: int = 80,
        channel: str = "101",
        timeout_sec: float = 15,
        capture_dir: str = "data/captures",
        *,
        name_label: str = "",
        side: str = "",
    ):
        self.host = host.strip()
        self.port = int(port)
        self.user = user
        self.password = password
        self.channel = normalize_stream_channel(channel)
        self.timeout_sec = timeout_sec
        self.capture_dir = Path(capture_dir)
        self.name_label = str(name_label or side or "").strip()
        self.side = str(side or "").strip()

    def _urls(self) -> List[str]:
        """优先标准 101 路径，再试物理通道号写法（部分固件兼容）。"""
        base = f"http://{self.host}:{self.port}/ISAPI/Streaming/channels"
        urls = [f"{base}/{self.channel}/picture"]
        if self.channel.endswith("01") and len(self.channel) == 3:
            phys = str(int(self.channel) // 100)
            alt = f"{base}/{phys}/picture"
            if alt not in urls:
                urls.append(alt)
        return urls

    @property
    def picture_url(self) -> str:
        return self._urls()[0]

    def _auths(self) -> List[Tuple[str, Any]]:
        return [
            ("digest", HTTPDigestAuth(self.user, self.password)),
            ("basic", HTTPBasicAuth(self.user, self.password)),
        ]

    def capture(self) -> Path:
        """抓拍 JPEG 并落盘，返回文件路径。"""
        last_err: Optional[str] = None
        for url in self._urls():
            for auth_name, auth in self._auths():
                try:
                    resp = requests.get(
                        url,
                        auth=auth,
                        timeout=self.timeout_sec,
                        headers={"Accept": "image/jpeg,*/*"},
                    )
                except requests.RequestException as e:
                    last_err = f"请求失败 ({auth_name}) {url}: {e}"
                    logger.warning(last_err)
                    continue

                if resp.status_code == 401:
                    last_err = (
                        f"401 Unauthorized ({auth_name}) {url} — "
                        f"请核对 cameras.*.user / password "
                        f"（当前用户={self.user!r}），并在相机网页确认账号可预览。"
                    )
                    logger.warning(last_err)
                    continue

                if resp.status_code != 200:
                    last_err = (
                        f"HTTP {resp.status_code} ({auth_name}) {url}, "
                        f"body={resp.text[:200]!r}"
                    )
                    logger.warning(last_err)
                    continue

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "image" not in ctype and not resp.content.startswith(b"\xff\xd8"):
                    last_err = (
                        f"响应不是图片 ({auth_name}) Content-Type={ctype}, "
                        f"size={len(resp.content)}"
                    )
                    logger.warning(last_err)
                    continue

                now = datetime.now()
                day = now.strftime("%Y%m%d")
                out_dir = self.capture_dir / day
                out_dir.mkdir(parents=True, exist_ok=True)
                path = _unique_time_path(out_dir, now, label=self.name_label)
                path.write_bytes(resp.content)
                from .camera_common import cam_log_prefix

                logger.info(
                    "%s抓图完成 path=%s bytes=%d type=isapi host=%s",
                    cam_log_prefix(self.side or self.name_label),
                    path,
                    len(resp.content),
                    self.host,
                )
                return path

        raise HikvisionCaptureError(last_err or "抓图失败：未知错误")


def build_camera_client(
    cam_cfg: Dict[str, Any],
    *,
    capture_dir: str,
    side: str = "",
):
    """按单路 cameras.{side} 配置构建抓图客户端。

    - mvs / gige / hikrobot：海康机器人 GigE（MVS SDK）
    - isapi：海康监控机 HTTP ISAPI
    """
    cam = dict(cam_cfg or {})
    ctype = str(cam.get("type") or "").strip().lower()
    if not ctype:
        raise HikvisionCaptureError(
            f"cameras.{side or '?'}.type 未配置（mvs | isapi）"
        )
    name_label = str(side or cam.get("name") or "").strip() or side
    host = str(cam.get("host") or "").strip()
    if not host:
        raise HikvisionCaptureError(
            f"cameras.{side or '?'}.host 未配置"
        )

    if ctype in ("mvs", "gige", "hikrobot"):
        from .hikrobot_mvs import HikrobotMvsClient

        return HikrobotMvsClient(
            host=host,
            timeout_sec=float(cam.get("timeout_sec", 15)),
            capture_dir=str(capture_dir),
            net_export=str(cam.get("net_export", "") or ""),
            jpeg_quality=int(cam.get("jpeg_quality", 80)),
            name_label=name_label,
            side=side,
        )

    if ctype != "isapi":
        raise HikvisionCaptureError(
            f"cameras.{side or '?'}.type={ctype!r} 不支持，请用 mvs 或 isapi"
        )

    return HikvisionIsapiClient(
        host=host,
        port=int(cam.get("port", 80)),
        user=cam.get("user", "admin"),
        password=cam.get("password", ""),
        channel=str(cam.get("channel", "101")),
        timeout_sec=float(cam.get("timeout_sec", 15)),
        capture_dir=str(capture_dir),
        name_label=name_label,
        side=side,
    )


def build_camera_client_for_side(
    cfg: Dict[str, Any],
    root_dir: Path,
    side: str,
    cam_cfg: Optional[Dict[str, Any]] = None,
):
    """从 cfg.cameras.{side} 构建客户端。"""
    from .config import resolve_project_path

    del root_dir
    cams = cfg.get("cameras") or {}
    side_cam = dict(cam_cfg if cam_cfg is not None else (cams.get(side) or {}))
    if not side_cam:
        raise HikvisionCaptureError(f"cameras.{side} 未配置")
    capture_dir = resolve_project_path(
        (cfg.get("capture") or {}).get("dir", "data/captures")
    )
    return build_camera_client(side_cam, capture_dir=str(capture_dir), side=side)
