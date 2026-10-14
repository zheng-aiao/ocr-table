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

logger = logging.getLogger(__name__)


class HikvisionCaptureError(RuntimeError):
    pass


def format_capture_stem(when: Optional[datetime] = None) -> str:
    """
    抓图文件名主干：H-MM（如 9-09、10-20）。
    Windows 文件名不能含 ':'，用 '-' 代替；日期由外层 YYYYMMDD 目录表达。
    """
    now = when or datetime.now()
    return f"{now.hour}-{now.minute:02d}"


def _unique_time_path(out_dir: Path, when: Optional[datetime] = None) -> Path:
    """生成不冲突的抓图路径：优先 9-09.jpg，同分钟重复则 9-09_2.jpg …"""
    stem = format_capture_stem(when)
    path = out_dir / f"{stem}.jpg"
    if not path.exists():
        return path
    n = 2
    while True:
        cand = out_dir / f"{stem}_{n}.jpg"
        if not cand.exists():
            return cand
        n += 1


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
    ):
        self.host = host.strip()
        self.port = int(port)
        self.user = user
        self.password = password
        self.channel = normalize_stream_channel(channel)
        self.timeout_sec = timeout_sec
        self.capture_dir = Path(capture_dir)

    def _urls(self) -> List[str]:
        """优先标准 101 路径，再试物理通道号写法（部分固件兼容）。"""
        base = f"http://{self.host}:{self.port}/ISAPI/Streaming/channels"
        urls = [f"{base}/{self.channel}/picture"]
        # 若规范后是 101，也试一下 /1/picture（少数固件）
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
                        f"请核对 config.yaml 中 camera.user / camera.password "
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
                # 文件名用直观时刻（Windows 不能用冒号，故用 9-09 / 10-20）；
                # 外层目录 YYYYMMDD 表示日期，文件名表示当日时刻。
                path = _unique_time_path(out_dir, now)
                path.write_bytes(resp.content)
                logger.info(
                    "抓图成功 %s (%d bytes) <- %s [%s]",
                    path,
                    len(resp.content),
                    url,
                    auth_name,
                )
                return path

        raise HikvisionCaptureError(last_err or "抓图失败：未知错误")


def build_camera_client(cfg: Dict[str, Any], root_dir: Path) -> HikvisionIsapiClient:
    from .config import resolve_project_path

    cam = cfg.get("camera", {})
    cap = cfg.get("capture", {})
    capture_dir = resolve_project_path(cap.get("dir", "data/captures"))
    return HikvisionIsapiClient(
        host=cam.get("host", "127.0.0.1"),
        port=int(cam.get("port", 80)),
        user=cam.get("user", "admin"),
        password=cam.get("password", ""),
        channel=str(cam.get("channel", "101")),
        timeout_sec=float(cam.get("timeout_sec", 15)),
        capture_dir=str(capture_dir),
    )
