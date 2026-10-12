# -*- coding: utf-8 -*-
"""
PI 客户端：ctypes 调用 piapi.dll。

对照 yingwei-realdb:
  - PIConfig.connect / piut_setservernode / piut_login
  - PIAPIUtils.findPoint / pipt_findpoint
  - PIAPIUtils.putVal / piar_putvalue
  - PIAPIUtils.getTagsRealCurves / piar_interpvalues
"""

from __future__ import annotations

import logging
import os
import threading
from ctypes import (
    CDLL,
    POINTER,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_longlong,
)
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ROOT_DIR

logger = logging.getLogger(__name__)


class PiClientError(RuntimeError):
    pass


def resolve_pi_dll_dir(dll_dir: str) -> Path:
    """相对路径按项目根解析，便于把 piapi.dll 放进仓库随包部署。"""
    p = Path(dll_dir)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p.resolve()


class PiClient:
    """线程内串行访问 piapi.dll（与 Java synchronizedLibrary 类似）。"""

    def __init__(
        self,
        dll_dir: str,
        server: str,
        user: str,
        password: str = "",
        dll_name: str = "piapi.dll",
    ):
        self.server = server
        self.user = user
        self.password = "" if password in ("-1", None) else password
        self._lock = threading.RLock()
        self._dll: Optional[CDLL] = None
        self._dll_dir = resolve_pi_dll_dir(dll_dir)
        self._dll_path = self._dll_dir / dll_name
        self._point_cache: Dict[str, int] = {}
        self._connected = False

    def _load(self) -> CDLL:
        if self._dll is not None:
            return self._dll
        if not self._dll_path.is_file():
            raise PiClientError(f"找不到 piapi.dll: {self._dll_path}")
        # Windows：把 DLL 所在目录加入搜索路径，便于同目录依赖被找到
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(self._dll_dir))
        dll = CDLL(str(self._dll_path))
        # 连接
        dll.piut_isconnected.restype = c_int
        dll.piut_setservernode.argtypes = [c_char_p]
        dll.piut_setservernode.restype = c_int
        dll.piut_login.argtypes = [c_char_p, c_char_p, POINTER(c_int)]
        dll.piut_login.restype = c_int
        # 测点
        dll.pipt_findpoint.argtypes = [c_char_p, POINTER(c_int)]
        dll.pipt_findpoint.restype = c_int
        # 写入（对照 PiApiMethod.piar_putvalue: Long timedate）
        dll.piar_putvalue.argtypes = [c_int, c_float, c_int, c_longlong, c_int]
        dll.piar_putvalue.restype = c_int
        # 快照
        dll.pisn_getsnapshot.argtypes = [
            c_int, POINTER(c_float), POINTER(c_int), POINTER(c_int)
        ]
        dll.pisn_getsnapshot.restype = c_int
        # 插值曲线
        dll.piar_interpvalues.argtypes = [
            c_int, POINTER(c_int), POINTER(c_int), POINTER(c_float), POINTER(c_int)
        ]
        dll.piar_interpvalues.restype = c_int
        self._dll = dll
        return dll

    def connect(self) -> None:
        with self._lock:
            dll = self._load()
            if dll.piut_isconnected() != 0:
                self._connected = True
                return
            rc = dll.piut_setservernode(self.server.encode("ascii"))
            if rc != 0:
                raise PiClientError(f"piut_setservernode 失败, code={rc}, server={self.server}")
            level = c_int(0)
            rc = dll.piut_login(
                self.user.encode("ascii"),
                self.password.encode("ascii"),
                byref(level),
            )
            if rc != 0:
                raise PiClientError(f"piut_login 失败, code={rc}, user={self.user}")
            self._connected = True
            logger.info("PI 连接成功: %s", self.server)

    def ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def pi_timestamp(self, dt: datetime) -> int:
        """本地墙钟 datetime → PI 时间戳（epoch 秒，不做时区偏移）。"""
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return int((dt - datetime(1970, 1, 1)).total_seconds())

    def from_pi_timestamp(self, pi_ts: int) -> datetime:
        """PI 时间戳 → 本地墙钟 datetime。"""
        return datetime(1970, 1, 1) + timedelta(seconds=int(pi_ts))

    def find_point(self, tag: str) -> int:
        with self._lock:
            self.ensure_connected()
            if tag in self._point_cache:
                return self._point_cache[tag]
            dll = self._load()
            pt = c_int(0)
            rc = dll.pipt_findpoint(tag.encode("ascii"), byref(pt))
            if rc != 0 or pt.value <= 0:
                raise PiClientError(f"测点不存在或查找失败: {tag}, code={rc}")
            self._point_cache[tag] = pt.value
            return pt.value

    def put_value(
        self,
        tag: str,
        value: float,
        when: Optional[datetime] = None,
        *,
        verify: bool = True,
        verify_tol: float = 1e-3,
    ) -> int:
        """
        对照 PIAPIUtils.putVal → piar_putvalue。
        返回 PI API 返回码：0 为 API 成功。
        verify=True 时再按写入时刻回读校验；早于建点等「API 成功但未入库」会返回自定义码 -9001。
        """
        with self._lock:
            self.ensure_connected()
            pt = self.find_point(tag)
            dt = when or datetime.now()
            ts = self.pi_timestamp(dt)
            dll = self._load()
            rc = dll.piar_putvalue(pt, c_float(value), 0, c_longlong(ts), 0)
            if rc != 0:
                logger.warning(
                    "piar_putvalue 失败 tag=%s value=%s when=%s code=%s",
                    tag,
                    value,
                    dt.strftime("%Y-%m-%d %H:%M:%S"),
                    rc,
                )
                return int(rc)
            if not verify:
                return 0
            ok, detail = self._verify_archived_value(
                tag, float(value), dt, tol=verify_tol
            )
            if ok:
                return 0
            logger.warning(
                "piar_putvalue 返回成功但回读未确认 tag=%s value=%s when=%s detail=%s",
                tag,
                value,
                dt.strftime("%Y-%m-%d %H:%M:%S"),
                detail,
            )
            # 自定义：API 成功但历史库未见有效值（常见于早于建点时间）
            return -9001

    def _verify_archived_value(
        self,
        tag: str,
        expected: float,
        when: datetime,
        *,
        tol: float = 1e-3,
    ) -> Tuple[bool, str]:
        """
        写后按时刻附近插值回读；istat!=0 或数值偏差过大视为未入库。
        """
        try:
            # 短窗插值，取最接近写入时刻的点
            start = when - timedelta(seconds=1)
            end = when + timedelta(seconds=1)
            series = self.interp_values(tag, start, end, count=3)
        except Exception as e:
            return False, f"回读失败: {e}"
        if not series:
            return False, "回读无点"
        target_ts = self.pi_timestamp(when)
        best = None
        best_dist = None
        for item in series:
            try:
                t = datetime.strptime(str(item["t"]), "%Y-%m-%d %H:%M:%S")
                dist = abs(self.pi_timestamp(t) - target_ts)
                if best_dist is None or dist < best_dist:
                    best = item
                    best_dist = dist
            except Exception:
                continue
        if best is None:
            return False, "回读无有效时刻"
        istat = int(best.get("s") or 0)
        got = float(best.get("v") or 0.0)
        if istat != 0:
            return False, f"回读状态码 istat={istat}（非好值，可能早于建点/无档案）"
        # 允许小浮点误差；0 附近用绝对容差
        abs_tol = max(float(tol), abs(expected) * 1e-4)
        if abs(got - expected) > abs_tol:
            return (
                False,
                f"回读值不符 got={got} expected={expected} tol={abs_tol}",
            )
        return True, "ok"

    def get_snapshot(self, tag: str) -> Tuple[float, int, datetime]:
        with self._lock:
            self.ensure_connected()
            pt = self.find_point(tag)
            dll = self._load()
            rval = c_float(0)
            istat = c_int(0)
            timedate = c_int(0)
            rc = dll.pisn_getsnapshot(pt, byref(rval), byref(istat), byref(timedate))
            if rc != 0:
                raise PiClientError(f"pisn_getsnapshot 失败: {tag}, code={rc}")
            return float(rval.value), int(istat.value), self.from_pi_timestamp(timedate.value)

    def interp_values(
        self,
        tag: str,
        start: datetime,
        end: datetime,
        count: int = 96,
    ) -> List[Dict[str, object]]:
        """对照 piar_interpvalues：首尾给定时间，返回插值序列。"""
        if count < 2:
            count = 2
        with self._lock:
            self.ensure_connected()
            pt = self.find_point(tag)
            dll = self._load()
            r_count = c_int(count)
            times = (c_int * count)()
            rvals = (c_float * count)()
            istats = (c_int * count)()
            times[0] = self.pi_timestamp(start)
            times[count - 1] = self.pi_timestamp(end)
            rc = dll.piar_interpvalues(pt, byref(r_count), times, rvals, istats)
            if rc != 0:
                raise PiClientError(f"piar_interpvalues 失败: {tag}, code={rc}")
            n = r_count.value
            out: List[Dict[str, object]] = []
            for i in range(n):
                out.append(
                    {
                        "t": self.from_pi_timestamp(times[i]).strftime("%Y-%m-%d %H:%M:%S"),
                        "v": float(rvals[i]),
                        "s": int(istats[i]),
                    }
                )
            return out


_client: Optional[PiClient] = None


def get_pi_client(cfg: dict) -> PiClient:
    global _client
    pi = cfg.get("pi", {})
    if _client is None:
        _client = PiClient(
            dll_dir=pi.get("dll_dir", "D:/piapi"),
            dll_name=pi.get("dll_name", "piapi.dll"),
            server=pi.get("server", "127.0.0.1"),
            user=pi.get("user", "piadmin"),
            password=pi.get("password", "") or "",
        )
    return _client
