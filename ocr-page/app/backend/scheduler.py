# -*- coding: utf-8 -*-
"""定时处置：按顶层 schedule 周期执行 抓图 → OCR → 入 PI。

规则：
- 认 enabled / interval_sec / capture / ocr / pi
- 开启后立刻跑第一轮；本轮结束后再等 interval_sec 秒开始下一周期
- 周期串行：上一轮未结束不会开下一轮
- 阶段开关按序截断：任一为 false 则该阶段及之后不执行
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .config import get_config, get_schedule_cfg, load_config
from .pipeline_dual import get_capture_mode, run_pipeline_dual

logger = logging.getLogger(__name__)

_busy = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_state: Dict[str, Any] = {
    "running": False,
    "enabled": False,
    "interval_sec": 10,
    "capture": True,
    "ocr": True,
    "pi": True,
    "busy": False,
    "last_run_at": None,
    "last_ok": None,
    "last_error": None,
    "last_skipped": None,
    "last_mode": None,
    "last_stages": None,
    "next_run_at": None,
}


def get_schedule_state() -> Dict[str, Any]:
    return dict(_state)


def _schedule_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return get_schedule_cfg(cfg)


def _sync_state_from_cfg(sch: Dict[str, Any]) -> None:
    _state["enabled"] = sch["enabled"]
    _state["interval_sec"] = sch["interval_sec"]
    _state["capture"] = sch["capture"]
    _state["ocr"] = sch["ocr"]
    _state["pi"] = sch["pi"]


def _execute_pipeline() -> Dict[str, Any]:
    """实际执行（按阶段开关截断；调用方须已持有 _busy）。"""
    try:
        load_config()
    except Exception:
        logger.exception("定时任务热加载配置失败，沿用内存配置")
    cfg = get_config()
    sch = _schedule_cfg(cfg)
    _sync_state_from_cfg(sch)
    mode = get_capture_mode(cfg)
    _state["last_mode"] = mode
    _state["last_run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _state["busy"] = True
    _state["last_skipped"] = None

    do_capture = bool(sch["do_capture"])
    do_ocr = bool(sch["do_ocr"])
    do_pi = bool(sch["do_pi"])
    stages = {"capture": do_capture, "ocr": do_ocr, "pi": do_pi}
    _state["last_stages"] = stages

    logger.info(
        "定时处置开始 mode=%s stages=capture:%s ocr:%s pi:%s",
        mode,
        do_capture,
        do_ocr,
        do_pi,
    )
    t0 = time.perf_counter()
    try:
        if not do_capture:
            reason = "schedule.capture=false，本轮跳过抓拍/OCR/写PI"
            _state["last_ok"] = True
            _state["last_error"] = None
            _state["last_skipped"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(reason)
            return {
                "ok": True,
                "skipped": True,
                "reason": "schedule_capture_disabled",
                "message": reason,
                "stages": stages,
            }

        # 本分支固定双摄流水线
        result = run_pipeline_dual(
            trigger="schedule", do_ocr=do_ocr, do_pi=do_pi
        )
        result = dict(result)
        result.setdefault("stages", stages)
        ok = bool(result.get("ok"))
        _state["last_ok"] = ok
        if ok:
            _state["last_error"] = None
        else:
            _state["last_error"] = (
                result.get("message")
                or result.get("reason")
                or "本轮识别未通过 min_points 或写 PI 失败"
            )
        total_ms = int((result.get("timing") or {}).get("total_ms") or 0)
        if total_ms <= 0:
            total_ms = max(0, int(round((time.perf_counter() - t0) * 1000)))
        logger.info(
            "定时处置结束 ms=%s ok=%s mode=%s stages=%s left_ok=%s right_ok=%s reason=%s",
            total_ms,
            ok,
            mode,
            stages,
            result.get("left_ok"),
            result.get("right_ok"),
            result.get("reason"),
        )
        return result
    except Exception as e:
        _state["last_ok"] = False
        _state["last_error"] = str(e)
        logger.exception("定时处置失败")
        return {"ok": False, "error": str(e), "stages": stages}
    finally:
        _state["busy"] = False


def run_scheduled_once(*, wait_if_busy: bool = False) -> Dict[str, Any]:
    """
    尝试执行一轮。
    默认非阻塞：上一轮未完成则跳过。
    wait_if_busy=True 时等待上一轮结束再跑（供手动接口可选）。
    """
    acquired = _busy.acquire(blocking=wait_if_busy)
    if not acquired:
        reason = "上一次定时任务仍在执行，本次跳过"
        _state["last_skipped"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(reason)
        return {"ok": False, "skipped": True, "reason": reason}
    try:
        return _execute_pipeline()
    finally:
        _busy.release()


def _loop() -> None:
    """串行周期：跑完一轮，再等待 interval_sec，然后开始下一轮。"""
    while not _stop.is_set():
        try:
            try:
                load_config()
            except Exception:
                logger.exception("定时调度热加载配置失败")
            sch = _schedule_cfg()
            _sync_state_from_cfg(sch)

            if not sch["enabled"]:
                _state["next_run_at"] = None
                _stop.wait(5.0)
                continue

            logger.info(
                "定时周期开始（本轮结束后等待 %d 秒再开下一轮）",
                sch["interval_sec"],
            )
            _state["next_run_at"] = None
            run_scheduled_once(wait_if_busy=True)
            if _stop.is_set():
                break

            try:
                load_config()
            except Exception:
                logger.exception("定时调度热加载配置失败")
            sch = _schedule_cfg()
            _sync_state_from_cfg(sch)
            if not sch["enabled"]:
                _state["next_run_at"] = None
                continue

            wait_sec = float(sch["interval_sec"])
            next_at = datetime.now() + timedelta(seconds=wait_sec)
            _state["next_run_at"] = next_at.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                "本轮已结束，%s 秒后开始下一周期 next=%s",
                sch["interval_sec"],
                _state["next_run_at"],
            )
            if _stop.wait(wait_sec):
                break
        except Exception:
            logger.exception("定时调度循环异常，5s 后重试")
            _stop.wait(5.0)


def start_scheduler() -> None:
    """启动后台定时线程（幂等）。"""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    sch = _schedule_cfg()
    _sync_state_from_cfg(sch)
    _state["running"] = True
    _thread = threading.Thread(target=_loop, name="capture-scheduler", daemon=True)
    _thread.start()
    logger.info(
        "定时调度已启动 enabled=%s interval_sec=%s capture=%s ocr=%s pi=%s",
        sch["enabled"],
        sch["interval_sec"],
        sch["capture"],
        sch["ocr"],
        sch["pi"],
    )


def stop_scheduler() -> None:
    global _thread
    _stop.set()
    _state["running"] = False
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _thread = None
