# -*- coding: utf-8 -*-
"""珙县 Demo：海康抓图 + OCR + PI + 执行履历前端。"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 尽早限制 OpenMP/BLAS，避免 RapidOCR 默认占满 CPU（识别时浏览器极慢）
for _k, _v in (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"),
    ("NUMEXPR_NUM_THREADS", "2"),
):
    os.environ.setdefault(_k, _v)

# 日志经 bat >> 重定向到 ocr.log：中文 Windows 默认 GBK，与 UTF-8 头混写会乱码
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import (
    DATA_DIR,
    ROOT_DIR,
    get_config,
    get_ocr_onnx_threads,
    load_config,
    update_capture_mode,
    update_capture_tag,
)
from .excel_export import export_ocr_excel
from .pipeline import capture_only, curve, ocr_only, run_pipeline, write_points_to_pi
from .pipeline_dual import (
    get_camera_defs,
    get_capture_mode,
    get_dual_defs,
    ocr_side_images,
    run_pipeline_dual,
    split_and_ocr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gx-capture-demo")


def _parse_base_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    return datetime.strptime(raw.strip() + " 00:00:00", "%Y-%m-%d %H:%M:%S")


app = FastAPI(title="珙县抓拍OCR-PI Demo", version="0.2.0")

STATIC_DIR = ROOT_DIR / "app" / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DATA_DIR.mkdir(parents=True, exist_ok=True)

CAPTURES_DIR = DATA_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
# /data/* 为新布局；根下 /captures|/ocr_runs|/exports 兼容旧履历 URL
app.mount("/data/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="data_captures")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")

OCR_RUNS_DIR = DATA_DIR / "ocr_runs"
OCR_RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data/ocr_runs", StaticFiles(directory=str(OCR_RUNS_DIR)), name="data_ocr_runs")
app.mount("/ocr_runs", StaticFiles(directory=str(OCR_RUNS_DIR)), name="ocr_runs")

EXPORTS_DIR = DATA_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="data_exports")
app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _lower_process_priority() -> None:
    """识别与网页同机时，降低本进程优先级，避免 OCR 抢光 CPU 导致浏览器卡死。"""
    try:
        if sys.platform == "win32":
            import ctypes

            # BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                0x00004000,
            )
            logger.info("process priority set to BELOW_NORMAL (UI stays responsive during OCR)")
        else:
            os.nice(5)
            logger.info("process nice raised by 5 (UI stays responsive during OCR)")
    except Exception as e:
        logger.warning("failed to lower process priority: %s", e)


@app.on_event("startup")
def _startup() -> None:
    from .config import resolve_ocr_artifacts_root
    from .scheduler import start_scheduler

    load_config()
    _lower_process_priority()
    n = get_ocr_onnx_threads()
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = str(n)
    arts = resolve_ocr_artifacts_root()
    arts.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "config loaded from %s ; ocr_runs=%s ; onnx_threads=%s",
        ROOT_DIR / "config.yaml",
        arts,
        n,
    )
    start_scheduler()


@app.on_event("shutdown")
def _shutdown() -> None:
    from .scheduler import stop_scheduler

    stop_scheduler()


def _page(name: str) -> FileResponse:
    path = STATIC_DIR / name
    if not path.is_file():
        raise HTTPException(404, f"static/{name} 不存在")
    return FileResponse(path)


@app.get("/")
def home_page():
    """执行履历列表（新首页）。"""
    return _page("home.html")


@app.get("/runs/{run_id}")
def run_detail_page(run_id: str):
    """执行详情页（前端按 path 拉 /api/runs/{id}）。"""
    return _page("run.html")


@app.get("/debug")
def debug_page():
    """分步联调页。"""
    return _page("debug.html")


@app.get("/single")
def single_page():
    """单屏一键 Demo。"""
    return _page("demo.html")


@app.get("/dual")
def dual_page():
    """双屏一键 Demo。"""
    return _page("split.html")


@app.get("/demo")
def demo_redirect():
    return RedirectResponse(url="/single", status_code=302)


@app.get("/split")
def split_redirect():
    return RedirectResponse(url="/dual", status_code=302)


@app.get("/api/runs")
def api_list_runs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    mode: Optional[str] = Query(None, description="single|dual"),
    ok: Optional[bool] = Query(None, description="兼容：true=三步成功"),
    result: Optional[str] = Query(
        None,
        description="结果: ok|partial_ok|capture_fail|ocr_fail|pi_fail|pending_pi",
    ),
    day: Optional[str] = Query(None, description="yyyy-MM-dd 或 yyyyMMdd"),
    pi: Optional[str] = Query(
        None, description="写PI: ok|unwritten|fail"
    ),
):
    from .run_store import list_runs

    return list_runs(
        limit=limit,
        offset=offset,
        mode=mode,
        ok=ok,
        result=result,
        day=day,
        pi=pi,
    )


class RewritePiRequest(BaseModel):
    mode: Optional[str] = Field(None, description="可选：只补录 single|dual")
    day: Optional[str] = Field(None, description="可选：只补录当天 yyyy-MM-dd")
    limit: int = Field(200, ge=1, le=500, description="最多补录条数")


@app.post("/api/runs/pi/rewrite")
def api_rewrite_pending_pi(body: RewritePiRequest = RewritePiRequest()):
    """一键补录：OCR 已有点数但未写成功 PI 的履历。"""
    from .run_store import rewrite_pending_pi

    try:
        return rewrite_pending_pi(mode=body.mode, day=body.day, limit=body.limit)
    except Exception as e:
        logger.exception("batch rewrite pi failed")
        raise HTTPException(500, str(e)) from e


@app.post("/api/runs/{run_id}/pi/rewrite")
def api_rewrite_run_pi(run_id: str):
    """单条履历补写 PI。"""
    from .run_store import rewrite_run_pi

    try:
        result = rewrite_run_pi(run_id)
    except Exception as e:
        logger.exception("rewrite pi failed: %s", run_id)
        raise HTTPException(500, str(e)) from e
    if result.get("reason") == "not_found":
        raise HTTPException(404, result.get("message") or "执行记录不存在")
    return result


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str):
    from .run_store import get_run

    data = get_run(run_id)
    if not data:
        raise HTTPException(404, f"执行记录不存在: {run_id}")
    return {"ok": True, "run": data}


@app.post("/api/capture")
def api_capture():
    try:
        return capture_only()
    except Exception as e:
        logger.exception("capture failed")
        raise HTTPException(500, str(e)) from e


class OcrRequest(BaseModel):
    image_path: str = Field(..., description="本地图片绝对/相对路径")
    base_date: Optional[str] = Field(None, description="业务日期 yyyy-MM-dd（单屏）")


@app.post("/api/ocr")
def api_ocr(body: OcrRequest):
    path = Path(body.image_path)
    if not path.is_file():
        raise HTTPException(400, f"图片不存在: {body.image_path}")
    try:
        return ocr_only(
            str(path),
            base_date=_parse_base_date(body.base_date),
            trigger="debug",
        )
    except Exception as e:
        logger.exception("ocr failed")
        raise HTTPException(500, str(e)) from e


class DualOcrRequest(BaseModel):
    image_path: Optional[str] = Field(
        None, description="兼容：单图路径（左右共用同一张，调试用）"
    )
    left_image: Optional[str] = Field(None, description="左摄像头整屏图路径")
    right_image: Optional[str] = Field(None, description="右摄像头整屏图路径")
    left_date: Optional[str] = Field(
        None, description="左机组业务日期回退 yyyy-MM-dd（日期 OCR 失败时使用）"
    )
    right_date: Optional[str] = Field(
        None, description="右机组业务日期回退 yyyy-MM-dd（日期 OCR 失败时使用）"
    )
    base_date: Optional[str] = Field(
        None, description="左右未单独指定时的公共业务日期回退"
    )


@app.post("/api/ocr/dual")
def api_ocr_dual(body: DualOcrRequest):
    """双摄 OCR：左右各一张整屏图独立识别（不写 PI）。也可只传 image_path 调试。"""
    left = (body.left_image or body.image_path or "").strip() or None
    right = (body.right_image or body.image_path or "").strip() or None
    if not left and not right:
        raise HTTPException(400, "请提供 left_image/right_image 或 image_path")
    for label, p in (("left", left), ("right", right)):
        if p and not Path(p).is_file():
            raise HTTPException(400, f"{label} 图片不存在: {p}")
    try:
        if body.image_path and not body.left_image and not body.right_image:
            return split_and_ocr(
                str(body.image_path),
                write_pi=False,
                export_excel=True,
                trigger="debug",
                persist=True,
            )
        return ocr_side_images(
            left,
            right,
            write_pi=False,
            export_excel=True,
            trigger="debug",
            persist=True,
        )
    except Exception as e:
        logger.exception("dual ocr failed")
        raise HTTPException(500, str(e)) from e


@app.post("/api/ocr/dual/upload")
async def api_ocr_dual_upload(
    file: Optional[UploadFile] = File(None),
    left_file: Optional[UploadFile] = File(None),
    right_file: Optional[UploadFile] = File(None),
    left_date: Optional[str] = None,
    right_date: Optional[str] = None,
    base_date: Optional[str] = None,
):
    """上传识别：可传 left_file+right_file 两张整屏图；或单 file（左右共用，调试）。"""
    import tempfile

    del left_date, right_date, base_date
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    tmp_paths: List[Path] = []

    async def _save_one(up: UploadFile, prefix: str) -> Path:
        suffix = Path(up.filename or "upload.jpg").suffix.lower() or ".jpg"
        if suffix not in allowed:
            raise HTTPException(400, f"不支持的图片格式: {suffix}")
        content = await up.read()
        if not content:
            raise HTTPException(400, f"空文件: {up.filename}")
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix=prefix
        ) as tmp:
            tmp.write(content)
            path = Path(tmp.name)
        tmp_paths.append(path)
        return path

    try:
        if left_file is not None or right_file is not None:
            left_path = (
                await _save_one(left_file, "ocr_dual_L_") if left_file else None
            )
            right_path = (
                await _save_one(right_file, "ocr_dual_R_") if right_file else None
            )
            result = await run_in_threadpool(
                ocr_side_images,
                str(left_path) if left_path else None,
                str(right_path) if right_path else None,
                write_pi=False,
                export_excel=True,
                trigger="debug",
                persist=True,
            )
            result["uploaded"] = True
            return result

        if file is None:
            raise HTTPException(400, "请上传 left_file/right_file 或 file")
        tmp_path = await _save_one(file, "ocr_dual_upload_")
        upload_name = file.filename or "upload"
        result = await run_in_threadpool(
            split_and_ocr,
            str(tmp_path),
            write_pi=False,
            export_excel=True,
            trigger="debug",
            persist=True,
            upload_name=upload_name,
        )
        result["upload_name"] = upload_name
        result["image_path"] = result.get("source_copy") or upload_name
        result["uploaded"] = True
        result["persisted"] = False
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("dual ocr upload failed")
        raise HTTPException(500, str(e)) from e
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                logger.warning("failed to remove temp dual upload: %s", p)


@app.post("/api/ocr/upload")
async def api_ocr_upload(file: UploadFile = File(...)):
    """上传本机图片并 OCR：仅用临时文件，不写入 captures。"""
    import tempfile

    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        raise HTTPException(400, f"不支持的图片格式: {suffix}")
    tmp_path: Optional[Path] = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "空文件")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="ocr_upload_") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        upload_name = file.filename or "upload"
        # 同步 OCR 必须进线程池，否则会堵死事件循环，历史页 /api/runs 全部无响应
        result = await run_in_threadpool(
            ocr_only,
            str(tmp_path),
            trigger="debug",
            upload_name=upload_name,
        )
        result["upload_name"] = upload_name
        result["image_path"] = result.get("source_copy") or upload_name
        result["uploaded"] = True
        result["persisted"] = False
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ocr upload failed")
        raise HTTPException(500, str(e)) from e
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("failed to remove temp upload: %s", tmp_path)


class WriteRequest(BaseModel):
    points: list
    tag: Optional[str] = None
    run_id: Optional[str] = Field(
        None, description="联调 OCR 落盘履历 id；写入后回写执行结果状态"
    )


@app.post("/api/pi/write")
def api_pi_write(body: WriteRequest):
    try:
        result = write_points_to_pi(body.points, tag=body.tag)
    except Exception as e:
        logger.exception("pi write failed")
        if body.run_id:
            from .run_store import apply_pi_write_result

            apply_pi_write_result(
                body.run_id,
                {"written": 0, "total": len(body.points or [])},
                error=str(e),
            )
        raise HTTPException(500, str(e)) from e
    if body.run_id:
        from .run_store import apply_pi_write_result

        synced = apply_pi_write_result(body.run_id, result)
        result["run_id"] = body.run_id
        result["run"] = {
            "id": body.run_id,
            "pi_status": synced.get("pi_status"),
            "result_status": synced.get("result_status"),
            "ok": synced.get("ok"),
        }
        if synced.get("reason") == "not_found":
            result["run_sync"] = "not_found"
        else:
            result["run_sync"] = "ok"
    return result


@app.get("/api/config/pi")
def api_get_pi_config():
    cfg = get_config()
    pi = cfg.get("pi", {}) or {}
    cams = get_camera_defs(cfg)
    return {
        "ok": True,
        "capture_mode": "double_camera",
        "cameras": {
            side: {
                "tag": sc.get("tag"),
                "host": sc.get("host"),
                "type": sc.get("type"),
            }
            for side, sc in cams.items()
        },
        "row_tags": [],
        "server": pi.get("server", ""),
    }


class PiTagRequest(BaseModel):
    tag: str = Field(..., description="单屏测点前缀（写入 capture.single.tag）")


@app.put("/api/config/pi/tag")
def api_put_pi_tag(body: PiTagRequest):
    try:
        return update_capture_tag(body.tag)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("update capture tag failed")
        raise HTTPException(500, str(e)) from e


class DemoExcelRequest(BaseModel):
    name: str = "demo"
    headers: list
    rows: list
    row_labels: Optional[list] = None


@app.post("/api/demo/export-excel")
def api_demo_export_excel(body: DemoExcelRequest):
    try:
        path = export_ocr_excel(
            headers=[str(h) for h in body.headers],
            rows=[[str(c) for c in row] for row in body.rows],
            image_path=body.name,
            row_labels=[str(x) for x in body.row_labels] if body.row_labels else None,
        )
        if path is None:
            return {"ok": False, "skipped": True, "reason": "excel.enabled=false"}
        return {"ok": True, "excel_path": path}
    except Exception as e:
        logger.exception("demo excel export failed")
        raise HTTPException(500, str(e)) from e


class RunRequest(BaseModel):
    base_date: Optional[str] = Field(
        None, description="业务日期回退 yyyy-MM-dd（日期 OCR 失败时与表头时刻拼接）"
    )


class DualRunRequest(BaseModel):
    left_date: Optional[str] = Field(
        None, description="左机组业务日期回退 yyyy-MM-dd（日期 OCR 失败时使用）"
    )
    right_date: Optional[str] = Field(
        None, description="右机组业务日期回退 yyyy-MM-dd（日期 OCR 失败时使用）"
    )
    base_date: Optional[str] = Field(
        None, description="左右未单独指定时的公共业务日期回退"
    )


@app.get("/api/config/view")
def api_get_view_config():
    """双摄配置：cameras + capture.double 裁剪（date_crop=搜索带）。"""
    cfg = get_config()
    double = get_dual_defs(cfg)
    cams = get_camera_defs(cfg)
    return {
        "ok": True,
        "mode": "double",
        "capture_mode": "double_camera",
        "cameras": {
            side: {
                "type": sc.get("type"),
                "host": sc.get("host"),
                "tag": sc.get("tag"),
            }
            for side, sc in cams.items()
        },
        "double": {
            side: {
                "id": sc.get("id"),
                "title": sc.get("title"),
                "table_crop": sc.get("table_crop"),
                "date_crop": sc.get("date_crop"),
                "tag": sc.get("tag"),
                "host": sc.get("host"),
                "type": sc.get("type"),
            }
            for side, sc in double.items()
        },
        # 兼容旧前端字段名
        "dual": {
            side: {
                "id": sc.get("id"),
                "title": sc.get("title"),
                "table_crop": sc.get("table_crop"),
                "date_crop": sc.get("date_crop"),
                "tag": sc.get("tag"),
                "host": sc.get("host"),
                "type": sc.get("type"),
            }
            for side, sc in double.items()
        },
    }


class ViewModeRequest(BaseModel):
    mode: str = Field(..., description="single | dual | double")


@app.put("/api/config/view/mode")
def api_put_view_mode(body: ViewModeRequest):
    """切换联调页画面模式，写入 capture.mode。"""
    try:
        return update_capture_mode(body.mode)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("update capture mode failed")
        raise HTTPException(500, str(e)) from e


@app.post("/api/run")
def api_run(body: RunRequest = RunRequest()):
    """兼容旧入口：本分支转发到双摄流水线。"""
    del body
    try:
        return run_pipeline_dual(trigger="manual")
    except Exception as e:
        logger.exception("pipeline failed")
        raise HTTPException(500, str(e)) from e


@app.post("/api/run/dual")
def api_run_dual(body: DualRunRequest = DualRunRequest()):
    """双摄像头：并行抓拍两路 → 各自 OCR（测点前缀 dual.*.tag）→ 写 PI。"""
    del body
    try:
        return run_pipeline_dual(trigger="manual")
    except Exception as e:
        logger.exception("dual pipeline failed")
        raise HTTPException(500, str(e)) from e


@app.get("/api/curve")
def api_curve(
    start: str = Query(..., description="yyyy-MM-dd HH:mm:ss"),
    end: str = Query(..., description="yyyy-MM-dd HH:mm:ss"),
    tag: Optional[str] = None,
):
    """读取测点在时间范围内的 PI 档案值，有多少返回多少。"""
    try:
        return curve(start=start, end=end, tag=tag)
    except Exception as e:
        logger.exception("curve failed")
        raise HTTPException(500, str(e)) from e


@app.get("/api/schedule")
def api_schedule_status():
    """定时处置状态（配置见 capture.schedule）。"""
    from .scheduler import get_schedule_state

    cfg = get_config()
    sch = (cfg.get("capture") or {}).get("schedule") or {}
    try:
        interval = int(sch.get("interval_sec", 10) or 0)
    except (TypeError, ValueError):
        interval = 10
    return {
        "ok": True,
        "config": {
            "enabled": bool(sch.get("enabled", False)),
            "interval_sec": max(0, interval),
        },
        "state": get_schedule_state(),
        "mode": get_capture_mode(cfg),
    }


@app.post("/api/schedule/run")
def api_schedule_run_now():
    """立刻执行一轮定时处置（抓图+OCR+入PI），不改变 enabled。"""
    from .scheduler import run_scheduled_once

    try:
        return run_scheduled_once()
    except Exception as e:
        logger.exception("manual schedule run failed")
        raise HTTPException(500, str(e)) from e


@app.get("/api/health")
def health():
    from .scheduler import get_schedule_state

    return {"ok": True, "schedule": get_schedule_state()}
