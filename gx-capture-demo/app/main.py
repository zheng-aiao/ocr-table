# -*- coding: utf-8 -*-
"""珙县 Demo：海康抓图 + OCR + PI + 执行履历前端。"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# 尽早限制 OpenMP/BLAS，避免 RapidOCR 默认占满 CPU（识别时浏览器极慢）
for _k, _v in (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"),
    ("NUMEXPR_NUM_THREADS", "2"),
):
    os.environ.setdefault(_k, _v)

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    DATA_DIR,
    ROOT_DIR,
    get_capture_tag,
    get_config,
    get_ocr_onnx_threads,
    load_config,
    update_capture_mode,
    update_capture_tag,
)
from .excel_export import export_ocr_excel
from .pipeline import capture_only, curve, ocr_only, run_pipeline, write_points_to_pi
from .pipeline_dual import (
    get_capture_mode,
    get_dual_defs,
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

STATIC_DIR = ROOT_DIR / "static"
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


@app.on_event("startup")
def _startup() -> None:
    from .config import resolve_ocr_artifacts_root
    from .scheduler import start_scheduler

    load_config()
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
    image_path: str = Field(..., description="整图路径（含左右两个标签页）")
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
    """双屏分步 OCR：裁左右 → 独立识别（不写 PI）。左右日期/测点前缀互不影响。"""
    path = Path(body.image_path)
    if not path.is_file():
        raise HTTPException(400, f"图片不存在: {body.image_path}")
    try:
        return split_and_ocr(
            str(path),
            base_date=_parse_base_date(body.base_date),
            left_date=_parse_base_date(body.left_date),
            right_date=_parse_base_date(body.right_date),
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
    file: UploadFile = File(...),
    left_date: Optional[str] = None,
    right_date: Optional[str] = None,
    base_date: Optional[str] = None,
):
    """本机双屏图上传：临时文件左右独立 OCR（不写 captures；过程图仍走 ocr_runs）。"""
    import tempfile

    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        raise HTTPException(400, f"不支持的图片格式: {suffix}")
    tmp_path: Optional[Path] = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "空文件")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="ocr_dual_upload_") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        upload_name = file.filename or "upload"
        result = split_and_ocr(
            str(tmp_path),
            base_date=_parse_base_date(base_date),
            left_date=_parse_base_date(left_date),
            right_date=_parse_base_date(right_date),
            write_pi=False,
            export_excel=True,
            trigger="debug",
            persist=True,
            upload_name=upload_name,
        )
        # 临时文件马上删除：预览用 ocr_runs 的 00_source；展示名用原始上传文件名
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
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("failed to remove temp dual upload: %s", tmp_path)


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
        result = ocr_only(
            str(tmp_path), trigger="debug", upload_name=upload_name
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
    prefix = get_capture_tag(cfg)
    dual = get_dual_defs(cfg)
    return {
        "ok": True,
        "tag": prefix,
        "mode": get_capture_mode(cfg),
        "single": {"tag": prefix},
        "dual": {
            side: {"tag": sc.get("tag"), "title": sc.get("title"), "id": sc.get("id")}
            for side, sc in dual.items()
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
    """单屏 / 双屏配置（capture.mode 对应 capture.single / capture.dual）。"""
    from .config import get_single_cfg

    cfg = get_config()
    mode = get_capture_mode(cfg)
    single = get_single_cfg(cfg)
    dual = get_dual_defs(cfg)
    return {
        "ok": True,
        "mode": mode,
        "tag": get_capture_tag(cfg),
        "single": {
            "tag": single.get("tag") or get_capture_tag(cfg),
            "crop": single.get("crop") or "",
            "table_crop": single.get("table_crop") or "",
        },
        "dual": {
            side: {
                "id": sc.get("id"),
                "title": sc.get("title"),
                "crop": sc.get("crop"),
                "table_crop": sc.get("table_crop"),
                "tag": sc.get("tag"),
            }
            for side, sc in dual.items()
        },
    }


class ViewModeRequest(BaseModel):
    mode: str = Field(..., description="single | dual")


@app.put("/api/config/view/mode")
def api_put_view_mode(body: ViewModeRequest):
    """切换首页单屏/双屏模式，写入 capture.mode。"""
    try:
        return update_capture_mode(body.mode)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("update capture mode failed")
        raise HTTPException(500, str(e)) from e


@app.post("/api/run")
def api_run(body: RunRequest = RunRequest()):
    """单屏单页流水线（保持原行为，不受 capture.mode 影响）。"""
    try:
        return run_pipeline(
            base_date=_parse_base_date(body.base_date),
            trigger="manual",
        )
    except Exception as e:
        logger.exception("pipeline failed")
        raise HTTPException(500, str(e)) from e


@app.post("/api/run/dual")
def api_run_dual(body: DualRunRequest = DualRunRequest()):
    """双屏左右：抓图 → 左右独立 OCR（各自日期/测点前缀）→ 按点内 tag 写 PI。"""
    try:
        return run_pipeline_dual(
            base_date=_parse_base_date(body.base_date),
            left_date=_parse_base_date(body.left_date),
            right_date=_parse_base_date(body.right_date),
            trigger="manual",
        )
    except Exception as e:
        logger.exception("dual pipeline failed")
        raise HTTPException(500, str(e)) from e


@app.get("/api/curve")
def api_curve(
    start: str = Query(..., description="yyyy-MM-dd HH:mm:ss"),
    end: str = Query(..., description="yyyy-MM-dd HH:mm:ss"),
    tag: Optional[str] = None,
    count: Optional[int] = None,
):
    try:
        return curve(start=start, end=end, tag=tag, count=count)
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
        interval = int(sch.get("interval_minutes", 15) or 15)
    except (TypeError, ValueError):
        interval = 15
    return {
        "ok": True,
        "config": {
            "enabled": bool(sch.get("enabled", False)),
            "interval_minutes": max(1, interval),
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
