# -*- coding: utf-8 -*-
"""OCR bench: layout-aware single vs dual testing.

- dual: 按 config 分屏 0~0.56 / 0.56~1，两侧分别 OCR，与左右真值比对
- single: 整图 OCR（单屏模式），与有数据的那一侧真值比对
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "image-table-extract"))

from extract_table import TableImageExtractor  # noqa: E402

# 与 config.yaml 对齐
DUAL_SPLIT = 0.56
SINGLE_TABLE_CROP = (0.10, 0.52, 0.995, 0.90)
DUAL_LEFT_TABLE_CROP = (0.02, 0.52, 0.985, 0.90)
DUAL_RIGHT_TABLE_CROP = (0.0, 0.52, 1.0, 0.90)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _norm_cell(v: Any) -> str:
    s = "" if v is None else str(v).strip()
    s = s.replace(",", "").replace(" ", "").replace("−", "-").replace("?", "")
    if s in {"", "-", "--", "—", "null", "None"}:
        return ""
    return s


def _cells_equal(a: str, b: str, *, rel_tol: float = 0.015) -> bool:
    a = _norm_cell(a)
    b = _norm_cell(b)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        fa = float(a)
        fb = float(b)
        if fa == fb:
            return True
        denom = max(abs(fb), 1e-6)
        return abs(fa - fb) / denom <= rel_tol
    except ValueError:
        return False


def compare_tables(
    pred_headers: List[str],
    pred_rows: List[List[str]],
    gt: Dict[str, Any],
) -> Dict[str, Any]:
    gt_headers = [str(x) for x in (gt.get("headers") or [])]
    gt_rows = [[str(c) for c in r] for r in (gt.get("rows") or [])]
    usable_cols = []
    for i, h in enumerate(gt_headers):
        col_vals = [(_norm_cell(r[i]) if i < len(r) else "") for r in gt_rows]
        if any(v and v != "?" for v in col_vals) or h:
            usable_cols.append(i)
    if usable_cols:
        gt_headers = [gt_headers[i] for i in usable_cols if i < len(gt_headers)]
        gt_rows = [
            [(_norm_cell(r[i]) if i < len(r) else "") for i in usable_cols]
            for r in gt_rows
        ]

    if gt_headers and pred_headers:
        common = [h for h in gt_headers if h in pred_headers]
        if len(common) >= max(3, len(gt_headers) // 2):
            gt_idx = [gt_headers.index(h) for h in common]
            pr_idx = [pred_headers.index(h) for h in common]
            gt_rows = [[r[i] if i < len(r) else "" for i in gt_idx] for r in gt_rows]
            pred_rows = [[r[i] if i < len(r) else "" for i in pr_idx] for r in pred_rows]
            gt_headers = common
            pred_headers = common

    n_gt = 0
    matched = 0
    mismatches: List[Dict[str, Any]] = []
    for ri, gr in enumerate(gt_rows):
        pr = pred_rows[ri] if ri < len(pred_rows) else []
        for ci, g in enumerate(gr):
            g = _norm_cell(g)
            if not g or g == "?":
                continue
            n_gt += 1
            p = pr[ci] if ci < len(pr) else ""
            if _cells_equal(p, g):
                matched += 1
            else:
                mismatches.append(
                    {
                        "row": ri,
                        "col": ci,
                        "header": gt_headers[ci] if ci < len(gt_headers) else "",
                        "gt": g,
                        "pred": p,
                    }
                )
    rate = (matched / n_gt) if n_gt else 0.0
    header_ok = sum(1 for h in gt_headers if h in pred_headers) if gt_headers else 0
    return {
        "gt_cells": n_gt,
        "matched_cells": matched,
        "cell_rate": round(rate, 4),
        "header_rate": round(header_ok / len(gt_headers), 4) if gt_headers else 0.0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:30],
        "pred_headers": pred_headers,
        "pred_rows": pred_rows,
    }


def split_dual(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    mid = int(w * DUAL_SPLIT)
    return img[:, :mid].copy(), img[:, mid:].copy()


def run_pane_extract(
    extractor: TableImageExtractor,
    pane_path: Path,
    crop_ratio: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = extractor.extract(str(pane_path), crop_ratio=crop_ratio)
        ms = int((time.perf_counter() - t0) * 1000)
        return {
            "ok": True,
            "ms": ms,
            "headers": list(result.headers or []),
            "rows": [list(r) for r in (result.rows or [])],
            "raw_row_count": int(getattr(result, "raw_row_count", 0) or 0),
        }
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        return {
            "ok": False,
            "ms": ms,
            "error": f"{type(e).__name__}: {e}",
            "headers": [],
            "rows": [],
        }


def resolve_gt_for_item(
    m: Dict[str, Any], gt_doc: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    by_id = gt_doc.get("by_id") or {}
    if m["id"] in by_id:
        return by_id[m["id"]]
    by_burst = gt_doc.get("by_burst") or {}
    key = f"{m['slug']}|{m['burst_key']}"
    if key in by_burst:
        return by_burst[key]
    return None


def _gt_side_cells(side_gt: Optional[Dict[str, Any]]) -> int:
    if not side_gt:
        return 0
    n = 0
    for r in side_gt.get("rows") or []:
        for c in r:
            s = _norm_cell(c)
            if s and s != "?":
                n += 1
    return n


def pick_single_gt_side(gt: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """单屏真值：取有数据更多的一侧。"""
    left = gt.get("left") or {"headers": [], "rows": []}
    right = gt.get("right") or {"headers": [], "rows": []}
    lc, rc = _gt_side_cells(left), _gt_side_cells(right)
    if rc > lc:
        return "right", right
    return "left", left


def resolve_layout(
    m: Dict[str, Any], layout_doc: Dict[str, Any]
) -> str:
    by_id = layout_doc.get("by_id") or {}
    if m["id"] in by_id:
        return str(by_id[m["id"]])
    by_burst = layout_doc.get("by_burst") or {}
    key = f"{m['slug']}|{m['burst_key']}"
    if key in by_burst:
        return str(by_burst[key])
    # 默认：近景/远景按双屏；old 未标注时按双屏
    return "dual"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "data" / "ocr_bench"))
    ap.add_argument("--gt", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bad-threshold", type=float, default=0.70)
    ap.add_argument("--only-slug", default="")
    ap.add_argument("--only-layout", default="", help="single|dual")
    ap.add_argument("--ids", default="", help="comma-separated ids")
    ap.add_argument(
        "--merge-existing",
        action="store_true",
        help="合并写入 ocr_results.json（按 id 覆盖），不全量覆盖",
    )
    args = ap.parse_args()

    bench = Path(args.bench)
    panes_dir = bench / "panes"
    panes_dir.mkdir(parents=True, exist_ok=True)
    bad_dir = bench / "bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    ocr_out = bench / "ocr_out"
    ocr_out.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(bench / "manifest.json")
    gt_path = Path(args.gt) if args.gt else (bench / "ground_truth.json")
    gt_doc: Dict[str, Any] = _read_json(gt_path) if gt_path.exists() else {}
    layout_path = bench / "layout_map.json"
    layout_doc: Dict[str, Any] = (
        _read_json(layout_path) if layout_path.exists() else {"by_id": {}, "by_burst": {}}
    )

    extractor = TableImageExtractor()
    items = [m for m in manifest if m.get("include_ocr")]
    items.sort(key=lambda x: (x["slug"], x["seq"]))
    if args.only_slug:
        items = [m for m in items if m["slug"] == args.only_slug]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [m for m in items if m["id"] in want]
    if args.only_layout:
        items = [
            m
            for m in items
            if resolve_layout(m, layout_doc) == args.only_layout.strip().lower()
        ]
    if args.limit > 0:
        items = items[: args.limit]

    results: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []

    for i, m in enumerate(items, 1):
        img_path = bench / "images" / m["filename"]
        layout = resolve_layout(m, layout_doc)
        print(f"[{i}/{len(items)}] {m['id']} layout={layout}", flush=True)
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            results.append({"id": m["id"], "error": "read_fail", "layout": layout})
            continue

        row: Dict[str, Any] = {
            "id": m["id"],
            "filename": m["filename"],
            "category": m["category"],
            "slug": m["slug"],
            "burst_key": m["burst_key"],
            "layout": layout,
        }

        if layout == "single":
            # 整图单屏 OCR，禁止再切半屏
            single_p = panes_dir / f"{m['id']}__single.jpg"
            cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(
                str(single_p)
            )
            ocr = run_pane_extract(extractor, single_p, crop_ratio=SINGLE_TABLE_CROP)
            # 也跑一遍 auto（crop=None）若 fixed 很差可作参考，评测主用 fixed+auto 择优简单取更好填格
            ocr_auto = run_pane_extract(extractor, single_p, crop_ratio=None)
            def _fill(o: Dict[str, Any]) -> int:
                return sum(1 for r in (o.get("rows") or []) for c in r if _norm_cell(c))

            if _fill(ocr_auto) > _fill(ocr):
                ocr = ocr_auto
                ocr["crop_mode"] = "auto"
            else:
                ocr["crop_mode"] = "fixed"
            row["single"] = ocr
            # 兼容字段：把结果放到有真值的一侧标签下
            gt = resolve_gt_for_item(m, gt_doc)
            if gt:
                side_name, side_gt = pick_single_gt_side(gt)
                row[side_name] = ocr
                other = "right" if side_name == "left" else "left"
                row[other] = {"ok": True, "ms": 0, "headers": [], "rows": [], "skipped": True}
                cmp = compare_tables(
                    ocr.get("headers") or [], ocr.get("rows") or [], side_gt
                )
                row[f"{side_name}_compare"] = cmp
                row["cell_rate"] = cmp["cell_rate"]
                row["min_side_rate"] = cmp["cell_rate"]
                if cmp["gt_cells"] > 0 and cmp["cell_rate"] < args.bad_threshold:
                    bad.append(row)
                    dst = bad_dir / m["filename"]
                    if not dst.exists():
                        shutil.copy2(img_path, dst)
                if gt.get("unsalvageable"):
                    row["unsalvageable"] = True
                    bad_u = bench / "bad_unsalvageable"
                    bad_u.mkdir(parents=True, exist_ok=True)
                    dst = bad_u / m["filename"]
                    if not dst.exists():
                        shutil.copy2(img_path, dst)
        else:
            left, right = split_dual(img)
            left_p = panes_dir / f"{m['id']}__L.jpg"
            right_p = panes_dir / f"{m['id']}__R.jpg"
            cv2.imencode(".jpg", left, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(
                str(left_p)
            )
            cv2.imencode(".jpg", right, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(
                str(right_p)
            )
            left_ocr = run_pane_extract(extractor, left_p, crop_ratio=None)
            right_ocr = run_pane_extract(extractor, right_p, crop_ratio=None)
            left_ocr["crop_mode"] = "auto"
            right_ocr["crop_mode"] = "auto"
            row["left"] = left_ocr
            row["right"] = right_ocr

            gt = resolve_gt_for_item(m, gt_doc)
            if gt and not gt.get("skip"):
                side_rates = []
                for side in ("left", "right"):
                    side_gt = gt.get(side)
                    if not side_gt or _gt_side_cells(side_gt) == 0:
                        continue
                    ocr = row[side]
                    cmp = compare_tables(
                        ocr.get("headers") or [], ocr.get("rows") or [], side_gt
                    )
                    row[f"{side}_compare"] = cmp
                    if cmp["gt_cells"] > 0:
                        side_rates.append(cmp["cell_rate"])
                if side_rates:
                    row["cell_rate"] = round(sum(side_rates) / len(side_rates), 4)
                    row["min_side_rate"] = round(min(side_rates), 4)
                    if row["cell_rate"] < args.bad_threshold or row[
                        "min_side_rate"
                    ] < (args.bad_threshold - 0.15):
                        bad.append(row)
                        dst = bad_dir / m["filename"]
                        if not dst.exists():
                            shutil.copy2(img_path, dst)
                if gt.get("unsalvageable"):
                    row["unsalvageable"] = True
                    bad_u = bench / "bad_unsalvageable"
                    bad_u.mkdir(parents=True, exist_ok=True)
                    dst = bad_u / m["filename"]
                    if not dst.exists():
                        shutil.copy2(img_path, dst)

        results.append(row)
        (ocr_out / f"{m['id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    rates = [r["cell_rate"] for r in results if "cell_rate" in r]
    summary: Dict[str, Any] = {
        "total": len(results),
        "with_gt": len(rates),
        "bad_count": len(bad),
        "bad_threshold": args.bad_threshold,
        "mean_cell_rate": round(sum(rates) / len(rates), 4) if rates else None,
        "min_cell_rate": round(min(rates), 4) if rates else None,
        "max_cell_rate": round(max(rates), 4) if rates else None,
        "by_layout": {},
        "by_slug": {},
    }
    for lay in ("single", "dual"):
        rs = [
            r["cell_rate"]
            for r in results
            if r.get("layout") == lay and "cell_rate" in r
        ]
        summary["by_layout"][lay] = {
            "n": sum(1 for r in results if r.get("layout") == lay),
            "with_gt": len(rs),
            "mean_cell_rate": round(sum(rs) / len(rs), 4) if rs else None,
        }
    for slug in sorted({r.get("slug") for r in results if r.get("slug")}):
        rs = [
            r["cell_rate"]
            for r in results
            if r.get("slug") == slug and "cell_rate" in r
        ]
        summary["by_slug"][slug] = {
            "n": sum(1 for r in results if r.get("slug") == slug),
            "with_gt": len(rs),
            "mean_cell_rate": round(sum(rs) / len(rs), 4) if rs else None,
            "bad": sum(
                1
                for r in results
                if r.get("slug") == slug
                and "cell_rate" in r
                and r["cell_rate"] < args.bad_threshold
            ),
            "single": sum(
                1 for r in results if r.get("slug") == slug and r.get("layout") == "single"
            ),
            "dual": sum(
                1 for r in results if r.get("slug") == slug and r.get("layout") == "dual"
            ),
        }

    out_path = bench / "ocr_results.json"
    if args.merge_existing and out_path.exists():
        prev = _read_json(out_path)
        by_id = {r["id"]: r for r in (prev.get("results") or []) if r.get("id")}
        for r in results:
            by_id[r["id"]] = r
        results = [by_id[k] for k in sorted(by_id.keys())]
        bad = [
            r
            for r in results
            if "cell_rate" in r and r["cell_rate"] < args.bad_threshold
        ]
        rates = [r["cell_rate"] for r in results if "cell_rate" in r]
        summary = {
            "total": len(results),
            "with_gt": len(rates),
            "bad_count": len(bad),
            "bad_threshold": args.bad_threshold,
            "mean_cell_rate": round(sum(rates) / len(rates), 4) if rates else None,
            "min_cell_rate": round(min(rates), 4) if rates else None,
            "max_cell_rate": round(max(rates), 4) if rates else None,
            "by_layout": {},
            "by_slug": {},
        }
        for lay in ("single", "dual"):
            rs = [
                r["cell_rate"]
                for r in results
                if r.get("layout") == lay and "cell_rate" in r
            ]
            summary["by_layout"][lay] = {
                "n": sum(1 for r in results if r.get("layout") == lay),
                "with_gt": len(rs),
                "mean_cell_rate": round(sum(rs) / len(rs), 4) if rs else None,
            }
        for slug in sorted({r.get("slug") for r in results if r.get("slug")}):
            rs = [
                r["cell_rate"]
                for r in results
                if r.get("slug") == slug and "cell_rate" in r
            ]
            summary["by_slug"][slug] = {
                "n": sum(1 for r in results if r.get("slug") == slug),
                "with_gt": len(rs),
                "mean_cell_rate": round(sum(rs) / len(rs), 4) if rs else None,
                "bad": sum(
                    1
                    for r in results
                    if r.get("slug") == slug
                    and "cell_rate" in r
                    and r["cell_rate"] < args.bad_threshold
                ),
                "single": sum(
                    1
                    for r in results
                    if r.get("slug") == slug and r.get("layout") == "single"
                ),
                "dual": sum(
                    1
                    for r in results
                    if r.get("slug") == slug and r.get("layout") == "dual"
                ),
            }

    out = {
        "summary": summary,
        "results": results,
        "bad_ids": [b["id"] for b in bad],
    }
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bench / "bad_list.json").write_text(
        json.dumps(
            [
                {
                    "id": b["id"],
                    "filename": b["filename"],
                    "layout": b.get("layout"),
                    "cell_rate": b.get("cell_rate"),
                    "min_side_rate": b.get("min_side_rate"),
                    "left_rate": (b.get("left_compare") or {}).get("cell_rate"),
                    "right_rate": (b.get("right_compare") or {}).get("cell_rate"),
                }
                for b in bad
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
