# ocr_engine（内置表格 OCR）

原 `image-table-extract`，现位于 `app/ocr_engine/`，随 `app` 整体打包。

由 `app.backend.ocr_bridge` 通过 `ocr.workdir`（默认 `app/ocr_engine`）加载 `extract_table.TableImageExtractor`。

依赖见项目根目录 `requirements.txt`。
