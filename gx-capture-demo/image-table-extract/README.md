# image-table-extract（内置）

表格 OCR 引擎，已纳入 `gx-capture-demo` 本仓库，随目录整体打包即可，**无需再依赖外层 `../image-table-extract`**。

由 `app/ocr_bridge.py` 通过 `ocr.workdir`（默认 `image-table-extract`）加载 `extract_table.TableImageExtractor`。

依赖见项目根目录 `requirements.txt`。
