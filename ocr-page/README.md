# 珙县抓拍 OCR → PI Demo

全 Python：`DS-2CD6855FWD-PL` ISAPI 抓图 → 内置 `app/ocr_engine` OCR → `ctypes` 调 `piapi.dll` 写/读 PI → 简易页面。

目录约定：程序在 [`app/`](app/)（`backend` 业务、`static` 页面、`ocr_engine` 识别、`pidll`），与 [`data/`](data/)、[`scripts/`](scripts/) 同级。

`yingwei-realdb` 仅作 API 对照，运行时不依赖。

## 统一入口：`ocr.bat`

根目录入口 [`ocr.bat`](ocr.bat)，其余脚本在 [`scripts/`](scripts/)。

```bat
ocr.bat                 :: 交互菜单
ocr.bat install         :: 安装 Python3.12（可 winget）+ 创建 .venv + pip
ocr.bat offline         :: 由 .venv 生成便携 python\
ocr.bat start           :: 后台启动 http://127.0.0.1:8088/
ocr.bat stop            :: 停止服务
ocr.bat restart         :: 停止后重启（更新代码后用）
ocr.bat pack            :: 轻量更新包 → ..\gx-capture-demo-update.zip
ocr.bat pack -all       :: 全量离线包；若无 python\ 则自动 install/offline 再打包
```

## 现场离线包（推荐）

目录内已带便携 `python\`（含依赖与 OCR 模型），现场**无需联网、无需安装 Python**：

1. 开发机：`ocr.bat install` → `ocr.bat offline` → `ocr.bat pack -all`
2. 拷到现场解压 → 改 `config.yaml` → `ocr.bat start`
3. 浏览器打开 http://127.0.0.1:8088/ ；停止：`ocr.bat stop`

轻量更新（不含 `python\`）：`ocr.bat pack`

详见 [`部署说明.txt`](部署说明.txt)。

## 准备（开发机联网）

**重要：不要用 Python 3.14。** OCR 库 `rapidocr-onnxruntime` 仅支持 **Python 3.10 ~ 3.12**。

```bat
cd d:\yw\gx\gx-capture-demo
ocr.bat install
```

然后编辑 [`config.yaml`](config.yaml)：

- `camera.*`：相机 IP / 账号 / 通道（默认 `101`）
- `capture.mode`：`single` / `dual`，与下方 `capture.single` / `capture.dual` 对应
- `capture.single.tag` / `crop`：单屏测点与裁剪；`capture.dual.*.tag/crop`：双屏左右
- `schedule`：定时处置（`enabled` / `interval_sec` / `capture` / `ocr` / `pi`）；开启后立刻跑一轮，本轮结束后再等 `interval_sec` 秒开始下一周期；阶段开关任一为 false 则该阶段及之后不执行
- `pi.*`：`dll_dir`（默认 `app/pidll/`）、服务器、账号
- `ocr.min_points`：有效点数绝对下限；低于则本轮（双屏为该侧）识别失败、不写 PI，等下一轮

OCR 引擎位于 [`app/ocr_engine/`](app/ocr_engine/)（`ocr.workdir`，默认即可）。

## 启动服务

```bat
cd d:\yw\gx\gx-capture-demo
ocr.bat start
```

浏览器打开：

- 执行履历（首页）：http://127.0.0.1:8088/
- 执行详情：http://127.0.0.1:8088/runs/{id}
- 分步联调：http://127.0.0.1:8088/debug
- 单屏一键：http://127.0.0.1:8088/single
- 双屏一键：http://127.0.0.1:8088/dual

（旧地址 `/demo`、`/split` 会 302 到 `/single`、`/dual`）

## 单屏 / 双屏

| 模式 | 配置 | 入口 | API |
|------|------|------|-----|
| 单屏单页（原逻辑） | `capture.mode: single` + `capture.single` | `/single`、`/debug` | `POST /api/run`、`/api/ocr` |
| 双屏两机组 | `capture.mode: dual` + `capture.dual` | `/dual`、`/debug` | `POST /api/run/dual`、`/api/ocr/dual` |

**单屏代码路径未改**：`pipeline.run_pipeline` / `/api/run` 行为与原来一致。  
双屏在 `app/pipeline_dual.py`：抓一整帧 → 按 `capture.dual.*.crop` 裁半屏 → **各自 OCR**（各自 `tag`、各自业务日期）→ 按点内 tag 写 PI。

同刻不同天示例：左填 `2025-09-15`、右填 `2025-09-16`，两边都有 `15:00` 时：

- 左 → `GX.DEMO.OCR.L_N` @ `2025-09-15 15:00:00`
- 右 → `GX.DEMO.OCR.R_N` @ `2025-09-16 15:00:00`

现场若左右不是精确对半，只需改 `capture.dual.*.crop`。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/capture` | 仅抓图存盘 |
| POST | `/api/ocr` | body: `{"image_path":"..."}`；可按开关写 Excel |
| POST | `/api/ocr/upload` | multipart 上传本机图片并 OCR（临时文件，不写入 captures） |
| POST | `/api/run` | **单屏**：抓图 + OCR 整表 +（可选 Excel）+ 按行写 PI |
| POST | `/api/run/dual` | **双屏**：抓图 + 左右独立 OCR（`left_date`/`right_date`）+ 按点内 tag 写 PI |
| POST | `/api/ocr/dual` | 双屏分步：对已有整图左右裁剪 OCR（不写 PI） |
| POST | `/api/ocr/dual/upload` | 双屏本机图上传并左右 OCR（临时文件，不写入 captures） |
| GET | `/api/runs` | 执行履历列表 |
| GET | `/api/runs/{id}` | 执行详情 |
| GET | `/api/config/view` | 读 `capture.mode` / `single` / `dual` |
| GET | `/api/schedule` | 定时处置配置与运行状态 |
| POST | `/api/schedule/run` | 立刻执行一轮定时处置（按 schedule.capture/ocr/pi 截断） |
| GET | `/api/curve?start=&end=` | 读 PI 测点档案时间与值，有多少返回多少 |
| POST | `/api/pi/write` | 写点列表；可选 `tag` 前缀 |
| GET | `/api/config/pi` | 读当前测点前缀等 |
| PUT | `/api/config/pi/tag` | 更新 `capture.single.tag`（单屏前缀） |
| POST | `/api/demo/export-excel` | Demo 导出 Excel 备份 |

## 说明

- 抓图 URL：`http://{host}:{port}/ISAPI/Streaming/channels/{channel}/picture`（Digest）
- OCR 按内置 `app/ocr_engine` 整表识别：第 1 行时间表头，其余均为数据行；**仅识别画面上可见列**
- 写 PI 固定规则：一行数据 → 前缀_1…_N；单屏前缀 `capture.single.tag`，双屏 `capture.dual.left/right.tag`
- 执行履历：每次流水线写入 `data/runs/{id}.json`（并同步到 `data/ocr_runs/.../run.json`）；首页列表 + `/runs/{id}` 详情
- OCR 质量：`ocr.min_points`（默认 8）。有效点数不足视为切屏/熄屏/模糊等失败：**不写 PI**；双屏左右各自判断，合格侧照常写
- 抓图目录 `data/captures/YYYYMMDD/`，文件名按时刻：`9-09.jpg`、`10-20.jpg`（Windows 不能用 `:`，用 `-`；同分钟重复为 `_2`…）；外层日期 + 文件名即完整时间
- OCR 过程中间文件：`ocr.artifacts_dir`（默认 `data/ocr_runs/`，始终落盘，无开关）
- 定时处置：`schedule.enabled=true` 后立刻跑第一轮，本轮结束后再等 `interval_sec` 秒开始下一周期；按 `schedule.capture` / `ocr` / `pi` 截断执行；改 YAML 后下一轮前热加载
- Excel 备份：`excel.enabled` / `excel.dir`，按日分区 `data/exports/YYYYMMDD/`（写 PI **不读** Excel）
- 通道拍不到目标画面时改 `camera.channel`
