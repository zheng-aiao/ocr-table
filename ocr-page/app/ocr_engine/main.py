# -*- coding: utf-8 -*-
"""命令行：从指定位置读取图片，识别表格并导出 CSV/TXT。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config_loader import load_config, parse_crop
from extract_table import TableImageExtractor


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从指定路径读取图片，识别表格并导出 CSV/TXT")
    parser.add_argument(
        "image", nargs="?",
        help="输入图片路径（绝对路径或相对路径）；省略时读 config.properties 中 input.image")
    parser.add_argument("-i", "--input", dest="image_alt",
                        help="输入图片路径（与 positional 二选一）")
    parser.add_argument("-c", "--config", help="配置文件路径，默认 config.properties")
    parser.add_argument("-o", "--output", help="输出文件路径（默认与图片同目录）")
    parser.add_argument("-f", "--format", choices=["csv", "txt"],
                        help="输出格式，默认 csv 或配置文件 output.format")
    parser.add_argument("--scale", type=float, help="识别前放大倍数，默认 2.5")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="OCR 最低置信度，默认 0.3")
    parser.add_argument("--crop", help="裁剪区域 x0,y0,x1,y1（比例 0~1）")
    return parser


def resolve_input_path(args, config: dict, script_dir: Path) -> Path:
    """
    确定要读取的图片路径。

    @param args       命令行参数
    @param config     配置文件内容
    @param script_dir 脚本所在目录（相对路径基准）
    @return 图片路径
    """
    raw = args.image_alt or args.image or config.get("input.image")
    if not raw:
        raise ValueError(
            "请指定图片路径：\n"
            "  python main.py f:\\path\\to\\image.png\n"
            "  python main.py -i f:\\path\\to\\image.png\n"
            "或在 config.properties 中设置 input.image=...")
    return TableImageExtractor.resolve_image_path(raw, base_dir=script_dir)


def default_output(image_path: Path, fmt: str, config: dict) -> Path:
    """
    生成默认输出路径。

    @param image_path 输入图片
    @param fmt        输出格式
    @param config     配置
    @return 输出路径
    """
    out_dir = config.get("output.dir")
    name = image_path.stem + "_table." + fmt
    if out_dir:
        return Path(out_dir).expanduser().resolve() / name
    return image_path.with_name(name)


def main(argv=None) -> int:
    """
    程序入口。

    @param argv 命令行参数
    @return 退出码
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    config = load_config(args.config)

    try:
        image_path = resolve_input_path(args, config, script_dir)
    except (ValueError, FileNotFoundError) as e:
        print("错误:", e, file=sys.stderr)
        return 1

    fmt = args.format or config.get("output.format", "csv")
    scale = args.scale
    if scale is None:
        scale = float(config.get("scale", "2.5"))
    crop = None
    crop_raw = args.crop or config.get("crop")
    if crop_raw:
        crop = parse_crop(crop_raw)

    out_path = Path(args.output) if args.output else default_output(image_path, fmt, config)

    print("读取图片:", image_path)
    extractor = TableImageExtractor(scale=scale, min_confidence=args.min_confidence)
    try:
        result = extractor.extract(str(image_path), crop_ratio=crop)
        saved = extractor.export(result, str(out_path), fmt=fmt)
    except Exception as e:
        print("识别失败:", e, file=sys.stderr)
        return 1

    print("识别完成")
    print("  输出:", saved)
    print("  表头列数:", len(result.headers))
    print("  数据行数:", len(result.rows))
    if result.headers:
        print("  表头:", ", ".join(result.headers[:8]),
              ("..." if len(result.headers) > 8 else ""))
    for i, row in enumerate(result.rows[:5]):
        preview = ", ".join(row[:8])
        print("  行%d: %s%s" % (i + 1, preview, "..." if len(row) > 8 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
