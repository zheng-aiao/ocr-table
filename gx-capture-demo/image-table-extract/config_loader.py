# -*- coding: utf-8 -*-
"""读取 config.properties 中的默认图片路径等配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple


def load_config(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    加载配置文件（key=value，# 开头为注释）。

    @param config_path 配置文件路径，默认与本模块同目录的 config.properties
    @return 配置字典
    """
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent / "config.properties"
    result: Dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value:
            result[key.strip()] = value
    return result


def parse_crop(value: str) -> Tuple[float, float, float, float]:
    """
    解析裁剪参数 x0,y0,x1,y1。

    @param value 逗号分隔字符串
    @return 四元组
    """
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("crop 格式应为 x0,y0,x1,y1")
    return tuple(parts)
