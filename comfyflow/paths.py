"""插件安装目录，独立于内部模块的文件位置。"""

from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
