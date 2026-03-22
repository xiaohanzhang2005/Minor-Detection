# 模块说明：
# - pytest 的路径引导文件。
# - 保证本地测试时项目包导入稳定。

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
