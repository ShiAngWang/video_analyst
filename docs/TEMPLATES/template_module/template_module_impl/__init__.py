# -*- coding: utf-8 -*-
import glob
from os.path import basename, dirname, isfile

modules = glob.glob(dirname(__file__) + "/*.py")
modules = [m for m in modules if not m.endswith(('_bak.py'))
           ]  # filter file with name ending with '_bak' (debugging)
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f)
    and not f.endswith("__init__.py") and not f.endswith("utils.py")
]  # 从 __all__ 中滤除 __init__.py 与 "*utils.py", "*_bak.py"
