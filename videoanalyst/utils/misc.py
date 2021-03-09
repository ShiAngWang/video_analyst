# -*- coding: utf-8 -*
import hashlib
import time
from typing import Dict

from loguru import logger
from yacs.config import CfgNode as CN


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict, print(
        module_name, module_dict, 'defined in several script files')
    module_dict[module_name] = module


class Registry(dict):
    r"""
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    usually declared in XXX_base.py, e.g. videoanalyst/model/backbone/backbone_base.py

    used as decorator when declaring the module:

    @some_registry.register
    def foo():
        ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]

    注册处类，继承于 dict 类，提供一个 register 函数用于将 模块名称：模块类对象 这样的键值对添加（注册）到注册处实例中。
    后期可以通过注册处实例检索模块类 （f = some_registry["foo_module"]），并使用预定义的配置文件实例化模块类对象来构建整个 pipeline（例如，
    构建 pipeline 的跟踪器/分割器/训练器等）

    关于注册机制的介绍可以查阅 DEVELOP.md
    """
    def __init__(self, *args, **kwargs):
        self.name = 'Registry'
        if len(args) > 0 and isinstance(args[0], str):
            name, *args = args  # 数据拆包
            self.name = name  # Registry 字典命名
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module):  # 将 模块名称：模块类对象 这样的键值对添加（注册）到 Registry 字典实例中
        name = module.__name__  # 键 模块名称
        _register_generic(self, name,
                          module)  # 添加 模块名称：模块类对象 键值对到 Registry 字典实例
        #logger.debug('%s: %s registered' % (self.name, name))
        return module


def load_cfg(path: str):
    r"""
    Load yaml with yacs

    Arguments
    ---------
    path: str
        yaml path
    """
    with open(path, 'r') as f:
        config_node = CN.load_cfg(f)

    return config_node


def merge_cfg_into_hps(cfg: CN, hps: Dict):
    for hp_name in hps:
        if hp_name in cfg:
            new_value = cfg[hp_name]
            hps[hp_name] = new_value
    return hps


class Timer():
    r"""
    Mesure & print elapsed time witin environment
    """
    def __init__(self,
                 name: str = "",
                 output_dict: Dict = None,
                 verbose: bool = False):
        """Timing usage
        
        Parameters
        ----------
        name : str, optional
            name of timer, used in verbose & output_dict, by default ''
        output_dict : Dict, optional
            dict-like object to receive elapsed time in output_dict[name], by default None
        verbose : bool, optional
            verbose or not via logger, by default False
        """
        self.name = name
        self.output_dict = output_dict
        self.verbose = verbose

    def __enter__(self, ):
        self.tic = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.time()
        elapsed_time = self.toc - self.tic
        if self.output_dict is not None:
            self.output_dict[self.name] = elapsed_time
        if self.verbose:
            print_str = '%s elapsed time: %f' % (self.name, elapsed_time)
            logger.info(print_str)


def md5sum(file_path) -> str:
    """Get md5sum string
    
    Parameters
    ----------
    file_path : str
        path to file to calculate md5sum
    
    Returns
    -------
    str
        md5 value string in hex
    """
    with open(file_path, "rb") as f:
        md5sum_str = hashlib.md5(f.read()).hexdigest()
    return md5sum_str
