# -*- coding: utf-8 -*-
from typing import Dict, List

from yacs.config import CfgNode

from videoanalyst.utils import merge_cfg_into_hps

from ..dataset.builder import get_config as get_dataset_cfg
from ..filter.builder import get_config as get_filter_cfg
from ..sampler.builder import build as build_sampler
from ..target.builder import build as build_target
from ..transformer.builder import build as build_transformer
from .datapipeline_base import TASK_DATAPIPELINES, DatapipelineBase


# datapipeline主要由下面三部分构成 sampler transformers target
def build(task: str, cfg: CfgNode, seed: int = 0) -> DatapipelineBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: data
    seed: int
        seed for rng initialization
    """
    assert task in TASK_DATAPIPELINES, "invalid task name"
    MODULES = TASK_DATAPIPELINES[task]

    sampler = build_sampler(
        task, cfg.sampler,
        seed=seed)  # 从同一个序列中产生一个positive pair或者是从不同序列中产生一个negative pair
    transformers = build_transformer(
        task, cfg.transformer, seed=seed
    )  # 数据增广，主要有scale和shift，使search patch和templete patch并不是正好目标中心对着的，会有一点偏移
    target = build_target(task, cfg.target)  # 构造label

    pipeline = []
    pipeline.extend(transformers)
    pipeline.append(target)

    cfg = cfg.datapipeline
    name = cfg.name
    module = MODULES[name](sampler, pipeline)

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    cfg_dict: dict = {name: CfgNode() for name in task_list}

    for cfg_name, modules in TASK_DATAPIPELINES.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = ""

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

        cfg["submodules"] = CfgNode()
        cfg["submodules"] = get_filter_cfg(task_list)[cfg_name]
        cfg["submodules"] = get_dataset_cfg(task_list)[cfg_name]

    return cfg_dict
