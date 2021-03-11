# -*- coding: utf-8 -*-

from ..template_module_base import (TRACK_TEMPLATE_MODULES,
                                    VOS_TEMPLATE_MODULES, TemplateModuleBase)


@TRACK_TEMPLATE_MODULES.register
@VOS_TEMPLATE_MODULES.register
class TemplateModuleImplementation(TemplateModuleBase):
    r"""
    Template Module Implementation

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(template_module_hp="",
                                )  # 这部分定义模块的hyper-parameters(name/type)

    def __init__(self, ) -> None:
        super().__init__()

    def update_params(self) -> None:
        r"""
        定义了用配置文件提供的超参数更新超参数的行为

        Returns:

        """
