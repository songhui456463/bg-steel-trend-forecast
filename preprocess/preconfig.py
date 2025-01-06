"""
predata config
pretest config
"""

from enum import Enum

from pre_enums import EnumRepairMissingMethod


class PreConfig:
    # 预处理参数ss

    CHECK_MISSING: dict = {
        "repairable_ratio": 0.9,  # numerical_total_ratio <= 0.6 可修补
        "repairable_longest_consec_ratio": 0.3,  # numerical_longest_consecutive_missing <= int(0.3 * len(df), 可修补
        "is_print": True,  # bool
    }
    REPAIR_MISSING: dict = {
        "method": EnumRepairMissingMethod.LINEAR,
        # 'method': 'polynomial',  # 'drop', 'ma', 'linear', 'polynomial','nearest'
        # 'method_ma':{'moving_window':1,
        #              },
        "is_plot": True,  # bool
    }

    # 定义异常值的检测规则，后续可根据数据情况或人工知识调整
    # check_outliers_rules: dict = {'check rules':[]}
    # 定义异常值的范围，后续可根据数据情况或人工知识调整,定义可以修复的异常值情况

    CHECK_OUTLIER: dict = {
        "upper bound": 100000,
        "lower bound": 1,
        "repairable_ratio": 0.6,
        "repairable_longest_consec_ratio": 0.3,
    }

    REPAIR_OUTLIER: dict = {"is_plot": True}  # bool


preconfig: PreConfig = PreConfig()
