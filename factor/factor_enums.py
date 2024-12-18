"""
多因子分析过程中的枚举类
"""

from enum import Enum


class Enum_allfactor_lt_corr_res_DF(Enum):
    # 多因子分析 提前期和相关性分析结果的df 的列名（固定）
    y_name = "y_name"
    x_name = "x_name"
    corr = "corr"
    beta = "beta"
    t_statis = "t_statis"
    p_value = "p_value"
    is_significant_corr = "is_significant_corr"
    best_lag = "best_lag"
