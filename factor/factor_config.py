"""
因子分析模块参数
"""

from typing import Dict
from utils.enum_family import EnumFreq


class FACTORCONFIG:  # todo 参数调整
    """提前期相关性分析"""

    MAX_LT_CHECK: Dict[EnumFreq, int] = {
        EnumFreq.DAY: 6,  # 频度为day时，要分析的最大提前期数
        EnumFreq.WEEK: 6,
        EnumFreq.MONTH: 6,
    }

    """共线性分析"""
    N_CLUSTERS: int = 1  # 聚类簇数
    VIF_THRED: int = 10  # vif的阈值，大于该阈值认为共线性较严重
    VIF_MAX_CYCLE: int = 10  # 循环迭代求vif的最大迭代次数


FactorConfig = FACTORCONFIG()
