"""
预处理过程中的枚举类
"""

from enum import Enum, IntEnum


class EnumExpectedDtype(IntEnum):
    INT = 1
    FLOAT = 2


class EnumProcessedDF(Enum):
    # 单因子预处理模块的self.processed_df的列名（固定）
    MISSING_TYPE = "missing_type"
    MISSING_NUMERILIZED = "missing_numerilized"
    MISSING_REPAIRED = "missing_repaired"
    IS_OUTLIER = "is_outlier"
    OUTLIER_TYPE = "outlier_type"
    OUTLIERS_REPAIRED = "outliers_repaired"


class EnumRepairMissingMethod(Enum):
    DROP = "drop"
    MA = "ma"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    NEAREST = "nearest"


class EnumRepairOutliersMethod(Enum):
    REMOVE = "remove"
    REPLACE_MEAN = "replace_mean"
    REPLACE_MEDIAN = "replace_median"
    INTERPOLATE = "interpolate"
    MA = "moving_average"
    EMA = "exponential_moving_average"
    QR = "quantile_replacement"


class EnumPretestingReturn(Enum):
    """pretesting.py中所有检验的返回字典的key"""

    # 自相关性检测的return dict的key
    autocorrTest_is_corr = "is_corr"
    # 正态性检检验
    gaussianTest_is_gaussian = "is_gaussian"
    gaussianTest_skew = "skew"
    gaussianTest_kurtosis = "kurtosis"
    # 白噪声检验
    whitenoiseTest_is_whitenoise = "is_whitenoise"
    # 平稳性检验
    stationaryTest_is_stationary = "is_stationary"
    stationaryTest_stationary_d = "stationary_d"
    # 异方差检验
    hetetoTest_is_het = "is_het"
