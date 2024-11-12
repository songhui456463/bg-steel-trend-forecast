"""
预处理过程中的枚举类
"""

from enum import Enum, IntEnum


class EnumExpectedDtype(IntEnum):
    INT = 1
    FLOAT = 2


class EnumProcessedDF(Enum):
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
