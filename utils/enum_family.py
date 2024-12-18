"""
公共枚举类们
"""

from enum import Enum


class EnumVariety(Enum):
    LENGZHA = "冷轧"
    REZHA = "热轧"
    LUOWENGANG = "螺纹钢"
    HOUBAN = "厚板"
    DUXIN = "镀锌"


class EnumSource(Enum):
    MYSTEEL = "Mysteel"
    TONGHUASHUN = "TongHuaShun"  # IFIND
    # 待补充


class EnumFreq(Enum):
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    SEASON = "SEASON"
    HALFYEAR = "HALFYEAR"
    YEAR = "YEAR"


class EnumForecastMethodType(Enum):
    SINGLE = "single"
    MULTI = "multi"
    SIMPLE_FIT = "simple_fit"  # 简单拟合数据序列


class EnumForecastMethod(Enum):
    """单因子预测"""

    ARIMA = ("arima", EnumForecastMethodType.SINGLE)
    HOLTWINTERS = ("holtwinters", EnumForecastMethodType.SINGLE)  # holtwinters
    FBPROPHET = ("fbprophet", EnumForecastMethodType.SINGLE)  # fbprophet
    LSTM_SINGLE = (
        "lstm_single",
        EnumForecastMethodType.SINGLE,
    )  # 使用单因子预测价格序列

    """多因子"""
    LSTM_MULTIPLE = (
        "lstm_multiple",
        EnumForecastMethodType.MULTI,
    )  # 使用多因子预测价格序列
    VAR = ("var", EnumForecastMethodType.MULTI)

    """简单拟合"""
    NORMAL_FIT = ("normalfit", EnumForecastMethodType.SIMPLE_FIT)
    T_FIT = ("t_fit", EnumForecastMethodType.SIMPLE_FIT)
    GARCH_FIT = ("garchfit", EnumForecastMethodType.SIMPLE_FIT)

    def __init__(self, value, type_):
        self._value_ = value
        self.type = type_


if __name__ == "__main__":
    lst = [EnumForecastMethod.ARIMA, EnumForecastMethod.VAR]
