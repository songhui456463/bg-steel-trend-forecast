"""
公共枚举类们
"""

from enum import Enum, IntEnum


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
    # DAY = 1
    # WEEK = 2
    # MONTH = 3
    # SEASON = 4
    # HALFYEAR = 5
    # YEAR = 6
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    SEASON = "SEASON"
    HALFYEAR = "HALFYEAR"
    YEAR = "YEAR"


class EnumForecastMethod(Enum):
    # 单因子预测
    ARIMA = "arima"
    HOLTWINTERS = "holtwinters"  # holtwinters
    FBPROPHET = "fbprophet"  # fbprophet
    LSTM_SINGLE = "lstm_single"  # 使用单因子预测价格序列
    # 多因子
    LSTM_MULTIPLE = "lstm_multiple"  # 使用多因子预测价格序列
    VAR = "var"
    # todo 其他模型
