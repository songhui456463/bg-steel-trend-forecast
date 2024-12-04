"""
预测模型的参数配置
"""

import pandas as pd

from utils.log import mylog
from utils.enum_family import EnumFreq, EnumForecastMethod
from utils.data_read import harmonize_weekfreq_date, harmonize_monthfreq_date
from forecasting.local_data_map import factor_location_map


class FORECASTCONFIG:
    """序列名称"""

    # 价格序列
    # TARGET_NAME: str = '国际热轧板卷汇总价格：中国市场（日）'  # 日频测试
    # TARGET_NAME: str = '国际热轧板卷汇总价格：中国市场（周）'  # 周频测试
    TARGET_NAME: str = "国际热轧板卷汇总价格：中国市场（月）"  # 月频测试

    # 候选因子名称列表
    # candi_factors_name: list = list(factor_location_map.keys())  # 使用所有因子
    CANDI_FACTORS_NAME: list = [  # 使用自选因子
        "销量:液压挖掘机:主要企业:出口(外销):当月值",
        "30大中城市:商品房成交面积",
        "制冷:冰箱:销量:当月值",
        "家电:洗衣机:销量:当月值",
    ]

    """date range"""
    # 因子分析 ANALYSE_Y_STARTDATE <= date < ANALYSE_Y_ENDDATE     #  注意：2个date最好是标的序列中有的日期；若不是，则会根据y_freq被统一到当周周五或当月第一天
    # 日频测试
    # ANALYSE_Y_STARTDATE: str = "2020-01-02"  # 对price序列的[y_start_date, y_end_date]期间的子序列进行相关性检测
    # ANALYSE_Y_ENDDATE: str = "2024-01-02"  # today期
    # 周频测试
    # ANALYSE_Y_STARTDATE: str = "2017-01-01"
    # ANALYSE_Y_ENDDATE: str = "2023-01-01"
    # 月频测试
    ANALYSE_Y_STARTDATE: str = "2016-10-01"
    ANALYSE_Y_ENDDATE: str = "2022-11-01"

    # 预测
    # PRE_START_DATE: str = '2024-01-02'  # analyse_y_enddate的下一期。实际预测的第一期是pre_start_date及之后的第一个按频度的日子
    # PRE_START_DATE: str = '2023-01-01'  # 周频测试
    PRE_START_DATE: str = "2022-12-01"  # 月频测试

    ROLL_STEPS: int = 10  # 滚动测试的滚动步数
    PRE_STEPS: int = 6  # 每次预测的预测步数

    """预测过程"""
    # 训练数据长度
    LEN_TRAIN_DAY = 750  # 暂取三年
    LEN_TRAIN_WEEK = 150
    LEN_TRAIN_MONTH = 36

    # 预测pre_steps步过程中模型更新的频率  # 每预测几步重新训练一次模型，=1表示每次roll都重新训练模型
    MODEL_UPDATE_FREQ: dict = {
        # 单因子模型
        EnumForecastMethod.ARIMA: 9999,
        EnumForecastMethod.HOLTWINTERS: 9999,
        EnumForecastMethod.FBPROPHET: 9999,
        EnumForecastMethod.LSTM_SINGLE: 9999,
        # 多因子模型
        EnumForecastMethod.LSTM_MULTIPLE: 9999,
        EnumForecastMethod.VAR: 9999,
    }

    def harmonize_param_date(self, y_freq: EnumFreq):
        # 规范输入的日期参数，使日期改到输入日期所在的周或月的规范日期
        if y_freq == EnumFreq.MONTH:
            self.ANALYSE_Y_STARTDATE = harmonize_monthfreq_date(
                pd.to_datetime(self.ANALYSE_Y_STARTDATE)
            ).strftime("%Y-%m-%d")
            self.ANALYSE_Y_ENDDATE = harmonize_monthfreq_date(
                pd.to_datetime(self.ANALYSE_Y_ENDDATE)
            ).strftime("%Y-%m-%d")
            self.PRE_START_DATE = harmonize_monthfreq_date(
                pd.to_datetime(self.PRE_START_DATE)
            ).strftime("%Y-%m-%d")
            mylog.info(
                f"harmonize month freq date: ANALYSE_Y_STARTDATE={self.ANALYSE_Y_STARTDATE}, ANALYSE_Y_ENDDATE={self.ANALYSE_Y_ENDDATE}, PRE_START_DATE={self.PRE_START_DATE}"
            )

        if y_freq == EnumFreq.WEEK:
            self.ANALYSE_Y_STARTDATE = harmonize_weekfreq_date(
                pd.to_datetime(self.ANALYSE_Y_STARTDATE)
            ).strftime("%Y-%m-%d")
            self.ANALYSE_Y_ENDDATE = harmonize_weekfreq_date(
                pd.to_datetime(self.ANALYSE_Y_ENDDATE)
            ).strftime("%Y-%m-%d")
            self.PRE_START_DATE = harmonize_weekfreq_date(
                pd.to_datetime(self.PRE_START_DATE)
            ).strftime("%Y-%m-%d")
            mylog.info(
                f"harmonize week freq date: ANALYSE_Y_STARTDATE={self.ANALYSE_Y_STARTDATE}, ANALYSE_Y_ENDDATE={self.ANALYSE_Y_ENDDATE}, PRE_START_DATE={self.PRE_START_DATE}"
            )

    def check_param_date_existence(self, origin_y_df: pd.DataFrame):
        if not pd.to_datetime(self.ANALYSE_Y_STARTDATE) in origin_y_df.index:
            mylog.warning(
                f"参数ANALYSE_Y_STARTDATE={self.ANALYSE_Y_STARTDATE} 不存在于价格序列中，需要更换参数"
            )  # todo 自动替换为寻找最近的

        if not pd.to_datetime(self.ANALYSE_Y_ENDDATE) in origin_y_df.index:
            mylog.warning(
                f"参数ANALYSE_Y_ENDDATE={self.ANALYSE_Y_ENDDATE} 不存在于价格序列中，需要更换参数"
            )  # todo 自动替换为寻找最近的

        if not pd.to_datetime(self.PRE_START_DATE) in origin_y_df.index:
            # 实际预测的第一期是pre_start_date及之后的第一个按频度的有效的日子
            mylog.warning(
                f"参数PRE_START_DATE={self.PRE_START_DATE} 不存在于价格序列中，第一个T+1预测日期将为下一个"
            )


ForecastConfig = FORECASTCONFIG()
