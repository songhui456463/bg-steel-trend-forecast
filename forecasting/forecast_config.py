"""
预测模型的参数配置
"""

import pandas as pd

from forecasting.local_data_map import factor_location_map, price_location_map
from utils.data_read import harmonize_weekfreq_date, harmonize_monthfreq_date
from utils.enum_family import EnumFreq, EnumForecastMethod
from utils.log import mylog


class FORECASTCONFIG:
    """序列名称"""

    # 候选因子名称列表
    # CANDI_FACTORS_NAME: list = list(factor_location_map.keys())  # 使用所有因子 (不包含价格序列，但里面仍有价格指数相关序列)
    CANDI_FACTORS_NAME: list = [  # 使用自选因子
        "金属集装箱:产量:当月值",
        # '螺纹钢：消费量：中国（周）',
        # '线材：消费量：中国（周）',
        "热轧板卷：消费量：中国（周）",
        "冷轧板卷：消费量：中国（周）",
        "中厚板：消费量：中国（周）",
        "销量:液压挖掘机:主要企业:出口(外销):当月值",
        "销量:液压挖掘机:主要企业:总计:当月值",
        "30大中城市:商品房成交面积",
        "空调:家用空调:销量:当月值",
        "制冷:冷柜:销量:当月值",
        "制冷:冰箱:销量:当月值",
        "家电:洗衣机:销量:当月值",
        "家用空调产量（月）",
        "家用空调销量（月）",
        "冰箱产量（月）",
        "冰箱销量（月）",
        "洗衣机产量（月）",
        "洗衣机销量（月）",
        "彩电产量（月）",
        "彩电销量（月）",
        "家用空调内销量（月）",
        "家用空调出口数量（月）",
        # '冰箱内销量（月）',
        "冰箱出口数量（月）",
        # '洗衣机内销量（月）',
        "洗衣机出口数量（月）",
        "彩电内销量（月）",
        "彩电出口数量（月）",
        "挖掘机:产量:当月值",
        "水泥：出库量：中国（周）",
        "重交沥青：产能利用率：中国（周）",
        "销量:平地机:出口(外销):当月值",
        "销量:装载机:出口(外销):当月值",
        "机床:出口数量:当月值",
        "变压器:出口数量:当月值",
        "船舶:出口数量:当月值",
        "集装箱:出口数量:当月值",
        "家用电器:出口数量:当月值",
        "出口数量:机电产品:汽车(包括底盘):当月值",
        "摩托车:出口数量:当月值",
        "自行车:出口数量:当月值",
        "出口数量:钢材:当月值",
        "制造业PMI",
        "制造业PMI:生产",
        "制造业PMI:新订单",
        # 'CPI:环比',
        "美国:耐用品:库存:季调",
        "美国:CPI:当月环比",
        "美国:PPI:所有商品:当月环比",
        "美国:ISM:制造业PMI:物价",
        "出口总值(美元计价):当月值",
        "汽车产量:当月值",
        "汽车销量:当月值",
        "汽车:出口数量:当月值",
        "房地产开发投资:累计同比",
        "新承接船舶订单:累计同比",
        "炼焦煤：价格指数：中国（日）",
        "粗钢:产量:当月值",
        # 上游
        "价格指数:铁矿石:62%Fe:CFR:青岛港",
        "186家矿山企业（363座矿山）：铁精粉：日均产量（周）",
        "247家钢铁企业：铁水：日均产量：中国（周）",
        "低硫主焦煤：价格指数：中国（月）",
        "冶金焦：全样本：独立焦化企业：日均产量：中国（周）",
        "焦炭：全样本：独立焦化企业：产能利用率：中国（周）",
        "Mysteel焦炭价格指数：综合平均价格指数：中国（月）",
        "高硫主焦煤：价格指数：中国（月）",
        "中硫主焦煤：价格指数：中国（月）",
        "1/3焦煤：价格指数：中国（月）",
        "300家钢铁企业：废钢：消耗量：中国（日）",
        "动力煤：进口数量合计：全球→中国（月）",
        "铁矿：进口：库存：45个港口（周）",
        "铁矿：进口：库存消费比：247家钢铁企业（周）",
        "焦炭：库存：全样本（周）",
        "精煤：样本洗煤厂（110家）：库存：中国（周）",
        "船板废钢：厚≥6mm：市场价：江阴（日）",
        "重废：厚≥6mm：市场价：江阴（日）",
        # 根据corr>0.8选择的
        "汽车:出口数量:累计值:同比",
        "南华期货:金属指数",
        "PPI:当月同比",
        "南华期货:黑色指数",
        "美国:ISM:制造业PMI:客户库存",
        "美国:ISM:制造业PMI:订单库存",
        "集装箱:出口数量:当月值",
        "销量:平地机:出口(外销):当月值",
        "汽车:出口数量:当月值:同比",
        "出口总值(美元计价):累计同比",
        "南华期货:有色金属指数",
        "焦炭：247家钢铁企业：库存：中国（周）",
        "出口总额(美元计价):美国:累计同比",
        "美国:零售和食品服务销售额:零售:总计",
        "销量:液压挖掘机:主要企业:出口(外销):当月同比",
        "南华期货:商品指数",
        "Mysteel焦炭价格指数：综合平均价格指数：焦炭（周）",
        "销量:平地机:出口(外销):当月同比",
        "Mysteel焦炭价格指数：综合绝对价格指数：焦炭（日）",
        "铁矿：进口数量：美国→中国（月）",
        "洗衣机销量：累计同比（月）",
        "金属集装箱:产量:累计同比",
        "动力煤：港口库存：京唐老港（日）",
        "洗衣机产量：累计同比（月）",
        "美国:供应管理协会(ISM):制造业PMI",
        "销量:液压挖掘机:主要企业:出口(外销):当月值",
        "集装箱:出口数量:当月同比",
        "美国:CPI:当月同比",
        "制造业PMI:主要原材料购进价格",
        "美国:耐用品:库存:季调",
        "手持船舶订单量:累计同比",
        "房屋竣工面积:累计同比",
        "美国:ISM:PMI:供应商交付",
        "美国:零售和食品服务销售额:总计:同比",
        "销量:压路机:出口(外销):当月值",
        "炼焦煤：价格指数：中国（月）",
        "金属集装箱:产量:当月同比",
    ]

    """价格标的序列"""
    # TARGET_NAME: str = '冷轧无取向硅钢：50WW1300：0.5*1200*C：市场价：等权平均（日）'
    # TARGET_NAME: str = '冷轧无取向硅钢：50WW1300：0.5*1200*C：市场价：等权平均（周）'
    # TARGET_NAME: str = '冷轧无取向硅钢：50WW1300：0.5*1200*C：市场价：等权平均（月）'
    # TARGET_NAME: str = '冷轧取向硅钢：30Q120：0.3*980*C：市场价：等权平均（日）'
    # TARGET_NAME: str = '冷轧取向硅钢：30Q120：0.3*980*C：市场价：等权平均（周）'
    # TARGET_NAME: str = '冷轧取向硅钢：30Q120：0.3*980*C：市场价：等权平均（月）'  # 标的平稳序列之后多项存在自相关，测试一下
    # TARGET_NAME: str = '冷卷：SPCC：1*1250*C：市场价：等权平均（日）'
    # TARGET_NAME: str = '冷卷：SPCC：1*1250*C：市场价：等权平均（周）'
    # TARGET_NAME: str = '冷卷：SPCC：1*1250*C：市场价：等权平均（月）'
    # TARGET_NAME: str = '热轧板卷：Q235B：5.75*1500*C：市场价：等权平均（日）'
    # TARGET_NAME: str = '热轧板卷：Q235B：5.75*1500*C：市场价：等权平均（周）'
    # TARGET_NAME: str = '热轧板卷：Q235B：5.75*1500*C：市场价：等权平均（月）'
    # TARGET_NAME: str = '热轧酸洗板卷：SPHC：3*1250*C：市场价：等权平均（日）'
    # TARGET_NAME: str = '热轧酸洗板卷：SPHC：3*1250*C：市场价：等权平均（周）'
    # TARGET_NAME: str = '热轧酸洗板卷：SPHC：3*1250*C：市场价：等权平均（月）'
    # TARGET_NAME: str = '中厚板：Q235B：20mm：价格指数：辽宁（周）'
    # TARGET_NAME: str = '中厚板：Q235B：20mm：价格指数：辽宁（月）'
    # TARGET_NAME: str = '螺纹钢：HRB400E：Φ20：汇总价格：上海（日）'
    # TARGET_NAME: str = '螺纹钢：HRB400E：Φ20：汇总价格：上海（周）'
    # TARGET_NAME: str = '螺纹钢：HRB400E：Φ20：汇总价格：上海（月）'
    # TARGET_NAME: str = '无取向电工钢：中低牌号：牌号等权平均：0.5*1200*C：出厂价：本钢集团（日）'
    # TARGET_NAME: str = '无取向电工钢：中低牌号：牌号等权平均：0.5*1200*C：出厂价：本钢集团（周）'
    # TARGET_NAME: str = '无取向电工钢：中低牌号：牌号等权平均：0.5*1200*C：出厂价：本钢集团（月）'
    # TARGET_NAME: str = '国际冷轧板卷汇总价格：中国市场（日）'
    # TARGET_NAME: str = '国际冷轧板卷汇总价格：中国市场（周）'
    # TARGET_NAME: str = '国际冷轧板卷汇总价格：中国市场（月）'
    # TARGET_NAME: str = '国际热轧板卷汇总价格：中国市场（日）'  # 日频测试
    TARGET_NAME: str = "国际热轧板卷汇总价格：中国市场（周）"  # 周频测试
    # TARGET_NAME: str = '国际热轧板卷汇总价格：中国市场（月）'  # 月频测试

    """date range"""
    # 因子分析 ANALYSE_Y_STARTDATE <= date < ANALYSE_Y_ENDDATE     #  注意：2个date最好是标的序列中有的日期；若不是，则会根据y_freq被统一到当周周五或当月第一天
    # 日频测试（日频意义不大）
    # ANALYSE_Y_STARTDATE: str = "2020-01-02"  # 对price序列的[y_start_date, y_end_date]期间的子序列进行相关性检测
    # ANALYSE_Y_ENDDATE: str = "2023-03-02"  # today期
    # PRE_START_DATE: str = '2023-03-03'  # analyse_y_enddate的下一期。实际预测的第一期是pre_start_date及之后的第一个按频度的有效的日子

    # 周频测试
    ANALYSE_Y_STARTDATE: str = "2019-01-01"  # 2016-01-01
    ANALYSE_Y_ENDDATE: str = "2022-06-02"
    PRE_START_DATE: str = (
        "2022-06-09"  # 周频测试 取值使analyse_y_enddate和pre_start_date在相邻的不同周
    )

    # 月频测试
    # ANALYSE_Y_STARTDATE: str = "2016-10-01"
    # ANALYSE_Y_ENDDATE: str = "2021-11-01"
    # PRE_START_DATE: str = '2021-12-01'  # 月频测试

    """预测期数"""
    ROLL_STEPS: int = 1  # 滚动测试的滚动步数
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
        EnumForecastMethod.TRANSFORMER_SINGLE: 9999,
        EnumForecastMethod.GRU_SINGLE: 9999,
        # 多因子模型
        EnumForecastMethod.LSTM_MULTIPLE: 9999,
        EnumForecastMethod.TRANSFORMER_MULTIPLE: 9999,
        EnumForecastMethod.GRU_MULTIPLE: 9999,
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
            mylog.error(
                f"参数ANALYSE_Y_STARTDATE={self.ANALYSE_Y_STARTDATE} 不存在于价格序列中，需要更换参数"
            )

        if not pd.to_datetime(self.ANALYSE_Y_ENDDATE) in origin_y_df.index:
            mylog.error(
                f"参数ANALYSE_Y_ENDDATE={self.ANALYSE_Y_ENDDATE} 不存在于价格序列中，需要更换参数"
            )

        if not pd.to_datetime(self.PRE_START_DATE) in origin_y_df.index:
            # 实际预测的第一期是pre_start_date及之后的第一个按频度的有效的日子
            mylog.warning(
                f"参数PRE_START_DATE={self.PRE_START_DATE} 不存在于价格序列中，实际第一个T+1预测日期将为下一个当前频度日期"
            )


ForecastConfig = FORECASTCONFIG()
