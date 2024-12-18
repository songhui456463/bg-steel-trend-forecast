"""
预测部分manager
"""

import copy
from typing import Optional, Dict

import pandas as pd

from factor.factor_config import FACTORCONFIG, FactorConfig
from factor.factor_enums import Enum_allfactor_lt_corr_res_DF
from factor.factor_manager import multifactor_align_index, multifactor_ayalysis
from factor.factor_resampling import check_freq
from forecasting.forecast_config import FORECASTCONFIG, ForecastConfig
from forecasting.forecast_manager import roll_forecast
from forecasting.results_assessment import (
    forecast_res_plot,
    forecast_evaluation,
)
from utils.data_read import read_x_by_map
from utils.enum_family import EnumFreq, EnumForecastMethod
from utils.log import mylog


class OneForecast:
    """一次forecast的数据"""

    def __init__(self):
        """准备序列"""
        # 标的和因子名称
        self.y_name: Optional[str] = None  # 标的序列的名称
        self.candi_xs_name: list = []  # 所有参与多因子分析的候选因子名称
        # 标的序列
        self.y_freq: Optional[EnumFreq] = None
        self.y_lt_spare_df = None  #  =.iloc[analyse_y_startdate_idx - max_lt_param_map.get(f.y_freq) - 5 : analyse_y_startdate_idx]
        self.y_train_df = (
            None  # .loc[analyse_y_startdate : analyse_y_end_date]
        )
        self.y_test_df = None  # .iloc[get_loc(analyse_y_end_date)+1 : get_loc(analyse_y_end_date)+1 + (roll_steps+pre_steps-1)]
        self.y_df = None  # 参与到多因子分析中的Price列，= y_lt_spare_df + y_analyse_df + y_test_df）
        self.y_history_df = (
            None  # 参与到预测模型中的标的序列，= y_train_df + y_test_df
        )
        # 所有候选因子序列（对齐y_df日期）
        self.resampling_xs_df = (
            None  # 以y_df为基准，重采样后，对齐日期索引的协从因子们
        )

        """因子分析"""
        # 所有候选因子的相关性分析结果（提前期、显著性）
        self.allfactors_lt_corr_df = None
        # 经过共线性分析筛选后的关键因子名称
        self.xs_name: list = (
            []
        )  # 多因子分析后，参与到多因子预测模型中的因子名称

        """预测"""
        # 根据因子分析结果提取xs序列
        self.all_history_df = None  # 参与到预测模型中的所有列的历史数据。columns=[price,f1,f2,...]
        self.forecastmethod_list: list = []
        # 预测结果
        self.method_realpredf_dict: Dict[str, pd.DataFrame] = (
            {}
        )  # 各预测方法的预测结果{'method_name':realpre_df}
        self.optimal_ws_dict: dict = {}  # 各预测方法的训练权重
        self.weighted_prerealT1df = None  # 真实值和各预测方法的T+1期预测值。形如columns={real,arima_pre_T+1,hw_pre_T+1, ..., weighted_pre_T+1 }

        """预测结果评估"""
        self.mae_dict = {}  # 各预测方法的T+1期预测值与真实值的mae
        self.mse_dict = {}
        self.mape_dict = {}
        self.mape_unfold_df = None


def run(
    ForecastConfig: FORECASTCONFIG,
    FactorConfig: FACTORCONFIG,
):
    """
    总运行
    :param ForecastConfig: 预测模块(forecasting)参数
    :param FactorConfig: 银子分析模块(factor)参数
    :return:
    """

    """创建预测对象"""
    f = OneForecast()
    f.y_name = ForecastConfig.TARGET_NAME
    f.candi_xs_name = ForecastConfig.CANDI_FACTORS_NAME

    """0-1 读取价格序列"""
    origin_y_df = read_x_by_map(factor_name=ForecastConfig.TARGET_NAME)
    # mylog.info(f'origin_y_df: \n{origin_y_df}')

    # 对输入端的参数进行规范（周频用周五，月频用月第一天）
    f.y_freq = check_freq(origin_y_df)
    ForecastConfig.harmonize_param_date(y_freq=f.y_freq)
    ForecastConfig.check_param_date_existence(origin_y_df=origin_y_df)

    # 截取三节 标的序列。
    analyse_y_startdate_idx = origin_y_df.index.get_loc(
        ForecastConfig.ANALYSE_Y_STARTDATE
    )
    analyse_y_enddate_idx = origin_y_df.index.get_loc(
        ForecastConfig.ANALYSE_Y_ENDDATE
    )
    # 截取三节 标的序列。按因子分析的开始日期(analyse_y_startdate)和要检查的最大提前期数(max_lag)来截取标的序列
    f.y_lt_spare_df = origin_y_df.iloc[
        analyse_y_startdate_idx
        - FactorConfig.MAX_LT_CHECK.get(f.y_freq)
        - 5 : analyse_y_startdate_idx
    ]  # 5是一个任取的‘安全库存’
    f.y_train_df = origin_y_df.loc[
        ForecastConfig.ANALYSE_Y_STARTDATE : ForecastConfig.ANALYSE_Y_ENDDATE
    ]
    f.y_test_df = origin_y_df.iloc[
        analyse_y_enddate_idx
        + 1 : analyse_y_enddate_idx
        + 1
        + (ForecastConfig.ROLL_STEPS + ForecastConfig.PRE_STEPS - 1)
    ]

    # 参与多因子分析的标的序列
    f.y_df = pd.concat([f.y_lt_spare_df, f.y_train_df, f.y_test_df], axis=0)
    # 参与预测过程的标的序列
    f.y_history_df = pd.concat([f.y_train_df, f.y_test_df], axis=0)

    """0-2 重采样协从因子序列"""
    resampling_xs_df = multifactor_align_index(
        y_df=f.y_df, candi_xs_name_list=ForecastConfig.CANDI_FACTORS_NAME
    )

    """1 多因子分析"""  # -> keyfactors_df
    _, allfactors_lt_corr_df, keyfactors_df = multifactor_ayalysis(
        y_df=f.y_df,
        xs_df=resampling_xs_df,
        # 对price序列的[analyse_y_startdate, analyse_y_enddate]期间的子序列进行相关性检测
        y_start=ForecastConfig.ANALYSE_Y_STARTDATE,
        y_end=ForecastConfig.ANALYSE_Y_ENDDATE,
        freq=check_freq(f.y_df),
    )
    mylog.info(f"keyfactors_df.columns:\n{keyfactors_df.columns}")

    """2 合并预测所需的历史数据：合并价格列和keyfactors列"""
    if not keyfactors_df.empty:
        # 从resampling_xs_df中按各因子的提前期重新取序列(长度与y_df一致)，拼接成价格和各关键因子的all_history_df
        f.all_history_df = copy.deepcopy(f.y_history_df)

        for factor in keyfactors_df.columns:
            # 当前因子的最好提前期
            best_lt = allfactors_lt_corr_df.loc[
                allfactors_lt_corr_df[
                    Enum_allfactor_lt_corr_res_DF.x_name.value
                ]
                == factor,
                Enum_allfactor_lt_corr_res_DF.best_lag.value,
            ].iloc[0]

            # 当前因子序列的开始date的idx
            temp_idx = (
                resampling_xs_df.index.get_loc(
                    ForecastConfig.ANALYSE_Y_STARTDATE
                )
                - best_lt
            )

            # 取出当前因子滞后best_lt的序列，长度和y_df一致，合并到all_history_df中
            f.all_history_df.loc[:, factor] = (
                resampling_xs_df[factor]
                .iloc[temp_idx : temp_idx + len(f.y_history_df)]
                .tolist()
            )
    else:
        f.all_history_df = copy.deepcopy(f.y_history_df)
    mylog.info(f"all_history_df:\n{f.all_history_df}")

    """3 单因子和多因子 预测"""
    # 3.1 决定预测方法：价格序列 时间特征分析
    # f.forecastmethod_list = run_pretesting(f.y_df)
    f.forecastmethod_list = [
        EnumForecastMethod.ARIMA,
        EnumForecastMethod.HOLTWINTERS,
        # EnumForecastMethod.VAR  # 注意：若数据量不够，不能var建模
    ]

    # 3.2 滚动预测
    (f.method_realpredf_dict, f.optimal_ws_dict, f.weighted_realpreT1df) = (
        roll_forecast(
            all_history_df=f.all_history_df,
            devp_pre_start_date=ForecastConfig.PRE_START_DATE,
            forecastmethod_list=f.forecastmethod_list,
            roll_steps=ForecastConfig.ROLL_STEPS,
            pre_steps=ForecastConfig.PRE_STEPS,
            is_save=True,
        )
    )
    # mylog.info(f'================= 所有方法预测完成的结果 method_realpredf_dict:')
    # for method, realpredf in f.method_realpredf_dict.items():
    #     mylog.info(f'method=【{method}】, realpredf:\n{realpredf}')
    #     pass

    # 3.3 T+1期预测值绘图和评估
    mylog.info(
        f"\n"
        f"============================================== 预测结果评估 =============================================="
    )
    # 绘图
    forecast_res_plot(
        f.weighted_realpreT1df,
        f.optimal_ws_dict,
        is_save=True,
        is_show=True,
    )
    # 评价指标
    f.mae_dict, f.mse_dict, f.mape_dict, f.mape_unfold_df = (
        forecast_evaluation(
            f.weighted_realpreT1df,
            is_save=True,
        )
    )
    mylog.info(f"mae_dict:\n{f.mae_dict}")
    mylog.info(f"mse_dict:\n{f.mse_dict}")
    mylog.info(f"mape_dict:\n{f.mape_dict}")
    mylog.info(f"mape_unfold_df:\n{f.mape_unfold_df}")


if __name__ == "__main__":
    run(ForecastConfig, FactorConfig)
    pass
