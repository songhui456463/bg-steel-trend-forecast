"""
预测
"""

import copy
import os
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from config.config import settings
from factor.factor_resampling import check_freq
from forecast_config import ForecastConfig
from forecasting.simplefit import (
    gaussian_modeling,
    gaussian_forecasting,
    t_dist_modeling,
    t_dist_forecasting,
    arch_forecasting,
    GarchInput,
)
from modeling_arima import arima_model
from modeling_holtwinters import (
    holtwinters_model_by_cv,
)
from modeling_var import var_forcast
from modeling_weight import (
    weight_optimization,
    weight_optimization_scipy,
    weight_forcast,
)
from results_assessment import forecast_res_plot
from utils.enum_family import (
    EnumForecastMethod,
    EnumForecastMethodType,
    EnumFreq,
)
from utils.log import mylog

# 设置显示的最大行数
pd.set_option("display.max_rows", 60)  # 设置为 10 行
# 设置显示的最大列数
pd.set_option("display.max_columns", 20)  # 设置为 10 列
# # 设置字体为支持中文的字体，如 SimHei（黑体）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 解决负号 '-' 显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False


def single_factor_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecastmethod_list: List[EnumForecastMethod],
    pre_steps: int = 6,
) -> pd.DataFrame:
    """
    一次预测，预测pre_steps步（即test_df的长度），判断使用哪种单变量的基础预测模型进行预测
    :param train_df: 价格序列，的历史数据
    :param test_df: 价格序列，要预测的period范围的真实值
    :param forecast_method_list:
    :return:
    """
    # 滚动预测的历史序列（滚动增加新的数据）
    history_df = copy.deepcopy(train_df)
    pre_steps = len(test_df)

    # 第一种结果保存：存放真实值和各基础模型的预测值
    real_pre_df = copy.deepcopy(
        test_df
    )  # 真实值列。devp_env:真实值列，prod_enc:空列(且无datetime index,只有price_columns_name)
    # real_pre_df[method.value + '_pre'] = np.nan  # 当前模型的预测值列
    pre_cols_name = [f"{method.value}_pre" for method in forecastmethod_list]
    real_pre_df[pre_cols_name] = np.nan  # 各模型的预测值列
    # 第二种结果保存
    pres_list = []

    # 进行pre_steps期预测
    for method in forecastmethod_list:
        if method.type != EnumForecastMethodType.SINGLE:
            continue
        if method == EnumForecastMethod.ARIMA:
            preroll_history_df = copy.deepcopy(history_df)
            arima_model_res = arima_model(preroll_history_df)  # arima 建模
            forecast_res = arima_model_res.forecast(
                steps=pre_steps
            )  # pd.series
            pres_list = forecast_res.tolist()
            real_pre_df.loc[:, f"{method.value}_pre"] = pres_list

        elif method == EnumForecastMethod.HOLTWINTERS:
            preroll_history_df = copy.deepcopy(history_df)
            # hw_order_tuple, hw_model_res = holtwinters_model_by_gridsearch(preroll_history_df)  # hw 建模
            hw_order_tuple, hw_model_res = holtwinters_model_by_cv(
                preroll_history_df
            )  # hw 建模
            forecast_res = hw_model_res.forecast(steps=pre_steps).reset_index(
                drop=True
            )  # pd.series
            pres_list = forecast_res.tolist()
            real_pre_df.loc[:, f"{method.value}_pre"] = pres_list
        else:
            mylog.warning(f"<method:{method.value}> method取值错误")
        # mylog.info(f'<method:{method.value}> 预测值保存 real_pre_df:\n{real_pre_df}')

    return real_pre_df


def multi_factor_forecast(
    train_xs_df: pd.DataFrame,
    test_df: pd.DataFrame,
    # real_pre_df: pd.DataFrame,
    forecastmethod_list: List[EnumForecastMethod],
    pre_steps: int = 6,
    cur_roll_r: int = -1,
):
    """
    多因子预测模型的滚动预测
    :param train_df: 多因子训练数据，第一列为价格序列，后面列为因子序列
    :param real_pre_df: 已经初始化了所有的forecast_method(包含单因子和多因子)，并含有单因子pre列的值
    :param forecastmethod_list:
    :return:
    """
    # 第一种结果保存：存放真实值和各基础模型的预测值
    real_pre_df = copy.deepcopy(
        test_df
    )  # 真实值列。devp_env:真实值列，prod_enc:空列(且无datetime index,只有price_columns_name)
    pre_cols_name = [f"{method.value}_pre" for method in forecastmethod_list]
    real_pre_df[pre_cols_name] = np.nan  # 各模型的预测值列
    # 第二种结果保存
    pres_list = []

    # 使用逐个模型进行预测（一次预测 预测出pre_steps期的预测值）
    for method in forecastmethod_list:
        if method.type != EnumForecastMethodType.MULTI:
            continue
        if method == EnumForecastMethod.VAR:
            pre_price_df = var_forcast(
                train_xs_df=train_xs_df,
                pre_steps=pre_steps,
                varmodel_update_freq=9999,
                is_ir_fevd=(cur_roll_r == 0),
            )  # pre_price_df: 没有datetime index的预测值df (单列)
            real_pre_df.loc[:, f"{method.value}_pre"] = pre_price_df.values
            pres_list = pre_price_df.iloc[:, 0].tolist()

        elif method == EnumForecastMethod.LSTM_MULTIPLE:
            pass
            preprice_df = None
            real_pre_df.loc[:, f"{method.value}_pre"] = pres_list

        else:
            mylog.warning(f"<method:{method.value}> method取值错误")
        # mylog.info(f'<method:{method.value}> 预测值保存 real_pre_df:\n{real_pre_df}')

    return real_pre_df


def simplefit_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecastmethod_list: List[EnumForecastMethod],
    pre_steps: int = 6,
) -> pd.DataFrame:
    """
    一次预测，预测pre_steps步（即test_df的长度），判断使用哪种单变量的基础预测模型进行预测
    :param train_df: 价格序列，的历史数据
    :param test_df: 价格序列，要预测的period范围的真实值
    :param forecast_method_list:
    :return:
    """
    # 第一种结果保存：存放真实值和各基础模型的预测值
    real_pre_df = copy.deepcopy(
        test_df
    )  # 真实值列。devp_env:真实值列，prod_enc:空列(且无datetime index,只有price_columns_name)
    pre_cols_name = [f"{method.value}_pre" for method in forecastmethod_list]
    real_pre_df[pre_cols_name] = np.nan  # 各模型的预测值列
    # 第二种结果保存
    pres_list = []

    # 使用逐个模型进行预测（一次预测 预测出pre_steps期的预测值）
    for method in forecastmethod_list:
        if method.type != EnumForecastMethodType.SIMPLE_FIT:
            continue
        if method == EnumForecastMethod.NORMAL_FIT:
            gaussian_model_res = gaussian_modeling(train_df)
            pres_list = gaussian_forecasting(
                param_mean=gaussian_model_res.get("mean_estimate"),
                param_variance=gaussian_model_res.get("variance_estimate"),
                pre_steps=pre_steps,
            )
            real_pre_df.loc[:, f"{method.value}_pre"] = pres_list

        elif method == EnumForecastMethod.T_FIT:
            t_model_res = t_dist_modeling(train_df)
            pres_list = t_dist_forecasting(
                param_df=t_model_res.get("df_estimate"),
                param_loc=t_model_res.get("loc_estimate"),
                param_variance=t_model_res.get("variance_estimate"),
                pre_steps=pre_steps,
            )
            real_pre_df.loc[:, f"{method.value}_pre"] = pres_list

        elif method == EnumForecastMethod.GARCH_FIT:
            pres_list = arch_forecasting(
                origin_df=train_df,
                pre_steps=pre_steps,
                garch_input=GarchInput.ORIGIN,
            )
            real_pre_df.loc[:, f"{method.value}_pre"] = pres_list

        else:
            mylog.warning(f"<method:{method.value}> method取值错误")
        # mylog.info(f'<method:{method.value}> 预测值保存 real_pre_df:\n{real_pre_df}')

    return real_pre_df


def roll_forecast(
    all_history_df: pd.DataFrame,
    devp_pre_start_date: str,
    forecastmethod_list: List[EnumForecastMethod],
    roll_steps: int,
    pre_steps: int = 6,
    is_save: bool = True,
):
    """
    :param all_history_df: 所有历史序列
    :param pre_start_date: 真正的预测第一天是 pre_start_date及之后的第一个按频度的期
    :param roll_steps: 滚动次数。满足 pre_start_date及之后的期数 >= len(最终的test_df) = roll_steps+pre_steps-1
    :param forecastmethod_list: 预测方法列表
    :param pre_steps: 每次roll中的预测期数
    :param is_save: 预测结果是否保存本地
    :return:
    """
    # 1 根据pre_start_date和roll_steps来划分出train和test，test_df可能有空值（即没有实际值）
    # 若all_history_df为价格单序列，则train_df和test_df为价格单序列；
    # 若all_history_df为价格和因子的多序列，则train_df和test_df为价格和因子的多序列；
    train_df, test_df = get_train_test_df_by_prestartdate_presteps(
        all_history_df, pre_start_date=devp_pre_start_date
    )
    # test_df长度=roll_steps + pre_steps -1
    if len(test_df) < roll_steps + pre_steps - 1:
        mylog.warning(
            f"<更新roll_steps> old roll_steps:{roll_steps} -> roll_steps:{len(test_df) - pre_steps + 1}"
        )
        roll_steps = len(test_df) - pre_steps + 1
    else:
        test_df = test_df.iloc[: roll_steps + pre_steps - 1, :]
    mylog.info(f"截取后的test_df:\n{test_df}")
    if roll_steps == 0:
        raise ValueError(f"实际的roll_steps=={roll_steps}")

    # 2 创建real_pre_df  # real_pre_df.columns=[real价格名称，'pre_T+1'，'pre_roll_1', ... , 'pre_roll_r',]
    real_pre_df = copy.deepcopy(test_df.iloc[:, [0]])
    real_pre_df["pre_T+1"] = np.nan  # 新增一列
    # mylog.info(f'real_pre_df:\n{real_pre_df}')

    # 3 每种method进行roll预测，每个roll预测pre_steps步
    method_realpredf_dict = {}  # {EnumForecastMethod: real_pre_df}
    for method in forecastmethod_list:
        print(f"\n")
        mylog.info(f"========== method:{method}")
        method_realpredf = copy.deepcopy(
            real_pre_df
        )  # 存放当前method的roll预测值
        # mylog.info(f'method_realpredf:\n{method_realpredf}')

        # train_df可能是单列（价格）或多列（价格和多因子）
        if method.type == EnumForecastMethodType.SINGLE:
            method_testdf = copy.deepcopy(
                test_df.iloc[:, [0]]
            )  # 价格真实值序列
            roll_history_df = copy.deepcopy(
                train_df.iloc[:, [0]]
            )  # 若当前是单因子模型，则历史数据只需要单列价格序列
        else:
            method_testdf = copy.deepcopy(
                test_df
            )  # 价格和因子们的真实值序列，用其中的真实值更新roll_history_df
            roll_history_df = copy.deepcopy(train_df)
        # mylog.info(f'method_testdf:\n{method_testdf}')
        # mylog.info(f'roll_history_df:\n{roll_history_df}')

        # 滚动roll_steps次进行预测
        for roll_r in range(roll_steps):
            mylog.info(f"------roll_r: {roll_r}")

            # 每次roll的预测开始日期
            rollr_prestartdate = method_realpredf.index[roll_r]
            roll_test_df = copy.deepcopy(
                method_realpredf.iloc[roll_r : roll_r + pre_steps, [0]]
            )
            mylog.info(
                f"<method={method.value}> <roll_r={roll_r}>  rollr_prestartdate: {rollr_prestartdate}"
            )
            # mylog.info(f'<method={method.value}> <roll_r={roll_r}> cur roll roll_history_df:\n{roll_history_df}')
            # mylog.info(f'<method={method.value}> <roll_r={roll_r}> cur roll roll_test_df:\n{roll_test_df}')

            # 一次预测，预测得到pre_steps期的预测值
            if method.type == EnumForecastMethodType.SINGLE:
                pre_value_df = single_factor_forecast(
                    train_df=roll_history_df,
                    test_df=roll_test_df,
                    forecastmethod_list=[method],
                    pre_steps=pre_steps,
                )
            elif method.type == EnumForecastMethodType.MULTI:
                pre_value_df = multi_factor_forecast(
                    train_xs_df=roll_history_df,
                    test_df=roll_test_df,
                    forecastmethod_list=[method],
                    pre_steps=pre_steps,
                    cur_roll_r=roll_r,
                )
            elif method.type == EnumForecastMethodType.SIMPLE_FIT:
                pre_value_df = simplefit_forecast(
                    train_df=roll_history_df,
                    test_df=roll_test_df,
                    forecastmethod_list=[method],
                    pre_steps=pre_steps,
                )
            else:
                mylog.warning(f"<method={method.value}> method.type 赋值错误")
            # mylog.info(f'<method={method.value}> <roll_r={roll_r}> pre_value_df:\n{pre_value_df}')

            # 记录预测结果
            # 'pre_T+1'列
            method_realpredf.loc[rollr_prestartdate, "pre_T+1"] = (
                pre_value_df.loc[rollr_prestartdate, f"{method.value}_pre"]
            )
            # f'pre_roll_{roll_r}'列
            pre_value_df.rename(
                columns={f"{method.value}_pre": f"pre_roll_{roll_r}"},
                inplace=True,
            )
            method_realpredf = pd.merge(
                left=method_realpredf,
                right=pre_value_df.loc[:, [f"pre_roll_{roll_r}"]],
                how="left",
                left_index=True,
                right_index=True,
            )
            # mylog.info(f'<method={method.value}> <roll_r={roll_r}> 预测结果保存 method_realpredf:\n{method_realpredf}')

            # 更新roll_history_dfd: 追加T+1期的真实值
            # real_cols = method_realpredf.columns[:method_realpredf.columns.get_loc('pre_T+1')].tolist()  # 价格列的列名 [pricecol_name]
            real_cols = test_df.columns.tolist()
            roll_history_df = pd.concat(
                [roll_history_df, method_testdf.loc[[rollr_prestartdate], :]],
                axis=0,
            )

        # 保存当前method的realpredf
        method_realpredf_dict[method.value] = method_realpredf

    # 4 合并所有模型的T+1预测结果。columns = [real, 'arima_pre_T+1', 'hw_pre_T+1']
    allmethod_realpreT1df = copy.deepcopy(test_df.iloc[:roll_steps, [0]])
    for method_value, method_realpredf in method_realpredf_dict.items():
        allmethod_realpreT1df.loc[:, f"{method_value}_pre_T+1"] = (
            method_realpredf.loc[:, ["pre_T+1"]]
        )
    # mylog.info(f'合并所有模型的T+1预测结果：allmethod_realpreT1df:\n{allmethod_realpreT1df}')

    # 5.1 使用有real值的期来训练权重
    nona_allmethod_realpre_df = allmethod_realpreT1df.dropna()
    # mylog.info(f'nona_allmethod_realpre_df:\n{nona_allmethod_realpre_df}')
    # optimal_ws_dict = weight_optimization(nona_allmethod_realpre_df)  # {'arima_pre_T+1': 1.0}
    optimal_ws_dict = weight_optimization_scipy(nona_allmethod_realpre_df)
    mylog.info(f"optimal_ws_dict: \n{optimal_ws_dict}")

    # 5.2 权重预测
    weighted_realpreT1df = weight_forcast(
        allmethod_realpreT1df, optimal_ws_dict
    )
    mylog.info(f"weighted_preT1_df:\n{weighted_realpreT1df}")

    # 6 预测结果保存本地
    if is_save:
        # 6.1
        middle_file_path = os.path.join(
            settings.OUTPUT_DIR_PATH,
            "[Total] (rollprocess)method_realpre_df_dict.xlsx",
        )
        for method_name, method_realpredf in method_realpredf_dict.items():
            sheet_name = method_name
            if not os.path.exists(middle_file_path):
                with pd.ExcelWriter(
                    middle_file_path, engine="openpyxl"
                ) as writer:
                    method_realpredf.to_excel(
                        writer, sheet_name=sheet_name, index=True
                    )
            else:
                with pd.ExcelWriter(
                    middle_file_path, mode="a", engine="openpyxl"
                ) as writer:
                    method_realpredf.to_excel(
                        writer, sheet_name=sheet_name, index=True
                    )
        mylog.info(
            f"[Total] (rollprocess)method_realpre_df_dict.xlsx 已保存本地!"
        )
        # 6.2
        final_file_path = os.path.join(
            settings.OUTPUT_DIR_PATH,
            "[Total] (rollfinal)weighted_realpreT1df.xlsx",
        )
        with pd.ExcelWriter(final_file_path, engine="openpyxl") as writer:
            weighted_realpreT1df.to_excel(
                writer, sheet_name=f"weighted_realpreT1df", index=True
            )
        with pd.ExcelWriter(
            final_file_path, mode="a", engine="openpyxl"
        ) as writer:
            pd.DataFrame([optimal_ws_dict]).to_excel(
                writer, sheet_name="optimal_ws"
            )
        mylog.info(f"[Total] (rollfinal)weighted_realpreT1df.xlsx 已保存本地!")

    # 7 T+1预测值绘图和评估
    # forecast_res_plot(weighted_realpreT1df, optimal_ws_dict)
    #
    # mae_dict, mse_dict = forecast_evaluation(weighted_realpreT1df,)
    # mylog.info(f'mae_dict:\n{mae_dict}')
    # mylog.info(f'mse_dict:\n{mse_dict}')

    return (
        method_realpredf_dict,  # 滚动预测过程记录
        optimal_ws_dict,
        weighted_realpreT1df,  # T+1期预测值的 训练权重和权重预测
    )


def run_forecast(
    all_history_df: pd.DataFrame,
    devp_pre_start_date: str,
    devp_pre_roll_steps: int,
    forecast_method_list: List[EnumForecastMethod],
):
    """
    预测总run
    :param all_history_df: 价格序列及所有因子序列，单因子 or 多因子
    :param devp_pre_start_date: devp_env的预测开始日期，不一定是真实的预测第一天日期（真实的预测第一天日期是'pre_start_date及之后'的第一个按频度的日子）
    :param devp_pre_roll_steps: devp_env预测多少期（取值要满足：devp_pre_start_date及之后的第一个按频度的期date + (devp_per_roll_steps - 1) <= all_history_df.index[-1]
    :param forecast_method_list:
    :return:
    """
    """devp_env pre : 历史数据预测一次，训练出预测模型的weight"""
    # 1 根据pre_start_date和roll_steps来划分出train和test，test_df可能有空值（即没有实际值）
    # 若all_history_df为价格单序列，则train_df和test_df为价格单序列；
    # 若all_history_df为价格和因子的多序列，则train_df和test_df为价格和因子的多序列；
    train_df, test_df = get_train_test_df_by_prestartdate_presteps(
        all_history_df,
        pre_start_date=devp_pre_start_date,
        steps=devp_pre_roll_steps,
    )

    # 2.1 单因子预测模型
    single_train_df = train_df.iloc[:, [0]]
    single_real_pre_df = single_factor_forecast(
        train_df=single_train_df,
        test_df=test_df.iloc[:, [0]],
        forecastmethod_list=forecast_method_list,
    )
    # 2.2 多因子预测模型
    multi_real_pre_df = multi_factor_forecast(
        train_xs_df=train_df,
        test_df=test_df,
        forecastmethod_list=forecast_method_list,
    )
    # 拼接两个real_pre_df
    real_pre_df = pd.merge(
        left=single_real_pre_df,
        right=multi_real_pre_df,
        how="outer",
        left_index=True,
        right_index=True,
    )

    # 2.3 训练 权重优化
    # 使用有real值的期来训练权重
    nona_real_pre_df = real_pre_df.dropna()
    optimal_ws_dict = weight_optimization(nona_real_pre_df)
    # optimal_ws_dict = weight_optimization_scipy(nona_real_pre_df)
    # 权重预测
    weighted_real_pre_df = weight_forcast(real_pre_df, optimal_ws_dict)

    # 3 结果展示和评估
    forecast_res_plot(weighted_real_pre_df, optimal_ws_dict)
    return None


def get_train_test_df_by_prestartdate_presteps(
    history_df: pd.DataFrame, pre_start_date: str, steps: int = None
):
    """
    根据想要预测的date范围，获取训练数据
    :param history_df: 单列价格df或多列df
    :param pre_start_date: 真实的预测开始日期是pre_start_date及之后的第一个按频度的日子
    :param steps: 要预测/滚动的步数
    :return:
    """
    # 参数
    price_df = history_df.iloc[:, [0]]
    freq = check_freq(price_df)
    if freq == EnumFreq.DAY:
        len_train = (
            ForecastConfig.LEN_TRAIN_DAY
        )  # 若为日频预测，取训练长度为三年工作日
    elif freq == EnumFreq.WEEK:
        len_train = ForecastConfig.LEN_TRAIN_WEEK  # 三年工作日
    elif freq == EnumFreq.MONTH:  # 月频容易训练数据量太少
        len_train = ForecastConfig.LEN_TRAIN_MONTH  # 三年工作日
        # len_train = 72
    else:
        raise ValueError(f"freq={freq}, 频度判断失败")

    # index是否有效
    if not is_datetime64_any_dtype(history_df.index):
        mylog.error(f"<df:{history_df.columns}> 不是datetime索引")
        raise AttributeError(f"<df:{history_df.columns}> 不是datetime索引")

    # 获取test_df：根据参数想要预测的date范围
    # test_df = price_df.loc[(pre_start_date <= price_df.index)]  # 单列，仅包含价格列
    test_df = history_df.loc[
        (pre_start_date <= price_df.index)
    ]  # 多列，包含价格列和因子列
    # if len(test_df) >= steps:  # 若test_df长度足够，则截取steps需要的长度
    #     test_df = test_df.iloc[:steps]
    # else:  # len(test_df) < pre_steps  # 若test_df长度不够，则steps缺少的长度由长度序号来补齐index，对应real值为空
    #     test_df_index = test_df.index.tolist()
    #     # mylog.info(f"test_df_index:\n{test_df_index}")
    #     while len(test_df_index) < steps:
    #         test_df_index.append(len(test_df_index) + 1)  # 原本的test_df长度小于steps，则test_df的index增加序号（对应test_df中没有real值）
    #     test_df = test_df.reindex(test_df_index)
    # # mylog.info(f'test_df:\n{test_df}')

    # 获取train_df
    test_start = test_df.index[
        0
    ]  # 注意不一定是pre_start_date，比如pre_start_date是周三，则test_start是pre_start_date之后的第一个周五
    mylog.info(f"test_df start_date: {test_start}")
    temp_df = history_df.loc[history_df.index < test_start]  # test_df的前一天
    train_df = get_train_df_by_current_date(
        history_df=temp_df, len_train=len_train, current_date=temp_df.index[-1]
    )

    mylog.info(f"train_df的数据量：{len(train_df)}")
    mylog.info(f"test_df的数据量：{len(test_df)}")
    mylog.info(f"train_df:\n{train_df}")
    mylog.info(f"test_df:\n{test_df}")
    return train_df, test_df


def get_train_df_by_current_date(
    history_df: pd.DataFrame, len_train: int, current_date: str = None
):
    """
    prod_env, 直接根据当前最新的日期，获取以前的历史数据
    :history_df: 单列df，dateindex
    :current_date: 所要获取的train_df的最后一个日子，即last_train_date
    :return:
    """
    if current_date is None:
        current_date = history_df.index[-1]
    current_date_idx = history_df.index.get_loc(current_date)
    # mylog.info(f'last_train_date: {current_date}, last_train_date_idx: {current_date_idx}')

    # 取train_df
    if current_date_idx + 1 >= len_train:
        # price_df中的历史数据足够
        train_df = history_df.iloc[
            current_date_idx - len_train + 1 : current_date_idx + 1
        ]  # iloc取值规则：[),loc取值规则：[]
        # 若current_date_idx + 1 超出index，则会取到底
    else:
        # price_df中的历史数据不够
        train_df = history_df.iloc[: current_date_idx + 1]

    return train_df


def get_train_test_df_by_prestartdate_presteps_from_db(
    price_df_name, current_date: str = None
):
    pass


def get_train_df_by_current_date_from_db(
    price_df_name, current_date: str = None
):
    pass


if __name__ == "__main__":
    # pass
    forecastmethod_list = [
        EnumForecastMethod.ARIMA,
        EnumForecastMethod.HOLTWINTERS,
        EnumForecastMethod.VAR,
    ]

    """test"""
    path = r"../data/钢材new.csv"
    usecols = [
        "日期",
        "热轧板卷4.75mm(raw)",
        "国产铁矿石价格指数",
        "铁矿62%Fe现货交易基准价",
        "焦炭综合绝对价格指数",
    ]

    history_df = pd.read_csv(path, usecols=usecols)
    history_df = history_df[usecols]  # 第一列必须是标的价格序列
    history_df.set_index(keys=["日期"], drop=True, inplace=True)
    history_df.index = pd.to_datetime(history_df.index)
    history_df.sort_index(inplace=True)
    # mylog.info(f"history_df: \n{history_df}")

    # 定义预测区间
    pre_start_date = "2024-08-22"
    roll_steps = 3  # 滚动次数
    pre_steps = 5  # 预测期数
    # 预测：单因子和多因子
    roll_forecast(
        all_history_df=history_df,
        devp_pre_start_date=pre_start_date,
        roll_steps=roll_steps,
        forecastmethod_list=forecastmethod_list,
        pre_steps=5,
    )
