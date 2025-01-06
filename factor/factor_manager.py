"""
多因子分析模块总run
"""

import copy
import numpy as np
import os.path
import pandas as pd
from enum import Enum

from config.config import settings
from factor.factor_colinearity_analysis import factor_colinearity_filter
from factor.factor_config import FactorConfig
from factor.factor_correlation_analysis import (
    calculate_max_abs_cov_for_factors,
    factor_correlation_filter,
)
from factor.factor_resampling import check_freq, auto_resampling
from utils.data_read import read_x_by_map
from utils.date_utils import generate_date_pairs
from utils.enum_family import EnumFreq
from utils.log import mylog

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def multifactor_align_index(
    y_df: pd.DataFrame,
    candi_xs_name_list: list,
):
    """
    将所有协从因子与价格序列的日期对齐，并将所有对齐的协从因子放到一个大xs_df
    :param y_df: 价格序列，其频度是想要预测的freq
    :param candi_xs_name_list:
    :param analyse_y_startdate:
    :param analyse_y_enddate:
    # :param max_lt_check: 提前期分析中的检查的最大提前期期数
    :return: 所有因子日期对齐的xs_df
    """
    price_freq = check_freq(y_df)

    # 依次对每个因子重采样，使与价格序列对齐
    resampling_xs_df = pd.DataFrame(index=y_df.index)

    for x_name in candi_xs_name_list:
        # 1 读取该因子
        unresampling_x_df = read_x_by_map(
            factor_name=x_name, start_date=None, end_date=None
        )
        try:
            # 2 重采样
            # 检查x_df的长度是否足够（比如y_df为2014-01-01至2024-01-01，但x_df只有2022-01-01以后的数据，则抛弃该因子）
            if y_df.index[0] < unresampling_x_df.index[0]:
                mylog.warning(
                    f"<{unresampling_x_df.columns[0]}> 该因子的历史数据不足，无法用于对当前标的序列y进行相关性分析: "
                    f'y_df:{y_df.index[0].strftime("%Y-%m-%d")}--{y_df.index[-1].strftime("%Y-%m-%d")}, '
                    f'x_df:{unresampling_x_df.index[0].strftime("%Y-%m-%d")}--{unresampling_x_df.index[-1].strftime("%Y-%m-%d")}'
                )
                raise ValueError
            # 重采样
            resampling_x_df = auto_resampling(
                base_df=y_df, todo_df=unresampling_x_df
            )
            # mylog.warning(f'<{resampling_x_df.columns[0]}> resampling_x_df:\n{resampling_x_df}')

            # 3 拼接到xs_df
            resampling_xs_df = pd.merge(
                left=resampling_xs_df,
                right=resampling_x_df,
                how="left",
                left_index=True,
                right_index=True,
            )

        except Exception as e:
            mylog.warning(
                f"<{unresampling_x_df.columns[0]}> 该因子重采样失败，忽略该因子"
            )

    # mylog.info(f"resampling_xs_df:\n{resampling_xs_df}")
    return resampling_xs_df


def multifactor_ayalysis(
    y_df: pd.DataFrame,
    xs_df: pd.DataFrame,
    y_start: str,
    y_end: str,
    freq: EnumFreq,
):
    """
    分析众多协从因子与标的因子的相关性和共线性，筛选出相关性高的且共线性低的协从因子
    :param y_df: 单列标的因子，index为dateindex,如'热轧汇总价格日频'.y_start和y_end需要存在于y_df当中
    :param xs_df: 要分析的多列协从因子，频度相同且已对齐dateindex，如均升采样为日频（频度与标的序列一致）
    :param y_start: 对价格序列分析的开始时间
    :param y_end: 对价格序列分析的结束时间
    :param freq: 价格序列的freq，（所有因子与价格序列的freq应该是一致的）
    :return: 可以直接input到多因子预测模型的 filted_xs_df
    """
    # 参数n: 最多检查多长滞后期的提前期相关性
    max_lt_check = FactorConfig.MAX_LT_CHECK.get(freq, 6)

    # 1 每个协从因子的相关提前期分析
    bestleadtime_result_df = calculate_max_abs_cov_for_factors(
        y_df, xs_df, t=y_start, T=y_end, n=max_lt_check
    )
    bestleadtime_result_df = bestleadtime_result_df[
        ["factor_name", "max_abs_cov", "best_lag"]
    ].set_index(["factor_name"])
    # mylog.info(f"相关提前期分析结果：\n{bestleadtime_result_df}")

    # 2 筛选显著相关因子
    # 2.1 取出每个协从因子的相应提前期的子序列 -> 没有行index的xs_df，但有列名
    lead_xs_df = pd.DataFrame(columns=xs_df.columns)
    start_idx = xs_df.index.get_loc(y_start)
    end_idx = xs_df.index.get_loc(y_end)
    for col_i in range(len(lead_xs_df.columns)):
        col = lead_xs_df.columns[col_i]
        # 当前因子的best提前期
        col_best_lag = bestleadtime_result_df.loc[col, "best_lag"]
        # 取出比price序列之后best_lag期的当前因子的子序列
        temp_idx = end_idx - col_best_lag + 1
        if temp_idx >= len(xs_df.index):
            lead_xs_df[col] = xs_df.iloc[
                (start_idx - col_best_lag) :, [col_i]
            ].reset_index(drop=True)
        else:
            lead_xs_df[col] = xs_df.iloc[
                (start_idx - col_best_lag) : temp_idx, [col_i]
            ].reset_index(drop=True)

    # 2.2 检验相关性是否显著，筛选出显著的因子
    sub_y_df = y_df.loc[y_start:y_end].reset_index(drop=True)
    all_factor_corr_res, corr_filted_xs_df = factor_correlation_filter(
        y_df=sub_y_df, xs_df=lead_xs_df
    )  # all_factor_corr_res中按corr列.abs降序
    # mylog.info(f'all_factor_corr_res:\n{all_factor_corr_res}')

    # 2.3 所有因子显著相关分析结果 拼接 提前期信息
    allfactor_lt_corr_res = pd.merge(
        all_factor_corr_res,
        bestleadtime_result_df[["best_lag"]],
        how="outer",
        left_on=["x_name"],
        right_index=True,
    ).sort_values(by=["p_value"])
    mylog.info(
        f"所有因子提前期及相关性显著分析结果 all_factor_lt_corr_res:\n{allfactor_lt_corr_res}"
    )
    os.makedirs(
        os.path.join(settings.OUTPUT_DIR_PATH, r"factor_analysis"),
        exist_ok=True,
    )
    allfactor_lt_corr_res.to_csv(
        os.path.join(
            settings.OUTPUT_DIR_PATH,
            r"factor_analysis\all_factor_lt_corr_res.csv",
        ),
        index=True,
        encoding="utf-8-sig",
    )

    # poc 场景4展示
    poc_factor_list_display_df = poc_factor_res_output(allfactor_lt_corr_res)
    # mylog.info(f"poc:场景4结果输出：\n{poc_factor_list_display_df}")

    # 3 共线性分析，筛选出代表性因子
    colinear_filted_xs_df = factor_colinearity_filter(
        xs_df=corr_filted_xs_df,
        n_clusters=FactorConfig.N_CLUSTERS,
        vif_thred=FactorConfig.VIF_THRED,
        vif_max_cycle=FactorConfig.VIF_MAX_CYCLE,
    )
    # colinear_filted_xs_df = None
    keyfactor_bestleadtime_result_df = bestleadtime_result_df.loc[
        colinear_filted_xs_df.columns.values.tolist()
    ]
    # mylog.info(f'keyfactor_bestleadtime_result_df:\n{keyfactor_bestleadtime_result_df}')

    # colinear_filted_xs_df可以直接参与到多因子预测模型中
    # return poc_factor_list_display_df, keyfactor_bestleadtime_result_df, colinear_filted_xs_df
    return (
        poc_factor_list_display_df,
        allfactor_lt_corr_res,
        colinear_filted_xs_df,
    )


def poc_factor_res_output(allfactor_lt_corr_res_df: pd.DataFrame):
    """
    场景4：价格影响因素列表 展示：因子名称，因子所属范畴（宏观面/基本面/技术面），相关强度度量，提前期数
    :param allfactor_lt_corr_res_df: 经过提前期和相关显著性分析的结果大表
    :return:
    """
    # 读取因子分类df
    factor_category_df = pd.read_csv(r"../data/factor_category.csv").set_index(
        keys=["factor_name"]
    )
    # 读取提前期和相关性结果
    display_df = allfactor_lt_corr_res_df.loc[
        :,
        [
            # Enum_allfactor_lt_corr_res_DF.y_name.value,
            # Enum_allfactor_lt_corr_res_DF.x_name.value,
            # Enum_allfactor_lt_corr_res_DF.best_lag.value,
            # Enum_allfactor_lt_corr_res_DF.corr.value,
            # Enum_allfactor_lt_corr_res_DF.p_value.value,
            "y_name",
            "x_name",
            "best_lag",
            "corr",
            # 'p_value'
        ],
    ]
    # 按照相关性强度进行排序
    display_df = display_df.sort_values(by="corr", ascending=False)
    display_df = pd.merge(
        display_df,
        factor_category_df,
        how="left",
        left_on="x_name",
        right_index=True,
    )  # 拼接因子所属分类
    display_df.index = pd.RangeIndex(start=1, stop=len(display_df) + 1, step=1)
    # display效果设置
    col_seq = [
        "y_name",
        "x_name",
        "category",
        "best_lag",
        "corr",
        # 'p_value'
    ]
    display_df = display_df[col_seq]  # 顺序
    display_df = display_df.rename(
        columns={
            "y_name": "标的名称",
            "x_name": "因子名称",
            "category": "因子范畴",
            "best_lag": "因子提前期",
            "corr": "相关强度",
            # 'p_value': '因子相关显著性(p_value)'
        }
    )
    return display_df


def poc_multi_cycle_validation():
    origin_df = pd.read_csv(
        r"../data/02六品种数据整理-月_test.csv", index_col=["date"]
    )
    origin_df.index = pd.to_datetime(origin_df.index)
    y_df = origin_df[["热轧板卷价格"]]
    xs_df = origin_df.drop(columns=["热轧板卷价格"])
    # print(y_df)
    # print(xs_df)
    resampling_xs_df = multifactor_align_index(y_df, xs_df)

    acc = []
    start_date = "2022-01-01"
    frequency = "month"
    periods = 10
    interval = 16
    multi_cycle_date = generate_date_pairs(
        start_date, frequency, periods, interval
    )
    print(multi_cycle_date)
    previous_factors_tmp = None

    # multi_cycle_date = [("2022-01-01", "2024-03-01"), ("2022-02-01", "2024-04-01"), ]
    for y_start_date, y_end_date in multi_cycle_date:
        mylog.info(f"正在分析[{y_start_date}~{y_end_date}]期间的因子")
        # 注意：y_start_date和y_end_date的选择必须在目标序列的dateindex内
        # y_start_date = "2022-01-01"  # 对price序列的[y_start_date, y_end_date)期间的子序列进行相关性检测
        # y_end_date = "2024-03-01"  # y_df的第一个date差不多离y_start_date至少半年(根据提前期检测的最大滞后期数来判断)
        # y_start的设置必须在所有因子都有值以后，比如价格序列有近10年的，而因子1只有近3年的，因子2只有近5年的
        # 则y_start要晚于 “ 因子1最早时间 + price_freq下的提前期最长回溯长度 * price_freq ”
        result, _, _ = multifactor_ayalysis(
            y_df,
            resampling_xs_df,
            y_start=y_start_date,
            y_end=y_end_date,
            freq=check_freq(y_df),
        )
        factors_tmp = result.iloc[:10,]
        mylog.info(
            f"本期[{y_start_date}~{y_end_date}] Top10因子:\n{factors_tmp}"
        )
        if previous_factors_tmp is None:
            previous_factors_tmp = factors_tmp
        else:
            # 找出交集（重叠的因子名称）
            common_factors = set(factors_tmp["因子名称"]).intersection(
                set(previous_factors_tmp["因子名称"])
            )
            # 计算重叠百分比
            overlap_percentage = (
                len(common_factors) / len(factors_tmp["因子名称"])
            ) * 100
            acc.append(overlap_percentage)
            previous_factors_tmp = factors_tmp
    mylog.info(f"因子周期: {multi_cycle_date}")
    mylog.info(f"因子重叠度: {[f'{x:.3f}%' for x in acc]}")
    matched = [x for x in acc if x >= 70]
    mylog.info(
        f"最终指标: \n"
        f"指标1：{np.mean(acc):.3f}%, 指标2:{len(matched)/len(acc)*100:.3f}%\n"
        f"指标说明:\n"
        f"\t指标1: 每次重叠因子个数除以上一期因子个数, 最终求算术平均值\n"
        f"\t指标2: 每次重叠度>=70%则为成功一次"
    )


if __name__ == "__main__":
    origin_df = pd.read_csv(
        r"../data/02六品种数据整理-月_test.csv", index_col=["date"]
    )
    origin_df.index = pd.to_datetime(origin_df.index)
    y_df = origin_df[["热轧板卷价格"]]  # 标的价格序列
    xs_df = origin_df.drop(columns=["热轧板卷价格"])  # 所有因子序列
    # print(y_df)
    # print(xs_df)

    # 1 所有因子对齐price序列
    resampling_xs_df = multifactor_align_index(y_df, xs_df)
    # 2.1 多因子分析测试：T期
    # 注意：y_start_date和y_end_date的选择必须在目标序列的dateindex内
    y_start_date = "2022-01-01"  # 对price序列的[y_start_date, y_end_date)期间的子序列进行相关性检测
    y_end_date = "2024-03-01"  # y_df的第一个date差不多离y_start_date至少半年(根据提前期检测的最大滞后期数来判断)
    # y_start的设置必须在所有因子都有值以后，比如价格序列有近10年的，而因子1只有近3年的，因子2只有近5年的
    # 则y_start要晚于 “ 因子1最早时间 + price_freq下的提前期最长回溯长度 * price_freq ”
    result_tuple = multifactor_ayalysis(
        y_df,
        resampling_xs_df,
        y_start=y_start_date,
        y_end=y_end_date,
        freq=check_freq(y_df),
    )
    # result_tuple[0].to_csv(r'../outputs/poc4_result_2022-01-01--2024-03-01.csv', encoding='UTF-8-SIG')
    # # 需要用到各关键因子，输入到多因子预测模型中时，可以根据keyfactor_bestleadtime_result_df中的best_lag重新取序列，也可以直接用filted_xs_df
    #
    # print(
    #     f"==================================================================================="
    # )
    # # 2.2多因子分析测试：T+1期
    # y2_start_date = "2022-02-01"  # 对price序列的[y_start_date, y_end_date)期间的子序列进行相关性检测
    # y2_end_date = "2024-04-01"  # y_df的第一个date差不多离y_start_date至少半年
    # # 多因子分析
    # result_tuple2 = multifactor_ayalysis(
    #     y_df,
    #     resampling_xs_df,
    #     y_start=y2_start_date,
    #     y_end=y2_end_date,
    #     freq=check_freq(y_df),
    # )
    # result_tuple2[0].to_csv(r'../outputs/poc4_result_2022-02-01--2024-04-01.csv', encoding='UTF-8-SIG')

    # poc_multi_cycle_validation()
