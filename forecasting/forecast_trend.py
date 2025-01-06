"""
基于预测数值来计算trend预测
与T期相比: 涨/跌, 涨跌幅度, 这个 涨/跌 以及涨跌幅度 的概率  (用各个模型的权重来表示概率,相当于投票票数)
"""

import copy
import numpy as np
import os
import pandas as pd
from typing import Dict

from config.config import settings
from utils.log import mylog


def cal_trend_by_value(
    last_real_df,
    method_realpredf_dict: Dict[str, pd.DataFrame],
    optimal_ws_dict: Dict[str, float],
    is_save: bool = True,
):
    """
    计算相邻日期的trend和相应的prob
    :param last_real_df:
    :param method_realpredf_dict:
    :param optimal_ws_dict:
    :return:
    """
    # mylog.info(f'\n*******************************************************n*****************************************')
    # 1 计算实际趋势
    # 拼接最后一行实际价格
    arbi_realpredf = next(iter(method_realpredf_dict.values()))
    test_df = copy.deepcopy(
        arbi_realpredf.iloc[:, [0]]
    )  # index为日期，columns=[real_trend, roll_0_pre_trend, roll_0_pre_trend_prob,  roll_1_pre_trend, roll_1_pre_trend_prob, ...]
    concat_real_df = pd.concat([last_real_df, test_df], axis=0)

    price_name = concat_real_df.columns[0]
    roll_steps = arbi_realpredf.shape[1] - 2
    pre_steps = arbi_realpredf.iloc[:, [-1]].dropna().shape[0]

    # 计算实际趋势 = (T+1_pre - T_real）/ T_real
    concat_real_df[f"trend_{price_name}"] = (
        concat_real_df[price_name] - concat_real_df[price_name].shift(1)
    ) / concat_real_df[price_name].shift(1)

    # 2 计算各method的预测趋势
    trend_method_realpredf_dict = {}  # {method.value: trend_realpredf, }
    for method_name, realpredf in method_realpredf_dict.items():
        # mylog.warning(f'method_name: {method_name}')
        trend_realpredf = copy.deepcopy(concat_real_df)

        # 计算每次roll的预测趋势
        for col_i in range(2, realpredf.shape[1]):
            col_name = realpredf.columns[col_i]
            trend_realpredf[f"trend_{col_name}"] = (
                realpredf[col_name] - trend_realpredf[price_name].shift(1)
            ) / trend_realpredf[price_name].shift(1)

        # 删除非趋势实际值列 及 第一行
        trend_realpredf.drop(columns=[price_name], inplace=True)
        trend_realpredf.dropna(how="all", inplace=True)

        # 记录
        trend_method_realpredf_dict[method_name] = trend_realpredf

    # 3 计算加权预测趋势
    arbi_trend_realpredf = next(iter(trend_method_realpredf_dict.values()))
    trend_weighted_realpredf = copy.deepcopy(
        arbi_trend_realpredf.iloc[:, [0]]
    )  # 实际趋势列

    # trend_weighted_realpredf.columns = [trend_real, trend_pre_roll_0, prob_trend_pre_roll_0,  trend_pre_roll_1, prob_trend_pre_roll_1, ...]
    for trend_pre_roll_r in arbi_trend_realpredf.columns[1:]:
        # mylog.info(f'------trend_pre_roll_r:{trend_pre_roll_r}')

        # 新增当前roll_r的两列
        trend_weighted_realpredf[
            [
                trend_pre_roll_r,
                f"state_{trend_pre_roll_r}",
                f"prob_{trend_pre_roll_r}",
            ]
        ] = np.nan

        # 计算每一个pre_date的加权预测趋势
        pre_dates = arbi_trend_realpredf.index[
            arbi_trend_realpredf[trend_pre_roll_r].notna()
        ]
        for date in pre_dates:
            up_dict = {}  # {method_name: trend}
            down_dict = {}

            for (
                method_name,
                trend_realpredf,
            ) in trend_method_realpredf_dict.items():
                trend_value = trend_realpredf.loc[date, trend_pre_roll_r]
                if trend_value >= 0:  # 涨趋势
                    up_dict[method_name] = trend_value
                else:  # 跌趋势
                    down_dict[method_name] = trend_value

            if up_dict:
                up_total_weight = sum(
                    [
                        optimal_ws_dict[f"{method_name}_pre_T+1"]
                        for method_name in up_dict.keys()
                    ]
                )
                up_dict[f"weighted_trend"] = sum(
                    [
                        up_dict[method_name]
                        * optimal_ws_dict[f"{method_name}_pre_T+1"]
                        / up_total_weight
                        for method_name in up_dict.keys()
                    ]
                )
            else:
                up_total_weight = 0
                up_dict[f"weighted_trend"] = None
            if down_dict:
                down_total_weight = sum(
                    [
                        optimal_ws_dict[f"{method_name}_pre_T+1"]
                        for method_name in down_dict.keys()
                    ]
                )
                down_dict[f"weighted_trend"] = sum(
                    [
                        down_dict[method_name]
                        * optimal_ws_dict[f"{method_name}_pre_T+1"]
                        / down_total_weight
                        for method_name in down_dict.keys()
                    ]
                )
            else:
                down_total_weight = 0
                down_dict[f"weighted_trend"] = None

            # 选择>50%的趋势作为显式趋势预测值
            if up_total_weight >= down_total_weight:
                if (up_dict[f"weighted_trend"] > -0.02) and (
                    up_dict[f"weighted_trend"] < 0.02
                ):
                    trend_state = "平稳"
                elif up_dict[f"weighted_trend"] > 0.02:
                    trend_state = "涨"  # >0.02,
                else:
                    trend_state = "跌"  # <-0.02
                trend_weighted_realpredf.loc[date, trend_pre_roll_r] = up_dict[
                    f"weighted_trend"
                ]  # 加权趋势值
                trend_weighted_realpredf.loc[
                    date, f"state_{trend_pre_roll_r}"
                ] = trend_state  # 趋势状态
                trend_weighted_realpredf.loc[
                    date, f"prob_{trend_pre_roll_r}"
                ] = up_total_weight  # 当前趋势值的概率
            else:
                if (down_dict[f"weighted_trend"] > -0.02) and (
                    down_dict[f"weighted_trend"] < 0.02
                ):
                    trend_state = "平稳"
                elif down_dict[f"weighted_trend"] >= 0.02:
                    trend_state = "涨"  # >0.02,
                else:
                    trend_state = "跌"  # <-0.02
                trend_weighted_realpredf.loc[date, trend_pre_roll_r] = (
                    down_dict[f"weighted_trend"]
                )
                trend_weighted_realpredf.loc[
                    date, f"state_{trend_pre_roll_r}"
                ] = trend_state  # 趋势状态
                trend_weighted_realpredf.loc[
                    date, f"prob_{trend_pre_roll_r}"
                ] = down_total_weight

    # mylog.info(f'trend_weighted_realpredf:\n{trend_weighted_realpredf}')

    if is_save:
        # 1 计算trend的中间过程：每个method的趋势计算
        middle_file_path = os.path.join(
            settings.OUTPUT_DIR_PATH,
            "[TotalTrend] (rollprocess)trend_method_realpredf_dict.xlsx",
        )
        for (
            method_name,
            trend_realpredf,
        ) in trend_method_realpredf_dict.items():
            sheet_name = method_name
            if not os.path.exists(middle_file_path):
                with pd.ExcelWriter(
                    middle_file_path, engine="openpyxl"
                ) as writer:
                    trend_realpredf.to_excel(
                        writer, sheet_name=sheet_name, index=True
                    )
            else:
                with pd.ExcelWriter(
                    middle_file_path, mode="a", engine="openpyxl"
                ) as writer:
                    trend_realpredf.to_excel(
                        writer, sheet_name=sheet_name, index=True
                    )
        mylog.info(
            f"[TotalTrend] (rollprocess)trend_method_realpredf_dict.xlsx 已保存本地!"
        )
        # 2 计算
        final_file_path = os.path.join(
            settings.OUTPUT_DIR_PATH,
            "[Total] (rollfinal)trend_weighted_realpredf.xlsx",
        )
        with pd.ExcelWriter(final_file_path, engine="openpyxl") as writer:
            trend_weighted_realpredf.to_excel(writer, index=True)
        mylog.info(
            f"[TotalTrend] (rollfinal)trend_weighted_realpredf.xlsx 已保存本地!"
        )

    return trend_weighted_realpredf


def trend_demo(trend_weighted_realpredf):
    """
    roll=1时，趋势预测值展示
    :param trend_weighted_realpredf:
    :return:
    """
    display_trend_weighted_realpredf = copy.deepcopy(trend_weighted_realpredf)
    old_columns = display_trend_weighted_realpredf.columns
    if len(old_columns) == 3:
        display_trend_weighted_realpredf.rename(
            columns={
                old_columns[0]: old_columns[0].replace("trend_", ""),
                old_columns[1]: "预测涨跌幅度",
                old_columns[2]: "预测涨跌幅度（概率）",
            },
            inplace=True,
        )
        # 保存本地
        display_file_path = os.path.join(
            settings.OUTPUT_DIR_PATH, "[TotalTrend] 趋势预测结果.xlsx"
        )
        with pd.ExcelWriter(display_file_path, engine="openpyxl") as writer:
            display_trend_weighted_realpredf.to_excel(writer, index=True)
        mylog.info(f"[TotalTrend] 趋势预测结果.xlsx 已保存本地!")

    else:
        mylog.warning(f"当前预测结果为滚动预测结果，不直接展示趋势计算！")
    return display_trend_weighted_realpredf


def cal_trend_by_weighted_value(
    last_real_df,
    method_realpredf_dict: Dict[str, pd.DataFrame],
    optimal_ws_dict: Dict[str, float],
    is_save: bool = True,
):
    """
    根据weighted预测值来计算预测趋势：涨/跌/平稳
    :param last_real_df:
    :param method_realpredf_dict:
    :param optimal_ws_dict:
    :param is_save:
    :return:
    """
    # mylog.info(f'\n*********************************************************************************************************')

    # 1 根据各method的预测结果计算weighted预测结果
    weighted_realpredf = sum(
        optimal_ws_dict[f"{method_name}_pre_T+1"] * realpredf
        for method_name, realpredf in method_realpredf_dict.items()
    )
    weighted_realpredf.drop(columns=[f"pre_T+1"], inplace=True)

    # 2 拼接最后一行实际价格
    concat_realpredf = pd.concat([last_real_df, weighted_realpredf], axis=0)

    trend_realpredf = copy.deepcopy(
        concat_realpredf.iloc[:, [0]]
    )  # columns=[price_name, f'trend_{price_name}', f'trend_{pre_roll_r}', f'trendstate_{pre_roll_r}', ]
    price_name = concat_realpredf.columns[0]

    # 3 根据实际价格计算实际趋势（作为对照）
    trend_realpredf[f"trend_{price_name}"] = (
        concat_realpredf[price_name] - concat_realpredf[price_name].shift(1)
    ) / concat_realpredf[price_name].shift(1)

    # 4 计算各次roll中pre_steps步的weighted预测趋势
    for pre_roll_r in concat_realpredf.columns[1:]:
        # mylog.info(f'------ pre_roll_r:{pre_roll_r}')

        # 每个roll新增两列
        # trend_realpredf[[f'trend_{pre_roll_r}', f'trendstate_{pre_roll_r}']] = np.nan

        # 计算当前roll中每个predate的预测趋势
        pre_dates = concat_realpredf.index[
            concat_realpredf[pre_roll_r].notna()
        ]
        # base_date = concat_realpredf.index[concat_realpredf.index.get_loc(pre_dates[0]) - 1]  # 当前predates们的趋势计算的真实值basedate

        for date in pre_dates:
            date_idx = concat_realpredf.index.get_loc(date)

            # 找出计算趋势的basedate
            base_date = concat_realpredf.index[date_idx - 1]

            # 计算趋势
            date_trend = (
                concat_realpredf.loc[date, pre_roll_r]
                - concat_realpredf.loc[base_date, price_name]
            ) / concat_realpredf.loc[base_date, price_name]
            trend_realpredf.loc[date, f"trend_{pre_roll_r}"] = date_trend

            # 判断趋势状态
            if (0.02 >= date_trend) and (
                date_trend >= -0.02
            ):
                date_trendstate = "平稳"  # （0.02,-0.02）
            elif date_trend > 0.02:
                date_trendstate = "涨"  # >0.02,
            else:
                date_trendstate = "跌"  # <-0.02
            trend_realpredf.loc[date, f"trendstate_{pre_roll_r}"] = (
                date_trendstate
            )

    # 去除第一列（价格真实值列）以及拼接的首行
    trend_realpredf.drop(columns=[price_name], inplace=True)
    trend_realpredf.dropna(how="all", inplace=True)

    # mylog.info(f'222222 trend_realpredf: \n{trend_realpredf}')

    # 保存
    if is_save:
        final_file_path = os.path.join(
            settings.OUTPUT_DIR_PATH,
            "[Total] (rollfinal-roll0)trend_realpredf.xlsx",
        )
        with pd.ExcelWriter(final_file_path, engine="openpyxl") as writer:
            trend_realpredf.to_excel(writer, index=True)
        mylog.info(
            f"[TotalTrend] (rollfinal-roll0)trend_realpredf.xlsx 已保存本地!"
        )

    return trend_realpredf
