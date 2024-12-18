"""
检测序列缺失值，输出缺失值的描述性统计指标
"""

import copy
from itertools import groupby
from typing import Union

import numpy as np
import pandas as pd

from preconfig import PreConfig
from utils.log import mylog

# pd.set_option('display.precision', 4)  # 设置浮点数精度
pd.set_option("display.float_format", "{:,.4f}".format)  # 右对齐浮点数


def isinstance_int(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    elif np.issubdtype(type(value), np.integer):
        return True
    else:
        return False


def isinstance_float(value):
    if isinstance(value, float):
        return True
    elif np.issubdtype(type(value), np.floating):
        return True
    else:
        return False


def isinstance_int_float(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    elif np.issubdtype(type(value), np.floating) or np.issubdtype(
        type(value), np.integer
    ):
        return True
    else:
        return False


def missing_type_mapping(value):
    if pd.isna(value):
        return "nan"  # np.nan,None,其他nan类型
    elif isinstance_int(value):  # 不包含bool
        return "int"  # 认为不是缺失值
    elif isinstance_float(value):
        return "float"  # 认为不是缺失值
    elif isinstance(value, str):
        return "str"
    elif isinstance(value, bool):
        return "bool"
    else:
        return "other"


def check_missing(
    origin_df: pd.DataFrame,
    check_start: Union[str, pd.DatetimeIndex],
    check_end: Union[str, pd.DatetimeIndex],
    # check_missing_params: Optional[dict] = None,
    preconfig: PreConfig,
):
    """
    检查并统计df中start_date-end_date之间的子序列的缺失值信息
    :param origin_df: 单列
    :param check_start: 检查缺失值的起始日期（包含）
    :param check_end: 检查缺失值的终止日期（包含）
    :return: 缺失值统计信息，以及数值化转换后的新的df（只有：float类型、空值）
    """
    # 准备参数
    # if check_missing_params is None:
    #     check_missing_params = {}
    check_missing_params = preconfig.CHECK_MISSING
    repairable_ratio = check_missing_params.get("repairable_ratio", 0.6)
    repairable_longest_consec_ratio = check_missing_params.get(
        "repairable_longest_consec_ratio", 0.3
    )
    is_print = check_missing_params.get("is_print", True)

    y_name = origin_df.columns[0]
    # 1 截取子序列
    mylog.info(f"origin_df:\n{origin_df}")
    mylog.info(f"check_start:\n{check_start}")
    mylog.info(f"check_end:\n{check_end}")
    df = copy.deepcopy(
        origin_df.loc[
            (origin_df.index >= check_start) & (origin_df.index <= check_end)
        ]
    )
    # mylog.info(f'df:\n{df}')

    # 2 检查原始序列的各类缺失值
    # 找出各类缺失值
    miss_mask = df.iloc[:, 0].isnull() | df.iloc[:, 0].apply(
        lambda x: not isinstance_int_float(x)
    )
    df_miss_mask = copy.deepcopy(df)
    df_miss_mask["is_miss_mask"] = (
        miss_mask  # 原始子序列数据df的基础上，加一列miss_mask
    )
    # 缺失值所在位置
    miss_dates = df.index[miss_mask]
    # 缺失值总占比
    miss_total_ratio = len(miss_dates) / len(df)
    # 检查缺失值类型
    all_types = (
        df.iloc[:, 0].apply(lambda x: missing_type_mapping(x)).to_frame()
    )  # 所有位置的类型
    all_types.columns = ["missing_type"]
    miss_types = (
        df.iloc[:, 0]
        .apply(lambda x: missing_type_mapping(x))[miss_mask]
        .to_frame()
    )  # 仅包含缺失位置的缺失类型
    miss_types.columns = ["missing_type"]
    # 各类型的数量及比例
    type_counts = miss_types.value_counts().to_frame()
    type_ratios = type_counts / len(df)
    type_counts_ratios = pd.concat(
        [type_counts, type_ratios], axis=1
    ).rename_axis("miss_type")
    type_counts_ratios.columns = ["counts", "ratios"]
    # 最长连续缺失数量
    longest_consecutive_missing = max(
        (len(list(group)) for k, group in groupby(miss_mask) if k), default=0
    )

    def convert_to_numeric(value):
        if pd.isna(value):  # 首先排除float类型的np.nan
            return None  # 所有nan类型，None,np.nan，..
        if isinstance_int_float(value):
            return value  # 非bool的int类型，float类型
        elif isinstance(value, str):
            try:
                return (
                    float(value) if "." in value else int(value)
                )  # 可转为int或float的str
            except (ValueError, TypeError):
                return None  # 不可转为int或float的str
        else:
            return None  # 其他类型，比如bool

    # 3 [repair]尽可能将str表示的数字转为数值型数据，不能转换的数据（None,其他str等）转为nan
    numerical_df = copy.deepcopy(df)
    # numerical_df.iloc[:, 0] = pd.to_numeric(numerical_df.iloc[:, 0], errors='coerce')  # int会全部变成float型
    numerical_df[y_name] = numerical_df[y_name].apply(convert_to_numeric)
    mylog.info(f"numerical_df:\n{numerical_df}")

    # 4 找出数值化后的序列有效子区间
    numerical_miss_mask = numerical_df[
        y_name
    ].isnull()  # nan位置为True，非nan位置为False
    # 子序列数据numerical_df的基础上，加一列miss_mask
    numerical_df_miss_mask = copy.deepcopy(numerical_df)
    numerical_df_miss_mask["is_miss_mask"] = (
        numerical_miss_mask  # nan位置为True，非nan位置为False
    )
    # 更新数值化之后的有效start_date和有效end_date(两个方向不为空的首个位置)
    valid_date_index = numerical_df.index[~numerical_miss_mask]
    valid_start_date = valid_date_index[0]
    valid_end_date = valid_date_index[-1]
    # 截取有效子区间
    numerical_df_sub = numerical_df.loc[
        (numerical_df.index >= valid_start_date)
        & (numerical_df.index <= valid_end_date)
    ]
    numerical_df_sub.columns = [
        "missing_numerilized"
    ]  # 修改转为数值后的列名，以便存入processed_df
    numerical_miss_mask_sub = numerical_miss_mask.loc[
        (numerical_miss_mask.index >= valid_start_date)
        & (numerical_miss_mask.index <= valid_end_date)
    ]
    mylog.info(f"numerical_df_sub:\n{numerical_df_sub}")

    # 5 统计数值化后有效子区间的缺失值info
    # 计算数值化后的nan总比例
    numerical_sub_miss_num = numerical_miss_mask_sub.sum()
    numerical_sub_total_ratio = numerical_miss_mask_sub.sum() / len(
        numerical_df
    )
    # 计算数值化后的最长连续缺失长度
    numerical_sub_longest_consecutive_missing = max(
        (
            len(list(group))
            for k, group in groupby(numerical_miss_mask_sub)
            if k
        ),
        default=0,
    )

    # 3 判断是否可以修复
    if (
        numerical_sub_total_ratio <= repairable_ratio
        and numerical_sub_longest_consecutive_missing
        <= int(repairable_longest_consec_ratio * len(df))
    ):
        is_repairable = True
    else:
        is_repairable = False

    # 打印缺失值信息
    miss_results = {
        "origin_len": len(df),
        "all_types": all_types,
        "miss_types": miss_types,
        "miss_total_ratio": miss_total_ratio,
        "type_counts_ratios": type_counts_ratios,
        "longest_consecutive_missing": longest_consecutive_missing,
        "df_miss_mask": df_miss_mask,
        # after numerical
        "numerical_sub_len": len(numerical_df_sub),
        "numerical_sub_miss_num": numerical_sub_miss_num,
        "numerical_sub_total_ratio": numerical_sub_total_ratio,
        "numerical_sub_longest_consecutive_missing": numerical_sub_longest_consecutive_missing,
        "numerical_df_miss_mask": numerical_df_miss_mask,
        "is_repairable": is_repairable,  #
    }

    def check_missing_print(miss_results: dict):
        mylog.info(
            f"\n================= check missing results =================="
            f"\n------------------- before numeralize --------------------"
            f"\n原始序列-数据总量:\n {miss_results['origin_len']}"
            f"\n原始序列-缺失值位置:\n {miss_results['miss_types']}"
            f"\n原始序列-缺失值各类型的数量、比例:\n {miss_results['type_counts_ratios']}"
            f"\n原始序列-缺失值总占比: {miss_results['miss_total_ratio']}"
            f"\n原始序列-最长连续缺失数量: {miss_results['longest_consecutive_missing']}"
            f"\n-------------------- after numeralize --------------------"
            f"\n有效子序列-数据总量: {miss_results['numerical_sub_len']}"
            f"\n有效子序列-缺失值总数: {miss_results['numerical_sub_miss_num']}"
            f"\n有效子序列-缺失值总占比: {miss_results['numerical_sub_total_ratio']}"
            f"\n有效子序列-最长连续缺失数量: {miss_results['numerical_sub_longest_consecutive_missing']}"
            f"\n有效子序列-是否可以修补: {miss_results['is_repairable']}"
            ""
            f"\n=========================================================="
        )

    if is_print:
        check_missing_print(miss_results)
    return (
        all_types,  # 原始序列所有位置的类型
        numerical_df_sub,
        valid_start_date,
        valid_end_date,  # 原始序列数值化后的有效子序列
        miss_results,
    )


if __name__ == "__main__":
    # origin_df = pd.read_csv(r'E:\Project_yyang\bg-forecast\data\钢材.csv', usecols=['日期', '热轧板卷4.75mm', ],
    #                         index_col=['日期'])
    # origin_df.index = pd.to_datetime(origin_df.index)
    # numerical_df, miss_results = check_missing(origin_df, check_start='2019-09-03', check_end='2019-12-29')

    dates = pd.date_range(start="2024-01-01", end="2024-01-13")
    data = {
        "price": [
            np.nan,
            4520,
            "-4540",
            0,
            -4600,
            "%",
            4679,
            True,
            4610,
            "4640",
            "text",
            None,
            4670,
        ]
    }
    origin_df = pd.DataFrame(data=data, index=dates, columns=["price"])
    # numerical_df, miss_results = check_missing(origin_df, '2024-01-01', check_end='2024-01-12')
    # print(numerical_df)
