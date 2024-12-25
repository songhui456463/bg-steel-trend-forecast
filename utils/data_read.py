"""
读取本地excel\csv中的因子数据
"""

import sys
import traceback
from typing import Union, Optional
import numpy as np
import pandas as pd
import copy
import os
import warnings

from utils.log import mylog
from factor.factor_resampling import check_freq
from utils.enum_family import EnumFreq
from forecasting.local_data_map import factor_location_map, price_location_map


# 全局忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def harmonize_monthfreq_date(
    target: Union[pd.DataFrame, pd.Timestamp]
) -> Union[pd.DataFrame, pd.Timestamp]:
    """
    将月频序列的dateindex统一用每月的第一天来表示
    :param target: 单列df，行索引为datetime ;或者 一个时间戳日期
    :return:
    """
    if isinstance(target, pd.DataFrame):
        if check_freq(target) == EnumFreq.MONTH:
            # 将索引转换为每个月的第一天
            target.index = target.index.normalize() + pd.offsets.MonthBegin(
                -1
            )  # 创建一个偏移量对象，表示向前移动到当月的第一天。
    elif isinstance(target, pd.Timestamp):
        # 将时间戳日期转换为当月的第一天
        target.index = target + pd.offsets.MonthBegin(-1)
    else:
        raise TypeError(
            f"target must be pd.Timestamp or DataFrame(with pd.DatetimeIndex)"
        )
    return target


def harmonize_weekfreq_date(
    target: Union[pd.DataFrame, pd.Timestamp]
) -> Union[pd.DataFrame, pd.Timestamp]:
    """
    将周频序列的dateindex统一用每周的周五来表示
    :param target: 单列df，行索引为datetime ;或者 一个时间戳日期
    :return:
    """
    if isinstance(target, pd.DataFrame):
        if check_freq(target) == EnumFreq.WEEK:
            # 将日期索引转换为每周的周五
            target.index = target.index + pd.offsets.Week(
                n=0,  # n：表示向前或向后移动多少个星期。正数表示向后移动，负数表示向前移动。
                weekday=4,
            )  # 0:周一
    elif isinstance(target, pd.Timestamp):
        # 将时间戳日期转换为每周的周五
        target = target + pd.offsets.Week(
            n=0,  # n：表示向前或向后移动多少个星期。正数表示向后移动，负数表示向前移动。
            weekday=4,
        )  # 0:周一
    else:
        raise TypeError(
            f"target must be pd.Timestamp or DataFrame(with pd.DatetimeIndex)"
        )
    return target


def read_x_from_local(
    file_path, col_idx: int, start_date: str = None, end_date: str = None
):
    """
    从excel文件中读取loc_idx对应的列
    :param file_path:
    :param col_idx:
    :return: x_df index为datetime日期，仅有一列数据
    """
    # 1读取因子列
    filename = os.path.basename(file_path)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        raw_x_df = pd.read_excel(file_path).iloc[
            :, [0, col_idx]
        ]  # date列，因子列
    elif filename.endswith(".csv"):
        raw_x_df = pd.read_csv(file_path).iloc[:, [0, col_idx]]
    else:
        raise TypeError(f"file_path must be .xlsx, .xls or .csv")

    # 2取出指标名称行作为x_df的columns
    factor_name_row_df = raw_x_df.loc[raw_x_df.iloc[:, 0] == "指标名称"]

    # 3取出数据行，日期列转为datetime, 排除掉表头几行无效文字说明
    raw_x_df.iloc[:, 0] = pd.to_datetime(
        raw_x_df.iloc[:, 0], errors="coerce", format=None
    )
    x_df = raw_x_df.loc[raw_x_df.iloc[:, 0].notna()].infer_objects()

    # 拼接，指标名称行做columns
    if not factor_name_row_df.empty:
        x_df.columns = ["日期"] + [
            factor_name_row_df.iloc[0, i]
            for i in range(1, factor_name_row_df.shape[1])
        ]
    else:
        x_df.columns = ["日期"] + [
            factor_name_row_df.columns[i]
            for i in range(1, factor_name_row_df.shape[1])
        ]
    # 设置datetime列为行索引
    x_df.set_index(keys=[x_df.columns[0]], drop=True, inplace=True)
    # 按日期升序排序
    x_df.sort_index(inplace=True, ascending=True)

    # 4若输入date参数，则根据date参数取序列
    if start_date is not None:
        start_date = max(
            x_df.index[0], pd.to_datetime(start_date)
        )  # datetime之间比较
        x_df = x_df.loc[start_date <= x_df.index]
    if end_date is not None:
        end_date = min(x_df.index[-1], pd.to_datetime(end_date))
        x_df = x_df.loc[x_df.index <= end_date]

    # 5排除掉最开始的连续nan值
    x_df = x_df.dropna()

    # 6对于周频和月频序列，统一用周五和月第一天来表示
    if check_freq(x_df) == EnumFreq.MONTH:
        harmonize_monthfreq_date(target=x_df)
    if check_freq(x_df) == EnumFreq.WEEK:
        harmonize_weekfreq_date(target=x_df)

    mylog.info(
        f"<{x_df.columns[0]}>,"
        f"first_date:{x_df.index[0].strftime('%Y-%m-%d')},last_date:{x_df.index[-1].strftime('%Y-%m-%d')},"
        f"len={x_df.shape[0]}"
    )

    return x_df


def read_x_by_map(
    factor_name: str, start_date: str = None, end_date: str = None
):
    """
    根据因子名称查询当前的factor_location_map，获取该因子所在的文件路径和列idx，读取该因子序列
    :param factor_name: 要读取的x序列的名称
    :param start_date: 可以不指定日期，即读取整列数据
    :param end_date:
    :return: x_df index为datetime日期，仅有一列数据
    """
    if factor_name in list(price_location_map.keys()):
        location_map = price_location_map
    else:
        location_map = factor_location_map

    # 查询当前的factor_location_map
    cur_factor_location_dict = location_map.get(factor_name, None)
    if cur_factor_location_dict is not None:
        file_path = cur_factor_location_dict.get("path", None)
        if sys.platform != "win32":
            file_path = file_path.replace("\\", "/")
        col_idx = cur_factor_location_dict.get("col_idx", None)
        # 从本地读取数据
        if file_path is not None and col_idx is not None:
            try:
                x_df = read_x_from_local(
                    file_path=file_path,
                    col_idx=col_idx,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                x_df = None
                mylog.warning(
                    f"<{factor_name}> 序列读取失败. traceback:\n{traceback.format_exc()}"
                )
        else:
            mylog.warning(
                f"<{factor_name}> factor_location_map中的path和col_idx无效，序列读取失败"
            )
            x_df = None
    else:
        mylog.warning(
            f"<{factor_name}> factor_location_map中没有当前factor_name，序列读取失败"
        )
        x_df = None
    return x_df


if __name__ == "__main__":
    pass

    # factor_name = '国际热轧板卷汇总价格：中国市场（日）'
    # factor_name = '销量:液压挖掘机:主要企业:出口(外销):当月值'
    # factor_name = '国际热轧板卷汇总价格：中国市场（周）'
    # x_df = read_x_by_map(factor_name)
