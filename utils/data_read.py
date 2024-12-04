"""
读取本地excel中的因子数据
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
from forecasting.local_data_map import factor_location_map


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


def get_factor_path_from_folder(folder_path):
    """
    获取folder_path目录下的所有xlsx或xls结尾的文件中的指标名称（因子名称）
    :param folder_path:
    :return: {'factor_name':{'path': .. , 'col_idx': ..}}
    """
    # 遍历文件夹中的所有文件
    map_dict = {}
    for filename in os.listdir(folder_path):
        if (
            filename.endswith(".xlsx")
            or filename.endswith(".xls")
            or filename.endswith(".csv")
        ):
            file_path = os.path.join(folder_path, filename)
            file_path = rf"{file_path}"
            mylog.info(f"========= file_path: {file_path}")
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                data = pd.read_excel(file_path).dropna(how="all")
            else:
                data = pd.read_csv(file_path).dropna(how="all")

            # 找到‘指标名称’这一行的行索引，取出这一行
            index_of_target_value = data.index[
                data.iloc[:, 0] == "指标名称"
            ].tolist()  # 找到第0列中值为 '指标名称' 的行索引
            factor_name_df = copy.deepcopy(data.iloc[index_of_target_value, :])

            if factor_name_df.empty:  # 针对自生成的csv等
                factor_name_df = pd.concat(
                    [
                        factor_name_df,
                        pd.DataFrame(
                            [factor_name_df.columns],
                            columns=factor_name_df.columns,
                        ),
                    ],
                    axis=0,
                )

            # 找到包含'指标名称'的列
            columns_to_ignore = [
                col for col in data.columns if (data[col] == "指标名称").any()
            ]
            factor_name_df.dropna(axis=1, how="all", inplace=True)

            # 逐列找到对应列的相关info
            for col_idx in range(factor_name_df.shape[1]):
                if factor_name_df.columns[col_idx] in columns_to_ignore:
                    continue
                factor_name = factor_name_df.iloc[0, col_idx]

                col_df = data.iloc[:, [0, col_idx]]
                col_df.iloc[:, 0] = pd.to_datetime(
                    col_df.iloc[:, 0], errors="coerce", format=None
                )  # 注意：只能用下表索引 iloc
                col_df = col_df.loc[col_df.iloc[:, 0].notna()].dropna(
                    how="any"
                )

                map_dict[factor_name] = {
                    "path": file_path,
                    "col_idx": col_idx,
                    "first_date": col_df.iloc[0, 0].strftime("%Y/%m/%d"),
                    "end_date": col_df.iloc[-1, 0].strftime("%Y/%m/%d"),
                }
                mylog.info(
                    f"col_idx:{col_idx}, factor_name:{factor_name}, first_date: {col_df.iloc[0, 0]}, end_date: {col_df.iloc[-1, 0]}"
                )

    mylog.info(f"=============================")
    mylog.info(f"map_dict:\n{map_dict}")
    return map_dict


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
    # 查询当前的factor_location_map
    cur_factor_location_dict = factor_location_map.get(factor_name, None)
    if cur_factor_location_dict is not None:
        file_path = cur_factor_location_dict.get("path", None)
        if sys.platform != "win":
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
    folder_path = r"..\data\市场数据"
    map_dict = get_factor_path_from_folder(folder_path)

    # factor_name = '国际热轧板卷汇总价格：中国市场（日）'
    # factor_name = '销量:液压挖掘机:主要企业:出口(外销):当月值'
    # x_df = read_x_by_map(factor_name)
