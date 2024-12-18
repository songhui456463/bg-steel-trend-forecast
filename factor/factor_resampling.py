"""
对齐不同因子的dateindex（降采样或升采样）
1、不同因子之间相关性、共线性分析，需要对齐dateindex
    比如日频价格序列和周/月频协从因子的相关性，日频价格降采样为周/月频
2、相关性筛选后的有效因子 输入到多因子模型中：
    若价格序列降采样为月频，那预测模型的预测输出值只能是月频的；
    若协从因子升采样为日频，那就需要对协从因子做大量插值，预测效果怎么保证？

"""

from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from utils.enum_family import EnumFreq
from utils.log import mylog

# 设置显示的最大行数
pd.set_option("display.max_rows", 200)  # 设置为 10 行
# 设置显示的最大列数
pd.set_option("display.max_columns", 20)  # 设置为 10 列


def check_freq(single_df: pd.DataFrame) -> Optional[EnumFreq]:
    """
    判断single_df的频度：日频\周频\月频
    :param single_df: 单列df，dateindex
    :return:
    """
    # 检查index是否为datetime类型
    if not is_datetime64_any_dtype(single_df.index):
        mylog.error(f"<df:{single_df.columns[0]}> 的行索引不是datetime")
        return None

    # 参数：检查连续几期的diff
    # param_step = min(10, len(single_df))  # 10为参数
    param_step = len(single_df)  # 用这个，防止印度刺客
    diff_day_count_list = [
        single_df.index[i + 1] - single_df.index[i]
        for i in range(param_step - 1)
    ]
    diff_day_count_min = min(diff_day_count_list).days
    # mylog.info(f'diff_day_count_list: \n{diff_day_count_list}')
    # mylog.info(f'diff_day_count_min: \n{diff_day_count_min}')

    # 判断频度
    if diff_day_count_min >= 28:
        freq = EnumFreq.MONTH
    elif diff_day_count_min == 7:
        freq = EnumFreq.WEEK
    elif diff_day_count_min == 1:
        freq = EnumFreq.DAY
    else:
        mylog.error(f"数据频度判断失败")
        freq = None
    # mylog.info(f'<df:{single_df.columns[0]}> freq={freq.value}')
    return freq


def downsampling_by_dateindex(highfreq_df, base_lowfreq_df):
    """
    对高频序列降采样，降为与低频序列一致的频度（相同的dateindex）
    :param highfreq_df: 单列df，如日频价格序列
    :param base_lowfreq_df: 单列df，如周频/月频协从因子
    :return: 原高频序列downsampling后的df
    """
    highfreq_name = highfreq_df.columns[0]

    """法一：低频序列全部展开为日频，再提取"""
    # 均展开成完全的日频
    date_start = base_lowfreq_df.index[0]
    date_end = base_lowfreq_df.index[-1]
    flatten_highfreq_df = pd.DataFrame(
        index=pd.date_range(date_start, date_end, freq="D")
    )
    flatten_highfreq_df = pd.merge(
        left=flatten_highfreq_df,
        right=highfreq_df,
        how="outer",  # outer是为了 如果highfreq的日期更早，可以给第一个空值直接插上值
        left_index=True,
        right_index=True,
    )

    # 基于时间进行插值（因为时间可能不是均匀的）
    flatten_highfreq_df[highfreq_name] = flatten_highfreq_df[
        highfreq_name
    ].interpolate(method="time")
    flatten_highfreq_df[highfreq_name] = flatten_highfreq_df[
        highfreq_name
    ].bfill()  # 当第一个值为空时，进行二次插值

    # 取均值降频
    flatten_highfreq_df[f"downsampling_{highfreq_name}"] = np.nan
    for low_date_i in range(len(base_lowfreq_df.index)):
        low_date = base_lowfreq_df.index[low_date_i]
        if low_date_i == 0:
            flatten_highfreq_df.loc[
                low_date, f"downsampling_{highfreq_name}"
            ] = flatten_highfreq_df.loc[low_date, highfreq_name]
            continue
        flatten_highfreq_df.loc[low_date, f"downsampling_{highfreq_name}"] = (
            np.mean(
                flatten_highfreq_df.loc[
                    (
                        flatten_highfreq_df.index
                        > base_lowfreq_df.index[low_date_i - 1]
                    )
                    & (flatten_highfreq_df.index <= low_date),
                    [highfreq_name],
                ]
            )
        )
    # mylog.info(f"flatten_highfreq_df: \n{flatten_highfreq_df}")

    # 返回填充后的单列因子df
    downsampling_highfreq_df = flatten_highfreq_df.loc[
        base_lowfreq_df.index, [f"downsampling_{highfreq_name}"]
    ]
    downsampling_highfreq_df.columns = [f"{highfreq_name}"]

    """法二：高频序列和base_lowfreq merge后基于时间插值，再提取"""
    # # 与低频序列按索引外拼接，获取低频索引
    # concat_df = pd.merge(
    #     left=highfreq_df,
    #     right=base_lowfreq_df,
    #     how="outer",  # outer是为了 如果highfreq的日期更早，可以给第一个空值直接插上值
    #     left_index=True,
    #     right_index=True,
    # )
    # # mylog.info(f'concat_df\n{concat_df}')
    # concat_df[highfreq_name] = concat_df[highfreq_name].interpolate(method="time")  # 基于时间进行插值（因为时间可能不是均匀的）
    # concat_df.columns = [highfreq_name, base_lowfreq_df.columns[0]]
    # # mylog.info(f'concat_df: \n{concat_df}')
    #
    # # 法一：直接按低频index取出
    # # downsampling_highfreq_df = concat_df.loc[base_lowfreq_df.index,[highfreq_name]]
    # # 法二：取均值
    # concat_df[f"downsampling_{highfreq_name}"] = np.nan
    # for low_date_i in range(len(base_lowfreq_df.index)):
    #     low_date = base_lowfreq_df.index[low_date_i]
    #     if low_date_i == 0:
    #         concat_df.loc[low_date, f"downsampling_{highfreq_name}"] = (
    #             concat_df.loc[low_date, highfreq_name]
    #         )
    #         continue
    #     concat_df.loc[low_date, f"downsampling_{highfreq_name}"] = np.mean(
    #         concat_df.loc[
    #             (concat_df.index > base_lowfreq_df.index[low_date_i - 1])
    #             & (concat_df.index <= low_date),
    #             [highfreq_name],
    #         ]
    #     )
    # mylog.info(f'concat_df: \n{concat_df}')
    # # 返回填充后的单列因子df
    # downsampling_highfreq_df = concat_df.loc[
    #     base_lowfreq_df.index, [f"downsampling_{highfreq_name}"]
    # ]
    # downsampling_highfreq_df.columns = [f'{highfreq_name}']

    # mylog.info(f'downsampling_lowfreq_df: \n{downsampling_highfreq_df}')
    return downsampling_highfreq_df


def upsampling_by_dateindex(
    base_highfreq_df: pd.DataFrame, lowfreq_df: pd.DataFrame
):
    """
    对低频序列升采样，升为与高频序列一致的频度（相同的dateindex）
    :param base_highfreq_df: 单列df(含dateindex)，如日频价格序列
    :param lowfreq_df: 单列df(含dateindex)，如周频/月频协从因子
    :return: 原低频序列upsampling后的df
    """
    lowfreq_name = lowfreq_df.columns[0]

    """法一：低频序列全部展开为日频，再提取"""
    # 均展开成完全的日频
    date_start = base_highfreq_df.index[0]
    date_end = base_highfreq_df.index[-1]
    flatten_lowfreq_df = pd.DataFrame(
        index=pd.date_range(date_start, date_end, freq="D")
    )
    flatten_lowfreq_df = pd.merge(
        left=flatten_lowfreq_df,
        right=lowfreq_df,
        how="outer",
        left_index=True,
        right_index=True,
    )
    flatten_lowfreq_df[lowfreq_name] = flatten_lowfreq_df[
        lowfreq_name
    ].interpolate(
        method="time"
    )  # 基于时间进行插值（因为时间可能不是均匀的）
    flatten_lowfreq_df[lowfreq_name] = flatten_lowfreq_df[
        lowfreq_name
    ].bfill()  # 当第一个值为空时，进行二次插值
    # 提取出bases_highfreq对应的行
    upsampling_lowfreq_df = flatten_lowfreq_df.loc[base_highfreq_df.index]

    """法二：低频序列展开为base_highfreq，再提取"""
    # # 与高频序列拼接，获取高频dateindex
    # concat_df = pd.merge(
    #     left=base_highfreq_df,
    #     right=lowfreq_df,
    #     how="outer",
    #     left_index=True,
    #     right_index=True,
    # )
    # concat_df.columns = [base_highfreq_df.columns[0], lowfreq_name]
    # # mylog.info(f'concat_df: \n{concat_df}')
    # # 低频列填充
    # concat_df[lowfreq_name] = concat_df[lowfreq_name].interpolate(method="time")  # 基于时间进行插值（因为时间可能不是均匀的）
    # concat_df[lowfreq_name] = concat_df[lowfreq_name].bfill()  # 当第一个值为空时，进行二次插值
    # # mylog.info(f'concat_df: \n{concat_df}')
    #
    # # 按高频index取出填充后的低频因子
    # upsampling_lowfreq_df = concat_df.loc[
    #     base_highfreq_df.index, [lowfreq_name]
    # ]

    # mylog.info(f'upsampling_lowfreq_df: \n{upsampling_lowfreq_df}')
    return upsampling_lowfreq_df


def auto_resampling(base_df, todo_df):
    """
    判断todo_df需要upsampling还是downsampling，与base_df对齐
    :param base_df: 单列df,基准dateindex，比如价格序列
    :param todo_df: 单列df,需要与base_df的频度对其（dateindex需要一致）
    :return: todo_df按base_df重采样后的、按base_df.index的序列
    """
    # 检查两个序列的频度
    base_freq = check_freq(base_df)
    todo_freq = check_freq(todo_df)

    # 判断todo_df应该up还是down采样
    base_flag = {EnumFreq.DAY: 3, EnumFreq.WEEK: 2, EnumFreq.MONTH: 1}.get(
        base_freq, -1
    )
    todo_flag = {EnumFreq.DAY: 3, EnumFreq.WEEK: 2, EnumFreq.MONTH: 1}.get(
        todo_freq, -1
    )
    if base_flag > 0 and todo_flag > 0:
        if base_flag > todo_flag:
            new_todo_df = upsampling_by_dateindex(
                base_highfreq_df=base_df, lowfreq_df=todo_df
            )
        elif base_flag < todo_flag:
            new_todo_df = downsampling_by_dateindex(
                highfreq_df=todo_df, base_lowfreq_df=base_df
            )
        else:
            # 频度相同。但存在标的dateindex和因子dateindex不能对齐的情况，因此仍然重采样
            new_todo_df = upsampling_by_dateindex(
                base_highfreq_df=base_df, lowfreq_df=todo_df
            )  # 部分因子存在固定月份数据缺失的情况，因此升采样
            # new_todo_df = todo_df
    else:
        mylog.warning(
            f"price_df<{base_df.columns[0]}>, factor_df<{todo_df.columns[0]}> : 重采样对齐失败, 忽略当前factor"
        )
        raise Exception(
            f"price_df<{base_df.columns[0]}>, factor_df<{todo_df.columns[0]}> : 重采样对齐失败, 忽略当前factor"
        )

    return new_todo_df


if __name__ == "__main__":
    # 创建一个包含日频数据的DataFrame
    dates_y = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
    values_y = np.random.randn(len(dates_y))  # 随机生成日频数据
    df_y = pd.DataFrame({"day_data": values_y}, index=dates_y)
    print(f"日频数据 df_y:\n{df_y}")

    dates_x = pd.date_range(start="2023-01-01", end="2023-03-31", freq="W")
    values_x = np.random.randn(len(dates_x))  # 随机生成周频数据
    df_x = pd.DataFrame({"week_data": values_x}, index=dates_x)
    print(f"周频数据 df_x:\n{df_x}")

    dates_z = pd.date_range(start="2023-01-01", end="2023-03-31", freq="ME")
    values_z = np.random.randn(len(dates_z))  # 随机生成月频数据
    df_z = pd.DataFrame({"month_data": values_z}, index=dates_z)
    print(f"月频数据 df_z:\n{df_z}")

    # new_todo_df1 = auto_resampling(base_df=df_y, todo_df=df_x)
    # print("==============================")
    # new_todo_df2 = auto_resampling(base_df=df_x, todo_df=df_y)
    # print(
    #     "=========================================================================================="
    # )
    new_todo_df3 = auto_resampling(base_df=df_z, todo_df=df_x)
    # print("==============================")
    # new_todo_df4 = auto_resampling(base_df=df_x, todo_df=df_z)
    # print(
    #     "=========================================================================================="
    # )
    # new_todo_df5 = auto_resampling(base_df=df_z, todo_df=df_y)
    # print("==============================")
    # new_todo_df6 = auto_resampling(base_df=df_y, todo_df=df_z)
