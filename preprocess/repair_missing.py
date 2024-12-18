"""
修补时间序列缺失值，使用不同的方法
"""

import copy

import matplotlib.pyplot as plt
import pandas as pd

from check_missing import check_missing
from pre_enums import EnumRepairMissingMethod
from preconfig import PreConfig
from utils.log import mylog

# pd.set_option('display.precision', 4)  # 设置浮点数精度
pd.set_option("display.float_format", "{:,.4f}".format)  # 右对齐浮点数


def is_repairable_missing(
    origin_df: pd.DataFrame,
    check_start: str,
    check_end: str,
    preconfig: PreConfig,
) -> bool:
    """
    检查序列的缺失值信息，根据缺失值信息判断是否可以修补
    :param origin_df:
    :param check_start:
    :param check_end:
    :return:
    """
    # input改为check_missing模块的最终结果
    numerical_df, miss_results = check_missing(
        origin_df, check_start, check_end, preconfig
    )

    is_repairable = miss_results["is_repairable"]
    mylog.info(
        f"<{numerical_df.columns[0]}> check missing -> is_repairable: {is_repairable}"
    )
    return is_repairable


def fill_missing_with_MoveAverage(
    df: pd.DataFrame, moving_window=1
) -> pd.DataFrame:
    """
    修复空值的具体方法：用移动均值来修复空值
    :param df: 需要repair的df
    :return: repaired_df
    """
    nan_rows_index = df.index[df.isnull().any(axis=1)]
    nan_rows_i_list = [df.index.get_loc(index) for index in nan_rows_index]
    for i in nan_rows_i_list:
        if i == 0:
            slice_len = int(0.5 * len(df))
            if slice_len != 0:
                df.iat[i, 0] = df.iloc[
                    :slice_len, 0
                ].mean()  # 存在仍为nan的极端情况
            else:
                df.iat[i, 0] = df.iloc[:, 0].mean()
        elif i < moving_window:
            df.iat[i, 0] = df.iloc[:i, 0].mean()
        else:
            df.iat[i, 0] = df.iloc[i - moving_window : i, 0].mean()
    # print(f'new_df:\n{df}')
    repaired_df = df
    return repaired_df


def repair_missing(
    y_name: str,
    numerical_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    check_miss_results,
    preconfig: PreConfig,
):
    """
    检查序列的缺失值信息，根据缺失值信息判断如果可以修补，则进行修补 (这里不考虑int or float)
    :param y_name: 所处理的序列的名称，比如‘热轧4.75mm汇总价格’
    :param numerical_df: 经过check_missing方法数值化后的有效子序列
    :param start_date:
    :param end_date:
    :return:
    """
    # 准备参数
    method = preconfig.REPAIR_MISSING.get(
        "method", EnumRepairMissingMethod.POLYNOMIAL
    )
    is_plot = preconfig.REPAIR_MISSING.get("is_plot", False)

    # check缺失值
    # numerical_df, miss_results = check_missing(origin_df, check_start, check_end, preconfig)

    # 根据check missing的结果判断是否可以修补
    if not check_miss_results["is_repairable"]:
        mylog.warning(
            f"<{y_name}> is_reparable is False, not repaired. Not available"
        )
        return None

    # 1 截取序列有效区间
    numerical_df_sub = copy.deepcopy(
        numerical_df.loc[
            (numerical_df.index >= start_date)
            & (numerical_df.index <= end_date)
        ]
    )
    repaired_df = copy.deepcopy(numerical_df_sub)
    repaired_df.columns = ["missing_repaired"]  # 列名存储至processed_df
    repaired_col = repaired_df.columns[0]

    # 2 repair nan
    repaired_df = repaired_df.infer_objects(
        copy=False
    )  # handle future warning of 'interpolate'
    if method == EnumRepairMissingMethod.DROP:
        repaired_df = repaired_df.dropna()
    elif method == EnumRepairMissingMethod.MA:
        repaired_df = fill_missing_with_MoveAverage(repaired_df, 1)
    elif method == EnumRepairMissingMethod.LINEAR:
        repaired_df[repaired_col] = repaired_df[repaired_col].interpolate(
            method="linear"
        )
    elif method == EnumRepairMissingMethod.POLYNOMIAL:
        repaired_df[repaired_col] = repaired_df[repaired_col].interpolate(
            method="polynomial", order=2
        )
    elif method == EnumRepairMissingMethod.NEAREST:
        repaired_df[repaired_col] = repaired_df[repaired_col].interpolate(
            method="nearest"
        )
    else:
        mylog.warning(
            f"<{y_name}> repair missing method is unsupported. Not available"
        )
        raise ValueError(
            "Unsupported method. Choose from 'drop', 'ma', 'linear', 'polynomial','nearest'."
        )

    # 3 final pass
    if repaired_df[repaired_col].isnull().sum() > 0:
        # 先推断对象类型
        repaired_df = repaired_df.infer_objects()  # handle future warning
        repaired_df[repaired_col] = repaired_df[repaired_col].ffill()

    mylog.info(f"repaired_df:\n{repaired_df}")
    if repaired_df[repaired_col].isnull().sum() == 0:
        # 找出最新的有效start_date和end_date(两个方向非空的首个位置)
        valid_start_date = repaired_df.index[0]
        valid_end_date = repaired_df.index[-1]
        mylog.info(f"<{y_name}> repair missing successfully.")
        # mylog.info(f'repaired_missing_df:\n{repaired_df}')
    else:
        mylog.warning(f"<{y_name}> repair missing failed. Not available.")
        return None, start_date, end_date

    # 4 plot
    if is_plot:
        plot_miss_repair(
            y_name,
            numerical_df_sub,
            repaired_df,
        )
    return repaired_df, valid_start_date, valid_end_date


def plot_miss_repair(
    y_name: str, numerical_df: pd.DataFrame, repaired_df: pd.DataFrame
):
    """
    绘图对比缺失值修补前后的数据：plot, scatter：
    :param y_name: 所处理的序列字段的名称，比如‘热轧4.75mm汇总价格’
    :param numerical_df: 数值化后的有效子序列df
    :param repaired_df:
    :return:
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False  # 处理负号
    # 获取数据(for scatter)

    # 绘图
    fig, axs = plt.subplots(2, 1, figsize=(16, 12))
    axs[0].plot(
        repaired_df.index,
        repaired_df.iloc[:, 0],
        label="repaired data",
        linewidth=2,
        linestyle="--",
        color="orange",
    )
    axs[0].plot(
        numerical_df.index,
        numerical_df.iloc[:, 0],
        label="normal data",
        linewidth=2,
        color="blue",
    )
    axs[1].scatter(
        x=repaired_df.index,
        y=repaired_df.iloc[:, 0],
        label="repaired data",
        linewidth=0.1,
        color="orange",
    )  # 填补数据点
    axs[1].scatter(
        x=numerical_df.index,
        y=numerical_df.iloc[:, 0],
        label="normal data",
        linewidth=0.1,
        color="blue",
    )  # 正常数据点

    plt.suptitle(f"补缺效果比较--缺失值：{y_name}")
    for ax in axs.flat:
        ax.set_xlabel("日期")
        ax.set_ylabel("价格")
        ax.grid()
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test1
    # 创建日期范围
    # dates = pd.date_range(start='2024-01-01', end='2024-01-13')
    # # # prices = [4500, 4520, np.nan, 4550, 4600, np.nan, 4620, 4630, 4610, 4640, 4650, 4660, 4670]
    # data = {'price': [np.nan, 4520, np.nan, 4550, 4600, np.nan, np.nan, 4630, 4610, '4640', 'text', 4660, 4670]}
    # origin_df = pd.DataFrame(data=data, index=dates, columns=['price'])
    # test2
    origin_df = pd.read_csv(
        r"E:\Project_yyang\bg-forecast\data\钢材.csv",
        usecols=[
            "日期",
            "热轧板卷4.75mm",
        ],
        index_col=["日期"],
    )
    origin_df.index = pd.to_datetime(origin_df.index)

    # df, miss_results = check_missing(origin_df, check_start=pd.to_datetime('2024-01-01'), check_end=pd.to_datetime('2024-01-12'))
    # df, miss_results = check_missing(origin_df, check_start='2024-01-01', check_end='2024-01-12')
    # df, miss_results = check_missing(origin_df, check_start='2019-09-03', check_end='2019-12-29')

    # is_repairable = is_repairable_missing(origin_df, check_start='2019-09-03', check_end='2019-12-29')

    # repaired_df = repair_missing(origin_df, check_start='2024-01-01', check_end='2024-01-12')
    # repaired_df = repair_missing(origin_df, check_start='2019-09-03', check_end='2019-12-29')
