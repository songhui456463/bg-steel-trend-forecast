import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pre_enums import EnumRepairOutliersMethod
from preprocess.preconfig import preconfig
from utils.log import mylog

pd.set_option("display.float_format", "{:,.4f}".format)  # 右对齐浮点数


def repair_outliers(
    outliers_checked_result: dict,
    outliers_checked_df: pd.DataFrame,
    repair_start_date: str,
    repair_end_date: str,
    method: str,
    Preconfig=preconfig,
) -> pd.DataFrame:
    """
    :param outliers_checked_result: the dict of outlier check result, 包含是否可以修复的信息
    :param outliers_checked_df: index(datetime), 5 columns (price,missing_check_output,missing_repaired,is_outlier,outlier_types)
    :param repair_start_date: the time start to repair outliers ‘%Y-%m-%d’
    :param repair_end_date: the time end to repair outliers ‘%Y-%m-%d’
    :param method:'remove', 'replace_mean', 'replace_median', 'interpolate'
    'moving_average', 'exponential_moving_average', or 'quantile_replacement'
    :return: repaired dataframe
    """
    # 获取列名
    price_col = outliers_checked_df.columns[0]
    is_plot = Preconfig.REPAIR_OUTLIER["is_plot"]

    if not outliers_checked_result["is_repairable"]:
        mylog.info(f"\n<{price_col}> is_reparable is False, no repair")
        return None

    # 创建新的dataframe保存修改结果
    df_copy = outliers_checked_df.loc[repair_start_date:repair_end_date]
    df_copy.drop([price_col], axis=1, inplace=True)
    df_copy.rename(columns={"missing_repaired": price_col}, inplace=True)
    outlier_indexes = df_copy[df_copy["is_outlier"] == 1].index

    if method == EnumRepairOutliersMethod.REMOVE.value:
        df_copy = df_copy.drop(index=outlier_indexes)

    elif method == EnumRepairOutliersMethod.REPLACE_MEAN.value:
        # 用均值替代异常值
        mean_value = df_copy[price_col].mean()
        df_copy.loc[outlier_indexes, price_col] = mean_value

    elif method == EnumRepairOutliersMethod.REPLACE_MEDIAN.value:
        # 用中位数替代异常值
        median_value = df_copy.loc[repair_start_date:repair_end_date][
            "missing_repaired"
        ].median()
        df_copy.loc[outlier_indexes, price_col] = median_value

    elif method == EnumRepairOutliersMethod.INTERPOLATE.value:
        # 使用线性插值法填补异常值
        df_copy.loc[outlier_indexes, price_col] = np.nan  # 将异常值设为 NaN
        df_copy[price_col] = df_copy[price_col].interpolate(method="linear")

    elif method == EnumRepairOutliersMethod.MA.value:
        # 移动平均法替代异常值
        df_copy[price_col] = (
            df_copy[price_col]
            .rolling(window=3, min_periods=1, center=True)
            .mean()
        )

    elif method == EnumRepairOutliersMethod.EMA.value:
        # 指数加权移动平均替代异常值
        df_copy[price_col] = (
            df_copy[price_col].ewm(span=3, adjust=False).mean()
        )

    elif method == EnumRepairOutliersMethod.QR.value:
        # 用第25或75百分位数替代异常值
        lower_quantile = df_copy[price_col].quantile(0.25)
        upper_quantile = df_copy[price_col].quantile(0.75)
        df_copy.loc[outlier_indexes, price_col] = np.clip(
            df_copy.loc[outlier_indexes, price_col],
            lower_quantile,
            upper_quantile,
        )

    else:
        raise ValueError(
            "Unsupported method. Choose from 'remove',replace_mean', 'replace_median', "
            "'interpolate', 'moving_average', 'exponential_moving_average', or 'quantile_replacement'."
        )
    print(df_copy[price_col])
    if is_plot:
        plot_repaired(
            outliers_checked_df, df_copy, repair_start_date, repair_end_date
        )

    # 修改列名为 ‘outliers_repaired’表示修复后数据
    df_copy.rename(columns={price_col: "outliers_repaired"}, inplace=True)
    outliers_repaired_df = pd.concat(
        [outliers_checked_df[price_col], df_copy], axis=1
    )
    mylog.info(
        f"\n<{price_col}>Outliers Repaired df:\n {outliers_repaired_df}"
    )

    # 更新有效起止时间
    valid_start_date = df_copy.index[0]
    valid_end_date = df_copy.index[-1]

    return df_copy, valid_start_date, valid_end_date


def plot_repaired(
    df: pd.DataFrame, repaired_df: pd.DataFrame, start_date: str, end_date: str
):

    # 选择特定时间范围的数据
    original_data = df[start_date:end_date]
    repaired_data = repaired_df[start_date:end_date]
    # 获取列名
    price = original_data.columns[0]
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False  # 处理负号

    plt.figure(figsize=(12, 6))

    fig, axs = plt.subplots(2, 1, figsize=(16, 12))
    axs[0].plot(
        repaired_data.index,
        repaired_data[price],
        label="repaired data",
        linewidth=2,
        linestyle="--",
        color="orange",
    )
    axs[0].plot(
        original_data.index,
        original_data["missing_repaired"],
        label="missing repaired data",
        linewidth=2,
        color="blue",
    )
    axs[1].scatter(
        x=repaired_data.index,
        y=repaired_data[price],
        label="repaired data",
        linewidth=0.5,
        color="orange",
    )
    axs[1].scatter(
        x=original_data.index,
        y=original_data["missing_repaired"],
        label="missing repaired data",
        linewidth=0.5,
        color="blue",
    )
    #
    plt.suptitle(f"补缺效果比较--异常值：{price}")
    for ax in axs.flat:
        ax.set_xlabel("日期")
        ax.set_ylabel("价格")
        ax.grid()
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
    # 显示图形
    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     date_rng = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
#     df = pd.DataFrame({'price': [10, 20, 30.5, -1000, 50, 70, 80, 90, 200,3000]}, index=date_rng)
#     result = repair_outliers(df, '2024-01-01', '2024-01-10','replace_median',preconfig)
