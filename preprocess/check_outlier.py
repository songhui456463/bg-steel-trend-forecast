import pandas as pd

from pre_enums import EnumProcessedDF
from preconfig import PreConfig
from utils.log import mylog


def check_outliers(
    missing_repaired_df: pd.DataFrame,
    check_start: str,
    check_end: str,
    preconfig: PreConfig,
):
    """
    :param missing_repaired_df: index(datetime), 3 columns (price,missing_check_out,missing_repaired)
    :param check_start: the time start to check outliers ‘%Y-%m-%d’
    :param check_end: the time end to check outliers ‘%Y-%m-%d’
    :return: outlier points(dataframe),longest_consecutive_outliers,outlier_ratios
    """
    # 获取价格列名称
    price = missing_repaired_df.columns[0]

    # 过滤指定日期范围
    df_filtered = missing_repaired_df.loc[check_start:check_end]
    # 从preconfig中提取定义的数据正常上下界
    ub = preconfig.CHECK_OUTLIER["upper bound"]
    lb = preconfig.CHECK_OUTLIER["lower bound"]

    # 判断是否为异常值的函数
    def is_outliers(x):
        if x < lb or x > ub:  # 数据超出上下界
            return 1
        else:
            return 0

    def outlier_type(x):
        if x < 0:
            return "negative values"
        elif x < lb or x > ub:  # 数据超出上下界
            return "out of bounds"
        else:
            return "Not an outlier"

    # 新增一列用来存储是否为异常值
    df_filtered.loc[:, EnumProcessedDF.IS_OUTLIER.value] = df_filtered[
        EnumProcessedDF.MISSING_REPAIRED.value
    ].apply(lambda x: is_outliers(x))
    # 新增一列用来存储数据的异常值类型，原始结果为 No an outlier
    df_filtered.loc[:, EnumProcessedDF.OUTLIER_TYPE.value] = df_filtered[
        EnumProcessedDF.MISSING_REPAIRED.value
    ].apply(lambda x: outlier_type(x))

    # 从 preconfig中提取最大可修复的异常值比例
    repairable_ratio = preconfig.CHECK_OUTLIER["repairable_ratio"]

    # 从preconfig中提取最大可修复的连续异常值比例
    repairable_longest_consec_ratio = preconfig.CHECK_OUTLIER[
        "repairable_longest_consec_ratio"
    ]

    # 计算序列长度
    total_count = len(df_filtered)

    # 计算outliers的比例
    total_outliers_ratio = df_filtered[EnumProcessedDF.IS_OUTLIER.value].mean()

    # 计算最大连续异常值的长度
    max_consecutive_outliers = (
        df_filtered[EnumProcessedDF.IS_OUTLIER.value]
        .groupby((df_filtered[EnumProcessedDF.IS_OUTLIER.value] == 0).cumsum())
        .sum()
        .max()
    )

    # 判断是否可以repair
    if (
        total_outliers_ratio <= repairable_ratio
        and max_consecutive_outliers
        <= int(repairable_longest_consec_ratio * total_count)
    ):
        is_repairable = True
    else:
        is_repairable = False

    # 筛选出outlier_points
    outlier_points = df_filtered[
        df_filtered[EnumProcessedDF.IS_OUTLIER.value] == 1
    ]

    # 统计不同outlier的类型占比
    outlier_types = outlier_points[
        EnumProcessedDF.OUTLIER_TYPE.value
    ].value_counts(normalize=True)

    # 返回结果，并格式化为百分比
    result = {
        "total_outliers_ratio": total_outliers_ratio,
        "outlier_points": outlier_points,
        "longest_consecutive_outliers": max_consecutive_outliers,
        "outlier_types": outlier_types,
        "is_repairable": is_repairable,
    }

    mylog.info(
        f"\n================= check outliers results =================="
        f"\n<{price}> 异常数据点比例：{result['total_outliers_ratio']*100:.2f}%"
        f"\n<{price}>异常数据点:{result['outlier_points']}"
        f"\n<{price}>最长连续异常值数量:{result['longest_consecutive_outliers']}"
        f"\n<{price}>异常值类型统计:{result['outlier_types']}"
        f"\n<{price}>是否可以修正：{result['is_repairable']}"
        f"\n==========================================================="
    )

    # 定义有效的开始和结束时间
    # 取第一个有效数值的位置为valid start date
    valid_start_date = df_filtered[
        df_filtered[EnumProcessedDF.IS_OUTLIER.value] == 0
    ].index[0]
    # 有效结束时间为最后一行index
    valid_end_date = df_filtered.index[-1]
    df_filtered = df_filtered[
        [EnumProcessedDF.IS_OUTLIER.value, EnumProcessedDF.OUTLIER_TYPE.value]
    ]
    return result, df_filtered, valid_start_date, valid_end_date


# if __name__ == '__main__':
#
#     date_rng = pd.date_range(start='2024-01-01', end='2024-01-10')
#     df = pd.DataFrame({'price': [10, 20, 30.5, -10000, 50, 70, 80, 90, 200,3000]}, index=date_rng)
#     result,df_filter = check_outliers(origin_df=df,check_start='2024-01-01', check_end='2024-01-10',preconfig=preconfig)
#     print(result)
#     #
