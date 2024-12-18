"""
标的因子与协从因子的相关性分析
    1、确定协从因子与标的序列相关性最强的提前期
    2、筛选出与标的序列的相关性显著的因子
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils.log import mylog

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def max_abs_cov(price_df, factor_df, t, T, n):
    """
    计算 price s[t,T] 和 factor r[t-1,T-1], ..., r[t-n,T-n] 的相关系数

    :param price_df: (pd.DataFrame) 包含价格的单列 DataFrame，index 为日期
    :param factor_df: (pd.DataFrame) 包含因子的单列 DataFrame，index 为日期
    :param t: (str) 起始时间点
    :param T: (str) 截止时间点
    :param n: (int) 待检查的最大提前期数
    :return: (tuple) max_abs_cov_value, lag 最大绝对相关系数值及其对应的提前期
    """

    try:
        # 确保输入的 t 和 T 在 price_df, factor_df 的有效时间范围内
        if t not in price_df.index or t not in factor_df.index:

            mylog.error(
                f"Invalid date for t ({t}) or T ({T}) in the DataFrame index."
            )
            raise ValueError(
                f"Invalid date for t ({t}) or T ({T}) in the DataFrame index."
            )

        if T not in price_df.index or T not in factor_df.index:
            mylog.error(f"Invalid date for T ({T}) in the DataFrame index.")
            raise ValueError(
                f"Invalid date for T ({T}) in the DataFrame index."
            )

        if n < 0:
            mylog.error("The value of n cannot be negative.")
            raise ValueError("The value of n cannot be negative.")

        t_index = price_df.index.get_loc(t)
        if t_index < n:
            mylog.error(
                f"The value of n ({n}) is too large for the current index of t ({t})."
            )
            raise ValueError(
                f"The value of n ({n}) is too large for the current index of t ({t})."
            )

    except Exception as e:
        mylog.error(f"Parameter validation error: {e}")
        return -1, -1

    # 获取价格和因子在指定日期范围的数据
    price_s = price_df.loc[t:T].squeeze()  # 转为 Series
    max_abs_cov_value = -1
    best_lag = -1  # 保存最优提前期

    for i in range(1, n + 1):
        # 获取提前 i 的因子数据
        factor_r = factor_df.iloc[
            t_index - i : t_index - i + len(price_s)
        ].squeeze()  # 转为 Series

        # 检查因子数据长度是否与价格数据匹配
        if len(factor_r) == len(price_s):
            # 去掉行索引，否则corr方法内部会自动对齐行索引
            price_s = price_s.reset_index(drop=True)
            factor_r = factor_r.reset_index(drop=True)
            cov_value = price_s.corr(factor_r)  # 计算相关系数
            abs_cov_value = abs(cov_value)  # 取绝对值

            # 更新最大绝对协方差值和最优提前期
            if abs_cov_value > max_abs_cov_value:
                max_abs_cov_value = abs_cov_value
                best_lag = i

    return max_abs_cov_value, best_lag


def calculate_max_abs_cov_for_factors(price_df, factor_dfs, t, T, n=6):
    """
    计算多个因子与价格之间的最大绝对协方差，并将结果存储到一个 DataFrame。

    :param price_df: (pd.DataFrame) 包含价格的单列 DataFrame，index 为日期
    :param factor_dfs: factor_dataframe 多列
    :param t: (str) 起始时间点
    :param T: (str) 截止时间点
    :param n: (int) 待检查的最大提前期数
    :return: (pd.DataFrame) 包含所有因子的最优提前期及最大绝对协方差的 DataFrame
    """

    results = []

    for factor_name in factor_dfs.columns:
        factor_df = factor_dfs[[factor_name]]
        max_cov, lag = max_abs_cov(price_df, factor_df, t, T, n)

        if max_cov != -1:
            # 添加结果到列表，这里直接存储因子 DataFrame，以保留索引
            results.append(
                {
                    "factor_name": factor_name,
                    "factor_data": factor_df,
                    "best_lag": lag,
                    "max_abs_cov": max_cov,
                }
            )

    # 创建结果 DataFrame
    result_df = pd.DataFrame(results)

    # 根据 'max_abs_cov' 降序排列
    result_df_sorted = result_df.sort_values(
        by="max_abs_cov", ascending=False
    ).reset_index(drop=True)

    return result_df_sorted


def factor_cal_correlation(y_df: pd.DataFrame, x_df: pd.DataFrame) -> dict:
    """
    检验标的因子和协从因子的相关性是否显著
    :param y_df: 标的因子单列df
    :param x_df:  协从因子单列df，与y_df 等长且对齐
    :return:
    """
    y_name = y_df.columns[0]
    x_name = x_df.columns[0]

    # 0 计算两个序列的相关系数
    combined_df = pd.concat([y_df, x_df], axis=1, ignore_index=True)
    corr_mat = combined_df.corr(
        method="pearson"
    )  # 相关系数矩阵 'pearson', 'kendall', 'spearman'
    corr = round(corr_mat.loc[0, 1], 4)

    # 1 建立简单回归（有截距项）
    x_df = sm.add_constant(x_df)
    model_intercept = sm.OLS(y_df, x_df).fit()  # y_df和x_df的index需要一致
    # mylog.info(f'<{y_name}>--<{x_name}> simple_linear regression summary: \n{model_intercept.summary()}')

    # 2 t检验
    t_statis = model_intercept.tvalues.iloc[1]  # H0: beta==0
    p_value = model_intercept.pvalues.iloc[1]
    is_significant_corr = True if p_value <= 0.05 else False
    # mylog.info(f'beta:\n{model_intercept.params.values},{type(model_intercept.params.values)}')
    res_dict = {
        "y_name": y_name,
        "x_name": x_name,
        "corr": corr,  # 相关系数
        "beta": [
            np.round(model_intercept.params.values, decimals=6)
        ],  # 一元线性回归的拟合参数
        "t_statis": t_statis,  # 一元线性回归的t检验统计量
        "p_value": p_value,  # 一元线性回归的t检验p值
        "is_significant_corr": is_significant_corr,  # 一元线性回归的t检验 相关性是否显著
    }
    # 3 检验模型的有效性
    from preprocess.pretesting import autocorr_test

    is_resid_corr = autocorr_test(pd.DataFrame(model_intercept.resid)).get(
        "is_corr"
    )
    if is_resid_corr:
        # mylog.warning(f'<{y_name}>--<{x_name}> simple_linear regression 残差存在自相关性，理论上建模无效')
        pass

    return res_dict


def factor_correlation_filter(y_df: pd.DataFrame, xs_df: pd.DataFrame):
    """
    根据各协从因子与标的因子的相关性是否显著，筛选高相关性因子
    :param y_df: 单列df，标的序列（有dateindex）
    :param xs_df: 不对齐的多列因子（没有行index，因为每一列的提前期不同，导致行index不能对齐）（xs_df中的列的频度一致）
    :return:
    """
    # 1 循环计算各胁从因子与标的因子的相关性是否显著
    allfactor_corr_res = (
        pd.DataFrame()
    )  # columns=['y_name', 'x_name', 'corr', 'beta', 't_statis', 'p_value', 'is_significant_corr'
    for x_name in xs_df.columns:
        mylog.info(f"======= 检验协从因子相关显著性：{x_name}")

        # 取协从因子x_df
        x_df = xs_df[[x_name]]
        # 计算corr并t检验
        cur_res_dict = factor_cal_correlation(y_df, x_df)
        allfactor_corr_res = pd.concat(
            [allfactor_corr_res, pd.DataFrame(cur_res_dict)],
            axis=0,
            ignore_index=True,
        )
        # mylog.info(f'allfactor_corr_res:\n{allfactor_corr_res}')

    # 2 所有因子相关性显著分析信息，并按相关强度的绝对值降序排序
    allfactor_corr_res["corr_abs"] = allfactor_corr_res["corr"].abs()
    allfactor_corr_res.sort_values(
        by=["corr_abs", "p_value"], ascending=[False, True], inplace=True
    )
    allfactor_corr_res.drop(columns=["corr_abs"], inplace=True)
    allfactor_corr_res.reset_index(drop=True, inplace=True)
    # mylog.info(f"allfactor_corr_res:\n{allfactor_corr_res}")

    # 3 对corr_res，按照相关性是否显著进行筛选，--> 显著corr_factor们的序列
    filted_xs_name = allfactor_corr_res.loc[
        allfactor_corr_res["is_significant_corr"], "x_name"
    ].values
    filted_xs_df = xs_df[filted_xs_name]  # 相关性显著的因子的序列
    # mylog.info(f"<{y_df.columns[0]}> 相关性显著的因子：\n {filted_xs_df}")

    return allfactor_corr_res, filted_xs_df


if __name__ == "__main__":
    # 使用6类品种数据进行测试
    origin_df = pd.read_csv(
        r"../data/02六品种数据整理-月_test.csv", index_col=["date"]
    )
    y_df = origin_df[["热轧板卷价格"]]
    xs_df = origin_df.drop(columns=["热轧板卷价格"])
    # print(y_df)
    # print(xs_df)

    # 哐哐测试
    # 检查的最大提前期设置为6个月
    #
    # result_df = calculate_max_abs_cov_for_factors(
    #     y_df, xs_df, t="2022-01-01", T="2024-03-01", n=6
    # )
    # print(result_df[["factor_name", "max_abs_cov", "best_lag"]])
    #
    # corr_filted_xs_df = factor_correlation_filter(y_df=y_df, xs_df=xs_df)
