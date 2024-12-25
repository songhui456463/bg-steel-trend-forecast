"""
直接对价格序列进行拟合（normal/t）
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t

from preprocess.pre_enums import EnumPretestingReturn
from preprocess.pretesting import (
    gaussian_test,
)
from utils.log import mylog


def gaussian_modeling(df: pd.DataFrame):
    """
    :param df:传入待拟合的单列df
    :return: guassian 拟合后的均值 方差
    """
    # 先进行高斯检测
    gaussian_test_result = gaussian_test(df)
    if not gaussian_test_result.get(
        EnumPretestingReturn.gaussianTest_is_gaussian
    ):
        mylog.info(f"该序列不是正态序列，不能用正态分布拟合！")
        return None

    else:
        mean_estimate, std_dev_estimate = stats.norm.fit(df)
        variance_estimate = std_dev_estimate**2
        mylog.info(
            f"正态拟合估计的均值: {mean_estimate}\n"
            f"正态拟合估计的方差: {variance_estimate}"
        )
        result = {
            "mean_estimate": mean_estimate,
            "variance_estimate": variance_estimate,
        }
        return result


def gaussian_forecasting(
    param_mean,  # 拟合估计出的gaussian期望
    param_variance,  # 拟合估计出的gaussian方差
    pre_steps,  # 预测步数
):
    """
    使用gaussian分布产生pre_steps步新值
    :param param_mean:
    :param param_variance:
    :param pre_steps:
    :return:
    """
    # 计算标准差
    std_dev = np.sqrt(param_variance)
    # 生成高斯分布的新值
    forecasted_values = np.random.normal(
        loc=param_mean, scale=std_dev, size=pre_steps
    )
    pre_value_list = forecasted_values.flatten().tolist()
    return pre_value_list


def t_dist_modeling(df: pd.DataFrame):
    """
    :param df:传入待拟合的单列df
    :return: t分布 拟合后的自由度，均值，方差
    """
    # 先进行高斯检测
    guassian_test_result = gaussian_test(df)

    if guassian_test_result.get(EnumPretestingReturn.gaussianTest_is_gaussian):
        mylog.info(f"该序列是正态序列，请用正态分布拟合！")
        return None
    else:
        df_estimate, loc_estimate, scale_estimate = stats.t.fit(df)
        variance_estimate = scale_estimate**2
        mylog.info(
            f" t分布拟合估计的自由度: {df_estimate}\n"
            f" t分布拟合估计的均值: {loc_estimate}\n"
            f" t分布拟合的方差: {variance_estimate}"
        )
        result = {
            "df_estimate": df_estimate,
            "loc_estimate": loc_estimate,
            "variance_estimate": variance_estimate,
        }

        return result


def t_dist_forecasting(
    param_df,  # 拟合估计出的t分布自由度
    param_loc,  # 拟合估计出的t分布期望
    param_variance,  # 拟合估计出的t分布方差
    pre_steps,  # 预测步数
):
    """
    使用t分布产生pre_steps步新值
    :param param_df:
    :param param_loc:
    :param param_varriance:
    :param pre_steps:
    :return:
    """
    # 计算标准差
    std_dev = np.sqrt(param_variance)
    # 生成t分布的新值
    forecasted_values = t.rvs(
        df=param_df, loc=param_loc, scale=std_dev, size=pre_steps
    )
    pre_values_list = forecasted_values.tolist()
    return pre_values_list
