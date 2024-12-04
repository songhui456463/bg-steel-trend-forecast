"""
单因子预测模型建模：HW
"""

import copy
import warnings
import numpy as np
import pandas as pd
from itertools import product
from typing import Tuple
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from utils.log import mylog
from preprocess.pretesting import autocorr_test, gaussian_test


def holtwinters_model_by_gridsearch(train_df: pd.DataFrame):
    """
    holtwinters模型，自适应确定阶数
    :param train_df: 训练序列
    :return: holtwintersResults
    """
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="statsmodels"
    )
    copy_train_df = copy.deepcopy(train_df).reset_index(
        drop=True
    )  # 非dateindex，因为不连续日的dateindex输入到ARIMA中，会warning

    # 网格参数
    trend = ["add", "mul"]
    damped_trend = [True, False]  # trend=None时，damped_trend不能为True
    seasonal = [None]
    # seasonal = ['add', 'mul', None]
    trend_dampedtrend_seasonal = list(
        product(trend, damped_trend, seasonal)
    ) + [(None, False, None)]

    # 初始值
    best_bic = np.inf
    # best_order = ('mul', False, None)
    best_order = None
    best_model_res = None

    # 网格搜索确定参数
    for order in trend_dampedtrend_seasonal:
        try:
            # 拟合
            if order[2] == None:  # seasonal==None
                model_res = ExponentialSmoothing(
                    copy_train_df,
                    trend=order[0],
                    damped_trend=order[1],
                    seasonal=None,
                ).fit()
            else:
                model_res = ExponentialSmoothing(
                    copy_train_df,
                    trend="add",
                    damped_trend=False,
                    seasonal="add",
                    seasonal_periods=250,
                ).fit()
            # 比较选择best_model_res
            if model_res.bic < best_bic:  # todo 使用aic or bic or else
                best_bic = model_res.bic
                best_order = order
                best_model_res = model_res
        except Exception as e:  # 忽略拟合失败的模型
            mylog.info(f"阶数组合 {order} 拟合失败，错误信息: {e}")

    # 使用默认参数
    if not best_model_res:
        default_order = ("mul", False, None)  # 短期预测偏好非阻尼
        default_model_res = ExponentialSmoothing(
            train_df,
            trend=default_order[0],
            damped_trend=default_order[1],
            seasonal=default_order[2],
            seasonal_periods=250,
        ).fit()
        best_order = default_order
        best_model_res = default_model_res

    # mylog.info(f'----------------- verify resid ------------------')
    resid = best_model_res.resid.to_frame()
    resid_is_autocorr = autocorr_test(resid).get("is_corr", None)
    resid_is_normal = gaussian_test(resid).get("is_gaussian", None)
    if resid_is_autocorr:
        mylog.warning(
            f"holtwinters_model with best_order=({best_order}), resid 存在自相关性, 理论上holtwinters建模失败"
        )
    # if not resid_is_normal:
    #     mylog.warning(f'holtwinters_model with order=({best_order}), resid 不是正态性, 理论上holtwinters建模失败')

    mylog.info(f"holtwinters_model best_order:\n {best_order}")
    return best_order, best_model_res


def holtwinters_model_apply(train_df: pd.DataFrame, order_tuple: Tuple):
    """
    直接根据输入的order_tuple进行建模
    :param train_df:
    :param order_tuple: （trend参数，damped_trend参数，seasonal参数）
    :return:
    """
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="statsmodels"
    )
    copy_train_df = copy.deepcopy(train_df).reset_index(
        drop=True
    )  # 非dateindex，因为不连续日的dateindex输入到ARIMA中，会warning

    # 拟合
    if order_tuple[2] == None:  # seasonal==None
        model_res = ExponentialSmoothing(
            copy_train_df,
            trend=order_tuple[0],
            damped_trend=order_tuple[1],
            seasonal=None,
        ).fit()
    else:
        model_res = ExponentialSmoothing(
            copy_train_df,
            trend=order_tuple[0],
            damped_trend=order_tuple[0],
            seasonal=order_tuple[0],
            seasonal_periods=7,
        ).fit()

    return model_res
