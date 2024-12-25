"""
单因子预测模型建模：HW
"""

import copy
import math
import warnings
from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from preprocess.pre_enums import EnumPretestingReturn
from preprocess.pretesting import autocorr_test, gaussian_test
from utils.log import mylog


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
    mylog.info(
        f"holtwinters_model_by_gridsearch with {trend_dampedtrend_seasonal} orders"
    )
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
                mylog.info(f"holtwinters_model with order=({order})")
            else:
                model_res = ExponentialSmoothing(
                    copy_train_df,
                    trend="add",
                    damped_trend=False,
                    seasonal="add",
                    seasonal_periods=250,
                ).fit()
            # 比较选择best_model_res
            if model_res.bic < best_bic:
                best_bic = model_res.bic
                best_order = order
                best_model_res = model_res
                mylog.info(
                    f"holtwinters_model with order=({order}) bic={best_bic}"
                )
        except FloatingPointError as e:  # 忽略拟合失败的模型
            mylog.info(f"阶数组合 {order} 拟合失败，错误信息: {e}")
        except Exception as e:  # 忽略拟合失败的模型
            mylog.info(f"阶数组合 {order} 拟合失败，错误信息: {e}")

    # 使用默认参数
    if not best_model_res:
        default_order = ("mul", False, None, None)  # 短期预测偏好非阻尼
        default_model_res = ExponentialSmoothing(
            train_df,
            trend=default_order[0],
            damped_trend=default_order[1],
            seasonal=default_order[2],
            seasonal_periods=None,
        ).fit()
        best_order = default_order
        best_model_res = default_model_res

    # mylog.info(f'----------------- verify resid ------------------')
    resid = best_model_res.resid.to_frame()
    resid_is_autocorr = autocorr_test(resid).get(
        EnumPretestingReturn.autocorrTest_is_corr
    )
    resid_is_normal = gaussian_test(resid).get(
        EnumPretestingReturn.gaussianTest_is_gaussian, None
    )
    if resid_is_autocorr:
        mylog.warning(
            f"holtwinters_model with best_order=({best_order}), resid 存在自相关性, 理论上holtwinters建模失败"
        )
    # if not resid_is_normal:
    #     mylog.warning(f'holtwinters_model with order=({best_order}), resid 不是正态性, 理论上holtwinters建模失败')

    mylog.info(f"holtwinters_model best_order:\n {best_order}")
    return best_order, best_model_res


def rolling_window_split(data, train_size=0.8, test_step=1):
    """
    创建滚动窗口分割器，基于固定比例的训练集和固定的测试步长。

    :param data: 输入的时间序列数据，假定已经按时间排序
    :param train_size: 训练集占总数据的比例，默认为80%
    :param test_step: 每次迭代测试集的大小（即滚动步长）,根据数据量调整，默认为1
    :return: 生成器，返回 (train_index, test_index) 对
    """
    n_samples = len(data)
    train_length = int(n_samples * train_size)
    test_step = math.ceil(train_length / 10)

    for start in range(0, n_samples - train_length, test_step):
        train_end = start + train_length
        test_start = train_end
        test_end = min(test_start + test_step, n_samples)

        train_index = list(range(start, train_end))
        test_index = list(range(test_start, test_end))

        if len(test_index) < test_step:
            break  # 如果测试集不足一个完整的步骤，则停止
        # if len(test_index) < test_step:
        # break  # 如果测试集不足一个完整的步骤，则停止

        yield train_index, test_index


def holtwinters_model_by_cv(
    train_df: pd.DataFrame, train_size=0.8, test_step=1
):
    """
    使用滚动窗口交叉验证选择最优Holt-Winters模型参数，不依赖时间索引
    :param train_df: 训练序列，假定已经按时间顺序排列
    :param train_size: 训练集占总数据的比例，默认为80%
    :param test_step: 每次滚动的步长（测试集大小），默认为20
    :return: 最优阶数组合及对应的模型结果
    """
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="statsmodels"
    )
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="statsmodels"
    )
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, module="statsmodels"
    )
    copy_train_df = copy.deepcopy(train_df).reset_index(
        drop=True
    )  # 非dateindex，因为不连续日的dateindex输入到ARIMA中，会warning

    # 构建网格参数组合
    trend = ["add", "mul"]
    damped_trend = [True, False]
    seasonal = ["add", "mul", None]
    seasonal_periods = [5, 7, 10]

    grid_params = []
    for t, d, s in product(trend, damped_trend, seasonal):
        if s is not None:
            for sp in seasonal_periods:
                grid_params.append((t, d, s, sp))
        else:
            grid_params.append((t, d, s, None))

    # mylog.info(f"holtwinters_model_by_rolling_cv with {grid_params} orders")

    # 初始化变量
    best_avg_mse = float("inf")
    best_order = None

    for order in grid_params:
        avg_mse = 0
        n_splits = 0
        # mylog.warning(f'order:{order}')

        for train_index, test_index in rolling_window_split(
            copy_train_df,
            # train_size,
            # test_step,
        ):
            train_split, test_split = (
                copy_train_df.iloc[train_index],
                copy_train_df.iloc[test_index],
            )

            try:
                model = ExponentialSmoothing(
                    train_split,
                    trend=order[0],
                    damped_trend=order[1],
                    seasonal=order[2],
                    seasonal_periods=order[3] if order[3] else None,
                    use_boxcox=False,
                )

                model_res = model.fit()
                predictions = model_res.forecast(steps=len(test_split))
                mse = mean_squared_error(test_split, predictions)
                avg_mse += mse
                n_splits += 1

                # mylog.info(f"Order ({order}) - Fold MSE: {mse},n_splits={n_splits}")

            except Exception as e:
                mylog.warning(f"阶数组合 {order} 拟合失败，错误信息: {e}")
                break

        if n_splits > 0:
            avg_mse /= n_splits
            # mylog.info(f"Order ({order}) - Avg MSE: {avg_mse},n_splits={n_splits}")
            if avg_mse < best_avg_mse:
                best_avg_mse = avg_mse
                best_order = order
                # mylog.info(f"New best order=({best_order}) with avg MSE={best_avg_mse}")

    # 使用最佳参数重新训练模型
    if best_order is not None:
        best_model = ExponentialSmoothing(
            copy_train_df,
            trend=best_order[0],
            damped_trend=best_order[1],
            seasonal=best_order[2],
            seasonal_periods=best_order[3] if best_order[3] else None,
            use_boxcox=False,
        )
        best_model_res = best_model.fit()

        # 检查残差
        resid = best_model_res.resid.to_frame()
        resid_is_autocorr = autocorr_test(resid).get(
            EnumPretestingReturn.autocorrTest_is_corr, None
        )

        if resid_is_autocorr:
            mylog.warning(
                f"holtwinters_model with best_order={best_order}, resid 存在自相关性, 理论上holtwinters建模失败"
            )
        else:
            mylog.info(
                f"holtwinters_model with best_order={best_order}, resid 不存在自相关性"
            )

    # mylog.info(f"holtwinters_model_by_rolling_cv best_order: {best_order}")
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
            seasonal_periods=order_tuple[3],
        ).fit()

    return model_res
