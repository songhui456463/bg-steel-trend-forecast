"""
单因子预测模型建模：ARIMA
"""

import copy
import numpy as np
import pandas as pd
import warnings
from itertools import product
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from forecasting.modeling_holtwinters import rolling_window_split
from preprocess.pre_enums import EnumPretestingReturn
from preprocess.pretesting import (
    autocorr_test,
    gaussian_test,
    stationary_test,
    whitenoise_test,
)
from utils.log import mylog


# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=ValueWarning, module='statsmodels')


def arima_model(train_df: pd.DataFrame):
    """
    ARIMA模型，自适应确定阶数
    :param train_df: 训练序列
    :return: ARIMAResults
    """
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="statsmodels"
    )

    # 【search order】
    copy_train_df = copy.deepcopy(train_df).reset_index(
        drop=True
    )  # 非dateindex，因为不连续日的dateindex输入到ARIMA中，会warning

    # mylog.info(f'====================== arma_order_select_ic ========================')
    # # 法一，已知一阶差分，使用tsa.arma_order_select_ic
    # t1 = time.time()
    # from statsmodels.tsa.stattools import arma_order_select_ic
    # train_df_diff = train_df.diff(1).dropna()
    # sel_res = arma_order_select_ic(train_df_diff, max_ar=3, max_ma=3, ic=['aic', 'bic'],
    #                                trend='c', )
    # mylog.info(f'aic_order:{sel_res.aic_min_order},\n'
    #           f'aic:\n{sel_res.aic.min()}\n'
    #           f'bic_order:{sel_res.bic_min_order},\n'
    #           f'bic:\n{sel_res.bic.min()}')
    # best_model_res = ARIMA(copy_train_df, order=(sel_res.bic_min_order[0], 1, sel_res.bic_min_order[1]))
    # t2 = time.time()

    # print(f'=========================== grid search ============================')
    test_res = stationary_test(copy_train_df)
    # mylog.warning(f'8888test_res:{test_res}')
    stationary_d = test_res.get(
        EnumPretestingReturn.stationaryTest_stationary_d
    )
    # mylog.warning(f'stationary_d: {stationary_d}')

    # 法二：grid_search
    p = range(3)  # AR阶数
    q = range(3)  # MA阶数
    # d = range(2)  # 差分阶数
    d = [stationary_d]
    pdq = list(product(p, d, q))
    best_bic = np.inf  # search criterion
    # best_order = (1, 1, 2)
    best_order = None

    best_model_res = None  #

    # 网格搜索阶数
    for order in pdq:
        # mylog.info(f'------------- {order} -------------')
        try:
            mod_res = ARIMA(copy_train_df, order=order).fit()
            if mod_res.aic < best_bic:
                best_bic = mod_res.aic
                best_order = order
                best_model_res = mod_res
        except Exception as e:  # 忽略拟合失败的模型
            print(f"阶数组合 {order} 拟合失败，错误信息: {e}")

    # # crossveri搜索阶数
    # best_avg_mse = float('inf')
    # best_order = None
    # for order in pdq:
    #     avg_mse = 0
    #     n_splits = 0
    #     # mylog.warning(f'order:{order}')
    #
    #     for train_index, test_index in rolling_window_split(copy_train_df):
    #         train_split, test_split = copy_train_df.iloc[train_index], copy_train_df.iloc[test_index]
    #         train_split.reset_index(drop=True, inplace=True)  # 使跳过future warning
    #         try:
    #             mod_res = ARIMA(train_split, order=order).fit()
    #             predictions = mod_res.forecast(steps=len(test_split))
    #             mse = mean_squared_error(test_split, predictions)
    #             avg_mse += mse
    #             n_splits += 1
    #             # mylog.info(f"Order ({order}) - Fold MSE: {mse},n_splits={n_splits}")
    #
    #         except Exception as e:
    #             mylog.warning(f"阶数组合 {order} 拟合失败，错误信息: {e}")
    #             break
    #     if n_splits > 0:
    #         avg_mse /= n_splits
    #         # mylog.info(f"Order ({order}) - Avg MSE: {avg_mse},n_splits={n_splits}")
    #         if avg_mse < best_avg_mse:
    #             best_avg_mse = avg_mse
    #             best_order = order
    #             # mylog.info(f"New best order=({best_order}) with avg MSE={best_avg_mse}")
    #
    # best_model_res = ARIMA(copy_train_df, order=best_order).fit()

    # 默认阶数（经验获得）
    if not best_model_res:
        default_order = (1, stationary_d, 1)
        # copy_train_df = copy.deepcopy(train_df).reset_index(drop=True)
        my_model_res = ARIMA(copy_train_df, order=default_order).fit()
        best_order = default_order
        best_model_res = my_model_res
    mylog.info(f"best_model_res.params: \n{best_model_res.params}")

    # mylog.info(f'----------------- verify resid ------------------')
    resid = best_model_res.resid.to_frame()
    resid_is_autocorr = autocorr_test(resid).get(
        EnumPretestingReturn.autocorrTest_is_corr
    )
    resid_is_normal = gaussian_test(resid).get(
        EnumPretestingReturn.gaussianTest_is_gaussian
    )
    if resid_is_autocorr:
        mylog.warning(
            f"arima_model with order=({best_order}), resid 存在自相关性, 理论上ARIMA建模失败"
        )
    # if not resid_is_normal:
    #     mylog.warning(f'arima_model with order=({best_order}), resid 不是正态性, 理论上ARIMA建模失败')

    mylog.info(f"arima_model best_order: {best_order}")
    return best_model_res
    # return best_order, best_model_res
