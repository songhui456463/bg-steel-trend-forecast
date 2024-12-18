"""
单因子预测模型建模：ARIMA
"""

import copy
import warnings
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from preprocess.pretesting import autocorr_test, gaussian_test
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

    # 【search order】  # 可选 aic or bic 为标准
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
    # 法二：grid_search
    p = range(3)  # AR阶数
    q = range(3)  # MA阶数
    d = range(2)  # 差分阶数
    pdq = list(product(p, d, q))
    best_bic = np.inf  # search criterion
    # best_order = (1, 1, 2)
    best_order = None

    best_model_res = None  #

    # 网格搜索阶数
    # for order in pdq:
    #     # mylog.info(f'------------- {order} -------------')
    #     try:
    #         mod_res = ARIMA(copy_train_df, order=order).fit()
    #         if mod_res.bic < best_bic:
    #             best_bic = mod_res.bic
    #             best_order = order
    #             best_model_res = mod_res
    #     except Exception as e:  # 忽略拟合失败的模型
    #         print(f"阶数组合 {order} 拟合失败，错误信息: {e}")
    # # if best_model_res:
    #     # print(f'best_order:{best_order}\n'
    #     #       f'best_bic: {best_bic}\n'
    #     #       f'best_model_res.params: \n{best_model_res.params}')

    # 默认阶数（经验获得）
    if not best_model_res:
        default_order = (1, 1, 2)
        # copy_train_df = copy.deepcopy(train_df).reset_index(drop=True)
        my_model_res = ARIMA(copy_train_df, order=default_order).fit()
        # mylog.info(f'my_order:{default_order}\n'
        #       f'my_bic: {my_model_res.bic}\n'
        #       f'my_model_res.params: \n{my_model_res.params}')
        best_order = default_order
        best_model_res = my_model_res

    # mylog.info(f'----------------- verify resid ------------------')
    resid = best_model_res.resid.to_frame()
    resid_is_autocorr = autocorr_test(resid).get("is_corr", None)
    resid_is_normal = gaussian_test(resid).get("is_gaussian", None)
    if resid_is_autocorr:
        mylog.warning(
            f"arima_model with order=({best_order}), resid 存在自相关性, 理论上ARIMA建模失败"
        )
    # if not resid_is_normal:
    #     mylog.warning(f'arima_model with order=({best_order}), resid 不是正态性, 理论上ARIMA建模失败')

    mylog.info(f"arima_model best_order: {best_order}")
    return best_model_res
    # return best_order, best_model_res
