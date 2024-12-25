"""
检验序列：
    1 是否为高斯白噪声：自相关性、正态性
    2 平稳/非平稳
    3 是否存在异方差性
    4 是否存在季节成分
输出 各项test的：
    True/False
    检验指标的值
"""

import copy

import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

from preprocess.pre_enums import EnumPretestingReturn
from utils.enum_family import EnumForecastMethod


def autocorr_test(df: pd.DataFrame):
    """
    检验序列的自相关性，比如原始序列的自相关性、残差的自相关性
    :param df: 待检验的序列
    :return:
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # 确定检验的最大滞后阶数
    max_lag = min(30, len(df) - 1)
    res_df = acorr_ljungbox(
        df.iloc[:, 0], lags=max_lag, auto_lag=False
    )  # H0:不存在自相关性

    if res_df["lb_pvalue"].min() >= 0.05:
        is_corr = False  # 所有阶数均不存在自相关性
        best_corrlag = None
    else:
        is_corr = True  # 存在自相关性
        # 不能直接通过pvalue或统计量值来找出自相关性最强的滞后阶数（统计量值是累积的，自然会随滞后阶数的增加而增加，因为它考虑了多个滞后期的自相关性。）
        # best_corrlag = res_df['lb_pvalue'].idxmin()  # 不管用
        best_corrlag = None

    # mylog.info(f'AutoCorr-Test Result:\n'
    #            f'==========================================================\n'
    #            # f'{res_df}\n'
    #            # f'\n'
    #            f'is_corr:{is_corr}\n'
    #            f'best_corrlag: {best_corrlag}\n'
    #            f'==========================================================')

    res_dict = {EnumPretestingReturn.autocorrTest_is_corr: is_corr}
    return res_dict
    # return is_corr, best_corrlag


def gaussian_test(df: pd.DataFrame) -> dict:
    """
    检验序列是否服从正态分布：JBtest, 计算峰度和偏度
    :param df:待检测的序列
    :return:
    """
    is_gaussian = None
    jb_statis, jb_pvalue = stats.jarque_bera(df)  # H0：服从正态分布

    if jb_pvalue > 0.05 and jb_pvalue <= 1:
        is_gaussian = True
    else:
        is_gaussian = False

    # 计算偏度和峰度
    data_skew = stats.skew(df)
    data_kurtosis = stats.kurtosis(df)
    # mylog.info(f'Gaussian Test Result:\n'
    #            f'==========================================================\n'
    #            f'Jarque-Bera Statistic: {jb_statis}\n'
    #            f'gaussian_test p-value: {jb_pvalue}\n'
    #            f'is_gaussian: {is_gaussian}\n'
    #            f'data_skew: {data_skew}\n'
    #            f'data_kurtosis: {data_kurtosis}\n'
    #            f'==========================================================')
    # =3是正态的峰度，>3:比正态更尖的峰，<3:比正态更宽的峰，与3的距离在1以内被认为接近正态，超过1可能表明数据的分布偏离正态性。
    # Excess Kurtosis = Kurtosis−3

    res_dict = {
        EnumPretestingReturn.gaussianTest_is_gaussian: is_gaussian,
        EnumPretestingReturn.gaussianTest_skew: data_skew,
        EnumPretestingReturn.gaussianTest_kurtosis: data_kurtosis,
    }
    return res_dict


def whitenoise_test(df):
    """
    检验序列是否为高斯白噪声：自相关性、正态分布、偏峰度
    e.g. ARIMA,线性回归的残差要求是whitenoise
    :param df:
    :return:
    """
    is_whitenoise = False
    # 自相关性
    is_autocorr = autocorr_test(df)
    # 正态性
    gaussian_res = gaussian_test(df)
    is_gaussian = gaussian_res.get(
        EnumPretestingReturn.gaussianTest_is_gaussian
    )
    # 偏峰度t分布

    if not is_autocorr and is_gaussian:  # e.g.检验残差
        is_whitenoise = True

    res_dict = {
        EnumPretestingReturn.whitenoiseTest_is_whitenoise: is_whitenoise
    }
    return res_dict


def stationary_test(df):
    """
    平稳性检验: 检验原始序列，若不平稳，继续检验差分序列，比如adfuller
    :param df: 原始序列
    :return: is_stationary:原始序列是否平稳, stationary_d：平稳序列的差分阶数，0、1、2
    """
    from statsmodels.tsa.stattools import adfuller

    copy_df = copy.deepcopy(df)
    is_stationary = None  # 原始序列是否平稳
    stationary_d = 0  # 平稳阶数即需要差分的阶数
    d_pvalue = {}  # 存放差分阶数和对应的pvalue
    for d in range(3):
        # mylog.info(f'------- diff order = {d} -------')
        adf_result = adfuller(
            copy_df.iloc[:, 0],
            maxlag=None,
            regression="c",
            autolag="AIC",
            store=False,
            regresults=False,
        )  # H0：存在单位根，不平稳
        adf_p = adf_result[1]
        # mylog.info(f'adfuller p-value: {adf_p}')  # p_value=0.6713348990495726  # 不平稳
        d_pvalue[d] = adf_p
        if adf_p > 0.05 and adf_p <= 1:
            if d == 0:
                is_stationary = False
        else:  # 平稳 p<0.05
            if d == 0:
                is_stationary = True
            stationary_d = d  # 需要差分d阶 才至平稳
            break
        copy_df = copy_df.diff(1).dropna()
    # 注意：如果到这里stationary_d仍为None，说明该序列差分3次都没有平稳，可能是“累计”型序列，次年首月和当年最后一月的差分值相对于其他差分值是异常的。
    # 即使再多差分几次也平稳不了。

    # mylog.info(
    #     f"Stationary-Test Result:\n"
    #     f"==========================================================\n"
    #     f"差分阶数d及对应pvalue: {d_pvalue}\n"
    #     f"is_stationary: {is_stationary}\n"
    #     f"stationary_d: {stationary_d}\n"
    #     f"=========================================================="
    # )

    res_dict = {
        EnumPretestingReturn.stationaryTest_is_stationary: is_stationary,
        EnumPretestingReturn.stationaryTest_stationary_d: stationary_d,
    }
    # return is_stationary, stationary_d
    return res_dict


def hetero_test(df):
    """
    检验序列是否存在异方差性(原始序列，预测残差)
    如果存在异方差性，可用的预测方法名称的List，不存在异方差性，可用的预测方法名称的list
    若存在异方差性，是否要进行repair
    :param df:
    :return:
    """

    is_het = False
    lm_statis, lm_pvalue, f_statis, f_pvalue = het_arch(
        df.iloc[:, 0].values
    )  # H0：不存在异方差性
    # LM_statis,LM_pvalue,F_statis,F_pvalue = het_breuschpagan(df.iloc[:, 0].values)

    if lm_pvalue < 0.05:
        is_het = True

    # mylog.info(
    #     f"Hetero-Test Result\n"
    #     f"==========================================================\n"
    #     f"LM-Statis value: {lm_statis}\n"
    #     f"LM_Pvalue: {lm_pvalue}\n"
    #     f"F_statis:{f_statis}\n"
    #     f"F_pvalue:{f_pvalue}\n"
    #     f"is_het:{is_het}\n"
    #     f"=========================================================="
    # )

    res_dict = {
        EnumPretestingReturn.hetetoTest_is_het: is_het,
    }
    return res_dict


# def season_test(df):
#     """
#     检验序列中是否含有季节成分，若含有，可以用SARIMA\ETS等模型
#     :param df:
#     :return:
#     """
#     pass


def run_pretesting(df: pd.DataFrame):
    """
    对缺失异常处理后的单变量df进行检验，判断进行那种预测方法
    :param df:
    :return:
    """
    forecastmethod_list = []
    # 1 检验自相关性
    is_autocorr = autocorr_test(df).get(
        EnumPretestingReturn.autocorrTest_is_corr
    )
    if not is_autocorr:
        # 2.1 若原始序列无自相关性，无需使用预测方法
        is_het = hetero_test(df).get(EnumPretestingReturn.hetetoTest_is_het)
        # 若序列的方差值是常数，则检测是否为正态分布，否则用t分布拟合
        if not is_het:
            is_normal = gaussian_test(df).get(
                EnumPretestingReturn.gaussianTest_is_gaussian
            )
            if is_normal:
                forecastmethod_list.append(EnumForecastMethod.NORMAL_FIT)
            else:
                forecastmethod_list.append(EnumForecastMethod.T_FIT)
        else:
            forecastmethod_list.append(EnumForecastMethod.GARCH_FIT)
    else:
        # 2.2 若原始序列存在自相关性
        # is_stationary, stationary_d = stationary_test(df)
        res_stationary = stationary_test(df)
        if res_stationary.get(
            EnumPretestingReturn.stationaryTest_is_stationary
        ):
            # 直接对平稳序列进行预测
            forecastmethod_list.extend(
                [
                    EnumForecastMethod.ARIMA,  # 需要平稳性条件，但预测方法模块会再进行平稳化
                    EnumForecastMethod.HOLTWINTERS,  # 不要求序列平稳
                    EnumForecastMethod.FBPROPHET,  # 需要平稳性条件，但...
                    EnumForecastMethod.LSTM_SINGLE,  # 需要平稳性条件，但...
                    EnumForecastMethod.VAR,  # 需要平稳性条件，但...
                    EnumForecastMethod.GARCH,
                ]
            )
    return forecastmethod_list


if __name__ == "__main__":
    pass
