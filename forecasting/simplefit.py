"""
直接对价格序列进行拟合（normal/t）或GARCH
"""

import copy
import itertools
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from scipy.stats import t

from preprocess.pretesting import (
    autocorr_test,
    gaussian_test,
    stationary_test,
    hetero_test,
)
from utils.log import mylog


def gaussian_modeling(df: pd.DataFrame):
    """
    :param df:传入待拟合的单列df
    :return: guassian 拟合后的均值 方差
    """
    # 先进行高斯检测
    gaussian_test_result = gaussian_test(df)
    if not gaussian_test_result["is_gaussian"]:
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

    if guassian_test_result["is_gaussian"]:
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


def arch_modeling(train_df: pd.DataFrame, pre_steps: int = 1):
    """
    对输入的序列（收益率序列或残差序列）进行ARCH/GARCH建模，预测未来期的值
    :param train_df: 存在异方差效应的价格序列
    :param pre_steps:
    :return:
    """

    y_name = train_df.columns[0]
    # mylog.info(f'train_vol: \n{train_df},\n{type(train_df)}')
    train_value = train_df.iloc[:, [0]]  # 用于训练GARCH的序列

    # 1 建模
    # 初始化参数
    p = [1]
    q = [1]
    dist = ["normal"]
    # p = [1,2]
    # q = [1,2]
    # dist = ['normal', 't']
    products = list(itertools.product(p, q, dist))
    # 网格定阶
    best_p, best_q, best_dist = 1, 1, "normal"
    best_aic = np.inf
    model_res = None
    for params in products:
        try:
            current_res = arch_model(
                train_value,
                mean="Constant",  # 均值模型,如‘AR’,'zero','Constant'
                # lags=1,  # 均值模型的滞后阶数
                vol="GARCH",
                p=params[0],
                q=params[1],
                dist=params[2],  # 对均值模型的残差的GARCH模型
            ).fit(disp="off")
            if current_res.aic < best_aic:
                best_aic = current_res.aic
                best_p, best_q, best_dist = params[0], params[1], params[2]
                model_res = current_res
        except:
            continue
    if not model_res:
        mylog.error(f"<{y_name}> GARCH 定阶失败，建模失败！")
        return None, None, None
    model_summary = model_res.summary()
    mylog.info(f"model_summary:\n{model_summary}")
    model_params = model_res.params

    # AIC = model_res.aic
    # BIC = model_res.bic

    # 2 检查garch模型的有效性: garch模型 残差是否为白噪声（自相关性、正态性）
    resid = model_res.resid.to_frame()
    mylog.info(f"<{y_name}> Garch-model resid:\n{resid}")
    is_autocorr = autocorr_test(resid)  # 自相关性
    is_normal = gaussian_test(resid)["is_gaussian"]  # 正态性
    if not is_autocorr and is_normal:
        resid_is_whitenoise = True
        mylog.info(
            f"<{y_name}> GARCH-model residuals 是白噪声. GARCH建模成功！"
        )
    else:
        resid_is_whitenoise = False
        mylog.warning(
            f"<{y_name}> GARCH-model residuals 不是白噪声. GARCH建模失败"
        )
    # 评估训练模型的效果
    estimate_value = (
        model_res.conditional_volatility
    )  # 训练集的模型得到的波动率
    train_mae = np.abs(estimate_value - train_value)

    resid_is_whitenoise = True
    # 3 预测
    pre_mean = None
    pre_variance = None
    if resid_is_whitenoise:  # GARCH模型有效
        mylog.info(f"<{y_name}> model_params:\n{model_params}")
        try:
            forecast_res = model_res.forecast(
                horizon=pre_steps,
                # start=,  # 默认以train_set中最后一个已知的时期开始往后预测
                # params=,  # 手动指定模型参数
            )
            pre_mean = forecast_res.mean.values[-1, :]  # 均值模型的预测值
            pre_variance = forecast_res.variance.values[
                -1, :
            ]  # 条件方差模型的条件方差（波动率）, 评估收益率波动风险

            mylog.info(f"<{y_name}> GARCH预测成功！")
            mylog.info(f"<{y_name}> pre_mean:\n{pre_mean}")
            mylog.info(f"<{y_name}> pre_variance:\n{pre_variance}")
        except:
            mylog.warning(f"<{y_name}> GARCH预测失败！")
    else:
        mylog.warning(
            f"<{y_name}> GARCH-model residuals 不是白噪声. GARCH建模失败！ "
        )
        return None, None, None

    return model_res, pre_mean, pre_variance


class GarchInput(IntEnum):
    ORIGIN = 0  # 直接对原始序列建模arch
    FLUCTUATION = 1  # 需要对输入的原始序列计算收益率再建模arch


def arch_forecasting(
    origin_df: pd.DataFrame,
    pre_steps: int = 1,
    garch_input: GarchInput = GarchInput.ORIGIN,
):
    """
    对输入的序列进行arch建模和预测，如输入‘热轧价格’
    :param origin_df: 价格的原始时间序列的训练集（不包括测试集），首先计算收益率（比例收益率 或 对数收益率），再收益率GARCH建模
    :param pre_steps: 预测步数
    :param garch_input: GARCH模型的input序列形式，
                        直接对价格序列建模GARCH或已经处理过的收益率序列，则GarchInput.ORIGIN；
                        需要转为收益率再GARCH，则GarchInput.FLUCTUATION
    :return: 预测结果df, ['pre_price']
    """
    y_name = origin_df.columns[0]
    df = copy.deepcopy(origin_df)
    # df.columns = [origin_price,'diff_0','diff_1','diff_2']

    # 1 处理GARCH的输入：原始价格or收益率
    if garch_input == GarchInput.FLUCTUATION:
        # 对原始价格序列计算收益率
        df.loc[:, "diff_0"] = (
            df.iloc[:, 0] / df.iloc[:, 0].shift(1)
        ) - 1  # 简单收益率
        # df.iloc[:,0] = np.log(df.iloc[:,0] / df.iloc[:,0].shift(1)).dropna()  # 对数收益率, 不适用于负值的情况
    else:
        # df.dropna(inplace=True)
        df.loc[:, "diff_0"] = df.iloc[:, 0].values

    # 2 检验价格/收益率序列：平稳性、异方差性
    res_stationary = stationary_test(df.loc[:, ["diff_0"]])
    is_stationary = res_stationary.get("is_stationary")
    stationary_d = res_stationary.get("stationary_d")
    is_het_diff_0 = hetero_test(df.loc[:, ["diff_0"]])
    if not is_stationary or not is_het_diff_0:
        mylog.warning(
            f"<{y_name}> 输入初始序列 非平稳or非异方差，理论上不能直接对其GARCH建模"
        )
        # return None
    # GARCH模型的输入需要是平稳的和异方差的
    if not is_stationary:
        for d in range(1, stationary_d + 1):
            df.loc[:, f"diff_{d}"] = df[df.columns[-1]].diff(1)
    else:  # 若平稳，不需要差分
        pass
    garch_input_df = df.loc[:, [f"diff_{stationary_d}"]].dropna()
    is_het = hetero_test(garch_input_df)
    if not is_het:
        mylog.warning(
            f"<{y_name}> 平稳序列（diff_{stationary_d}）无ARCH效应，理论上不能建模GARCH"
        )
    else:
        mylog.info(
            f"<{y_name}> 平稳序列（diff_{stationary_d}）存在ARCH效应，可以建模GARCH"
        )
    mylog.info(f"df:\n{df}")

    # 3 建模GARCH
    model_res_, pre_vol_mean, pre_vol_variance = arch_modeling(
        garch_input_df, pre_steps
    )

    # 4 由收益率预测值（mean和variance）转回价格预测值
    if (
        pre_vol_mean is not None and pre_vol_variance is not None
    ):  # GARCH模型有效
        # 预测收益率
        pre_returns = [
            pre_vol_mean[i] + np.random.normal(0, pre_vol_variance[i] ** 0.5)
            for i in range(pre_steps)
        ]
        # pre_returns = pre_vol_mean  # 预测收益率 仅使用均值模型

        # 逆转换差分
        # pre_df.columns = ['pre_returns','pre_diff_2','pre_diff_1','pre_diff_0','pre_price']
        pre_df = pd.DataFrame(
            {
                "pre_returns": pre_returns,
                f"pre_diff_{stationary_d}": pre_returns,
            }
        )
        for d in range(stationary_d - 1, -1, -1):
            last_before_diff = df[f"diff_{d}"].iloc[-1]
            for i in range(pre_steps):
                temp = last_before_diff + pre_df[f"pre_diff_{d + 1}"].iloc[i]
                pre_df.loc[pre_df.index[i], f"pre_diff_{d}"] = temp
                last_before_diff = temp

        # 预测价格
        if garch_input == GarchInput.FLUCTUATION:
            last_price = df.iat[-1, 0]
            for i in range(pre_steps):
                pre_df.loc[pre_df.index[i], "pre_price"] = last_price * (
                    1 + pre_df["pre_diff_0"].iloc[i]
                )  # garch直接预测的是收益率
                last_price = pre_df["pre_price"].iloc[-1]
        else:  # garch_input == GarchInput.ORIGIN
            for i in range(pre_steps):
                pre_df.loc[pre_df.index[i], "pre_price"] = pre_df[
                    "pre_diff_0"
                ].iloc[
                    i
                ]  # garch直接预测的是价格

        mylog.info(f"<{y_name}> GARCH 逆转换为价格预测值 成功！")
        mylog.info(f"<{y_name}> GARCH pre_price df:\n{pre_df}")
    else:
        return None

    pre_values_list = pre_df.loc[:, "pre_price"].tolist()
    return pre_values_list
    # return pre_df
    # return pre_df.loc[:, ["pre_price"]]


if __name__ == "__main__":
    # path = r'../data/钢材new.csv'
    # data1 = pd.read_csv(path, usecols=['日期', '冷轧板卷0.5mm', ], index_col=['日期']).head(100)
    # data1.index = pd.to_datetime(data1.index)
    # data1.sort_index(inplace=True)
    # data1.dropna(inplace=True)

    # # 生成日期范围
    # dates = pd.date_range(start='2020-01-01', periods=150, freq='D')
    # # 生成随机收益率数据，符合标准正态分布
    # np.random.seed(10)
    # returns = np.random.normal(loc=0, scale=1, size=len(dates))
    # # 创建 DataFrame
    # data1 = pd.DataFrame({'Returns': returns}, index=dates)

    np.random.seed(42)
    num = 150
    dates = pd.date_range(start="2020-01-01", periods=num, freq="D")
    X = np.linspace(1, 5, num)
    # 生成具有异方差性的因变量Y
    # Y = 2*X + 随机噪声，其中噪声的标准差随着X的增大而增大
    noise = np.random.randn(num) * X  # 噪声标准差与X成正比
    Y = 2 * 10 + noise
    Y2 = []
    het_variance = [3, 5, 12, 15, 9]
    for i in range(5):
        Y2.extend(list(np.random.normal(20, scale=het_variance[i], size=30)))

    # Y2 = np.random.normal(20, 4, size=num)
    data1 = pd.DataFrame({"price": Y2}, index=dates)
    print(f"data1:\n{data1}")

    def plot_test(df):
        plt.figure(figsize=(12, 8))
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
        plt.rcParams["axes.unicode_minus"] = False  # 处理负号
        plt.scatter(data1.index, data1["price"])
        plt.xlabel("日期")
        plt.ylabel("价格")
        plt.grid()
        plt.legend()
        plt.tick_params(axis="x", rotation=45)
        plt.show()

    # plot_test(data1)

    train_data1 = data1.iloc[:130, [0]]
    test_data1 = data1.iloc[130:, [0]]

    # 检验
    # is_autocorr, best_corrlag = autocorr_test(data1)  # 自相关性
    # # gaussian_test(data1)
    # stationary_test(data1)  # 平稳性
    # hetero_test(data1)  # 异方差性
    # # arch_modeling(data1)

    # 预测
    # pre_df = arch_forecasting(
    #     data1, pre_steps=len(test_data1), garch_input=GarchInput.ORIGIN
    # )
    # test_data1["pre_price"] = pre_df.values.flatten()
    # mylog.info(f"real-pre:\n{test_data1}")

    pre_values_list = arch_forecasting(
        data1, pre_steps=len(test_data1), garch_input=GarchInput.ORIGIN
    )
    mylog.info(f"pre_values_list:\n{pre_values_list}")
