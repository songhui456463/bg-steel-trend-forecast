"""
多因子预测模型：VAR
"""

import copy
import sys

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR

from preprocess.pre_enums import EnumPretestingReturn
from preprocess.pretesting import stationary_test
from utils.log import mylog

plt.rcParams["axes.unicode_minus"] = False  # 处理负号
if sys.platform == "win32":
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
else:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # Arial Unicode MS

def inverse_diff(before_diff_raw_df, diff_pre_df):
    """
    将差分预测序列逆差分一次
    :param before_diff_raw_df: 差分前的序列
    :param diff_pre_df: 预测序列，差分一次
    :return:
    """
    inverse_diff_pre_df = copy.deepcopy(diff_pre_df)

    last_value = before_diff_raw_df.iloc[-1, 0]
    for i in range(len(inverse_diff_pre_df)):
        inverse_value = last_value + diff_pre_df.iloc[i, 0]
        inverse_diff_pre_df.iloc[i, 0] = inverse_value
        # 更新前面值
        last_value = inverse_value

    return inverse_diff_pre_df


def var_model(
    diff_xs_df: pd.DataFrame,
    inf_criterion: str = "aic",
    is_ir_and_fred: bool = False,
    is_save_ir_and_fevd: bool = True,
):
    """
    输入逐列分别差分后的xs_df，自适应选择滞后阶数，返回训练模型
    :param diff_xs_df: 各列分别差分后（各列分别平稳）的多列df，第0列为价格序列
    :param inf_criterion: 选择最佳滞后阶数的信息准则
    :param is_ir_and_fred:  是否对var模型进行ir分析和方差分解，（画图）
    :param is_save_ir_and_fred: 是否保存分析图像至本地
    :return: 一个训练好的var model
    """

    """0 协整检验：逐个因子与价格序列"""  # 不存在协整关系的因子理论上不应该参与到对价格序列的预测中，剔除
    price_ser = diff_xs_df.iloc[:, 0]
    for col_idx in range(1, diff_xs_df.shape[1]):
        x_ser = diff_xs_df.iloc[:, col_idx]
        coint_t_statis, coint_pvalue, _ = sm.tsa.stattools.coint(
            price_ser, x_ser
        )  # H0:不存在长期协整关系
        if coint_pvalue <= 0.05:
            mylog.info(
                f"协整检验 ：<{x_ser.name}> coint_t_statis={coint_t_statis}, coint_pvalue={coint_pvalue} (<=0.05 存在协整关系)"
            )
        else:
            mylog.warning(
                f"协整检验 ：<{x_ser.name}> coint_t_statis={coint_t_statis}, coint_pvalue={coint_pvalue} (>0.05 不存在协整关系)"
            )

    """1 自动计算maxlags"""
    # 默认的自动搜索最优阶数的最大滞后阶数p
    maxlags = 5
    # 训练样本数可以计算的最大滞后阶数
    sample_n, variable_k = diff_xs_df.shape
    need_samples_num = (
        variable_k * variable_k * maxlags + variable_k
    )  # 训练maxlags之后阶数的var至少需要的样本量
    if sample_n < need_samples_num:
        maxlags = (sample_n - variable_k) // ((variable_k) ** 2)
        if maxlags < 1:
            mylog.error(
                f"根据实际样本量更新的的maxlags={maxlags}，小于1，不能有效var建模。需增加样本量。当前样本: sample_n={sample_n}, variable_k={variable_k}"
            )
            raise Exception(
                f"样本量不足以VAR建模.当前样本: sample_n={sample_n}, variable_k={variable_k}"
            )
        else:
            mylog.info(
                f"根据实际样本量自动更新: maxlags={maxlags}。当前样本: sample_n={sample_n}, variable_k={variable_k}"
            )

    """2 自适应选择滞后阶数p"""
    temp_diff_xs_df = copy.deepcopy(diff_xs_df).reset_index(drop=True)
    lags = VAR(temp_diff_xs_df).select_order(maxlags=maxlags).selected_orders
    mylog.info(f"选择滞后阶数: lags=\n{lags}")
    # 根据信息准则选择最优滞后阶数
    # inf_criterion = 'hqic'  # {'aic': 5, 'bic': 1, 'hqic': 4, 'fpe': 4}
    # inf_criterion, min_lag_value = min(lags.items(), key=lambda item: item[1])
    best_lags = lags.get(inf_criterion)

    """3 基于历史数据建模"""
    # varmax_model_res = VARMAX(diff_xs_df).fit(maxlags=best_lags)
    model_res = VAR(diff_xs_df).fit(maxlags=best_lags)
    model_res_resid = model_res.resid
    mylog.info(f"var_model_res params: \n{model_res.params}")

    """4 模型检验"""
    # 参数稳定性检验：AR根（VAR模型特征方程根的绝对值的倒数要在单位圆里面）只有通过参数稳定性检验的模型才具有预测能力,进行脉冲响应和方差分解分析才有意义。
    cusum_statis, cusum_pvalue, _ = (
        statsmodels.stats.diagnostic.breaks_cusumolsresid(model_res_resid)
    )  # H0:无漂移（平稳）
    if cusum_pvalue >= 0.05:
        mylog.info(
            f"cusum-ols-resid检验: cusum_statis={cusum_statis}, cusum_pvalue={cusum_pvalue} (>=0.05 var模型参数稳定)"
        )
    else:
        mylog.warning(
            f"cusum-ols-resid检验: cusum_statis={cusum_statis}, cusum_pvalue={cusum_pvalue} (<0.05 var模型参数不稳定)"
        )

    # if cusum_pvalue >= 0.05:
    # if is_ir_and_fred:
    #     # 脉冲响应
    #     # ax = varmax_model_res.impulse_responses(steps=12, orthogonalized=True).plot(figsize=(12, 8))  # statsmodels中VAR模型没有.impulse_responses，VARMAX中有
    #     # plt.show()
    #     impulse_res = model_res.irf(periods=12)
    #     impulse_res.plot(figsize=(24, 16), orth=True, signif=0.95, seed=10)  # MonteCarlo预测脉冲响应。横坐标为预测期数，纵坐标为受冲击的影响
    #     if is_save_ir_and_fevd:
    #         plt.savefig(os.path.join(settings.OUTPUT_DIR_PATH, '[var] impulse_response.png'))
    #     # plt.show()
    #
    #     # 方差分解
    #     fevd = model_res.fevd(periods=10)
    #     # mylog.info(f'var fevd.summary():\n{fevd.summary()}')
    #     fevd.plot(figsize=(12, 16))
    #     if is_save_ir_and_fevd:
    #         # 保存fevd数值
    #         file_name = os.path.join(settings.OUTPUT_DIR_PATH, '[var] fevd.xlsx')
    #         for col_idx in range(diff_xs_df.shape[1]):
    #             col_name = diff_xs_df.columns[col_idx]
    #             sheet_name = (f'FEVD_for_{col_name}'
    #                           .replace(":", "_")
    #                           .replace("：","_"))  # 注意excel sheetname的命名规范（不能有: ：）
    #             fevd_mat = fevd.decomp[col_idx]
    #             fevd_df = pd.DataFrame(data=fevd_mat,columns=diff_xs_df.columns,index=[i for i in range(1,fevd_mat.shape[0]+1)])
    #             if not os.path.exists(file_name):
    #                 with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
    #                     fevd_df.to_excel(writer, sheet_name=sheet_name, index=True)
    #             else:
    #                 with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
    #                     fevd_df.to_excel(writer, sheet_name=sheet_name, index=True)
    #
    #         # 保存fevd图像
    #         plt.savefig(os.path.join(settings.OUTPUT_DIR_PATH, '[var] fevd.png'))
    #     # plt.show()

    return best_lags, model_res


def var_forcast(
    train_xs_df: pd.DataFrame,
    pre_steps: int,
    varmodel_update_freq: int = 99999,
    is_ir_fevd: bool = False,
):
    """
    VAR滚动预测
    :param train_xs_df: 多因子df，已对齐dateindex，各因子序列未经过差分
    :param pre_steps: 滚动预测的步数
    :param varmodel_update_freq: var模型的更新频率（每roll多少步更新一次模型）
    :param is_ir_fevd: 是否进行ir和fevd分析
    :return: 根据训练数据确定var模型的滞后阶数
    """
    train_xs_df_copy = copy.deepcopy(train_xs_df)
    price_name = train_xs_df_copy.columns[0]
    diff_xs_df = copy.deepcopy(train_xs_df_copy)

    # 分别对各列因子做标准化, 消除不同因子量纲的影响
    # mylog.info(f'标准化前的diff_xs_df: \n{diff_xs_df}')
    scaler_dict = {}
    for col in diff_xs_df.columns:
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        diff_xs_df[col] = scaler.fit_transform(
            diff_xs_df[[col]]
        )  # 即使是input多列,scaler内部也是对每一列分别标准化
        scaler_dict[col] = scaler  # 【和不标准化的预测结果一样】
    # mylog.info(f'标准化后的diff_xs_df: \n{diff_xs_df}')

    # 1 逐列检查各因子的平稳性并差分
    xs_diff_degree = {}
    # 记录价格序列的差分过程，用于后面逆差分
    price_diff_middle = copy.deepcopy(diff_xs_df.iloc[:, [0]])
    price_diff_middle.columns = ["diff_0"]
    for col in diff_xs_df.columns:
        # 检查差分阶数
        cur_col_degree = stationary_test(diff_xs_df[[col]]).get(
            EnumPretestingReturn.stationaryTest_stationary_d
        )

        xs_diff_degree[col] = cur_col_degree
        if cur_col_degree == 0:
            # 原序列平稳，不需要差分
            continue
        else:
            # 逐次差分
            if diff_xs_df.columns.get_loc(col) == 0:
                for d in range(cur_col_degree):
                    price_diff_middle[f"diff_{d+1}"] = price_diff_middle[
                        f"diff_{d}"
                    ].diff(1)
                diff_xs_df[col] = price_diff_middle.iloc[:, -1]
            else:
                if cur_col_degree is not None:
                    for d in range(cur_col_degree):
                        diff_xs_df[col] = diff_xs_df[col].diff(1)
                else:  # 当前列因子无法平稳化，抛弃该因子
                    diff_xs_df.drop(columns=[col], inplace=True)
                    train_xs_df_copy.drop(columns=[col], inplace=True)
                    mylog.warning(f"<{col}> 当前因子无法平稳化，不参与到var中")

    # mylog.info(f'各列因子的差分次数：\n{xs_diff_degree}')
    # mylog.info(f'逐列差分后的diff_xs_df:\n{diff_xs_df}')

    # 去除差分产生的nan
    diff_xs_df.dropna(inplace=True)
    # mylog.info(f'逐列差分、去除nan后的diff_xs_df:\n{diff_xs_df}')

    # 2 各列都平稳后，滚动预测
    # pres_xs_df = pd.DataFrame(columns=train_xs_df_copy.columns)
    pres_xs_df = None
    for pre_i in range(pre_steps):
        # 每预测多少步更新一次模型
        if pre_i % varmodel_update_freq == 0:
            mylog.info(f"<---pre_i:{pre_i}> <var> 更新var模型")
            best_lags, model_res = var_model(
                diff_xs_df,
                inf_criterion="aic",
                is_ir_and_fred=is_ir_fevd,
            )
            mylog.info(
                f"<---pre_i:{pre_i}> <var> 更新var模型：best_lag={best_lags}"
            )
        else:
            mylog.info(f"<---pre_i:{pre_i}> <var> 不更新var模型")

        # 按滞后阶数取出input
        input_xs_df = diff_xs_df.iloc[-best_lags:]

        # 预测一步（input滞后期数据）
        pre_xs = model_res.forecast(input_xs_df.values, steps=1)
        pre_xs_df = pd.DataFrame(pre_xs, columns=train_xs_df_copy.columns)

        # 保存差分预测值
        if pres_xs_df is None:
            pres_xs_df = copy.deepcopy(pre_xs_df)
        else:
            pres_xs_df = pd.concat(
                [pres_xs_df, pre_xs_df], axis=0, ignore_index=True
            )
        # mylog.info(f'pre_xs_df:\n{pre_xs_df}')

        # 更新历史数据（符合prod_env: 将预测值追加到历史数据中）
        diff_xs_df = pd.concat(
            [diff_xs_df, pre_xs_df], axis=0, ignore_index=True
        )

    # mylog.info(f'pres_xs_df:\n{pres_xs_df}')

    # 3 价格序列预测值 逆差分
    price_degree = xs_diff_degree.get(price_name)
    if price_degree == 0:
        inverse_preprice_df = copy.deepcopy(pres_xs_df.iloc[:, [0]])
        # mylog.info(f'inverse_preprice_df:\n{inverse_preprice_df}')

    else:  # price_degree > 0
        diff_preprice_df = pres_xs_df.iloc[:, [0]]  # 逆差分前的预测序列
        # 逐步逆差分
        inverse_preprice_df = diff_preprice_df
        for d in range(price_degree):  # d=2
            inverse_preprice_df = inverse_diff(
                before_diff_raw_df=price_diff_middle.iloc[:, [-(d + 2)]],
                diff_pre_df=inverse_preprice_df,
            )
    # mylog.info(f'逆标准化前的inverse_preprice_df:\n{inverse_preprice_df}')

    # 逆标准化得到原尺度下的预测值
    inverse_preprice_df.iloc[:, 0] = (
        scaler_dict[price_name]
        .inverse_transform(inverse_preprice_df.copy())
        .flatten()
    )

    return inverse_preprice_df


if __name__ == "__main__":
    pass
    # origin_df = pd.read_csv(r'../data/02六品种数据整理-月_test.csv',
    #                         usecols=['date', '热轧板卷价格', '挖掘机销量', '家用空调'], index_col=['date'])
    # origin_df.index = pd.to_datetime(origin_df.index)
    # xs_df = origin_df.iloc[:-5]
    # test_df = origin_df.iloc[-5:]
    # print(xs_df)
    # print(test_df)
    #
    # # 测试
    # pre_steps = len(test_df)
    # var_forcast(train_xs_df=xs_df, roll_steps=pre_steps)
