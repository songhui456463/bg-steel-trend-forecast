"""
多因子预测模型：VAR
"""

import copy
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

from utils.log import mylog
from preprocess.pretesting import stationary_test
from preprocess.pre_enums import EnumPretestingReturn


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


def var_model(diff_xs_df: pd.DataFrame, inf_criterion: str = "aic"):
    """
    输入逐列分别差分后的xs_df，自适应选择滞后阶数，返回训练模型
    :param diff_xs_df: 各分别差分后（各列分别平稳）的多累df
    :param inf_criterion: 选择最佳滞后阶数的信息准则
    :return: 一个训练好的var model
    """
    # 自动计算maxlags
    maxlags = 5  # 默认的自动搜索最优阶数的最大滞后阶数p
    sample_n, variable_k = diff_xs_df.shape
    need_samples_num = (
        variable_k * variable_k * maxlags + variable_k
    )  # 训练maxlags之后阶数的var至少需要的样本量
    if sample_n < need_samples_num:
        maxlags = (sample_n - variable_k) // ((variable_k) ** 2)
        if maxlags < 1:
            mylog.error(
                f"根据实际样本量更新的的maxlags={maxlags}，小于1，不能有效var建模.需增加样本量."
            )
            raise Exception(
                f"样本量不足以VAR建模.目前样本中：sample_n={sample_n}, variable_k={variable_k}"
            )
        else:
            mylog.info(f"根据实际样本量更新的maxlags={maxlags}")

    # 2 自适应选择滞后阶数p
    temp_diff_xs_df = copy.deepcopy(diff_xs_df).reset_index(drop=True)
    lags = VAR(temp_diff_xs_df).select_order(maxlags=maxlags).selected_orders
    mylog.info(f"<select_order> lags:\n{lags}")
    # 根据信息准则选择最优滞后阶数
    # inf_criterion = 'hqic'  # {'aic': 5, 'bic': 1, 'hqic': 4, 'fpe': 4} todo 选择哪个标准？   # aic
    best_lags = lags.get(inf_criterion)

    # 3 基于历史数据建模
    model_res = VAR(diff_xs_df).fit(maxlags=best_lags)
    return best_lags, model_res


def var_forcast(
    train_xs_df: pd.DataFrame,
    roll_steps: int,
    varmodel_update_freq: int = 9999,
):
    """
    VAR滚动预测
    :param train_xs_df: 多因子df，已对齐dateindex，各因子序列未经过差分
    :param roll_steps: 滚动预测的步数
    :param varmodel_update_freq: var模型的更新频率（每roll多少步更新一次模型）
    :return: 根据训练数据确定var模型的滞后阶数
    """
    price_name = train_xs_df.columns[0]
    diff_xs_df = copy.deepcopy(train_xs_df)

    # 1 逐列检查各因子的平稳性并差分
    xs_diff_degree = {}
    # 记录价格序列的差分过程，用于后面逆差分
    price_diff_middle = copy.deepcopy(diff_xs_df.iloc[:, [0]])
    price_diff_middle.columns = ["diff_0"]
    for col in diff_xs_df.columns:
        # 检查差分阶数
        # cur_col_degree = stationary_test(diff_xs_df[[col]]).get('stationary_d')  # todo
        cur_col_degree = stationary_test(diff_xs_df[[col]]).get(
            EnumPretestingReturn.stationaryTest_stationary_d.value
        )
        xs_diff_degree[col] = cur_col_degree
        if cur_col_degree == 0:  # 原序列平稳，不需要差分
            continue
        else:  # 逐次差分
            if diff_xs_df.columns.get_loc(col) == 0:
                for d in range(cur_col_degree):
                    price_diff_middle[f"diff_{d+1}"] = price_diff_middle[
                        f"diff_{d}"
                    ].diff(1)
                diff_xs_df[col] = price_diff_middle.iloc[:, -1]
            else:
                for d in range(cur_col_degree):
                    diff_xs_df[col] = diff_xs_df[col].diff(1)
    # mylog.info(f'各列因子的差分次数：\n{xs_diff_degree}')
    # mylog.info(f'逐列差分后的diff_xs_df:\n{diff_xs_df}')
    # 去除差分产生的nan
    diff_xs_df.dropna(inplace=True)
    # mylog.info(f'逐列差分、去除nan后的diff_xs_df:\n{diff_xs_df}')

    # # 2 自适应选择滞后阶数p
    # temp_diff_xs_df = copy.deepcopy(diff_xs_df).reset_index(drop=True)
    # lags = VAR(temp_diff_xs_df).select_order(maxlags=5).selected_orders
    # mylog.info(f'<select_order> lags:\n{lags}')
    # # 根据信息准则选择最优滞后阶数
    # inf_criterion = 'hqic'  # {'aic': 5, 'bic': 1, 'hqic': 4, 'fpe': 4}
    # best_lags = lags.get(inf_criterion)
    #
    # # 3 基于历史数据建模
    # model_res = VAR(diff_xs_df).fit(maxlags=best_lags)

    # return model_res

    # 4 滚动预测
    pres_xs_df = pd.DataFrame(columns=train_xs_df.columns)
    for roll_i in range(roll_steps):
        # 每预测多少步更新一次模型
        if roll_i % varmodel_update_freq == 0:
            best_lags, model_res = var_model(diff_xs_df)
            mylog.info(
                f"<---roll_i:{roll_i}> <var> 更新var模型：updated best_lag={best_lags}"
            )
        else:
            mylog.info(f"<---roll_i:{roll_i}> <var> 不更新var模型")

        # 按滞后阶数取出input
        input_xs_df = diff_xs_df.iloc[-best_lags:]
        # 预测一步（input滞后期数据）
        pre_xs = model_res.forecast(input_xs_df.values, steps=1)
        pre_xs_df = pd.DataFrame(pre_xs, columns=train_xs_df.columns)
        # 保存差分预测值
        pres_xs_df = pd.concat([pres_xs_df, pre_xs_df], axis=0)
        # mylog.info(f'pre_xs_df:\n{pre_xs_df}')
        # 更新历史数据（符合prod_env: 将预测值追加到历史数据中）
        diff_xs_df = pd.concat([diff_xs_df, pre_xs_df], axis=0)
    # mylog.info(f'pres_xs_df:\n{pres_xs_df}')

    # 5 目标价格序列预测值 逆差分
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

        # mylog.info(f'inverse_preprice_df:\n{inverse_preprice_df}')
    # inverse_preprice_df.columns = ['price_pre']

    return inverse_preprice_df


if __name__ == "__main__":
    origin_df = pd.read_csv(
        r"../data/02六品种数据整理-月_test.csv",
        usecols=["date", "热轧板卷价格", "挖掘机销量", "家用空调"],
        index_col=["date"],
    )
    origin_df.index = pd.to_datetime(origin_df.index)
    xs_df = origin_df.iloc[:-5]
    test_df = origin_df.iloc[-5:]
    print(xs_df)
    print(test_df)

    # 测试
    pre_steps = len(test_df)
    var_forcast(train_xs_df=xs_df, roll_steps=pre_steps)
