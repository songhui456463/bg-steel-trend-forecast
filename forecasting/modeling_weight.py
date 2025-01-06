"""
各基础预测模型的权重优化模型：comprehensive_weight_model
"""

import numpy as np
import pandas as pd
import traceback
from gurobipy import Model, quicksum, GRB
from scipy.optimize import minimize

from utils.log import mylog


def weight_optimization(real_pre_df: pd.DataFrame):
    """
    根据训练数据 优化预测模型的权重
    :param real_pre_df: 训练数据的real_pre_df
    :return: 一个weight_model
    """
    premodels = real_pre_df.columns[1:].tolist()
    premodels_num = len(premodels)

    """gurobi优化"""
    model = Model("Weight_Optimization")
    w = model.addVars(premodels_num, name="weight", lb=0, ub=1)  # 变量

    # 损失函数和约束
    mse_obj = quicksum(
        (
            real_pre_df.iloc[t, 0]
            - quicksum(
                w[i] * real_pre_df.iloc[t, i + 1] for i in range(premodels_num)
            )
        )
        ** 2
        for t in range(len(real_pre_df))
    )
    model.setObjective(mse_obj, GRB.MINIMIZE)
    model.addConstr(
        quicksum(w[i] for i in range(premodels_num)) == 1, "sum_to_one"
    )

    # 求解
    model.setParam("OutputFlag", 0)
    model.Params.TimeLimit = 300  # 限制求解时间
    model.optimize()
    if model.status == GRB.OPTIMAL:
        optimal_ws_dict = {}
        for i in range(premodels_num):
            pre_model = premodels[i]
            optimal_ws_dict[pre_model] = w[i].X
    else:
        mylog.warning(f"< pre_weight > 优化失败")
        optimal_ws_dict = {}

    mylog.info(f"optimal_ws_dict: \n{optimal_ws_dict}")
    # mylog.info(f"optimal_ws_dict: \n{optimal_ws_dict}")
    return optimal_ws_dict


def weight_optimization_cvxpy(real_pre_df: pd.DataFrame):
    """
    根据训练数据 优化预测模型的权重
    :param real_pre_df: 训练数据的real_pre_df
    :return: 一个weight_model
    """
    premodels = real_pre_df.columns[1:].tolist()
    premodels_num = len(premodels)

    # # 求解
    # model.optimize()
    # if model.status == GRB.OPTIMAL:
    #     optimal_ws_dict = {}
    #     for i in range(premodels_num):
    #         pre_model = premodels[i]
    #         optimal_ws_dict[pre_model] = w[i].X
    # else:
    #     mylog.warning(f'< pre_weight > 优化失败')
    #     optimal_ws_dict = {}

    # mylog.info(f'optimal_ws_dict: \n{optimal_ws_dict}')
    # return optimal_ws_dict


def weight_optimization_scipy(real_pre_df: pd.DataFrame):
    """
    根据训练数据 优化预测模型的权重
    :param real_pre_df: 训练数据的real_pre_df
    :return: 一个weight_model
    """
    premodels = real_pre_df.columns[1:].tolist()
    premodels_num = len(premodels)

    # 目标函数：均方误差
    def objective(weights):
        predictions = real_pre_df[premodels].values
        true_values = real_pre_df.iloc[:, 0].values
        weighted_predictions = np.dot(predictions, weights)
        mse = np.mean((true_values - weighted_predictions) ** 2)
        return mse

    # 约束条件：权重之和为1
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    # 权重范围：0到1
    bounds = [(0, 1)] * premodels_num

    # 初始猜测值
    initial_guess = [1 / premodels_num] * premodels_num

    # 进行优化
    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        optimal_ws_dict = dict(zip(premodels, result.x))
    else:
        mylog.warning("< pre_weight > 优化失败")
        optimal_ws_dict = {}

    # mylog.info(f"optimal_ws_dict: \n{optimal_ws_dict}")
    return optimal_ws_dict


def weight_forcast(pre_df: pd.DataFrame, optimal_ws_dict: dict):
    """
    根据各模型的预测值，得到权重预测值
    :param pre_df:
    :param optimal_ws_dict:
    :return:
    """
    if optimal_ws_dict:  # 不为空时
        pre_df["weighted_pre"] = sum(
            pre_df[premodel] * w for premodel, w in optimal_ws_dict.items()
        )

    # mylog.info(f"weighted_pre_df:\n{pre_df}")
    return pre_df
