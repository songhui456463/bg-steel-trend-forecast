"""
单因子预测模型建模：FBprophet
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import ParameterGrid

from factor.factor_resampling import check_freq
from preprocess.pretesting import autocorr_test, gaussian_test
from utils.enum_family import EnumFreq
from utils.log import mylog


def prophet_model(train_df: pd.DataFrame):
    """
    fbprophet建模
    :param train_df: 训练序列，单因子（with dateindex）
    :return: prophet_model_res
    """
    # 复制训练集
    copy_train_df = copy.deepcopy(train_df)
    copy_train_df["日期"] = copy_train_df.index
    copy_train_df.reset_index(drop=True, inplace=True)

    # 准备数据
    # for column_name in copy_train_df.columns:
    copy_train_df = copy_train_df[["日期", train_df.columns[0]]].rename(
        columns={"日期": "ds", train_df.columns[0]: "y"}
    )

    # 初始化最佳参数和最小误差
    best_params = None
    best_rmse = float("inf")

    # # 使用默认参数训练
    # if best_params is None:
    #     my_parms = {
    #         "growth": "linear",
    #         "changepoint_prior_scale":0.5, #增大
    #         "seasonality_prior_scale": 10,
    #         "yearly_seasonality": True,
    #         "weekly_seasonality": False,
    #         "daily_seasonality": False,
    #         "interval_width": 0.8,
    #     }
    #     best_prarms = my_parms
    #     model = Prophet(**my_parms)
    #     my_model_res = model.fit(copy_train_df)
    #     best_model_res = my_model_res

    # 创建 Prophet 模型实例
    model = Prophet(
        changepoint_prior_scale=1.5,  # 调整趋势变化点的先验分布
        yearly_seasonality=False,  # 关闭默认的年度季节性
        seasonality_mode="additive",  # 或者 'multiplicative'
    )

    # 添加自定义的年度季节性，设置 fourier_order
    model.add_seasonality(
        name="yearly",
        period=365.25,  # 年周期
        fourier_order=4,  # 调整 Fourier 级数的阶数
    )

    model_res = model.fit(copy_train_df)
    best_model_res = model_res

    # mylog.info(f"prophet_model best_prams:\n {best_params}")
    return best_model_res
    # return model


def prophet_model_cv(train_df: pd.DataFrame, horizon: int = 3):
    """
    使用滚动窗口方法进行交叉验证，并直接在分割出来的测试集上评估模型性能。

    :param train_df: 训练数据集，包含 'ds' 和 'y' 列
    :param horizon: 滚动窗口的预测步长，默认为3个月
    :return: 最佳模型实例及其对应的参数配置
    """

    # 复制训练集
    copy_train_df = copy.deepcopy(train_df)
    copy_train_df["日期"] = copy_train_df.index
    copy_train_df.reset_index(drop=True, inplace=True)

    # 准备数据
    # for column_name in copy_train_df.columns:
    copy_train_df = copy_train_df[["日期", train_df.columns[0]]].rename(
        columns={"日期": "ds", train_df.columns[0]: "y"}
    )

    # 初始化最佳参数和最小误差
    best_rmse = float("inf")
    best_params = None
    best_model_res = None

    # 定义要优化的参数空间
    param_grid = {
        "fourier_order": [2, 3, 4, 5],
        "changepoint_prior_scale": [0.5, 1.0, 2.0, 3.0],
    }

    # 动态调整参数，确保不会超出数据范围
    total_data_points = len(copy_train_df)
    initial_train_size = int(total_data_points * 0.8)
    test_size = total_data_points - initial_train_size

    for params in ParameterGrid(param_grid):
        rmses = []
        for i in range(test_size):
            # 分割数据集
            train_subset = copy_train_df.iloc[: initial_train_size + i]
            test_subset = copy_train_df.iloc[
                initial_train_size + i : initial_train_size + i + horizon
            ]

            if len(test_subset) == 0:
                break

            # 创建 Prophet 模型实例
            model = Prophet(
                changepoint_prior_scale=params["changepoint_prior_scale"],
                yearly_seasonality=False,
                seasonality_mode="additive",
            )

            # 添加自定义的年度季节性，设置 fourier_order
            model.add_seasonality(
                name="yearly",
                period=365.25,  # 年周期
                fourier_order=params["fourier_order"],
            )

            # 训练模型
            model.fit(train_subset)

            # 预测（直接在已有数据上预测）
            forecast = model.predict(test_subset[["ds"]])

            # 获取预测值和真实值
            y_true = test_subset["y"].values
            y_pred = forecast["yhat"].values

            # 计算 RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rmses.append(rmse)

        avg_rmse = np.mean(rmses) if rmses else float("inf")

        # 更新最佳参数和模型
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params
            best_model_res = model

        mylog.info(
            f"prophet_model_cv params: {best_params}, avg_rmse: {avg_rmse}"
        )

    return best_model_res


# 创建未来数据框
def create_future_dataframe(
    train_df: pd.DataFrame, pre_roll_steps: int
) -> pd.DataFrame:
    """
    创建未来数据框
    :param df: 输入数据框, index为datetime
    :param pre_roll_steps: 预测期数
    :return: 未来数据框
    """
    # mylog.warning(f'0000train_df:\n{train_df}')
    # 复制训练集
    copy_train_df = copy.deepcopy(train_df)
    copy_train_df["日期"] = copy_train_df.index
    copy_train_df.reset_index(drop=True, inplace=True)

    # 准备数据
    # for column_name in copy_train_df.columns:
    copy_train_df = copy_train_df[["日期", train_df.columns[0]]].rename(
        columns={"日期": "ds", train_df.columns[0]: "y"}
    )

    # freq = pd.infer_freq(copy_train_df['ds']) # 获取数据频率
    # if freq is None:
    #     # 如果频率推断失败，手动设置为工作日频率 'B'
    #     freq = 'MS'
    #     mylog.info("Frequency inferred as None, setting to 'MS' (business day frequency)")
    # last_date = copy_train_df['ds'].iloc[-1]
    # if freq == 'D':
    #     future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=pre_roll_steps, freq=freq) # 日频
    # elif freq == 'W':
    #     future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=pre_roll_steps, freq=freq) # 周频
    # elif freq == 'MS':
    #     future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=pre_roll_steps, freq=freq) # 月频
    # elif freq == 'B':
    #     future_dates = pd.bdate_range(start=last_date + pd.DateOffset(days=1), periods=pre_roll_steps) # 使用bdate，自动跳过周末
    # else:
    #     raise ValueError(f"Unsupported frequency: {freq}")

    freq = check_freq(train_df.iloc[:, [0]])
    last_date = copy_train_df["ds"].iloc[-1]
    if freq == EnumFreq.DAY:
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1),
            periods=pre_roll_steps,
            freq="D",
        )  # 日频
    elif freq == EnumFreq.WEEK:
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(weeks=1),
            periods=pre_roll_steps,
            freq="W",
        )  # 周频
    elif freq == EnumFreq.MONTH:
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=pre_roll_steps,
            freq="MS",
        )  # 月频
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    future_df = pd.DataFrame({"ds": future_dates})
    # mylog.info(f"future_df:\n {future_df}")
    return future_df


# return a dataframe


# 检查并更新 roll_history_df 的最后一个索引
def update_index_with_future_ds(roll_history_df, future_df, pre_value):
    # 获取 future_df 中 ds 列的最后一行值
    future_ds = future_df["ds"].iloc[-1]

    # 创建新的预测值 DataFrame，保持索引不变
    pre_value_df = pd.DataFrame(
        {roll_history_df.columns[0]: [pre_value]}, index=[future_ds]
    )

    # 更新 roll_history_df
    roll_history_df = pd.concat([roll_history_df, pre_value_df], axis=0)

    # 检查最后一个索引是否为空值或非预期值
    last_index = roll_history_df.index[-1]
    if pd.isnull(last_index) or not isinstance(last_index, type(future_ds)):
        # 如果是空值或者类型不匹配，则替换为 future_df 中 ds 列的最后一行值
        roll_history_df.index = list(roll_history_df.index[:-1]) + [future_ds]

    return roll_history_df
