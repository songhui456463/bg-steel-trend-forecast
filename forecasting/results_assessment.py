"""
计算预测结果的评价指标、评价方法
"""

import copy
import os
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from config.config import settings
from utils.log import mylog


def cal_fluc_statis(real_pred_df: pd.DataFrame) -> dict:
    """
    计算df其中两列的平均波动率（绝对值)
    :param real_pred_df: df.columns=[real,pred]
    :return:
    """
    fluc_dict = {}
    df_diff = real_pred_df.diff(1).abs().dropna()
    fluc_dict["mean"] = round(df_diff.mean(), 6)
    fluc_dict["max"] = round(df_diff.max(), 6)
    fluc_dict["min"] = round(df_diff.min(), 6)
    # real_fluc_dict = {}
    # df_diff = real_pred_df.diff(1).abs().dropna()
    # real_fluc_dict['mean'] = round(df_diff.iloc[:, 0].mean(), 6)
    # real_fluc_dict['max'] = round(df_diff.iloc[:, 0].max(), 6)
    # real_fluc_dict['min'] = round(df_diff.iloc[:, 0].min(), 6)
    # pred_fluc_dict = {}
    # df_diff = real_pred_df.diff(1).abs().dropna()
    # pred_fluc_dict['mean'] = round(df_diff.iloc[:, 1].mean(), 6)
    # pred_fluc_dict['max'] = round(df_diff.iloc[:, 1].max(), 6)
    # pred_fluc_dict['min'] = round(df_diff.iloc[:, 1].min(), 6)
    return fluc_dict


def accuracy_updowntrend(real_pred_df: pd.DataFrame) -> Tuple[dict, float]:
    """
    计算预测涨跌的准确率，只看涨跌的趋势，不看具体数值。
    若: 实际涨预测涨（11），实际涨预测跌（12），实际跌预测涨（21），实际跌预测跌（22）
    :param real_pred_df: 必须有两列：真实价格、预测价格，且长度必须大于1.
    :return: dict,四种情况分别的ratio
    """
    accuracy_dict = {
        "11": 0,
        "12": 0,
        "21": 0,
        "22": 0,
    }  # sum should be (roll_steps - trend_cycle)
    trend_cycle = 1  # 涨跌趋势以前n天的real值来判断涨or跌

    roll_steps = len(
        real_pred_df
    )  # 共预测了roll_steps步，但trend计算只能计算(roll_steps - trend_cycle)次
    # 只有roll_steps大于1的时候，才能计算accuracy
    if roll_steps <= trend_cycle:
        # 标记：没有顺利计算趋势预测准确率
        return accuracy_dict, -1
    for i in range(trend_cycle, roll_steps):
        before_real = real_pred_df.iloc[i - trend_cycle, 0]
        real = real_pred_df.iloc[i, 0]
        pred = real_pred_df.iloc[i, 1]
        # ‘1’代表涨，‘2’代表跌
        if real >= before_real and pred >= before_real:
            accuracy_dict["11"] += 1  # 实际涨预测涨
        elif real >= before_real > pred:
            accuracy_dict["12"] += 1  # 实际涨预测跌
        elif real < before_real <= pred:
            accuracy_dict["21"] += 1  # 实际跌预测涨
        elif real < before_real and pred < before_real:
            accuracy_dict["22"] += 1  # 实际跌预测跌
        else:
            raise ValueError(f"real pred updown trend error")
    # 计算预测准确率
    accuracy_ratio = round(
        (accuracy_dict["11"] + accuracy_dict["22"])
        / (roll_steps - trend_cycle),
        6,
    )
    return accuracy_dict, accuracy_ratio


def res_plot_oneroll_trend(
    method_realpredf_dict: Dict[str, pd.DataFrame],
    optimal_ws_dict,
    roll_r: int = 0,
    is_save: bool = True,
):
    """
    绘制一次roll的trend预测绝对值，默认是第0次roll
    :param method_realpredf_dict:
    :return:
    """
    method_list = list(method_realpredf_dict.keys())
    arbi_realpredf = next(iter(method_realpredf_dict.values()))
    price_name = arbi_realpredf.columns[0]
    pre_roll_r = [
        colname
        for colname in arbi_realpredf.columns
        if f"roll_{roll_r}" in colname
    ][0]

    # 从method_realpredf_dict中取出roll0这一次预测的pre_steps步预测值，放到一个df中
    oneroll_trend_realpredf = copy.deepcopy(arbi_realpredf[[price_name]])
    for method_name, realpredf in method_realpredf_dict.items():
        oneroll_trend_realpredf[f"{method_name}_{pre_roll_r}"] = realpredf[
            pre_roll_r
        ]
    # 计算加权列
    oneroll_trend_realpredf[f"weighted_{pre_roll_r}"] = sum(
        [
            oneroll_trend_realpredf[f"{method_name}_{pre_roll_r}"]
            * optimal_ws_dict[f"{method_name}_pre_T+1"]
            for method_name in method_list
        ]
    )
    # 去除不在当前次roll的预测日期
    oneroll_trend_realpredf.dropna(how="any", inplace=True)

    # 绘图
    plt.figure(figsize=(14, 10))
    # 真实值
    plt.plot(
        oneroll_trend_realpredf.index,
        oneroll_trend_realpredf[price_name],
        label=price_name,
        linewidth=1.5,
    )
    # 每一个method的预测列
    for method_name in method_list:
        plt.plot(
            oneroll_trend_realpredf.index,
            oneroll_trend_realpredf[f"{method_name}_{pre_roll_r}"],
            label=f"{method_name}_{pre_roll_r}(w={optimal_ws_dict[f'{method_name}_pre_T+1']})",
            linewidth=1.5,
        )
    # weighted预测
    plt.plot(
        oneroll_trend_realpredf.index,
        oneroll_trend_realpredf[f"weighted_{pre_roll_r}"],
        label=f"weighted_{pre_roll_r}",
        linewidth=1.5,
    )

    plt.xticks(
        ticks=oneroll_trend_realpredf.index,
        labels=oneroll_trend_realpredf.index.strftime("%Y-%m-%d"),
    )
    # 添加标题和标签
    plt.title(
        f"roll_{roll_r}价格预测结果: pre_steps={len(oneroll_trend_realpredf)}"
    )
    plt.xlabel("预测期")
    plt.ylabel("预测值")
    plt.tick_params(axis="x", rotation=45)
    # 添加图例
    plt.legend()
    if is_save:
        oneroll_res_plot_dir = os.path.join(
            settings.OUTPUT_DIR_PATH, "[Total] trend_oneroll_realpre_plot"
        )
        os.makedirs(oneroll_res_plot_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                oneroll_res_plot_dir,
                f"[Total] trend_roll_{roll_r}_realpre_plot.png",
            )
        )
    plt.show()


def forecast_res_plot(
    real_pre_df: pd.DataFrame,
    optimal_ws_dict: dict,
    is_save: bool = True,
    is_show: bool = False,
):
    """
    绘制各预测模型的预测结果
    :param real_pre_df: datetime，第一列真实值，后面列为各模型的预测值
    :optimal_ws_dict: 各预测模型的预测结果权重
    :return:
    """
    # 绘制折线图
    plt.figure(figsize=(14, 10))
    # 遍历每一列，
    for col in real_pre_df.columns:
        if col not in optimal_ws_dict.keys():  # 真实值
            plt.plot(
                real_pre_df.index,
                real_pre_df[col],
                label=f"{col}",
                linewidth=1.5,
            )
        else:  # 预测值
            plt.plot(
                real_pre_df.index,
                real_pre_df[col],
                label=f"{col} (weight={optimal_ws_dict.get(col)})",
            )
    plt.xticks(
        ticks=real_pre_df.index, labels=real_pre_df.index.strftime("%Y-%m-%d")
    )

    # 添加标题和标签
    plt.title(f"T+1价格预测结果: roll_steps={len(real_pre_df)}")
    plt.xlabel("预测期数")
    plt.ylabel("预测值")
    plt.tick_params(axis="x", rotation=45)
    # 添加图例
    plt.legend()
    if is_save:
        plt.savefig(
            os.path.join(settings.OUTPUT_DIR_PATH, "[Total] real_pre_plot.png")
        )
    if is_show:
        plt.show()


def forecast_evaluation(real_pre_df: pd.DataFrame, is_save: bool = True):
    """
    评价预测结果
    :param real_pre_df: index为dateindex,第一列为real值，后面列是各种预测方法的T+1期pre值
    :return:
    """
    price_df = real_pre_df.iloc[:, [0]]
    multimethod_pre_df = real_pre_df.iloc[:, 1:]

    # 计算指标
    mae_dict = (
        {}
    )  # {'fbprophet_pre_T+1': mae value ,'lstm_single_pre_T+1': mae value ,'weighted_pre': mae value}
    mse_dict = (
        {}
    )  # {'fbprophet_pre_T+1': mse value ,'lstm_single_pre_T+1': mse value ,'weighted_pre': mse value}
    mape_dict = {}
    mape_unfold_df = pd.DataFrame(
        columns=multimethod_pre_df.columns, index=multimethod_pre_df.index
    )
    for method_pre_name in multimethod_pre_df.columns:
        # 计算mse
        mae_dict[method_pre_name] = round(
            mean_absolute_error(
                price_df, multimethod_pre_df[[method_pre_name]]
            ),
            6,
        )
        # 计算mae
        mse_dict[method_pre_name] = round(
            mean_squared_error(
                price_df, multimethod_pre_df[[method_pre_name]]
            ),
            6,
        )
        # 计算mape
        mape_unfold_df[method_pre_name] = abs(
            multimethod_pre_df[method_pre_name] - price_df.iloc[:, 0]
        ) / (price_df.iloc[:, 0] + 1e-4)
        mape_dict[method_pre_name] = round(
            mape_unfold_df[method_pre_name].mean(), 8
        )

    # 保存
    if is_save:
        file_path = os.path.join(
            settings.OUTPUT_DIR_PATH, "[Total] accuracy_evaluation.xlsx"
        )
        accuracy_df = pd.DataFrame([mse_dict], index=["mse"])
        accuracy_df = pd.concat(
            [
                accuracy_df,
                pd.DataFrame(mae_dict, index=["mae"]),
                pd.DataFrame(mape_dict, index=["mape"]),
            ],
            axis=0,
        )
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            accuracy_df.to_excel(
                writer, sheet_name=f"accuracy_value", index=True
            )
        with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
            mape_unfold_df.to_excel(writer, sheet_name="mape_unfold_df")
        mylog.info(f"[Total] accuracy_evaluation.xlsx 已保存本地!")

    return (
        mae_dict,
        mse_dict,
        mape_dict,
        mape_unfold_df,
    )
