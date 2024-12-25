"""
单因子预测模型建模：lstm_single
多因子：lstm_multi
"""

import copy
from enum import Enum

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from utils.log import mylog


def get_scaler():
    """
    :return: 一个scaler对象
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()

    return scaler


class EnumForecastPattern(Enum):
    """lstm预测的模式"""

    ONE = 1
    TWO = 2  # 预测


class LSTMModel(nn.Module):
    """lstm模型"""

    def __init__(self, input_dim, hidden_dim, layer_num, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.layer_num, x.size(0), self.hidden_dim
        ).requires_grad_()
        c0 = torch.zeros(
            self.layer_num, x.size(0), self.hidden_dim
        ).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class LSTM:
    """lstm"""

    def __init__(
        self,
        pattern: int = EnumForecastPattern.TWO.value,
        pre_steps: int = 1,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 100,
        layer_num: int = 1,
        look_back: int = 10,
        default_epochs: int = 150,
        learning_rate: float = 0.01,
    ):
        self.pattern = pattern  # 预测模式。=1：滚动预测presteps期。=2：一次性预测presteps期
        self.pre_steps = pre_steps

        # lstm net参数
        self.input_dim = input_dim  # input特征数：=1只有价格因子 ，=3 三个因子（包含价格因子本身）
        self.output_dim = output_dim  # output特征数，只有价格因子
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.look_back = look_back  # 每个特征的样本长度
        # train参数
        self.default_epochs = default_epochs
        self.learning_rate = learning_rate

        # 其他参数
        self.batch_size = 32
        self.lr_step_size = 50
        self.lr_gamma = 0.5

    def create_trainx_trainy(
        self,
        train_array: np.ndarray,
    ):
        """
        从历史序列中创建所有样本x和对应的样本y
        :param train_array: 2-d。 当单因子预测时，shape=(len, 1)，当多因子预测时，shape=(len, factor_num)，其中第一列为预测标的
        :param look_back: 每个样本的x数据长度
        :return:
        """
        # 从历史序列中划分出每个样本的x和y
        all_sample_x, all_sample_y = [], []
        samples_num = (
            train_array.shape[0] - self.look_back
        )  # train_array中能划分出的样本总数
        # 构造每个sample
        for sample_i in range(samples_num):
            sample_x = train_array[
                sample_i : sample_i + self.look_back, :
            ]  # 长度为look_back的单个样本的x：look_back长度的单因子or多因子
            sample_y = train_array[sample_i + self.look_back, :]  # 预测标的列
            all_sample_x.append(
                sample_x
            )  # 当单因子预测时，=3-d, (samples_num, look_back, 1)；# 当多因子预测时，=3-d, (samples_num, look_back，factor_num)；
            all_sample_y.append(
                sample_y
            )  # 当单因子预测时，=2-d, (samples_num, 1)；# 当多因子预测时，(samples_num, factor_num)

        # mylog.info(f'len__all_sample_x: \n{np.array(all_sample_x).shape}')
        # mylog.info(f'len__all_sample_y: \n{np.array(all_sample_y).shape}')
        return np.array(all_sample_x), np.array(all_sample_y)

    def create_trainx_trainy2(self, train_array: np.ndarray):
        all_sample_x, all_sample_y = [], []
        samples_num = train_array.shape[0] - self.look_back - self.pre_steps
        for sample_i in range(samples_num):
            sample_x = train_array[sample_i : sample_i + self.look_back, :]
            sample_y = train_array[
                sample_i
                + self.look_back : sample_i
                + self.look_back
                + self.pre_steps,
                [0],
            ]
            all_sample_x.append(sample_x)
            all_sample_y.append(sample_y)

        # mylog.info(f'len__all_sample_x: \n{np.array(all_sample_x).shape}')
        # mylog.info(f'len__all_sample_y: \n{np.array(all_sample_y).shape}')
        return np.array(all_sample_x), np.array(all_sample_y)

    def train(
        self,
        train_x_array: np.ndarray,
        train_y_array: np.ndarray,
    ):
        """
        训练lstm模型
        :param train_x_array:
        :param train_y_array:
        :return:
        """
        # 1 转换历史训练样本batch
        # 转换为PyTorch张量
        # tensor_train_x = torch.tensor(train_x_array, dtype=torch.float32).view(-1, self.look_back, 1)  # when batch_first=True, shape=(batch_size, look_back, input_dim=input_factor_num)
        tensor_train_x = torch.tensor(train_x_array, dtype=torch.float32).view(
            -1, self.look_back, self.input_dim
        )  # when batch_first=True, shape=(batch_size, look_back, input_dim=input_factor_num)
        tensor_train_y = torch.tensor(
            train_y_array,
            dtype=torch.float32,
            # ).view(-1, self.pre_steps)  # pattern=2时
        ).view(
            -1, self.output_dim
        )  # pattern=1 or 2

        # mylog.info(f'tensor_train_x: \n{tensor_train_x}')
        # mylog.info(f'len(tensor_train_x): \n{len(tensor_train_x)}')
        # mylog.info(f'tensor_train_y: \n{tensor_train_y}')
        # mylog.info(f'len(tensor_train_y): \n{len(tensor_train_y)}')

        # 2 创建样本池和batch加载器
        train_xy_pool = TensorDataset(tensor_train_x, tensor_train_y)
        batch_xy_loader = DataLoader(
            train_xy_pool, batch_size=self.batch_size, shuffle=True
        )  # iterator

        # 3 训练lstm
        lstm_model = LSTMModel(
            self.input_dim, self.hidden_dim, self.layer_num, self.output_dim
        )  # pattern=1 or 2
        # lstm_model = LSTMModel(self.input_dim, self.hidden_dim, self.layer_num, self.pre_steps)  # pattern=2时
        loss_func = nn.MSELoss()  # mse 损失函数
        optimizer = optim.Adam(
            lstm_model.parameters(), lr=self.learning_rate
        )  # 优化器adam
        lr_scheduler = StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )  # lr调度器, 每50个epoch将学习率乘以0.1
        for epoch in range(self.default_epochs):
            # 训练阶段
            lstm_model.train()
            # 从样本池中加载随机batch
            loss_mean = 0
            batch_i = 0
            for batch_xy in batch_xy_loader:
                batch_i += 1
                batch_ins, batch_y = batch_xy
                # 前向
                optimizer.zero_grad()
                batch_outs = lstm_model(batch_ins)
                # 反向和迭代更新model
                loss = loss_func(batch_outs, batch_y)
                loss.backward()
                optimizer.step()

                loss_mean += loss
            loss_mean /= batch_i
            lr_scheduler.step()  # 更新学习率

            if (epoch + 1) % 10 == 0:
                mylog.info(
                    f"<lstm training...> epoch:{epoch + 1}/{self.default_epochs}, loss:{loss_mean:.8f}"
                )

        return lstm_model

    def modeling(self, train_df: pd.DataFrame):
        """
        构建lstm模型
        :param train_df: 历史训练序列,单列
        :return: 训练好的lstm模型
        """
        # 1 预处理训练数据
        train_array = train_df.values

        # 各因子分别归一化 # lstm对input的尺度敏感
        scaled_train_array = copy.deepcopy(train_array)
        for factor_i in range(train_array.shape[1]):
            scaler = get_scaler()
            scaled_train_array[:, factor_i : factor_i + 1] = (
                scaler.fit_transform(train_array[:, factor_i : factor_i + 1])
            )

        # 2 构建历史数据的样本x和样本y
        create_trainx_trainy_pattern_map = {
            EnumForecastPattern.ONE.value: self.create_trainx_trainy,
            EnumForecastPattern.TWO.value: self.create_trainx_trainy2,
        }
        train_x_array, train_y_array = create_trainx_trainy_pattern_map.get(
            self.pattern
        )(scaled_train_array)

        # 3 转换和训练数据
        lstm_model = self.train(train_x_array, train_y_array)

        return lstm_model  # lstm模型参数

    def forecast(
        self,
        history_df: pd.DataFrame,
        lstm_model: LSTMModel,
        pre_steps: int = 1,
    ):
        """
        基于训练好的lstm_model预测未来几步
        :param lstm_model: 训练好的模型
        :param history_df: 用于预测的历史序列
        :param pre_steps: 预测步数
        :return:
        """
        if len(history_df) < self.look_back:
            mylog.warning(f"历史序列长度不够，无法预测")
            raise Exception(f"历史序列长度不够，无法预测")

        # 1 取出历史序列中最近的look_back个数据组成一个sample_x
        his_deque = history_df.iloc[-self.look_back :].values.astype(
            float
        )  # 2-darray, shape=(look_back, factor_num)
        # astype(float)不能丢，否则his_deque为int，会将标准化的值强制为int

        # 2 逐列标准化（lstm对input的尺度敏感）
        scaled_his_deque = copy.deepcopy(
            his_deque
        )  # 2-darray, shape=(look_back, factor_num)
        scalers = []
        for factor_i in range(0, his_deque.shape[1]):
            scaler = get_scaler()
            scaled_his_deque[:, factor_i : factor_i + 1] = (
                scaler.fit_transform(
                    scaled_his_deque[:, factor_i : factor_i + 1]
                )
            )  # 2-d, shape=(len, factor_num). # scalar expect 2dim array, 根据pre_steps滚动
            scalers.append(scaler)

        # 2 预测pre_steps步
        # 构造第一步预测的input，即一个sample_x
        scaled_input_ndarray = scaled_his_deque.reshape(
            1, self.look_back, self.input_dim
        )  # 1指只有一个sample_x
        predictions = []  # 存放标的序列的pre_steps步预测值
        lstm_model.eval()  # 评估环境
        with torch.no_grad():
            if self.pattern == EnumForecastPattern.ONE.value:
                # 预测模式一： 逐步预测。从第2步开始，用含预测值的sample_x 预测下一期
                for step in range(pre_steps):
                    # mylog.info(f'------- pre_step:{step}/{pre_steps} -------')
                    # mylog.info(f'scaled_his_deque:\n{scaled_his_deque}')
                    # mylog.info(f'scaled_input_ndarray: \n{scaled_input_ndarray}')

                    # 2darray to tensor
                    scaled_input_tensor = torch.tensor(
                        scaled_input_ndarray, dtype=torch.float32
                    ).view(-1, self.look_back, self.input_dim)
                    # lstm预测
                    scaled_tensor_out = lstm_model(
                        scaled_input_tensor
                    )  # tensor shape=(1, output_dim)
                    # mylog.info(f'scaled_tensor_out: \n{scaled_tensor_out}')

                    # 保存当前step的标的预测值
                    prediction_2darray = scalers[0].inverse_transform(
                        scaled_tensor_out.numpy()[:, :1]
                    )  # scalar expect 2dim array
                    predictions.append(round(prediction_2darray.item(), 6))
                    # mylog.info(f'predictions_list:\n{predictions}')

                    # 更新scaled_input_ndarray
                    scaled_his_deque = np.append(
                        scaled_his_deque, scaled_tensor_out.numpy(), axis=0
                    )  # 从tensor(1,1,1)提取出标量
                    # mylog.info(f'updated scaled_his_deque: \n{scaled_his_deque}')
                    scaled_input_ndarray = scaled_his_deque[
                        -self.look_back :
                    ].reshape(
                        1, self.look_back, self.output_dim
                    )  # 1指只有一个sample_x
                    # mylog.info(f'updated scaled_input_2darray: \n{scaled_input_ndarray}')

            elif self.pattern == EnumForecastPattern.TWO.value:
                # 预测模式二：直接预测Presteps期
                # 2darray to tensor
                scaled_input_tensor = torch.tensor(
                    scaled_input_ndarray, dtype=torch.float32
                ).view(-1, self.look_back, self.input_dim)
                # lstm预测
                scaled_tensor_out = lstm_model(
                    scaled_input_tensor
                )  # tensor shape=(1, output_dim)
                # mylog.info(f'scaled_tensor_out: \n{scaled_tensor_out}')

                # 保存presteps期的预测值
                prediction_2darray = scalers[0].inverse_transform(
                    scaled_tensor_out.numpy()
                )  # scalar expect 2dim array
                for i in prediction_2darray[0]:
                    predictions.append(round(i, 6))
                # mylog.info(f'predictions_list:\n{predictions}')

            else:
                raise ValueError(f"pattern只有1和2两种选择")

        return predictions  # list


class LSTMSingle(LSTM):
    """单因子预测的lstm"""

    def __init__(
        self, pattern=EnumForecastPattern.TWO.value, pre_steps: int = 1
    ):

        if (
            pattern == EnumForecastPattern.ONE.value
        ):  # 滚动预测pre_steps期，每次仅预测一期，因此output_dim=1
            output_dim = 1
        elif (
            pattern == EnumForecastPattern.TWO.value
        ):  # 一次性预测pre_steps期，每次直接预测pre_steps期
            output_dim = pre_steps
        else:
            raise Exception(f"pattern只有1和2两种选择")

        super().__init__(
            pattern=pattern,  # 预测模式
            pre_steps=pre_steps,
            # lstm net参数
            input_dim=1,  # input特征数：=1只有价格因子 ，=3 三个因子（包含价格因子本身）
            output_dim=output_dim,
            hidden_dim=50,
            layer_num=2,
            look_back=15,  # 每个特征的样本长度
            # train参数
            default_epochs=200,
            learning_rate=0.01,
        )


class LSTMMultiple(LSTM):
    """多因子预测的lstm（也可以进行单因子）"""

    def __init__(
        self,
        pattern=EnumForecastPattern.TWO.value,
        factor_num: int = 1,
        pre_steps: int = 1,
    ):

        if (
            pattern == EnumForecastPattern.ONE.value
        ):  # 滚动预测pre_steps期，每次仅预测一期，因此output_dim=因子个数，即只有一期的值但有多个因子
            output_dim = factor_num
        elif (
            pattern == EnumForecastPattern.TWO.value
        ):  # 一次性预测pre_steps期，每次直接预测pre_steps期的价格序列值，即因子只有一个但有presteps期
            output_dim = pre_steps
        else:
            raise Exception(f"pattern只有1和2两种选择")

        super().__init__(
            pattern=pattern,  # 预测模式
            pre_steps=pre_steps,
            # lstm net参数
            input_dim=factor_num,  # input特征数：=1只有价格因子 ，=3 三个因子（包含价格因子本身）
            output_dim=output_dim,  # output特征数(pattern=1)或预测期数(pattern=2)
            hidden_dim=100,
            layer_num=1,
            look_back=10,  # 每个特征的样本长度
            # train参数
            default_epochs=200,
            learning_rate=0.01,
        )


# lstm_single = LSTMSingle()


if __name__ == "__main__":
    pass
    # path = r"../data/钢材new.csv"
    # data1 = pd.read_csv(
    #     path,
    #     usecols=[
    #         "日期",
    #         "热轧板卷4.75mm(raw)",
    #     ],
    #     index_col=["日期"],
    # )
    # data1.index = pd.to_datetime(data1.index)
    # data1.sort_index(inplace=True)
    #
    # # 划分train_df和test_df
    # from forecast_manager import get_train_test_df_by_forecast_daterange
    #
    # train_df, test_df = get_train_test_df_by_forecast_daterange(
    #     data1, pre_start_date="2024-08-20", pre_end_date="2024-08-22"
    # )
    # # mylog.info(f'train_df:\n{train_df},{type(train_df)}')
    # # mylog.info(f'test_df:\n{test_df},{type(test_df)}')
    #
    # # lstm_model = lstm_single.modeling(train_df=train_df)
    # # # mylog.info(f'lstm_model:\n{lstm_model}')
    # # pre_list = lstm_single.forecast(history_df=train_df, lstm_model=lstm_model, pre_steps=1)
    # # # mylog.info(f'pre_list:\n{pre_list}')
    #
    # lstm_multiple = LSTMMultiple(factor_num=len(train_df.columns))
    #
    # lstm_multiple_model = lstm_multiple.modeling(train_df=train_df)
    # pre_list = lstm_multiple.forecast(
    #     history_df=train_df, lstm_model=lstm_multiple_model, pre_steps=1
    # )