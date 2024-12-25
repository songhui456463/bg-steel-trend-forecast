""""
单因子预测模型建模：transformer_single
多因子：transformer_multi
"""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.log import mylog


class TransformerModel(nn.Module):
    """
    Transformer模型
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embedding_size,
        layer_num,
        head_num,
        dim_forward,
        dropout_rate,
    ):
        """
        初始化Transformer模型

        Args:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            embedding_size (int): embedding维度
            layer_num (int): Transformer层的数量
            head_num (int): 多头注意力的头数
            dim_forward (int): 前向传播维度
            dropout_rate (float): dropout率
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=head_num,
            dim_feedforward=dim_forward,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=layer_num
        )
        self.fc = nn.Linear(embedding_size, output_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 输出张量
        """
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_size)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_size)
        x = x[:, -1, :]  # (batch_size, embedding_size)
        x = self.fc(x)
        return x


class Transformer:
    """
    Transformer类
    """

    def __init__(
        self,
        input_dim: int = 1,  # 输入维度
        output_dim: int = 1,  # 输出维度
        embedding_size: int = 128,  # embedding维度
        layer_num: int = 6,  # Transformer层的数量
        head_num: int = 8,  # 多头注意力的头数
        dim_forward: int = 512,  # 前向传播维度
        dropout_rate: float = 0.1,  # dropout率
        look_back: int = 15,  # 每个特征的样本长度
        default_epochs: int = 200,  # 默认的训练轮数
        learning_rate: float = 1e-4,  # 学习率
        batch_size: int = 32,  # batch大小
        lr_step_size: int = 10,  # 学习率衰减步长
        lr_gamma: float = 0.1,  # 学习率衰减系数
    ):
        """
        初始化Transformer类
        """
        # transformer net参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_size = embedding_size
        self.layer_num = layer_num
        self.head_num = head_num
        self.dim_forward = dim_forward
        self.dropout_rate = dropout_rate
        self.look_back = look_back
        # train参数
        self.default_epochs = default_epochs
        self.learning_rate = learning_rate
        # 其他参数
        self.batch_size = batch_size
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

    def create_trainx_trainy(
        self,
        train_array: np.ndarray,
    ):
        """
        从历史序列中创建所有样本x和对应的样本y

        Args:
            train_array (np.ndarray): 历史序列数据

        Returns:
            np.ndarray: 样本x和样本y
        """
        all_sample_x, all_sample_y = [], []
        samples_num = train_array.shape[0] - self.look_back
        for sample_i in range(samples_num):
            sample_x = train_array[sample_i : sample_i + self.look_back, :]
            sample_y = train_array[sample_i + self.look_back, :]
            all_sample_x.append(sample_x)
            all_sample_y.append(sample_y)

        return np.array(all_sample_x), np.array(all_sample_y)

    def train(
        self,
        train_x_array: np.ndarray,
        train_y_array: np.ndarray,
    ):
        """
        训练Transformer模型

        Args:
            train_x_array (np.ndarray): 训练样本x
            train_y_array (np.ndarray): 训练样本y

        Returns:
            TransformerModel: 训练好的Transformer模型
        """

        tensor_train_x = torch.tensor(train_x_array, dtype=torch.float32).view(
            -1, self.look_back, self.input_dim
        )
        tensor_train_y = torch.tensor(train_y_array, dtype=torch.float32).view(
            -1, self.output_dim
        )
        train_xy_pool = TensorDataset(tensor_train_x, tensor_train_y)
        batch_xy_loader = DataLoader(
            train_xy_pool, batch_size=self.batch_size, shuffle=True
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transformer_model = TransformerModel(
            self.input_dim,
            self.output_dim,
            self.embedding_size,
            self.layer_num,
            self.head_num,
            self.dim_forward,
            self.dropout_rate,
        )
        transformer_model.to(device)
        loss_func = nn.MSELoss()
        loss_func.to(device)
        optimizer = optim.Adam(
            transformer_model.parameters(), lr=self.learning_rate
        )
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        for epoch in range(self.default_epochs):
            transformer_model.train()
            loss_mean = 0
            batch_i = 0
            for batch_xy in batch_xy_loader:
                batch_i += 1
                batch_ins, batch_y = batch_xy
                batch_ins, batch_y = batch_ins.to(device), batch_y.to(device)
                optimizer.zero_grad()
                batch_outs = transformer_model(batch_ins)
                loss = loss_func(batch_outs, batch_y)
                loss.backward()
                optimizer.step()
                loss_mean += loss
            loss_mean /= batch_i
            lr_scheduler.step()
            if (epoch + 1) % 10 == 0:
                mylog.info(
                    f"<transformer training...> epoch:{epoch + 1}/{self.default_epochs}, loss:{loss_mean:.8f}"
                )

        return transformer_model

    def modeling(self, train_df: pd.DataFrame):
        """
        构建Transformer模型

        Args:
            train_df (pd.DataFrame): 训练数据

        Returns:
            TransformerModel: 训练好的Transformer模型
        """
        train_array = train_df.values
        scaled_train_array = copy.deepcopy(train_array)
        for factor_i in range(train_array.shape[1]):
            scaler = MinMaxScaler()
            scaled_train_array[:, factor_i : factor_i + 1] = (
                scaler.fit_transform(train_array[:, factor_i : factor_i + 1])
            )

        train_x_array, train_y_array = self.create_trainx_trainy(
            scaled_train_array
        )
        transformer_model = self.train(train_x_array, train_y_array)

        return transformer_model

    def forecast(
        self,
        history_df: pd.DataFrame,
        transformer_model: TransformerModel,
        pre_steps: int = 1,
    ):
        """
        Args:
            history_df (pd.DataFrame): 历史数据
            transformer_model (TransformerModel): 训练好的Transformer模型
            pre_steps (int): 预测步数

        Returns:
            list: 预测值
        """
        if len(history_df) < self.look_back:
            mylog.info(f"历史序列长度不够，无法预测")
            raise Exception(f"历史序列长度不够，无法预测")

        his_deque = history_df.iloc[-self.look_back :].values.astype(float)
        scaled_his_deque = copy.deepcopy(his_deque)
        scalers = []
        for factor_i in range(0, his_deque.shape[1]):
            scaler = MinMaxScaler()
            scaled_his_deque[:, factor_i : factor_i + 1] = (
                scaler.fit_transform(
                    scaled_his_deque[:, factor_i : factor_i + 1]
                )
            )
            scalers.append(scaler)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transformer_model.to(device)
        scaled_input_ndarray = scaled_his_deque.reshape(
            1, self.look_back, self.input_dim
        )
        predictions = []
        transformer_model.eval()
        with torch.no_grad():
            for step in range(pre_steps):
                scaled_input_tensor = (
                    torch.tensor(scaled_input_ndarray, dtype=torch.float32)
                    .view(-1, self.look_back, self.input_dim)
                    .to(device)
                )
                scaled_tensor_out = transformer_model(scaled_input_tensor)
                prediction_2darray = scalers[0].inverse_transform(
                    scaled_tensor_out.cpu().numpy()[:, :1]
                )
                predictions.append(round(prediction_2darray.item(), 6))
                # 更新scaled_input_ndarray
                scaled_his_deque = np.append(
                    scaled_his_deque, scaled_tensor_out.cpu().numpy(), axis=0
                )  # 从tensor(1,1,1)提取出标量
                # mylog.info(f'updated scaled_his_deque: \n{scaled_his_deque}')
                scaled_input_ndarray = scaled_his_deque[
                    -self.look_back :
                ].reshape(
                    1, self.look_back, self.output_dim
                )  # 1指只有一个sample_x
                # mylog.info(f'updated scaled_input_2darray: \n{scaled_input_ndarray}')

        return predictions


class TransformerSingle(Transformer):
    """
    单因子预测的Transformer
    """

    def __init__(self):
        """
        初始化单因子预测的Transformer
        """
        super().__init__(
            input_dim=1,
            output_dim=1,
            embedding_size=64,
            layer_num=6,
            head_num=8,
            dim_forward=256,
            dropout_rate=0.1,
            look_back=10,
            default_epochs=100,
            learning_rate=1e-4,
            batch_size=32,
            lr_step_size=10,
            lr_gamma=0.1,
        )


class TransformerMultiple(Transformer):
    """
    多因子预测的Transformer
    """

    def __init__(self, factor_num):
        """
        初始化多因子预测的Transformer
        """
        super().__init__(
            input_dim=factor_num,
            output_dim=factor_num,
            embedding_size=64,
            layer_num=6,
            head_num=8,
            dim_forward=256,
            dropout_rate=0.1,
            look_back=10,
            default_epochs=10,
            learning_rate=1e-4,
            batch_size=32,
            lr_step_size=100,
            lr_gamma=0.1,
        )
