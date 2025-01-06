"""
从本地数据目录中生成因子路径map
"""

import copy
import os
import pandas as pd
import warnings

from factor.factor_resampling import check_freq
from forecasting.local_data_map import factor_location_map
from utils.enum_family import EnumFreq
from utils.log import mylog

# 全局忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def get_factor_path_from_folder(folder_path):
    """
    获取folder_path目录下的所有xlsx或xls结尾的文件中的指标名称（因子名称）
    :param folder_path:
    :return: {'factor_name':{'path': .. , 'col_idx': ..}}
    """
    # 遍历文件夹中的所有文件
    map_dict = {}
    for filename in os.listdir(folder_path):
        if (
            filename.endswith(".xlsx")
            or filename.endswith(".xls")
            or filename.endswith(".csv")
        ):
            file_path = os.path.join(folder_path, filename)
            file_path = rf"{file_path}"
            mylog.info(f"========= file_path: {file_path}")
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                data = pd.read_excel(file_path).dropna(how="all")
            else:
                data = pd.read_csv(file_path).dropna(how="all")

            # 找到‘指标名称’这一行的行索引，取出这一行
            index_of_target_value = data.index[
                data.iloc[:, 0] == "指标名称"
            ].tolist()  # 找到第0列中值为 '指标名称' 的行索引
            factor_name_df = copy.deepcopy(data.iloc[index_of_target_value, :])

            if factor_name_df.empty:  # 针对自生成的csv等
                factor_name_df = pd.concat(
                    [
                        factor_name_df,
                        pd.DataFrame(
                            [factor_name_df.columns],
                            columns=factor_name_df.columns,
                        ),
                    ],
                    axis=0,
                )

            # 找到包含'指标名称'的列
            columns_to_ignore = [
                col for col in data.columns if (data[col] == "指标名称").any()
            ]
            factor_name_df.dropna(axis=1, how="all", inplace=True)

            # 逐列找到对应列的相关info
            for col_idx in range(factor_name_df.shape[1]):
                if factor_name_df.columns[col_idx] in columns_to_ignore:
                    continue
                factor_name = factor_name_df.iloc[0, col_idx]

                col_df = data.iloc[:, [0, col_idx]]
                col_df.iloc[:, 0] = pd.to_datetime(
                    col_df.iloc[:, 0], errors="coerce", format=None
                )  # 注意：只能用下表索引 iloc
                col_df = col_df.loc[col_df.iloc[:, 0].notna()].dropna(
                    how="any"
                )  # 去除该银子的空值（主要是针对首尾空值）

                map_dict[factor_name] = {
                    "path": file_path,
                    "col_idx": col_idx,
                    "first_date": col_df.iloc[0, 0].strftime("%Y/%m/%d"),
                    "end_date": col_df.iloc[-1, 0].strftime("%Y/%m/%d"),
                }
                mylog.info(
                    f"col_idx:{col_idx}, factor_name:{factor_name}, first_date: {col_df.iloc[0, 0]}, end_date: {col_df.iloc[-1, 0]}"
                )

    mylog.info(f"=============================")
    mylog.info(f"map_dict:\n{map_dict}")
    return map_dict


if __name__ == "__main__":
    pass
    folder_path = r"..\data\市场数据"
    # folder_path = r'..\data\标的数据'
    map_dict = get_factor_path_from_folder(folder_path)
