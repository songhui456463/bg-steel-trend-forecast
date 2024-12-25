"""
保存模型训练的中间结果或结果
"""

import hashlib
import json
import os

from config.config import settings
from utils.legalization import sanitize_filename
from utils.log import mylog


def get_lstm_hash(lstm_model_object):
    """
    根据Lstm_single的参数结构生成唯一的hash
    :param lstm_model_object: lstm_single或lstm_multiple的object
    :return:
    """
    params_dict = lstm_model_object.__dict__

    # 转为json字符串
    params_str = json.dumps(params_dict, ensure_ascii=False)

    # 哈希
    hash_object = hashlib.sha256(params_str.encode())
    hash_id = hash_object.hexdigest()

    return hash_id


def get_lstm_model_path(
    price_history_df,
    model_object,
):
    """
    根据当前训练数据和模型类生成模型文件名
    :param price_history_df: 用来训练模型的序列df
    :param model_object: lstm_single或lstm_multiple的object，用于生成模型结构的hash
    :return:
    """
    # 形如：outputs \lstm \price_name \xxx.pt

    save_base_path = os.path.join(settings.BASE_PATH, "outputs")
    lstm_dir_path = os.path.join(save_base_path, "lstm")

    # 按价格序列的名称创建一个文件夹
    price_name = price_history_df.columns[0]
    legal_price_name = sanitize_filename(price_name)
    price_dir_path = os.path.join(lstm_dir_path, legal_price_name)
    if not os.path.exists(price_dir_path):
        os.makedirs(price_dir_path)

    # 按训练数据创建model文件的名字
    model_hash = get_lstm_hash(model_object)
    cur_model_name = (
        f'{model_hash}__{price_history_df.index[0].strftime("%Y%m%d")}-{price_history_df.index[-1].strftime("%Y%m%d")}'
        + ".pth"
    )
    cur_model_path = os.path.join(price_dir_path, cur_model_name)
    mylog.info(f"current model path: {cur_model_path}")

    return cur_model_path
