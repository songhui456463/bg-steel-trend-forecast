"""
从日频数据中生成周频和月频数据，备用
"""

import pandas as pd
from typing import List, Dict
import os

from utils.log import mylog
from utils.enum_family import EnumFreq
from utils.data_read import read_x_by_map
from factor.factor_resampling import auto_resampling, check_freq


def extract_week_from_day(price_day_df):
    """
    从价格的日频序列中提取出该价格的周频序列
    :param day_df: index列为datetime日期，仅有一列price
    :return: 返回所提取得到的周频价格序列
    """
    # base 周频索引 (如果当前base的日期范围不够长，则换一个base_week_df)
    base_week_df = pd.read_excel(
        r"..\data\市场数据\D- 重交沥青产能利用率（周）.xlsx"
    ).iloc[
        3:, [0, 1]
    ]  # 2014-01-10 -- 2024-10-11
    base_week_df = base_week_df.infer_objects()
    base_week_df.columns = ["日期", "重交沥青：产能利用率：中国（周）"]
    base_week_df.set_index(["日期"], drop=True, inplace=True)
    mylog.info(f"base_week_df:\n{base_week_df}")
    mylog.info(f"type(base_week_df.index:\n{base_week_df.index}")

    # 从日频价格中提取出周频价格
    price_week_df = auto_resampling(base_df=base_week_df, todo_df=price_day_df)
    mylog.info(f"price_week_df:\n{price_week_df}")


def multi_extract_from_day(
    to_extract_name: list[str],
    to_freq: List[EnumFreq],
    base_seq_info_dict: Dict[EnumFreq, dict],
    file_name_dict: Dict[EnumFreq, str],
):
    """
    从日频价格序列中提取出周频和月频价格序列，保存为静态。
    :param to_extract_name:
    :return:
    """
    # 分别指定周频和月频的base index（daterange小于等于price序列的daterange）
    # base_seq_info_dict: Dict[EnumFreq, dict] = {
    #     EnumFreq.WEEK: {'factor_name': '重交沥青：产能利用率：中国（周）', 'start_date': None},
    #     EnumFreq.MONTH: {'factor_name': '粗钢:产量:当月值', 'start_date': '2005-10-01'},
    # }
    for freq in to_freq:
        mylog.info(f"current_to_frep:{freq.value}")
        mylog.info(f"current_to_frep_base_name:{base_seq_info_dict.get(freq)}")

        # 读取base序列
        base_df = read_x_by_map(
            factor_name=base_seq_info_dict.get(freq).get("factor_name"),
            start_date=base_seq_info_dict.get(freq).get("start_date"),
        )
        # mylog.info(f'base_df:\n{base_df}')

        # 分别创建周频和月频的表格
        table = pd.DataFrame(index=base_df.index)
        table.index.name = "指标名称"  # 使符合同花顺/mysteel下载的excel的标准格式：'指标名称'所在列为日期列，'指标名称'所在行为各因子名称

        for price_name in to_extract_name:
            # 读取price_df
            price_day_df = read_x_by_map(factor_name=price_name)
            # 降采样为周频和月频
            downsampling_price_df = auto_resampling(
                base_df=base_df, todo_df=price_day_df
            )
            # 保存至周频和月频table中
            table = pd.concat([table, downsampling_price_df.round(1)], axis=1)

        # 修改列名（必须，否则生成map时会被日频key覆盖）
        if freq == EnumFreq.WEEK:
            table.rename(
                columns=lambda x: x.replace("（日）", "（周）"), inplace=True
            )
        else:
            table.rename(
                columns=lambda x: x.replace("（日）", "（月）"), inplace=True
            )
        mylog.info(f"{freq.value}_table:\n{table}")

        # 保存table
        file_path = r"..\data\市场数据"
        file_path = os.path.join(file_path, file_name_dict.get(freq))
        table.to_csv(file_path, index=True, encoding="utf-8-sig")
        mylog.info(f"{freq.value} table 写入完毕！")


if __name__ == "__main__":
    to_extract_name = [
        # 【冷轧】
        # '国际冷轧板卷汇总价格：美国钢厂（中西部）（日）',
        # '国际冷轧板卷汇总价格：美国进口（CFR）（日）',
        # '国际冷轧板卷汇总价格：欧盟钢厂（日）',
        # '国际冷轧板卷汇总价格：欧盟进口（CFR）（日）',
        # '国际冷轧板卷汇总价格：日本市场（日）',
        # '国际冷轧板卷汇总价格：日本出口（FOB）（日）',
        # '国际冷轧板卷汇总价格：印度进口（日）',
        # '国际冷轧板卷汇总价格：中东进口（迪拜CFR）（日）',
        # '国际冷轧板卷汇总价格：独联体出口（FOB黑海）（日）',
        # '国际冷轧板卷汇总价格：南美出口（FOB）（日）',
        # '国际冷轧板卷汇总价格：中国市场（日）',
        # '国际冷轧板卷汇总价格：中国出口（FOB）（日）',
        # 【热轧】
        # '国际热轧板卷汇总价格：美国进口（CFR）（日）',
        # '国际热轧板卷汇总价格：欧盟钢厂（日）',
        # '国际热轧板卷汇总价格：欧盟进口（CFR）（日）',
        # '国际热轧板卷汇总价格：日本市场（日）',
        # '国际热轧板卷汇总价格：日本出口（FOB）（日）',
        # '国际热轧板卷汇总价格：印度出口（FOB）（日）',
        # '国际热轧板卷汇总价格：土耳其出口（FOB）（日）',
        # '国际热轧板卷汇总价格：东南亚进口（CFR）（日）',
        # '国际热轧板卷汇总价格：独联体出口（FOB黑海）（日）',
        # '国际热轧板卷汇总价格：中东进口（迪拜CFR）（日）',
        # '国际热轧板卷汇总价格：南美出口（FOB）（日）',
        # '国际热轧板卷汇总价格：中国市场（日）',
        # '国际热轧板卷汇总价格：中国出口（FOB）（日）',
        # 【各等权平均标的序列】
        "冷轧无取向硅钢：50WW1300：0.5*1200*C：市场价：等权平均（日）",
        "冷轧取向硅钢：30Q120：0.3*980*C：市场价：等权平均（日）",
        "冷卷：SPCC：1*1250*C：市场价：等权平均（日）",
        "热轧板卷：Q235B：5.75*1500*C：市场价：等权平均（日）",
        "热轧酸洗板卷：SPHC：3*1250*C：市场价：等权平均（日）",
        "中厚板：Q235B：20mm：价格指数：辽宁（日）",
        "螺纹钢：HRB400E：Φ20：汇总价格：上海（日）",
        "无取向电工钢：中低牌号：牌号等权平均：0.5*1200*C：出厂价：本钢集团（日）",
    ]
    new_file_name_dict = {
        # EnumFreq.WEEK: 'Mysteel国际冷轧板卷汇总价格-周.csv',
        # EnumFreq.MONTH: 'Mysteel国际冷轧板卷汇总价格-月.csv',
        # EnumFreq.WEEK: 'Mysteel国际热轧板卷汇总价格-周.csv',
        # EnumFreq.MONTH: 'Mysteel国际热轧板卷汇总价格-月.csv',
        EnumFreq.WEEK: "Mysteel各标的序列等权平均-周.csv",
        EnumFreq.MONTH: "Mysteel各标的序列等权平均-月.csv",
    }

    to_freq = [
        EnumFreq.WEEK,
        EnumFreq.MONTH,
    ]
    base_seq_info_dict: Dict[EnumFreq, dict] = {
        EnumFreq.WEEK: {
            "factor_name": "重交沥青：产能利用率：中国（周）",
            "start_date": None,
        },  # 周频的base
        EnumFreq.MONTH: {
            "factor_name": "粗钢:产量:当月值",
            "start_date": "2005-10-01",
        },  # 月频的base
    }
    multi_extract_from_day(
        to_extract_name,
        to_freq=to_freq,
        base_seq_info_dict=base_seq_info_dict,
        file_name_dict=new_file_name_dict,
    )
