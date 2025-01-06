"""
价格预测对象：
比如，“热轧475汇总价格”
"""

import copy
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from typing import Optional, Union

from check_missing import check_missing
from check_outlier import check_outliers
from pre_enums import EnumProcessedDF, EnumRepairOutliersMethod
from preconfig import preconfig
from pretesting import (
    autocorr_test,
    gaussian_test,
    stationary_test,
    hetero_test,
)
from repair_missing import is_repairable_missing, repair_missing
from repair_outlier import repair_outliers
from utils.log import mylog


class Target:
    """单变量原始序列的基本信息及处理后的序列"""

    def __init__(
        self,
        origin_df: pd.DataFrame,
        name: str = None,
        start: Union[pd.DatetimeIndex, str] = None,
        end: Union[pd.DatetimeIndex, str] = None,
        source: Optional[str] = None,
        freq: Optional[str] = None,
        expected_dtype: Optional[str] = None,
    ):
        # origin
        self.origin_df = origin_df
        self.name = name  # '热轧475汇总价格'
        self.source = source  # ['local', 'db']
        self.freq = freq  # ['Day','Week','Month','Season','HalfYear','Year']
        self.expected_dtype = expected_dtype  # 值类型是int or float
        # process
        self.start_date = start  # '%Y-%m-%d',包含
        self.end_date = end  # 包含
        self.processed_df = None  # date,origin_col,'missing_type','missing_numerilized','missing_repaired','is_outlier','outlier type','outliers_repaired'
        # 注意：如果'missing_numerilized','missing_repaired'这两列的列名修改后，需要同步修改check_outliers和repair_outliers
        # 注意：因为是检查和处理'missing_repaired'而不是原始price列
        self.fully_repaired_df = None  # 经过缺失异常修复后的df，self.processed_df['outliers_repaired']，变更col_name为self.name

        # result
        self.check_missing_result = None
        self.check_outliers_result = None
        self.test_autocorr_result = None
        self.test_hetero_result = None
        self.test_stationary_result = None
        self.test_gaussian_result = None

        # 根据origin_df初始化self.name, self.start_date,self.end_date, self.processed_df
        self.init_all()

    def init_all(self):
        # 检查创建对象的origin_df形式
        if not isinstance(self.origin_df, pd.DataFrame):
            mylog.error(f"origin_df应该是pd.DataFrame")
            raise TypeError("origin_df应该是pd.DataFrame")
        is_date_index_exist = is_datetime64_any_dtype(self.origin_df.index)
        if not is_date_index_exist:
            mylog.error(f"origin_df的行索引不是datetime")
            raise TypeError("origin_df的行索引不是datetime")
        # 按索引datetime排序
        self.origin_df.sort_index(
            inplace=True, ascending=True
        )  # 按dateindex升序，start_date应该小于等于end_date
        # 初始化 name
        if self.name is None:
            self.name = self.origin_df.columns[0]
        # 初始化 date
        if self.start_date is None:
            self.start_date = self.origin_df.index[0]
        if self.end_date is None:
            self.end_date = self.origin_df.index[-1]
        # 初始化 processed_df
        if self.processed_df is None:
            self.processed_df = copy.deepcopy(self.origin_df)

    def get_name(self):
        return self.name

    def get_start_date(self):
        return self.start_date

    def get_end_date(self):
        return self.end_date

    def update_start_date(self, date):
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        if self.start_date == date:
            mylog.info(f"<{self.name}> 更新start date: 没有改变")
        else:
            mylog.info(
                f"<{self.name}> 更新start date: {self.start_date} 改变为 {date}"
            )
            self.start_date = date

    def update_end_date(self, date):
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        if self.end_date == date:
            mylog.info(f"<{self.name}> 更新end date: 没有改变")
        else:
            mylog.info(
                f"<{self.name}> 更新end date: {self.end_date} 改变为 {date}"
            )
            self.end_date = date

    def update_processed_df(self, df):
        self.processed_df = pd.concat(
            [self.processed_df, df], axis=1, join="outer"
        )
        mylog.info(
            f"<{self.name}> 更新processed_df，新增列：{df.columns.values}"
        )

    def process_check_missing(self):
        """检查缺失值"""
        if (
            EnumProcessedDF.MISSING_NUMERILIZED.value
            not in self.processed_df.columns
        ):
            mylog.info(f"<{self.name}> process check_missing ......")
            (
                all_types,
                numerilized_df,
                valid_start_date,
                valid_end_date,
                missing_results,
            ) = check_missing(
                self.origin_df, self.start_date, self.end_date, preconfig
            )
            # 记录检查结果
            self.update_processed_df(all_types)
            self.update_processed_df(numerilized_df)
            self.update_start_date(valid_start_date)
            self.update_end_date(valid_end_date)
            self.check_missing_result = missing_results
        else:
            mylog.info(f"<{self.name}> missing_checked 已存在")

    def process_repair_missing(self):
        """修补缺失值"""
        if (
            EnumProcessedDF.MISSING_REPAIRED.value
            not in self.processed_df.columns
        ):
            if (
                EnumProcessedDF.MISSING_NUMERILIZED.value
                not in self.processed_df.columns
            ):
                self.process_check_missing()
            mylog.info(f"<{self.name}> process repair_missing ......")
            repaired_df, valid_start_date, valid_end_date = repair_missing(
                self.name,
                self.processed_df[[EnumProcessedDF.MISSING_NUMERILIZED.value]],
                self.start_date,
                self.end_date,
                self.check_missing_result,
                preconfig,
            )
            if repaired_df is not None:  # 修补成功
                # 记录修补结果
                self.update_processed_df(repaired_df)
                self.update_start_date(valid_start_date)
                self.update_end_date(valid_end_date)
        else:
            mylog.info(f"<{self.name}> missing_repaired 已存在")

    def process_check_outliers(self):
        if EnumProcessedDF.IS_OUTLIER.value not in self.processed_df.columns:
            if (
                EnumProcessedDF.MISSING_REPAIRED.value
                not in self.processed_df.columns
            ):
                self.process_repair_missing()
            mylog.info(f"<{self.name}> process check_outliers ......")
            result, df_filtered, valid_start_date, valid_end_date = (
                check_outliers(
                    self.processed_df,
                    self.start_date,
                    self.end_date,
                    preconfig,
                )
            )
            # 更新check_outliers_result,self.start_date,self.end_date
            self.update_start_date(valid_start_date)
            self.update_end_date(valid_end_date)
            self.check_outliers_result = result
            self.update_processed_df(df_filtered)
        else:
            mylog.info(f"<{self.name}> outliers_checked 已存在")

    def process_repair_outliers(self):
        if (
            EnumProcessedDF.OUTLIERS_REPAIRED.value
            not in self.processed_df.columns
        ):
            if (
                EnumProcessedDF.IS_OUTLIER.value
                not in self.processed_df.columns
            ):  # 如果已经做完异常值检测
                self.process_check_outliers()
            mylog.info(f"<{self.name}> process repair_outliers ......")
            df_repaired, valid_start_date, valid_end_date = repair_outliers(
                self.check_outliers_result,
                self.processed_df,
                self.start_date,
                self.end_date,
                EnumRepairOutliersMethod.INTERPOLATE.value,
                preconfig,
            )
            # 更新check_outliers_result,self.start_date,self.end_date
            self.update_start_date(valid_start_date)
            self.update_end_date(valid_end_date)
            self.update_processed_df(df_repaired)
            # 更新fully_repaired_df
            self.fully_repaired_df = self.processed_df[
                [EnumProcessedDF.OUTLIERS_REPAIRED.value]
            ].dropna()
            self.fully_repaired_df.columns = [self.name]
        else:
            mylog.info(f"<{self.name}> outliers_repaired 已存在")

    def get_checked_missing_df(self):
        if (
            EnumProcessedDF.MISSING_NUMERILIZED.value
            not in self.processed_df.columns
        ):
            self.process_check_missing()
        return self.processed_df[
            [self.name, "missing_type", "missing_numerilized"]
        ]  # 注意 不dropna
        # return self.processed_df[[self.name, 'missing_type', 'missing_numerilized']].loc[
        #     (self.processed_df.index >= self.start_date) & (self.processed_df.index <= self.end_date)]

    def get_repaired_missing_df(self):
        if (
            EnumProcessedDF.MISSING_REPAIRED.value
            not in self.processed_df.columns
        ):
            self.process_repair_missing()
        # latest start_date--latest end_date
        return self.processed_df[["missing_repaired"]].loc[
            (self.processed_df.index >= self.start_date)
            & (self.processed_df.index <= self.end_date)
        ]

    def get_checked_outliers_df(self):
        if EnumProcessedDF.IS_OUTLIER.value in self.processed_df.columns:
            return self.processed_df[
                [
                    self.name,
                    EnumProcessedDF.IS_OUTLIER.value,
                    EnumProcessedDF.OUTLIER_TYPE.value,
                ]
            ].dropna()
        else:
            # mylog.info(f"<{self.name}> 异常值检测结果为空！正在进行异常值检测！")
            self.process_check_outliers()
            return self.processed_df[["is_outlier", "outlier_type"]].dropna()

    def get_repaired_outliers_df(self):
        if (
            EnumProcessedDF.OUTLIERS_REPAIRED.value
            in self.processed_df.columns
        ):
            # return self.processed_df[self.name, EnumProcessedDF.OUTLIERS_REPAIRED.value].dropna()
            return self.processed_df[
                [EnumProcessedDF.OUTLIERS_REPAIRED.value]
            ].dropna()
        else:
            # mylog.info(f"<{self.name}> 异常值修复结果为空！正在进行异常值修复！")
            self.process_repair_outliers()
            # return self.processed_df[self.name, EnumProcessedDF.OUTLIERS_REPAIRED.value].dropna()
            return self.processed_df[
                [EnumProcessedDF.OUTLIERS_REPAIRED.value]
            ].dropna()

    def get_fully_repaired_df(self):
        if self.fully_repaired_df is None:
            self.fully_repaired_df = self.get_repaired_outliers_df()
            # 变更列名
            self.fully_repaired_df.columns = [self.name]
        return self.fully_repaired_df

    def update_pretesting(self):
        """对缺失异常处理后的序列进行检验"""
        # 检查待检验的序列是否存在
        # if EnumProcessedDF.OUTLIERS_REPAIRED.value not in self.processed_df.columns:
        #     self.process_repair_outliers()
        # tobetest_df = self.processed_df[[EnumProcessedDF.OUTLIERS_REPAIRED.value]].dropna()
        totest_df = self.get_repaired_outliers_df()
        # 检验
        self.test_autocorr_result = autocorr_test(totest_df)
        self.test_hetero_result = hetero_test(totest_df)
        self.test_stationary_result = stationary_test(totest_df)
        self.test_gaussian_result = gaussian_test(totest_df)
        # 打印
        mylog.info(
            f"pretesting results:\n"
            f"====================================================\n"
            f"test_autocorr_result: {self.test_autocorr_result}\n"
            f"test_hetero_result: {self.test_hetero_result}\n"
            f"test_stationary_result: {self.test_stationary_result}\n"
            f"test_gaussian_result: {self.test_gaussian_result}\n"
            f"====================================================\n"
        )

    def get_test_autocorr_result(self):
        if self.test_autocorr_result is None:
            self.update_pretesting()
        return self.test_autocorr_result

    def get_test_hetero_result(self):
        if self.test_hetero_result is None:
            self.update_pretesting()
        return self.test_hetero_result

    def get_test_stationary_result(self):
        if self.test_stationary_result is None:
            self.update_pretesting()
        return self.test_stationary_result

    def get_test_gaussian_result(self):
        if self.test_gaussian_result is None:
            self.update_pretesting()
        return self.test_gaussian_result

    def __str__(self):
        return f"<name: {self.name}> latest start_date:{self.start_date}, latest end_date: {self.end_date}"

    def __repr__(self):
        return self.__str__()


def check_validity_start_end(
    df: pd.DataFrame,
    start_date: Union[pd.DatetimeIndex, str],
    end_date: Union[pd.DatetimeIndex, str],
):
    """
    检查开始日期和结束日期的有效性，返回更新后的有效日期
    :param df:
    :param start_date:
    :param end_date:
    :return:
    """
    df_copy = copy.deepcopy(df)
    y_name = df_copy.columns[0]
    if start_date >= end_date:
        raise ValueError("start date should be earlier than end date ")

    # 检查start_date和end_date是否在df的日期范围内
    index_date = df_copy.index
    valid_start_date = max(index_date[0], pd.to_datetime(start_date))
    valid_end_date = min(index_date[-1], pd.to_datetime(end_date))

    # 检查有效子序列首尾的缺失值情况，取第一个有效数值的位置为valid start date
    sub_df = df_copy.loc[
        (df_copy.index >= valid_start_date) & (df_copy.index <= valid_end_date)
    ]
    sub_df.loc[:, y_name] = pd.to_numeric(
        sub_df.loc[:, y_name], errors="coerce"
    )
    # valid_index_mask = sub_df.index[sub_df[y_name].apply(lambda x: x > 1)]  # >1 排除掉bool True/False, 但是对于某些因子，负数和0-1小数属于正常值
    valid_index_mask = sub_df.index[
        sub_df[y_name].apply(lambda x: not pd.isna(x))
    ]
    valid_start_date = valid_index_mask[0]
    valid_end_date = valid_index_mask[-1]

    return valid_start_date, valid_end_date


# def preprocess_run(tg: Target):
#     """
#     对单序列对象进行预处理：缺失值和异常值
#     :param tg: 需要预处理缺失值和异常值的序列对象
#     :return:
#     """
#     # 验证开始和结束日期是否有效，并更新为有效日期
#     tg.valid_start_date, tg.valid_end_date = check_validity_start_end(tg.origin_df, tg.start_date, tg.end_date)
#     mylog.info(f'<{tg.name}> start_date: {tg.start_date} -> {tg.valid_start_date.date()}, end_date: {tg.end_date} -> {tg.valid_end_date.date()}')
#
#     # 缺失值处理
#     # is_repairable_miss = is_repairable_missing(tg.origin_df, tg.valid_start_date, tg.valid_end_date, preconfig)
#     tg.repaired_missing_df = repair_missing(tg.origin_df, tg.valid_start_date, tg.valid_end_date, preconfig)
#
#     # 异常值处理
#     tg.repaired_outliers_df = repair_outliers(tg.repaired_missing_df, tg.valid_start_date, tg.valid_end_date, 'replace_median',preconfig)
#
#     tg.fully_repaired_df = tg.repaired_outliers_df
#     return tg.fully_repaired_df
#     # return None


if __name__ == "__main__":
    # 读取数据和创建实例

    # target1
    path = r"../data/钢材new.csv"
    # data1 = pd.read_csv(path, usecols=['日期', '热轧板卷4.75mm',], index_col=['日期'])
    data1 = pd.read_csv(
        path,
        usecols=[
            "日期",
            "铁矿62%Fe现货交易基准价",
        ],
        index_col=["日期"],
    )
    data1.index = pd.to_datetime(data1.index)
    data1.sort_index(inplace=True)
    # y2 = Target(origin_df=data1, start='2019-09-03', end='2023-12-29')
    y2 = Target(origin_df=data1, start="2019-09-05", end="2019-12-31")
    # y2 = Target(origin_df=data1)

    # target2
    # dates = pd.date_range(start='2024-01-01', end='2024-01-13')
    # price_data = {'price': [np.nan, '-4540', 0, 4520, -4600, '%', 4679, 4610, True, '4640', 'text', None, 4670]}
    # data2 = pd.DataFrame(data=price_data, index=dates, columns=['price'])
    # # y2 = Target(origin_df=data2, start='2024-01-01', end='2024-01-12')
    # y2 = Target(origin_df=data2)

    # fully_repaired_df = preprocess_run(tg=y1)
    # fully_repaired_df = preprocess_run(tg=y2)

    mylog.info(f"target--y2: {y2}")
    # y2.process_check_missing()
    # mylog.info(f'processed_df:\n{y2.processed_df}')
    # y2.process_repair_missing()
    # mylog.info(f'processed_df:\n{y2.processed_df}')
    # mylog.info(f'repaired_missing_df sub:\n{y2.get_repaired_missing_df()}')

    # y2.process_check_outliers()
    y2.process_repair_outliers()
    mylog.info(f"repaired_outlier_df :\n{y2.get_repaired_outliers_df()}")

    y2.update_pretesting()
