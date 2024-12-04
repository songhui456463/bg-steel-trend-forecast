# -*- coding: utf-8 -*-
# @Project : bg-steel-price-trend-forecast
# @Date    : 2024/11/8 11:29
# @Author  : Xanto
# @File    : date_utils.py
# @Software: PyCharm

from datetime import datetime, timedelta


def generate_date_pairs(
    start_date_str: str, frequency: str, periods: int, interval: int
) -> list[tuple]:
    """
    根据输入的开始日期生成对应的日期序列组合
    :param start_date_str: 开始日期文本
    :param frequency: 频度, 'month', 'week', 'day', 'year'
    :param periods: 生成多少期的数据
    :param interval: 每期的开始和结束日期中间间隔多少周期
    :return: 返回对应的组合数据 [(start_date_str, end_date_str),(start_date_str, end_date_str),]


    Usage:
        >>> start_date = "2022-01-01"
        >>> frequency = "month"  # 'month', 'week', 'day', 'year'
        >>> periods = 6
        >>> interval = 3
        >>> date_pairs = generate_date_pairs(start_date, frequency, periods, interval)
        >>> print(date_pairs)

        [('2022-01-01', '2022-04-01'), ('2022-02-01', '2022-05-01'), ('2022-03-01', '2022-06-01'), ('2022-04-01', '2022-07-01'), ('2022-05-01', '2022-08-01'), ('2022-06-01', '2022-09-01')]
    """

    # 将输入的字符串转换为日期对象
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    # 定义一个空列表来存放生成的日期对
    date_pairs = []

    for i in range(periods):
        # 计算每个周期的结束日期
        if frequency == "month":
            # 按月份增加
            start_month = start_date.month + i
            start_year = start_date.year + (start_month - 1) // 12
            start_month = (start_month - 1) % 12 + 1

            end_month = start_date.month + i + interval
            end_year = start_date.year + (end_month - 1) // 12
            end_month = (end_month - 1) % 12 + 1

            # if end_month > 12:
            #     # end_month -= 12
            #     # start_year += 1
            #     end_year = start_year + ((end_month-2) // 12)
            #     end_month = (end_month-1) % 12 + 1
            # else:
            #     end_year = start_year
            start_date_new = start_date.replace(
                year=start_year, month=start_month, day=1
            )
            end_date = start_date_new.replace(
                year=end_year, month=end_month, day=1
            )

        elif frequency == "week":
            # 按周增加
            start_date_new = start_date + timedelta(weeks=i)
            end_date = start_date_new + timedelta(weeks=interval)

        elif frequency == "day":
            # 按天增加
            start_date_new = start_date + timedelta(days=i)
            end_date = start_date_new + timedelta(days=interval)

        elif frequency == "year":
            # 按年增加
            start_year = start_date.year + i
            end_year = start_year + interval
            start_date_new = start_date.replace(
                year=start_year, day=1, month=1
            )
            end_date = start_date_new.replace(year=end_year, day=1, month=1)
        else:
            raise ValueError("frequency input error")
        # 将生成的日期对添加到列表中
        date_pairs.append(
            (
                start_date_new.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
        )

    return date_pairs


if __name__ == "__main__":
    start_date = "2021-01-01"
    frequency = "month"  # 'month', 'week', 'day', 'year'
    periods = 3
    interval = 3
    date_pairs = generate_date_pairs(start_date, frequency, periods, interval)
    print(date_pairs)
