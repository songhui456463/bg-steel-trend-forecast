import pandas as pd


class PriceCorrelationAnalyzer:
    def __init__(self, data):
        """
        初始化 PriceCorrelationAnalyzer 类

        :param data: 包含多个价格数据列的 DataFrame，第一列为关注的价格
        """
        self.data = data

    def handle_missing_values(self, method="drop"):
        """
        处理缺失值

        :param method: 处理缺失值的方法 ('drop' 或 'fill')
        """
        if method == "drop":
            self.data = self.data.dropna()
        elif method == "fill":
            self.data = self.data.fillna(self.data.mean())  # 用均值填充缺失值
        else:
            raise ValueError("Invalid method. Choose 'drop' or 'fill'.")

    def calculate_correlations(self, max_lag):
        """
        计算每一列与第一列之间不同提前期的相关性

        :param max_lag: 最大提前期
        :return: 包含最高相关性的提前期及其相关性的信息
        """
        results = {}

        for column in self.data.columns[1:]:
            correlations = []

            for lag in range(0, max_lag + 1):  # 从0开始，表示提前期
                if lag < len(self.data):
                    corr = self.data[self.data.columns[0]].corr(
                        self.data[column].shift(lag)
                    )
                    correlations.append((lag, corr))

            correlation_df = pd.DataFrame(
                correlations, columns=["Lag", "Correlation"]
            )
            if not correlation_df.empty:  # 确保有计算结果
                max_corr_row = correlation_df.loc[
                    correlation_df["Correlation"].idxmax()
                ]
                results[column] = (
                    max_corr_row["Lag"],
                    max_corr_row["Correlation"],
                )

        return results


# 示例用法
if __name__ == "__main__":

    df = pd.read_csv("../data/钢材.csv", index_col="日期")

    # 创建 PriceCorrelationAnalyzer 对象
    analyzer = PriceCorrelationAnalyzer(df)

    # 处理缺失值，这里选择用均值填充
    analyzer.handle_missing_values(method="fill")  # 可以选择 'drop' 或 'fill'

    # 计算相关性
    results = analyzer.calculate_correlations(max_lag=90)

    # 输出结果
    for column, (lag, correlation) in results.items():
        print(
            f"{column} - 与{df.columns[0]}价格最高相关性的提前期: {lag}, 相关性: {correlation}"
        )
