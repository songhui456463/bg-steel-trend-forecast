"""
多协从因子共线性：去除多共线性的因子，筛选出有代表性的因子
"""

import copy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils.log import mylog
from factor.factor_config import FactorConfig

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# todo 如果标的因子和协从因子的频度不同怎么处理？ 比如标的价格序列是日频，协从因子是周频或月频
# 处理方法，先将低频的序列降采样，比如日频的价格序列取每周5天的平均值，得到价格的周频序列，再和原本就是周频的协从因子进行相关性分析
# todo 那预测的时候怎么办？比如要预测日频的价格，但有效因子有日频、周频、月频的。
# todo 如果将低频的价格序列降采样到月频，那预测结果就只有月频的； 只能升采样了？


def factor_colinearity_filter(
    xs_df: pd.DataFrame,
    n_clusters: int = 1,
    vif_thred: int = 10,
    vif_max_cycle: int = 10,  # 循环迭代求vif的最大迭代次数
) -> pd.DataFrame:
    """
    检验所有协从因子的共线性，筛选出有代表性的因子
    :param xs_df: 没有date索引的多列协从因子，各列协从因子的频度一致，由于提前期不同，所以没有对齐dateindex(没有行索引)
    :return: 没有日期索引的keyfactors多列df
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import SpectralClustering
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 1 所有协从因子的相关系数矩阵
    xs_df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(xs_df),
        columns=xs_df.columns,
        index=xs_df.index,
    )  # z-score，kmeans对量纲敏感
    corr_mat = xs_df_scaled.corr()  # df
    # mylog.info(f"allfactors_corr_mat:\n{corr_mat}")
    dist_mat = 1 - corr_mat

    # 2 聚类分组  todo 选择合适的聚类方法 以及 clu数量 很重要
    # clusters = KMeans(n_clusters=3, n_init='auto').fit_predict()  # kmeans不能用距离矩阵作输入
    # labels = fcluster(linkage(dist_mat,method='ward'), t=1, criterion=)
    clu_labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize",
    ).fit_predict(
        dist_mat
    )  # 这个
    # clu_labels = hdbscan.HDBSCAN(min_cluster_size=5, metric='precomputed').fit_predict(dist_mat)

    vif_res_df = pd.DataFrame(
        columns=["clu", "factor_name", "reserved"]
    )  # 必有的列
    vif_res_df["clu"] = clu_labels
    vif_res_df["factor_name"] = xs_df.columns
    # mylog.info(f'vif_res_df:\n{vif_res_df}')

    # 2 计算各因子的vif，并 3 排除共线性高的因子
    # vif_thred = 10  # 5, 3
    # vif_thred = 10
    for clu in set(vif_res_df["clu"].values):
        # mylog.info(f'\n============== clu={clu} ===============\n')
        # 取出每一簇的xs_df
        cur_clu_factor = vif_res_df.loc[
            vif_res_df["clu"] == clu, ["factor_name"]
        ].values.flatten()
        # mylog.info(f'cur_clu_factor:\n{cur_clu_factor}')
        cur_clu_xs_df = xs_df[cur_clu_factor]
        # mylog.info(f'cur_clu_xs_df:\n{cur_clu_xs_df}')

        # 计算各因子的vif 并筛选因子
        """法一：简单排除"""
        # cur_clu_xs_num = cur_clu_xs_df.shape[1]
        # for col_i in range(cur_clu_xs_num):
        #     vif = variance_inflation_factor(cur_clu_xs_df.values, col_i)
        #     vif_res_df.loc[vif_res_df['factor_name'] == cur_clu_factor[col_i], 'clu_vif'] = vif
        # vif_res_df.loc[vif_res_df['clu_vif'] <= vif_thred, 'reserved'] = True
        """法二：迭代排除"""
        cur_clu_xs_df_copy = copy.deepcopy(cur_clu_xs_df)
        condition_clu = vif_res_df["clu"] == clu
        cycle_i = 1
        while True:
            # print(f'--------- cucly_i={cycle_i} ---------')
            # 计算当前组的因子vif并保存
            cur_clu_xs_df_copy_num = cur_clu_xs_df_copy.shape[1]
            for col_i in range(cur_clu_xs_df_copy_num):
                vif = variance_inflation_factor(
                    cur_clu_xs_df_copy.values, col_i
                )  # todo 明明因子之间的相关系数并不大，为什么计算出的vif值比较大？？？
                vif_res_df.loc[
                    vif_res_df["factor_name"]
                    == cur_clu_xs_df_copy.columns[col_i],
                    f"clu_vif_{cycle_i}",
                ] = vif
            # mylog.info(f'vif_res_df:\n{vif_res_df}')

            # 保留vif值小的因子，进入下一轮
            reserved_factor = vif_res_df.loc[
                condition_clu
                & (vif_res_df[f"clu_vif_{cycle_i}"] <= vif_thred),
                ["factor_name"],
            ].values.flatten()
            # mylog.info(f'reserved_factor:\n{reserved_factor}')  # 当只有两个因子时，vif值相同，但仍有大于10和小于10之分

            # 停止条件
            if cur_clu_xs_df_copy_num == len(
                reserved_factor
            ):  # 所有因子的最新vif值都小于thred
                vif_res_df.loc[
                    vif_res_df["factor_name"].isin(reserved_factor.tolist()),
                    "reserved",
                ] = True
                break

            if (
                len(reserved_factor)
                == 1  # 不需要再对只有一个因子的xs_df_copy求vif
                or cycle_i >= vif_max_cycle
            ):  # 设置迭代求vif的最大迭代次数
                vif_res_df.loc[
                    vif_res_df["factor_name"].isin(reserved_factor.tolist()),
                    "reserved",
                ] = True
                break

            if (
                len(reserved_factor) == 0
            ):  # 所有因子的最新vif都大于thred：可能是两个因子(相同的vif)，可能是3个及以上因子
                max_vif_factor = vif_res_df.at[
                    vif_res_df[f"clu_vif_{cycle_i}"].idxmax(), "factor_name"
                ]
                # mylog.info(f'max_vif_factor:\n{max_vif_factor}')
                reserved_factor = cur_clu_xs_df_copy.columns.tolist()
                reserved_factor.remove(max_vif_factor)  # 排除掉vif值最大的因子
                # mylog.info(f'reserved_factor:\n{reserved_factor}')
                # 停止
                if (
                    len(reserved_factor) == 0
                ):  # 没有剩余的小于thred的因子，直接结束
                    break
                if len(reserved_factor) == 1:  # 不能对1个因子计算vif
                    vif_res_df.loc[
                        vif_res_df["factor_name"].isin(reserved_factor),
                        "reserved",
                    ] = True
                    break
                # 更新
                cur_clu_xs_df_copy = cur_clu_xs_df_copy.loc[:, reserved_factor]
                # mylog.info(f'cur_clu_xs_df_copy.columns:\n{cur_clu_xs_df_copy.columns}')
                cycle_i += 1
                continue
            # 更新
            cur_clu_xs_df_copy = cur_clu_xs_df_copy.loc[:, reserved_factor]
            # mylog.info(f'cur_clu_xs_df_copy.columns:\n{cur_clu_xs_df_copy.columns}')
            cycle_i += 1

        # mylog.info(f"vif_res_df:\n{vif_res_df}")

    # 4 返回有代表性的因子们的序列
    filted_factor_name = vif_res_df.loc[
        vif_res_df["reserved"] == True, ["factor_name"]
    ].values.flatten()
    filted_xs_df = xs_df[filted_factor_name]
    # mylog.info(f'filted_xs_df:\n{filted_xs_df}')
    return filted_xs_df


if __name__ == "__main__":
    # 示例数据
    # import numpy as np
    # np.random.seed(0)
    # dates = pd.date_range('20200101', periods=100)
    # y_df = pd.DataFrame({'y_name': np.random.normal(1, 4, 100)}, index=dates)
    # xs_df = pd.DataFrame({
    #     'Factor1': np.random.randn(100).cumsum(),
    #     'Factor2': 0.5 * np.random.randn(100).cumsum(),
    #     'Factor3': y_df['y_name'].values - 1,
    #     'Factor4': 0.1 * np.random.randn(100).cumsum(),
    #     'Factor5': 0.8 * np.random.randn(100).cumsum(),
    #     'Factor6': 0.2 * np.random.randn(100).cumsum(),
    #     'Factor7': 0.4 * np.random.randn(100).cumsum(),
    #     'Factor8': 0.6 * np.random.randn(100).cumsum(),
    #     'Factor9': 0.7 * np.random.randn(100).cumsum(),
    #     'Factor10': 0.9 * np.random.randn(100).cumsum()
    # }, index=dates)

    # test
    origin_df = pd.read_csv(
        r"../data/02六品种数据整理-月_test.csv", index_col=["date"]
    )
    y_df = origin_df[["热轧板卷价格"]]
    xs_df = origin_df.drop(columns=["热轧板卷价格"])
    # print(y_df)
    # print(xs_df)

    # corr_filted_xs_df = factor_correlation_filter(y_df=y_df, xs_df=xs_df)

    # colinear_filted_xs_df = factor_colinearity_filter(corr_filted_xs_df)
