"""
多协从因子共线性：去除多共线性的因子，筛选出有代表性的因子
"""

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.log import mylog

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def factor_colinearity_filter(
    xs_df: pd.DataFrame,
    n_clusters: int = 1,
    vif_thred: int = 10,
    vif_max_cycle: int = 10,  # 循环迭代求vif的最大迭代次数
) -> pd.DataFrame:
    """
    检验所有协从因子的共线性，筛选出有代表性的因子
    :param xs_df: 没有date索引的多列协从因子，各列协从因子的频度一致，由于提前期不同，所以没有对齐dateindex(没有行索引)
    :param n_clusters:
    :param vif_thred:
    :param vif_max_cycle:
    :return: 没有日期索引的keyfactors多列df
    """
    # 1 所有协从因子的相关系数矩阵
    xs_df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(xs_df),
        columns=xs_df.columns,
        index=xs_df.index,
    )  # z-score，kmeans对量纲敏感
    corr_mat = xs_df_scaled.corr()  # df
    mylog.info(f"allfactors_corr_mat:\n{corr_mat}")
    dist_mat = 1 - corr_mat

    # 2 聚类分组 选择合适的聚类方法 以及 clu数量 很重要
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
        cur_clu_xs_num = cur_clu_xs_df.shape[1]
        for col_i in range(cur_clu_xs_num):
            vif = variance_inflation_factor(cur_clu_xs_df.values, col_i)
            vif_res_df.loc[
                vif_res_df["factor_name"] == cur_clu_factor[col_i], "clu_vif"
            ] = vif
        vif_res_df.loc[vif_res_df["clu_vif"] <= vif_thred, "reserved"] = True

        mylog.info(f"vif_res_df:\n{vif_res_df}")

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
