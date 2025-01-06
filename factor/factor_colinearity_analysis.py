"""
多协从因子共线性：去除多共线性的因子，筛选出有代表性的因子
"""

import copy
import matplotlib.pyplot as plt
import operator
import os.path
import pandas as pd
import statsmodels.api as sm

from config.config import settings
from factor.factor_config import FactorConfig
from utils.log import mylog

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def factor_colinearity_filter(
    xs_df: pd.DataFrame,
    n_clusters: int,
    vif_thred: int,
    vif_max_cycle: int,  # 循环迭代求vif的最大迭代次数
) -> pd.DataFrame:
    """
    检验所有协从因子的共线性，筛选出有代表性的因子
    :param xs_df: 没有date索引的多列协从因子，各列协从因子的频度一致。由于提前期不同，所以没有对齐dateindex(没有行索引)
    :return: 筛选后的 没有日期索引的keyfactors多列df
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import SpectralClustering
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    allfactors_num = len(xs_df.columns)
    xs_df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(xs_df),
        columns=xs_df.columns,
        index=xs_df.index,
    )  # z-score，kmeans对量纲敏感

    # 1 所有协从因子的相关系数矩阵
    corr_mat = xs_df_scaled.corr()  # df
    # mylog.info(f"allfactors_corr_mat:\n{corr_mat}")
    mylog.info(f"进入共线性分析的allfactors个数：{allfactors_num}")

    # 2 按照corr初筛
    temp_corr_mat = copy.deepcopy(corr_mat)
    filted_factor_names = list(xs_df.columns)
    flag = 1
    while True:
        print(f"---------------- flag={flag} ----------------")
        high_corr_count_dict = {}
        high_corr_thred = 0.8  # 0.9

        for col_name in temp_corr_mat.columns:
            high_corr_count = sum(temp_corr_mat[col_name] > high_corr_thred)
            high_corr_count_dict[col_name] = high_corr_count
        high_corr_count_dict = dict(
            sorted(
                high_corr_count_dict.items(),
                key=operator.itemgetter(1),
                reverse=True,
            )
        )
        print(f"high_corr_count_dict:\n{high_corr_count_dict}")

        if not high_corr_count_dict:
            break
        if flag >= (allfactors_num * 0.45):
            break

        max_factor_name, max_count = max(
            high_corr_count_dict.items(), key=operator.itemgetter(1)
        )
        filted_factor_names.remove(max_factor_name)
        # print(f'filted_factor_names:{filted_factor_names}')

        temp_corr_mat = xs_df_scaled[filted_factor_names].corr()
        # print(f'temp_corr_mat:\n{temp_corr_mat}')
        flag += 1

    filted_xs_df_scaled = xs_df_scaled[filted_factor_names]
    dist_mat = 1 - temp_corr_mat

    # 3 聚类分组 选择合适的聚类方法 以及 clu数量 很重要
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
    vif_res_df["factor_name"] = filted_xs_df_scaled.columns
    vif_res_df.index = range(1, len(vif_res_df) + 1)
    # mylog.info(f'vif_res_df:\n{vif_res_df}')

    # 2 计算各因子的vif，并 3 排除共线性高的因子
    for clu in set(vif_res_df["clu"].values):
        # mylog.info(f'\n============== clu={clu} ===============\n')

        # 取出每一簇的xs_df
        cur_clu_factor = vif_res_df.loc[
            vif_res_df["clu"] == clu, ["factor_name"]
        ].values.flatten()
        cur_clu_xs_df = filted_xs_df_scaled[cur_clu_factor]

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
                )
                vif_res_df.loc[
                    vif_res_df["factor_name"]
                    == cur_clu_xs_df_copy.columns[col_i],
                    f"clu_vif_{cycle_i}",
                ] = vif

            # 保留vif值小的因子，准备进入下一轮
            reserved_factor = vif_res_df.loc[
                condition_clu
                & (vif_res_df[f"clu_vif_{cycle_i}"] <= vif_thred),
                ["factor_name"],
            ].values.flatten()  # 当只有两个因子时，vif值相同，但仍有大于10和小于10之分

            # 针对所有vif值都很大（比如e10, inf），而没有vif小的因子，则排除掉最高vif的因子
            if len(reserved_factor) <= 2:
                # 排除掉vif最大的因子
                temp_vif_res_df = copy.deepcopy(vif_res_df)
                nonatemp = (
                    temp_vif_res_df[
                        (temp_vif_res_df["clu"] == clu)
                        & (temp_vif_res_df[f"clu_vif_{cycle_i}"].notna())
                    ]
                ).copy()
                max_vif_factor_idx = nonatemp[f"clu_vif_{cycle_i}"].idxmax()
                if (
                    nonatemp[f"clu_vif_{cycle_i}"].loc[max_vif_factor_idx]
                    > vif_thred
                ):
                    nonatemp.drop(max_vif_factor_idx, inplace=True)
                reserved_factor = nonatemp["factor_name"].values

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
                reserved_factor = cur_clu_xs_df_copy.columns.tolist()
                reserved_factor.remove(max_vif_factor)  # 排除掉vif值最大的因子
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
                cycle_i += 1
                continue

            # 更新
            cur_clu_xs_df_copy = cur_clu_xs_df_copy.loc[:, reserved_factor]
            cycle_i += 1

        mylog.info(f"vif_res_df:\n{vif_res_df}")
    os.makedirs(
        os.path.join(settings.OUTPUT_DIR_PATH, r"factor_analysis"),
        exist_ok=True,
    )
    vif_res_df.to_csv(
        os.path.join(
            settings.OUTPUT_DIR_PATH, r"factor_analysis\vif_res_df.csv"
        ),
        encoding="utf-8-sig",
        index=True,
    )

    # 4 返回有代表性的因子们的序列
    filted_factor_name = vif_res_df.loc[
        vif_res_df["reserved"] == True, ["factor_name"]
    ].values.flatten()
    filted_xs_df = xs_df[filted_factor_name]
    # mylog.info(f'filted_xs_df:\n{filted_xs_df}')

    return filted_xs_df


if __name__ == "__main__":
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
