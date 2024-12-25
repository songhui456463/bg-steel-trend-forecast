"""
检查标的序列，差分平稳后的性质：是否还具有自相关性、acf/pacf
关系到是否还要用自回归相关的预测方法
"""

import os
import traceback
from typing import List

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config.config import settings
from preprocess.pre_enums import EnumPretestingReturn
from preprocess.pretesting import (
    autocorr_test,
    stationary_test,
)
from utils.data_read import read_x_by_map
from utils.legalization import sanitize_filename
from utils.log import mylog


def check_targets_property(target_name_list: List[str]):

    try:
        for price in target_name_list:
            # 读取标的序列
            price_df = read_x_by_map(
                factor_name=price, start_date="2015-02-01", end_date=None
            )
            pricename_forpath = sanitize_filename(price)

            # 检测原序列的平稳性，并差分至平稳
            price_station_res = stationary_test(price_df)
            mylog.info(f"price_station_res:{price_station_res}")

            # 若原序列不平稳，则差分后，检测差分后序列的自相关性
            if not price_station_res.get(
                EnumPretestingReturn.stationaryTest_is_stationary
            ):
                stationary_d = price_station_res.get(
                    EnumPretestingReturn.stationaryTest_stationary_d
                )
                for d in range(stationary_d):
                    diff_price_df = price_df.diff(1).dropna()

                diffprice_autocorr_res = autocorr_test(diff_price_df)
                mylog.info(f"diffprice_autocorr_res:{diffprice_autocorr_res}")

                fig, axs = plt.subplots(2, 1, figsize=(16, 12))

                # 绘图：原序列
                axs[0].plot(price_df.index, price_df[price])
                axs[0].set_title(f"{price}--原始序列")
                axs[0].set_xlabel("日期")
                axs[0].set_ylabel("价格")
                axs[0].grid()
                # axs[0].legend()
                axs[0].tick_params(axis="x", rotation=45)

                # 绘图：差分后的平稳序列
                axs[1].plot(diff_price_df.index, diff_price_df[price])
                axs[1].set_title(f"{price}--{stationary_d}阶差分序列(已平稳)")
                axs[1].set_xlabel("日期")
                axs[1].set_ylabel("价格")
                axs[1].grid()
                # axs[1].legend()
                axs[1].tick_params(axis="x", rotation=45)
                plt.tight_layout()
                fig.savefig(
                    os.path.join(
                        settings.OUTPUT_DIR_PATH,
                        f"{pricename_forpath}———原序列和差分序列.png",
                    )
                )
                # plt.show()

                # 绘图：差分后的平稳序列的acf/pacf
                fig2, axs2 = plt.subplots(2, 1, figsize=(16, 12))
                plot_acf(diff_price_df, lags=20, ax=axs2[0])
                plot_pacf(diff_price_df, lags=20, ax=axs2[1])
                fig2.suptitle(f"{price}--{stationary_d}阶差分序列(已平稳)")
                plt.tight_layout()
                fig2.savefig(
                    os.path.join(
                        settings.OUTPUT_DIR_PATH,
                        f"{pricename_forpath}———平稳后的acf和pacf.png",
                    )
                )
                # plt.show()

            # 若原序列平稳，则直接检测原序列自相关性
            else:
                price_autocorr_res = autocorr_test(price_df)
                mylog.info(f"price_autocorr_res:{price_autocorr_res}")

                # 绘图：原序列
                fig, axs = plt.subplots()
                # plt.figure(figsize=(16, 6))

                # 绘图：原序列
                axs.plot(price_df.index, price_df[price])
                axs.set_title(f"{price}--原始序列（已平稳）")
                axs.set_xlabel("日期")
                axs.set_ylabel("价格")
                axs.grid()
                # plt.legend()
                axs.tick_params(axis="x", rotation=45)
                plt.tight_layout()
                fig.savefig(
                    os.path.join(
                        settings.OUTPUT_DIR_PATH,
                        f"{pricename_forpath}———原序列.png",
                    )
                )
                # plt.show()

                # 绘图：原序列的acf/pacf
                fig2, axs2 = plt.subplots(2, 1, figsize=(16, 12))
                plot_acf(price_df, lags=20, ax=axs2[0])
                plot_pacf(price_df, lags=20, ax=axs2[1])
                fig2.suptitle(f"{price}--原始序列（已平稳）")
                plt.tight_layout()
                fig2.savefig(
                    os.path.join(
                        settings.OUTPUT_DIR_PATH,
                        f"{pricename_forpath}———平稳后的acf和pacf.png",
                    )
                )
                # plt.show()

    except Exception as e:
        mylog.error(f"error traceback: \n{traceback.format_exc()}")


if __name__ == "__main__":
    pass
    # 标的序列 20241212

    # target_name_list = list(price_location_map.keys())

    target_name_list = [
        "国际冷轧板卷汇总价格：中国市场（日）",
        "国际冷轧板卷汇总价格：中国市场（周）",
        "国际冷轧板卷汇总价格：中国市场（月）",
        "国际热轧板卷汇总价格：中国市场（日）",
        "国际热轧板卷汇总价格：中国市场（周）",
        "国际热轧板卷汇总价格：中国市场（月）",
        "冷轧无取向硅钢：50WW1300：0.5*1200*C：市场价：等权平均（日）",
        "冷轧无取向硅钢：50WW1300：0.5*1200*C：市场价：等权平均（周）",
        "冷轧无取向硅钢：50WW1300：0.5*1200*C：市场价：等权平均（月）",
        "冷轧取向硅钢：30Q120：0.3*980*C：市场价：等权平均（日）",
        "冷轧取向硅钢：30Q120：0.3*980*C：市场价：等权平均（周）",
        "冷轧取向硅钢：30Q120：0.3*980*C：市场价：等权平均（月）",
        "冷卷：SPCC：1*1250*C：市场价：等权平均（日）",
        "冷卷：SPCC：1*1250*C：市场价：等权平均（周）",
        "冷卷：SPCC：1*1250*C：市场价：等权平均（月）",
        "热轧板卷：Q235B：5.75*1500*C：市场价：等权平均（日）",
        "热轧板卷：Q235B：5.75*1500*C：市场价：等权平均（周）",
        "热轧板卷：Q235B：5.75*1500*C：市场价：等权平均（月）",
        "热轧酸洗板卷：SPHC：3*1250*C：市场价：等权平均（日）",
        "热轧酸洗板卷：SPHC：3*1250*C：市场价：等权平均（周）",
        "热轧酸洗板卷：SPHC：3*1250*C：市场价：等权平均（月）",
        "中厚板：Q235B：20mm：价格指数：辽宁（周）",
        "中厚板：Q235B：20mm：价格指数：辽宁（月）",
        "螺纹钢：HRB400E：Φ20：汇总价格：上海（日）",
        "螺纹钢：HRB400E：Φ20：汇总价格：上海（周）",
        "螺纹钢：HRB400E：Φ20：汇总价格：上海（月）",
        "无取向电工钢：中低牌号：牌号等权平均：0.5*1200*C：出厂价：本钢集团（日）",
        "无取向电工钢：中低牌号：牌号等权平均：0.5*1200*C：出厂价：本钢集团（周）",
        "无取向电工钢：中低牌号：牌号等权平均：0.5*1200*C：出厂价：本钢集团（月）",
    ]

    # print(f'target_name_list: \n{target_name_list}')
    check_targets_property(target_name_list)
