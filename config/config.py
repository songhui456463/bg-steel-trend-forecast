"""
总config
"""

import datetime
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BASE_PATH: str = str(
        Path(__file__).absolute().parent.parent
    )  # project path
    # 创建每次运行的输出目录
    _OUTPUT_DIR_PATH_INITIALIZED: bool = False  # 标志变量。
    _OUTPUT_DIR_PATH: str = os.path.join(
        os.path.join(BASE_PATH, "outputs"),
        f"RunResults{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    )

    # 日志相关
    LOG_DIR_NAME: str = "logs"  # 日志文件夹名
    LOG_FILE_NAME: str = "app.log"  # 日志文件名
    LOG_LEVEL: str = "INFO"  # 日志等级：'INFO','DEBUG'
    LOG_WHEN: str = (
        "midnight"  # 间隔时间: S:秒 M:分 H:小时 D:天 W:每星期（interval==0时代表星期一） midnight: 每天凌晨
    )
    LOG_INTERVAL: int = (
        1  # 日志旋转周期, 和 LOGGER_WHEN 配合使用，例如: LOGGER_WHEN='S', LOGGER_INTERVAL=5, 则每5秒切分一次日志
    )
    LOG_BACKUP_COUNT: int = 10  # 最大备份数, 0为不限制最大备份个数

    # 1 predata params
    # from preprocess.preconfig import preconfig

    # 2 testing params
    # 3 predict params
    # 4 evaluation params

    @property
    def OUTPUT_DIR_PATH(self):
        # 第一次访问OUTPUT_DIR_PATH时，创建输出目录
        if not self._OUTPUT_DIR_PATH_INITIALIZED:
            os.makedirs(self._OUTPUT_DIR_PATH, exist_ok=True)
            # 更新标志变量
            self._OUTPUT_DIR_PATH_INITIALIZED = True
        return self._OUTPUT_DIR_PATH


settings = Settings()
if __name__ == "__main__":
    print(settings.BASE_PATH)
    # print(settings.OUTPUT_DIR_PATH)
    pass
