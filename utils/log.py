"""
create log tool
"""

import logging
import os
from logging import config, Logger

from config.config import settings


class CreateLog:
    @staticmethod
    def get_logger() -> logging.Logger:
        # log地址
        log_dir_path = os.path.join(
            settings.BASE_PATH, settings.LOG_DIR_NAME
        )  # 日志文件夹路径
        if not os.path.exists(log_dir_path):
            os.mkdir(log_dir_path)
        log_file_path = os.path.join(
            log_dir_path, settings.LOG_FILE_NAME
        )  # 日志文件路径
        logger_name = settings.LOG_FILE_NAME.split(".")[
            0
        ]  # 日志记录器的名称，'my'
        # 参数
        level = settings.LOG_LEVEL
        when = (
            settings.LOG_WHEN
        )  # S:秒,M:分,H:小时,D:天,W:每星期（interval==0时代表星期一）midnight: 每天凌晨
        interval = settings.LOG_INTERVAL
        backup_count = (
            settings.LOG_BACKUP_COUNT
        )  # 备份文件的个数，若超过该值，就会自动删除

        LOGGER_CONFIG = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(module)s.%(funcName)s:[%(lineno)d] | %(message)s"
                },
            },
            "handlers": {
                "default": {
                    "level": level,
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",  # Default is stderr
                },
                "file_handler": {
                    "level": level,
                    "formatter": "standard",
                    "filename": log_file_path,
                    "encoding": "utf-8",
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "when": when,
                    "interval": interval,
                    "backupCount": backup_count,
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["default"],
                    "level": level,
                    "propagate": False,
                },
                f"{logger_name}": {
                    "handlers": ["default", "file_handler"],
                    "level": level,
                    "propagate": False,
                },
            },
        }
        logging.config.dictConfig(LOGGER_CONFIG)
        logger = logging.getLogger(f"{logger_name}")
        return logger


# 初始化日志logger 所有需要日志输出的地方都从此处导入使用
mylog: Logger = CreateLog().get_logger()
if __name__ == "__main__":
    print(mylog)
    aa = "normal info"
    mylog.info(f"{aa}")
    mylog.info("普通信息")
    mylog.debug("调试信息")
    mylog.warning("警告信息")
    mylog.error("错误信息")
    mylog.critical("重大")
