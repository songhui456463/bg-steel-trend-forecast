"""
使路径名字中的非法字符合法化
"""


def sanitize_filename(filename):
    """
    清理文件名中的非法字符，确保文件名在所有操作系统上都是合法的。
    :param filename: 需要清理的文件名
    :return: 清理后的文件名
    """
    # 定义非法字符及其替换规则
    illegal_chars = {
        # '<': '_lt_',  # 小于符号
        # '>': '_gt_',  # 大于符号
        ":": "：",  # 冒号
        # '"': "'",  # 双引号
        # '/': '_',  # 斜杠
        # '\\': '_',  # 反斜杠
        # '|': '_',  # 竖线
        # '?': '_',  # 问号
        "*": "乘",  # 星号
    }

    # 使用正则表达式替换非法字符
    sanitized = filename
    for char, replacement in illegal_chars.items():
        sanitized = sanitized.replace(char, replacement)
    # 移除开头和结尾的空格
    sanitized = sanitized.strip()
    # 如果文件名为空，返回默认名称
    if not sanitized:
        return "default_filename"
    return sanitized
