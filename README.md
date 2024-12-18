### 宝钢价格趋势预测项目

### 一、目录结构说明
```.
├── README.md
├── config                                              # 基础配置文件夹
│   └── config.py                                       # 基础配置设置
├── data                                                # 数据目录
├── factor                                              # 因子分析模块
│   ├── factor_colinearity_analysis.py                  # 因子共线性分析模块
│   ├── factor_config.py                                # 因子分析模块参数配置
│   ├── factor_correlation_analysis.py                  # 因子相关性分析模块
│   ├── factor_enums.py                                 # 因子分析枚举类
│   ├── factor_manager.py                               # 因子分析程序主入口文件
│   └── factor_resampling.py                            # 重采样模块
├── forecasting                                         # 预测模块
│   ├── forecast_config.py                              # 预测参数配置文件
│   ├── forecast_manager.py                             # 单次价格预测模块
│   ├── forecast_trend.py                               # 滚动周期预测模块
│   ├── local_data_map.py                               # 本地文件夹文件路径映射
│   ├── manager.py                                      # 预测程序调用主入口文件
│   ├── modeling_arima.py                               # arima模型
│   ├── modeling_holtwinters.py                         # holtwinters模型
│   ├── modeling_var.py                                 # var模型
│   ├── modeling_weight.py                              # 加权模型
│   ├── results_assessment.py                           # 结果评估模块
│   └── simplefit.py                                    # 基础模型
├── logs                                                # 日志存储目录
├── outputs                                             # 结果输出目录
├── preprocess                                          # 预处理模块
│   ├── check_missing.py                                # 缺失值处理模块
│   ├── check_outlier.py                                # 异常值处理模块
│   ├── pre_enums.py                                    # 预处理过程中的枚举值模块
│   ├── preconfig.py                                    # 预处理参数配置文件模块
│   ├── pretesting.py                                   # 预检测模块
│   ├── repair_missing.py                               # 缺失值修复模块
│   ├── repair_outlier.py                               # 异常值修复模块
│   └── target.py                                       # 预测目标序列处理模块
├── requirements.txt                                    # 项目依赖
└── utils                                               # 工具模块文件夹
    ├── data_read.py                                    # 数据读取模块
    ├── date_utils.py                                   # 日期序列生成模块
    ├── enum_family.py                                  # 枚举值类模块
    ├── genarate_price_data.py                          # 价格序列生成模块
    └── log.py                                          # 日志模块
```

### 二、使用说明
#### 2.1 预处理
- 缺失值检查
```
python preprocess/check_missing.py
```
- 缺失值处理
```
python preprocess/repair_missing.py
```
- 异常值检查
``` 
python preprocess/check_outlier.py
```
- 异常值处理
```
python preprocess/repair_outlier.py
```

#### 2.2 因子分析
- 因子共线性分析
```
python factor/factor_colinearity_analysis.py
```
- 因子相关性分析
```
python factor/factor_correlation_analysis.py
```
- 因子重采样
```
python factor/factor_resampling.py
```

#### 2.3 预测
1. 修改参数

   编辑`forecasting/forecast_config.py`, 修改对应的频度的参数, 如数据价格序列(TARGET_NAME)、因子分析区间(ANALYSE_Y_STARTDATE、ANALYSE_Y_ENDDATE)、预测开始时间(PRE_START_DATE)、滚动步数(ROLL_STEPS)、预测步数(PRE_STEPS)等

2. 执行预测
```
python forecasting/manager.py
```

3. 查看结果

   结果保存在 `outputs/RunResultsYYYYMMDDHHmmSS` 的文件夹中, 每次运行会单独生成文件夹保存

