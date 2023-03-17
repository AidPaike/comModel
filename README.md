# comModel

## 模型训练及统计基本指标
### 下载环境
`pip install -r requirements.txt -i https://pypi.douban.com/simple`

### 01 下载模型
[01_downloadModel.py](workspace/train_model/01_downloadModel.py) 更换model_name即可。

### 02 训练模型
[02_trainModel.py](workspace/train_model/02_trainModel.py) `nohup 02_trainModel.py model_config.json >{model_name}.out 2>&1 ` 

### 03 模型基本指标对比
[03_evaluateEffection.py](workspace/train_model/03_evaluateEffection.py) 测试用例正确率、重复率等指标



## 模型覆盖率统计

直接运行[coverageCombine.py](workspace/engineCoverage/coverageCombine.py) 即可，需要保证微调后的模型已经在对应路径。

# 文件说明
[distilgpt2_config.json](workspace/train_model/distilgpt2_config.json)  训练distilgpt2模型的配置文件  
[gpt2_config.json](workspace/train_model/gpt2_config.json)  训练gpt2模型的配置文件  
[distilgpt2_finetune.out](workspace/train_model/distilgpt2_finetune.out)  训练gpt2模型过程日志  
[gpt2_finetune.out](workspace/train_model/gpt2_finetune.out)  训练gpt2模型过程日志  
[modelConfig.ini](workspace/train_model/modelConfig.ini)  评价模型的配置文件，非特殊情况不用更改    
[coverageCombine.py](workspace/engineCoverage/coverageCombine.py)   覆盖率统计及合并  
[coverageCombine.ini](workspace/engineCoverage/coverageCombine.ini)  覆盖率统计及合并配置文件
