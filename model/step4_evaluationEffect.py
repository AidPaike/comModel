# 1.jshint语法通过率
# 2.生成数据自身的重复率
# 3.生成数据与训练集的重复率
# 4.生成数据的平均行数
import sys
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(BASE_DIR)
# 测试的模型，随机200个方法头，每个头生成50个方法，共1万个方法。
from multiprocessing.dummy import Pool as ThreadPool
import subprocess
import sys
import time
import tempfile
import os

from workline.model.step3_generationTextPipe import generationTextPipe

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def cmd_jshint(temp_file_path):
    """
    使用jshint对生成的function进行检查\n
    :param temp_file_path: 临时文件位置
    :return: 语法正确返回true,语法错误返回false
    """
    # cmd = ['timeout', '60s', 'jshint', temp_file_path]
    cmd = ['timeout', '10s', 'jshint', '-c', '/root/Comfort_all/data/.jshintrc', temp_file_path]

    if sys.platform.startswith('win'):  # 假如是windows
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    else:  # 假如是linux
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    # if stdout:
    #     print(stdout)
    # if stderr:
    #     print("error")
    #     print(stderr)

    if stdout.__len__() > 0:
        jshint_flag = False
    else:  # 通过了检查，此时 test_file_name中就是美化后的代码
        jshint_flag = True
        # print(f"{temp_file_path}right!")
    return jshint_flag


def testJshintPassRate(function):
    global testJshintPassRateSet
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        temp_file_path = tmpfile.name
        # print(temp_file_name)  # /tmp/tmp73zl8gmn
        fine_code = 'var NISLFuzzingFunc = ' + function
        # fine_code = function

        tmpfile.write(fine_code.encode())
        tmpfile.seek(0)
        # tmpTxt = tmpfile.read().decode()
        # print(tmpTxt)
        result = cmd_jshint(temp_file_path)
        if result:
            testJshintPassRateSet.add(function)


def repetitionRateGeneratedDataItself(allFunctions):
    noRepeatFunctions = set(allFunctions)
    noRepeatFunctionsSize = len(noRepeatFunctions)
    return noRepeatFunctionsSize


def generateDataWithRepetitionRateTrainingSet(function):
    # trainDataFile = '/root/Comfort_all/data/datasets/fzy_train1.txt'
    trainDataFile = '/root/Comfort_all/data/datasets/train_data_bos.txt'
    with open(trainDataFile, 'r') as f:
        trainDatasetContents = f.read()
        if function in trainDatasetContents:
            global generateDataWithRepetitionRateTrainingSetCount
            generateDataWithRepetitionRateTrainingSetCount += 1


def averageNumberRowsToGenerateData(function):
    global averageNumberRowsToGenerateDataCount
    averageNumberRowsToGenerateDataCount += len(function.splitlines())


def multithreadedAnalysis(function):
    testJshintPassRate(function)
    generateDataWithRepetitionRateTrainingSet(function)


# model_name = "/root/Comfort_all/data/train_model/distilgpt2_fzy1/checkpoint-30000"
model_name = "/root/Comfort_all/data/train_model/distilgpt2_new/checkpoint-640000"
# model_name = "/root/Comfort_all/data/train_model/distilgpt2_finetune/checkpoint-160000"

num = 50

max_length = 512

temperature = 1

p = 0.9

k = 0

start_gen = time.time()

# print(f'---------------第{i+1}次测试---------------')
prefixList = []

# prefix1 = """function("""
# prefix2 = """function(a,b"""
prefix3 = """function(b,"""
# prefix4 = """function(c,"""
# prefix5 = """function(d,"""
# prefix6 = """function(e,"""
# prefix7 = """function(f,"""
# prefix8 = """function(g,"""
# prefix9 = """function(h,"""
# prefix10 = """function(i,"""
# prefix11 = """function(g,"""
# prefix12 = """function(h,"""
# prefix13 = """function(q,"""
# prefix14 = """function(w,"""
# prefix15 = """function(e,"""
# prefixList.append(prefix1)
# prefixList.append(prefix2)
prefixList.append(prefix3)
# prefixList.append(prefix4)
# prefixList.append(prefix5)
# prefixList.append(prefix6)
# prefixList.append(prefix7)
# prefixList.append(prefix8)
# prefixList.append(prefix9)
# prefixList.append(prefix10)
# prefixList.append(prefix11)
# prefixList.append(prefix12)
# prefixList.append(prefix13)
# prefixList.append(prefix14)
# prefixList.append(prefix15)
allFunctions = generationTextPipe(model_name_or_path=model_name, prefixList=prefixList, num_return_sequences=100)
# for item in allFunctions:
#     print(item)
end_time = time.time()

totalSize = len(allFunctions)

print(f'max length为{max_length}')

print(f'总共生成{totalSize}个方法,生成速度:{int(len(allFunctions) / (end_time - start_gen))}个/秒')

generateDataWithRepetitionRateTrainingSetCount = 0

testJshintPassRateSet = set()

averageNumberRowsToGenerateDataCount = 0

pool = ThreadPool()

pool.map(multithreadedAnalysis, allFunctions)

pool.close()

pool.join()

noRepeatFunctionsSize = repetitionRateGeneratedDataItself(allFunctions)

print("生成的用例语法正确率为{:.2%},".format(len(testJshintPassRateSet) / totalSize))

print("生成数据本身的重复率为{:.2%}".format(1 - noRepeatFunctionsSize / totalSize))

print("生成数据与训练集的重复率为{:.2%}".format(generateDataWithRepetitionRateTrainingSetCount / totalSize))

# 统计通过语法检查的代码行数
for testJshintPassRate in testJshintPassRateSet:
    averageNumberRowsToGenerateData(testJshintPassRate)

print("语法正确方法的平均行数为{}行".format(int(averageNumberRowsToGenerateDataCount / totalSize)))
