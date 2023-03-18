# 1.jshint语法通过率
# 2.生成数据自身的重复率
# 3.生成数据与训练集的重复率
# 4.生成数据的平均行数
# 测试的模型，随机200个方法头，每个头生成50个方法，共1万个方法。

from assemble_tools.callable_processor import CallableProcessor
import sys
import os
import random
from pathlib import Path
import configparser
from multiprocessing.dummy import Pool as ThreadPool
import subprocess
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import time
import re
import logging
import shutil
import uuid
import sqlite3
from tqdm import tqdm
import time

from rich.live import Live
from rich.table import Table

# 获取项目的绝对路径
BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(BASE_DIR)
'''
定义环境变量
'''
# 加载配置文件
cf = configparser.ConfigParser()
cf.read(os.path.join(BASE_DIR, "workspace/temp/modelConfig.ini"), encoding='utf-8')
model_name_logg = cf.get('comModel', 'model_name')
# 初始化日志
logger = logging.getLogger()
# 创建格式器,并将console，handler设置对应的格式
formatter = logging.Formatter(
    '%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
# # 创建console控制台处理器
# console = logging.StreamHandler()
# # 设置日志输出的最低等级,低于当前等级则会被忽略
# console.setLevel(logging.INFO)
# console.setFormatter(formatter)
# 创建handler文件处理器
handler = logging.FileHandler(filename="report/"+model_name_logg+"_report.log", mode="w")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
# 将处理器，添加至日志器中
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# jshintrc 路径
jshintrcPath = os.path.join(BASE_DIR, cf.get('comModel', 'jshintrcPath'))
# 生成模型的路径
model_name_or_path = os.path.join(BASE_DIR, cf.get(
    'comModel', 'model_path'), cf.get('comModel', 'model_name'))
# 获取原始数据，进行重复率的计算
trainDataFile = os.path.join(BASE_DIR, cf.get('comModel', 'trainDataFile'))
# 获取comModel文件
database_path = os.path.join(BASE_DIR, cf.get('comModel', 'database_path'))
# 获取相同模型所有的模型路径
model_list_all = os.listdir(model_name_or_path)
model_list_all = list(filter(lambda x: re.match('checkpoint.*', x) != None, model_list_all))
model_list_all.sort(key=lambda x: int(x.split('-')[1]))
model_list = [os.path.join(model_name_or_path, model_path)
              for model_path in model_list_all]
# 使用哪几个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = cf.get('comModel', 'CUDA_VISIBLE_DEVICES')
# 禁止显示进程分叉告警
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class modelEvaluate:

    def __init__(self) -> None:
        self.testJshintPassRateSet = set()
        self.generateDataWithRepetitionRateTrainingSetCount = 0
        self.averageNumberRowsToGenerateDataCount = 0

    def generationTextPipe(self,
                           model_name_or_path=model_list[-1],
                           prefixList=["""function("""],
                           num_return_sequences=50,
                           max_length=512,
                           temperature=1,
                           p=0.9,
                           k=0,
                           ):
        start = time.time()
        logger.info("正在加载{}模型,大约需要10秒,请稍等".format(model_name_or_path))
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        generator = pipeline(task="text-generation",
                             model=model, tokenizer=tokenizer, device=0, warnings=False)
        logger.info("模型加载完成：{}".format(time.time() - start))
        allFunctions = []
        if num_return_sequences % 100 > 0:
            allGeneration = generator(prefixList, num_return_sequences=num_return_sequences % 100, max_length=max_length,
                                    pad_token_id=tokenizer.eos_token_id, temperature=temperature, k=k, p=p)
            for generationItem in allGeneration:
                for idx, item in enumerate(generationItem):
                    # print('-'*30+'NO.'+ str(idx+1)+'-'*30)
                    # print(item['generated_text'])
                    allFunctions.append(item['generated_text'])
        for batch_sequences in range(num_return_sequences//100):
            allGeneration = generator(prefixList, num_return_sequences=int(100), max_length=max_length,
                                    pad_token_id=tokenizer.eos_token_id, temperature=temperature, k=k, p=p)
            for generationItem in allGeneration:
                for idx, item in enumerate(generationItem):
                    # print('-'*30+'NO.'+ str(idx+1)+'-'*30)
                    # print(item['generated_text'])
                    allFunctions.append(item['generated_text'])
        return allFunctions

    def cmd_jshint(self, temp_file_path):
        """
        使用jshint对生成的function进行检查\n
        :param temp_file_path: 临时文件位置
        :return: 语法正确返回true,语法错误返回false
        """
        # cmd = ['timeout', '60s', 'jshint', temp_file_path]
        cmd = ['timeout', '60s', 'jshint', '-c', jshintrcPath, temp_file_path]

        if sys.platform.startswith('win'):  # 假如是windows
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        else:  # 假如是linux
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    def testJshintPassRate(self, function):
        '''
        记录模型生成的function语法通过率
        '''
        with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
            temp_file_path = tmpfile.name
            # print(temp_file_name)  # /tmp/tmp73zl8gmn
            fine_code = 'var NISLFuzzingFunc = ' + function
            tmpfile.write(fine_code.encode())
            tmpfile.seek(0)
            # tmpTxt = tmpfile.read().decode()
            # print(tmpTxt)
            result = self.cmd_jshint(temp_file_path)
            if result:
                self.testJshintPassRateSet.add(function)

    def jshint_check_testcases(self, all_testcases):
        # start_time = time.time()
        all_testcases_pass = set()
        for testcase in all_testcases:
            testcase_no_print = testcase[:testcase.rfind('\n')]
            # print(testcase_no_print)
            with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
                temp_file_path = tmpfile.name
                # print(temp_file_name)  # /tmp/tmp73zl8gmn
                tmpfile.write(testcase_no_print.encode())
                tmpfile.seek(0)
                # tmpTxt = tmpfile.read().decode()
                # print(tmpTxt)
                result = self.cmd_jshint(temp_file_path)
                if result:
                    all_testcases_pass.add(testcase)
        # end_time = time.time()
        return list(all_testcases_pass)

    def repetitionRateGeneratedDataItself(self, allFunctions):
        '''
        模型自身生成的重复率
        '''
        noRepeatFunctions = set(allFunctions)
        noRepeatFunctionsSize = len(noRepeatFunctions)
        return noRepeatFunctionsSize

    def generateDataWithRepetitionRateTrainingSet(self, function):
        '''
        记录生成的重复率
        '''
        with open(trainDataFile, 'r') as f:
            trainDatasetContents = f.read()
            if function in trainDatasetContents:
                self.generateDataWithRepetitionRateTrainingSetCount += 1

    def averageNumberRowsToGenerateData(self, function):
        '''
        生成数据的平均行数
        '''
        self.averageNumberRowsToGenerateDataCount += len(function.splitlines())

    def multithreadedAnalysis(self, function):
        # 记录语法通过率
        self.testJshintPassRate(function)
        # 记录生成的重复率
        self.generateDataWithRepetitionRateTrainingSet(function)

    def saveJs(self, functions, save_path, times=5, rm=True):
        callable_processor = CallableProcessor()

        def muti_assemble(func):
            try:
                function_assemle_list = set()
                for i in range(times):
                    function_assemle = callable_processor.get_self_calling(
                        func)
                    function_assemle_list.add(function_assemle)
                all_testcases_pass = self.jshint_check_testcases(
                    function_assemle_list)
                # 数据量不是很大，如果用查看文件最大值再命名不太行，因为是多线程，那么只能试试uuid了，选择之后还是uuid3比较合适
                for i in range(len(all_testcases_pass)):
                    name = uuid.uuid1().hex
                    with open(os.path.join(save_path, str(name)+".js"), "wb") as f:
                        f.write(all_testcases_pass[i].encode())
            except:
                pass

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info("目录暂无，新建！{save_path}".format(save_path=save_path))
        else:
            if rm:
                shutil.rmtree(save_path)
                os.makedirs(save_path)
                logger.info("清空文件夹！{save_path}".format(save_path=save_path))

        pool = ThreadPool()
        results = pool.map(muti_assemble, functions)
        pool.close()
        pool.join()

        logger.info("已将生成{functions}测试用例存入{save_path}文件夹".format(
            functions=len(functions), save_path=save_path))
        return 

    def prefixGenerate(self, table_function):
        def getJSfileSize(code: str, cut_max_line: int) -> int:
            lines_list = code.splitlines()
            return min(len(lines_list), cut_max_line)

        def getPrefix(code: str, cut_line: int):
            lines_list = code.splitlines(True)
            line_list_cut = lines_list[0:cut_line]
            function_cut = ''
            for i in line_list_cut:
                function_cut += i
            return function_cut

        def count_var_lines(code: str):
            regex = r'function.*\n( {4}"use strict";\n)?( {4}var.*\n)*'
            matches = re.finditer(regex, code, re.MULTILINE)
            count = 0
            for matchNum, match in enumerate(matches, start=1):
                # print(match.group(0))
                for line in match.group(0).splitlines():
                    count = count + 1
            return count

        prefix_list = []
        for prefix_line in range(count_var_lines(table_function), getJSfileSize(table_function, 10)):
            # Got the prefix
            function_prefix = getPrefix(table_function, prefix_line)
            prefix_list.append(function_prefix)
        return prefix_list

    def sqliteProcess(self):
        conn = sqlite3.connect(database_path)
        c = conn.cursor()

        def getId(c):
            sql = "select ID from prefix"
            result = c.execute(sql).fetchall()
            return result

        def idToTestcase(self, id):
            sql = "select TESTCASE from prefix where ID=:id"
            prames = (id,)
            result = c.execute(sql, prames)
            conn.commit()
            return result.fetchall()
        getId_list = [id[0] for id in getId(c)]
        result = idToTestcase(c, random.choice(getId_list))
        conn.close()
        return result[0][0].strip()

    def main(self, modelevaluate, model, sequence_num, temperature, max_length):
        # 加载生成参数
        model_name = model
        p = cf.get('generateConf', 'p')
        k = cf.get('generateConf', 'k')
        save = cf.get('generateConf', 'save')
        save_path = os.path.join(model_name, "testcases")

        # 记录生成时间
        start_gen = time.time()

        # 续写开头,进行随机生成头部进行续写
        prefixList = []
        prefix = """function("""
        prefixList.append(prefix)
        # 想个好名字
        table_function = self.sqliteProcess()
        prefix_list = self.prefixGenerate(table_function)
        prefixList = prefixList + prefix_list

        allFunctions = modelevaluate.generationTextPipe(
            model_name_or_path=model_name,
            prefixList=prefixList,
            num_return_sequences=sequence_num,
            temperature = temperature
        )

        if save:
            self.saveJs(allFunctions, save_path, rm=True)

        # 生成结束时间
        end_time = time.time()
        # 一共生成方法数量
        totalSize = len(allFunctions)
        # 最大生成长度
        logger.info(f'max length为{max_length}')
        load_model_time = int(len(allFunctions) / (end_time - start_gen))
        logger.info(
            f'总共生成{totalSize}个方法,生成速度:{int(len(allFunctions) / (end_time - start_gen))}个/秒')

        pool = ThreadPool()
        pool.map(modelevaluate.multithreadedAnalysis, allFunctions)
        pool.close()
        pool.join()
        # 模型自身生成的重复率
        noRepeatFunctionsSize = modelevaluate.repetitionRateGeneratedDataItself(allFunctions)
        accuracy = len(modelevaluate.testJshintPassRateSet) / totalSize
        logger.info("生成的用例语法正确率为{:.2%},".format(accuracy))
        model_repetitionRate = 1 - noRepeatFunctionsSize / totalSize
        logger.info("生成数据本身的重复率为{:.2%}".format(model_repetitionRate))
        # 生成数据与训练集的重复率
        training_repetitionRate = modelevaluate.generateDataWithRepetitionRateTrainingSetCount / totalSize
        logger.info("生成数据与训练集的重复率为{:.2%}".format(training_repetitionRate))
        # 统计通过语法检查的代码行数
        for testJshintPassRate in modelevaluate.testJshintPassRateSet:
            modelevaluate.averageNumberRowsToGenerateData(testJshintPassRate)
        lineAve = int(modelevaluate.averageNumberRowsToGenerateDataCount / totalSize)
        logger.info("语法正确方法的平均行数为{}行".format(lineAve))
        return load_model_time, accuracy, model_repetitionRate, training_repetitionRate, lineAve


if __name__ == '__main__':
    # 生成多少原始用例
    sequence_num = eval(cf.get('generateConf', 'num'))
    temperature = eval(cf.get('generateConf', 'temperature'))
    max_length = eval(cf.get('generateConf', 'max_length'))
    table = Table()
    table.add_column("row", style="green")
    table.add_column("model_name")
    table.add_column("checkpoint")
    table.add_column("load_model_time")
    table.add_column("accuracy", style="red")
    table.add_column("model_repetitionRate")
    table.add_column("training_repetitionRate")
    table.add_column("lineAve", style="red")
    table.add_column("temperature")
    table.add_column("max_length")
    table.add_column("sequence_num")
    with Live(table, refresh_per_second=5) as live:
        for row in range(8):
            modelevaluate = modelEvaluate()
            # new 生成实例
            model = model_list[-1]
            temperature=temperature + 0.1
            load_model_time, accuracy, model_repetitionRate, training_repetitionRate, lineAve = modelevaluate.main(
                modelevaluate,
                model=model,
                sequence_num=sequence_num,
                temperature=round(temperature,2),
                max_length = max_length
            )
            checkpoint = model.split("-")[-1]
            table.add_row(
                f"{row}",
                f"{cf.get('comModel', 'model_name')}",
                f"{checkpoint}",
                f"{load_model_time}s",
                f"{round(accuracy, 2)}",
                f"{round(model_repetitionRate,2)}",
                f"{round(training_repetitionRate,2)}", 
                f"{lineAve}", # 语法正确率平均行数
                f"{round(temperature,2)}", 
                f"{max_length}", 
                f"{sequence_num}", 
            )
