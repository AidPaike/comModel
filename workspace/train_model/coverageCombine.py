import configparser
import os, sys, re
import numpy as np
from pathlib import Path
import logging
from multiprocessing.pool import ThreadPool
import subprocess
from tqdm import tqdm
from tqdm.auto import trange
import threading
import time

# 初始化日志
logger = logging.getLogger()
# 创建格式器,并将console，handler设置对应的格式
formatter = logging.Formatter(
    '%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
# 创建handler文件处理器
handler = logging.FileHandler(filename="report/coverageCombine_report.log", mode="w")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
# 将处理器，添加至日志器中
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# 获取项目的绝对路径
BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(BASE_DIR)
# 加载配置文件
cf = configparser.ConfigParser()
cf.read(os.path.join(BASE_DIR, "workspace/train_model/coverageCombine.ini"), encoding='utf-8')


# def howManyProfrawCreated(start, path):  不优雅
#     for _ in os.listdir(path):  # file 表示的是文件名
#         start = start + 1
#     return start

# def getAllTestcases(testcases_path) -> list:
#     '''
#     将所有的js文件读取存入列,很容易内存溢出,不采用
#     '''
#     testcaseAll = []
#     for file in os.listdir(testcases_path):
#         file = open(os.path.join(testcases_path,file))
#         try:
#             file_context = file.read()
#             testcaseAll.append(file_context)
#         finally:
#             file.close()
#     return testcaseAll

def getCoverageFileNumber(path):  # 优雅QAQ
    '''
    获取文件夹下有多少文件,用于profdata、profraw
    retrun file_num path_list
    '''
    path_list = os.listdir(path)
    path_list = list(filter(lambda x: x.endswith("." + path.split('/')[-1]), path_list))
    file_num = sum([os.path.isfile(os.path.join(path, filename)) for filename in path_list])
    return file_num, path_list


def getTestcasesPath(testcase_path) -> list:
    '''
    获取存有测试用例的路径
    retrun: 将不同的模型分别存入不同的列表中,最外层包裹了一层列表
    '''
    # print(testcase_path)
    # /root/comModel/data/Finetune_model/
    list_all = []
    pathList = []
    model_path = [os.path.join(testcase_path, modelName) for modelName in os.listdir(testcase_path)]
    # model_checkpoins = [os.listdir(model_path[i]) for i in range(len(model_path))]
    for i in range(len(model_path)):
        model_checkpoint_path = list(filter(lambda x: x.startswith("checkpoint-"), os.listdir(model_path[i])))
        model_checkpoint_path.sort(key=lambda x: int(x.split('-')[1]))
        model_checkpoint_path = [os.path.join(model_path[i], x) for x in model_checkpoint_path]
        list_all.append(model_checkpoint_path)
    # list_all 一个大列表包括了每个模型的checkpoints路径组成的小列表
    for list_item in range(len(list_all)):
        pathList.append([])
        for item in range(len(list_all[list_item])):
            # 判断该目录是否存在及是否有文件
            list_all[list_item][item] = os.path.join(list_all[list_item][item], "testcases")
            # list_all[list_item][item] ------> /root/comModel/data/Finetune_model/Finetune_gpt2/checkpoint-50000/testcases
            if os.path.exists(list_all[list_item][item]) and len(list_all[list_item][item]) > 0:
                pathList[list_item].append(list_all[list_item][item])

    return pathList


def getCoverageProfraw(path):
    coverage_path = os.path.join(BASE_DIR, cf.get('coverageCombine', 'coverage'))
    engine_path = os.path.join(BASE_DIR, cf.get('coverageCombine', 'engine'))
    path_data = path.split("/")
    # print(path_data)
    LLVM_PROFILE_FILE = os.path.join(BASE_DIR, coverage_path, path_data[-4], path_data[-3], path_data[-1] + ".profraw")
    LLVM_PROFDATA_FILE = os.path.join(BASE_DIR, coverage_path, path_data[-4], path_data[-3],
                                      path_data[-1] + ".profdata")
    cmd = ["timeout", "-s9", "30s", engine_path, path]
    my_env = os.environ.copy()
    my_env['LLVM_PROFILE_FILE'] = LLVM_PROFILE_FILE
    # print(LLVM_PROFILE_FILE)
    # print(cmd)
    pro = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False, env=my_env,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = pro.communicate()
    # cmd_coverage = f"mv {path} {path}.changed"
    cmd_coverage = "llvm-profdata-10 merge -o " + LLVM_PROFDATA_FILE + " " + LLVM_PROFILE_FILE + " && rm " + LLVM_PROFILE_FILE + " && mv " + path + " " + path + ".changed"
    print(cmd_coverage)
    pro = subprocess.Popen(cmd_coverage, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = pro.communicate()
    # print(stdout)
    # print(stderr)


def recover_filename(testcase_path):
    '''
    恢复js文件名称,需要传入js文件所在的目录
    '''

    # /root/comModel/data/Finetune_model/
    def recover_filter(f):
        if f[-8:] in ['.changed']:
            return True
        else:
            return False

    recover_testcase_path = filter(recover_filter, os.listdir(testcase_path))
    for file in tqdm(list(recover_testcase_path), position=5, desc="recoverFile", leave=False, ncols=180):
        cmd_coverage = f"mv {os.path.join(testcase_path, file)} {os.path.join(testcase_path, file[:-8])}"
        pro = subprocess.Popen(cmd_coverage, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True,
                               stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = pro.communicate()


def combine_cov(checkpoints_path):
    path_data = checkpoints_path.split("/")
    coverage_path = os.path.join(BASE_DIR, cf.get('coverageCombine', 'coverage'))
    # engine_path = os.path.join(BASE_DIR, cf.get('coverageCombine', 'engine'))
    cov_list = os.path.join(BASE_DIR, coverage_path, path_data[-3], path_data[-2])
    # covProfraw_list = os.path.join(coverage_path , os.listdir(cov_list))
    # for i in trange(10, position=6, desc="combineCoverage", leave=False, ncols=180):
    #     time.sleep(1)
    # LLVM_PROFILE_FILE = os.path.join(BASE_DIR,coverage_path,path_data[-4],path_data[-3],path_data[-1]+".profraw")
    my_env = os.environ.copy()
    flag = 0
    # my_env['LLVM_PROFILE_FILE'] = LLVM_PROFILE_FILE

    # frawTodata_list = [path for path in covProfraw_list if path.endswith(".profraw")]
    # def frawTodata(profraw):
    #     cmd = ["llvm-profdata-10","merge","-o",profraw,profraw.replace(".profraw",".profdata"),"&&","rm",profraw]
    #     pro = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False,
    #                             stderr=subprocess.PIPE, universal_newlines=True)
    #     stdout, stderr = pro.communicate()
    # profrawTodata_pool = ThreadPool()
    # r = list(tqdm(profrawTodata_pool.imap(frawTodata, frawTodata_list), total=len(frawTodata_list),position=7,desc="profrawToData", leave=False, ncols=180))
    # profrawTodata_pool.close()
    # profrawTodata_pool.join()
    aim_profdata = path_data[-3] + "_" + path_data[-2] + ".profdata"
    while True:
        if os.path.exists(cov_list):
            break
    while True:
        combine_cmd = []
        profdata_list = [profdata for profdata in os.listdir(cov_list) if profdata.endswith(".profdata")]
        if aim_profdata not in os.listdir(cov_list):
            if len(profdata_list) >= 2:
                flag = 1
                profdata_combine = [os.path.join(cov_list, profdata_path) for profdata_path in profdata_list]
                print(not eval(cf.get('coverageCombine', 'profdata_save')))
                if not eval(cf.get('coverageCombine', 'profdata_save')):
                    # combine_cmd = ["llvm-profdata-10 ","merge","-sparse",profdata_combine[0],profdata_combine[1],
                    #     "-o",os.path.join(coverage_path,aim_profdata),"&&","rm",profdata_combine[0],profdata_combine[1]]
                    combine_cmd = "llvm-profdata-10 merge -sparse {p0} {p1} -o {Paim} && rm {p0} {p1}".format(
                        p0=profdata_combine[0], p1=profdata_combine[1], Paim=os.path.join(coverage_path, aim_profdata)
                    )
                    print("两者合并并删除")
                    print(combine_cmd)
                else:
                    # combine_cmd = ["llvm-profdata-10 ","merge","-sparse",profdata_combine[0],profdata_combine[1],
                    #     "-o",os.path.join(coverage_path,aim_profdata),"&&","mv",profdata_combine[0],profdata_combine[0]+".changed",
                    #     "mv",profdata_combine[1],profdata_combine[1]+".changed"]
                    combine_cmd = "llvm-profdata-10 merge -sparse {p0} {p1} -o {Paim} && mv {p0} {p0}.changed mv {p1} {p1}.changed".format(
                        p0=profdata_combine[0], p1=profdata_combine[1], Paim=os.path.join(coverage_path, aim_profdata)
                    )
                    print("两者合并并改名")
                    print(combine_cmd)
            else:
                pass

        else:
            if len(profdata_list) >= 1:
                flag = 1
                if not eval(cf.get('coverageCombine', 'profdata_save')):
                    # combine_cmd = clear["llvm-profdata-10 ","merge","-sparse",os.path.join(coverage_path,aim_profdata),profdata_combine[0],
                    #     "-o",os.path.join(coverage_path,aim_profdata),"&&","rm",profdata_combine[0]]
                    combine_cmd = "llvm-profdata-10 merge -sparse {Paim} {p0} -o {Paim} && rm {p0}".format(
                        Paim=os.path.join(coverage_path, aim_profdata), p0=profdata_combine[0]
                    )
                    print("两者合并并删除")
                    print(combine_cmd)
                else:
                    # combine_cmd = ["llvm-profdata-10 ","merge","-sparse",os.path.join(coverage_path,aim_profdata),profdata_combine[0],
                    #     "-o",os.path.join(coverage_path,aim_profdata),"&&","mv",profdata_combine[0],profdata_combine[0]+".changed"]
                    combine_cmd = "llvm-profdata-10 merge -sparse {Paim} {p0} -o {Paim} && mv {p0} {p0}.changed".format(
                        Paim=os.path.join(coverage_path, aim_profdata), p0=profdata_combine[0]
                    )
                    print("两者合并并改名")
                    print(combine_cmd)
            else:
                pass

        if len(profdata_list) == 0 and flag:
            break
            if profdata_list[0] == aim_profdata:
                break
        if combine_cmd:
            # print(combine_cmd)
            pro = subprocess.Popen(combine_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, env=my_env,
                                   stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = pro.communicate()
            # print(stdout)
            # print(stderr)
    # print(cov_list)


def getTestcases(pathList):
    for i in trange(len(pathList), position=1, desc="pathList", leave=False, ncols=180):
        ## TODO 覆盖率统计
        ## 不知道如果这个文件里短时间内完成还是继续生成，需要考虑合并完是删除还是保留，还有就是单个checkpoint数量多的话怎么办
        ## 那就先试试一个checkpoint生成完 等着合并完再继续下一步
        thead_combine = threading.Thread(target=combine_cov, args=(pathList[i],))
        thead_combine.start()

        js_path = [os.path.join(pathList[i], x) for x in os.listdir(pathList[i]) if x.split(".")[-1] == "js"]
        # thead_combine.join()
        pool = ThreadPool()
        r = list(
            tqdm(pool.imap(getCoverageProfraw, js_path), total=len(js_path), position=2, desc="fileList", leave=False,
                 ncols=180), daemon=True)
        pool.close()
        pool.join()

    return pathList


def main() -> None:
    testcase_path = os.path.join(BASE_DIR, cf.get('coverageCombine', 'testcase'))
    testcaseAll = getTestcasesPath(testcase_path)

    logger.info("覆盖率生成准备,开始恢复js文件名。")
    for i in trange(len(testcaseAll), position=3, desc="model_list", leave=True, ncols=180, disable=False):
        pool = ThreadPool()
        r = list(
            tqdm(pool.imap(recover_filename, testcaseAll[1]), total=len(testcaseAll[1]), position=4, desc="recoverList",
                 leave=True, ncols=180))
        pool.close()
        pool.join()

    for i in trange(len(testcaseAll), position=0, desc="model_list", leave=True, ncols=180):
        # testcaseAll[1]---->['/root/comModel/data/Finetune_model/Finetune_distilgpt2/checkpoint-10000/testcases', '/root/comModel/data/Finetune_model/Finetune_distilgpt2/checkpoint-20000/testcases']
        model_testcases = getTestcases(testcaseAll[i])
        # print(model_testcases)


if __name__ == '__main__':
    main()
