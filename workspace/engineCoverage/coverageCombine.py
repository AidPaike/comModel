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
from rich.live import Live
from rich.console import Group
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from functools import partial

getcoverage_model_list_progress = Progress(
    TextColumn('[bold blue]Progress for getcoverage_model_list:'),
    BarColumn(),
    TimeElapsedColumn(), auto_refresh=False)
recover_model_list_progress = Progress(auto_refresh=False)
model_checkpoints_progress = Progress(auto_refresh=False)
recoverFile_progress = Progress(auto_refresh=False)
fileList_progress = Progress(auto_refresh=False)
combine_cov_progress = Progress(auto_refresh=False)
pathList_progress = Progress(auto_refresh=False)
group = Group(
    getcoverage_model_list_progress, recover_model_list_progress, model_checkpoints_progress,
    recoverFile_progress, fileList_progress, combine_cov_progress, pathList_progress
)
live = Live(group)

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
cf.read(os.path.join(BASE_DIR, "workspace/engineCoverage/coverageCombine.ini"), encoding='utf-8')


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


def getCoverageProfraw(path, fileList_progress_id, len):
    fileList_progress.update(fileList_progress_id, total=len)
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
    # print(cmd_coverage)
    pro = subprocess.Popen(cmd_coverage, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = pro.communicate()
    fileList_progress.update(fileList_progress_id, advance=1)
    # print(stdout)
    # print(stderr)


def recover_filename(testcase_path, model_checkpoints_task_id):
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
    # for file in recoverFile_progress.track(recover_testcase_path,description="recoverFile",total=len(list(recover_testcase_path))):
    for file in recover_testcase_path:
        cmd_coverage = f"mv {os.path.join(testcase_path, file)} {os.path.join(testcase_path, file[:-8])}"
        pro = subprocess.Popen(cmd_coverage, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True,
                               stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = pro.communicate()
    model_checkpoints_progress.update(model_checkpoints_task_id, advance=1, refresh=True)


def combine_cov(checkpoints_path, console):
    with console.status("[green]{checkpoints_path} doing combine coverage ...[/]".format(
            checkpoints_path=checkpoints_path)) as status:
        # from my_project import my_console
        # with Live(console=my_console) as live:
        #     my_console.print("[bold blue]Starting work!")
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
        # status = Status("[green]正在处理->>>>>>{}".format(aim_profdata))
        # status.start()
        while True:
            if os.path.exists(cov_list):
                # status.stop()
                break
        while True:
            combine_cmd = []
            profdata_list = [profdata for profdata in os.listdir(cov_list) if profdata.endswith(".js.profdata")]
            if len(profdata_list) == 0 and flag == 1:
                if aim_profdata in os.listdir(cov_list):
                    break
            profdata_combine = [os.path.join(cov_list, profdata_path) for profdata_path in profdata_list]
            if aim_profdata not in os.listdir(cov_list):
                if len(profdata_list) >= 2:
                    flag = 1
                    if not eval(cf.get('coverageCombine', 'profdata_save')):
                        # combine_cmd = ["llvm-profdata-10 ","merge","-sparse",profdata_combine[0],profdata_combine[1],
                        #     "-o",os.path.join(coverage_path,aim_profdata),"&&","rm",profdata_combine[0],profdata_combine[1]]
                        combine_cmd = "llvm-profdata-10 merge -sparse {p0} {p1} -o {Paim} && rm {p0} {p1}".format(
                            p0=profdata_combine[0], p1=profdata_combine[1], Paim=os.path.join(cov_list, aim_profdata)
                        )
                        # print("两者合并并删除-----合成新的")
                        # print(combine_cmd)
                    else:
                        # combine_cmd = ["llvm-profdata-10 ","merge","-sparse",profdata_combine[0],profdata_combine[1],
                        #     "-o",os.path.join(coverage_path,aim_profdata),"&&","mv",profdata_combine[0],profdata_combine[0]+".changed",
                        #     "mv",profdata_combine[1],profdata_combine[1]+".changed"]
                        combine_cmd = "llvm-profdata-10 merge -sparse {p0} {p1} -o {Paim} && mv {p0} {p0}.changed mv {p1} {p1}.changed".format(
                            p0=profdata_combine[0], p1=profdata_combine[1], Paim=os.path.join(cov_list, aim_profdata)
                        )
                        # print("两者合并并改名")
                        # print(combine_cmd)
                else:
                    pass

            else:
                if len(profdata_list) >= 1:
                    flag = 1
                    if not eval(cf.get('coverageCombine', 'profdata_save')):
                        # combine_cmd = clear["llvm-profdata-10 ","merge","-sparse",os.path.join(coverage_path,aim_profdata),profdata_combine[0],
                        #     "-o",os.path.join(coverage_path,aim_profdata),"&&","rm",profdata_combine[0]]
                        combine_cmd = "llvm-profdata-10 merge -sparse {Paim} {p0} -o {Paim} && rm {p0}".format(
                            Paim=os.path.join(cov_list, aim_profdata), p0=profdata_combine[0]
                        )
                        # print("两者合并并删除-----------单纯合并")
                        # print(combine_cmd)
                    else:
                        # combine_cmd = ["llvm-profdata-10 ","merge","-sparse",os.path.join(coverage_path,aim_profdata),profdata_combine[0],
                        #     "-o",os.path.join(coverage_path,aim_profdata),"&&","mv",profdata_combine[0],profdata_combine[0]+".changed"]
                        combine_cmd = "llvm-profdata-10 merge -sparse {Paim} {p0} -o {Paim} && mv {p0} {p0}.changed".format(
                            Paim=os.path.join(cov_list, aim_profdata), p0=profdata_combine[0]
                        )
                        # print("两者合并并改名")
                        # print(combine_cmd)
                else:
                    pass

                if combine_cmd:
                    # print(combine_cmd)
                    pro = subprocess.Popen(combine_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True,
                                           env=my_env,
                                           stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = pro.communicate()
                    # print(stdout)
                    # print(stderr)
            # print(cov_list)


def getTestcases(pathList, pathList_progress_id):
    # pathList_progress.update(pathList_progress_id,total=len(pathList))
    for i in range(len(pathList)):
        fileList_progress_id = fileList_progress.add_task(
            "get {} coverage profraw:".format(pathList[i].split("/")[-3] + "_" + pathList[i].split("/")[-2]))
        ## TODO 覆盖率统计
        ## 不知道如果这个文件里短时间内完成还是继续生成，需要考虑合并完是删除还是保留，还有就是单个checkpoint数量多的话怎么办
        ## 那就先试试一个checkpoint生成完 等着合并完再继续下一步
        console = Console()
        thead_combine = threading.Thread(target=combine_cov, args=(pathList[i], console))
        thead_combine.start()
        js_path = [os.path.join(pathList[i], x) for x in os.listdir(pathList[i]) if x.split(".")[-1] == "js"]
        pathList_progress.update(pathList_progress_id, total=len(pathList))
        # thead_combine.join()
        pool = ThreadPool(3)
        # r = list(fileList_progress.track(pool.imap(getCoverageProfraw, js_path), total=len(js_path),description="fileList"))
        r = pool.imap(partial(getCoverageProfraw, fileList_progress_id=fileList_progress_id, len=len(js_path)), js_path)
        pool.close()
        pool.join()
        # pathList_progress.update(pathList_progress_id , advance=1,refresh=True)

    return pathList


def main() -> None:
    testcase_path = os.path.join(BASE_DIR, cf.get('coverageCombine', 'testcase'))
    testcaseAll = getTestcasesPath(testcase_path)
    # getcoverage_model_list_progress_id = getcoverage_model_list_progress.add_task("getcoverage_model_list")

    with live:
        live.console.print(f"覆盖率生成准备,开始恢复js文件名。")
        logger.info("覆盖率生成准备,开始恢复js文件名。")
        model_checkpoints_task_id = model_checkpoints_progress.add_task('recover js name:')
        for i in recover_model_list_progress.track(range(len(testcaseAll)), description="recover_model_list"):
            pool = ThreadPool()
            # r = list(pool.imap(recover_filename, testcaseAll[i]), total=len(testcaseAll[i]), desc="model_checkpoints",leave=False))
            model_checkpoints_progress.update(model_checkpoints_task_id, total=len(testcaseAll[i]))
            recover_filename_partil = partial(recover_filename, model_checkpoints_task_id=model_checkpoints_task_id)
            r = pool.imap(recover_filename_partil, testcaseAll[i])
            pool.close()
            pool.join()

        pathList_progress_id = pathList_progress.add_task('one_model_pathList:', expand=True)
        # getcoverage_model_list_progress.update(getcoverage_model_list_progress_id,total=len(testcaseAll))
        for i in getcoverage_model_list_progress.track(range(len(testcaseAll)), description="getcoverage_model_list"):
            #  recover_model_list_progress.track(range(len(testcaseAll)), description="recover_model_list")
            # print(i)
            # testcaseAll[1]---->['/root/comModel/data/Finetune_model/Finetune_distilgpt2/checkpoint-10000/testcases', '/root/comModel/data/Finetune_model/Finetune_distilgpt2/checkpoint-20000/testcases']
            model_testcases = getTestcases(testcaseAll[i], pathList_progress_id)
            # print(model_testcases)
            # getcoverage_model_list_progress.update(getcoverage_model_list_progress_id,advance=1)
            # getcoverage_model_list_progress.refresh()


if __name__ == '__main__':
    main()
