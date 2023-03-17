import sqlite3
from tqdm import tqdm


def createPrefix(c):
    print ("数据库打开成功")
    c.execute('''CREATE TABLE  IF NOT EXISTS prefix
        (ID INTEGER PRIMARY KEY autoincrement,
        TESTCASE           TEXT    NOT NULL
        );''')
    print ("数据表创建成功")
    conn.commit()
    conn.close()

def insertPrefix(c,a):
    sql = "INSERT INTO prefix (TESTCASE) VALUES (:a)"
    prames = (a,)
    c.execute(sql,prames)
    conn.commit()

def rmSame(c):
    # 去重数据库
    sql = f"DELETE FROM prefix WHERE rowid NOT IN(SELECT MAX(ID) ID FROM prefix GROUP BY TESTCASE);"
    # sql = f"SELECT ID FROM prefix WHERE TESTCASE in (SELECT DISTINCT TESTCASE FROM prefix);"
    # sql = f"SELECT ID FROM prefix WHERE TESTCASE='jenn';"
    # sql = f"SELECT DISTINCT TESTCASE FROM prefix"
    print(sql)
    result = c.execute(sql)
    conn.commit()

def load_data(file):
     with open(file, 'r') as f:
        trainDatasetContents = f.read()
        trainDatasetContent_list = trainDatasetContents.split("<|endoftext|>")
        return trainDatasetContent_list

def getId(c):
    sql = "select ID from prefix"
    result = c.execute(sql).fetchall()
    return result

def idToTestcase(c,id):
    sql = "select TESTCASE from prefix where ID=:id"
    prames = (id,)
    result = c.execute(sql,prames)
    conn.commit()
    return result.fetchall()

if __name__ == '__main__':
    file = "/root/comModel/dataset/train_data_bos.txt"
    conn = sqlite3.connect('comModel.db')
    c = conn.cursor()

    # 创建数据库表
    # createPrefix(c)

    # 插入生成头库
    # trainDatasetContent_list = load_data(file)
    # for testcase in tqdm(trainDatasetContent_list):
    #     if len(testcase)==0:
    #         continue
    #     insertPrefix(c,testcase)

    # 去重数据库
    # rmSame(c)
    
    # 查询
    getId_list = [id[0] for id in getId(c)]
    result = idToTestcase(c,getId_list[10])
    print(result[0][0].strip())
    conn.close()