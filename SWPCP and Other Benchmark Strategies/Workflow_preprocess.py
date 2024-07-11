import numpy as np
import xml.etree.ElementTree as ET
import re
from Class_declaration import JOB


def Read_Workflow(workflow):
    Data_path = "Data\\" + workflow + ".xml"
    tree = ET.parse(Data_path)  # Epigenomics_100.xml 得到xml的树状图
    root = tree.getroot()   # 获得根，也就是获得最前面的那一行
    TaskN = eval(root.attrib['jobCount'])
    Job = []    # 生成作业列表
    InData = [{} for _ in range(TaskN)]     # 对应任务的输入数据字典
    OutData = [{} for _ in range(TaskN)]
    # Indegree = [0] * TaskN
    # Ref = [[False] * TaskN for _ in range(TaskN)]
    count = -1
    for child in root:    # 访问所有的孩子
        if re.search('job$', child.tag):
            info1 = child.attrib
            Job.append(JOB(info1['id'], info1['name'], eval(info1['runtime'])))
            count += 1
            for offspring in child:
                info2 = offspring.attrib
                if info2['link'] == 'input':
                    InData[count][info2['file']] = eval(info2['size'])
                elif info2['link'] == 'output':
                    OutData[count][info2['file']] = eval(info2['size'])
        # elif re.search('child$', child.tag):
        #     son = eval(re.search('[1-9]\\d*$|0$', child.attrib['ref']).group(0))
        #     for offspring in child:
        #         father = eval(re.search(r'[1-9]\d*$|0$', offspring.attrib['ref']).group(0))
        #         Ref[father][son] = True
        #         Indegree[son] += 1
    return [TaskN, Job, InData, OutData]  # Indegree, Ref


def Pre_Process(taskN, job, inData, outData):
    data_ij = [[-1] * taskN for _ in range(taskN)]      # 记录各个任务对本任务的数据输入量
    amount = [-1] * taskN   # 每个任务的数据量
    Father_son = [[-2] * taskN for _ in range(taskN)]   # 记录各个任务有哪些孩子
    Son_father = [[-2] * taskN for _ in range(taskN)]   # 记录各个任务有哪些父节点
    num_father = [0] * taskN    # 记录各个节点父节点的个数
    for i in range(taskN):
        num_son = 0
        amount[i] = job[i].amount
        for j in range(taskN):
            # 获得第i个任务的输出数据与第j个任务的输入数据一样的文件名
            Data_key = list(outData[i].keys() & inData[j].keys())
            if Data_key:
                num_son += 1
                Father_son[i][num_son] = j  # 标记i个第几个孩子是j
                num_father[j] += 1  # j的父节点数量+1
                Son_father[j][num_father[j]] = i    # j个第num_father[j]个父节点是i，我感觉直接用ij标记也是可以的
                data_ij[i][j] = 0
                for key in Data_key:
                    data_ij[i][j] += outData[i][key] / 1e6
    return [data_ij, amount, Father_son, Son_father]  # 形参，与实参区别


def Merge_Process(taskN, data_ij, amount, father_son, son_father):
    MergedList = [-1]  # 初始化待合并列表
    data_ij = np.array(data_ij)
    np.set_printoptions(suppress=True, linewidth=250)
    # print("原始数据集：\n", data_ij, "\n", amount)
    while len(MergedList) != 0:
        MergedList.clear()
        for i in range(taskN):
            if i in MergedList:
                continue
            # 这个节点只有一个孩子,这条边的出度为1
            if father_son[i][1] != -2 and father_son[i][2] == -2:
                currentSon = father_son[i][1]
                # 这个孩子节点只有一个付清
                if son_father[currentSon][1] != -2 and son_father[currentSon][2] == -2:
                    MergedList.append(currentSon)
                    data_ij[i] = data_ij[currentSon] # ？？？？data_ij表示从i节点出发到其他节点的数据量，因此跟新为下一个
                    amount[i] += amount[currentSon]
                    # print("father: " + str(i) + ", son: " + str(currentSon))
        data_ij = np.delete(data_ij, MergedList, axis=0) # 表示删除数组的对应行
        data_ij = np.delete(data_ij, MergedList, axis=1) # 删除对应列
        amount = np.delete(amount, MergedList)

        [taskN, father_son, son_father] = Find_Father_And_Son(data_ij)  # 代码复用
        # print("预处理中的数据集：\n", data_ij, "\n", amount)
    # print("预处理后的数据集：\n", data_ij, "\n", amount)
    return [taskN, data_ij, amount, father_son, son_father]


def Find_Father_And_Son(data_ij):
    taskN = data_ij.shape[0] # 表示这是一个什么样的矩阵，得到的是一个元组，比如：（3，3）, 得到第一行元素的个数
    father_son = [[-2] * taskN for _ in range(taskN)]
    son_father = [[-2] * taskN for _ in range(taskN)]
    num_father = [0] * taskN
    for i in range(taskN):
        num_son = 0
        for j in range(taskN):
            if data_ij[i][j] >= -1e-6:
                num_son += 1
                father_son[i][num_son] = j
                num_father[j] += 1
                son_father[j][num_father[j]] = i
    return [taskN, father_son, son_father]


def Append_Entry_And_Exit_Task(taskN, data_ij, amount, father_son, son_father):
    data_ij = np.insert(data_ij, 0, [-1] * taskN, axis=1)
    data_ij = np.insert(data_ij, data_ij.shape[1], [-1] * taskN, axis=1)  # 插入列 np.shape返回一个几行几列的矩阵
    taskN += 2
    data_ij = np.insert(data_ij, 0, [-1] * taskN, axis=0)
    data_ij = np.insert(data_ij, data_ij.shape[0], [-1] * taskN, axis=0)  # 插入行

    # 入任务，加了虚拟入任务不是应该son_father要多一个，对data_ij进行操作，其他不用管的
    for i in range(len(son_father)):
        # 因为它只有一个负2，所以其给与虚拟父任务之间有连接且数据量为0，因为多了一个虚拟入任务所以任务都加了1
        if len(set(son_father[i])) == 1:
            data_ij[0][i + 1] = 0
    # 出任务
    for i in range(len(father_son)):
        if len(set(father_son[i])) == 1:
            data_ij[i + 1][taskN - 1] = 0

    amount = np.insert(amount, 0, [0])
    amount = np.insert(amount, amount.shape[0], [0])

    [taskN, father_son, son_father] = Find_Father_And_Son(data_ij)  # 代码复用
    # print(np.array(father_son), '\n\n', np.array(son_father))
    # print("添加入任务和出任务后的数据集：\n", data_ij, "\n", amount)
    return [taskN, data_ij.tolist(), amount.tolist(), father_son, son_father]
