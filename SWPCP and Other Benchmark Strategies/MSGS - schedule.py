import os
import time
from math import ceil  # 向上取整函数，返回值为小数
import copy
from queue import Queue

import numpy as np
import xlrd

from Fuzzy_operations import Fuzzified_Data, fuzzy_fitness
from Fuzzy_operations import fuzzy_sum, fuzzy_minus, fuzzy_num_multiply, fuzzy_divide, fuzzyToreal
from Fuzzy_operations import fuzzy_max, fuzzy_min, fuzzy_more_than, fuzzy_less_than
from Workflow_preprocess import Read_Workflow, Pre_Process, Merge_Process, Append_Entry_And_Exit_Task
# from Print_result import Print_Result, Output_File


INF = (float("inf"), float("inf"), float("inf"))
NEGINF = (-float("inf"), -float("inf"), -float("inf"))


class DataCenter(object):
    def __init__(self):
        self.dataCenter = [[], []]
        self.initDataCenter()

    def initDataCenter(self):
        for j in range(0, 5):
            self.dataCenter[0].append(self.Server(0, Fuzzified_Data(Capability[j]), CostRate[j]))
        for j in range(5, 10):
            self.dataCenter[1].append(self.Server(1, Fuzzified_Data(Capability[j]), CostRate[j]))

    class Server(object):
        def __init__(self, environment, capability, costRate):
            self.environment = environment  # 环境：0为云，1和2为边缘
            self.capability = capability
            self.costRate = costRate
            self.booting = False
            self.startTime = (0.000, 0.000, 0.000)  # 开启时间
            self.endTime = (0.000, 0.000, 0.000)  # 关闭时间
            self.rentTime = (0.000, 0.000, 0.000)  # 租赁时间

        def __str__(self):
            return "environment: {}; capability: {}; costRate: {}; booting: {}; [startTime, endTime]: {}".format(
                self.environment, self.capability, self.costRate, self.booting, [self.startTime, self.endTime])


class Workflow(object):
    def __init__(self, workflowType, arrivalTime, deadline, taskN, amount, data, fatherSon, sonFather):
        self.workflowType = workflowType  # 工作流类型
        self.arrivalTime = (arrivalTime, arrivalTime, arrivalTime)
        self.deadline = (deadline, deadline, deadline)
        self.taskN = taskN
        self.amount = amount
        self.data = data
        self.fatherSon = fatherSon  # 父亲的儿子
        self.sonFather = sonFather  # 儿子的父亲
        self.topolist = [-2] * taskN  # 拓扑序列

        self.server = [(-2, -2)] * taskN  # 服务器 = [(云或边缘中心, 服务器)]
        self.EFT = [(0.000, 0.000, 0.000)] * taskN  # 最早完成时间
        self.EST = [(0.000, 0.000, 0.000)] * taskN  # 最早开始时间

        self.toposort()  # 拓扑排序获得topolist

    def toposort(self):
        self.topolist.clear()
        Indegree = []
        myQueue = Queue()
        cnt = 0
        for ti in range(self.taskN):
            Indegree.append(self.taskN - self.sonFather[ti].count(-2))
            if Indegree[ti] == 0:
                myQueue.put_nowait(ti)
        while not myQueue.empty():
            father = myQueue.get_nowait()
            self.topolist.append(father)
            cnt += 1
            for son in self.fatherSon[father][1:]:
                if son < 0:
                    break
                Indegree[son] -= 1
                if Indegree[son] == 0:
                    myQueue.put_nowait(son)
        if cnt != self.taskN:
            time.sleep(0.01)
            raise Exception("There exists a loop in the workflow: " + self.workflowType + "!")

    def scheduleAllTasks(self):
        self.EFT[0] = self.arrivalTime
        self.EST[0] = self.arrivalTime
        for i in range(1, self.taskN):  # 任务0表示伪入任务
            ti = self.topolist[i]  # 获取拓扑序列下标

            finishTime, usedServer = INF, (-2, -2)
            for di in range(len(dataCenter)):
                for si in range(len(dataCenter[di])):
                    maxTime = (-1, -1, -1)
                    for father in self.sonFather[ti][1:]:
                        if father < 0:
                            break
                        startTime = fuzzy_max(self.EFT[father], dataCenter[di][si].endTime)
                        tranTime = fuzzy_divide(self.data[father][ti], Fuzzified_Data(BandWidth(self.server[father], (di, si))))
                        execTime = fuzzy_divide(self.amount[ti], dataCenter[di][si].capability)
                        tempTime = fuzzy_sum(fuzzy_sum(startTime, tranTime), execTime)
                        if fuzzy_more_than(tempTime, maxTime):  # 对所有父任务取大
                            maxTime = tempTime
                    if fuzzy_less_than(maxTime, finishTime):  # 对所有服务器取小
                        finishTime = maxTime
                        usedServer = (di, si)
            if fuzzy_less_than(finishTime, self.deadline):
                self.EFT[ti], self.server[ti] = finishTime, usedServer
                (di, si) = self.server[ti]
                self.EST[ti] = fuzzy_minus(self.EFT[ti], fuzzy_divide(self.amount[ti], dataCenter[di][si].capability))

                if not dataCenter[di][si].booting:
                    dataCenter[di][si].booting = True
                    dataCenter[di][si].startTime = self.EST[ti]
                    dataCenter[di][si].endTime = self.EFT[ti]
                    if di != 0:
                        for eachServer in dataCenter[di][:4:-1]:
                            if not eachServer.booting:
                                dataCenter[di].remove(eachServer)
                        for si in range(5):
                            if dataCenter[di][si].booting:
                                dataCenter[di].append(
                                    DataCenter.Server(di, dataCenter[di][si].capability, dataCenter[di][si].costRate))
                else:
                    dataCenter[di][si].endTime = self.EFT[ti]
            else:
                for typeId in range(5):
                    tempServer = DataCenter.Server(0, Fuzzified_Data(Capability[typeId]), CostRate[typeId])
                    maxTime = (-1, -1, -1)
                    for father in self.sonFather[ti][1:]:
                        if father < 0:
                            break
                        startTime = fuzzy_max(self.EFT[father], (0, 0, 0))
                        tranTime = fuzzy_divide(self.data[father][ti], Fuzzified_Data(BandWidth(self.server[father], (0, len(dataCenter[0])))))
                        execTime = fuzzy_divide(self.amount[ti], tempServer.capability)
                        tempTime = fuzzy_sum(fuzzy_sum(startTime, tranTime), execTime)
                        if fuzzy_more_than(tempTime, maxTime):  # 对所有父任务取大
                            maxTime = tempTime
                    if fuzzy_less_than(maxTime, self.deadline):
                        dataCenter[0].append(tempServer)
                        self.EFT[ti], self.server[ti] = maxTime, (0, len(dataCenter[0]) - 1)
                        self.EST[ti] = fuzzy_minus(self.EFT[ti], fuzzy_divide(self.amount[ti], tempServer.capability))
                        dataCenter[0][-1].booting = True
                        dataCenter[0][-1].startTime = self.EST[ti]
                        dataCenter[0][-1].endTime = self.EFT[ti]
                        break


def MSGS():
    for wi in range(workflowN):
        # print(wi, workflowNames[wi])
        workflows[wi].scheduleAllTasks()

    execCost = (0.000, 0.000, 0.000)
    for di in range(len(dataCenter)):
        # print("----------------------------------")
        for si in range(len(dataCenter[di])):
            server = dataCenter[di][si]
            # print(server)
            server.rentTime = fuzzy_minus(server.endTime, server.startTime)
            execCost = fuzzy_sum(execCost, fuzzy_num_multiply(tuple(map(ceil, fuzzy_divide(server.rentTime, pertime))), server.costRate))
    execCost = tuple([round(execCost[i], 3) for i in range(3)])
    execCostFitness = round(fuzzy_fitness(execCost), 3)
    print("总执行代价：", execCost, execCostFitness)
    return execCost

    # for wi in range(workflowN):
    #     if fuzzy_more_than(workflows[wi].EFT[workflows[wi].taskN - 1], workflows[wi].deadline):
    #         print(False)
    #         return
    #     print(workflows[wi].EFT[workflows[wi].taskN - 1], workflows[wi].deadline)
    # print(True)


def BandWidth(server1, server2):
    if server1[0] == server2[0]:
        if server1[1] == server2[1]:
            return BandWidthDict['same_server']
        elif server1[0] == 0:
            return BandWidthDict['cloud']
        else:
            return BandWidthDict['edge']
    else:
        return BandWidthDict['cloud-edge']


if __name__ == '__main__':
    pertime = 60  # 计费单位时间
    # Capability = [3.5, 5, 10, 2.5, 2]  # 单位：Mbps（待定）
    # CostRate = [3.5 / 60, 6 / 60, 13.5 / 60, 2 / 60, 1.5 / 60]  # 单位：$/pertime
    Capability = [2.5, 3.5, 5, 7.5, 20, 2.5, 2.6, 2.2, 2.3, 2.7]  # 单位：Gbps（待定）
    CostRate = [1.5 / 60, 3.5 / 60, 6.0 / 60, 10.0 / 60, 12.5 / 60,  # 1.5 3 5 8 15
                3.0 / 60, 3.2 / 60, 2.9 / 60, 3.1 / 60, 3.5 / 60]  # 单位：$/pertime
    dataCenter = DataCenter().dataCenter  # 1个云数据中心和1个边缘数据中心

    BandWidthDict = {'same_server': float('inf'), 'cloud': 2.5, 'edge': 12.5, 'cloud-edge': 1.0}  # 单位：MB/s
    TranCostDict = {'same_server': 0.0, 'cloud': 0.003, 'edge': 0.01, 'cloud-edge': 0.0012}  # 单位：$/s

    workflowN = 100
    wb = xlrd.open_workbook('Output\\Random_Workflows.xls')  # 打开Excel文件
    sheet = wb.sheet_by_name('Workflows')  # 通过excel表格名称(rank)获取工作表
    cells = sheet.row_values(0)
    workflowNames = cells[:workflowN]  # 取第0行微型工作流的前20个数据
    print("随机生成的20个工作流如下：")
    for row in range(4):  # 输出20个工作流
        print(workflowNames[5 * row:5 * (row + 1)])

    # CyberShake_100  Epigenomics_100  Inspiral_100  Montage_100  Sipht_97
    # CyberShake_50  Epigenomics_47  Inspiral_50  Montage_50  Sipht_58
    # CyberShake_30  Epigenomics_24  Inspiral_30  Montage_25  Sipht_29

    Output_path = "Output\\"
    if workflowN <= 5:
        for workflowName in workflowNames:
            Output_path += workflowName + "+"
        Output_path = Output_path[:-1] + "\\"
    else:
        Output_path += str(workflowN) + " Workflows\\"
    if not os.path.exists(Output_path):
        os.makedirs(Output_path)

    interval_avg = 2500  # ms
    base = 2.5  # 控制截止时间的松紧程度

    # start = time.process_time()
    # N = 10  # 进行10次调度
    # for k in range(N):
    #     MSGS()
    # end = time.process_time()
    # print("Process Exe_time:  {:.2f}s.".format(end - start))

    sum_cost = (0, 0, 0)
    average = (0, 0, 0)
    worst = (0, 0, 0)
    optimal = (-1, -1, -1)
    worst_makespan = (0, 0, 0)
    optimal_makespan = (0, 0, 0)
    worst_deadline = 0
    optimal_deadline = 0
    result = []
    iteration_cost = [[] for _ in range(1001)]
    iteration_time = [[] for _ in range(1001)]

    processStartTime = time.process_time()
    N = 10  # 进行10次调度
    for k in range(N):
        print("--------------------------K = " + str(k+1) + "--------------------------")
        dataCenter = DataCenter().dataCenter  # 1个云数据中心和1个边缘数据中心

        allArrivalTime = [0]
        allDeadline = [0] * len(workflowNames)
        workflows = []
        while True:
            interval_time = np.random.exponential(interval_avg, [len(workflowNames) - 1])
            if np.fabs(np.min(interval_time) - interval_avg) < interval_avg or len(workflowNames) == 2:
                # interval_avg * 0.5 < np.min(interval_time) < interval_avg < np.max(interval_time) < interval_avg * 1.5
                break
        for workflow_i in range(1, len(workflowNames)):
            allArrivalTime.append(allArrivalTime[workflow_i - 1] + interval_time[workflow_i - 1])

        for workflow_i in range(len(workflowNames)):
            workflowPath = "Output\\Workflow\\" + workflowNames[workflow_i] + "\\"
            with open(workflowPath + 'Deadline.txt', 'r') as f:
                allDeadline[workflow_i] = allArrivalTime[workflow_i] + base * eval(f.read())

        for workflow_i in range(workflowN):  # len(workflowNames)
            [TaskN, job, inData, outData] = Read_Workflow(workflowNames[workflow_i])  # indegree, ref
            [Data_ij, Amount, Father_son, Son_father] = Pre_Process(TaskN, job, inData, outData)
            [TaskN, Data_ij, Amount, Father_son, Son_father] = Merge_Process(TaskN, Data_ij, Amount, Father_son,
                                                                             Son_father)
            [TaskN, Data_ij, Amount, Father_son, Son_father] = Append_Entry_And_Exit_Task(TaskN, Data_ij, Amount,
                                                                                          Father_son, Son_father)
            workflows.append(Workflow(workflowNames[workflow_i], allArrivalTime[workflow_i], allDeadline[workflow_i],
                                      TaskN, Amount, Data_ij, Father_son, Son_father))

        print("Workflows' deadline is {} ms.".format(allDeadline))

        currCost = MSGS()
        sum_cost = fuzzy_sum(sum_cost, currCost)
        average = fuzzy_divide(sum_cost, N)
        if fuzzy_fitness(worst) < fuzzy_fitness(currCost):
            worst = currCost
        if optimal == (-1, -1, -1) or fuzzy_fitness(optimal) > fuzzy_fitness(currCost):
            optimal = currCost
    processEndTime = time.process_time()

    print("---------------------------------------")

    print("Process Exe_time:  {:.2f}s.".format(processEndTime - processStartTime))
    avg = [tuple([round(i, 3) for i in average])] + [fuzzy_fitness(average)]
    result.append(avg)
    print("Average Run_cost: ", avg)
    wor = [tuple([round(i, 3) for i in worst])] + [fuzzy_fitness(worst)]
    result.append(wor)
    print("Worst Run_cost: ", wor)
    opt = [tuple([round(i, 3) for i in optimal])] + [fuzzy_fitness(optimal)]
    result.append(opt)
    print("Optimal Run_cost: ", opt)
    # Output_File(Output_path, "MSPG", base, iteration_cost, iteration_time, result)
