import os
import time
from math import ceil  # 向上取整函数，返回值为小数
import copy
from queue import Queue, LifoQueue

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
        self.dataCenter = [[], []]  # 里面是5个云的server，以及5个边缘的server
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
            self.occupiedTime = []  # 占用时间

        def __str__(self):
            # return "environment: {}; capability: {}; costRate: {}; booting: {}".format(
            #     self.environment, self.capability, self.costRate, self.booting)
            return "environment: {}; capability: {}; costRate: {}; booting: {}; occupiedTime: {}".format(
                self.environment, self.capability, self.costRate, self.booting, self.occupiedTime)


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
        self.LST = [(0.000, 0.000, 0.000)] * taskN  # 最迟开始时间
        self.LFT = [(0.000, 0.000, 0.000)] * taskN  # 最迟完成时间
        self.AST = [(0.000, 0.000, 0.000)] * taskN  # 实际开始时间
        self.AFT = [(0.000, 0.000, 0.000)] * (taskN - 1) + [deadline]  # 实际完成时间 这边的加上就是再后面加一个deadline的元组

        self.scheduled = [False] * taskN
        self.PCP = []  # 局部关键路径
        self.CP = [-2] * taskN  # 关键父任务

        self.toposort()  # 拓扑排序获得topolist
        self.initStratTimeAndFinishTime()  # 计算EFT、EST、LST、LFT

    def toposort(self): # 啥啊
        self.topolist.clear()
        Indegree = []
        myQueue = Queue()
        cnt = 0
        for ti in range(self.taskN):
            # 计算入度
            Indegree.append(self.taskN - self.sonFather[ti].count(-2))
            # 如果入度为0 则这个是入任务，入队
            if Indegree[ti] == 0:
                myQueue.put_nowait(ti)
        while not myQueue.empty():
            father = myQueue.get_nowait()
            self.topolist.append(father)
            cnt += 1
            # 消去拓扑排序的出队的出度
            for son in self.fatherSon[father][1:]:
                if son < 0:
                    break
                Indegree[son] -= 1
                if Indegree[son] == 0:
                    myQueue.put_nowait(son)
        if cnt != self.taskN:
            time.sleep(0.01)
            raise Exception("There exists a loop in the workflow: " + self.workflowType + "!")

    # noinspection DuplicatedCode
    def initStratTimeAndFinishTime(self):
        self.EFT[0] = self.arrivalTime
        self.EST[0] = self.arrivalTime
        for i in range(1, self.taskN):  # 任务0表示伪入任务
            ti = self.topolist[i]  # 获取拓扑序列下标
            if self.data[0][ti] == 0:  # 真实入任务
                self.EST[ti] = self.EFT[0]
                self.EFT[ti] = fuzzy_divide(self.amount[ti], dataCenter[0][4].capability)
                self.server[ti] = (0, 4)
            else:  # 不是入任务
                finishTime = INF
                usedServer = (-2, -2)
                for di in range(len(dataCenter)):
                    for si in range(len(dataCenter[di])):
                        maxTime = (-1, -1, -1)
                        for father in self.sonFather[ti][1:]:
                            if father < 0:
                                break
                            tranTime = fuzzy_divide(self.data[father][ti], Fuzzified_Data(BandWidth(self.server[father], (di, si))))
                            execTime = fuzzy_divide(self.amount[ti], dataCenter[di][si].capability)
                            tempTime = fuzzy_sum(fuzzy_sum(self.EFT[father], tranTime), execTime)
                            if fuzzy_more_than(tempTime, maxTime):  # 对所有父任务取大
                                maxTime = tempTime
                        if fuzzy_less_than(maxTime, finishTime):  # 对所有服务器取小
                            finishTime = maxTime
                            usedServer = (di, si)
                self.EFT[ti] = finishTime
                self.server[ti] = usedServer

            (di, si) = self.server[ti]
            self.EST[ti] = fuzzy_minus(self.EFT[ti], fuzzy_divide(self.amount[ti], dataCenter[di][si].capability))


        self.LST[self.taskN - 1] = self.deadline
        self.LFT[self.taskN - 1] = self.deadline
        for i in range(self.taskN - 2, -1, -1):  # 任务taskN-1表示伪出任务，从后往前倒
            ti = self.topolist[i]
            if self.data[ti][self.taskN - 1] == 0:  # 真实出任务
                self.LST[ti] = fuzzy_minus(self.deadline, fuzzy_divide(self.amount[ti], dataCenter[0][4].capability))
                self.LFT[ti] = self.deadline
                self.server[ti] = (0, 4)
            else:  # 不是出任务
                startTime = (-1, -1, -1)
                usedServer = (-2, -2)
                for di in range(len(dataCenter)):
                    for si in range(len(dataCenter[di])):
                        minTime = INF
                        for son in self.fatherSon[ti][1:]:
                            if son < 0:
                                break
                            execTime = fuzzy_divide(self.amount[ti], dataCenter[di][si].capability)
                            tranTime = fuzzy_divide(self.data[ti][son], Fuzzified_Data(BandWidth((di, si), self.server[son])))
                            tempTime = fuzzy_minus(fuzzy_minus(self.LST[son], execTime), tranTime)
                            if fuzzy_less_than(tempTime, minTime):  # 对所有父任务取小
                                minTime = tempTime
                        if fuzzy_more_than(minTime, startTime):  # 对所有服务器取大
                            startTime = minTime
                            usedServer = (di, si)
                self.LST[ti] = startTime
                self.server[ti] = usedServer
                # print('FST,', self.server[ti])

            (di, si) = self.server[ti]
            self.LFT[ti] = fuzzy_sum(self.LST[ti], fuzzy_divide(self.amount[ti], dataCenter[di][si].capability))

    # 使得满足最早完成时间的server和最迟完成时间的server不会冲突吗
    # noinspection DuplicatedCode
    def updateStratTimeAndFinishTime(self, pcp):
        index = self.topolist.index(pcp)  # 找出这个任务在拓扑排序中的下标
        successorTask = self.topolist[index + 1: -1]  # 正序 -1表示最后一个位置，其这个坐标折后的所有任务。没有到最后，因为最后是虚拟如任务可以不要
        precursorTask = self.topolist[index - 1:: -1]  # 倒序

        for i in range(len(successorTask)):  # 更新所有未调度后继任务的EST和EFT
            ti = successorTask[i]
            if self.scheduled[ti] or self.data[0][ti] == 0:  # 真实入任务或已调度任务
                continue
            finishTime = INF
            usedServer = (-2, -2)
            for di in range(len(dataCenter)):
                for si in range(len(dataCenter[di])):
                    maxTime = (-1, -1, -1)
                    for father in self.sonFather[ti][1:]:
                        if father < 0:
                            break
                        tranTime = fuzzy_divide(self.data[father][ti], Fuzzified_Data(BandWidth(self.server[father], (di, si))))
                        execTime = fuzzy_divide(self.amount[ti], dataCenter[di][si].capability)
                        tempTime = fuzzy_sum(fuzzy_sum(self.EFT[father], tranTime), execTime)
                        if fuzzy_more_than(tempTime, maxTime):  # 对所有父任务取大
                            maxTime = tempTime
                    if fuzzy_less_than(maxTime, finishTime):  # 对所有服务器取小
                        finishTime = maxTime
                        usedServer = (di, si)
            self.EFT[ti] = finishTime
            self.server[ti] = usedServer
            (di, si) = self.server[ti]
            self.EST[ti] = fuzzy_minus(self.EFT[ti], fuzzy_divide(self.amount[ti], dataCenter[di][si].capability))

        for i in range(len(precursorTask)):  # 更新所有未调度前驱任务的LST和LFT
            ti = precursorTask[i]
            if self.scheduled[ti] or self.data[ti][self.taskN - 1] == 0:  # 真实出任务或已调度任务
                continue
            startTime = (-1, -1, -1)
            usedServer = (-2, -2)
            for di in range(len(dataCenter)):
                for si in range(len(dataCenter[di])):
                    minTime = INF
                    for son in self.fatherSon[ti][1:]:
                        if son < 0:
                            break
                        execTime = fuzzy_divide(self.amount[ti], dataCenter[di][si].capability)
                        tranTime = fuzzy_divide(self.data[ti][son], Fuzzified_Data(BandWidth((di, si), self.server[son])))
                        tempTime = fuzzy_minus(fuzzy_minus(self.LST[son], execTime), tranTime)
                        if fuzzy_less_than(tempTime, minTime):  # 对所有子任务取小
                            minTime = tempTime
                    if fuzzy_more_than(minTime, startTime):  # 对所有服务器取大
                        startTime = minTime
                        usedServer = (di, si)
            self.LST[ti] = startTime
            self.server[ti] = usedServer
            (di, si) = self.server[ti]
            self.LFT[ti] = fuzzy_sum(self.LST[ti], fuzzy_divide(self.amount[ti], dataCenter[di][si].capability))

    def hasUnscheduledParent(self, ti):
        for father in self.sonFather[ti][1:]:
            if father < 0:
                break
            if self.scheduled[father]:
                continue
            else:
                return True
        return False

    def scheduleAllParents(self, ti):
        while self.hasUnscheduledParent(ti):  # 存在直接未调度父任务
            currTask = ti
            myStack = LifoQueue()  # 一个栈
            # 找出所有关键父任务
            while self.hasUnscheduledParent(currTask):  # 存在直接未调度父任务
                maxTime = (-1, -1, -1)
                for father in self.sonFather[currTask][1:]:  # 算所有当前父任务的执行时间得到关键父任务
                    if father < 0:
                        break
                    if self.scheduled[father]:
                        continue
                    tempTime = fuzzy_sum(self.EFT[father],  # 父任务的最早完成时间时间+最小传输时间得到的
                                         fuzzy_divide(self.data[father][currTask],  # 在初始化最早开始时间和最迟开始时间的过程中已经初始化任务的服务器
                                                      Fuzzified_Data(BandWidth(self.server[father], self.server[currTask])))) # ？ 什么时候确实了父任务的server
                    # print('server[f]:{0}, and server[curr]:{1}'.format(self.server[father], self.server[currTask]))
                    if fuzzy_more_than(tempTime, maxTime):  # 对所有直接未调度父任务取大
                        maxTime = tempTime
                        self.CP[currTask] = father  # 关键父任务
                myStack.put_nowait(self.CP[currTask])
                currTask = self.CP[currTask]

            self.CP = [-2] * self.taskN  # 关键父任务
            self.PCP.clear()
            while not myStack.empty():
                self.PCP.append(myStack.get_nowait())
            # print(self.PCP)
            self.schedulePath()

            for pcp in self.PCP[::-1]:
                self.updateStratTimeAndFinishTime(pcp)
                self.scheduleAllParents(pcp)

    def tryAssignPCP(self, server, beginTime, endTime):
        startTime = fuzzy_max(beginTime, self.EST[self.PCP[0]])
        finishTime = startTime
        for pcp in self.PCP:
            startTime = fuzzy_max(finishTime, self.EST[pcp])
            finishTime = fuzzy_sum(startTime, fuzzy_divide(self.amount[pcp], server.capability))
            if fuzzy_more_than(finishTime, fuzzy_min(endTime, self.LFT[pcp])):
                return [False, (-1, -1, -1), (-1, -1, -1)]
        return [True, startTime, finishTime]

    def assignPCP(self, serverId, server, position, beginTime, endTime):
        execInterval = []

        startTime = fuzzy_max(beginTime, self.EST[self.PCP[0]]) # 这边有取大所以
        finishTime = startTime
        for pcp in self.PCP:
            self.server[pcp] = serverId
            self.scheduled[pcp] = True
            startTime = fuzzy_max(finishTime, self.EST[pcp])
            finishTime = fuzzy_sum(startTime, fuzzy_divide(self.amount[pcp], server.capability))

            if fuzzy_more_than(finishTime, endTime):
                time.sleep(0.01)
                # print(position)
                raise Exception("Assigning PCP Failed!")

            self.EST[pcp] = self.LST[pcp] = self.AST[pcp] = startTime  # 开始时间确定
            self.EFT[pcp] = self.LFT[pcp] = self.AFT[pcp] = finishTime  # 完成时间确定
            execInterval.append([self.AST[pcp], self.AFT[pcp]]) # 添加执行时间段  每个任务执行的时间段
            # print(execInterval)

        # print(position)
        if position == "booting":
            di = int(serverId[0])
            server.booting = True
            if di != 0:
                for eachServer in dataCenter[di][:4:-1]:
                    if not eachServer.booting:
                        dataCenter[di].remove(eachServer)
                for si in range(5):
                    if dataCenter[di][si].booting:
                        dataCenter[di].append(
                            DataCenter.Server(di, dataCenter[di][si].capability, dataCenter[di][si].costRate))
                # typeId = random.randint(5, 9)
                # dataCenter[di].append(DataCenter.Server(di, Capability[typeId], CostRate[typeId]))
                # dataCenter[di].append(DataCenter.Server(di, server.capability, server.costRate))
            server.occupiedTime = copy.deepcopy(execInterval)  # 添加使用的时间端
        elif position == "insert": # 时间段之间
            i = eval(position.split(" ")[1])
            server.occupiedTime = server.occupiedTime[:i] + execInterval + server.occupiedTime[i:]
            print()
        elif position == "before": # 时间段前
            server.occupiedTime = execInterval + server.occupiedTime
        elif position == "after": # 时间段后
            server.occupiedTime = server.occupiedTime + execInterval
        server.startTime = server.occupiedTime[0][0]
        server.endTime = server.occupiedTime[-1][1] # 所有server的开始时间和结束时间有点一刀切的感觉

    def isUnavailable(self, server, PCPAmount):
        if not server.booting:  # 未启动
            if self.tryAssignPCP(server, NEGINF, INF)[0]:
                T1 = (0.000, 0.000, 0.000)
                actualExecTime = fuzzy_divide(PCPAmount, server.capability)
                T2 = tuple(map(ceil, fuzzy_divide(actualExecTime, pertime))) # 模糊数向上取整
                increaseCost = fuzzy_num_multiply(fuzzy_minus(T2, T1), server.costRate)
                restTime = fuzzy_minus(fuzzy_num_multiply(T2, pertime), actualExecTime)
                return [True, "booting", NEGINF, INF, increaseCost, actualExecTime, restTime]
            # return [False, "false", (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)]

        else:  # 已启动
            # 寻找空闲时间槽进行插空
            if len(server.occupiedTime) > 2:
                for i in range(1, len(server.occupiedTime)):
                    if fuzzy_less_than(server.occupiedTime[i][0], self.EST[self.PCP[0]]):  # 不满足PCP的最早开始时间
                        continue
                    if fuzzy_more_than(server.occupiedTime[i - 1][1], self.LFT[self.PCP[-1]]):  # 不满足PCP的最迟完成时间
                        break
                    if fuzzy_more_than(fuzzy_minus(server.occupiedTime[i][0], server.occupiedTime[i - 1][1]),
                                       fuzzy_divide(PCPAmount, server.capability)):
                        flag, startTime, finishTime = self.tryAssignPCP(server, server.occupiedTime[i - 1][1],
                                                                        server.occupiedTime[i][0])
                        if flag:
                            increaseCost = (0.000, 0.000, 0.000)  # server.costRate * 0
                            actualExecTime = fuzzy_minus(server.endTime, server.startTime)
                            restTime = fuzzy_minus(
                                fuzzy_num_multiply(tuple(map(ceil, fuzzy_divide(actualExecTime, pertime))), pertime),
                                actualExecTime)
                            return [True, "insert" + str(i), server.occupiedTime[i - 1][1], server.occupiedTime[i][0],
                                    increaseCost, actualExecTime, restTime]

            # 插入到第一个任务之前
            if self.LFT[self.PCP[-1]] < server.occupiedTime[0][0]:
                flag, startTime, finishTime = self.tryAssignPCP(server, NEGINF, server.occupiedTime[0][0])
                if flag:  # 具体逻辑
                    T1 = tuple(map(ceil, fuzzy_divide(fuzzy_minus(server.endTime, server.startTime), pertime)))
                    T2 = tuple(map(ceil, fuzzy_divide(fuzzy_minus(finishTime, startTime), pertime)))
                    increaseCost = fuzzy_num_multiply(fuzzy_minus(T2, T1), server.costRate)
                    actualExecTime = fuzzy_minus(finishTime, startTime)
                    restTime = fuzzy_minus(
                        fuzzy_num_multiply(tuple(map(ceil, fuzzy_divide(actualExecTime, pertime))), pertime),
                        actualExecTime)
                    return [True, "before", NEGINF, server.occupiedTime[0][0], increaseCost, actualExecTime, restTime]

            # 插入到最后一个任务之后
            if self.EST[self.PCP[0]] > server.occupiedTime[-1][1]:
                flag, startTime, finishTime = self.tryAssignPCP(server, server.occupiedTime[-1][1], INF)
                if flag:  # 简化计算
                    T1 = tuple(map(ceil, fuzzy_divide(fuzzy_minus(server.endTime, server.startTime), pertime)))
                    actualExecTime = fuzzy_minus(finishTime, startTime)
                    T2 = tuple(map(ceil, fuzzy_divide(actualExecTime, pertime)))
                    increaseCost = fuzzy_num_multiply(fuzzy_minus(T2, T1), server.costRate)
                    restTime = fuzzy_minus(fuzzy_num_multiply(T2, pertime), actualExecTime)
                    return [True, "after", server.occupiedTime[-1][1], INF, increaseCost, actualExecTime, restTime]

        return [False, "false", (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)]

    def schedulePath(self):
        """
        # 保证每个边缘中心一定存在一个未启动的服务器
        for di in range(1, len(dataCenter)):
            idleFlag = False
            for server in dataCenter[di]:
                if not server.booting:
                    idleFlag = True
                    break
            if not idleFlag:
                typeId = random.randint(5, 9)
                dataCenter[di].append(DataCenter.Server(di, Capability[typeId], CostRate[typeId]))
        """

        # 寻找最适合的服务器
        PCPAmount = sum([self.amount[ti] for ti in self.PCP]) # PCP上存的所有数据量的和
        usedServer = (-2, -2)
        position = "false"
        beginTime, endTime = (-1, -1, -1), (-1, -1, -1)
        usedFlag = False
        minIncreaseCost = INF
        maxActualExecTime = (-1, -1, -1)
        minRestTime = INF  # 不超过60min

        for di in range(len(dataCenter)):
            for si in range(len(dataCenter[di])):
                currServer = dataCenter[di][si]
                # 返回是否这个服务器有效即是否能满足截止日期，在服务器中是怎么插入的
                flag, tempPosition, begin, end, increaseCost, actualExecTime, restTime = self.isUnavailable(
                    currServer, PCPAmount)
                if flag:
                    if fuzzy_less_than(increaseCost, minIncreaseCost):  # 执行增长代价最低
                        usedFlag = True
                    elif fuzzy_less_than(fuzzy_minus(increaseCost, minIncreaseCost), (1e-6, 1e-6, 1e-6)):  # 执行增长代价相等
                        if fuzzy_more_than(actualExecTime, maxActualExecTime):  # 实际执行时间最长
                            usedFlag = True
                        elif fuzzy_less_than(fuzzy_minus(maxActualExecTime, actualExecTime), (1e-6, 1e-6, 1e-6)):  # 实际执行时间相等
                            if fuzzy_less_than(restTime, minRestTime):  # 剩余时间（当前窗口时间-实际执行时间）最少
                                usedFlag = True
                    if usedFlag:  # 满足以上三个条件
                        position = tempPosition
                        beginTime, endTime = begin, end
                        minIncreaseCost = increaseCost
                        maxActualExecTime = actualExecTime
                        minRestTime = restTime
                        usedServer, usedFlag = (di, si), False

        # 租赁满足截止时间约束最便宜的云数据中心服务器，有加到，在上面找不到合适的服务器了
        if position == "false":
            for typeId in range(5):
                tempServer = DataCenter.Server(0, Fuzzified_Data(Capability[typeId]), CostRate[typeId])
                flag, tempPosition, begin, end, increaseCost, actualExecTime, restTime = self.isUnavailable(
                    tempServer, PCPAmount)
                if flag:
                    position = tempPosition
                    beginTime, endTime = begin, end
                    dataCenter[0].append(tempServer)
                    usedServer = (0, len(dataCenter[0]) - 1)
                    break

        # 调度PCP到服务器usedServer上
        (di, si) = usedServer
        self.assignPCP((di, si), dataCenter[di][si], position, beginTime, endTime)


def MSPCP():
    for wi in range(workflowN):
        # print(wi, workflowNames[wi])
        currWorkflow = workflows[wi]
        currWorkflow.scheduled[0] = currWorkflow.scheduled[-1] = True  # 标记伪入任务和伪出任务为已调度任务
        currWorkflow.scheduleAllParents(currWorkflow.taskN - 1)     # 从虚拟出任务往上找关键路径，调度关键路径

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
    #     print("----------------------------------")
    #     print(wi, workflowNames[wi], workflows[wi].scheduled)


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
    Capability = [2.5, 3.5, 5, 7.5, 10, 2.5, 2.6, 2.2, 2.3, 2.7]  # 单位：Gbps（待定）
    CostRate = [1.5 / 60, 3.5 / 60, 6.0 / 60, 10.0 / 60, 12.5 / 60,  # 1.5 3 5 8 15
                3.0 / 60, 3.2 / 60, 2.9 / 60, 3.1 / 60, 3.5 / 60]  # 单位：$/pertime
    dataCenter = DataCenter().dataCenter  # 1个云数据中心和1个边缘数据中心

    # print("dataCenter[0][0].per_cost", dataCenter)

    BandWidthDict = {'same_server': float('inf'), 'cloud': 2.5, 'edge': 12.5, 'cloud-edge': 1.0}  # 单位：MB/s
    TranCostDict = {'same_server': 0.0, 'cloud': 0.003, 'edge': 0.01, 'cloud-edge': 0.0012}  # 单位：$/s

    workflowN = 20
    wb = xlrd.open_workbook('Output\\Random_Workflows.xls')  # 打开Excel文件
    sheet = wb.sheet_by_name('Workflows')  # 通过excel表格名称(rank)获取工作表
    cells = sheet.row_values(0)
    workflowNames = cells[:workflowN]  # 取第0行微型工作流的前20个数据 这里又100个数据
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
        Output_path = Output_path[:-1] + "\\" # 截取最后一个字母，关键有点画蛇添足的感觉
    else:
        Output_path += str(workflowN) + " Workflows\\"
    if not os.path.exists(Output_path):
        os.makedirs(Output_path)

    interval_avg = 2500  # ms  # 这个intercal_avg表示的是什么
    base = 2.5  # 控制截止时间的松紧程度

    sum_cost = (0, 0, 0)  # 总的代价
    average = (0, 0, 0)    # 平均？
    worst = (0, 0, 0)
    optimal = (-1, -1, -1)
    worst_makespan = (0, 0, 0)
    optimal_makespan = (0, 0, 0)
    worst_deadline = 0
    optimal_deadline = 0
    result = []
    iteration_cost = [[] for _ in range(1001)]  #
    iteration_time = [[] for _ in range(1001)]

    processStartTime = time.process_time()
    N = 10  # 进行10次调度
    for k in range(N):
        print("--------------------------K = " + str(k + 1) + "--------------------------")
        dataCenter = DataCenter().dataCenter  # 1个云数据中心和1个边缘数据中心 每次都得重新启动一下数据中心，初始化每次都应该不一样
        # print(dataCenter[0][0])
        allArrivalTime = [0]    # 初始化所有任务的到达时间0
        allDeadline = [0] * len(workflowNames)  # 初始化所有任务的截止时间0
        # print('allDeadline', allDeadline)
        workflows = []
        while True:
            interval_time = np.random.exponential(interval_avg, [len(workflowNames) - 1])  # 生成指数分布的一个时间间隔，为什么要len-1是因为len个工作流，只有len-1个间隔
            if np.fabs(np.min(interval_time) - interval_avg) < interval_avg or len(workflowNames) == 2:
                # interval_avg * 0.5 < np.min(interval_time) < interval_avg < np.max(interval_time) < interval_avg * 1.5
                break
        for workflow_i in range(1, len(workflowNames)): # 根据时间间隔算出每个任务的到达时间
            allArrivalTime.append(allArrivalTime[workflow_i - 1] + interval_time[workflow_i - 1])

        # 读文件生成所有的截止时间
        for workflow_i in range(len(workflowNames)):
            workflowPath = "Output\\Workflow\\" + workflowNames[workflow_i] + "\\"
            with open(workflowPath + 'Deadline.txt', 'r') as f:
                # 生成每个工作流的截止时间为到达时间+截止时间*2.5
                allDeadline[workflow_i] = allArrivalTime[workflow_i] + base * eval(f.read())

        # 对工作流预处理，得到所有工作流的截止时间，开始时间，以及整个拓扑结构
        for workflow_i in range(workflowN):  # len(workflowNames)
            [TaskN, job, inData, outData] = Read_Workflow(workflowNames[workflow_i])  # indegree, ref
            [Data_ij, Amount, Father_son, Son_father] = Pre_Process(TaskN, job, inData, outData)
            # 融合有向割边
            [TaskN, Data_ij, Amount, Father_son, Son_father] = Merge_Process(TaskN, Data_ij, Amount, Father_son,
                                                                             Son_father)
            # 添加虚拟入任务和出任务
            [TaskN, Data_ij, Amount, Father_son, Son_father] = Append_Entry_And_Exit_Task(TaskN, Data_ij, Amount,
                                                                                          Father_son, Son_father)
            # 初始化工作流，包括工作流名称，到达时间，截止时间，任务数量，每个任务的计算量，每个任务的子任务，每个任务的父任务
            workflows.append(Workflow(workflowNames[workflow_i], allArrivalTime[workflow_i], allDeadline[workflow_i],
                                      TaskN, Amount, Data_ij, Father_son, Son_father))

        print("Workflows' deadline is {} ms.".format(allDeadline))

        currCost = MSPCP()
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
    wor = [tuple([round(i, 3) for i in worst])] + [fuzzy_fitness(worst)] # round（a, b）保留a中的b位小
    result.append(wor)
    print("Worst Run_cost: ", wor)
    opt = [tuple([round(i, 3) for i in optimal])] + [fuzzy_fitness(optimal)]
    result.append(opt)
    print("Optimal Run_cost: ", opt)
    # Output_File(Output_path, "MSPG", base, iteration_cost, iteration_time, result)
