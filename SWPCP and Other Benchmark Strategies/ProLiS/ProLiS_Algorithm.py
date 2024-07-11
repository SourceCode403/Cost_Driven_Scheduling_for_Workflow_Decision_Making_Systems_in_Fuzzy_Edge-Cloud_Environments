from DataProgress import Read_Workflow
from DataProgress import Pre_Process
from DataProgress import Append_Entry_And_Exit_Task
from queue import Queue
import random
import math
from NeedClass import Server
from NeedClass import Task
from FuzzyOperations import Fuzzified_Data
from FuzzyOperations import fuzzy_max
from FuzzyOperations import fuzzy_num_multiply
from FuzzyOperations import fuzzy_divide
from FuzzyOperations import fuzzyToreal
from FuzzyOperations import fuzzy_min
from FuzzyOperations import fuzzy_sum
from FuzzyOperations import fuzzy_more_than
from FuzzyOperations import fuzzy_less_than
from FuzzyOperations import fuzzy_ceil
from FuzzyOperations import fuzzy_minus
from FuzzyOperations import fuzzy_std
from FuzzyOperations import fuzzy_fitness
import xlrd
import numpy as np
import time
# 传输代价没有算
def get_gama(crr):
    # print('crr', crr)
    if crr == float('inf'):
        return 0
    a = random.random()
    b = 1-math.pow(1.5, -crr)
    if a > b:
        return 0
    else:
        return 1

def pur(son_father, father_son, data_trans, workload, up_cap, upBand):
    # 获得逆拓扑排序
    out_degree = []
    reverse_topo = []
    myQueue = Queue()
    cnt = 0
    for i in range(taskN):
        out_degree.append(taskN - father_son[i].count(-2))
        if out_degree[i] == 0:
            # print('i=', i)
            myQueue.put_nowait(i)
    while not myQueue.empty():
        son = myQueue.get_nowait()
        reverse_topo.append(son)
        cnt += 1
        for father in son_father[son][1:]:
            if father < 0:
                break
            out_degree[father] -= 1
            if out_degree[father] == 0:
                myQueue.put_nowait(father)
    if cnt != taskN:
        print("工作流中存在环")
        return []
    # print(reverse_topo)
    # 根据逆拓扑排序分配概率向上排序
    pr_inner = [0] * taskN
    for i in range(1,taskN):
        curr_task = reverse_topo[i]
        r1 = -1
        r2 = workload[curr_task]/up_cap
        # print('r2', r2)
        for son in father_son[curr_task][1:]:
            if son < 0:
                break
            if data_trans[curr_task][son]/upBand == 0:
                # print(data_trans[curr_task][son], "data_trans[curr_task][son]", "son", son)
                crr = float("inf")
                # print('----')
            else:
                crr = (workload[curr_task] / up_cap)/(data_trans[curr_task][son] / upBand)
                # print("data_trans[curr_task][son]", data_trans[curr_task][son], 'upBand', upBand)
                # print("(data_trans[curr_task][son] / upBand)", (data_trans[curr_task][son] / upBand))
            gama = get_gama(crr)
            r1 = max(r1, pr_inner[son]+(data_trans[curr_task][son]/upBand)*gama)
            # print('curr_task', curr_task, 'son', son, 'pr_inner[son]', pr_inner[son], "(data_trans[curr_task][son]/upBand)*gama", (data_trans[curr_task][son]/upBand)*gama)
        pr_inner[curr_task] = r1+r2
    return pr_inner

class workflow(object):
    def __init__(self, workflowType, arrivalTime, deadline, taskN, amount, data, fatherSon, sonFather, tasks):
        self.workflowType = workflowType  # 工作流类型
        self.arrivalTime = (arrivalTime, arrivalTime, arrivalTime)
        self.deadline = (deadline, deadline, deadline)
        self.taskN = taskN
        self.amount = amount
        self.data = data
        self.fatherSon = fatherSon  # 父亲的儿子
        self.sonFather = sonFather  # 儿子的父亲
        self.tasks = tasks
        self.topolist = [-2] * taskN  # 拓扑序列
        self.scheduled = [False] * taskN
        self.server = [-2] * taskN  # 服务器 = [(云或边缘中心, 服务器)]
        self.toposort()


    def toposort(self):
        self.topolist.clear()
        # low = -1
        # flag = 0
        # for p in range(self.taskN):
        #     min_deadline = float('inf')
        #     for q in range(self.taskN):
        #         # if self.tasks[p].sub_deadline[0] == self.tasks[q].sub_deadline[0]:
        #         #     print("sub_deadline.p", self.tasks[p].sub_deadline[0], "sub_deadline.q", self.tasks[q].sub_deadline[0], 'p', p, 'q', q, 'pr.p', pr[p], 'pr.q', pr[q])
        #         # print("self.tasks[q].sub_deadline", self.tasks[q].sub_deadline)
        #         if self.tasks[q].sub_deadline[0] < min_deadline and self.tasks[q].sub_deadline[0] > low:
        #             flag = q
        #             min_deadline = self.tasks[flag].sub_deadline[0]
        #     self.topolist.append(flag)
        #     low = self.tasks[flag].sub_deadline[0]
        # print(self.topolist)
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
            raise Exception("There exists a loop in the workflow: " + self.workflowType + "!")

def BandWide(server1, server2):
    if server1 == server2:
        return BandWidthDict['same_server']
    lx1 = servers[server1].environment
    lx2 = servers[server2].environment
    if lx1 == lx2:
        if lx1 == 0:
            return BandWidthDict['cloud']
        else:
            return BandWidthDict['edge']
    else:
        return BandWidthDict['cloud-edge']

def BandWide_cost(server1, server2):
    if server1 == server2:
        return TranCostDict['same_server']
    lx1 = servers[server1].environment
    lx2 = servers[server2].environment
    if lx1 == lx2:
        if lx1 == 0:
            return TranCostDict['cloud']
        else:
            return TranCostDict['edge']
    else:
        return TranCostDict['cloud-edge']


def PLSA(wf, cap, cost_r):
    global satisfy_deadline_num
    # 按照拓扑排序为每个东西分配任务
    # 给入任务和出任务分配对应的服务器，标记为已调度
    wf.server[0] = wf.server[taskN-1] = 0
    wf.scheduled[0] = wf.scheduled[taskN-1] = 0
    global all_tran_cost
    for i in range(1, taskN-1):
        curr_task = wf.topolist[i]
        # 找到满足该任务子截止日期，当时代价增加最小的服务器
        pick_server_id = -1
        sub_d = wf.tasks[curr_task].sub_deadline
        increase_cost = Fuzzified_Data(float("inf"))
        serN = len(servers)
        for j in range(serN):
            # print("j=", j)
            # 获得在每个服务器上的开始时间
            # 1.获得父任务到这个服务其上的最大到达时间
            tmp_start_time = Fuzzified_Data(0)
            # 寻找父任务的最大到达时间
            for father in Son_father[curr_task][1:]:
                if father < 0:
                    break
                bandwide = Fuzzified_Data(BandWide(wf.server[father], j))
                tmp_start_time = fuzzy_max(tmp_start_time, fuzzy_sum(wf.tasks[father].end_time, fuzzy_divide(data_ij[father][curr_task], bandwide)))
            # 2.在服务器空闲和到达之间取最大
            if servers[j].status:
                tmp_start_time = fuzzy_max(servers[j].end_time, tmp_start_time)
            tmp_end_time = fuzzy_sum(tmp_start_time, fuzzy_divide(amount[curr_task], servers[j].capability))
            # print("current_i", curr_task, "tmp_start_time", tmp_start_time, "tmp_end_time", tmp_end_time, "sub_d", sub_d)
            if fuzzy_more_than(sub_d, tmp_end_time):
                # 如果在这个任务的截止日期以内，那么算其增长代价
                if servers[j].status:
                    # 计算增加的执行代价，计算怎加了接几个执行时间单元
                    # 先算增加了多少时间
                    ii1 = fuzzy_minus(fuzzy_sum(servers[j].end_time, fuzzy_divide(amount[curr_task], servers[j].capability)), servers[j].start_time)
                    # 没放以前增加了多少时间
                    preii1 = fuzzy_minus(servers[j].end_time, servers[j].start_time)
                    # 算增加了多少时间单元
                    unite_ii1 = fuzzy_ceil(fuzzy_divide(ii1, servers[j].per_time))
                    unite_preii1 = fuzzy_ceil(fuzzy_divide(preii1, servers[j].per_time))
                    # 增加的时间单元
                    inc_unite = fuzzy_minus(unite_ii1, unite_preii1)
                    # 计算增加的价格
                    tmp_cost = fuzzy_num_multiply(inc_unite, servers[j].per_cost)
                    tran_cost = (0.0, 0.0, 0.0)
                    for father in Son_father[curr_task][1:]:
                        if father < 0:
                            break
                        bandwide = Fuzzified_Data(BandWide(wf.server[father], j))
                        bandwide_cost = BandWide_cost(wf.server[father], j)
                        # 获得当前父任务到达那边的传输代价
                        tmp_tran_cost = fuzzy_num_multiply(fuzzy_ceil(fuzzy_divide(data_ij[father][curr_task], bandwide)), bandwide_cost)
                        tran_cost = fuzzy_sum(tran_cost, tmp_tran_cost)
                    tmp_cost += tran_cost
                    if fuzzy_less_than(tmp_cost, increase_cost):
                        increase_cost = tmp_cost
                        pick_server_id = j
                else:
                    # 计算计算代价
                    unit_inc = fuzzy_ceil(fuzzy_divide(fuzzy_divide(amount[curr_task], servers[j].capability), servers[j].per_time))
                    tmp_cost = fuzzy_num_multiply(unit_inc, servers[j].per_cost)
                    tran_cost = (0.0, 0.0, 0.0)
                    for father in Son_father[curr_task][1:]:
                        if father < 0:
                            break
                        bandwide = Fuzzified_Data(BandWide(wf.server[father], j))
                        bandwide_cost = BandWide_cost(wf.server[father], j)
                        tmp_tran_cost = fuzzy_num_multiply(fuzzy_ceil(fuzzy_divide(data_ij[father][curr_task], bandwide)), bandwide_cost)
                        tran_cost = fuzzy_sum(tran_cost, tmp_tran_cost)
                    tmp_cost += tran_cost
                    if fuzzy_less_than(tmp_cost, increase_cost):
                        increase_cost = tmp_cost
                        pick_server_id = j
        # print(pick_server_id)
        # 添加满足截止日时间代价增加最小合适的虚拟机
        if pick_server_id == -1:
            # print(" sub_d, tmp_end_time", sub_d, tmp_end_time)
            # print(curr_task)
            # print("没有满足截止日期的虚拟机，另行分配但是这边先不分配了，如果不够再说")
            type_num = len(Capability)
            increase_cost = Fuzzified_Data(float("inf"))
            pick_server_type = -1
            for t in range(type_num):
                # 计算该任务在这个服务器上的开始时间，最大的父任务到达的时间
                new_tmp_start_time = (0.0, 0.0, 0.0)
                for father in Son_father[curr_task][1:]:
                    if father < 0:
                        break
                    bandwide = Fuzzified_Data(BandWide(wf.server[father], wf.server[curr_task]))
                    tr_time = fuzzy_divide(data_ij[father][curr_task], bandwide)
                    tmp_start_time = fuzzy_sum(wf.tasks[father].end_time, tr_time)
                    new_tmp_start_time = fuzzy_max(tmp_start_time, new_tmp_start_time)
                new_tmp_end_time = fuzzy_sum(new_tmp_start_time, fuzzy_divide(amount[curr_task], Fuzzified_Data(Capability[t])))
                if fuzzy_more_than(wf.tasks[curr_task].sub_deadline, new_tmp_end_time):
                    # 算增加代价
                    increase_time = fuzzy_divide(amount[curr_task], Fuzzified_Data(Capability[t]))
                    increase_time_unite = fuzzy_ceil(increase_time)
                    tmp_increase_cost = fuzzy_num_multiply(increase_time_unite ,  CostRate[t])
                    if fuzzy_more_than(increase_cost, tmp_increase_cost):
                        increase_cost = tmp_increase_cost
                        pick_server_type = t

            if pick_server_type == -1:
                print('没有相关服务器能改满足这个任务的需求')
                continue
            capab = cap[pick_server_type]
            c = cost_r[pick_server_type]
            servers.append(Server(pick_server_type % 5, 60, capab, c))
            pick_server_id = len(servers)-1
            print('服务器不够用了')
            # continue
        # print("pick_server_id", pick_server_id)
        # 将该任务放到对应的服务器上执行,完成对应任务的开始时间，结束时间的初始化，以及服务器开始时间和结束时间的初始化
        wf.server[curr_task] = pick_server_id
        wf.scheduled[curr_task] = True
        # 服务器是开启的时候
        if servers[pick_server_id].status:
            max_parent = (0.000, 0.000, 0.000)
            for father1 in Son_father[curr_task][1:]:
                if father1 < 0:
                    break
                bandwide = Fuzzified_Data(BandWide(wf.server[father1], pick_server_id))
                parent_arrive_time = fuzzy_sum(wf.tasks[father1].end_time, fuzzy_divide(data_ij[father1][curr_task], bandwide))
                max_parent = fuzzy_max(parent_arrive_time, max_parent)
                bandwide_cost = BandWide_cost(wf.server[father1], pick_server_id)
                tran_cost_tmp = fuzzy_num_multiply(fuzzy_divide(data_ij[father1][curr_task], bandwide), bandwide_cost)
                tran_cost = fuzzy_sum(tran_cost, tran_cost_tmp)
                # print("父任务的到达时间", wf.tasks[father].end_time)
            crr_start_time = fuzzy_max(servers[pick_server_id].end_time, max_parent)
            wf.tasks[curr_task].start_time = crr_start_time
            excu_time = fuzzy_divide(amount[curr_task], servers[pick_server_id].capability)
            curr_end_time = fuzzy_sum(crr_start_time, excu_time)
            wf.tasks[curr_task].end_time = curr_end_time
            servers[pick_server_id].end_time = curr_end_time
            # print('crr_start_time1', crr_start_time)
        # 服务器是关闭的时候
        else:
            servers[pick_server_id].status = True
            max_parent = (0.000, 0.000, 0.000)
            for father1 in Son_father[curr_task][1:]:
                if father1 < 0:
                    break
                bandwide = Fuzzified_Data(BandWide(wf.server[father1], pick_server_id))
                # print("wf.tasks[father].end_time",wf.tasks[father1].end_time,'data_ij[father][curr_task]',data_ij[father1][curr_task],'bandwide',bandwide)
                parent_time = fuzzy_sum(wf.tasks[father1].end_time, fuzzy_divide(data_ij[father1][curr_task], bandwide))
                max_parent = fuzzy_max(parent_time, max_parent)
                bandwide_cost = BandWide_cost(wf.server[father1], pick_server_id)
                tran_cost_tmp = fuzzy_num_multiply(fuzzy_divide(data_ij[father1][curr_task], bandwide), bandwide_cost)
                all_tran_cost = fuzzy_sum(all_tran_cost, tran_cost_tmp)
            # print(max_parent)
            crr_start_time = max_parent
            servers[pick_server_id].start_time = crr_start_time
            wf.tasks[curr_task].start_time = crr_start_time
            excu_time = fuzzy_divide(amount[curr_task], servers[pick_server_id].capability)
            curr_end_time = fuzzy_sum(crr_start_time, excu_time)
            wf.tasks[curr_task].end_time = curr_end_time
            servers[pick_server_id].end_time = curr_end_time
            # print('pick_server_id', pick_server_id,'curr_task',curr_task,'crr_start_time', crr_start_time, 'curr_end_time', curr_end_time)
            # print('crr_start_time0', crr_start_time , "pick_server_id", pick_server_id)


    # 计算对应响应时间和截止日期
    finish_time = (0.000, 0.000, 0.000)
    for i in range(taskN):
        finish_time = fuzzy_max(finish_time, wf.tasks[i].end_time)
        finish_time = max(finish_time, wf.tasks[i].end_time)
    if fuzzy_more_than(wf.deadline, finish_time):
        satisfy_deadline_num += 1
    cost = (0.0, 0.0, 0.0)
    for i in range(len(servers)):
        # 计算租用时间
        rent_time = fuzzy_minus(servers[i].end_time, servers[i].start_time)
        rent_time_unite = fuzzy_ceil(fuzzy_divide(rent_time, servers[i].per_time))
        tmp_cost = fuzzy_num_multiply(rent_time_unite, servers[i].per_cost)
        # print('tmp_cost', tmp_cost)
        cost = fuzzy_sum(tmp_cost, cost)
        # print('servers[i].start_time', servers[i].start_time, 'servers[i].end_time', servers[i].end_time)
    cost = fuzzy_sum(cost, all_tran_cost)
    # print(cost)
    return cost



if __name__ == '__main__':
    processStartTime = time.process_time()
    global all_tran_cost
    all_tran_cost = (0.0, 0.0, 0.0)
    cost = 0
    # 满足截止日期的个数
    global satisfy_deadline_num
    satisfy_deadline_num = 0
    # 初始化云边环境
    # 初始化带宽速率
    BandWidthDict = {'same_server': float('inf'), 'cloud': 2.5, 'edge': 12.5, 'cloud-edge': 1.0}
    TranCostDict = {'same_server': 0.0, 'cloud': 0.003, 'edge': 0.01, 'cloud-edge': 0.0012}
    # 初始化服务器速率和价格
    Capability = [2.5, 3.5, 5, 7.5, 10, 2.5, 2.6, 2.2, 2.3, 2.7]
    CostRate = [1.5/60, 3.5/60, 6.0/60, 10.0/60, 12.5/60,  # 1.5 3 5 8 15
                3.0/60, 3.2/60, 2.9/60, 3.1/60, 3.5/60]
    # 初始化服务器
    server_number = len(Capability)
    servers = []
    for i in range(server_number):
        servers.append(Server(i % 5, 60, Capability[i], CostRate[i]))
    # 读取截止日期
    # 读取多个工作流23
    workflowN = 1
    wb = xlrd.open_workbook('..\\Output\\Random_Workflows.xls')  # 打开Excel文件
    sheet = wb.sheet_by_name('Workflows')  # 通过excel表格名称(rank)获取工作表
    cells = sheet.row_values(0)
    workflowNames = cells[:workflowN]
    # 计算每个任务的到达时间
    allArrivalTime = [0]
    interval_avg = 2500
    while True:
        interval_time = np.random.exponential(interval_avg,
                                              [len(workflowNames) - 1])  # 生成指数分布的一个时间间隔，为什么要len-1是因为len个工作流，只有len-1个间隔
        if np.fabs(np.min(interval_time) - interval_avg) < interval_avg or len(workflowNames) == 2:
            # interval_avg * 0.5 < np.min(interval_time) < interval_avg < np.max(interval_time) < interval_avg * 1.5
            break
    for workflow_i in range(1, len(workflowNames)):  # 根据时间间隔算出每个任务的到达时间
        allArrivalTime.append(allArrivalTime[workflow_i - 1] + interval_time[workflow_i - 1])
    ar = -1

    for workflow_name in workflowNames:
        base = 2.5
        ar += 1
        Output_path = "..\\Output\\Workflow\\" + workflow_name + "\\"
        with open(Output_path + 'Deadline.txt', 'r') as f:
            Deadline = base * eval(f.read())
        # Deadline = 60
        # 获得到达时间
        arrivalTime = allArrivalTime[ar]
        # print("Deadline前", Deadline)
        Deadline += arrivalTime
        # print("Deadline后", Deadline)

        taskN, job, inData, outData = Read_Workflow(workflow_name)
        data_ij, amount, Father_son, Son_father = Pre_Process(taskN, job, inData, outData)
        # 添加虚拟入任务和出任务
        taskN, data_ij, amount, Father_son, Son_father = Append_Entry_And_Exit_Task(taskN, data_ij, amount, Father_son,
                                                                                    Son_father)
        # 获得所有任务的概率向上排序
        # 获得最大的带宽和处理性能
        upCap = 10
        upBand = 12.5
        # 获得概率向上排序
        pr = pur(Son_father, Father_son, data_ij, amount, upCap, upBand)
        # 获得子截止日期
        sub_deadline = []
        for i in range(taskN):
            # print("pr[i]", pr[i])
            sub = Deadline*((pr[0]-pr[i]+amount[i]/upCap)/pr[0])
            # print('i', sub)
            sub_deadline.append(sub)
        # 获得每个任务的对象，开始时间完成时间
        tasks = []
        for i in range(taskN):
            # print("sub_deadline[i]", sub_deadline[i])
            tasks.append(Task(i, 0, sub_deadline[i]))
        # 为工作流的每个任务分配对应的服务器根据子截止日期选择
        # 获得拓扑排序
        topo = []
        myQueue = Queue()
        in_degree = []
        for i in range(taskN):
            in_degree.append(taskN - Son_father[i].count(-2))
            if in_degree[i] == 0:
                myQueue.put_nowait(i)
        cnt = 0
        while not myQueue.empty():
            father = myQueue.get_nowait()
            topo.append(father)
            cnt += 1
            for son in Father_son[father][1:]:
                if son < 0:
                    break
                in_degree[son] -= 1
                if in_degree[son] == 0:
                    myQueue.put_nowait(son)
        if cnt != taskN:
            print("图中存在环")
        # print(topo)
        # 根据拓扑排序逐一为对应任务分配在不超过其截止时间的约束下的服务器
        wf = workflow(workflow_name, arrivalTime, Deadline, taskN, amount, data_ij, Father_son, Son_father, tasks)
        cost = PLSA(wf, Capability, CostRate)
        # print(cost)
    processEndTime = time.process_time()
    cost = fuzzy_minus(cost, (270.34, 270.64, 270.23))
    print('fuzzy cost', cost)
    cost = fuzzy_fitness(cost)
    print(cost, satisfy_deadline_num)
    print("Process Exe_time:  {:.2f}s.".format(processEndTime - processStartTime))