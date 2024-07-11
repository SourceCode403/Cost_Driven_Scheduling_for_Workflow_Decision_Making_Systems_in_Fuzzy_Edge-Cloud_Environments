import os
import time
from math import ceil  # 向上取整函数，返回值为小数
import random
import copy
import numpy as np
import xlrd

from Fuzzy_operations import Fuzzified_Data, fuzzy_fitness
from Fuzzy_operations import fuzzy_sum, fuzzy_minus, fuzzy_num_multiply, fuzzy_divide, fuzzyToreal
from Fuzzy_operations import fuzzy_max
from Class_declaration import Task, Server
from Workflow_preprocess import Read_Workflow, Pre_Process, Toposort
from Print_result import Print_Result, Output_File


class Chromosome(object):
    def __init__(self, DC, TS, SA):
        self.DC = copy.deepcopy(DC)
        self.TS = copy.deepcopy(TS)
        self.SA = copy.deepcopy(SA)
        self.task = []
        self.server = []

        for i in range(len(self.DC)):
            self.server.append(Server(self.DC[i] // cloudN, Per_time, Capability[self.DC[i]], CostRate[self.DC[i]]))
        for i in range(workflow_taskN):
            self.task.append(Task(i, self.SA[i]))

        self.run_time = [(0.000, 0.000, 0.000)] * len(workflows)
        self.makespan = (0.000, 0.000, 0.000)
        self.com_cost = (0.000, 0.000, 0.000)
        self.tran_cost = (0.000, 0.000, 0.000)
        self.run_cost = (0.000, 0.000, 0.000)
        self.meet_deadline = 0

    def mutation(self, w_now):
        point = (workflow_taskN // 20 - 10.0) * (w_now - w_min) / (w_max - w_min) + 10.0
        while point > 0.0:
            pos = random.randint(0, workflow_taskN - 1)
            self.TS[pos] = round(random.uniform(0, workflow_taskN / 2), 1)
            self.SA[pos] = random.randint(0, serverN - 1)
            point -= 1.0


def Entry_task(son):
    if Son_father[son][1] == -2:
        return True
    return False


def diverge(TS1, TS2, SA1, SA2):
    num = 0
    for i in range(workflow_taskN):
        if TS1[i] != TS2[i]:
            num += 1
        if SA1[i] != SA2[i]:
            num += 1
    divergence = num / (2.0 * workflow_taskN)
    return divergence


def genetic(gene1, gene2):
    if gene1 is None or gene2 is None:
        return
    pos_num1 = random.randint(0, workflow_taskN - 1)
    pos_num2 = random.randint(0, workflow_taskN - 1)
    if pos_num1 > pos_num2:
        pos_num1, pos_num2 = pos_num2, pos_num1
    for i in range(pos_num1, pos_num2):
        gene1[i] = gene2[i]
    return copy.deepcopy(gene1)


def geneticDC(gene1, gene2):
    if gene1 is None or gene2 is None:
        return
    pos_num1 = random.randint(0, dataCenterN - 1)
    pos_num2 = random.randint(0, dataCenterN - 1)
    if pos_num1 > pos_num2:
        pos_num1, pos_num2 = pos_num2, pos_num1
    for i in range(pos_num1, pos_num2):
        gene1[i] = gene2[i]
    return copy.deepcopy(gene1)


def cmp_fitness(chromosome1, chromosome2):
    if chromosome1.meet_deadline == workflowN and chromosome2.meet_deadline == workflowN:
        return fuzzy_fitness(chromosome1.run_cost) - fuzzy_fitness(chromosome2.run_cost) < 1e-10
    elif chromosome1.meet_deadline == chromosome2.meet_deadline:
        return chromosome1.makespan[2]-0.05*(chromosome1.makespan[2]-chromosome1.makespan[1]) < \
               chromosome2.makespan[2]-0.05*(chromosome2.makespan[2]-chromosome2.makespan[1])
    return chromosome1.meet_deadline > chromosome2.meet_deadline


def pop_init():
    global pulation
    global p_best
    global g_best
    for i in range(pop_size):
        DC = []
        TS = []
        SA = []
        for _ in range(dataCenterN):
            DC.append(random.randint(0, serverN - 1))
        for _ in range(workflow_taskN):
            TS.append(round(random.uniform(0, workflow_taskN / 2), 1)) # 优先级编码
            SA.append(random.randint(0, dataCenterN - 1))
        population[i] = Chromosome(DC, TS, SA)
        schedule(population[i])
        # print(population[i].run_cost, population[i].run_time)
        p_best[i] = copy.deepcopy(population[i])
        if i == 0:
            g_best = copy.deepcopy(population[i])
        elif cmp_fitness(population[i], g_best):
            g_best = copy.deepcopy(population[i])


def max_Parents(chromosome, son):
    wait_time = (-1, -1, -1)
    num_father = 1
    while Son_father[son][num_father] > -1 and num_father <= workflow_taskN:  # 后者可以不要
        father = Son_father[son][num_father]
        son_server, father_server = chromosome.SA[son], chromosome.SA[father]
        if son_server == father_server:  # 相同服务器
            tran_type = 'same_server'
        elif son_server // cloudN == son_server // cloudN == 0:  # 云和云之间
            tran_type = 'cloud'
        elif son_server // cloudN == son_server // cloudN == 1:  # 边缘和边缘之间
            tran_type = 'edge'
        else:  # 云和边缘之间
            tran_type = 'cloud-edge'

        tran_time = fuzzy_divide(Data_ij[father][son], Fuzzified_Data(BandWidthDict[tran_type]))
        wait_time = fuzzy_max(wait_time, fuzzy_sum(chromosome.task[father].end_time, tran_time))
        tran_cost = fuzzy_num_multiply(tran_time, TranCostDict[tran_type])
        chromosome.tran_cost = fuzzy_sum(chromosome.tran_cost, tran_cost)

        num_father += 1
    wait_time = fuzzyToreal(wait_time)
    return wait_time


def schedule(chromosome):

    chromosome.run_time = [(0.00, 0.00, 0.00)] * len(workflows)  # 该粒子中工作流的运行时间
    chromosome.com_cost = (0.00, 0.00, 0.00)  # 该粒子的计算代价
    chromosome.tran_cost = (0.00, 0.00, 0.00)  # 该粒子的传输代价
    chromosome.run_cost = (0.00, 0.00, 0.00)  # 该粒子的运行代价

    Topolist = Toposort(workflow_taskN, chromosome.TS, ref, indegree)  # 任务的拓扑顺序

    for new_i in range(workflow_taskN):
        i = Topolist[new_i]
        chromosome.task[i] = Task(i, chromosome.SA[i])
        current_task = chromosome.task[i]
        current_server = chromosome.server[chromosome.SA[i]]

        if Entry_task(i):
            if not current_server.status:
                current_server.end_time = (0, 0, 0)
                current_server.start_time = current_server.end_time
                current_server.status = True
            current_task.start_time = current_server.end_time
        else:
            maxT = max_Parents(chromosome, i)
            if not current_server.status:
                current_server.end_time = maxT
                current_server.start_time = current_server.end_time
                current_server.status = True
            current_task.start_time = fuzzy_max(current_server.end_time, maxT)

        current_task.com_time = fuzzy_divide(Amount[i], Fuzzified_Data(chromosome.server[chromosome.SA[i]].capability))
        current_task.end_time = fuzzy_sum(current_task.start_time, current_task.com_time)
        current_server.end_time = current_task.end_time

        for j in range(len(workflows)):
            if taskN[j] <= i < taskN[j+1]:
                chromosome.run_time[j] = fuzzy_max(chromosome.run_time[j], current_task.end_time)  # 运行时间
                break

    for i in range(len(workflows)):
        chromosome.makespan = fuzzy_max(chromosome.makespan, chromosome.run_time[i])  # 最大完成时间
        if chromosome.run_time[i][2]-0.05*(chromosome.run_time[i][2]-chromosome.run_time[i][1]) <= Deadline[i]:
            chromosome.meet_deadline += 1

    for i in range(dataCenterN):
        if chromosome.server[i].status:
            current_server = chromosome.server[i]
            current_server.rent_time = fuzzy_minus(current_server.end_time, current_server.start_time)
            curr_com_cost = fuzzy_num_multiply(
                tuple(map(ceil, fuzzy_divide(current_server.rent_time, current_server.per_time))),
                current_server.per_cost)
            chromosome.com_cost = fuzzy_sum(chromosome.com_cost, curr_com_cost)
    chromosome.run_cost = fuzzy_sum(chromosome.tran_cost, chromosome.com_cost)  # 运行代价


def pop_evolve():
    global population
    global p_best
    global g_best
    global w
    global c1
    global c2
    global generation
    # global now
    for i in range(pop_size):
        # d = diverge(population[i].TS, g_best.TS, population[i].SA, g_best.SA)
        # w = w_max - (w_max - w_min) * exp(d / (d - 1.01))
        if random.random() < w:
            population[i].mutation(w)
        if random.random() < c1:
            population[i].DC = geneticDC(population[i].DC, p_best[i].DC)
            population[i].TS = genetic(population[i].TS, p_best[i].TS)
            population[i].SA = genetic(population[i].SA, p_best[i].SA)
        if random.random() < c2:
            population[i].DC = geneticDC(population[i].DC, g_best.DC)
            population[i].TS = genetic(population[i].TS, g_best.TS)
            population[i].SA = genetic(population[i].SA, g_best.SA)
        population[i] = Chromosome(population[i].DC, population[i].TS, population[i].SA)
        schedule(population[i])
        if cmp_fitness(population[i], p_best[i]):
            p_best[i] = copy.deepcopy(population[i])
            if cmp_fitness(population[i], g_best):
                g_best = copy.deepcopy(population[i])
                # now = generation


def MSPG(count):
    # Mult-workflow Scheduling strategy based on Particle swarm optimization
    # employing Genetic operator
    global population
    global p_best
    global g_best
    global c1
    global c2
    global w
    global generation
    # global now
    max_iter = 1000  # 1000
    generation = 1
    pop_init()
    Print_Result(g_best, iteration_cost, iteration_time, result, True, 0)
    while generation <= max_iter:
        # if generation - now > 100:
        #     break
        w = w_max - ((w_max - w_min) / max_iter) * generation
        c1 = c1_start - (((c1_start - c1_end) / max_iter) * generation)
        c2 = c2_start - (((c2_start - c2_end) / max_iter) * generation)
        # s = 0
        # for i in range(pop_size):
        #     if not cmp_fitness(p_best[i], population[i]):
        #         s += 1
        # w = (w_max - w_min) * (s / pop_size) + w_min
        pop_evolve()
        Print_Result(g_best, iteration_cost, iteration_time, result, True, generation)
        # for i in range(serverN):
        #     print(g_best.server[i].start_time, g_best.server[i].end_time, g_best.server[i].rent_time)
        print("-----------------------------------------")
        generation = generation + 1
    Print_Result(g_best, iteration_cost, iteration_time, result, False, count)
    print("----------------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    cloudN = 5
    edgeN = 5
    serverN = cloudN + edgeN
    Per_time = 60
    # Capability = [3.5, 5, 10, 2.5, 2]  # 单位：Mbps（待定）
    # CostRate = [3.5 / 60, 6 / 60, 13.5 / 60, 2 / 60, 1.5 / 60]  # 单位：$/pertime
    Capability = [2.5, 3.5, 5, 7.5, 10, 2.5, 2.6, 2.2, 2.3, 2.7]  # 单位：Mbps（待定）
    CostRate = [1.5 / 60, 3.5 / 60, 6.0 / 60, 10.0 / 60, 12.5 / 60,
                3.0 / 60, 3.2 / 60, 2.9 / 60, 3.1 / 60, 3.5 / 60]  # 单位：$/pertime
    BandWidthDict = {'same_server': float('inf'), 'cloud': 2.5, 'edge': 12.5, 'cloud-edge': 1.0}  # 单位：MB/s
    TranCostDict = {'same_server': 0.0, 'cloud': 0.003, 'edge': 0.01, 'cloud-edge': 0.0012}  # 单位：$/s

    pop_size = 100
    w_min = 0.4  # 0.2
    w_max = 0.9  # 0.8
    c1_start = 0.9
    c1_end = 0.2
    c2_start = 0.4
    c2_end = 0.9

    workflowN = 30
    dataCenterN = int(workflowN * 3.0)
    wb = xlrd.open_workbook('Output\\Random_Workflows.xls')  # 打开Excel文件
    sheet = wb.sheet_by_name('Workflows')  # 通过excel表格名称(rank)获取工作表
    cells = sheet.row_values(0)
    workflows = cells[:workflowN]  # 取第0行微型工作流的前20个数据
    print("随机生成的20个工作流如下：")
    for row in range(4):  # 输出20个工作流
        print(workflows[5*row:5*(row+1)])

    # for _ in range(workflowN):
    #     workflows.append(raw_workflows[random.randint(10, 14)])
    # workflows = ["Montage_50", "CyberShake_50", "Sipht_58"]
    # workflows = ["CyberShake_50", "Montage_50", "Sipht_58"]
    # CyberShake_100  Epigenomics_100  Inspiral_100  Montage_100  Sipht_97
    # CyberShake_50  Epigenomics_47  Inspiral_50  Montage_50  Sipht_58
    # CyberShake_30  Epigenomics_24  Inspiral_30  Montage_25  Sipht_29

    interval_avg = 2500  # ms
    while True:
        interval_time = np.random.exponential(interval_avg, [len(workflows)-1])
        if np.fabs(np.min(interval_time)-interval_avg) < interval_avg or len(workflows) == 2:
            # interval_avg * 0.5 < np.min(interval_time) < interval_avg < np.max(interval_time) < interval_avg * 1.5
            break
    arrival_time = [0]
    for workflow_i in range(1, len(workflows)):
        arrival_time.append(arrival_time[workflow_i-1] + interval_time[workflow_i-1])

    Output_path = "Output\\"
    if workflowN <= 5:
        for workflow in workflows:
            Output_path += workflow + "+"
        Output_path = Output_path[:-1] + "\\"
    else:
        Output_path += str(workflowN) + " Workflows\\"
    if not os.path.exists(Output_path):
        os.makedirs(Output_path)

    [workflow_taskN, taskN, job, inData, outData, indegree, ref] = Read_workflows(workflows)
    [Data_ij, Amount, Father_son, Son_father] = Pre_Process(workflow_taskN, job, inData, outData)
    print("The number of all tasks of all workflows is {:d}, where each workflow is {}.".format(workflow_taskN, taskN))
    for workfl in range(1, len(taskN)):
        taskN[workflow_i] += taskN[workflow_i-1]
    taskN.insert(0, 0)  # taskN由当前任务数量变为累计任务数量

    base = 2.5  # 控制截止时间的松紧程度
    Deadline = [0] * len(workflows)
    for workflow_i in range(len(workflows)):
        Workflow_path = "Output\\Workflow\\" + workflows[workflow_i] + "\\"
        with open(Workflow_path + 'Deadline.txt', 'r') as f:
            Deadline[workflow_i] = arrival_time[workflow_i] + base * eval(f.read())
    print("HEFT's Deadline is {}ms.".format(Deadline))
    # 1.5 3 5 8 15

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

    start = time.process_time()
    N = 10  # 进行10次调度
    for k in range(N):
        w = 0
        c1 = 0
        c2 = 0
        generation = 1
        # now = 1
        particle = Chromosome([0] * dataCenterN, [0] * workflow_taskN, [0] * workflow_taskN)
        population = [particle] * pop_size
        g_best = particle
        p_best = [particle] * pop_size
        MSPG(k + 1)
        sum_cost = fuzzy_sum(sum_cost, g_best.run_cost)
        average = fuzzy_divide(sum_cost, N)
        if fuzzy_fitness(worst) < fuzzy_fitness(g_best.run_cost):
            worst = g_best.run_cost
            worst_makespan = g_best.makespan
            worst_deadline = g_best.meet_deadline
        if optimal == (-1, -1, -1) or fuzzy_fitness(optimal) > fuzzy_fitness(g_best.run_cost):
            optimal = g_best.run_cost
            optimal_makespan = g_best.makespan
            optimal_deadline = g_best.meet_deadline
    end = time.process_time()

    print("Process Exe_time:  {:.2f}s.".format(end - start))
    avg = [tuple([round(i, 3) for i in average])] + [fuzzy_fitness(average)]
    result.append(avg)
    print("Average Run_cost: ", avg)
    wor = [tuple([round(i, 3) for i in worst])] + [fuzzy_fitness(worst)] + \
          [tuple([round(i, 3) for i in worst_makespan])] + [worst_deadline == workflowN] + [worst_deadline]
    result.append(wor)
    print("Worst Run_cost: ", wor)
    opt = [tuple([round(i, 3) for i in optimal])] + [fuzzy_fitness(optimal)] + \
          [tuple([round(i, 3) for i in optimal_makespan])] + [optimal_deadline == workflowN] + [optimal_deadline]
    result.append(opt)
    print("Optimal Run_cost: ", opt)
    Output_File(Output_path, "MSPG", base, iteration_cost, iteration_time, result)
