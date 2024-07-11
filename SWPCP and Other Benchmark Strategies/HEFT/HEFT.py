import os
import csv
import sys


def Read_workflow():
    data_ij = [0]
    w_ij = [0]
    with open(Output_path + 'Data_ij.csv', 'r', newline='') as file:
        reader = csv.reader(file, dialect='excel')
        for line in reader:
            temp = [0]
            temp.extend(list(map(float, line))+[0])
            data_ij.append(temp)
        # print("Read Data_ij Succeed! ", end='  ')
    with open(Output_path + 'W_ij.csv', 'r', newline='') as file:
        reader = csv.reader(file, dialect='excel')
        for line in reader:
            w_ij.append(list(map(float, line)))
        # print("Read W_ij Succeed! ")
    TaskN = len(data_ij) - 1

    data_ij[0] = [-1.0] * (TaskN + 2)
    data_ij.append([-1.0] * (TaskN + 2))  # data_ij[TaskN + 1]
    w_ij[0] = [0.0] * serverN
    w_ij.append([0.0] * serverN)  # w_ij[TaskN + 1]
    return [TaskN, data_ij, w_ij]  # 形参，与实参区别


class Task(object):
    def __init__(self,):
        self.EST = -1.0  # 最早开始时间
        self.EFT = -1.0  # 最早完成时间
        # self.LFT = -1  # 最迟完成时间
        # self.AST = -1  # 实际开始时间
        self.server = -100  # 运行的服务器编号


def pre_process():
    outdegree = [0] * (taskN + 2)
    indegree = [0] * (taskN + 2)
    for i in range(1, taskN + 1):
        for j in range(1, taskN + 1):
            if Data_ij[i][j] > -1:
                outdegree[i] += 1
                indegree[j] += 1
    for i in range(1, taskN + 1):
        if indegree[i] < 1e-10:
            Data_ij[0][i] = 0.0
        if outdegree[i] < 1e-10:
            Data_ij[i][taskN + 1] = 0.0
    Task[0].EST = 0.0
    Task[0].EFT = 0.0
    Task[0].server = -1
    # Task[0].LFT = 0
    # Task[0].AST = 0


def min_server(task):  # 任务task完成时间最小的服务器
    min_s = -100
    finish_time = float('inf')
    for server in range(serverN):
        if Server[server] + W_ij[task][server] < finish_time:
            finish_time = Server[server] + W_ij[task][server]
            min_s = server
    return [finish_time, min_s]


def HEFT():
    for i in range(1, taskN + 1):
        if Data_ij[0][i] > -1:  # 实际入节点
            [Task[i].EFT, Task[i].server] = min_server(i)
            Task[i].EST = Task[i].EFT - W_ij[i][Task[i].server]
            Server[Task[i].server] = Task[i].EFT

    for i in range(2, taskN + 2):
        if Task[i].server > -1:
            continue
        new_server = -100
        flag = False
        EFT_min = float('inf')
        for s in range(serverN):  # 选择完成时间最小的服务器
            EST_max = -1
            for j in range(1, i):  # 非入节点
                if Data_ij[j][i] > -1:
                    flag = True
                    EFT = max(Task[j].EFT, Server[s])
                    tran_time = Data_ij[j][i] / BandWidth[Task[j].server][s]
                    if EFT + tran_time > EST_max:
                        EST_max = EFT + tran_time
            if EST_max + W_ij[i][s] < EFT_min:
                EFT_min = EST_max + W_ij[i][s]
                new_server = s

        if flag:
            Task[i].EFT = EFT_min
            Task[i].EST = EFT_min - W_ij[i][new_server]
            Task[i].server = new_server
            Server[Task[i].server] = Task[i].EFT  # 差点忘了，很重要
    # for i in range(1, taskN + 1):
    #     print(i, Task[i].server, Task[i].EFT)


if __name__ == '__main__':
    cloudN = 5
    edgeN = 5
    serverN = cloudN + edgeN
    Band_Width = {'same_server': float('inf'), 'cloud': 2.5, 'edge': 12.5, 'cloud-edge': 1.0}  # 单位：MB/s
    # Band_Width = {'same_server': float('inf'), 'cloud': 0.20, 'edge': 1.00, 'cloud-edge': 0.15}  # 单位：MB/s
    BandWidth = [[0.0] * serverN for _ in range(serverN)]
    for ii in range(serverN):
        for jj in range(serverN):
            if ii == jj:
                BandWidth[ii][jj] = float('inf')
            elif ii < cloudN and jj < cloudN:
                BandWidth[ii][jj] = Band_Width['cloud']
            elif ii >= cloudN and jj >= cloudN:
                BandWidth[ii][jj] = Band_Width['edge']
            else:
                BandWidth[ii][jj] = Band_Width['cloud-edge']

    workflow = sys.argv[1]  # "CyberShake_50"
    os.system("python HEFT/Read_workflow.py " + workflow)
    Output_path = "Output\\Workflow\\" + workflow + "\\"
    [taskN, Data_ij, W_ij] = Read_workflow()
    Task = [Task() for _ in range(taskN + 2)]
    Server = [0.0] * serverN

    pre_process()
    HEFT()
    # Deadline = 0
    # for task_i in range(1, taskN + 1):
    #     if Task[task_i].EFT > Deadline:
    #         Deadline = Task[task_i].EFT
    Deadline = Task[taskN + 1].EFT
    with open(Output_path + 'Deadline.txt', 'w') as f:
        f.write('%.4f' % Deadline)
    print("The deadline of " + sys.argv[1] + " is {:.4f}ms.".format(Deadline))
    # print(Task[i].server for i in range(taskN)])
