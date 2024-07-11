import sys
import xml.etree.ElementTree as ET
import re
import queue
import csv
import os


class Job(object):
    def __init__(self, _id, name, runtime):
        self.id = _id
        self.name = name
        self.amount = runtime * computing_capability  # 计算量=运行时间*计算能力


def Toposort():
    que = queue.Queue()
    topolist = []
    cnt = 0
    for i in range(jobsize):
        if Indegree[i] == 0:
            que.put_nowait(i)
    while not que.empty():
        v = que.get_nowait()
        topolist.append(v)
        cnt += 1
        for w in range(jobsize):
            if ref[v][w]:
                Indegree[w] -= 1
                if Indegree[w] == 0:
                    que.put_nowait(w)
    if cnt != jobsize:
        print("Error: There exists a loop in the workflow! \n")
        return None
    # print("Topolist =", topolist)
    return topolist


def PreProcess():
    for new_i in range(jobsize):
        i = Topolist[new_i]
        amount = job[i].amount
        Capability = [2.5, 3.5, 5, 7.5, 10, 2.5, 2.6, 2.2, 2.3, 2.7]  # 单位：Mbps（待定）
        for server in range(0, serverN):
            W_ij[new_i][server] = amount / Capability[server]
        for new_j in range(new_i + 1, jobsize):
            j = Topolist[new_j]
            Data_key = list(outData[i].keys() & inData[j].keys())
            if Data_key:
                Data_ij[new_i][new_j] = 0
                for key in Data_key:
                    Data_ij[new_i][new_j] += outData[i][key] / 1e6
    with open(Output_path + 'Data_ij.csv', 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerows(Data_ij)
        # print("Write Data_ij Succeed! ", end='  ')
    with open(Output_path + 'W_ij.csv', 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerows(W_ij)
        # print("Write W_ij Succeed! ")
    Total_Runtime = 0.0
    for i in range(jobsize):
        Total_Runtime += W_ij[i][0]
    print("The total runtime of " + sys.argv[1] + " is {:.4f}ms.".format(Total_Runtime))


if __name__ == '__main__':
    workflow = sys.argv[1]  # "CyberShake_50"
    Data_path = "Data\\" + workflow + ".xml"
    Output_path = "Output\\Workflow\\" + workflow + "\\"
    if not os.path.exists(Output_path):
        os.makedirs(Output_path)
    tree = ET.parse(Data_path)  # Epigenomics_100.xml
    root = tree.getroot()
    cloudN = 5
    edgeN = 5
    serverN = cloudN + edgeN
    computing_capability = 12.5  # 计算能力，单位：Mbps（待定）
    jobsize = eval(root.attrib['jobCount'])
    job = []
    inData = [{} for _ in range(jobsize)]
    outData = [{} for _ in range(jobsize)]
    Indegree = [0] * jobsize
    ref = [[False] * jobsize for _ in range(jobsize)]
    Data_ij = [[-1] * jobsize for _ in range(jobsize)]
    W_ij = [[0.0] * serverN for _ in range(jobsize)]
    count = -1
    for child in root:
        if re.search('job$', child.tag):
            info1 = child.attrib
            job.append(Job(info1['id'], info1['name'], eval(info1['runtime'])))
            count += 1
            for offspring in child:
                info2 = offspring.attrib
                if info2['link'] == 'input':
                    inData[count][info2['file']] = eval(info2['size'])
                elif info2['link'] == 'output':
                    outData[count][info2['file']] = eval(info2['size'])
        elif re.search('child$', child.tag):
            son = eval(re.search('[1-9]\\d*$|0$', child.attrib['ref']).group(0))
            for offspring in child:
                father = eval(re.search(r'[1-9]\d*$|0$', offspring.attrib['ref']).group(0))
                ref[father][son] = True
                Indegree[son] += 1
    Topolist = Toposort()
    PreProcess()
