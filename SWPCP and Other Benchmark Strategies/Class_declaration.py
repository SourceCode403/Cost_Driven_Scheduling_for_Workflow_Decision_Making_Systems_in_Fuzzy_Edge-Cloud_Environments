# -*- coding: utf-8 -*-

computing_capability = 12.5  # 计算能力，单位：Gbps（待定）


class JOB(object):
    def __init__(self, _id, name, runtime):
        self.id = _id
        self.name = name
        self.amount = runtime * computing_capability  # 计算量=运行时间*计算能力


class CmpObj:  # 可比较对象，放入优先队列中
    def __init__(self, num, pri):
        self.number = num
        self.priority = pri

    def __lt__(self, other):  # 自定义小于运算符'<'
        return self.priority > other.priority


class Task(object):
    def __init__(self, _id, server):
        self._id = _id  # 任务编号
        self.server = server  # 运行的服务器编号
        self.start_time = (0.000, 0.000, 0.000)  # 开始时间
        self.end_time = (0.000, 0.000, 0.000)  # 结束时间
        self.com_time = (0.000, 0.000, 0.000)  # 计算时间


class Server(object):
    def __init__(self, environment, per_time, capability, per_cost):
        self.environment = environment  # 环境：0为云，1为边缘
        self.status = False  # 服务器的状态：开启或关闭
        self.per_time = per_time  # 单位要价时间
        self.capability = capability  # 计算能力
        self.per_cost = per_cost  # 单位计算代价
        self.start_time = (0.000, 0.000, 0.000)  # 开启时间
        self.end_time = (0.000, 0.000, 0.000)  # 关闭时间
        self.rent_time = (0.000, 0.000, 0.000)  # 租赁时间
