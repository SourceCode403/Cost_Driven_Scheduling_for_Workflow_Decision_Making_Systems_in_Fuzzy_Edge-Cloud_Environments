# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
import xlwt


def print_ok(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'OK, {name}!')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    raw_workflows = [
        "CyberShake_100", "Epigenomics_100", "Inspiral_100", "Montage_100", "Sipht_97",
        "CyberShake_50", "Epigenomics_47", "Inspiral_50", "Montage_50", "Sipht_58",
        "CyberShake_30", "Epigenomics_24", "Inspiral_30", "Montage_25", "Sipht_29"
    ]
    for workflow in raw_workflows:
        os.system("python HEFT/HEFT.py " + workflow)
    print_ok('Success!')

    workflows = []
    for _ in range(50):
        workflows.append(raw_workflows[random.randint(10, 14)])

    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet，并给出工作表名（sheet）
    worksheet = workbook.add_sheet('Workflows')

    # 写入excel，参数对应 行, 列, 值
    for i in range(3):
        for j in range(200):
            worksheet.write(i, j, label=raw_workflows[random.randint(10-i*5, 14-i*5)])

    # 保存，文件不存在就自动新建，否则在原本文件的基础上添加数据，指定文件路径即名称
    workbook.save('Output\\Random_Workflows.xls')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
