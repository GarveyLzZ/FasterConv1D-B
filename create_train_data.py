import xlrd
import random
from random import randint, sample

#读取excel文件
def Yexcel(filepath, sheetnum):
    wb = xlrd.open_workbook(filepath)# 打开Excel文件
    sheet = wb.sheet_by_name(sheetnum)#通过excel表格名称(rank)获取工作表
    dat = []  #创建空list
    for a in range(sheet.nrows):  #循环读取表格内容（每次读取一行数据）
                cells = sheet.row_values(a)  # 每行数据赋值给cells
                data=cells[0]#因为表内可能存在多列数据，0代表第一列数据，1代表第二列，以此类推
                data = data.split(',')
                data = list(map(float, data))
                dat.append(data) #把每次循环读取的数据插入到list
    return dat

def Xexcel(filepath, sheetnum):
    wb = xlrd.open_workbook(filepath)  # 打开Excel文件
    sheet = wb.sheet_by_name(sheetnum)  # 通过excel表格名称(rank)获取工作表
    dat = []  # 创建空list
    for a in range(sheet.nrows):  # 循环读取表格内容（每次读取一行数据）
        cells = sheet.row_values(a)  # 每行数据赋值给cells
        data = cells[0]  # 因为表内可能存在多列数据，0代表第一列数据，1代表第二列，以此类推
        dat.append(data)  # 把每次循环读取的数据插入到list
    return dat

def getTrainFile(train_path):
    train_file = open(train_path, 'w')
    print("训练数据: ", len(train_data))
    for line in train_data:
        for i in range(len(line)):
            flag = 0
            # 获取每一行的数据 每行最后一个数字不加","并换行
            if i != len(line) - 1:
                train_file.write(str(line[i]) + ",")
            else:
                train_file.write(str(line[i]) + " ")
                for i in box:
                    ran = (X[i] + random.uniform(-6, 6)) / 4
                    train_file.write(str(ran) + ",1,")
                    flag += 1
                    if flag % 2 == 0:
                        if flag == len(box):
                            train_file.write("0")
                        else:
                            train_file.write("0 ")
                train_file.write("\n")
    train_file.close()

def getPredictFile(predict_path, groundtrue_path):
    predict_file = open(predict_path, 'w')
    groundtrue_file = open(groundtrue_path, 'w')
    print("测试数据: ", len(predict_data))

    for line in predict_data:
        for i in range(len(line)):
            flag = 0
            # 获取每一行的数据 每行最后一个数字不加","并换行
            if i != len(line) - 1:
                groundtrue_file.write(str(line[i]) + ",")
                predict_file.write(str(line[i]) + ",")
            else:
                groundtrue_file.write(str(line[i]) + " ")
                predict_file.write(str(line[i]) + "\n")
                for i in box:
                    ran = (X[i] + random.uniform(-6, 6)) / 4
                    groundtrue_file.write(str(ran) + ",1,")
                    flag += 1
                    if flag % 2 == 0:
                        if flag == len(box):
                            groundtrue_file.write("0")
                        else:
                            groundtrue_file.write("0 ")
                groundtrue_file.write("\n")
    predict_file.close()
    groundtrue_file.close()

raman_shift_path = './raman_data/raw_data/RamanShift.xlsx'
all_data_path = './raman_data/raw_data/jia/no_origin_label1.xlsx'
train_path = 'raman_data/jia/train_data.txt'
predict_path = 'raman_data/jia/predict_data.txt'
groundtrue_path = 'raman_data/jia/groundtrue_data.txt'

X = Xexcel(raman_shift_path, 'Sheet1') # raman_shift

all_data = Yexcel(all_data_path, 'no_origin_label1') # cancer 获取表中所有数据
train_data = sample(all_data, int(len(all_data) * 0.85)) # 训练数据占总数据的百分比
predict_data = [ i for i in all_data if i not in train_data ]


# box = [183, 222, 264, 306, 606, 667] # she数据
# box = [130, 160, 176, 208, 222, 235, 255, 292, 312, 336, 608, 697] # yayin数据
box = [126, 165, 174, 209, 214, 250, 308, 338, 608, 697] # jia数据
print("所有数据: ", len(all_data))

getTrainFile(train_path)
getPredictFile(predict_path, groundtrue_path)


