#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import numpy as np

raman_path = './raman_data/jia/groundtrue_data.txt'
with open(raman_path) as f:
    raman_data = f.readlines()
    num = 0
    for line in raman_data:
        raman_path = "./Object-Detection-Metrics-master/ground-truth/raman_" + str(num) + ".txt"
        print(raman_path)
        new_f = open(raman_path, "w")
        line = line.split()
        box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
        box_data = np.zeros((len(box), 5))
        box_data[:len(box)] = box
        boxes = np.array(box_data[:, :4], dtype=np.float32)
        for i in boxes:
            left = i[0] * 4
            right = i[2] * 4
            new_f.write("%s %s %s %s %s\n" % ("cancer", str(int(left)), str(1), str(int(right)), str(2)))
        num += 1

print("Conversion completed!")
