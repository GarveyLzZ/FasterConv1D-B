import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
import nets.frcnn as frcnn
import rampy
from nets.frcnn_training import get_new_img_size
from keras import backend as K
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image,ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
from tool import *
import copy
import math
import xlrd
import matplotlib.pyplot as plt
from tool import rampy
import scipy.signal as sp

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
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

# 所有数据按列取平均
def Ygetmean(excel):
    mean = [0] * len(excel[0])
    for each in excel:
        for num in range(len(each)):
            mean[num] += each[num]

    for num in range(len(mean)):
        mean[num] = mean[num] / len(excel)

    return mean

def Xexcel(filepath, sheetnum):
    wb = xlrd.open_workbook(filepath)  # 打开Excel文件
    sheet = wb.sheet_by_name(sheetnum)  # 通过excel表格名称(rank)获取工作表
    dat = []  # 创建空list
    for a in range(sheet.nrows):  # 循环读取表格内容（每次读取一行数据）
        cells = sheet.row_values(a)  # 每行数据赋值给cells
        data = cells[0]  # 因为表内可能存在多列数据，0代表第一列数据，1代表第二列，以此类推
        dat.append(data)  # 把每次循环读取的数据插入到list
    return dat

class FRCNN(object):
    _defaults = {
        "model_path": 'model_data/jia_v1.h5',
        "classes_path": 'model_data/voc_classes.txt',
        "confidence": 0.7,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = Config()
        self.generate()
        self.bbox_util = BBoxUtility()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算总的种类
        self.num_classes = len(self.class_names)+1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model_rpn,self.model_classifier = frcnn.get_predict_model(self.config,self.num_classes)
        self.model_rpn.load_weights(self.model_path,by_name=True)
        self.model_classifier.load_weights(self.model_path,by_name=True,skip_mismatch=True)
                
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    
    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3,1,0,0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
            return input_length
        return get_output_length(width), get_output_length(height) 
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, raman_data):

        old_raman = copy.deepcopy(raman_data)

        raman_data = np.array(list(map(float, raman_data)), dtype=np.float32).reshape(-1, 1, 1)
        raman_shape = np.array(np.shape(raman_data)[0:2])
        old_width = raman_shape[0]
        old_height = raman_shape[1]

        raman = np.array(raman_data,dtype = np.float64)

        raman = (raman - (np.min(raman))) / (np.max(raman) - np.min(raman))
        raman = np.expand_dims(raman,0)
        # raman shape = [1,1044,1,1]
        preds = self.model_rpn.predict(raman)
        # 将预测结果进行解码
        anchors = get_anchors((66, 1), old_width, old_height)
        # preds rpn的预测结果 共有三个维度
        # 第一纬度 (1,198,1) 是包含物体的置信的
        # 第二维度 (1,198,4) 是先验框的调整参数
        # 第三个维度 (1,66,1,1024) 是feature map
        preds[1][..., 3] = 1
        anchors[:, 1] = 0
        rpn_results = self.bbox_util.detection_out(preds,anchors,1,confidence_threshold=0)
        R = rpn_results[0][:, 2:]
        
        R[:,0] = np.array(np.round(R[:, 0]*old_width/self.config.rpn_stride),dtype=np.int32)
        R[:,1] = np.array(np.round(R[:, 1]*old_height),dtype=np.int32)
        R[:,2] = np.array(np.round(R[:, 2]*old_width/self.config.rpn_stride),dtype=np.int32)
        R[:,3] = np.array(np.round(R[:, 3]*old_height),dtype=np.int32)
        
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]
        
        delete_line = []
        for i,r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R,delete_line,axis=0)
        
        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0]//self.config.num_rois + 1):
            ROIs = np.expand_dims(R[self.config.num_rois*jk:self.config.num_rois*(jk+1), :], axis=0)
            
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//self.config.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.config.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            
            [P_cls, P_regr] = self.model_classifier.predict([base_layer,ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :-1]) < self.confidence:
                    continue

                label = np.argmax(P_cls[0, ii, :-1])

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :-1])

                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w/2.
                cy = y + h/2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1/2.
                y1 = cy1 - h1/2.

                x2 = cx1 + w1/2
                y2 = cy1 + h1/2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1,y1,x2,y2])
                probs.append(np.max(P_cls[0, ii, :-1]))
                labels.append(label)

        if len(bboxes)==0:
            print("None boxes")

            Raman_shift = Xexcel('./raman_data/raw_data/RamanShift.xlsx', 'Sheet1')
            Normal_data = Yexcel('./raman_data/raw_data/yayin/no_origin_label0.xlsx', 'no_origin_label0')  # Normal
            Normal_data = Ygetmean(Normal_data)

            Cancer_data = np.array(old_raman)
            Raman_shift = np.array(Raman_shift)
            Normal_data = np.array(Normal_data)

            # 截取数据350~4000cm-1 #
            Lower_limit = np.max(np.where(Raman_shift < 350)) + 1
            Upper_limit = np.min(np.where(Raman_shift > 4000)) + 1

            Raman_shift_limit = Raman_shift[Lower_limit:Upper_limit]
            Cancer_data_limit = Cancer_data[Lower_limit:Upper_limit]
            Normal_data_limit = Normal_data[Lower_limit:Upper_limit]

            # SG平滑处理#
            Cancer_data_SG = sp.savgol_filter(Cancer_data_limit, 11, 2)
            Normal_data_SG = sp.savgol_filter(Normal_data_limit, 11, 2)

            # 去基线处理 #
            roi = np.array([[350, 4000]])
            Cancer_data_final, Cancer_base_Intensity = rampy.baseline(Raman_shift_limit, Cancer_data_SG, roi,
                                                                      'arPLS', lam=10 ** 6, ratio=0.001)
            Normal_data_final, Normal_base_Intensity = rampy.baseline(Raman_shift_limit, Normal_data_SG, roi,
                                                                      'arPLS', lam=10 ** 6, ratio=0.001)

            plt.plot(Raman_shift_limit, Normal_data_final, ls="-", lw=2, c="c", label="Normal")
            plt.plot(Raman_shift_limit, Cancer_data_final, ls="-", lw=1, c="b", label="Cancer")

            plt.legend()
            plt.xlabel("yayin")
            # plt.savefig('./raman_data/raw_data/yayin/yayin_alter.jpg')
            plt.show()
        
        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes,dtype=np.float32)
        boxes[:,0] = boxes[:,0]*self.config.rpn_stride/old_width
        boxes[:,1] = boxes[:,1]*old_height
        boxes[:,2] = boxes[:,2]*self.config.rpn_stride/old_width
        boxes[:,3] = boxes[:,3]*old_height
        results = np.array(self.bbox_util.nms_for_out(np.array(labels),np.array(probs),np.array(boxes),self.num_classes-1,0.4))
        
        top_label_indices = results[:,0]
        top_conf = results[:,1]
        boxes = results[:,2:]
        boxes[:,0] = boxes[:,0]*old_width
        boxes[:,1] = boxes[:,1]*old_height
        boxes[:,2] = boxes[:,2]*old_width
        boxes[:,3] = boxes[:,3]*old_height

        # 画基本图
        Raman_shift = Xexcel('./raman_data/raw_data/RamanShift.xlsx', 'Sheet1')
        Normal_data = Yexcel('./raman_data/raw_data/yayin/no_origin_label0.xlsx', 'no_origin_label0')  # Normal
        Normal_data = Ygetmean(Normal_data)

        Cancer_data = np.array(old_raman)
        Raman_shift = np.array(Raman_shift)
        Normal_data = np.array(Normal_data)

        # 截取数据350~4000cm-1 #
        Lower_limit = np.max(np.where(Raman_shift < 350)) + 1
        Upper_limit = np.min(np.where(Raman_shift > 4000)) + 1

        Raman_shift_limit = Raman_shift[Lower_limit:Upper_limit]
        Cancer_data_limit = Cancer_data[Lower_limit:Upper_limit]
        Normal_data_limit = Normal_data[Lower_limit:Upper_limit]

        # SG平滑处理#
        Cancer_data_SG = sp.savgol_filter(Cancer_data_limit, 11, 2)
        Normal_data_SG = sp.savgol_filter(Normal_data_limit, 11, 2)

        # 去基线处理 #
        roi = np.array([[350, 4000]])
        Cancer_data_final, Cancer_base_Intensity = rampy.baseline(Raman_shift_limit, Cancer_data_SG, roi,
                                                                  'arPLS', lam=10 ** 6, ratio=0.001)
        Normal_data_final, Normal_base_Intensity = rampy.baseline(Raman_shift_limit, Normal_data_SG, roi,
                                                                  'arPLS', lam=10 ** 6, ratio=0.001)

        plt.plot(Raman_shift_limit, Normal_data_final, ls="-", lw=2, c="c", label="Normal")
        plt.plot(Raman_shift_limit, Cancer_data_final, ls="-", lw=1, c="b", label="Cancer")

        plt.legend()
        plt.xlabel("yayin")

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = boxes[i]

            # left = max(1, np.floor(left + 0.5).astype('int32'))
            # right = min(1043, np.floor(right + 0.5).astype('int32'))

            left = max(-30, np.floor(left-0.5).astype('int32')*4)
            right = min(4080, np.floor(right-0.5).astype('int32')*4)

            label = '{} {:.2f}'.format(predicted_class, score)
            label = label.encode('utf-8')

            # print(label ,"  ", "[", left , ", " , right , "]", " ", "[",  X[left-1], ",", X[right-1], "]")
            # plt.axvspan(xmin=X[left-1], xmax=X[right-1], facecolor='y', alpha=0.3)

            print(label ,"  ", "[", left , ", " , right , "]")
            plt.axvspan(xmin=left, xmax=right, facecolor='y', alpha=0.3)

        plt.show()

    def close_session(self):
        self.sess.close()
