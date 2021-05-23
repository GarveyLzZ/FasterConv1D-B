#----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
from keras.layers import Input
from frcnn import FRCNN
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from utils.anchors import get_anchors
from utils.utils import BBoxUtility
from nets.frcnn_training import get_new_img_size
import math
import copy
import numpy as np
import os
class mAP_FRCNN(FRCNN):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, raman_data, raman_path):
        self.confidence = 0.05
        f = open(raman_path,"w")
        print(raman_path)
        raman_data = np.array(list(map(float, raman_data)), dtype=np.float32).reshape(-1, 1, 1)
        raman_shape = np.array(np.shape(raman_data)[0:2])
        old_width = raman_shape[0]
        old_height = raman_shape[1]

        raman = np.array(raman_data, dtype=np.float64)

        raman = (raman - (np.min(raman))) / (np.max(raman) - np.min(raman))
        raman = np.expand_dims(raman, 0)

        preds = self.model_rpn.predict(raman)
        # 将预测结果进行解码
        anchors = get_anchors((66, 1), old_width, old_height)
        preds[1][..., 3] = 1
        anchors[:, 1] = 0
        rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
        R = rpn_results[0][:, 2:]

        R[:, 0] = np.array(np.round(R[:, 0] * old_width / self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * old_height), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * old_width / self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * old_height), dtype=np.int32)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]

        delete_line = []
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R, delete_line, axis=0)

        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)

            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :-1]) < self.confidence:
                    continue

                label = np.argmax(P_cls[0, ii, :-1])

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :-1])

                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w / 2.
                cy = y + h / 2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.

                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(P_cls[0, ii, :-1]))
                labels.append(label)

        if len(bboxes) == 0:
            return

        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / old_width
        boxes[:, 3] = boxes[:, 3] * old_height
        results = np.array(
            self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

        top_label_indices = results[:, 0]
        top_conf = results[:, 1]
        boxes = results[:, 2:]
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height


        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = str(top_conf[i])

            left, top, right, bottom = boxes[i]

            # left = max(1, np.floor(left + 0.5).astype('int32'))
            # right = min(1043, np.floor(right + 0.5).astype('int32'))

            left = max(-30, np.floor(left - 0.5).astype('int32') * 4)
            right = min(4080, np.floor(right - 0.5).astype('int32') * 4)

            # print(label ,"  ", "[", left , ", " , right , "]", " ", "[",  X[left-1], ",", X[right-1], "]")
            # plt.axvspan(xmin=X[left-1], xmax=X[right-1], facecolor='y', alpha=0.3)
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)+1), str(int(right)),str(int(bottom)+1)))

        f.close()
        return 

frcnn = mAP_FRCNN()

with open('./raman_data/jia/predict_data.txt', 'r') as raman_file:
    raman_data = raman_file.readlines()
    for i in range(0, len(raman_data)):
        raman = raman_data[i].rstrip('\n').split(',')
        for j in range(len(raman)):
            raman[j] = float(raman[j])
        raman_path = "./Object-Detection-Metrics-master/detection-results/raman_" + str(i) + ".txt"
        frcnn.detect_image(raman, raman_path)
        # print("raman_" + str(i) + ".txt completed")

print("Conversion completed!")
