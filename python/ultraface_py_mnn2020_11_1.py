#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
""" 
@author:linzai 
@file: ultraface_py_mnn.py 
@time: 2019-11-25 
"""
from __future__ import print_function

import os
import requests
import json
import base64
import argparse
import sys
import time
from math import ceil

import MNN
import cv2
import numpy as np
import torch
import multiprocessing as mp
import face_recognition
from utils import file_processing,image_processing

sys.path.append('../../')
import box_utils_numpy as box_utils

parser = argparse.ArgumentParser(description='run ultraface with MNN in py')
parser.add_argument('--model_path', default="../models/MNN_model/version-RFB/RFB-320.mnn", type=str, help='model path')
parser.add_argument('--input_size', default="320,240", type=str,
                    help='define network input size,format: width,height')
parser.add_argument('--threshold', default=0.7, type=float, help='score threshold')
parser.add_argument('--imgs_path', default="../imgs", type=str, help='imgs dir')
parser.add_argument('--videos_path', default="../video", type=str, help='video dir')
parser.add_argument('--results_path', default="results", type=str, help='results dir')
args = parser.parse_args()

resize_width = 160
resize_height = 160
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
strides = [8, 16, 32, 64]

# url = "http://127.0.0.1:5000/upFaceData"
url = "http://124.71.140.59:8080/api/faceCompute/faceInfo"
payload = {
    "robotid":"123456abc",
    "monitorDeviceid":"qwe123456",
    "detectionResult":""
}

def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [ceil(size / stride) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def compare_embadding(pred_emb, dataset_emb, names_list,threshold=0.7):
    # 为bounding_box 匹配标签
    pred_num = len(pred_emb)
    dataset_num = len(dataset_emb)
    pred_name = []
    pred_score=[]
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
            dist_list.append(dist)
        min_value = min(dist_list)
        pred_score.append(min_value)
        if (min_value > threshold):
            pred_name.append('unknow')
        else:
            pred_name.append(names_list[dist_list.index(min_value)])
    return pred_name,pred_score

def load_dataset(dataset_path,filename):
    '''
    加载人脸数据库
    :param dataset_path: embedding.npy文件（faceEmbedding.npy）
    :param filename: labels文件路径路径（name.txt）
    :return:
    '''
    embeddings=np.load(dataset_path)
    names_list=file_processing.read_data(filename,split=None,convertNum=False)
    return embeddings,names_list

def inferenceHtptsVideo(window_name,frame,dataset_emb,names_list,interpreter,input_tensor,session,face_net):
    input_size = [int(v.strip()) for v in args.input_size.split(",")]
    priors = define_img_size(input_size)
    interpreter.runSession(session)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(input_size))
    image = image.astype(float)
    image = (image - image_mean) / image_std
    image = np.asarray(image, dtype=np.float32)
    image = image.transpose((2, 0, 1))
    tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    time_time = time.time()
    scores = interpreter.getSessionOutput(session, "scores").getData()
    boxes = interpreter.getSessionOutput(session, "boxes").getData()
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = box_utils.center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(frame.shape[1], frame.shape[0], scores, boxes, args.threshold)
    files = []
    all_face_results = []
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        # remove to small face
        min_size = 5
        if ((box[3] - box[1]) < min_size and (box[2] - box[0]) > min_size)\
                or ((box[3] - box[1]) > min_size and (box[2] - box[0]) < min_size)\
                or ((box[3] - box[1]) < min_size and (box[2] - box[0]) < min_size):
            continue

        # up image ============
        tmp_cut_face = frame[max(0,box[1] - 100):box[3]+100, max(0,box[0]-100):box[2]+100]
        tmp_cut_face = cv2.resize(tmp_cut_face, (400, 500))

        #  cut face from image
        cut_face = frame[max(0,box[1]):box[3], max(0,box[0]):box[2]]
        cut_face = cv2.resize(cut_face, (160, 160))
        cut_face = cut_face.reshape((1,160,160,3))
        # FaceNet chrack ================================================ people / 20ms
    #     pred_emb = face_net.get_embedding(cut_face)
    #     pred_name, pred_score = compare_embadding(pred_emb, dataset_emb, names_list,0.7)
    #     # 在图像上绘制人脸边框和识别的结果
    #     show_info = [n + ':' + str(s)[:5] for n, s in zip(pred_name, pred_score)]
    #     all_face_results.append(pred_name[0])
    #     files.append(("face", np.array(cv2.imencode('.png', tmp_cut_face)[1]).tobytes()))
    #
    #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #     cv2.putText(frame, show_info[0], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    # # 保存或者上传数据
    # if len(files) != 0:
    #     # 拼接参数
    #     headers = {}
    #     payload["detectionResult"] = ",".join(all_face_results)
    #     print(payload)
    #     res = requests.request("POST", url, headers=headers, data=payload, files=files)
    #     print(res.text)
    print("inference time: {} s".format(round(time.time() - time_time, 4)))
    # cv2.imshow(window_name, frame)
    # cv2.waitKey(1)

# url = 'rtsp://admin:flm2019hb@192.168.101.107:554'
def image_put(q, user, pwd, ip, channel=1):
    # cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))
    # if cap.isOpened():
    #     print('HIKVISION')
    # else:
    #     cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
    #     print('DaHua')
    cap = cv2.VideoCapture("rtsp://admin:flm2019hb@192.168.101.107:554")
    start_time = time.time()
    while cap.isOpened():
        # 每处理100帧后 重新链接摄像头 及清除摄像头缓存 防止崩溃
        print((time.time() - start_time))
        if (time.time() - start_time) > 10:
            cap.release()
            start_time = time.time()
            print("开始重新连接摄像头")
            cap = cv2.VideoCapture("rtsp://admin:flm2019hb@192.168.101.107:554")
        q.put(cap.read()[1])
        if q.qsize() > 9:
            q.get()
        else:
            time.sleep(0.4)
def image_get(q, window_name):
    # 读取本地视频
    # cap = cv2.VideoCapture("../video/Crystal.mp4")
    while True:
        frame = q.get()
        print("========================")
        time.sleep(0.02)
        # _, tmp_frame=cap.read()
        # inferenceHtptsVideo(window_name,frame,dataset_emb,names_list,interpreter,input_tensor,session,face_net)
# url = 'rtsp://admin:flm2019hb@192.168.101.107:554'
def run_multi_camera():
    # rtsp://admin:Xhs88283111@192.168.2.5/Streaming/Channels/1
    user_name, user_pwd = "admin", "flm2019hb"
    camera_ip_l = [
        "192.168.2.5",  # ipv4
        # "192.168.2.30",
        # "192.168.2.23",
        # "192.168.2.22"
    ]
    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=10) for _ in camera_ip_l]

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()

# 初始化facenet
faceNet_model_path = '../models/FaceNet_model/20180408-102900'
face_net = face_recognition.facenetEmbedding(faceNet_model_path)
# 初始化数据库特征值
dataset_path = '../dataset/emb/faceEmbedding.npy'
filename = '../dataset/emb/name.txt'
dataset_emb, names_list = load_dataset(dataset_path, filename)
# 初始化MNN
interpreter = MNN.Interpreter(args.model_path)
session = interpreter.createSession()
input_tensor = interpreter.getSessionInput(session)
if __name__ == "__main__":
    run_multi_camera() # with 1 + n threads