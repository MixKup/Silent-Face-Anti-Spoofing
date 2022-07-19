# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

def crop(org_img, bbox, scale, crop=True):

    if not crop:
        dst_img = cv2.resize(org_img, (80, 80))
    else:
        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, \
            right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)

        img = org_img[left_top_y: right_bottom_y+1,
                      left_top_x: right_bottom_x+1]
        dst_img = cv2.resize(img, (80, 80))
    return dst_img



SAMPLE_IMAGE_PATH = "./images/sample/"


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(model_dir, frame, image_bbox, model_test27, model_test4):


    #image_cropper = CropImage() # padding
    #image = img 

    #image_bbox = model_test.get_bbox(image) # face detect
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    # for model_name in os.listdir(model_dir):
    #     h_input, w_input, model_type, scale = parse_model_name(model_name)
    #     param = {
    #         "org_img": img,
    #         "bbox": image_bbox,
    #         "scale": scale,
    #         "out_w": w_input,
    #         "out_h": h_input,
    #         "crop": True,
    #     }
    #     if scale is None:
    #         param["crop"] = False
    #     img = image_cropper.crop(**param)
    img27 = crop(frame, image_bbox, 2.7, crop=True)
    img4 = crop(frame, image_bbox, 4, crop=True)
    start = time.time()
    prediction += model_test27.predict(img27)
    prediction += model_test4.predict(img4)
    test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image is Real Face. Score: {:.2f}.".format(value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image is Fake Face. Score: {:.2f}.".format(value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        frame,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        frame,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)

    return frame


if __name__ == "__main__":
    desc = "test"
    # parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument(
    #     "--device_id",
    #     type=int,
    #     default=0,
    #     help="which gpu id, [0/1/2/3]")
    # parser.add_argument(
    #     "--model_dir",
    #     type=str,
    #     default="./resources/anti_spoof_models",
    #     help="model_lib used to test")
    # parser.add_argument(
    #     "--image_name",
    #     type=str,
    #     default="image_F1.jpg",
    #     help="image used to test")
    # args = parser.parse_args()
    cap = cv2.VideoCapture(-1)
    #caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
    caffemodel = '/home/mix_kup/Desktop/intact/intact/soft_alko_nuc/res10_300x300_ssd_iter_140000.caffemodel'
    #deploy = "./resources/detection_model/deploy.prototxt"
    deploy = '/home/mix_kup/Desktop/intact/intact/soft_alko_nuc/deploy.prototxt'
    
    model_test1 = AntiSpoofPredict(0, 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth') # device_id - for cuda:{}
    model_test2 = AntiSpoofPredict(0, 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth')
    net = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
    while True:
        rects = []
        _, frame = cap.read()
        if _:

            blob = cv2.dnn.blobFromImage(frame, 1.0, (frame.shape[1], int(frame.shape[1]*0.75)))
        
            #print(blob.shape)
            net.setInput(blob)
            detections = net.forward()
            #rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            # bboxes = facedet.predict(rgb, 0.8)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #rect_dlib = dlib_det(gray, 0)
            #print(rect_dlib)
            for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
                if detections[0, 0, i, 2] >= 0.99:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object, then update the bounding box rectangles list
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    #bbs.append(np.array([box[0], box[1], box[2], box[3], detections[0, 0, i, 2]]))
                    (startX, startY, endX, endY) = box.astype("int")
                    #print(rects)
                    if (endX > frame.shape[0] or endY > frame.shape[1]): continue
                    rects.append(np.array([box[0], box[1], box[2]-box[0], box[3]-box[1]]).astype("int"))

            for rec in rects:
                #face = crop_image(frame, rec)
                img = test('./resources/anti_spoof_models', frame, rec, model_test1, model_test2)
        cv2.imshow('ids.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break