# -*- coding: utf-8 -*-
# @Project : image-enhancement-utils
# @File    : lime-example.py
# @Author  : MQB
# @Email   : mengqingbin@aliyun.com
# @Time    : 2018-05-18 23:03


import cv2
import sys
import os

from algorithms import lime

file_path = sys.argv[1]
file_name = os.path.split(file_path)[1]
raw_image = cv2.imread(file_path)

img_ws1 = lime.lime(raw_image, weight_strategy=1)
img_ws2 = lime.lime(raw_image, weight_strategy=2)
img_ws3 = lime.lime(raw_image, weight_strategy=3)
img_ws4 = lime.lime(raw_image, weight_strategy=4)

cv2.imwrite('./%s_ws1.jpg' % file_name, img_ws1)
cv2.imwrite('./%s_ws2.jpg' % file_name, img_ws2)
cv2.imwrite('./%s_ws3.jpg' % file_name, img_ws3)
cv2.imwrite('./%s_ws4.jpg' % file_name, img_ws4)

cv2.imshow('raw_image', raw_image)
cv2.imshow('new_image_ws1', img_ws1)
cv2.imshow('new_image_ws2', img_ws2)
cv2.imshow('new_image_ws3', img_ws3)
cv2.imshow('new_image_ws4', img_ws4)

cv2.waitKey()
