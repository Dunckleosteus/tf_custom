import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from IPython.display import display, HTML
import sys
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import openpyxl
from openpyxl import load_workbook
import os 
import xlsxwriter
from PIL import Image 
from openpyxl import Workbook
from tqdm import tqdm
# the dimension of the images can be ajusted here

width = 1000
height = 1028

path_to_model = "efficientdet_lite2_detection_1"
detector = tf.saved_model.load(path_to_model)  # load the saved model
# setup csv fields
data_root_folder = r"/media/throgg/KINGSTON/Output/" # root folder containing all data 
# fuse all the csv in a folder
csv_name = "LM003_CL1_vt.csv"


filenames = glob.glob(data_root_folder + "*.csv")
finalexcelsheet = pd.DataFrame()
for file in tqdm(filenames):
    df= pd.read_csv(file)
    path_field = "image1" # add a gui element 
    df["car_in_pic"] = "no"  # create new field, default value is no 
    df["number_cars"] = 0
    for i in tqdm(df.index):
        image = data_root_folder+df.at[int(i), path_field]
        # Load image by Opencv2
        img = cv2.imread(image)
        # Resize to respect the input_shape
        inp = cv2.resize(img, (width, height))
        # Convert img to RGB
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        img_boxes = rgb
        # COnverting to uint8
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        # Add dims to rgb_tensor
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
        # Loading csv with labels of classes
        boxes, scores, classes, num_detections = detector(rgb_tensor)
        # Processing outputs
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]
        pred_labels = classes.numpy().astype('int')[0]
        
        number_of_cars = 0
        for score, (ymin, xmin, ymax, xmax), lab in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.5:
                continue
            if lab in [3, 4, 6, 8]:
                df.at[int(i), "car_in_pic"] = "yes"
                img_boxes = cv2.rectangle(rgb, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)
                number_of_cars += 1 
        df.at[int(i), "number_cars"] = number_of_cars
        # displays the image and the bounding box briefly on the screen 
        cv2.imshow("title", img_boxes)
        cv2.waitKey(1)
    finalexcelsheet = pd.concat([finalexcelsheet, df])
finalexcelsheet.to_excel(r"merged.xlsx")




