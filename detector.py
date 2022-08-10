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
def convert_2_raw(input_string):
    return r'{}'.format(input_string)
def choose_image_field(df):
    for i in range (0, len(df.columns)):
        print(f"{str(i)} => {df.columns[i]}")
    try: 
        result = int(input("enter the columns number to select: ")) # TODO: check that input is an integer
    except:
        print("invalid input")
        result = int(input("enter the columns number to select: ")) # TODO: check that input is an integer
    return df.columns[result]
width = 800
height = 600
#
path_to_model = os.path.join("efficientdet_lite2_detection_1")
detector = tf.saved_model.load(path_to_model)  # load the saved model
# setup csv fields
# r"/media/throgg/KINGSTON/Output/"
print ("enter the directory containing the vt files and images")
print ("here is an input example on linux : /media/throgg/KINGSTON/Output/")
data_root_folder =  convert_2_raw(input("enter the directory containing vt files: "))# root folder containing all data 
# fuse all the csv in a folder
filenames = glob.glob(data_root_folder + "*.csv")
finalexcelsheet = pd.DataFrame()
path_field = ""
for file in tqdm(filenames):
    df= pd.read_csv(file)
    if path_field == "": # the user only has to be asked about the path field if the program doesn't know it
        path_field = choose_image_field(df)
    df["car_in_pic"] = "no"  # create new field, default value is no 
    df["number_cars"] = 0
    for i in tqdm(df.index):
        image = data_root_folder+df.at[int(i), path_field]
        # Load image by Opencv2
        img = cv2.imread(image)
        # Resize to respect the input_shape
        inp = cv2.resize(img, (width, height))
        cropped = inp[150:600, 0:800]
        # Convert img to RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
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
        cv2.imshow("resized", img_boxes)
        cv2.imshow("original", inp)
        cv2.waitKey(1)
    finalexcelsheet = pd.concat([finalexcelsheet, df])
finalexcelsheet.to_excel(os.path.join("output.xlsx"))




