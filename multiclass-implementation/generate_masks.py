import os
import math
import cv2
import pandas as pd
import numpy as np
import shutil

def get_colors_for_classes(classes):
    color_info = {}
    temp = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255], [0, 0, 0]]
    for cls in classes:
        if temp:
            color_info[cls.strip()] = temp.pop(0)
        else:
            color_info[cls.strip()] = np.random.randint(255, size=3, dtype=np.uint8).tolist() #(np.random.rand(3)*255).astype(np.uint8)
    print(color_info)
    return color_info

def process_records_mult_output(input_path):
    records = pd.read_csv(os.path.join(input_path, 'train.csv'))
    classes = records['class'].unique().tolist()
    print(classes)
    color_map = get_colors_for_classes(classes)
    
    masked_directory = os.path.join(input_path, 'masked')
    
    if os.path.exists(masked_directory):
        shutil.rmtree(masked_directory)
    os.makedirs(masked_directory)
    
    for index, row in records.iterrows():
        print('Processing record at index:', index)
        image_path = row["filename"].strip()
        image_path = os.path.join(input_path, image_path)
        if not os.path.isfile(image_path):
            continue
        filename = os.path.basename(image_path)
        masked_img_path = os.path.join(masked_directory, filename)
        
        if os.path.isfile(masked_img_path):
            image = cv2.imread(masked_img_path)
        else:
            input = cv2.imread(image_path)
            height, width, _ = input.shape
            image = np.zeros((height, width), dtype=np.uint8)
            #image = np.full((height, width, 3), 255, dtype=np.uint8)
        
        class_name = row["class"].strip()
        color = color_map[class_name]
        
        xmin = math.floor(float(row["xmin"]))
        ymin = math.floor(float(row["ymin"]))
        xmax = math.ceil(float(row["xmax"]))
        ymax = math.ceil(float(row["ymax"]))
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, -1)
        cv2.imwrite(masked_img_path, image)
        
def process_records_binary(input_path):
    records = pd.read_csv(os.path.join(input_path, 'train.csv'))
    classes = records['class'].unique().tolist()
    print(classes)
    color_map = get_colors_for_classes(classes)
    
    masked_directory = os.path.join(input_path, 'masked')
    
    if os.path.exists(masked_directory):
        shutil.rmtree(masked_directory)
    os.makedirs(masked_directory)
    
    for index, row in records.iterrows():
        image_path = row["filename"].strip()
        image_path = os.path.join(input_path, image_path)
        if not os.path.isfile(image_path):
            continue
        filename = os.path.basename(image_path)
        masked_img_path = os.path.join(masked_directory, filename)
        
        if os.path.isfile(masked_img_path):
            image = cv2.imread(masked_img_path, 0)
        else:
            input = cv2.imread(image_path)
            height, width, _ = input.shape
            image = np.zeros((height, width, 1), dtype=np.uint8)
            #image = np.full((height, width, 3), 255, dtype=np.uint8)
        
        class_name = row["class"].strip()
        if class_name == 'BT_RasterPicture':
            print('Processing record at index:', index)
            color = 255
            xmin = math.floor(float(row["xmin"]))
            ymin = math.floor(float(row["ymin"]))
            xmax = math.ceil(float(row["xmax"]))
            ymax = math.ceil(float(row["ymax"]))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, -1)
        cv2.imwrite(masked_img_path, image)     

def process_records_new(input_path):
    records = pd.read_csv(os.path.join(input_path, 'train.csv'))
    classes = records['class'].unique().tolist()
    print(classes)
    color_map = get_colors_for_classes(classes)
    
    masked_directory = os.path.join(input_path, 'masked')
    
    if os.path.exists(masked_directory):
        shutil.rmtree(masked_directory)
    os.makedirs(masked_directory)
    
    for index, row in records.iterrows():
        print('Processing record at index:', index)
        image_path = row["filename"].strip()
        image_path = os.path.join(input_path, image_path)
        if not os.path.isfile(image_path):
            continue
        filename = os.path.basename(image_path)
        masked_img_path = os.path.join(masked_directory, filename)
        
        if os.path.isfile(masked_img_path):
            image = cv2.imread(masked_img_path)
        else:
            input = cv2.imread(image_path)
            height, width, _ = input.shape
            image = np.zeros((height, width, 3), dtype=np.uint8)
            #image = np.full((height, width, 3), 255, dtype=np.uint8)
        
        class_name = row["class"].strip()
        if class_name == 'BT_Text':
            color = [0, 0, 255]
            xmin = math.floor(float(row["xmin"]))
            ymin = math.floor(float(row["ymin"]))
            xmax = math.ceil(float(row["xmax"]))
            ymax = math.ceil(float(row["ymax"]))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, -1)
        elif class_name == 'BT_RasterPicture' or class_name == 'BT_VectorPicture':
            color = [255, 0, 0]
            xmin = math.floor(float(row["xmin"]))
            ymin = math.floor(float(row["ymin"]))
            xmax = math.ceil(float(row["xmax"]))
            ymax = math.ceil(float(row["ymax"]))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, -1)
        cv2.imwrite(masked_img_path, image)
    print(classes)

def process_records_dilated(input_path):
    records = pd.read_csv(os.path.join(input_path, 'train.csv'))
    classes = records['class'].unique().tolist()
    print(classes)
    
    masked_directory = os.path.join(input_path, 'dilated')
    if os.path.exists(masked_directory):
        shutil.rmtree(masked_directory)
    os.makedirs(masked_directory)

    grouped = records.groupby('filename')
    for key, df_group in grouped:
        image_path = os.path.join(input_path, key.strip())
        if not os.path.isfile(image_path):
            continue
        filename = os.path.basename(image_path)
        masked_img_path = os.path.join(masked_directory, filename)
        
        roi_list = []
        for row_index, row in df_group.iterrows():
            class_name = row["class"].strip()
            if class_name == 'BT_Text':
                xmin = math.floor(float(row["xmin"]))
                ymin = math.floor(float(row["ymin"]))
                xmax = math.ceil(float(row["xmax"]))
                ymax = math.ceil(float(row["ymax"]))
                roi_list.append((xmin, ymin, xmax, ymax))

        input_image = cv2.imread(image_path)
        height, width, _ = input_image.shape
        temp = np.zeros((height, width, 1), dtype=np.uint8)

        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        for (xmin, ymin, xmax, ymax) in roi_list:
            cv2.rectangle(temp, (xmin, ymin), (xmax, ymax), 255, -1)

            roi = thresh[ymin:ymax, xmin:xmax]
            roi[:] = cv2.dilate(roi, kernel, iterations=5)
        
        masked_image = cv2.bitwise_and(thresh, temp)
        cv2.imwrite(masked_img_path, masked_image)

input_path = r'./Data/'
process_records_new(input_path)
