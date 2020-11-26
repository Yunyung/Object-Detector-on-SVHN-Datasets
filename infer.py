from mmdet.apis import init_detector, inference_detector
import numpy as np
import mmcv
import os
from tqdm import tqdm
import json

CLASS_NUM = 10 # digit labels 1~10

test_dataset_path = "dataset/test"
class Box():
    def __init__(self, y_min, x_min, y_max, x_max, confidence, label):
        # bbox coordinate
        self.four_coordinate = [y_min, x_min, y_max, x_max] 
        # confidence probability
        self.confidence = confidence
        # predicted label
        self.label = label

def write_json(data, out_filename):
    """
    args:
        data : data that format meet the required format in testing 
        out_filename: output file name

        Write result to .json file
    """

    with open(out_filename, 'w') as fp:
        json.dump(data, fp)

    print("Sucessfully write json file")
    pass

def get_all_imgs_filename():
    fileNames = os.listdir(test_dataset_path)
    fileNames = sorted(fileNames, key= lambda x: int(x[:-4])) # oorted by filename
    # sort fileName
    
    
    return fileNames

def parse_and_store_detection_result(data, result, threshold=0.5):
    """
        parse detection result and store it
    """
    #print(f'data status:{data}')
    #print(f'result:{result}')

    # data info in a image that format meet the required format in testing 
    one_image_data = {} 
    one_image_data["bbox"] = []
    one_image_data["score"] = []
    one_image_data["label"] = []

    for label_idx in range(CLASS_NUM):
        #print(f'label_idx={label_idx}, len:{len(result[label_idx])}')
        result[label_idx] = sorted(result[label_idx], key=lambda x:x[4], reverse=True)
        for row_idx in range(len(result[label_idx])):
            # if there are object detected for this label, it will enter this loop
            # result[label_dx] contain arbitrary # of rows, 
            # each row have 5 element -> [x_min, y_min, x_max, y_max, confidence]
            # Note: 1.For meet the testing requirement that digit label 1 ~ 10, so predicted label should also be 1 ~ 10.
            # 2. Bounding box sholud be [y_min, x_min, y_max, x_max] and integer type
            row = result[label_idx][row_idx]
            
            if (row[4] > threshold):
                row = [round(row[0]), round(row[1]), round(row[2]), round(row[3]), row[4].item()]
                box = Box(row[1], row[0], row[3], row[2], row[4], label_idx + 1) # used to store infomation of the detected box 
                one_image_data["bbox"].append(box.four_coordinate)
                one_image_data["score"].append(box.confidence)
                one_image_data["label"].append(box.label)
                #print(row)

    data.append(one_image_data)
    #print(f'data status:{data}')
    return data

# Specify the path to model config and checkpoint file
# config_file value can be one of [mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco_copy.py, mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_VOC.py, mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco_copy.py]
config_file = 'mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco_copy.py'
# modify checkpoint_file value to your checkpoint path 'checkpoints/{faster_rcnn.pth, cascade_rcnn.pth, YOLOv3.pth}'
checkpoint_file = 'checkpoints/cascade_rcnn.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# data that format meet the required format in testing 
data = []

# get all img's filename in testing set
img_fileNames = get_all_imgs_filename()   
count = 0
print("Start inference process...")
for img_fn in tqdm(img_fileNames, mininterval=3):
    # object detection - infer test img 
    img_path = os.path.join(test_dataset_path, img_fn)
    result = inference_detector(model, img_path)
    data = parse_and_store_detection_result(data, result, threshold=0) 
    count += 1
    # if (count == 100)
    #     print(data)

# write data to json file
out_filename = "infer_output.json"
write_json(data, out_filename)

print("End inference.")
print(data)







