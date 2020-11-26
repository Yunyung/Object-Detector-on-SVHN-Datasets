# Homework2 README.md

###### `tags: Selected Topics in Visual Recognition using Deep Learning`

This project is part of a series of projects for the course *Selected Topics in Visual Recognition using Deep Learning*. This Repository gathers the code for digit object detector.

In this project we use [mmdetection](https://github.com/open-mmlab/mmdetection), an open source object detection toolbox based on PyTorch, to train our model and conquer this task. See report.pdf for the report containing the representation and the analysis of the produced results.

## Environment
- Platform: Ubuntu
- Package: mmdetection
## Reproducing submissoin
Because we use [mmdetectoin](https://github.com/open-mmlab/mmdetection) as our project's backbone, there are some of tricks needed to do for training and testing on our custom dataset. That is, a lots of preparation and modification required to be done.

To reproduct my submission without retrainig, do the following steps:

1. [Installation](#Installation)
2. [Inference](#Inference)

## Installation
In this project, we use [mmdetectoin](https://github.com/open-mmlab/mmdetection) as our backbone, so the related package should be installed. There provided quick installed instructoin below. See [mmdetectoin official installation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) if you encounter any problems. 

```
#Assuming that you already have CUDA installed, here is a full script for setting up MMDetection with conda.
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch torchvision cudatoolkit -c pytorch -y

# install the latest mmcv
pip install mmcv-full

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e 

pip install tqdm
```
## Download Official Dataset
#### Dataset used for the course are a liitle bit different. Just provide the link below to official dataset
- [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)

## Project Structure
```
Root/
    infer.py # infer your testing data
    SpeedBenchmark.ipynb # Test model's inference speed
    parse_matFile.py # used to parse data in 'digitStruct.mat' file to .xml format(VOC format)
    override_setting # override the setting in mmdetection
    ├ faster_rcnn_r50_fpn_1x_VOC.py
    ├ ....
    checkpoints
    ├ cascade_rcnn.pth
    ├ faster_rcnn.pth
    ├ YOLOv3.pth
    TestDataset # testing dataset you want to infer
    ├ 1.png
    ├ 2.png
    ├ ...
    mmdetection
    ├── mmdet
    ├── tools
    ├── configs
    ├── data # data folder that you sould create manually
    │   ├── VOCdevkit
    │   │   ├── VOC2007
    │   │   │    ├── Annotations # Put your all 
    │   │   │    │   ├ 1.xml
    │   │   │    │   ├ 2.xml
    │   │   │    │   ├ ...
    │   │   │    ├── JPEGImages # All Image
    │   │   │    │   ├ 1.png
    │   │   │    │   ├ 2.png
    │   │   │    │   ├ ...
    │   │   │    ├── ImageSets # *.txt contain file's name (no file extension)
    │   │   │    │   ├── Main
    │   │   │    │   │   ├ train.txt
    │   │   │    │   │   ├ val.txt
```

## Dataset Preparation
- You can use ```parse_matFile.py``` to parse 'digitStruct.mat' file to .xml format(VOC format), or searching other tools that are also avaliable on the Internet 

- Use following instructoin to override setting in mmdetection for training and testing our models.
```
cd override_setting

mv cascade_rcnn_r50_fpn.py ../mmdetection/configs/_base_/models

mv cascade_rcnn_r101_caffe_fpn_1x_coco_copy.py ../mmdetection/configs/cascade_rcnn
mv faster_rcnn_r50_fpn_1x_VOC.py ../mmdetection/configs/faster_rcnn
mv yolov3_d53_mstrain-416_273e_coco_copy.py ../mmdetection/configs/yolo
mv yolov3_d53_mstrain-608_273e_coco_copy.py ../mmdetection/configs/yolo

mv xml_style.py ../mmdetection/mmdet/datasets
mv class_names.py ../mmdetection/mmdet/core/evaluation
```



## Training
- If all the training data are ready, there are there three different models we can use.  
```
cd mmdetection

# train YOLOv3
python tools/train.py configs/yolo/yolov3_d53_mstrain-416_273e_coco_copy.py  

# train Faster R-CNN
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_VOC.py

# train Cascade R-CNN
python tools/train.py configs/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco_copy.py
```

- Modify img scale to meet you custom dataset if needed in ```mmdetection/configs/_base_/datasets/voc0712.py```

## Inference
There are three trained model in ```checkpoints``` folder, so modify ```checkpoint_file``` and ```config_file``` path variables in 
```infer.py```.

```
python infer.py # Default using Cascade R-CNN to infer testing data
```

The result will save in as ```infer_output.json``` in root

## Result
| Models | YOLOv3 | Faster-RCNN | Cascade R-CNN |
| ------ | ------ | ----------- | ------------- |
| mAP    | 0.390  | 0.402       | 0.419         |
| Speed(ms)|20.6   | 59.4   | 94.1        | 

The three object detector shown above are measured by mAP score and Speed. The speed tested in ```SpeedBenchmark.ipynb``` file







