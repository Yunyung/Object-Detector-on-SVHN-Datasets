# Homework2
###### `tags: Selected Topics in Visual Recognition using Deep Learning`

Jhen-Yung Hsu, 0856167

Project repositoy: [https://github.com/Yunyung/Object-detector-on-SVHN-datasets](https://github.com/Yunyung/Object-detector-on-SVHN-datasets)

---
## Environment
Framework: PyTorch, mmdetection

Platform: Ubuntu, Google Colab (For Speech benchmark)

## Speed benchmark
- YOLOv3 - 20.6ms
![](https://i.imgur.com/Py71aIv.jpg)
- Faster R-CNN - 59.4ms
![](https://i.imgur.com/Aiu063z.jpg)
- Cascade R-CNN - 94.1ms
![](https://i.imgur.com/oIUUGZo.jpg)

## Introduction

&nbsp;&nbsp;&nbsp;&nbsp;In this project, we use [mmdetection [1]](https://github.com/open-mmlab/mmdetection) to do object detection. The state-of-the-arts models such as YOLOv3, Faster R-CNN, and [Cascade R-CNN[2]](https://arxiv.org/abs/1712.00726) are used to conquer this contest. We also do the experiment for this three model included mAP score and inference speed.

## Methodology

### Data preprocessing
&nbsp;&nbsp;&nbsp;&nbsp;To train our model on [mmdetection](https://github.com/open-mmlab/mmdetection), we need to transform our data to COCO format or VOC format.
### Transfer Learning
&nbsp;&nbsp;&nbsp;&nbsp;Transfer learning is a important factor in traiaing process. In this task, we load pretrained weights from [mmdetection](https://github.com/open-mmlab/mmdetection) and train our three models.

### Model architecture
&nbsp;&nbsp;&nbsp;&nbsp;In this object detection task, we try three  models, yoloV3, faster-RCNN, and cascade rcnn. They have different mAP, inference speed individually. The experiments are shown belowed:



| Models | YOLOv3 | Faster-RCNN | Cascade R-CNN |
| ------ | ------ | ----------- | ------------- |
| mAP    | 0.390  | 0.402       | 0.419         |
| Speed(ms)|20.6   | 59.4   | 94.1        | 
 
&nbsp;&nbsp;&nbsp;&nbsp;In the experiment above, Cascade R-CNN acheive highest mAP score but lowest Speed per image, in contrast, YOLOv3 has lowest mAP but highest speed.

### Findings or Summary
&nbsp;&nbsp;&nbsp;&nbsp;Object detection is a classical problem in computer vision and machine learning. It's my first time dealing with object detectoin task, which is quite interesting and difficult task. To acheive as high mAP as possible, i also try other model and tool like [EffientDet[4]](https://arxiv.org/abs/1911.09070) and YOLOv4. Because some training and data preprocessing problem, it's not successful in my experiment. 

### Hyperparameters
- **Epoch** - 12~20
- **Optimizer** - SGD, learning rate = 0.001, momentum=0.9

### References
[1] [mmdetection](https://github.com/open-mmlab/mmdetection)

[2] [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726)

[3] [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)