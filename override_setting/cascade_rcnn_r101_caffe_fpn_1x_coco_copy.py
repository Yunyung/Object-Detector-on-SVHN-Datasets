_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    #'../_base_/datasets/coco_detection.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101)
    )

classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

data = dict(
    train=dict(
        dataset=dict(
            classes=classes,
            ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
            img_prefix= data_root + 'VOC2007/'
        )
    )

)

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) 

load_from = 'checkpoints/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.423_20200504_175649-cab8dbd5.pth'

resume_from = 'work_dirs/cascade_rcnn_r101_caffe_fpn_1x_coco_copy/epoch_9.pth' # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
