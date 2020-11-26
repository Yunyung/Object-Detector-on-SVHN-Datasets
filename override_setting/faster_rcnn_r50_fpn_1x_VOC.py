_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    #'../_base_/datasets/coco_detection.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)  
    )
)

data = dict(
    train=dict(
        dataset=dict(
            classes=classes,
            ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt',
            img_prefix=data_root + 'VOC2007/'
        )
    )
)


load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.