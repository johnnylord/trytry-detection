# TryTry-Detection

This repo implements YOLOv3 from scratch with modified codes from [aladdinpersson](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3).

## YOLOv3 Model Prediction
By feeding an image into `YOLOv3`, it will generate dense prediction of objects for each scale. For each cell in each scale, there are three anchors responsible for detecting objects, and each anchor will predict the `(prob, x, y, w, h, classes)`.
> The output of YOLOv3 should be further processed as following:
> prob => sigmoid(prob)
> x\_offset => sigmoid(x)
> y\_offset => sigmoid(y)
> w\_cell => anchor\_width * torch.exp(w)
> h\_cell => anchor\_height * torch.exp(h)
> class\_idx => torch.argmax(classes)[0]
```python
from model.yolov3 import YOLOv3

in_channels = 3
num_classes = 20

model = YOLOv3(in_channels=in_channels,
                num_classes=num_classes)

imgs = torch.randn((1, 3, 416, 416))
outs = model(imgs)

print("Scale(13):", outs[0].shape) # (1, 3, 13, 13, 25)
print("Scale(26):", outs[1].shape) # (1, 3, 26, 26, 25)
print("Scale(52):", outs[2].shape) # (1, 3, 52, 52, 25)
```

## YOLOv3 Dataset Groundtruth
The groundtruth label for each image is the position information of each object in the image. Specifically, there are three kinds of target labels representing ground truth object position in each scale. The label for each grid cell is `(prob, x_offset, y_offset, w_cell, h_cell, class_idx)`.
```python
from data.dataset import YOLODataset
from data.transform import get_yolo_transform

# Hyperparameters
CSV_PATH = 'download/PASCAL_VOC/2examples.csv'
IMG_DIR = 'download/PASCAL_VOC/images/'
LABEL_DIR = 'download/PASCAL_VOC/labels/'
IMG_SIZE = 416
SCALES = [13, 26, 52]
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] # (3, 3, 2)

# Load Dataset
transform = get_yolo_transform(img_size=IMG_SIZE, mode='test')
dataset = YOLODataset(csv_file=CSV_PATH,
                    img_dir=IMG_DIR,
                    label_dir=LABEL_DIR,
                    anchors=ANCHORS,
                    transform=transform)
# Peek One Sample
img, targets = dataset[0]

print("Img shape:", img.shape) # (3, 416, 416)
print("Number of targets:", len(targets)) # 3
print("T1 shape:", targets[0].shape) # (3, 13, 13, 6)
print("T2 shape:", targets[1].shape) # (3, 26, 26, 6)
print("T3 shape:", targets[2].shape) # (3, 52, 52, 6)
```

## YOLOv3 Loss Calculation
```python

```

## Training Procedure
### 1. Train darknet53 on jmageNet dataset
### 2. Train YOLOv3 with pretrained darknet53 backbone
