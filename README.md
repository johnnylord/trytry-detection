# trytry-detection

This repo implements yolov3 and yolact with darknet53 backbone from scratch with pytorch.

## Pretrained Result
| Model           | Dataset        | mAP@0.5 | Model Checkpoint                                                               |
|-----------------|----------------|---------|--------------------------------------------------------------------------------|
| YOLOv3\_416x416 | PASCAL VOC2012 | 0.74    | [Download](https://www.dropbox.com/s/7gtcjbxnk012k3h/yolov3_voc2012.pth?dl=0)  |
| YOLOv3\_416x416 | MS COCO2014    | 0.47    | [Download](https://www.dropbox.com/s/rdaztvk7ap46f1i/yolov3_coco2014.pth?dl=0) |
| Maskv3\_416x416 | MS COCO2014    | 0.51    | [Download](https://www.dropbox.com/s/dy1weqzdr3rsl1h/maskv3_coco2014.pth?dl=0) |

## VOC2012 Visualization of Result
![VOC2012](https://i.imgur.com/EbYx6bU.png)

## COCO2104 Visualization of Result
![COCO2014](https://i.imgur.com/rijFc8r.png)

## Maskv3 with YOLACT Concept
![COCO2014Mask](https://i.imgur.com/JXcIKH8.jpg)

## How to train the model

### Download dataset
- [voc2012 dataset](https://www.dropbox.com/s/wpo5ht4rnphsn8k/voc2012.tar.gz?dl=0)
- [coco2014 dataset](https://www.dropbox.com/s/ct43kswckwpafpq/coco2014.tar.gz?dl=0)
```bash
$ cd download
$ wget -O voc2012.zip https://www.dropbox.com/s/wpo5ht4rnphsn8k/voc2012.tar.gz?dl=1
$ wget -O coco2014.zip https://www.dropbox.com/s/ct43kswckwpafpq/coco2014.tar.gz?dl=1
$ tar -xzvf voc2012.zip && rm voc2012.zip
$ tar -xzvf coco2014.zip && rm coco2014.zip
```

### Directory Structure of Dataset
Download the dataset and unzip them under `download` directory with the following structure.
```bash
$ tree -L download
download
├── COCO
│   ├── images
│   ├── labels
│   ├── masks
│   ├── test.csv
│   └── train.csv
└── PASCAL_VOC
    ├── 100examples.csv
    ├── 1examples.csv
    ├── 2examples.csv
    ├── 8examples.csv
    ├── images
    ├── labels
    ├── test.csv
    └── train.csv

6 directories, 8 files
```

### Train your model
```bash
$ pip install -r requrements.txt
$ python main.py --config config/yolov3_voc.yml
$ python main.py --config config/yolov3_coco.yml
```

## Development Note
### YOLOv3 model prediction
By feeding an image into `YOLOv3`, it will generate dense prediction of objects for each scale. For each cell in each scale, there are three anchors responsible for detecting objects, and each anchor will predict the `(x_raw, y_raw, w_raw, h_raw, prob_raw, classes_raw)`.
> the output of YOLOv3 should be further processed as following:  
> x\_offset => sigmoid(x\_raw)  
> y\_offset => sigmoid(y\_raw)  
> w\_cell => anchor\_w * torch.exp(w\_raw)  
> h\_cell => anchor\_h * torch.exp(h\_raw)  
> prob => sigmoid(prob\_raw)  
> class => torch.argmax(classes\_raw)[0]
```python
from model.yolov3 import YOLOv3

in_channels = 3
num_classes = 20

model = YOLOv3(in_channels=in_channels,
                num_classes=num_classes)

imgs = torch.randn((1, 3, 416, 416))
outs = model(imgs)

print("scale(13):", outs[0].shape) # (1, 3, 13, 13, 25)
print("scale(26):", outs[1].shape) # (1, 3, 26, 26, 25)
print("scale(52):", outs[2].shape) # (1, 3, 52, 52, 25)
```

### YOLOv3 dataset groundtruth
The groundtruth label for each image is the position information of each object in the image. Specifically, there are three kinds of target labels representing ground truth object position in each scale. the label for each grid cell is `(x_offset, y_offset, w_cell, h_cell, prob, class)`.
```python
from data.dataset import YOLODataset
from data.transform import get_yolo_transform

# hyperparameters
csv_path = 'download/pascal_voc/test.csv'
img_dir = 'download/pascal_voc/images/'
label_dir = 'download/pascal_voc/labels/'
img_size = 416
scales = [13, 26, 52]
anchors = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # scale 13
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # scale 26
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], # scale 52
] # (3, 3, 2)

# load dataset
transform = get_yolo_transform(img_size=img_size, mode='test')
dataset = YOLODataset(csv_file=csv_path,
                    img_dir=img_dir,
                    label_dir=label_dir,
                    anchors=anchors,
                    transform=transform)
# peek one sample
img, targets = dataset[0]

print("img shape:", img.shape) # (3, 416, 416)
print("number of targets:", len(targets)) # 3
print("t1 shape:", targets[0].shape) # (3, 13, 13, 6)
print("t2 shape:", targets[1].shape) # (3, 26, 26, 6)
print("t3 shape:", targets[2].shape) # (3, 52, 52, 6)
```
