# trytry-detection

This repo implements yolov3 from scratch with modified codes from [aladdinpersson](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3).

## Pretrained Result
TODO

## How to train the model
### Download dataset
Thanks to [aladdinperson](https://github.com/aladdinpersson), we can directly download organzied dataset from his kaggle account.
- [pascal voc2012 dataset](https://www.kaggle.com/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video)
- [coco2014 dataset](https://www.kaggle.com/dataset/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e)

### Directory Structure of Dataset
Download the dataset and unzip them under `download` directory with the following structure.
```bash
$ tree -L download
download
├── COCO
│   ├── images
│   ├── labels
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
The groundtruth label for each image is the position information of each object in the image. Specifically, there are three kinds of target labels representing ground truth object position in each scale. the label for each grid cell is `(prob, x_offset, y_offset, w_cell, h_cell, class)`.
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

## Visualization of Result
- **The following diagram shows the overfitted prediction result of 9 voc image.**
![voc-prediction](https://i.imgur.com/yEUEfnP.png)

- **The following diagram shows the ground truth result of 9 voc image.**
![voc-groundtruth](https://i.imgur.com/CNk5zdR.png)
