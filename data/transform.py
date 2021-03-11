import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_yolo_transform(img_size, mode='train'):
    if mode == 'train':
        scale = 1.1
        transform = A.Compose([
            A.LongestMaxSize(max_size=int(img_size*scale)),
            A.PadIfNeeded(
                min_height=int(img_size*scale),
                min_width=int(img_size*scale),
                border_mode=cv2.BORDER_CONSTANT),
            A.RandomCrop(width=img_size, height=img_size),
            A.ColorJitter(
                brightness=0.6,
                contrast=0.6,
                saturation=0.6,
                hue=0.6,
                p=0.4),
            A.OneOf([
                A.ShiftScaleRotate(
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.4),
                A.IAAAffine(
                    shear=10,
                    mode='constant',
                    p=0.4)
                ], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255),
            ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[]))

    elif mode == 'test':
        transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))
    else:
        raise ValueError("'mode' can only accept 'train' or 'test'")

    return transform
