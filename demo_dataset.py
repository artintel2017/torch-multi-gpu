# demo dataset

# 测试数据集

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

pre_image_size = (256, 256)
image_size     = (224, 224)

data_transform = transforms.Compose([
    transforms.Resize(pre_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=25, translate=(.2, .2) , 
        scale=(0.8, 1.2), shear=8, 
        resample=Image.BILINEAR, fillcolor=0),
    transforms.RandomCrop(image_size, padding=2, fill=(0,0,0) ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_set = CIFAR10('downloaded_models', train=True,  transform=data_transform, download=True)
valid_set = CIFAR10('downloaded_models', train=False, transform=test_transform, download=True)