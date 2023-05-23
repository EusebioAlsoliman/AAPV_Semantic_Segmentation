
import albumentations as albu
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = './data/CamVid/'

# load best saved checkpoint
best_model = torch.load('./models/model_car.pth')


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir,  
            preprocessing=None,
    ):
        self.img = images_dir
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        # image = cv2.imread(self.img)
        image = cv2.resize(self.img, (480,384))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image


# define a video capture object
vid = cv2.VideoCapture('data/video.mp4')

# Check if camera opened successfully
if (vid.isOpened()== False): 
  print("Error opening video stream or file")

while(vid.isOpened()):
    ret, webcam = vid.read()
    if ret == True:

        dataset = Dataset(webcam)
        dataset_inf = Dataset(webcam, preprocessing=get_preprocessing(preprocessing_fn))

        image_vis = dataset[0] # get some sample
        image = dataset_inf[0]

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        cv2.imshow('Imagen', image_vis)
        cv2.imshow('Inferencia', pr_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()