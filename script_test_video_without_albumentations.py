import cv2
import torch
import segmentation_models_pytorch as smp

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = './data/CamVid/'
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

# load best saved checkpoint
best_model = torch.load('./models/model_car.pth')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# define a video capture object
vid = cv2.VideoCapture('data/video.mp4')

# Check if camera opened successfully
if (vid.isOpened()== False): 
    print("Error opening video stream or file")

while True:
    ret, webcam = vid.read()
    if ret == True:

        image = cv2.resize(webcam, (480,384))
        image_vis = image # for visualization
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocessing_fn(image)

        image = image.transpose(2, 0, 1).astype('float32')

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