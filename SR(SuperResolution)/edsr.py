from model import resolve_single
from utils import load_image, plot_sample
import os


from data import DIV2K
from model.edsr import edsr
from train import EdsrTrainer


import glob
import numpy as np
from PIL import Image
# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

# Location of model weights (needed for demo)
weights_dir = f'weights/edsr-{depth}-x{scale}'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)
    sr = resolve_single(model, lr)
    return sr
model = edsr(scale=scale, num_res_blocks=depth)
model.load_weights('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/superresolution/super-resolution/weights/edsr-16-x4/weights/edsr-16-x4/weights.h5')
path ='/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/superresolution/super-resolution/CIFAR-10-images/test'
dst = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/superresolution/super-resolution/augmented_data/x4/test'
list_dir = os.listdir(path)
print(len(list_dir))
all_train = []
for dir in list_dir:
    image_list = glob.glob(path +'/'+ dir+'/*.jpg')
    for image in image_list:
        all_train.append(image)
        
print(len(all_train))    
print(all_train[0])    
for image in all_train:
    file_path = dst+'/edsr/'
    category = image.split('/')[9]
    name = image.split('/')[10]
    sr = resolve_and_plot(image)
    sr = np.array(sr)
    
    img = Image.fromarray(sr) # NumPy array to PIL image
    img.save(file_path+category+'/'+name)