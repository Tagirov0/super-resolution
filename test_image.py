import os
import gdown
import torch
import argparse
from PIL import Image
from model import Generator
import srgan_config as config
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms.functional as FT

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--image_name', type=str, help='low resolution image name')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
opt = parser.parse_args()

if opt.test_mode == 'GPU' and torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def convert_image(img, source, target):
    imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(config.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(config.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source

    if source == 'pil':
        img = FT.to_tensor(img)
    elif source == '[0, 1]':
        pass
    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    if target == 'pil':
        img = FT.to_pil_image(img)
    elif target == '[0, 255]':
        img = 255. * img
    elif target == '[-1, 1]':
        img = 2. * img - 1.
    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda
    
    return img

def visualize(img, srgan_generator):
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)))
    
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height))

    sr_img_srgan = srgan_generator(
        convert_image(
            lr_img, 
            source='pil', 
            target='-norm'
        ).unsqueeze(0).to(DEVICE))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    return sr_img_srgan

def main():
    if not os.path.exists(config.generator_weights_path):
        if not os.path.exists('weights'):
            os.mkdir('weights')
        output = config.generator_weights_path
        gdown.download(url=config.generator_weight_url, output=output, quiet=False, fuzzy=True)

    generator = Generator().eval()

    if torch.cuda.is_available():  
        generator.load_state_dict(torch.load(config.generator_weights_path))
    else:
        generator.load_state_dict(torch.load(config.generator_weights_path, map_location='cpu'))

    sr_img = visualize(opt.image_name, generator)
    sr_img.save('sr_x4_' + opt.image_name)

if __name__ == "__main__":
    main()