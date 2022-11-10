import torch

image_size = 96
batch_size = 64

epochs = 100
lr = 0.0001
model_betas = (0.9, 0.999)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset_path = 'img_align_celeba'
pretrained_weights = False
weights_url = 'https://drive.google.com/file/d/15sj6S-FtNjXgevGf2EU4yKChviU2YopA/view?usp=share_link'
pretrained_weights_path = 'weights/checkpoint_srgan_celeba.pth.tar'