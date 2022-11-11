import torch

image_size = 96
batch_size = 64

epochs = 100
lr = 0.0001
model_betas = (0.9, 0.999)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset_path = 'img_align_celeba'
weights_url = 'https://drive.google.com/file/d/15sj6S-FtNjXgevGf2EU4yKChviU2YopA/view?usp=share_link'
generator_weight_url = 'https://drive.google.com/file/d/1F8FmHVCQFxMkX3wJdRyAWaFtur0QH0Xi/view?usp=sharing'
pretrained_weights = False
generator_weights_path = 'weights/checkpoint_srgan.pth'
pretrained_weights_path = 'weights/checkpoint_srgan.pth.tar'