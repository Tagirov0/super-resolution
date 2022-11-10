import torch

imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(srgan_config.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(srgan_config.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

image_size = 96
batch_size = 64

epochs = 100
lr = 0.0001
model_betas = (0.9, 0.999)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset_path = 'img_align_celeba'
pretrained_weights = False
weights_url = 'https://drive.google.com/file/d/15sj6S-FtNjXgevGf2EU4yKChviU2YopA/view?usp=share_link'
pretrained_weights_path = 'checkpoint_srgan_celeba.pth.tar'