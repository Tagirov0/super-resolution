import os
import gdown
import torch
import srgan_config
import torch.nn as nn
from itertools import islice
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
from dataset import MyImageFolder
from vgg_features import TruncatedVGG19
from model import Generator, Discriminator
from torch.utils.data import Dataset, DataLoader

def train(
        epochs,
        train_loader,
        generator,
        discriminator,
        truncated_vgg19,
        content_loss_criterion,
        adversarial_loss_criterion,
        optimizer_g,
        optimizer_d,
        device
):
    for epoch in tqdm(range(epochs)):

        generator.train()
        discriminator.train()

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = generator(lr_imgs)

            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

            sr_discriminated = discriminator(sr_imgs)

            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
            perceptual_loss = content_loss + 1e-3 * adversarial_loss

            optimizer_g.zero_grad()
            perceptual_loss.backward()

            optimizer_g.step()

            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())

            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                               adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

            optimizer_d.zero_grad()
            adversarial_loss.backward()

            optimizer_d.step()

        if epoch == epochs - 1:
            save_samples(generator, epoch+1, lr_imgs, show=True)

        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict()},
            'checkpoint_srgan_celeba.pth.tar'
        )

def show_images(images, nmax=16):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images.detach()[:nmax], nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=16):
    for images, _ in dl:
        show_images(images, nmax)
        break

def save_samples(gen, index, latent_tensors, show=True):
    fake_images = gen(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, fake_fname, nrow=4)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=4).permute(1, 2, 0))

def main():
    data = MyImageFolder(
        srgan_config.train_dataset_path, 
        resolution=srgan_config.image_size
    )
    dataloader = DataLoader(data, batch_size=srgan_config.batch_size)
    
    discriminator = Discriminator(
        srgan_config.image_size, 
        srgan_config.image_size
    ).to(srgan_config.device)
    generator = Generator().to(srgan_config.device)

    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    content_loss_criterion = content_loss_criterion.to(srgan_config.device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(srgan_config.device)

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=srgan_config.lr, 
        betas=srgan_config.model_betas
    )
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=srgan_config.lr, 
        betas=srgan_config.model_betas
    )

    if srgan_config.pretrained_weights:
        if not os.path.exists(srgan_config.pretrained_weights_path):
            os.mkdir('weights')
            output = srgan_config.pretrained_weights_path
            gdown.download(url=srgan_config.weights_url, output=output, quiet=False, fuzzy=True)
        
        checkpoint = torch.load(srgan_config.pretrained_weights_path)
        generator.load_state_dict(checkpoint['generator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])

        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])

    vgg19_i = 5
    vgg19_j = 4

    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()
    truncated_vgg19 = truncated_vgg19.to(srgan_config.device)

    train(
        srgan_config.epochs, 
        dataloader, 
        generator, 
        discriminator, 
        truncated_vgg19,
        content_loss_criterion, 
        adversarial_loss_criterion, 
        optimizer_g,
        optimizer_d, 
        srgan_config.device
    )

if __name__ == "__main__":
    main()

