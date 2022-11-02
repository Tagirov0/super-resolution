import os
import torch
from itertools import islice
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid

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

        for i, (lr_imgs, hr_imgs) in enumerate(islice(train_loader, 100, 200)):
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
            'checkpoint_srgan.pth.tar'
        )

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images.detach()[:nmax], nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def save_samples(gen, index, latent_tensors, show=True):
    fake_images = gen(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join('hr', fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))