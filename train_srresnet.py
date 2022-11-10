import os
import torch
from itertools import islice
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid

def train_srresnet(
        epochs, 
        dataloader, 
        model, 
        criterion, 
        optimizer,
        device
    ):
   
    for epoch in tqdm(range(epochs)):
        model.train()      

        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)

            loss = criterion(sr_imgs, hr_imgs)  

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
          
        if epoch == epochs - 1:
            save_samples(model, epoch+1, lr_imgs, show=True)

        torch.save({'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            'checkpoint_srresnet.pth.tar')

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
    save_image(fake_images, fake_fname, nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))