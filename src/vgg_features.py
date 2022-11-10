import torch.nn as nn
from torchvision.models import vgg19

class TruncatedVGG19(nn.Module):

    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()
        vgg19_model = vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0

        for layer in vgg19_model.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            if maxpool_counter == i - 1 and conv_counter == j:
                break

        self.truncated_vgg19 = nn.Sequential(*list(vgg19_model.features.children())[:truncate_at + 1])

    def forward(self, input):
        output = self.truncated_vgg19(input)
        return output