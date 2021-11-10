import torch
from torch import nn
from torchsummary import summary
from torchvision.models import mobilenet_v2

import numpy as np



class SenceMobilenetV2(nn.Module):

    def __init__(self, n_classes, pretrained=False):
        super(SenceMobilenetV2, self).__init__()

        self.model_ = mobilenet_v2(pretrained=pretrained)
        for param in self.model_.parameters():
            param.requires_grad = True #False
        self.ft_mobilenetv2 = nn.Sequential(
            nn.Linear(in_features=np.int(1000), out_features=n_classes, bias=True)
        )

    def forward(self, input):

        midlevel_features = self.model_(input)
        output = self.ft_mobilenetv2(midlevel_features)
        return output



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SenceMobilenetV2().to(device)
    # print(str(model))
    summary(model, (3, 240, 320))