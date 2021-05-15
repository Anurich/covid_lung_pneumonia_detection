import torch
import torch.nn as nn
import torchvision.models as model
import torch.nn.functional as F
from einops import rearrange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, no_class):
        super().__init__()
        vgg = model.vgg16(pretrained=True)
        in_features = vgg.classifier[0].in_features
        vgg = vgg.eval()
        self.vgg_pretrained = nn.Sequential(*list(vgg.children())[:-1])
        # now we gonna make the layers parameters to false
        for param in self.vgg_pretrained.parameters():
            param.requires_grad = False
        # now we can add few more layers where
        self.linear = nn.Linear(in_features, 64)
        #self.linear1 = nn.Linear(64, 64)
        self.output = nn.Linear(64, no_class)

    def forward(self,x):
        vgg_features  = self.vgg_pretrained(x)
        # we can now pass these through dense layer
        flatten  = rearrange(vgg_features,'b c h w -> b (c h w)')
        output   = F.relu(self.linear(flatten))
        #output   = F.relu(self.linear1(output))
        output   = self.output(output)
        return output

