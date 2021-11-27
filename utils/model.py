import torch
import torch.nn as nn
import torchvision.models as models
# import timm

class my_model(nn.Module):
    def __init__(self, args):
        super(my_model, self).__init__()
        # self.model = models.resnet50(num_classes=args.num_classes)
        self.model = eval("models.{}".format(args.model))(num_classes=args.num_classes) # a trick: use of eval()
        # self.model = timm.create_model(args.model, num_classes=args.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
