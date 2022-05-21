import torch.nn as nn
from torchvision.models import resnet50
from torchsummary import summary
import torch


def get_base_model():
    resnet = resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(num_ftrs, 2)
    return resnet


def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    model = get_base_model()
    model = model.cuda()
    summary(model, (3, 224, 224))
    save_model(model, "./resnet50_model.pt")
    print(type(model))
