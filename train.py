import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import numpy as np
import os
from dataLoader import load_data
from model import get_base_model


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader,
          optimizer: optim, epoch: int, device: torch.device):
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    train_loader, val_loader = load_data('./data/', config['batch_size'])
    # Get model
    model = get_base_model()
    # Train
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    for epoch in range(1, config['epochs'] + 1):
        train(model, train_loader, optimizer, epoch, device)
        if epoch % config['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join('./model', 'model_{}.pth'.format(epoch)))
    torch.save(model.state_dict(), os.path.join('./model', 'model_final.pth'))
