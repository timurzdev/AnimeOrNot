import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from dataLoader import load_data
from model import get_base_model
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def check_accuracy(model, epoch, data_loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            predicted = torch.sigmoid(outputs)
            predicted = (predicted > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item() / 2
    print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.4f}%')
    torch.save(model.state_dict(), f'./model/model_{epoch}.pt')
    print(f'EPOCH: {epoch}" -- Accuracy ={correct}/{total} /{correct / total:.4f}')


def train(model, data_loaders, device):
    epochs = 1
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    data_loader = data_loaders['train']
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)
        model.train()
        for batch_id, data in enumerate(data_loader):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_id % 25 == 0:
                print(f'Epoch:{epoch}-{batch_id} - Loss: {loss.item():.4f}')
        check_accuracy(model, epoch, data_loaders['val'], device)
    print('Finished Training')
    print('-' * 20)
    print("TEST")
    check_accuracy(model, 9999, data_loaders['test'], device)


if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = get_base_model()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCELoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    data_loaders, dataset_sizes = load_data('./data/', config['batch_size'])
    train(model, data_loaders, device)
