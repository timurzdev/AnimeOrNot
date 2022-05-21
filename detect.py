import torch
import argparse
from model import get_base_model
from torchvision import transforms as T
import PIL
import timeit

transform = T.Compose(
    [
        T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

labels = {1: 'human', 0: 'anime'}


def predict(model: torch.nn.Module, image_path: str) -> str:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    try:
        image = PIL.Image.open(image_path)
    except FileNotFoundError:
        return "Error opening image"
    image = transform(image).to(device)
    output = torch.sigmoid(model(image.unsqueeze(0)))
    return '{}'.format(labels[output.argmax(dim=1).item()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/model_9999.pt')
    parser.add_argument('--image_path', type=str, default='./data/anime/1.jpg')
    args = parser.parse_args()
    model = get_base_model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    label = predict(model, args.image_path)
    print(label)
