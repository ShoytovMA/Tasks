import os
import time
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrainValDataset(Dataset):

    def __init__(self, path, max_len=50):
        self.path = path
        self.max_len = max_len
        self.data = pd.DataFrame(columns=['path', 'label'])
        for path in tqdm(scan_dir(self.path), 'Loading dataset'):
            self.data = pd.concat([self.data,
                                   pd.DataFrame({'path': path, 'label': path.split(os.sep)[-3]}, index=[0])],
                                  ignore_index=True)
        self.data = self.data.sample(frac=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        path, label = self.data.iloc[idx]
        spectrogram = np.load(path)
        start = np.random.randint(0, spectrogram.shape[0]-self.max_len, 1)[0]
        spectrogram = spectrogram[start:start+self.max_len]
        spectrogram = np.stack([spectrogram])
        label = np.array(0) if label == 'clean' else np.array(1)
        return torch.from_numpy(spectrogram).float(), torch.from_numpy(label).long()


class CustomModel(nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, (3, 3), padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))

        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, (3, 3), padding='same'),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))

        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, (3, 3), padding='same'),
                                   nn.ReLU(),
                                   nn.AdaptiveMaxPool2d((6, 10)))

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(1920, 960),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(960, 2),
                                        nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


def scan_dir(path):
    results = []
    for root, _, files in os.walk(path):
        for filename in files:
            results.append(os.path.join(root, filename))
    return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='MobileNet', help='"MobileNet" or "Custom"')
    parser.add_argument('--train-data', type=str, default=os.path.join('..', 'data', 'train'), help='train data path')
    parser.add_argument('--val-data', type=str, default=os.path.join('..', 'data', 'val'), help='validation data path')
    parser.add_argument('--epochs', type=int, default=25, help='epochs number')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--save-path', type=str, default=os.path.join('models', 'new_model.pt'), help='model save path')
    return parser.parse_args()


def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


def main(arguments):

    train_ds = TrainValDataset(arguments.train_data)
    val_ds = TrainValDataset(arguments.val_data)
    train_val_dl = {'train': DataLoader(train_ds, batch_size=arguments.batch_size, shuffle=True),
                    'val': DataLoader(val_ds, batch_size=arguments.batch_size, shuffle=True)}

    if arguments.model_type == 'MobileNet':
        model = torchvision.models.mobilenet_v3_small(pretrained=False).float()
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[3] = nn.Linear(in_features=1024, out_features=2, bias=True)
    elif arguments.model_type == 'Custom':
        model = CustomModel()
    else:
        raise Exception(f'Unknown model type: Expected "MobileNet" or "Custom" but got {arguments.model_type}.')

    model = train_model(model, train_val_dl, nn.CrossEntropyLoss(),
                        torch.optim.Adam(model.parameters()), arguments.epochs)
    model_scripted = torch.jit.script(model)
    model_scripted.save(arguments.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
