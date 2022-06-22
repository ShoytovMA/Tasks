import os
import re
import time
import copy
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrainValDataset(Dataset):

    def __init__(self, path, max_len=8):
        self.path = path
        self.max_len = max_len
        self.data = pd.DataFrame(columns=['path_clean', 'path_noisy'])
        for path_noisy in tqdm(scan_dir(self.path), 'Loading dataset'):
            path_clean = re.sub('noisy', 'clean', path_noisy)
            if os.path.exists(path_clean):
                self.data = pd.concat([self.data,
                                       pd.DataFrame({'path_clean': path_clean, 'path_noisy': path_noisy}, index=[0])],
                                      ignore_index=True)
        self.data = self.data.sample(frac=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        path_clean, path_noisy = self.data.iloc[idx]
        clean = np.load(path_clean)
        noisy = np.load(path_noisy)
        start = np.random.randint(0, clean.shape[0] - self.max_len, 1)[0]
        clean = clean[start:start + self.max_len]
        noisy = noisy[start:start + self.max_len]
        clean = np.stack([clean.T])
        noisy = np.stack([noisy.T])
        return torch.from_numpy(noisy).float(), torch.from_numpy(clean).float()


class Denoiser(nn.Module):

    def __init__(self):
        super(Denoiser, self).__init__()
        nf = 16
        self.b1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, nf * 2, (9, 8), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b2 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b3 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b4 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Conv2d(nf, nf * 2, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.Conv2d(nf * 2, nf * 4, (5, 1), (1, 1), (2, 0), bias=False),
        )
        self.b5 = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.Conv2d(nf * 4, nf, (9, 1), (1, 1), (4, 0), bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(nf),
            torch.nn.Dropout2d(0.2, True),
            torch.nn.Conv2d(nf, 1, (1, 1), (1, 1), (0, 0), bias=True),
        )


    def forward(self, input):
        b1 = self.b1(input)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3+b2)
        b5 = self.b5(b4+b1)

        return b5


def scan_dir(path):
    path = os.path.join(path, 'noisy')
    results = []
    for root, _, files in os.walk(path):
        for filename in files:
            results.append(os.path.join(root, filename))
    return results


def get_args():
    parser = argparse.ArgumentParser()
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
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
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
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model


def sdr_loss(pred, label):
    return -(torch.sum(label**2)/torch.sum((pred-label)**2))


def main(arguments):

    train_ds = TrainValDataset(arguments.train_data)
    val_ds = TrainValDataset(arguments.val_data)
    train_val_dl = {'train': DataLoader(train_ds, batch_size=arguments.batch_size, shuffle=True),
                    'val': DataLoader(val_ds, batch_size=arguments.batch_size, shuffle=True)}

    model = Denoiser()

    model = train_model(model, train_val_dl, sdr_loss,
                        torch.optim.Adam(model.parameters()), arguments.epochs)
    model_scripted = torch.jit.script(model)
    model_scripted.save(arguments.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
