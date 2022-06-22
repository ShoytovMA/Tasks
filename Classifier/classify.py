import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm


def scan_dir(path):
    results = []
    for root, _, files in os.walk(path):
        for filename in files:
            results.append(os.path.join(root, filename))
    return results


def model_predict(model, spectrogram):
    spectrogram = np.array([np.stack([spectrogram])])
    prediction = model(torch.from_numpy(spectrogram).float())
    label = torch.max(prediction, 1)[1].numpy()[0]
    return 'clean' if label == 0 else 'noisy'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=os.path.join('models', 'MobileNet.pt'), help='model path')
    parser.add_argument('--data', type=str, default=os.path.join('..', 'data', 'test'), help='data path')
    parser.add_argument('--results', type=str, default=os.path.join('predictions', 'new_preds.csv'), help='predictions path')
    return parser.parse_args()


def main(arguments):

    model = torch.jit.load(arguments.model_path)
    model.eval()

    if os.path.isdir(arguments.data):
        data = pd.DataFrame(columns=['path', 'label'])
        data['path'] = scan_dir(arguments.data)
    elif os.path.isfile(arguments.data):
        data = pd.DataFrame({'path': arguments.data, 'label': np.nan}, index=[0])
    else:
        raise FileNotFoundError

    preds = []
    for npy in tqdm(data['path'], 'Predicting'):
        spectrogram = np.load(npy)
        preds.append(model_predict(model, spectrogram))
    data['label'] = preds
    data.to_csv(arguments.results, index=False)


if __name__ == '__main__':
    args = get_args()
    main(args)
