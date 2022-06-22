import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def scan_dir(path):
    results = []
    for root, _, files in os.walk(path):
        for filename in files:
            results.append(os.path.join(root, filename))
    return results


def model_predict(model, spectrogram):
    with torch.no_grad():
        spectrogram = np.array([np.stack([spectrogram.T])])
        prediction = model(torch.from_numpy(spectrogram).float())
        result = prediction.numpy()[0][0].T
        return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=os.path.join('models', 'denoiser.pt'), help='model path')
    parser.add_argument('--data', type=str, default=os.path.join('..', 'data', 'test'), help='data path')
    parser.add_argument('--results', type=str, default='results', help='results path')
    return parser.parse_args()


def main(arguments):

    model = torch.jit.load(arguments.model_path)
    model.eval()

    data = pd.DataFrame(columns=['path'])
    data['path'] = scan_dir(arguments.data)

    for npy in tqdm(data['path'], 'Denoising'):
        new_npy = os.path.join(arguments.results, npy.split(os.sep)[-1])
        spectrogram = np.load(npy)
        result = model_predict(model, spectrogram)
        np.save(new_npy, result)


if __name__ == '__main__':
    args = get_args()
    main(args)
