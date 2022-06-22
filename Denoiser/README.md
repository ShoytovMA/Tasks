# Denoiser
**Избавляется от посторонних шумов в звуковом сигнале (mel-спектрограмме)**


## Usage

### Train

~~~~~~
python train.py [-h] [--train-data TRAIN_DATA] [--val-data VAL_DATA] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--save-path SAVE_PATH]
~~~~~~

| Param | Default               | Description          |
|-------|-----------------------|----------------------|
| --train-data | ../data/train/        | train data path      |
| --val-data | ../data/val/          | validation data path |
| --epochs | 25                    | epochs number        |
| --batch-size | 64                    | batch size     |
| --save-path | models/new_model.pt | model save path      |

### Denoise

~~~~~~
python  denoise.py [-h] [--model-path MODEL_PATH] [--data DATA] [--results RESULTS]
~~~~~~

| Param | Default            | Description |
|-------|--------------------|-------------|
| --model-path | models/denoiser.pt | path to load model .pt file |
| --data | ../data/test/      | path to dir or file to classify |
| --results | predictions/       | path to save predictions .csv file |
