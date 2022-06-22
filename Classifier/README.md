# Usage

### Train

~~~~~~
python train.py [-h] [--model-type MODEL_TYPE] [--train-data TRAIN_DATA] [--val-data VAL_DATA] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--save-path SAVE_PATH]
~~~~~~

| Param | Default | Description |
|-------|---------|-------------|
| --model-type | MobileNet | use MobileNet or Custom architecture |
| --train-data | ../data/train/ | path to train dir |
| --val-data | ../data/val/ | path to validation dir |
| --epochs | 25 | epochs number |
| --batch-size | 64 | batch size |
| --save-path | models/new_model.pt | path to save model .pt path |

### Classify

~~~~~~
python  classify.py [-h] [--model-path MODEL_PATH] [--data DATA] [--results RESULTS]
~~~~~~

| Param | Default | Description |
|-------|---------|-------------|
| --model-path | models/MobileNet.pt | path to load model .pt file |
| --data | ../data/test/ | path to dir or file to classify |
| --results | predictions/new_preds.csv | path to save predictions .csv file |
