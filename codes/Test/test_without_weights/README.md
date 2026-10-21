# Testing: Without Pre-trained Weights

Tests the models trained in [`codes/code_without_weights`](../../code_without_weights). Each script loads `codes/code_without_weights/<Model>/outputs/best_model.keras` by default.

| Model | Folder |
|---|---|
| ResNet50 | [`ResNet50/`](ResNet50) |
| InceptionV3 | [`InceptionV3/`](InceptionV3) |
| Xception | [`Xception/`](Xception) |

## Run

```bash
# one model
python Xception/test_xception.py --test_dir /path/to/real_test

# all three models
bash test_all.sh /path/to/real_test
```

## Reported metrics

Accuracy · per-class precision, recall and F1 · macro F1 · confusion matrix · one-vs-rest ROC curves with AUC · per-image predictions.
