# Test InceptionV3: With pre-trained weights

Evaluates the model trained in [`codes/code_with_weights/InceptionV3`](../../../code_with_weights/InceptionV3).

## Run

```bash
cd codes/Test/test_with_weights/InceptionV3
python test_inceptionv3.py --test_dir /path/to/test_data
```

By default it loads `codes/code_with_weights/InceptionV3/outputs/best_model.keras`. Use `--model_path` and `--label_encoder` to point elsewhere.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `test_results.json` | Accuracy, macro F1, mean AUC, per-class metrics, confusion matrix |
| `per_class_metrics.csv` | Precision, recall, F1, support and AUC per class |
| `predictions.csv` | Per-image prediction and class probabilities |
| `confusion_matrix.png` | Confusion matrix heat map |
| `roc_curves.png` | One-vs-rest ROC curves with AUC |
