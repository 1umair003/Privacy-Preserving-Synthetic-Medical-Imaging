# Xception: With pre-trained weights

ImageNet initialisation; the last 100 backbone layers are fine-tuned and earlier layers stay frozen.

## Run

```bash
cd codes/code_with_weights/Xception
python train_xception.py --train_dir /path/to/train_data
```

Optional arguments: `--epochs 30` · `--batch_size 32` · `--lr 1e-4` · `--folds 5` · `--output_dir outputs` · `--fine_tune_layers 100`

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `best_model.keras` | Best fold's model, used by the test script |
| `label_encoder.pkl` | Class-to-index mapping |
| `fold_results.json` / `.csv` | Per-fold validation accuracy, mean ± std |
| `accuracy_curve.png`, `loss_curve.png` | Training vs validation curves |
| `kfold_accuracy.png` | Per-fold accuracy bar chart |
| `fold_k/` | Each fold's model and `history.csv` |

Test it with [`codes/Test/test_with_weights/Xception`](../../Test/test_with_weights/Xception).
