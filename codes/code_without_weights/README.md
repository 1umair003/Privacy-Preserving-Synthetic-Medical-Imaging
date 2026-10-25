# Training: Without Pre-trained Weights

Random initialisation. **Every layer** is trained from scratch, with no transfer learning.

| Model | Folder | Input size |
|---|---|---|
| ResNet50 | [`ResNet50/`](ResNet50) | 224 × 224 |
| InceptionV3 | [`InceptionV3/`](InceptionV3) | 299 × 299 |
| Xception | [`Xception/`](Xception) | 299 × 299 |

## Run

```bash
# one model
python Xception/train_xception.py --train_dir /path/to/train_data

# all three models
bash train_all.sh /path/to/train_data
```

## Shared training setup

| Setting | Value |
|---|---|
| Validation | 5-fold stratified cross-validation (`random_state = 42`) |
| Epochs / batch | 30 / 32 |
| Optimiser | Adam, η₀ = 1 × 10⁻⁴ |
| LR schedule | Cosine annealing + plateau drops (factor 0.2, patience 3) |
| Loss | Class-weighted categorical cross-entropy |
| Head | GAP → Dense 512 → Dropout 0.5 → Dense 256 → Dropout 0.3 → Dense 5 (softmax) |
| Augmentation | Rotation ±30°, shift / shear / zoom 0.3, brightness 0.8–1.2, horizontal flip |

Matching tests: [`codes/Test/test_without_weights`](../Test/test_without_weights) · Other condition: [`codes/code_with_weights`](../code_with_weights)
