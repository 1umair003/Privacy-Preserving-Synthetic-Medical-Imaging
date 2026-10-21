# Data Preparation

`make_hybrid_dataset.py` builds the **Hybrid (50% real + 50% synthetic)** training set used in Group 1 of Table 2.

```bash
# from codes/
python data_preparation/make_hybrid_dataset.py \
    --real_dir data/real_train --synthetic_dir data/synthetic_train --output_dir data/hybrid_train
```

For each class, it takes the same number of real and synthetic images (the smaller of the two, or `--per_source`) and prefixes the files with `real_` / `synth_`.
