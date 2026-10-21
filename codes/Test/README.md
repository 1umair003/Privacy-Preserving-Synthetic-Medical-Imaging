# Test

| Folder / script | Purpose |
|---|---|
| [`test_with_weights/`](test_with_weights) | Tests models from `codes/code_with_weights` |
| [`test_without_weights/`](test_without_weights) | Tests models from `codes/code_without_weights` |
| `statistical_tests.py` | Paired t-test and Wilcoxon signed-rank test on fold accuracies |
| `summarize_results.py` | Collects every test result into one Table 2-style table |

## Run (from `codes/`)

```bash
bash Test/test_with_weights/test_all.sh data/real_test
bash Test/test_without_weights/test_all.sh data/real_test

python Test/statistical_tests.py --condition with_weights
python Test/statistical_tests.py --condition without_weights

python Test/summarize_results.py      # -> Test/results_summary.csv / .md
```

> With 5 folds, the smallest possible two-sided Wilcoxon p-value is 0.0625, which is why the paired t-test is used for significance at α = 0.05.
