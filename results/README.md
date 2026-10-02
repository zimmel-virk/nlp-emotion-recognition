# Reference results

The following accuracies are transcribed from the final model-comparison figure in the coursework report:

| Model | Accuracy |
|---|---:|
| Naive Bayes | 84.87% |
| SVM | 89.89% |
| Simple Neural Network | 90.02% |
| LSTM + GloVe | 91.50% |

`reference_metrics.csv` stores these report values.

`reference_model_comparison.png` visualizes the same values.

When you rerun the experiment, generated outputs will be written to `results/generated/`, which is excluded from Git by default so that you can decide which fresh results to commit.
