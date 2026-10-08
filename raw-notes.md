# RAW

Notes that need to be included in the rest of the book.

## Evaluation and Verification

The input to a model can be characterized into three different names based on
its context:

- training set: used to fit model parameters $w$

- validation set: used to choose hyperparameters (e.g. polynomial degree,
  regulation strength $\lambda$)

- test set: used _once_, at the very end, to report final performance

k-Fold Cross-Validation: split data into $k$ folds; train on $k-1$ folds,
validate on the remaining fold; repeat $k$ times and average. Make efficient use
of small datasets, common in chemistry, where labelled data is often expensive
to obtain.

Rooted mean square error and fraction of variance $R^2$, where 0 is perfect:

$$RMSE = \sqrt{\frac{1}{N}\sum_{n=1}^N (y_n - \hat{y}_n)^2}$$

$$R^2 = 1 - \frac{\sum_n (y_n - \hat{y}_n)^2}{\sum_n (y_n - \bar{y}_n)^2}$$
