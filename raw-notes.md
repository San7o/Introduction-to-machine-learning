# Notes

Notes that need to be included in the rest of the book.

## Probability distributions

#### Mixtures of Gaussians

A single Gaussian cannot capture _moltimodal_ data. A linear superposition of
$K$ Gaussians can:

$$p(x) = \sum_{k=1}^K \pi_k N(x | \mu_k, \Sigma_k)$$

where the _mixing coefficients_ satisfy $0 \le \pi_k \le 1$ and $\sum_k \pi_k =
1$.

We do not have a closed form, but we can use some iterative algorithms (_EM
algorithm_).

Slide 141
