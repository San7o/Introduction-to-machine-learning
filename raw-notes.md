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

## The Data

### Bayesian filter for spam

Use of Bayes theorem for spam detection. We can use the dataset from the UC
Irvine dataset (\path{https://archive.ics.uci.edu/dataset/228/}).

| Diagnostic test | Spam filter | Estimated from data as ... |
| --------------- | ----------- | ---------------------------|
| diesase $D$     | message in spam $S$ |                    |
| prior $Pr(D)$   | $Pr(S)$            | fraction of spam    |
| sensitivity $Pr(+ | D)$ | $Pr("free" | S)$ | fraction of spam containing "free" |
| false positive rate $Pr(+ | \lnot D)$ | $Pr("free" | \lnot S)$ | fraction of ham
containing "free" |

1. Estimate the three probabilities on 70% of the messages (training set)

2. Compute $Pr(S | "free")$ with Bayes' theorem, and check it on the other 30%

3. Second test: update with the word "txt", using the posterior as the new prior
   (this assumes the two words independent given the class)

4. Change the prior: what if only 1% of the messages were spam?

Note that if an event is rare, it is much more difficult to be recognized since
its prior probability will be rare too.

### Bernoulli and Beta: Bank marketing

Dataset of marketing campaigns (phone calls) of a Portuguese bank institution.
(\path{https://archive.ics.uci.edu/dataset/222/}), the classification goal is to
predict if the client will subscribe. The output is a Bernulli distribution.

Let $x_n = 1$ if call $n$ end with a subscription: $x_n \sim Bern(x | \mu )$,
$\mu$ is the _conversion rate_.

1. Maximum likelihood: $\mu_{ML} = m / N$; check it is the maximum of:

$$\ln p(D | \mu ) = m \ln \mu + (N - m) \ln (1 - \mu )$$

2. Binimial: an operator makes 20 calls a day; how many subscriptions? $m \sim
   Bin(m | 20, \mu_{ML})$

3. Beta prior: $Beta(\mu | a, b)$ with $a = 1, b = 9$ ("about one in ten");
   posterior

$$p(\mu | D) = Beta(\mu | a + m, b + M - m)$$

4. Sequential: update after the first 10, 100, 1000, ... calls

5. Grouped: one posterior per month, with a $95%$ credible interval: small
   months get wide intervals and are pulled towards the prior

### Multinomial and Dirichlet: Online Shoppers

In an e-commerce, we have a number of interactions with the website with some
information such as operating system and browser used
(\path{https://archive.ics.uci.edu/dataset/468/}).

The OS of a session is a 1-of-K variable $x \sim Mult(x | \mu),\ K = 8$.

1. Full data: $\mu_{k, ML} = m_k / N$ with $N = 12330$: our "ground truth"

2. Pilot: we only see $N = 200$ random sessions. Some systems are never seen :
   $\mu_{k, ML} = 0$

3. Dirichlet prior $Dir(\mu | \alpha)$ with $\alpha = (1, ..., 1)$ (uniform over
   the simplex); posterior:

$$p(\mu | D) = Dir(\mu | \alpha + m), \qquad \mathbb{E}[\mu_k | D] =
\frac{\alpha_k + m_k}{\alpha_0 + N}$$

4. Credible intervals: TODO

5. Multinomial: TODO

### The Gaussian Distribution

Partial census of the Kung San people by Nancy Howell
(\path{https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv}).
We want to study the distribution of human heights. For this dataset, we could
use either Gaussian or T-strudents distributions.

Model the height of an adult woman as $x \sim N(x | \mu, \sigma^2)$

1. Fit by maximum likelihood: $\mu_{ML} = \bar{x}$

TODO

### The multivariate gaussian

We want to study the distribution of pairs $x = (height, weight)^T$.

Model $x \sim N (x | \mu, \Sigma )$ with $D=2$

1. Fit: $\mu_{ML} = \frac{1}{N} \sum_n x_n$ and $\Sigma_{ML} = \frac{1}{N}\sum_n
   (x_n \mu_{ML}) (x_n - \mu_{ML})^T$

2. Predict: the conditional of a Gaussian is Gaussian:

$$p(w | h) = N (w | \mu_w + \frac{\Sigma_{wh}}{\Sigma_{hh}}(h - \mu_h ),
\Sigma_{ww} - \frac{\Sigma^2_{wh}}{\Sigma_{hh}})$$

3. Detect anomalies: the squared Mehalanobis distance $\Delta^2 = (x - \mu)^T
   \Sigma^{-1} (x - \mu)$ follows a $X^2$ distrivution with $D$ degrees of
   freedom; flag the points above its 99% quantile.

### Learning the parameters of a Gaussian

Focusing on the 187 heights of the Kung women dataset, read as a stream, one
woman at a time (in random order).

1. Bias of ML: thear the 187 women as a population and draw many samples of size
   $n$: on average $\sigma^2_{ML} = \frac{n-1}{n} \sigma^2$, where the
   $\frac{1}{n-1}$ estimator is unbiased

2. Sequential ML: $\mu^{(N)} = \mu^{(N - 1)} + \frac{1}{N} (x_N - \mu^{(N - 1)})$ gives exactly $\bar{x}$

3. Robbins-Monro: TODO

4. Bayesian known-precision: TODO

5. Bayesian unknown precision: TODO

### Student's T-distribution

We have a dataset containing the daily closing prices of major European stock
indices (\path{https://rdrr.io/r/datasets/EuStockMarkets.html}). We can use both
a Gaussian and a Student's T-distribution. We would notice that the Gaussian
distribution underestimates extreme losses by orders of magnitudes.
