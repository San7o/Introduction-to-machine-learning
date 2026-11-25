# Raw notes

Notes that need to be included in the rest of the book.

## Probability distributions

## The Data

Some exercises on real datasets.

### Bayesian filter for spam

In this first exercise, we use the Bayes theorem for spam detection. We can use
the dataset from the UC Irvine dataset
(\path{https://archive.ics.uci.edu/dataset/228/}).

Consider the following probability:

| Diagnostic test | Spam filter | Estimated from data as ... |
| --------------- | ----------- | ---------------------------|
| diesase $D$     | message in spam $S$ |                    |
| prior $Pr(D)$   | $Pr(S)$            | fraction of spam    |
| sensitivity $Pr(+ | D)$ | $Pr("free" | S)$ | fraction of spam containing "free" |
| false positive rate $Pr(+ | \lnot D)$ | $Pr("free" | \lnot S)$ | fraction of ham
containing "free" |

The steps perform spam detection are, in order:

1. Estimate the three probabilities (in the table) on 70% of the messages
   (training set)

2. Compute $Pr(S | "free")$ with Bayes' theorem, and check it on the other 30%

3. Second test: update with the word "txt", using the posterior as the new prior
   (this assumes the two words independent given the class)

4. Change the prior: what if only 1% of the messages were spam?

Note that if an event is rare, it is much more difficult to be recognized since
its prior probability will be rare too.

### Bernoulli and Beta: Bank marketing

Given a dataset of marketing campaigns (phone calls) of a Portuguese bank
institution, (\path{https://archive.ics.uci.edu/dataset/222/}), the
classification goal is to predict if the client will subscribe. Notice that the
output is a Bernulli distribution (either subscribed or not subscribed).

Let $x_n = 1$ if call $n$ end with a subscription: $x_n \sim Bern(x | \mu )$,
$\mu$ is the _conversion rate_.

1. Maximum likelihood: $\mu_{ML} = m / N$; check that it is the maximum of:

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

TODO

### The Keeling Curve

We have a NOAA global monitoring laboratory monitoring annual mean $CO_2$ (http://gml.noaa.gov/ccgg/trends). It has 67 rows, one per year.

1. Train on 1959-2000 only; forecast 2001-2025 and compare with what happened

2. Model: polynomial of order $M$ in $x = (year - 1990)/30$ (scaling keeps
   $\phi^T \phi$ well conditioned); $M=2$ to begin with.

3. Hyper-parameters: prior precision $\alpha = 10^{-3}$ (weak); noise precision
   $\beta = 1 / \hat{sigma}^2$ from the residuals of the least-squares fir

4. Posterior and predictive distribution:

$$S_N^{-1} = \alpha I + \beta \phi^T \phi, \qquad m_N = \beta S_N \phi^T t,
\qquad s^2(x) = \frac{1}{\beta} + \phi (x)^T S_N \phi(x)$$

5. Evaluate: fraction of the 25 feature years inside the 95% band $m(x) +- 1.96
   s(s)$; repeat for $M = 1, 3$

### Von Mises distribution: Wind at Ancona Airport

Meteostat blut hourly data, station 16191 (Ancona Flaconara, Marche), from
national weather services (http://bulk.meteostat.net/v2/hourly/16191.csv.gz). A
direction is an angle.

Model the direction at $14:00$ as $\theta \sim p(\theta | \theta_0, m) =
\frac{1}{2\pi I_0 (m)} exp(m \cos (\theta - \theta_0))$

1. The wrong way: the arithmetic mean of the degrees

2. ML mean direction: average the unit vectors $(\cos \theta_i, \sin \theta_i)$

$$\theta_0^{ML} = atan2 (\sum_i \sin \theta_i, \sum_i \cos \theta_i), \qquad
\bar{r} = ||\frac{1}{N} \sum_i (\cos \theta_i, \sin \theta_i)||$$

3. ML concentration: solve $A(m_{ML}) = \bar{r}$ with $A(m) = I_1(m) / I_0(m)$
   (bisection / brentq); check with _scipy.stats.vonmises.fit_.

4. Use the model: probability that the wind blows from the north-east quadrant
   $(0'-90')$, compared with the data

5. Compare with 03:00 (nights): does the direction change?

### The Old faithful Geyser

Dataset of 272 eruptions of Old Faithful: how long the time of the eruption, and
how long it waited before the next. Used throughout Bishop PRML.

1. 1D: waiting time $x$, micture $p(x) = \sum_{k=1}^K \pi_k N(x | \mu_k,
   \sigma_k^2)$ with $K=2$, fittend with the $M algorithm

   - E-step: responsibilities $\gamma_{nk} = \pi_k N(x_n | \mu_k, \sigma_k^2) /
     \sum_j \pi_j N(TODO)$

TODO

---

## Regression

$$y(x, w) = \sum_{j=0}^{M-1} w_j \phi_j (x) = w^T \phi(x)$$

$\phi(x)$ is a basis function and is used to morph the space, introducing
non-linearity. Example functions are:

- $\phi_j (x) = x_i$

- $\phi_j (x) = exp(- \frac{(x - \mu_j)^2}{2 s^2})$

- $\phi_j (x) = \sigma (\frac{x - \mu_j}{s})$
