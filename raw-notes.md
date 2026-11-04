# Notes

Notes that need to be included in the rest of the book.

## Probability distributions

### Gaussian Distribution

$$N(x| \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

Fully described by mean $\mu$ and variance $\sigma^2$

The _central limit theorem_ states that sums of many independent effects tend to
a Gaussian. For example, we assume Gaussian-distributed noise in linear
regression: $t = y(x, w) + \epsilon,\ \epsilon \sim N(0, \sigma^2)$.
Graphically, this looks like a bell curve, which is symmetric about $\mu$ and
its peak is $\frac{1}{\sqrt{2\pi}}$ at $x=0$.

We have a lot of interesting properties. Let $X\sim N(\mu, \sigma^2)$.

- Normalisation: from the Gaussian integral $\int_{-\infty}^\infty e^{-y^2/2}dy
  = \sqrt{2\pi}$

$$\int TODO$$

- Mean: TODO

- Variance: TODO

- Mean = median = mode = $\mu$ by symmetry

- Linear transformations: if $X\sim N(\mu, \sigma^2)$ and $a\ne 0$, then:

$$aX + b \sim N(a\mu + b, a^2 \sigma^2)$$

- Standardisation: as a special case:

$$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$

- Sums: if $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$
  are independent, then:

$$X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma^2_1 + \sigma_2^2)$$

- Averages: TODO

For a continuous variable, probabilities are areas under the density. With the
distribution function $F(x) = Pr(X\le x)$:

$$Pr(a\le X \le b) = \int_a^b N(x|\mu, \sigma^2) dx = F(b) - F(a)$$

There is a problem: $e^{-y^2}$ has no closed form (we cannot calculate it
analytically), so we use table values.

We only need the distribution function of $Z\sim N(0, 1)$, computed once and for
all:

$$\Phi = Pr(Z\le z) = \int_{-\infty}^z \frac{1}{\sqrt{2\pi}} TODO$$

TODO

There are some useful identities of $\Phi$:

- $\Phi (0) = 0.5$

- $\Phi (-z) = 1 - \Phi(z)$

- $P_r(X > x) = 1 - \Phi(z)$

- $P_r(|X - \mu| \le k \sigma) = 2\Phi (x) - 1$

The inverse $\Phi^{-1}$ gives quantiles: the value $x$ such that $P_r(X\le x) =
q$ is:

$$x = \mu + \sigma \Phi^{-1} (q)$$

The $68-95-99.7$ rule:

- $P_r(|X - \mu| \le \sigma) \approx 0.683$

- TODO

- TODO

For a D-dimensional vector $x$, the multivariate Gaussian distribution takes the
form:

$$N(x|\mu, \Sigma) = \frac{1}{(2\pi)^{D/2}}
\frac{1}{|\Sigma|^{1/2}}exp(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu))$$

where $\mu$ is the D-dimensional mean vector, $\Sigma$ is the $D \times D$
convariance matrix and $|\Sigma|$ is its determinant.

$$\Delta^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)$$

The quality $\Delta$ is called the _Mahalanobis distance_ form $\mu$ to $x$. It reduces
to the Euclidean distance when $\Sigma$ is the identity matrix.

The covariance matrix $\Sigma$ is _symmetric_, since $cov[x_i, x_j] = cov[x_j,
x_i]$, and is assumed to be _positive definite_. The eigenvectors of $\Sigma$
are real and can be chosen to form an _orthonormal_ set:

$$\Sigma u_i = \lambda_i u_i$$

$$u_i^T u_j = I_{ij} = TODO$$

Moreover:

 TODO

We have that $$\Delta^2 = ...$$


### Conditional Gaussian Distributions

Suppose $x$ is a D-dimensional vector with Gaussian distibution $N(x|\mu,
\Sigma)$, and that we partition $x$ into two disjoint subsets $x_a$ and $x_b$:

TODO...

Our goal is to find the distribution of $x_a$ given $x_b,\ p(x_a | x_b)$. This
can be obtained by fixing $x_b$ in $p(x)$ and renormalising. The result is again
a Gaussian:

$$p(x_a | x_b) = N(x_a | \mu_{a|b}, \Sigma_{a|b})$$

where:

$$\mu_{a|b} = \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b)$$

$$\Sigma_{a|b} = \Lambda_{aa}^{-1} = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1}
\Sigma_{ba}$$
