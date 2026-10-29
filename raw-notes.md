# Notes

Notes that need to be included in the rest of the book.

## Probability theory

### Multinomial Variables

Multinomial variables are binary variables describe quantities that can take one
of two possible values. Often, however, we encounter discrete variables that can
take one of $K$ possible mutually exlusive states. A particularly convenient
representation is the $1-of-K$ scheme: the variable is represented by a
$K$-dimensional vector $x$ in which exactly one element $x_k$ equals 1 and all
the others equal $0$. So, for instance, if we have a variable that can take
$K=6$ states and a particular observation of the variable happens to correspond
to the state where $x_3 = 1$, then $x$ will be represented by:

$$x=(0, 0, 1, 0, 0, 0)^T$$

Such vectors satisfy $\sum_{k=1}^K x_k = 1$. If we denote the probability that
$x_k = 1$ by the parameter $\mu_k$, then the distribution of $x$ is:

$$p(x|\mu) = \prod_{k=1}^K u_k^{x_k}$$

where $\mu = (\mu_1, ..., \mu_k)^T$, and the parameters $u_k$ are constrained to
satisfy $u_k \ge 0$ and $\sum_k \mu_k = 1$.

Let $D$ be a data set of $N$ ...

28 no lesson.
