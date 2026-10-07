
# Math background

We cannot begin our discussion without first having a decent background in
calculus, linear algebra, probability and statistics, among other things — sorry.
Fortunately, I have prepared some notes for you to study or revise the most
fundamental concepts that you should know about. Feel free to skip this chapter
if you already grasp these topics (nerd!).

## Eigenvalues and Eigenvectors

Let's look at Eigenvalues and Eigenvectors in linear algebra for now. I will use
the great book "Algebra Lineare" by Marco Abate as my reference. Eigenvectors
are an important theoretical and practical topic as they are used for some
tricks in unsupervised learning, and are often used as the preferred axis for
rotations or other transformations.

Let $T: V\rightarrow V$ be an endomorphism (maps from a space to itself) of a
vector space $V$. A vector $v_0 \ne 0$ of $V$ is an eigenvector of $T$ with
respect to the _eigenvalue_ $\lambda$ if:

$$T(v_0)=\lambda v_0$$

The set of _eigenvalues_ is called the _spectrum_ of $T$. If $\lambda \in
\ spectrum(T)$ then the set $V_{\lambda}$ is called _eigenspace_ and is defined as
follows:

$$V_{\lambda} = \{ v\in T\ |\ T(v)=\lambda v \}$$

Notice that from the definition of eigenvalue follows:

$$Tv_0 - \lambda v_0 = 0$$
$$(T - \lambda I)v_0 = 0$$

Therefore $T-\lambda I0$ is singular (non invertible), meaning that its
determinant is 0:

$$det(T-\lambda I)=0$$

This equation provides a practical way to compute the eigenvalues. Once you have
a specific eigenvalue $\lambda$ from solving this last equation, you can
substitute it back into the definition and solve $(T - \lambda I)v = 0$ to find
the corresponding eigenspace and its eigenvectors.

## Optimization and The Lagrangian dual Function

In this section we will delve into the realm of optimization and in particular
we will look at the _Lagrange dual function_. I will be using the book "Convex
Optimization" by Stephen Boyd and Lieven Vandenberghe as my reference.

### Optimization problems

A mathematical optimization problem, or just optimization problem, has
the form:

$$minimize\ f_0(x)$$

subject to:

$$f_i(x)\le b_i,\ i=1, ..., m$$
$$h_i(x) = 0,\ i=1, ..., p$$

Here the vector $x=(x_1, ..., x_n)$ is the optimization variable of the problem,
the function $f_0: \mathbb{R}^n \rightarrow \mathbb{R}$ is the objective
function, the functions $f_i, h_i: \mathbb{R}^n \rightarrow \mathbb{R},\ i=1,
..., m$ are the constraint functions, and the constants $b_i, ..., b_m$ are the
limits, or bounds, of the constraints. A vector $x*$ is called optimal, or a
solution of the problem, if it has the smallest objective value among all
vectors that satisfy the constraints.

The optimal value $x*$ of the above problem, also referred to as $p*$, is
defined as:

$$x* = inf\{ f_0(x) | f_i(x) \le 0,\ i=1, ..., m,\ h_i(x)=0,\ i=1,
..., p \}$$

The optimal value for a maximization problem would be the greater value instead.

The types of optimization problems can be divided into:

- _linear programming_ where the objective and all constraint functions are
  linear meaning they satisfy the equality $f_i(\alpha x+ \beta y) = \alpha f(x)
  + \beta f(y)$ for all $x, y \in \mathbb{R}^n$ and all $\alpha, \beta \in
  \mathbb{R}$. There is no simple analytical formula for the solution of a
  linear program, but there are a variety of very effective methods including
  Dantzig's simplex method and interior-point methods.

- _convex optimization_ where the objective and constraint functions are convex
  meaning they satisfy the equation $f_i(\alpha x+ \beta y) \le \alpha f(x) +
  \beta f(y)$ for all $x, y \in \mathbb{R}^n$ and all $\alpha , \beta \in
  \mathbb{R},\ \alpha + \beta = 1, \alpha \ge 0, \beta \ge 0$. We say a function
  $f$ is _concave_ if $-f$ is convex. If just the objective function is convex
  but not the constraints, the problem is called _quasiconvex_. Similar to
  linear programming, in general there is no analytical formula for solving
  these problems, but there are effective methods to do so like interior-point
  methods. Fortunately, those are quite efficient for computers and we are able
  to solve them quickly.

- _non linear optimization_ for problems with non linear objective or
  constraints which further divides into _local_ optimization (finding a local
  best solution) or _global_ optimization which is usually slower in terms of
  computation time. There is no general efficient solution.

### The Lagrangian

Consider a minimization problem as defined above, we assume the domain
$D=\prod_{i=1}^m dom(f_i) \cap \prod_{i=1}^p dom(h_i)$ is non empty
and denote the optimal value by $p*$.

The idea is to account for the constraints by augmenting the objective
function with a weighted sum of the constraints. We define the
_Lagrangian_ $L:\ \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p
\rightarrow \mathbb{R}$ as:

$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) +
\sum_{i=1}^p \nu_i h_i(x)$$

We refer to $\lambda_i$ as the _Lagrange multiplier_ associated with the
i-th inequality constraint $f_i\le 0$ and $\nu_i$ as the Lagrange
multiplier associated with the i-th equality constraint $h_i(x)=0$. The
vectors $\lambda$ and $\nu$ are called the dual variables or Lagrange
multiplier vectors associated with the problem.

We define the _Lagrange dual function_ $g: \mathbb{R}^m \times
\mathbb{R}^p \rightarrow \mathbb{R}$ as the minimum value of the
Lagrangian over $x$ associated to the previous problem: for $\lambda
\in \mathbb{R}^n,\ \nu \in \mathbb{R}^p$,

$$g(\lambda , \nu) = inf_{x\in D} L(x, \lambda , \nu ) = inf_{x\in
D}(f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{i=1}^p \nu_i
h_i(x))$$

It is easy to see that $g(\lambda , \nu)$ is the lower bound of the
optimal value $p*$, in fact the sum of $f_i(x)$ is negative and the
sum of $h_i(x)$ is zero for valid solutions, hence we are subtracting
from $f_0(x)$ and we are looking for the vectors $\lambda$ and $\nu$
that give the smaller valid value (or the greater subtraction), which
is the lower bound.

### The dual problem

We showed that $g(\lambda , \nu)$ is the lower bound of the optimal
value $p*$. But to find the optimal value we need to find the highest
lower bound. Hence, we need to maximize the dual function, meaning we need
to solve the problem:

$$maximize g(\lambda , \nu)$$

subject to:

$$\lambda \ge 0$$

This problem is called the Lagrange dual problem associated with the
original minimization problem. We refer to $(\lambda *, \nu *)$ as
dual optimal or optimal Lagrange multipliers if they are optimal for
the problem.

The Lagrange dual problem is a _convex optimization problem_, since
the objective to be maximized is concave (because "the dual function
is the point-wise infium of a family of affine functions", gl) and the
constraint is convex (indeed the constraint $f(x) < \lambda_i$ has
function $f(x)=0$ which satisfies the definition of convex).  This is
always the case whether or not the original problem (sometimes referred
to as primal problem) is convex.

### Convex optimization problems

Let's now consider convex optimization problems. A fundamental
property of convex optimization problems is that any locally optimal
point is also globally optimal, this can be proven my contradiction.

The convex optimization problem is called a quadratic program if the
objective function is convex and quadratic, and the constraint
functions are affine (linear). A quadratic program can be expressed in
the form:

$$minimize\ \frac{1}{2}x^TPx + q^Tx+r$$

subject to:

$$Gx \le h$$
$$Ax=b$$

where $P$ is a symmetric positive matrix where $x^TPx$ is always
positive.

There are many techniques to solve these kind of problems. If the problem is
unconstrained, you can use backtracking (move in steps and reduce the step size
incrementally), gradient descent, steepest descent or newton's method. If it is
also very simple, you can calculate the first and second derivatives and equal
them to 0. In equality constrained problems you can use more sophisticated
versions of the Newton step or using barriers.  In general, you can use the
interior-point method.

## Probability

In this section I will summarize the core ideas of probability theory.

### Basic Definitions

There are two main interpretations of what probability is. The first one is to
think of a probability as the frequency of a certain event occurring. If a coin
has 0.5 probability of landing heads, then we expect the coin to land heads
about half of the time. The other interpretation views probability as a quantity
of uncertainty or ignorance about something, this is more related to information
rather than repeated trials. In the coin example, here we mean that the coin is
equally likely to land heads or tails on the next toss.

We define the following terms:

- $\Omega$ as the sample space, it is composed of independent events
- $A\subset 2^{\Omega}$ is a subset of the sample space of a problem
- $a\in A$ is an event

### Probability Function

A function $P$ is a probability function if:

- $P$ is non negative: $P(A)\ge 0\ \forall A\in \Omega$
- $P$ is normalized: $P(\Omega)=1$
- $P$ is $\sigma$ -additive: if $A_i \cap A_j \ne \emptyset,\ A\in \Omega$ then  $P(\cup_i A_i)=\sum_i P(A_i)$

  From set theory, you can demonstrate that the following holds:

$$P(A\cup B) = P(A) + P(B) - P(A\cap B)$$

### Bayes Theorem:

Bayes theorem is a foundamental theorem that correlates the probabilty
of variables given another variable. First, we define $P(A|H)$ as the
probability of A given H, and we can calculate this as:

  $$P(A|H)=\frac{P(A\cap H)}{P(H)},\ P(H)>0$$

The Bayes theorem states:
  
  $$P(A_i|B)=\frac{P(B|A_i)P(A_i)}{\sum_j P(B|A_j)P(A_j)}=\frac{P(B|A_i)P(A_i)}{P(B)}$$

### Stochastic Independence

Events $A\in \Omega$ are said to be independent if:

$$P(A_{i1} \cap A_{i2} \cap ... \cap A_{ik}) = \prod_{j=1}^k P(A_{ij}) = P(A_{i1})\cdot P(A_{i2})\cdot ...\cdot P(A_{ik})$$

The same applies to bivaraite functions:

$$P_{X, Y}(x, y) = P_X(x)\cdot P_Y(y),\ \forall (x, y)\in R_x \times R_y$$

moreover:

$$P_{X|Y}(x|y) = P_X(x)$$
$$P_{Y|X}(y|x) = P_Y(y)$$

### Distribution Function

A function $F$ is a probability distribution function if:

- $F$ never decreases
- $F$ is right-continuous
- $F$ always has a left-limit
- $\lim_{x\to - \infty} f(x)=0$
- $\lim_{x\to\infty}f(x)=1$

  Then $P((a, b])=^{(discrete)}F(b)-F(a^-) =^{(continuous)} \int_a^b f(x)dx$

Where $f(x)$ is called density when if $F\in C^1$ in the continuous
formulation.

### Random Variable

Random variables are a mathematical formalization used to model
quantities which depend on random events, it lets us quantify
random events so that we can make probability calculations.

More formally, a random variable is a function that maps event in
some sample space $\Omega$ to a set of outcomes in measurable space $E$,
which is often $\mathbb{R}$.

$$X:\Omega \to E$$

The probability that $X$ takes on a value in a measurable set
$S\subseteq E$ is written as

$$P(X\in S) = P(\{ \omega \in \Omega \ |\ X(\omega)\in S \})$$

### Notable Random Variables

Some random variables appear more than others, so a few of them
are worthy of their name. You will see those everywhere in nature,
economics, populations, and more.

Bernoulli:

$$X(\omega)=\{0, 1\}$$

Rademacher:

$$Y(\omega)=\{ -1, 1 \}$$

Binomial $X\sim Bin(n, p)$:

$$P_x(J)=\{ \binom{n}{k}p^J(1-p)^{n-J},\ j=1...n\ |\ 0\ otherwise \}$$

Poissont $X\sim Pois(\lambda)$:

$$P_x\{\frac{\lambda^xe^{-\lambda}}{x!}, n\in \mathbb{N}\cup \{ 0 \}\ |\ 0\ otherwise\}$$

Geometric:

$$P(y)=\{ p(1-p)^{y-1},\ y\in \mathbb{N}\ |\ 0\ otherwise \}$$

Uniform $X\sim Unif[a, b]$:

$$f_x(x)=\{ \frac{1}{b-a},\ x\in [a, b]\ |\ 0\ otherwise \}$$

Normal (Gaussian) $X\sim N(\mu , \sigma^2)$:

$$f_x(x) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}}$$

Exponential $X\sim Exp(\lambda)$:

$$f_x(x)=\lambda e^{\lambda x}\mathbb{1}(x>0)$$

### Expected Value

We define the expected value in a discrete space as:

$$\mathbb{E}(x) = \sum_{x\in R_x} xp_x(x)$$

and in the continuous:

$$\mathbb{E}(x)=\int_{-\infty}^{\infty} xf_x(x)dx $$

The expected value is a linear function:

$$E(aX+b) = a\mathbb{E}(x)+b$$
$$E(g(x))=\sum_{x\in R_x} g(x)p_x(x)$$

Known formulas for notable random variables:

- Bernoulli: $\mathbb{E}(x)=p$
- Binomial: $\mathbb{E}(x)=np$
- Geometric: $\mathbb{E}(x)= \frac{1}{p}-1$
- Normal: $\mathbb{E}(x)=\mu$
- Exponential: $\mathbb{E}(x)=\frac{1}{\lambda}$
- Poisson: $\mathbb{E}(x)=\lambda$

### Variance

We define variance as:

$$\mathbb{V}ar(x)=\mathbb{E}(x^2)-\mathbb{E}(x)^2 = \mathbb{E}[(x-\mathbb{E}[x])^2]$$

Moreover:

$$\mathbb{V}ar(x)=\mathbb{E}(\mathbb{V}ar(x|y)) + \mathbb{V}ar(\mathbb{E}(x|y))$$

### Covariance

We define the covariance as:

$$\mathbb{C}ov(x, y)=\mathbb{E}((x-\mathbb{E}(x))(y-\mathbb{E}(y)))=\mathbb{E}(XY)-\mathbb{E}(x)\mathbb{E}(y)$$

### Standardization

$$z=g(x)=\frac{x-\mathbb{E}(x)}{\sqrt{\mathbb{V}ar(x)}}$$

After this transformation:

- $\mathbb{E}(z)=0$
- $\mathbb{V}ar(z)=1$

The opposite can be achieved:

$$x=\sigma z + \mu$$

### Markov Inequality

Let $Y$ be a random variable non negative, then $\forall a>0$:

$$P(Y\ge a)\le \frac{\mathbb{E}(y)}{a}$$

### Chebyshev Inequality

Let $Y$ be a random variable, $\mu = \mathbb{E}(y)$,
$\sigma^2=\mathbb{V}ar(y)$, then $\forall \epsilon > 0$:

$$P(|Y-\mu| \ge \epsilon)\le \frac{\sigma^2}{\epsilon^2}$$
