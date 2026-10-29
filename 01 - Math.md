
# A Mathematical background

We cannot begin our discussion without first having a decent background in
calculus, linear algebra, probability and statistics, among other things — sorry.
Fortunately, I have prepared some notes for you to study or revise the most
fundamental concepts that you should know about. Feel free to skip this chapter
if you already grasp these topics (nerd!).

## Foundations of mathematics

Just to warm up a bit.

### Set theory

Let's start from the very foundation of math (the standard one according to the
ZFC framework): set theory. A _set_ is a primitive notion, that is: it can only
described in terms of its properties. We can define a set with three approaches:

- by listing its elements (extensional): $A = \{2, 4, 6, 8\}$
- by stating a defining property (intensional): $A = \{x : P(x)\}$
- by enumeration: a rule generating elements one by one, e.g. $A=\{1, 2, 3,
  ...\}$

Some basic terminology now. We call the _empty set_ $\emptyset$ as the set such
that for any $x$, $x \notin \emptyset$. A _singleton_ is a set with a single
value. A _subset_ is defined in terms of $\in$:

$$A \subseteq B \iff  x, (x \in A \Rightarrow x \in B) $$

Some notable sets that you will find often:

- _Natural numbers_ $\mathbb{N}$: defined inductively (Peano)

- _Integers_ $\mathbb{Z}$

- _Rational numbers_ $\mathbb{Q}$

- _Real numbers_ $\mathbb{R}$ (informal description): all numbers expressible as
  a (possibly infinite, non-repeating) decimal expansion; $\mathbb{R}$ also
  contains the irrationals which cannot be written as $p/q$

Note that:

$$\mathbb{N} \subseteq \mathbb{Z} \subseteq \mathbb{Q} \subseteq \mathbb{R}
\subseteq{C}$$

There are several operations on sets. The membership operator $\in$ as in $x \in
A$ means "x is an element of A", while $x \notin A$ is its negation. We can
define basic pointwise operations in terms of membership:

- union: $A \cup B = \{ x: x\in A \lor x\in B \}$
- intersection: $A \cap B = \{ x: x\in A \land x\in B \}$
- difference: $A \slash B = \{ x: x\in A \land x\notin B \}$
- complement: $A^C = \{x\in A: x\notin A \}$
- cartesian prodict: $A \times B = \{(a, b): a\in A \land b\in B\}$

### Relations

Let $A$ and $B$ be sets. A _binary relation_ $R$ from $A$ ro $B$ is any subset
of the Cartesian product $A \times B$:

$$R \subseteq A \times B$$

- if $(a, b)\in R$, we say that $a$ is related to $b$ by $R$, often written as
  $aRb$
- if $(a, b)\notin R$, they are not related

A relation $R$ on $A$ is an equivalence relation if it satisfies, for all $a, b,
c\in A$ the following conditions:

- _reflective_: $a\ R\ a$
- _symmetric_: $a\ R\ b \Rightarrow b\ R\ a$
- _transitive_: $a\ R\ b \land b\ R\ c \Rightarrow a\ R\ c$

The _equivalence classes_ of a R partition A is defined as:

$$A / R = \{[x]: x\in A\}$$

A relation $R$ on $A$ is a _partial order_ if it satisfies, for all $a, b, c\in
A$:

- reflective: $a\ R\ a$
- antisymmetric: $a\ R\ b \land b\ R\ a \Rightarrow a = b$
- transitive: $a\ R\ b \land b\ R\ c \Rightarrow a\ R\ c$

A set $A$ equipped with a partial order is a _partially ordered set_ (poset),
usually written $(A, \le )$.

_Total_ (linear) order is a partial order where every pair is comparable.

$$\forall a, b\in A,\ a\ R\ b \lor \ b\ R\ a$$

### Functions

A _function_ $f$ from $A$ to $B$, written $f: A\rightarrow B$, is a relation
$f\subseteq A\times B$ such that:

$$\forall a\in A, \exists !\ b\in B\ such\ that\ (a, b)\in f$$

Where $A$ is the _domain_, $B$ is the _codomain_. The _image_ (or range) is
$f(A) = \{f(a): a\in A\} \subseteq B$. In other words, a function is a
particular relation mapping an element in a domain to another element in a
codomain: every $a\in A$ must have the same $b\in B$.

Now the three famous characterization of functions which I always forget:

- A function $f$ is called _injective_ if $\forall a_1, a_2 \in A, f(a_1) =
  f(a_2) \Rightarrow a_1 = a_2$
- A function is _surjective_ if $\forall b \in B, \exists a\in A$ such that
  $f(a)=b$.
- If $f$ is both injective and _surjective_, the relation is a bijection and has
  a well-defined _inverse_ $f^{-1}$.

### Cardinality

The sets $A, B$ have the same _cardinality_, written $|A| = |B|$, if there
exists a bijection $f: A\rightarrow B$. $A$ is _finite_ with $|A| = n$ if there
is a bijection $f:\{1, ..., n\}\rightarrow A$; $|\emptyset|=0$. $A$ is
_infinite_ if it is not finite.

A set $A$ is _countable_ if it is finite, or if there exists a bijection $f:
\mathbb{N} \rightarrow A$ (in which case $A$ is countably infinite).

A set is _uncountable_ if it is infinite and not countable: no bijection with
$\mathbb{N}$ exists. Because of Cantor's diagonal argument, $|\mathbb{N}| \neq
|\mathbb{R}|$.

Looking at our beloved sets:

- $\mathbb{N}$ is countable: this can be trivially demonstrated (the identity
  map), hence $|\mathbb{N}| = |\mathbb{N}|$
- $\mathbb{Z}$ is countable: list +1, -1, +2, -1... with the bijection $f(x) =
  x$ if x is even or $-x$ if it is odd. Hence $|\mathbb{N}| = |\mathbb{Q}|$.
- $\mathbb{Q}$ is countable: proof using a space filling curve. $|\mathbb{N}| =
  |\mathbb{Q}|$.
- $\mathbb{R}$ is not countable: because of Cantor's diagonalization,
  $|\mathbb{N}| \neq |\mathbb{R}|$.

### Distance functions

A _distance function_ (or _metric_) on a set $X$ is a function:

$$d: X \times X \rightarrow \mathbb{R}$$

such that, for all $x, y, z\in X$ the following conditions hold:

1. Identity of indiscernibles: $d(x, y) = 0 \iff x = y$
2. Symmetry: $d(x, y) = d(y, x)$
3. Triangle inequality: $d(x, z) \le d(x, y) + d(y, z)$
4. Non-negativity: this is not a separate axiom, it follows form the three
   above: $0 = d(x, x) \le d(x, y) + d(y, x) = 2 d(x, y)$, so $d(x, y \ge 0)$
   always.

A set $X$ together with a metric $d$ is called a _metric space_, written $(X,
d)$.

We can define $d$ in many ways:

- Euclidean: $d(x, y) = (\sum_i (x_i - y_i)^2)^{1/2}$, "$L2$"
- Manhattan: $d(x, y) = \sum_i |x_i - y_i|$, "$L1$"
- Chebyshev: $d(x, y) = max_i |x_i - y_i|$, "$L_{\infty}$"
- Minkowsky: $d(x, y) = (\sum_i |x_i - y_i|^p)^{1/p}$
- Discrete: $d(x, y)  = 0$ if $x=y$, else 1
- Hamming: number of positions where two equal-length strings differ
- Haccard / Tanimoto: $d(A, B)=1-\frac{|A \cup B|}{A \cap B}$

Note the the $LX$ notation is used as a special case of $p$ in the Minkowsky
distance function.

### Continuous functions

Now that we have formalized distance, we can properly define a continuous
function.

Let $(X, d_x)$ and $(Y, d_y)$ be metric spaces. A function $f: X \rightarrow Y$
is _continuous_ at $x_o$ if the following holds:

$$\forall \epsilon > 0, \exists \delta > 0\ such\ that\ d_x(x, x_o) < \delta
\Rightarrow d_y(f(x), f(x_0)) < \epsilon$$

Gradient-based optimisation requires the loss function to be countinuous,
ideally differentiable, in the parameters $w$.

## Linear Algebra

Some important notions about linear algebra.

### Eigenvalues and Eigenvectors

Let's look at Eigenvalues and Eigenvectors in linear algebra for now. I will use
the great book "Algebra Lineare" by Marco Abate as my reference. Eigenvectors
are an important theoretical and practical topic as they are used for some
tricks in unsupervised learning, and are often used as the preferred axis for
rotations or other transformations.

Let $T: V\rightarrow V$ be an endomorphism (a function that maps a space onto
itself) of a vector space $V$. A vector $v_0 \ne 0$ of $V$ is an eigenvector of
$T$ with respect to the _eigenvalue_ $\lambda$ if:

$$T(v_0)=\lambda v_0$$

The set of _eigenvalues_ is called the _spectrum_ of $T$. If $\lambda \in \
spectrum(T)$ then the set $V_{\lambda}$ is called _eigenspace_ and is defined as
follows:

$$V_{\lambda} = \{ v\in T\ |\ T(v)=\lambda v \}$$

Notice that from the definition of eigenvalue follows:

$$Tv_0 - \lambda v_0 = 0$$
$$(T - \lambda I)v_0 = 0$$

Therefore $T-\lambda I$ is singular (non invertible), meaning that its
determinant is 0:

$$det(T-\lambda I)=0$$

This equation provides a practical way to compute the eigenvalues. Once you have
a specific eigenvalue $\lambda$ from solving this last equation, you can
substitute it back into the definition and solve $(T - \lambda I)v = 0$ to find
the corresponding eigenspace and its eigenvectors.

## Optimization

In this section we will delve into the realm of optimization and in particular
we will look at the _Lagrange dual function_. I will be using the book "Convex
Optimization" by Stephen Boyd and Lieven Vandenberghe as my reference.

### Optimization problems

A mathematical optimization problem, or just _optimization problem_, has
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

### Multi variable optimization

The _derivative_ of $f: \mathbb{R} \rightarrow \mathbb{R}$ at $w$ (when the
limit exists) is the slope of the tangent to the graph of $f$ at $w$"

$$f'(w) = \frac{df}{dw}(w) = \lim_{h\rightarrow 0} \frac{f(w+h) - f(w)}{h}$$

Where if $f'(w) > 0$ then $f$ increases mobing right and if $f'(w) < 0$ then $f$
decreases: the sign of $f'$ tells us which way is _downhill_. At a minimum (or
maximum) $w^*$ in the interior of the domain, $f'(w^*) = 0$ implies that $w^*$
is a _stationary point_. The second derivative test tells us if $w^*$ is a local
minimum $f'(w^*) = 0 \land f''(w^*) > 0$ or maximum $f'(w^*) = 0 \land f''(w^*)
< 0$. If we follow the tangent line of the curve, which we can find via its
derivative (or a _gradient_ in a multi-variable function), we arrive at a local
minimum from the starting point.

The _partial derivative_ of $f$ with respect to $w_i$ is the derivative obtained
by varying $w_i$ only, keeping all the other variables fixed ($e_i$ is the
$i$-th unit vector):

$$\frac{\partial f}{\partial w_i} (w) = \lim_{h\rightarrow 0} \frac{f(w_1, ...,
w_i + h, ..., w_D) - f(w)}{h} = \lim_{h \rightarrow 0} \frac{f(w+he_i) -
f(w)}{h}$$

For exmple, let $f(w_1, w_2) = w_1^2 + w_1w_2 + w_2^2 - 3w_1$, then treating
the other variables as a constant:

$$\frac{\partial f}{\partial w_1} = 2w_1 + w_2 - 3$$

$$\frac{\partial f}{\partial w_2} = w_1 + 2 w_2$$

The _gradient_ of $f$ at $w$ is the vector of its partial derivatives"

$$\nabla f(w) = (\frac{\partial f}{\partial w_1} (w), ..., \frac{\partial f}{\partial w_D} (w_D))^T$$

The rate of change of $f$ along a unit direction $u$ (also called the _symplex_
direction) is the _directional derivative$ $\nabla f(w)^T u = ||\nabla
f(w)||\cos \theta$. The gradient points in the direction of steepest descent.
$\nabla f(w)$ is orthogonal to the contour line (level set) of $f$ through $w$.

For $f(w_1, w_2) = w_1^2 + w_1w_2 + w_2^2 - 3w_1$, setting the gradient to zero
gives a system of equations:

$$
\begin{cases}
2 w_1 + w_2 = 3 \\
w_1 + 2 w_2 = 0
\end{cases}
$$

In general, for a quadratic $f$ the stationary point solves a _linear_ system.
For most loss functions, $\nabla f=0$ has no closed form (we don't know it
exactly), so we need an iterative technique.

The _Hessian_ of $f$ is the $D \times D$ matrix of its second partial
derivatives, $H_{ij} = \frac{\partial^2 f}{\partial w_i \partial w_j}$
(symmetric for smooth $f$), where $D$ is the number of dimensions i.e. the
number of variables accepted by $f$.

At a stationary point $w^*$, generalising the test on $f''$:

- all eigenvalues of $H$ positive ($H$ positive definite) implies a local
  _minimum_

- all eigenvalues negative implies a local _maximum_

- eigenvalues of both signs implies a _saddle point_ (a minimum along some
  directions, a maximum along others)

Given data $\{(x_n, t_n)\}_{n=1}^N$, fit the line $y(x) = w_0 + w_1 x$ by
minimising:

$$E(w_0, w_1) = \frac{1}{2} \sum_{n=1}^N (w_0 + w_1 x_n - t_n)^2$$

We can find the partial derivatives of this and create a system of equations:

$$\frac{\partial E}{\partial w_0} = \sum_{n=1}^N (w_0 + w_1 x_n - t_n)$$

$$\frac{\partial E}{\partial w_1} = \sum_{n=1}^N (w_0 + w_1 x_n - t_n)x_n$$

To walk downhill, we repeatedly take a small step in the direction of steepest
descent, $- \nabla f$.

$$w_{t+1} = w_t - \eta \nabla f(w_i)$$

where $\eta > 0$ is the _learning rate_. The bigger the learning rate, the
bigger the jumps. The iterative loop can stop when some criterios is met,
$||\nabla f(w_t)|| < \epsilon$.

If we don't have a closed form of the error function, we can estimate the
derivative without using the limit:

$$f'(w) \approx \frac{f(w+h) - f(w)}{h}$$


### Polynomial Curve Fitting

Given a function $f$, the derivative $f'(x)$ tells us the slope of $f$ at $x$,
that is, the direction of steepest increase. For a multivariate function $f(w)$,
the gradient $\nabla f(w)$ is the vector of partial derivatives. At a minimum,
$\nabla f(w^*)=0$.

Given data points $\{(x_n, t_n)\}_{n=1}^N$, fit a polynomial of order $M$:

$$y(x, w) = w_0 + w_1 x + w_2 x^2 + ... + w_M x^M = \sum_{j=0}^M w_j x^j$$

Error function (sum of squares):

$$E(w) = \frac{1}{2}\sum_{n=1}^N (y(x_n, w) - t_n)^2$$

To minimize this error function we can use the gradient descent approach, where:

$$w_{t+1} = w_t - \delta \nabla f(w_t)$

$\delta$ is the learning rate. Iterating this finds $w$ that minimise $E(w)$.

## Probability Theory

In this section I will summarize the core ideas of probability theory.

### Basic Definitions

There are two main interpretations of what probability is. The first one is to
think of a probability as the frequency of a certain event occurring. If a coin
has 0.5 probability of landing heads, then we expect the coin to land heads
about half of the time. The other interpretation views probability as a quantity
of uncertainty or ignorance about something, this is more related to information
rather than repeated trials and is more general. In the coin example, it can be
interpreted so that the coin is equally likely to land heads or tails on the
next toss. This interpretation is usually referred to as the Bayesian view.

Generally, we say that _probability_ is the study of likelihood of events at
priori, while _statistic_ is the study of their likelihood at posteriori.
Probability gives us a principled language for uncertainty in both data and
model parameters.

We define the _sample space_ $\Omega$ as the set of all possible outcomes of an
experiment. A $\sigma$-algebra $\Sigma \subseteq 2^{\Omega}$ is a subset of the
sample space such that:

1. $\Sigma$ contains the sample space: $\Omega \in \Sigma$
2. $\Sigma$ is closed under _complement_: if $A \in \Sigma$ then $(\Omega
   \setminus A) \in \Sigma$. It follows that the empty set is in the sigma
   algebra.
3. $\Sigma$ is closed under _countable unions_: if $A_1, A_2, ... \in \Sigma$
   then $\bigcup_i A_i \in \Sigma$

Suppose we want to toss a count. On $\Omega_C = \{H, T\}$ the $\sigma$-algebra
$\Sigma_C$ contains: $\emptyset$ (neither head nor tail), $\{H\}, \{T\}, \{H,
T\}$.

### Probability measure

Let $\Sigma \subseteq 2^{\Omega}$ be a $\sigma$-algebra. A _probability measure_
is a function $P_r: \Sigma \rightarrow [0, 1]$ such that:

- $P$ is non negative: $P_r(A)\ge 0\ \forall A\in \Omega$
- $P$ is normalized: $P_r(\Omega)=1$
- $P$ is $\sigma$ -additive: if $A_i \cap A_j \ne \emptyset,\ A\in \Omega$ then
  $P_r(\cup_i A_i)=\sum_i P_r(A_i)$

A _probability space_ is a tripe $(\Omega, \Sigma, P_r)$ where:

- $\Omega$ is a smaple space

- $\Sigma$ is a $\sigma$-algebra

- $P_r$ is a probability function on $\Sigma$

Let $(\Omega, \Sigma, P_r)$ be a probability space. The following properties
hold:

- For any $A \in \sigma$, $P_r(\Omega \setminus A) = 1 - P_r(A)$

- From set theory, you can demonstrate that the following holds: $A, B \in
  \sigma$, $P_r(A \cup B) = P_r(A) + P_r(B) - P_r(A \cap B)$

- In particular, $P_r(\emptyset ) = 0 \land A \subseteq B \Rightarrow P_r(A) \le
  P_r(B)$

Let $A, B \in \Sigma$ with $P_r(B) > 0$. The conditional probability of $A$
occurring, given that $B$ has occurred, is:

$$P_r(A|B) = \frac{P_r(A \cap B)}{P_r(B)}$$

It follows that:

- if $A$ and $B$ are _mutually exclusive_ $(A \cap B = \emptyset)$, then
  $P_r(A|B) = 0$

- if $B$ is a _precondition_ for $A (A \subseteq B)$, then $P_r(A \cap B) -
  P_r(A)$, so $P_r(A|B) = P_r(A) / P_r(B)$

- Two events are _independent_ if knowing that one occurred tells us nothing
  about hte probability of the other, i.e. $P_r(A|B) = P_r(A)$, or equivalently:

  $$P_r(A \cap B) = P_r(A) \times P_r(B)$$

### The two rules of probability

There are two very important rules in probability:

- the sum rule:

$$p(X) = \sum_Y p(X, Y)$$

- the product rule:

$$p(X,Y) = p(Y|X)p(X)$$

They are not hard to demonstrate, if you are curious an explanation is provided
in chapter 1.2 of Bishop's "Pattern Recognition and Machine Learning" book
(2006).

### Bayes Theorem

Let $P(A|H)$ be the probability of A given H, then, from the product rule, it
follows that:

  $$P(A|H)=\frac{P(A\cap H)}{P(H)},\ P(H)>0$$

The _Bayes theorem_ states:

  $$P(A_i|B)=\frac{P(B|A_i)P(A_i)}{\sum_j P(B|A_j)P(A_j)}=\frac{P(B|A_i)P(A_i)}{P(B)}$$

In the Bayesian interpretation of probability, we could state this theorem as: a
_posterior_ observation is directly proportional to the _likelihood_ times the
_prior_ observation, where:

- $p(A_i)$ is the _prior_: the belief of $A_i$, before seeing B

- $p(B|A_i)$ is the _likelihood_: how probably the B is, given $A_i$

- $p(A_i|B)$ is the _posterior_: updated belief after seeing B

- $p(B)$ is a normalizing constant (evidence)

The denominator is computed with the _law of total probability_.

### Stochastic Independence

Events $A\in \Omega$ are said to be independent if:

$$P_r(A_{i1} \cap A_{i2} \cap ... \cap A_{ik}) = \prod_{j=1}^k P_r(A_{ij}) = P_r(A_{i1})\cdot P_r(A_{i2})\cdot ...\cdot P_r(A_{ik})$$

The same applies to bivaraite functions:

$$P_{X, Y}(x, y) = P_X(x)\cdot P_Y(y),\ \forall (x, y)\in R_x \times R_y$$

moreover:

$$P_{X|Y}(x|y) = P_X(x)$$
$$P_{Y|X}(y|x) = P_Y(y)$$

### Random Variable

Random variables are a mathematical formalization used to model quantities which
depend on random events, it lets us quantify random events so that we can make
probability calculations.

More formally, a random variable is a function that maps event in some sample
space $\Omega$ to a set of outcomes in measurable space $E$, which is often
$\mathbb{R}$.

$$X:\Omega \to E$$

The probability that $X$ takes on a value in a measurable set $S\subseteq E$ is
written as

$$P_r(X\in S) = P_r(\{ \omega \in \Omega \ |\ X(\omega)\in S \})$$

### Distribution Function

Let $X$ be a random variable on the probability space $(\Omega, \Sigma, P_r)$.
A function $F: \mathbb{R}\to [0, 1]$ is a _cumulative distribution function_
(CDF) if:

- $F$ never decreases
- $F$ is right-continuous
- $F$ always has a left-limit
- $\lim_{x\to - \infty} F(x)=0$
- $\lim_{x\to\infty} F(x)=1$

For a random variable $X$, its CDF is defined by:

$$F_x(x) = P_r(X \le x) = P_r(\{\omega | X(\omega) \le x \})$$

The distribution function describes the probability that $X$ takes a value less
than or equal to $x$. It is defined for every real-values random variable,
whether discrete or continuous.

For every $x \in \mathbb{R}$, we can consider the probability that $X$ takes
exactly the value $x$:

$$p_x(x) = P_r(X = x) = P_r(\{\omega | X(\omega ) = x\})$$

This is the _point probability_ of $X = x$. Its interpretation depends on the
type of random variable.

### Continuous and Discrete Random Variables

Now we need to discriminate two different cases of random variables, since they
behave differently in the continuous and in the discrete case. A random variable
$X$ is _continuous_ if $p_x(x) = 0$ for all real $x$, in other words: it does
not have point masses. A random variable is _discrete_ if it takes values in a
finite or countable set $S \subseteq \mathbb{R}$.

#### Discrete case

Let $X$ be a discrete random variable, then the _probability mass function_
(PMF) $p_X(x) = P_r(X=x), x\in \mathbb{R}$ satisfies the following two
conditions:

$$p_X(x) \ge 0$$

$$\sum_x p_X(x) = 1$$

The probability of an event involving a set $A \subseteq \mathbb{R}$ can be
calculated by summing the masses of the values in that set:

$$P_r(X\in A) = \sum_{x\in A} p_X(x)$$

#### Continuous Case

Let $X$ be a continuous random variable, the _probability density function_
$f_X(x) = \frac{dF_X(x)}{dx}$ (PDF) satisfies:

$$f_X(x) \ge 0$$

$$\int_{-\infty}^{\infty} f(x) dx = 1$$

$$P_r(a\le X \le b) = \int_a^b f(x) dx$$

A continuous random variable can assume infinitely (uncountably) many
values, so it is reasonable that the probability of it assuming any specific
value chosen is zero:

$$P_r(X = x) = 0, \qquad \forall x\in \mathbb{R}$$

Consequently, including or excluding the endpoints of an interval does not
change its probability. Note that the density $f_X(x)$ is not the probability
$P_r(X=x)$: it describes how probability is distributed over the real line, and
probabilities are obtained by integration.

Random variables allow us use standard operations on functions to model the
randomness of a system. In practice we work directly with the distribution (pmf
or density) of a random variable.

### Notable Random Variables

Some random variables appear more than others, so a few of them are worthy of
their name. You will see those everywhere in nature, economics, populations, and
more.

- Bernoulli:

$$X(\omega)=\{0, 1\}$$

- Rademacher:

$$Y(\omega)=\{ -1, 1 \}$$

- Binomial $X\sim Bin(n, p)$:

$$P_x(J)=\{ \binom{n}{k}p^J(1-p)^{n-J},\ j=1...n\ |\ 0\ otherwise \}$$

- Poissont $X\sim Pois(\lambda)$:

$$P_x\{\frac{\lambda^xe^{-\lambda}}{x!}, n\in \mathbb{N}\cup \{ 0 \}\ |\ 0\ otherwise\}$$

- Geometric:

$$P(y)=\{ p(1-p)^{y-1},\ y\in \mathbb{N}\ |\ 0\ otherwise \}$$

- Uniform $X\sim Unif[a, b]$:

$$f_x(x)=\{ \frac{1}{b-a},\ x\in [a, b]\ |\ 0\ otherwise \}$$

- Normal (Gaussian) $X\sim N(\mu , \sigma^2)$:

$$f_x(x) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}}$$

- Exponential $X\sim Exp(\lambda)$:

$$f_x(x)=\lambda e^{\lambda x}\mathbb{1}(x>0)$$

### Expected Value

Let $X$ be a discrete random variable taking values in $S$, with probability
mass function $p(\cdot)$. The _mean_ or _expected value_ is defined as:

$$\mu = \mathbb{E}[x] = \sum_{x\in S} x\cdot p_X(x)$$

If $X$ is a continuous random variable with density function $f_X(x) = \frac{dF(x)}{dx}$, the expected value of $X$ is:

$$\mu = \mathbb{E}[x] = \int_{-\infty}^{\infty} x\cdot f_x(x) dx $$

The expectation only gives us an idea of the average value assumed by a random
variable, not of how much individual values may differ from this average.

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

We define the _variance_ $\mathbb{V}ar(x)$ as:

$$\mathbb{V}ar[x]=\mathbb{E}[x^2]-\mathbb{E}[x]^2 = \mathbb{E}[(x-\mathbb{E}[x])^2]$$

This indicates the "spread" of values. Moreover:

$$\mathbb{V}ar[x]=\mathbb{E}(\mathbb{V}ar[x|y]) + \mathbb{V}ar(\mathbb{E}[x|y])$$


The _standard deviation_ is $\sigma = \sqrt{var(X)}$, and has the same units as
$X$.

### Covariance

We define the covariance as:

$$\mathbb{C}ov[x, y]=\mathbb{E}((x-\mathbb{E}[x])(y-\mathbb{E}[y]))=\mathbb{E}[XY]-\mathbb{E}[x]\mathbb{E}[y]$$

### Standardization

$$z=g(x)=\frac{x-\mathbb{E}[x]}{\sqrt{\mathbb{V}ar[x]}}$$

After this transformation:

- $\mathbb{E}[z]=0$
- $\mathbb{V}ar[z]=1$

The opposite can be achieved:

$$x=\sigma z + \mu$$

### Markov Inequality

Let $Y$ be a random variable non negative, then $\forall a>0$:

$$P(Y\ge a)\le \frac{\mathbb{E}[y]}{a}$$

### Chebyshev Inequality

Let $Y$ be a random variable, $\mu = \mathbb{E}[y]$,
$\sigma^2=\mathbb{V}ar[y]$, then $\forall \epsilon > 0$:

$$P(|Y-\mu| \ge \epsilon)\le \frac{\sigma^2}{\epsilon^2}$$

### Some distributions

#### Bernoulli Distribution

Let us consider a random variable $x$ assuming values in $\{0, 1\}$. The
probability of $x=1$ is denoted by a parameter $\mu \in [0,1]$, so that
$p(x=1|\mu ) = \mu$.

The _Bernoulli distribution_ over $x$ can be written in the form:

$$Bern(x | \mu ) = \mu^x (1 - \mu)^{1-x}$$

$$\mathbb{E}[Bern(x|\mu )] = \mu$$

$$var[Bern(x|\mu )] = \mu (1-\mu)$$

Let $D=\{x_1, ..., x_N\}$ be a set of independent observed values of $x$ drawn
from $p(x|\mu )$. The likelihood of $D$ is:

$$p(D|\mu ) = \prod_{i=1}^N p(x_i | \mu) = \prod_{i=1}^N \mu^{x_i}(1-\mu)^{1-x_i}$$

Let $D=\{x_1, ..., x_N\}$ be independent observations of variable $x$ we assume
Bernoulli-distributed with unknown parameter $\mu$.

How can we estimate the parameter $\mu$?

We maximise the likelihood function $f(\mu) = p(D|\mu)$, or equivalently its
logarithm:

$$\ln p(D|\mu) = \sum_{i=1}^N \ln p(x_i|\mu )$$

$$\ln p(D|\mu) = (\ln \mu) \sum_{i=1}^N x_i + \ln (1-\mu )\sum_{i=1}^N (1-x_i)$$

We derive this and set it equal to 0. We obtain the maximum likelihood
estimator:

$$\mu_{ML}=\frac{1}{N}\sum_{n=1}^N x_n$$

which is also known as the _sample mean_. If we denote the number of
observations of $x=1$ with $m$, then we can write this in the form:

$$\mu = \frac{m}{N}$$

Note that this only works for a high value of $N$.

#### Binomial distribution

We can also work out the distribution of the number $m$ of observations of
$x=1$, given that the data set has size $N$. This is called the _Binomial
distribution_:

$$Bin(m|N, \mu) = \binom{N}{m} \mu^m(1-\mu)^{N-m}$$

$$\binom{N}{m} \equiv \frac{N!}{(N-m)!m!}$$

We also have the following:

$$\mathbb{E}[m] = N \mu$$

$$\mathbb{V}ar[m] = N \mu (1-\mu )$$

#### Beta distributions

Let the _gamma function_ $\Gamma$ be:

$$\Gamma (x) = \int_0^{\infty} u^{x-1}e^{-u}du$$

With the following properties:

- $\Gamma (1) = 1$

- $\Gamma (x+1) = x \Gamma (x)$

Note that this behaves like a factorial, so it can be thought as the continuous
version of the factorial.

Beta distribution:

$$Beta(\mu | a, b) = \frac{\Gamma (a+b)}{\Gamma(a) \Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$

$$\int_0^1 Beta(\mu | a, b) d\mu = 1$$

In some sense, this distribution generalizes the binomial distribution.

$$\mathbb{E}[\mu] = \frac{a}{a+b}$$

$$var[\mu] = \frac{ab}{(a+b)^2 (a+b+1)}$$

The parameters $a$ and $b$ are often called _hyperparameters_ because they
control the distribution of the parameter $\mu$.

The develop a Bayesian treatment of parameter esitmation for a Bernoulli
distribution, we introduce a prior distribution $p(\mu)$ over $\mu$. We assume
$\mu$ to be Beta-distributed, with given hyperparameters $a$ and $b$: $p(\mu) =
Beta(\mu | a, b)$. Let $m$ and $l$ be the numbers of positive and negative
results, respectively $(N = m + l)$. Then:

$$p(\mu | m, l) \propto p(m, l | \mu ) p(\mu) \propto \mu^{m + a - 1} (1-\mu)^{l+b-1}$$

The posterior has the same form as the prior: it is again a Beta distribution:

$$p(\mu | m, l, a, b) = \frac{\Gamma (m+a+l+b)}{\Gamma (m+a) \Gamma (l+b)}\mu^{m+a-1} (1-\mu)^{l+b-1}$$

Our goal is to predict the outcome of the next trial, given the observed data
set $D$:

$$p(x=1 | D) = \int_0^1 p(x=1 | \mu) p(\mu | D) d\mu = \int_0^1 \mu p(\mu | D)
d\mu = \mathbb{E}[\mu | D]$$

$$p(x=1 | D) = \frac{m + a}{m + a + l + b}$$

When $m, l \rightarrow \infty,\ p(x=1 | D)$ reduces to the maximum likelihood
estimate $m/N$ (the sample mean)!

