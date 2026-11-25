
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

### Basic Combinatorics

Combinatorics is the branch of mathematics concerned with counting
the number of possible outcomes of a finite process. In probability,
counting techniques are particularly useful when outcomes are equally
likely.

#### Fundamental Counting Principle

If a procedure consists of $k$ successive steps, and step $i$ can
be performed in $n_i$ ways, then the total number of possible
outcomes is

$$N = \prod_{i=1}^{k} n_i$$

In particular, if $n$ distinct objects are arranged in a sequence, there are
$n!$ possible arrangements, where

$$n! = n(n-1)\cdots 2\cdot 1, \\ \qquad 0! = 1$$

#### Permutations

A permutation is an ordered arrangement of objects. The order in which the
objects appear matters.

Considering the permutations of $n$ distinct objects, the number of ways to
arrange all $n$ objects is

$$P_n = n!$$

Considering now the permutations of $n$ distinct objects taken $r$ at a time. If
we select and arrange $r$ objects from $n$ distinct objects, without repetition,
the number of possible arrangements is

$$
    P(n,r) = \frac{n!}{(n-r)!}, \\
    \qquad 0 \leq r \leq n.
$$

The first position has $n$ choices, the second has $n-1$, and so on, giving

$$P(n,r) = n(n-1)\cdots(n-r+1)$$

#### Combinations

A combination is a selection of objects in which order does not matter. We are
interested only in which objects are selected, not in how they are arranged.

The number of ways to select $r$ objects from $n$ distinct objects, without
repetition, is the binomial coefficient

$$
    \binom{n}{r} = \frac{n!}{r!(n-r)!}, \\
    \qquad 0 \leq r \leq n.
$$

To derive this formula, first arrange the $r$ selected objects:
there are $P(n,r)$ arrangements. Since each selection corresponds
to exactly $r!$ arrangements, we divide by $r!$:

$$
    \binom{n}{r} \\
    = \frac{P(n,r)}{r!} \\
    = \frac{n!}{r!(n-r)!}.
$$

Useful identities include:

$$
    \binom{n}{0} = \binom{n}{n} = 1,
    \qquad
    \binom{n}{r} = \binom{n}{n-r},
$$

and Pascal's identity:

$$
    \binom{n}{r}
    = \binom{n-1}{r-1} + \binom{n-1}{r}.
$$

#### Selections with Repetition

The counting formula depends on whether repetition is allowed and
whether order matters.

If each of $r$ positions can be filled by any of $n$ objects,
with repetition allowed, then

$$n^r$$

The number of ways to select $r$ objects from $n$ types, where
each type may be selected multiple times, is

$$\binom{n+r-1}{r}$$

for $n \geq 1$ and $r \geq 0$.

This is the classical _stars and bars_ result.

## Linear Algebra

Linear algebra is the study of vectors and certain rules to manipulate vectors.

### Groups

Consider a set $\mathcal{G}$ and an operation $\otimes : \mathcal{G} \times
\mathcal{G} \to \mathcal{G}$ defined on $\mathcal{G}$. Then $G := (\mathcal{G},
\otimes)$ is called a group if the following hold:

- Closure of $\mathcal{G}$ under $\otimes$:

$$\forall x,y \in \mathcal{G} : x \otimes y \in \mathcal{G}$$

- Associativity:

$$\forall x,y,z \in \mathcal{G} : (x \otimes y) \otimes z = x \otimes (y \otimes z)$$

- Neutral element:

$$\exists e \in \mathcal{G}\ \forall x \in \mathcal{G} : x \otimes e = x \text{ and } e \otimes x = x$$

- Inverse element:

$$\forall x \in \mathcal{G}\ \exists y \in \mathcal{G} : x \otimes y = e \text{ and } y \otimes x = e,$$

  where $e$ is the neutral element. We often write $x^{-1}$ to denote the inverse element of $x$.

### Vector Space

A real-valued vector space $V = (\mathcal{V}, +, \cdot)$ is a set $\mathcal{V}$ with two operations

$$+: \mathcal{V} \times \mathcal{V} \to \mathcal{V}$$
$$\cdot : \mathbb{R} \times \mathcal{V} \to \mathcal{V}$$

where

- $(\mathcal{V}, +)$ is an Abelian group.

- Distributivity:

  - $\forall \lambda \in \mathbb{R}, x,y \in \mathcal{V} : \lambda \cdot (x+y) =
    \lambda \cdot x + \lambda \cdot y$

  - $\forall \lambda,\psi \in \mathbb{R}, x \in \mathcal{V} : (\lambda+\psi)
    \cdot x = \lambda \cdot x + \psi \cdot x$

- Associativity (outer operation): $\forall \lambda,\psi \in \mathbb{R}, x \in
  \mathcal{V} : \lambda \cdot (\psi \cdot x) = (\lambda\psi) \cdot x$

- Neutral element with respect to the outer operation: $\forall x \in
  \mathcal{V} : 1 \cdot x = x$

The elements $x \in V$ are called vectors. The neutral element of $(\mathcal{V},
+)$ is the zero vector $0 = [0,\ldots,0]^\top$, and the inner operation $+$ is
called vector addition. The elements $\lambda \in \mathbb{R}$ are called scalars
and the outer operation $\cdot$ is a multiplication by scalars.

### Eigenvalues and Eigenvectors

Let's look at Eigenvalues and Eigenvectors now. I will use the great book
"Algebra Lineare" by Marco Abate as my reference. Eigenvectors are an important
theoretical and practical topic as they are used for some tricks in unsupervised
learning, and are often used as the preferred axis for rotations or other
transformations.

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
direction) is the _directional derivative_ $\nabla f(w)^T u = ||\nabla
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

$$w_{t+1} = w_t - \delta \nabla f(w_t)$$

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

#### Sample space

We define the _sample space_ $\Omega$ as the set of all possible outcomes of an
experiment.

#### Sigma algebra

A $\sigma$-algebra $\Sigma \subseteq 2^{\Omega}$ is a subset of the sample space
such that:

1. $\Sigma$ contains the sample space: $\Omega \in \Sigma$
2. $\Sigma$ is closed under _complement_: if $A \in \Sigma$ then $(\Omega
   \setminus A) \in \Sigma$. It follows that the empty set is in the sigma
   algebra.
3. $\Sigma$ is closed under _countable unions_: if $A_1, A_2, ... \in \Sigma$
   then $\bigcup_i A_i \in \Sigma$

Suppose we want to toss a coin. On $\Omega_C = \{H, T\}$ the $\sigma$-algebra
$\Sigma_C$ contains: $\emptyset$ (neither head nor tail), $\{H\}, \{T\}, \{H,
T\}$.

#### Probability measure

Let $\Sigma \subseteq 2^{\Omega}$ be a $\sigma$-algebra. A _probability measure_
is a function $P_r: \Sigma \rightarrow [0, 1]$ such that:

- $P$ is non negative: $P_r(A)\ge 0\ \forall A\in \Omega$
- $P$ is normalized: $P_r(\Omega)=1$
- $P$ is $\sigma$ -additive: if $A_i \cap A_j = \emptyset,\ A\in \Omega$ then
  $P_r(\cup_i A_i)=\sum_i P_r(A_i)$

#### Probability space

A _probability space_ is a tripe $(\Omega, \Sigma, P_r)$ where:

- $\Omega$ is a smaple space

- $\Sigma$ is a $\sigma$-algebra

- $P_r$ is a probability function on $\Sigma$

Let $(\Omega, \Sigma, P_r)$ be a probability space. The following properties
hold:

- For any $A \in \Sigma$, $P_r(\Omega \setminus A) = 1 - P_r(A)$

- From set theory, you can demonstrate that the following holds: $A, B \in
  \Sigma$, $P_r(A \cup B) = P_r(A) + P_r(B) - P_r(A \cap B)$

- In particular, $P_r(\emptyset ) = 0 \land A \subseteq B \Rightarrow P_r(A) \le
  P_r(B)$

Let $A, B \in \Sigma$ with $P_r(B) > 0$. The conditional probability of $A$
occurring, given that $B$ has occurred, is:

$$P_r(A|B) = \frac{P_r(A \cap B)}{P_r(B)}$$

It follows that:

- if $A$ and $B$ are _mutually exclusive_ $(A \cap B = \emptyset)$, then
  $P_r(A|B) = 0$

- if $B$ is a _precondition_ for $A$, meaning $(A \subseteq B)$, then $P_r(A
  \cap B) = P_r(A)$, so $P_r(A|B) = P_r(A) / P_r(B)$

- Two events are _independent_ if knowing that one occurred tells us nothing
  about hte probability of the other, i.e. $P_r(A|B) = P_r(A)$, or equivalently:

  $$P_r(A \cap B) = P_r(A) \times P_r(B)$$

### The two rules of probability

There are two very important rules in probability:

#### The sum rule

$$p(X) = \sum_Y p(X, Y)$$

#### The product rule

$$p(X,Y) = p(Y|X)p(X)$$

Where $p(X, Y)$ is a joint probability, meaning that it is the probability of
$X$ _and_ $Y$. $p(X|Y)$, instead, is a conditional probability, meaning the
probability of $X$ given $Y$.They are not hard to demonstrate, if you are
curious an explanation is provided in chapter 1.2 of Bishop's "Pattern
Recognition and Machine Learning" book (2006).

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

The denominator is computed with the _law of total probability_: $P(B) = \sum_i
P(B|A_i) P(A_i)$.

### Stochastic Independence

Events $A\in \Omega$ are said to be _independent_ if:

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

More formally, a random variable is a function that maps an event in a sample
space $\Omega$ to a set of outcomes in measurable space $E$, which is often
$\mathbb{R}$.

$$X:\Omega \to E$$

#### Probability of a random variable

The probability that $X$ takes on a value in a measurable set $S\subseteq E$ is
written as

$$P_r(X\in S) = P_r(\{ \omega \in \Omega \ |\ X(\omega)\in S \})$$

### Distribution Function

#### CDF

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

#### Discrete case (PMF)

Let $X$ be a discrete random variable, then the _probability mass function_
(PMF) $p_X: \mathbb{R} \rightarrow [0, 1]$ such that $p_X(x) = P_r(X=x), x\in
\mathbb{R}$ satisfies the following two conditions:

$$p_X(x) \ge 0$$

$$\sum_x p_X(x) = 1$$

The probability of an event involving a set $A \subseteq \mathbb{R}$ can be
calculated by summing the masses of the values in that set:

$$P_r(X\in A) = \sum_{x\in A} p_X(x)$$

#### Continuous Case (PDF)

Let $X$ be a continuous random variable, the _probability density function_
$f_X: \mathbb{R} \rightarrow [0, \infty]$ such that $f_X(x) =
\frac{dF_X(x)}{dx}$ (PDF) which satisfies:

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

- Bernoulli $X \sim Ber(p)$:

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

#### Discrete case

Let $X$ be a discrete random variable taking values in $S$, with probability
mass function $p(\cdot)$. The _mean_ or _expected value_ is defined as:

$$\mu = \mathbb{E}[x] = \sum_{x\in S} x\cdot p_X(x)$$

#### Continuous case

If $X$ is a continuous random variable with density function $f_X(x) =
\frac{dF(x)}{dx}$, the expected value of $X$ is:

$$\mu = \mathbb{E}[X] = \int_{-\infty}^{\infty} x\cdot f_x(x) dx $$

The expectation only gives us an idea of the average value assumed by a random
variable, not of how much individual values may differ from this average.

#### Properties

The expected value is a linear function:

$$E[aX+b] = a\mathbb{E}[x]+b$$
$$E[g(x)]=\sum_{x\in R_x} g(x)p_x(x)$$

#### Notable formulas

- Bernoulli: $\mathbb{E}[x]=p$
- Binomial: $\mathbb{E}[x]=np$
- Geometric: $\mathbb{E}[x]= \frac{1}{p}-1$
- Normal: $\mathbb{E}[x]=\mu$
- Exponential: $\mathbb{E}[x]=\frac{1}{\lambda}$
- Poisson: $\mathbb{E}[x]=\lambda$

### Variance

Let $X$ be a random variable on a probability space. We define the _variance_
$\mathbb{V}ar[X]$ as:

$$\mathbb{V}ar[X]=\mathbb{E}[X^2]-\mathbb{E}[X]^2 = \mathbb{E}[(X-\mathbb{E}[X])^2]$$

This indicates the "spread" of values. Moreover:

$$\mathbb{V}ar[X]=\mathbb{E}(\mathbb{V}ar[X|Y]) + \mathbb{V}ar(\mathbb{E}[X|Y])$$


The _standard deviation_ is $\sigma = \sqrt{\mathbb{V}ar[X]}$, and has the same
units as $X$.

### Covariance

Let $X$ and $Y$ be two random variables on a probability space. We define the
covariance as:

$$\mathbb{C}ov[X, Y]=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$$

### Standardization

Given a random variable $X$, we can standardize it with this formula:

$$Z=g(Z)=\frac{X-\mathbb{E}[X]}{\sqrt{\mathbb{V}ar[X]}}$$

After this transformation:

- $\mathbb{E}[Z]=0$
- $\mathbb{V}ar[Z]=1$

The inverse operation can be computed too:

$$X=\sigma Z + \mu$$

### Markov Inequality

Let $Y$ be a random variable non negative, then $\forall a>0$:

$$P(Y\ge a)\le \frac{\mathbb{E}[y]}{a}$$

### Chebyshev Inequality

Let $Y$ be a random variable, $\mu = \mathbb{E}[y]$,
$\sigma^2=\mathbb{V}ar[y]$, then $\forall \epsilon > 0$:

$$P(|Y-\mu| \ge \epsilon)\le \frac{\sigma^2}{\epsilon^2}$$

## Probability distributions

### Bernoulli Distribution

Let us consider a random variable $X$ with image $im(X) = \{0, 1\}$ and $x \in
im(X)$. The probability that $X=1$ is denoted by a parameter $\mu \in [0,1]$, so
that $P_X(X=1|\mu ) = \mu$.

The PMF of the _Bernoulli distribution_ over $x$ can be written in the form:

$$Bern(x | \mu ) = \mu^x (1 - \mu)^{1-x}$$

If $X \sim Bern(\mu)$, then:

$$\mathbb{E}[X] = \mu$$

$$\mathbb{V}ar[X] = \mu (1-\mu)$$

#### Likelihood

Let $X_1, ..., X_N \sim^{i.i.d.} Bern(\mu)$ with unknown parameter $\mu \in [0,
1]$, and let $x_i \in \{0, 1\}$ denote the observed realization of $X_i$ for $i
= 0, .., N$. For the dataset $D=\{x_1, ..., x_N\}$, the joint PMF (viewed as a
likelihood function of $\mu$) is:

$$L(\mu) = p_X(x_1,...,x_N|\mu ) = \prod_{i=1}^N p(x_i | \mu) = \prod_{i=1}^N \mu^{x_i}(1-\mu)^{1-x_i} = \mu^{m} (1-\mu)^{N-m}$$

where $m = \sum_{i=1}^N \mathbb{I}(x_i = 1)$ is the number of observations $x_i$
equal to 1.

Given $D$, how can we estimate the parameter $\mu$?

To do this, we maximise the likelihood function $L(\mu)$, or, equivalently (!!),
its logarithm:

$$\ln p_X(D|\mu) = \sum_{i=1}^N \ln p_X(x_i|\mu )$$

$$\ln p_X(D|\mu) = (\ln \mu) \sum_{i=1}^N x_i + \ln (1-\mu )\sum_{i=1}^N (1-x_i)$$

We derive this and set it equal to 0, note that $\frac{d \ln(f(x))}{dx} =
\frac{\frac{d f(x)}{dx}}{f(x)}$. We obtain the _maximum likelihood estimator_:

$$\mu_{ML}=\frac{1}{N}\sum_{n=1}^N x_n$$

which is also known as the _sample mean_. If we denote the number of
observations of $x=1$ with $m$, since $x \in \{0, 1\}$, we can write this in the
form:

$$\mu = \frac{m}{N}$$

Note that this only works for a high value of $N$.

### Binomial distribution

Suppose $X_1, ..., X_N \sim^{i.i.d.} Bern(\mu)$ with $\mu \in [0, 1]$, and
define the random variable $M = \sum_{i=1}^N X_i$ (the number of observations
equal to 1 in a sample of size $N$), with realization $m \in \{0, 1, ... N\}$.
The PMF of $M$ follows the _binomial distribution_:

$$Bin(m|N, \mu) = \binom{N}{m} \mu^m(1-\mu)^{N-m} \qquad \binom{N}{m} \equiv
\frac{N!}{(N-m)!m!}$$

Using the binomial theorem:

$$(x+y)^n = \sum_{k=0}^n \binom{n}{k} x^{n-k} y^k = \sum_{k=0}^n \binom{n}{k}
x^k y^{n-k}$$

and by using $m! = m (m-1)!$ and $m^2 = m(m - 1) + m$, we obtain:

$$\mathbb{E}[M] = N \mu \qquad \mathbb{V}ar[M] = N \mu (1-\mu )$$

### Beta distributions

Now we develop a Bayesian treatment of parameter estimation for a Bernoulli
distribution (the same technique can be used for other distributions, such as
binomial and geometric), meaning that we introduce a prior distribution $p(\mu)$
over $\mu$.

Let the _gamma function_ $\Gamma$ be:

$$\Gamma (x) = \int_0^{\infty} u^{x-1}e^{-u}du$$

With the following properties:

- $\Gamma (1) = 1$

- $\Gamma (x+1) = x \Gamma (x)$

The gamma function generalizes the factorial function from the positive integers
to $\mathbb{C} \backslash \{0, -1, -2, ...\}$, it can be thought as the
continuous version of the factorial in the complex numbers.


Let $U$ be a continuous random variable taking values $\mu \in (0, 1)$, and let
$a, b >0$ be shape parameters. We say $U \sim Beta(a, b)$ if its PDF is given
by:

$$Beta(\mu | a, b) = \frac{\Gamma (a+b)}{\Gamma(a) \Gamma(b)}\mu^{a-1}(1-\mu)^{b-1} \qquad \mu \in (0, 1)$$

and 0 otherwise, satisfying the normalization condition:

$$\int_0^1 Beta(\mu | a, b) d\mu = 1$$

For $U \sim Beta(a, b)$, the expectation and variance are:

$$\mathbb{E}[U] = \frac{a}{a+b}$$

$$\mathbb{V}ar[U] = \frac{ab}{(a+b)^2 (a+b+1)}$$

The parameters $a > 0$ and $b > 0$ are often called _hyperparameters_ because
they control the distribution of the parameter $\mu$. In some sense, they
represent our assumptions or our prior knowledge.

We assume $U$ to be Beta-distributed, with given hyperparameters $a$ and $b$ and
$\mu \in im(U)$: $p(\mu) = Beta(\mu | a, b)$. Let $m$ and $l$ be the numbers of
positive and negative results, respectively $(N = m + l)$. Then:

$$p(\mu | m, l) \propto p(m, l | \mu ) p(\mu) \propto \mu^{m + a - 1} (1-\mu)^{l+b-1}$$

The posterior has the same form as the prior: it is again a Beta distribution
(key fact about the beta distribution):

$$p(\mu | m, l, a, b) = \frac{\Gamma (m+a+l+b)}{\Gamma (m+a) \Gamma (l+b)}\mu^{m+a-1} (1-\mu)^{l+b-1}$$

Our goal is to predict the outcome of the next trial, given the observed data
set $D$:

$$P(X_{N+1}=1 | D) = \int_0^1 P(X_{N+1}=1 | \mu) p(\mu | D) d\mu = \int_0^1 \mu p(\mu | D)
d\mu = \mathbb{E}[U | D]$$

$$P(X_{N+1}=1 | D) = \frac{m + a}{m + a + l + b}$$

When $m, l \rightarrow \infty,\ P(X_{N+1}=1 | D)$ reduces to the maximum likelihood
estimate $m/N$ (the sample mean)!

### Multinomial distribution

Binary variables describe quantities that can take one of two possible values.
Often, however, we encounter discrete variables that can take one of $K$
possible mutually exlusive states. A particularly convenient representation is
the _1-of-K_ scheme: the random vector $X = (X_1, ..., X_K)^T$ takes values $x =
(x_1, ..., x_K)^T \in \{0, 1\}^K$ in which exactly one component $x_k = 1$ and
all others equal  $0$, satisfying $\sum_{k=1}^K x_k = 1$. So, for instance, if
we have a variable that can take $K=6$ states and a particular observation of
the variable happens to correspond to the state where $x_3 = 1$, then $x$ will
be represented by:

$$x=(0, 0, 1, 0, 0, 0)^T$$

If we denote the probability that $x_k = 1$ by the parameter $\mu_k$ for $k = 1,
..., K$, then the PMF of $X$ is:

$$p_X(x|\mu) = \prod_{k=1}^K \mu_k^{x_k}$$

where $\mu = (\mu_1, ..., \mu_k)^T$ satisfied $\mu_k \ge 0$ for all $K$, and
$\sum_k \mu_{k=1}^K = 1$.

This is a generalization of the Bernoulli distribution to more than two
outcomes. Let $X = (X_1, ..., X_K)^T$ be the random vector taking values $x =
(x_1, ..., x_K)^T \in \{0, 1\}$ with $\sum_{k=1}^K x_k = 1$, then:

$$\mathbb{E}[X|\ \mu] = \sum_x p_X(x|\ \mu) x = (\mu_1, ..., \mu_k)^T = \mu$$

Let $D$ be a data set of $N$ independent observations $x_1, ..., x_n$. The
corresponding likelihood function takes the form:

$$p_X(D|\ \mu) = \prod_{i=1}^N \prod_{k=1}^K \mu_k^{x_{nk}} = \prod_{k=1}^K
\mu_k^{\sum_{i=1}^N x_{ik}} = \prod_{k=1}^K \mu_k^{m_k}$$

we see that the likelihood function depends on the $N$ data point only through
the $K$ quantities:

$$m_k = \sum_n x_{nk}$$

which represent the number of observations of $x_k = 1$. These are called the
_sufficient statistics_ for this distribution.

To find the maximum likelihood solution for $\mu$, we maximize $\ln p_X(D|\mu)$
with respect to the components $\mu_k$ of $\mu$, subject to the constraint
$\sum_{k=1}^K \mu_k = 1$.

This can be achieved using a Lagrange multiplier $\lambda$, maximizing:

$$\sum_{k=1}^K m_k \ln \mu_k + \lambda (\sum_{k=1}^K \mu_k - 1)$$

Setting the derivative with respect to $\mu_k$ to zero, we obtain:

$$\mu_k = -m_k / \lambda$$

We can solve for the Lagrange multiplier $\lambda$ by substituting this into the
constraint $\sum_k \mu_k = 1$ to give $\lambda = - N$. Thus we obtain the
maximum likelihood solution in the form:

$$\mu_k^{ML} = \frac{m_k}{N}$$

which is the fraction of the $N$ observations for which $x_k = 1$.

We can consider the joint distribution of the quantities $m_1, ..., m_K$,
conditioned on the parameters $\mu$ and on the total number $N$ of observations.
So we have this PMF:

$$Mult(m_1, m_2, ..., m_K | \mu, N) = \binom{N}{m_1 m_2 ... m_k} \prod_{k=1}^K
\mu_k^{m_k}$$

which is known as the _multinomial_ distribution. The normalization coefficient
is the number of ways of partitioning $N$ objects into $K$ groups of size $m_1,
..., m_K$ and is given by:


$$\binom{N}{m_1 m_2 ... m_K} = \frac{N!}{m_1! m_2! ... m_K!}$$

Note that the variables $m_k$ are subject to the constraint:

$$\sum_{k=1}^K m_k = N$$

An example event that can be modeled in this way is the face of a dice.

### Dirichlet Distribution

We now introduce a family of prior distributions of the parameters $\{ \mu_k \}$
of the multinomial distribution. By inspection of the likelihood, the _conjugate
prior_ has the form:

$$p(\mu| \alpha) \propto \prod_{k=1}^K \mu_k^{\alpha_k - 1}$$

where $0 \le \mu_k \le 1$ and $\sum_k \mu_k = 1$. Here $\alpha_1, ..., \alpha_K$
are the parameters of the distribution, and $\alpha$ denotes $(\alpha_1, ...,
\alpha_K)^T$.

Because of the constraints, the distribution over the space of the $\{\mu_k\}$
is confined to a simplex of dimensionality $K-1$. The normalized form of the
distribution is:

$$Dir(\mu| \alpha) = \frac{\Gamma (\alpha_0)}{\Gamma (\alpha_1) \cdot \cdot
\cdot \Gamma (\alpha_K)} \prod_{k=1}^K \mu_k^{a_k - 1}$$

which is called the _Dirichlet_ distribution. Where $\alpha_0 = \sum_{k=1}^K
\alpha_k$

Multiplying the prior by the likelihood function, we obtain the posterior
distribution for the parameters $\{\mu_k \}$ in the form:

$$p(\mu| D, \alpha) \propto p(D|\mu) p(\mu | \alpha) \propto \prod_{k=1}^K
\mu_k^{a_k + m_k - 1}$$

The posterior preserved the form of a Dirichlet: the Diricklet is a conjugate
prior of the multinomial. This allows us to determine the normalization
coefficient so that:

$$p (\mu | D, \alpha) = Dir(\mu | \alpha + m) = \frac{\Gamma (\alpha_0 +
N)}{\Gamma (\alpha_1 + m_1) \cdot \cdot \cdot \Gamma(\alpha_K + m_K)} \prod_{k=1}^K \mu_k^{a_k + m_k - 1}$$

where we have denoted $m = (m_1, ..., m_K)^T$. As for the case of the binomial
distribution with its beta prior, we can interpret the parameters $\alpha_k$ of
the Dirichlet prior as an effective number of observations of $x_k = 1$.

### Gaussian Distribution

$$N(x| \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

Fully described by mean $\mu$ and variance $\sigma^2$

The _central limit theorem_ states that sums of many independent effects tend to
a Gaussian. For example, we assume Gaussian-distributed noise in linear
regression: $t = y(x, w) + \epsilon,\ \epsilon \sim N(0, \sigma^2)$.
Graphically, this looks like a bell curve, which is symmetric about $\mu$ and
its peak is $\frac{1}{\sqrt{2\pi}}$ at $x=0$.

#### Properties

We have a lot of interesting properties. Let $X\sim N(\mu, \sigma^2)$.

- Normalisation: from the Gaussian integral $\int_{-\infty}^\infty e^{-y^2/2}dy
  = \sqrt{2\pi}$

$$\int_{-\infty}^{\infty } N(x | \mu, \sigma^2 ) dx = 1$$

- Mean: with $y=x - \mu$, the term $y N(y | 0, \sigma^2 )$ is an odd function,
  so its integral is 0:

  $$\mathbb{E}[X] = \int_{-\infty}^{\infty} (y + \mu ) N(y|0, \sigma^2 ) dy = 0
  + \mu = \mu$$

- Variance: $\mathbb{V}ar[X] = \mathbb{E}[(X - \mu)^2] = \sigma^2$, and the
  _precision_ is $\beta = 1 / \sigma^2$

- Mean = median = mode = $\mu$ by symmetry

- Linear transformations: if $X\sim N(\mu, \sigma^2)$ and $a\ne 0$, then:

$$aX + b \sim N(a\mu + b, a^2 \sigma^2)$$

- Standardisation: as a special case, the _standard normal_ distribution is:

$$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$

- Sums: if $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$
  are independent, then:

$$X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma^2_1 + \sigma_2^2)$$

- Averages: the mean of $N$ independent copies of $X$ satisfied $\bar{X} \sim
  N(\mu , \sigma^2 / N)$: averaging reduces the spread by a factor of $\sqrt{N}$

For a continuous variable, probabilities are _areas under the density_. With the
distribution function $F(x) = Pr(X\le x)$:

$$Pr(a\le X \le b) = \int_a^b N(x|\mu, \sigma^2) dx = F(b) - F(a)$$

There is a problem: $e^{-y^2}$ has no closed form (we cannot calculate it
analytically), so we use table values.

#### The standard normal distribution function

We only need the distribution function of $Z\sim N(0, 1)$, computed once and for
all:

$$\Phi = Pr(Z\le z) = \int_{-\infty}^z \frac{1}{\sqrt{2\pi}} e^{-t^2 / 2} dt$$

Since $X \le x$ if and only if $Z = \frac{X - \mu }{\sigma } \le \frac{x - \mu}{\sigma}$, for any $X \sim N(\mu, \sigma^2)$:

$$F(x) = \Phi (\frac{x- \mu}{\sigma}) \Rightarrow Pr(a \le X \le b) =
\Phi(\frac{b - \mu}{\sigma}) - \Phi (\frac{a - \mu}{\sigma})$$

There are some useful identities of $\Phi$:

- $\Phi (0) = 0.5$

- $\Phi (-z) = 1 - \Phi(z)$

- $P_r(X > x) = 1 - \Phi(z)$

- $P_r(|X - \mu| \le k \sigma) = 2\Phi (x) - 1$

The inverse $\Phi^{-1}$ gives _quantiles_: the value $x$ such that $P_r(X\le x) =
q$ is:

$$x = \mu + \sigma \Phi^{-1} (q)$$

The $68-95-99.7$ rule:

- $P_r(|X - \mu| \le \sigma) \approx 0.683$

- $Pr(|X-\mu | \le 2 \sigma ) \approx 0.954$

- $Pr(|X-\mu | \le 3 \sigma ) \approx 0.997$

#### The Multivariate Gaussian Distribution

For a D-dimensional vector $x$, the multivariate Gaussian distribution takes the
form:

$$N(x|\mu, \Sigma) = \frac{1}{(2\pi)^{D/2}}
\frac{1}{|\Sigma|^{1/2}}exp(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu))$$

where $\mu$ is the D-dimensional mean vector, $\Sigma$ is the $D \times D$
convariance matrix and $|\Sigma|$ is its determinant.

Let $\Delta$ be:

$$\Delta^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)$$

The quantity $\Delta$ is called the _Mahalanobis distance_ form $\mu$ to $x$. It
reduces to the Euclidean distance when $\Sigma$ is the identity matrix.

#### The covariance Matrix

The covariance matrix $\Sigma$ is _symmetric_, since $\mathbb{C}ov[x_i, x_j] = \mathbb{C}ov[x_j,
x_i]$, and is assumed to be _positive definite_ (all eigenvalues are positive,
it is invertible and the determinant is greater than 0). The eigenvectors of
$\Sigma$ are real and can be chosen to form an _orthonormal_ set:

$$\Sigma u_i = \lambda_i u_i$$

$$u_i^T u_j = I_{ij} = 1\ if\ i=j,\ 0\ otherwise$$

Moreover:

$$\Sigma = \sum_{i=1}^D \lambda_i u_i u_i^T \qquad \Sigma^{-1} = \sum_{i=1}^D
\frac{1}{\lambda_i} u_i u_i^T$$

We have that:

$$\Delta^2 = \sum_{i=1}^D \frac{y_i^2}{\lambda_i}$$

where $y_u = u_i^T (x - \mu )$. We can interpret $y = (y_1, ..., y_d)^T$ as a
new coordinate system:

$$y = U (x - \mu)$$

where $U$ is the orthogonal matrix whose rows are the $u_i^T$, so that $UU^T =
U^T U = I$.

#### Conditional Gaussian Distributions

Suppose $x$ is a D-dimensional vector with Gaussian distribution $N(x|\mu,
\Sigma)$, and that we partition $x$ into two disjoint subsets $x_a$ and $x_b$:

$$
\bm{x} = \begin{pmatrix} \bm{x}_a \\ \bm{x}_b \end{pmatrix} 
\qquad\qquad 
\bm{\mu} = \begin{pmatrix} \bm{\mu}_a \\ \bm{\mu}_b \end{pmatrix} 
\qquad\qquad 
\bm{\Sigma} = \begin{pmatrix} \bm{\Sigma}_{aa} & \bm{\Sigma}_{ab} \\ \bm{\Sigma}_{ba} & \bm{\Sigma}_{bb} \end{pmatrix}
$$

It is convenient to work with the _precision matrix_ $\Lambda = \Sigma^{-1}$:

$$
\bm{\Lambda} = \begin{pmatrix} \bm{\Lambda}_{aa} & \bm{\Lambda}_{ab} \\ \bm{\Lambda}_{ba} & \bm{\Lambda}_{bb} \end{pmatrix}
$$

Our goal is to find the distribution of $x_a$ given $x_b,\ p(x_a | x_b)$. This
can be obtained by fixing $x_b$ in $p(x)$ and renormalising. The result is again
a Gaussian:

$$p(x_a | x_b) = N(x_a | \mu_{a|b}, \Sigma_{a|b})$$

where:

$$\mu_{a|b} = \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b)$$

$$\Sigma_{a|b} = \Lambda_{aa}^{-1} = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1}
\Sigma_{ba}$$

The _marginal_ distribution is Gaussian too:

$$p(x_a) = \int p(x_a, x_b) dx_b = N(x_a | \mu_a, \Sigma_{aa})$$

#### Bayes theorem for Gaussian Variables

Given a Gaussian distribution for $x$ and a conditional Gaussian distribution
for $y$ given $x$, of the form

$$p(x) = N(x|\mu, \Lambda^{-1}) \qquad \qquad p(y|x) = N(y | Ax + b, L^{-1})$$

where $\mu, A$ and $b$ are parameters governing the means, and $\Lambda$ and $L$
are precision matrices. The marginal distribution of $y$ and the conditional
distribution of $x$ given $y$ are:

$$p(x) = N(y | A\mu + b, L^{-1} + A \Lambda^{-1} A^T)$$

$$p(x | y) = N(x | \Sigma\{A^T L(y = b) + \Lambda \mu \}, \Sigma )$$

where $\Sigma = (\Lambda + A^T L A)^{-1}$

### Learning the Parameters of a Gaussian

Given i.i.d observations $X = (x_1, ..., x_N)^T$, the log likelihood function
is:

$$\ln p(X| \mu, \Sigma) = - \frac{ND}{2} \ln (2\pi) - \frac{N}{2} \ln |\Sigma| -
\frac{1}{2} \sum_{i=1}^N (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)$$

Sufficient statistics:

$$\sum_i x_i \qquad \sum_i x_i x_i^T$$

Solutions:

$$\mu_{ML} = \frac{1}{N} \sum_i x_i$$

$$\Sigma_{ML} = \frac{1}{N} \sum_i (x_i - \mu_{ML})(x_i - \mu_{ML})^T$$

$\Sigma_{ML}$ is biased: $\mathbb{E}[\Sigma_{ML}] = \frac{N-1}{N}\Sigma$.
Dividing by $N-1$ instead of $N$ removes the bias.

#### Sequential Estimation

We can update the mean as new experiments are gathered (sequential estimation):

$$\mu_{ML}^{(N)} = \frac{1}{N} \sum_{i=1}^N x_i = \frac{1}{N} x_N + \frac{1}{N} \sum_{i=1}^{N-1} x_i = \frac{1}{N} x_N + \frac{N-1}{N} \mu_{ML}^{N-1}$$

$$  = \mu_{ML}^{(N-1)} + \frac{1}{N} (x_N - \mu_{ML}^{(N-1)}) $$

This approach can be generalized to any distribution (Robbins-Monro algorithm).
Consider a pair of random variables $\theta$ and $z$ governed by a joint
distribution $p(z, \theta)$. The conditional expectation of $z$ given $\theta$
defines a deterministic function $f(\theta)$, called the _regression function_:

$$f(\theta) = \mathbb{E}[z|\theta ] = \int zp(z|\theta ) dz$$

Our goal is to find a root $\theta^*$ such that $f(\theta^*) = 0$.

We assume that the conditional variance of $z$ is finite:

$$\mathbb{E} [(z - f)^2 | \theta ] < \infty $$

and, without loss of generality, that $f(\theta) < 0$ if $\theta < \theta^*$ and
$f(\theta) > 0$ if $\theta > \theta^*$. The Robbins-Monro algorithms defines a
sequence of successive estimates of $\theta^*$:

$$\theta^{(N)} = \theta^{(N-1)} - a_{N-1} z(\theta^{(N01)})$$

where $z(\theta^{(N)})$ is an observed value of $z$ when $\theta$ takes the
value $\theta^{(N)}$, and the coefficients $\{a_N\}$ satisfy:

- $\lim_{N \rightarrow \infty} a_N = 0$

- $\sum_{N=1}^{\infty} a_N = \infty$

- $\sum_{N=1}^{\infty} a_N^2 < \infty$

Under these conditions the sequence converges to the root $\theta^*$ with
probability one.

How can a general maximum likelihood problem be solved sequentially using the
Robbins-Monro algorithm? By definitions, the maximum likelihood solution
$\theta_{ML}$ is a stationary point of the log likelihood function:

$$\frac{\partial }{\partial \theta}\{ \frac{1}{N} \sum_{i=1}^N \ln p(x_i |
\theta) \} = 0$$

Exchanging derivative and summation, and taking the limit $N \rightarrow \infty$:

$$\lim_{N \rightarrow \infty} \frac{1}{N} \sum_{i=1}^N \frac{\partial}{\partial
\theta} \ln p(x_i | \theta ) = \mathbb{E}_x[\frac{\partial}{\partial \theta} \ln
p(x|\theta )]$$

Finding the maximum likelihood solution is finding the root of a regression
function! The Robbins-Monro algorithm takes the form:

$$\theta^{(N)} = \theta^{(N-1)} + a_{N-1} \frac{\partial}{\partial \theta^{(n-1)}} \ln p(x_N | \theta^{(N-1)})$$

For the mean of a Gaussian distribution, $\theta^{(N)}$ is $\mu_{ML}^{(N)}$.

The random variable $z$ is given by:

$$z = - \frac{\partial}{\partial \mu_{ML}} \ln p(x | \mu_{ML}, \sigma^2) =
-\frac{1}{\sigma^2} (x - \mu_{ML})$$

Choosing the coefficients $a_N = \sigma^2 / N$ we obtain

$$\mu_{ML}^{(N)} = \mu_{ML}^{(N-1)} + \frac{1}{N} (x_N - \mu_{ML}^{(N-1)})$$

which is exactly the sequential formula derived before.

#### Example

A general recipe for maximum likelihood with a model $p(x|\theta )$:

1. Write the _negative log-likelihood_ $NLL(\theta) = -\sum_{i=1}^N \ln p(x_i|
   \theta )$

2. If $\nabla NLL(\theta ) = 0$ can be solved, use the _closed form_:

   - Bernoulli: $\mu_{ML} = m/N$
   - Multinomial: $\mu_{k, ML} = m_K/N$
   - Gaussian: $\mu_{ML} = \bar{x},\ \sigma_{ML}^2 = \frac{1}{N} \sum_i (x_i -
     \bar{x})^2$

3. Otherwise, minimise $NLL$ _numerically_, with various methods (gradient
   descent, Newton, BGFS). For instance, the shape parameter of the Gamma
   distribution has no closed form.

#### Example: unknown mean

Assume $\sigma^2$ is known, we consider the task of inferring the mean $\mu$
given a set of $N$ i.i.d. observations $x = \{ x_1, ..., x_N \}$. The likelihood
function for $\mu$ is:

$$p(x|\mu ) = \prod_{i=1}^N p(x_1 | \mu ) = \frac{1}{(2\pi \sigma^2)^{N/2}}
exp(-\frac{1}{2\sigma^2} \sum_{i=1}^N (x_i - \mu )^2)$$

This is a Gaussian shape as a function of $\mu$ (but not a distribution over
$\mu$). The conjugate prior is therefore a Gaussian over $\mu$:

$$p(\mu ) = N(\mu | \mu_0, \sigma_0^2)$$

which gives the posterior $p(\mu | x ) \propto x(x | \mu ) p(x)$.

By completing the square in the exponent we obtain:

$$p(\mu | x) = N(\mu | \mu_N, \sigma^2_N)$$

where:

$$\mu_N = \frac{\sigma^2}{N \sigma^2_0 + \sigma^2} \mu_0 + \frac{N \sigma_0^2}{N\sigma^2 + \sigma^2} \mu_{ML} \qquad \mu_{ML} = \frac{1}{N}\sum_{i=1}^{N}x_i \qquad \frac{1}{\sigma_N^2} = \frac{1}{\sigma_N^2} + \frac{N}{\sigma^2}$$

So the posterior mean is a compromise between the prior mean $\mu_0$ and the
maximum likelihood solution $\mu_{ML}$. We also see that each observation
increases the precision of the posterior by $1/\sigma^2$.

The posterior after $N-1$ observations become the prior when we observe the
$N$-th value:

$$p(\mu | x) \propto p(\mu)p(x | \mu ) = [p(\mu)\prod_{i=1}^{N-1}p(x_i | \mu)]
p(x_N | \mu)$$

$$\propto N(\mu | \mu_{N-1}, \sigma_{N-1}^2)p(x_N|\mu )$$

#### Example: Unknown precision

Let us now assume that $\mu$ is known, and infer the _precision_ $\lambda =
\frac{1}{\sigma^2}$. The likelihood function for $\lambda$ is:

$$p(x | \lambda ) = \prod_{i=1}^N N(x_i | \mu, \lambda^{-1}) \propto^{N/2} exp(-\frac{\lambda }{ 2} \sum_{i=1}^N (x_i - \mu)^2)$$

As a function of $\lambda$ this has the shape of a _gamma distribution_:

$$Gam(\lambda | a, b) = \frac{1}{\Gamma (a)} b^a \lambda^{a-1} e^{-b\lambda}
\qquad \mathbb{E}[ \lambda ] = \frac{a}{b} \qquad \mathbb{V}ar[\lambda] =
\frac{a}{b^2}$$

Combining a gamma prior $Gam(\lambda | a_0, b_0)$ with the likelihood function
for $\lambda$ we obtain:

$$p(\gamma | x) \propto \lambda^{a_0 - 1} \lambda^{N/2} exp (-b_0 \lambda -
\frac{\lambda}{2} \sum_{i=1}^N (x_i - \mu)^2)$$

This is again a gamma distribution, $Gam(\lambda | a_N, b_N)$ with:

$$a_N = a_0 + N / 2 \qquad b_N = b_0 + \frac{1}{2} \sum_{i=1}^N (x_i - \mu)^2 =
b_0 + \frac{N}{2} \sigma_{ML}^2$$

#### Exercise: Unknown Mean and Precision

If both $\mu$ and $\lambda$ are unknown, the joint likelihood function is:

$$p(x | \mu, \lambda ) = \prod_{i=1}^N
(\frac{\lambda}{2\pi})^{1/2}exp(-\frac{\lambda}{2}(x_i - \mu)^2)$$

$$\propto [\lambda^{1/2}exp(-\frac{\lambda \mu^2}{2})]^N exp(\lambda \mu
\sum_{i=1}^N x_i - \frac{\lambda}{2}\sum_{i=1}^N x_i^2)$$

We need a prior with the same functional dependence on $\mu$ and $\lambda$. The
_Gaussian-gamma_ (or normal-gamma) distribution:

$$p(\mu, \lambda) = N(\mu | \mu_0, (\beta \lambda)^{-1})Gam(\lambda | a, b)$$

#### Multivariate Conjugate Priors

For a D-dimensional Gaussian $N(x | \mu, \Lambda^{-1})$:

- $\mu$ unknown, $\Lambda$ is known: the conjugate prior $p(\mu)$ is a Gaussian

- $\Lambda$ is unknown, $\mu$ is known: the conjugate prior $p(\Lambda)$ is a
  Wishart distribution

  $$W(\Lambda | W, \nu) = B | \Lambda |^{(\nu - D - 1) / 2} exp(-\frac{1}{2}Tr(W^{-1}\Lambda))$$

  where $\nu$ is the number of degree of freedom, $W$ is a $D \times D$ scale
  matrix and $B$ a normalisation constant.

- $\mu$ and $\Lambda$ both unknown: the conjugate prior is the Gaussian-Wishart
  distribution:

  $$p(\mu, \Lambda | \mu_0, \beta, W, \nu) = N(\mu | \mu_0, (\beta \Lambda)^{-1}) W(\Lambda | W, \nu)$$

### Student's t-Distribution

If we have a univariate Gaussian $N(x | \mu, \tau^{-1})$ together with a gamma
prior $Gam(\tau | a, b)$ over its precision, and we integrate out the precision,
we obtain the marginal distribution of $x$:

$$p(x | \mu, a, b) = \int_{0}^{\infty } N(x | \mu, \tau^{-1}) Gam(\tau | a, b)
d\tau = $$

$$= \frac{\Gamma(\nu / 2 + 1/2)}{\Gamma (\nu / 2)} (\frac{\lambda}{\pi \nu})^{1/2} [ 1 + \frac{\lambda (x - \mu)^2}{\nu} ]^{-\nu / 2 - 1 / 2} = St(x | \mu, \lambda, \nu)$$

where $\nu = 2a$ (the _degrees of freedom_) and $\lambda = a / b$ (the precision
of the t-distribution). This is essentially an infinite mixture of Gaussians
with the same mean and different precisions. For $\nu = 1$ it is the _Cauchy_
distribution; for $\nu \rightarrow \infty$ it becomes $N(x | \mu, \lambda^{-1})$

Informally, with the t-Distribution we handle better the outliers: they do not
affect dramatically the variance.

#### Estimating the paramiters of a t-Distribution

We are now interested in estimating the parameters of a t-Distribution. There is
_no closed form_. Setting the derivatives of the log-likelihood to zero gives:

$$\mu = \frac{\sum_i w_i x_i}{\sum_i w_i}$$

$$\lambda^{-1} = \frac{1}{N} \sum_i w_i (x_i - \mu )^2$$

with: $w_i = \frac{\nu + 1}{\nu + \lambda (x_i - \mu )^2}$. But the weights
$w_i$ depend on $\mu$ and $\lambda$ themselves.

Iteratively reweithted estimation, for fixed $\nu$:

1. Start from $\mu = median(x)$ and $\lambda^{-1} =$ sample variance

2. Compute the weights $w_i$, then update $\mu$ (weighted mean) and
   $\lambda^{-1}$; repeat until convergence

The D-dimensional case is:

$$St(x | \mu, \Lambda , \nu) = \frac{\Gamma(D/2 + \nu / 2)}{\Gamma(\nu / 2)}
\frac{|\Lambda |^{1/2}}{(\pi \nu)^{D/2}} [1 + \frac{\Delta^2}{\nu}]^{-D/2 - \nu
/2}$$

where $\Delta^2 = (x - \mu)^T \Lambda (x-\mu)$ is the squared Mahalanobis
distance.

Properties:

- $\mathbb{E}[x] = \mu$, if $\nu > 1$

- $\mathbb{C}ov[c] = \frac{\nu}{\nu - 2} \Lambda^{-1}$, if $\nu > 2$

- $mode[x] = \mu$

### Bayesian curve fitting

- Frequentist view: find a single best $w^*$ that minimises $E(w)$

- Bayesian view: treat $w$ itself as a random variable with a prior, and compute
  a full distribution over $w$ given the data.

Given the training data $D$ composed of $w$ and $t$, along with a new test point
$x$, our goal is to predict the value of $t$, which is $p(t | x, w, t)$, suppose
that $\alpha$ and $\beta$ are known in advance. Assume Gaussian noise and a
Gaussian prior over weights:

$$p(t | x, w, \beta) = N(t | y(x, w), \beta^{-1}),\qquad p(w|\alpha ) = N(w | 0,
\alpha^{-1} I)$$

Because both are Gaussian, the posterior $p(w | D)$ is also Gaussian, and can be
computed in closed form using Bayes' theorem for Gaussian variables. We apply
the sum and product rules of probability to find the predictive distribution:

$$p(t | x, D) = \int p(t | x, w) p(w | D) dw = N(t | m(x), s^2(x))$$

- $m(x)$: predictive mean $\beta \phi(x)^T S \sum_{n=1}^N \phi(x_n)t_n$

- $s^2(x)$: predictive variance $\beta^{-1} + \phi(x)^T S \phi(x)$

Where $\phi(x)$ is a vector with elements $\phi_i(x) = x^i$ for $i = 0, ..., M$,
and the matrix $S$ is given by:

$$S^{-1} = \alpha I + \beta \sum_{i=1}^N \phi(x_n)\phi(x)^T$$

We are now interested in computing the posterior and the predictive
distribution. With the polynomial features $\phi (x) = (1, x, ..., x^M)^T$ and
the design matrix $\Phi$ whose rows are $\phi (x_n)^T$, Bayes' theorem for
Gaussian variables gives:

$$p(w | D) = N(w | m_N, S_N) \qquad S_N^{-1} = \alpha I + \beta \Phi^T \Phi
\qquad m_N = \beta S_N \Phi^T t$$

and the predictive distribution $N(t | m(x), s^2(x))$ has:

$$m(x) = \phi(x)^T m_N \qquad s^2(x) = \frac{1}{\beta} + \phi (x)^T S_N \phi
(x)$$

where $m_N = (\frac{\alpha}{\beta} I + \Phi^T \Phi)^{-1} \Phi^T t$ is the _ridge
regression_ solution with $\lambda = \alpha / \beta$, meaning that the prior
acts as a regulariser.

### Beyond the Gaussian

#### Periodic variables

Sometimes we need to model data with a periodic behaviour, such as when days and
hours are variables in the problem. Such quantities are naturally represented by
_angular_ (polar) coordinates $0 \le \theta < 2\pi$. Consider the problem of
evaluating the mean of a set of observations $D = \{\theta_1, ..., \theta_N\}$
(measured in radians). The simple average $\frac{1}{N}\sum_{i=1}^N \theta_i$ is
strongly coordinate dependent so it does not work.

Each $\theta_i$ can be represented as a point on the unit circle, i.e. a
two-dimensional unit vector:

$$x_i = (\cos \theta_i, \sin \theta_i )$$

and we can compute the mean as $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$.

Writing $\bar{x} = (\bar{r} \cos \bar{\theta}, \bar{r} \sin \bar{\theta})$ we
have:

$$\bar{r} \cos \bar{\theta} = \frac{1}{N} \sum_{i=1}^N \cos \theta_i \qquad
\bar{r} \sin \bar{\theta} = \frac{1}{N} \sum_{i=1}^N \sin \theta_i$$

Since $\tan \theta = \frac{\sin \theta}{\cos \theta}$, we obtain:

$$\bar{\theta} = tan^{-1} \{ \frac{\sum_{i=1}^N \sin \theta_i}{\sum_{i=1}^N \cos
\theta_i} \}$$

which does not depend on the choice of origin.

#### Von Mises Distribution

The _von Mises distribution_ is the periodic generalisation of the Gaussian
distribution. By convention, we consider distributions $p(\theta)$ with period
$2\pi$:

$$\forall \theta: p(\theta) \ge 0, \qquad \int_0^{2\pi} p(\theta) d\theta = 1
\qquad \forall \theta: p(\theta) = p(\theta + 2\pi)$$

Consider a Gaussian distribution over two variables $x = (x_1, x_2)$ with mean
$\mu = (\mu_1, \mu_2)$ and covariance matrix $\Sigma = \sigma^2 I$:

$$p(x_1, x_2) = \frac{1}{2\pi \sigma^2} exp(-\frac{(x_1 - \mu_1)^2 + (x_2 -
\mu_2)^2}{2 \sigma^2})$$

Transform from Cartesian coordinates $(x_1, x_2)$ to polar coordinates $(r,
\theta )$, and similarly for the mean:

$$x_1 = r \cos \theta,\qquad x_2 = r \sin \theta \qquad \mu_1 = t_0 \cos
\theta_0 \qquad \mu_2 = r_0 \sin \theta_0$$

Conditioning on the unit circle $(r=1)$, the exponent of the Gaussian becomes:

$$-\frac{1}{2\sigma^2} \{ (\cos \theta - r_0 \cos \theta_0)^2 + (\sin \theta -
r_0 \sin \theta_0)^2 \}$$
$$ = -\frac{1}{2\sigma^2} \{1 + r_0^2 - 2r_0 \cos \theta
\cos \theta_0 - 2 r_0 \sin \theta \sin \theta_0\}$$
$$ = \frac{r_0}{\sigma_2} \cos(\theta - \theta_0) + const$$

Defining $m = r_0 / \sigma^2$, the _von Mises distribution_, or _circular
normal_, is:

$$p(\theta | \theta_0, m) = \frac{1}{2 \pi I_0 (m)} exp(m \cos (\theta -
\theta_0))$$

where:

- $\theta_0$ is the mean of the distribution

- $m$ is the _concentration parameter_ (the analogue of the inverse variance of a
  Gaussian)

- $I_0(m)$ is the zeroth-order modified Bessel function of the first kind

$$I_0(m) = \frac{1}{2\pi} \int_0^{2\pi} exp(m \cos \theta) d\theta$$

The log likelihood function is:

$$\ln p(D | \theta_0, m) = -N \ln (2\pi) - N \ln I_0(m) + m\sum_{i=1}^N \cos
(\theta_i - \theta_0)$$

Setting the derivative with respect to $\theta_0$ equal to zero gives:

$$\sum_{i=1}^N \sin (\theta_i - \theta_0) = 0$$

Using $sin(\alpha - \beta) = \sin \alpha \cos \beta \sin \beta$:

$$\sum_{i=1}^N \sin(\theta_i - \theta_0) = 0$$

$$\Rightarrow \sum_{i=1}^N (\cos \theta_0 \sin \theta_i - \cos \theta_i \sin
\theta_0) = 0$$

$$\Rightarrow \cos \theta_0 (\sum_{i=1}^N \sin \theta_i) = \sin \theta_0
(\sum_{i=1}^N \cos \theta_i)$$

$$\Rightarrow \theta_0^{ML} = \tan^{-1} \{ \frac{\sum_i \sin \theta_i}{\sum_i \cos
\theta_i} \}$$

Which is the same result as the mean of the observations viewed as unit vectors.

Maximising with respect to $m$, and using $I_0'(m) = I_1(m)$, we obtain:

$$A(m_{ML}) = \frac{1}{N} \sum_{i=1}^N \cos (\theta_i - \theta_0^{ML})$$

where $A(m) = \frac{I_1 (m)}{I_0 (m)}$. The right-hand side can be written as:

$$A(m_{ML}) = (\frac{1}{N}\sum_{i=1}^N \cos \theta_i) \cos \theta_0^{ML} +
(\frac{1}{N} \sum_{i=1}^N \sin \theta_i) \sin \theta_0^{ML}$$

The function $A(m)$ can be inverted numerically to obtain $m_{ML}$.

#### Mixtures of Gaussians

A single Gaussian cannot capture _moltimodal_ data. A linear superposition of
$K$ Gaussians can:

$$p(x) = \sum_{k=1}^K \pi_k N(x | \mu_k, \Sigma_k)$$

where the _mixing coefficients_ satisfy $0 \le \pi_k \le 1$ and $\sum_k \pi_k =
1$.

We do not have a closed form, but we can use some iterative algorithms (_EM
algorithm_). For example, if we have 1000 response times of a web service behind
a cache, were most requests are hits (fast) and the others are misses (slow), we
can model this with a mixture of gaussians.

A problem that arises is to decide which gaussian component produced a certain
event. To work this out, think of a hidden choice $k$ with prior $p(k) = \pi_k$.
By bayes' theorem, the posterior of compenet $k$ given $x% is its
responsibility:

$$\gamma_k (x) = p(k | x) = \frac{\pi_k N(x | \mu_k, \sigma^2_k)}{\sum_{j=1}^K
\pi_j N(x | \mu_j, sigma^2_j)}$$

The log-likelihood is:

$$\ln p(x | \pi, \mu, \sigma) = \sum_{n=1}^N \ln \sum_{k=1}^K \pi_K N(x_n |
\mu_k, \sigma^2_k)$$

but this has no closed form. But if we knew the responsibilities $\gamma_{nk} =
\gamma_k (x_n)$, each component would be a weighted Gaussian NL problem. The
Expectation-Maximization (EM) algorithm would look like this:

1. E-step: compute $\gamma_{nk}$ with the current parameteres

2. M-step: with $N_k = \sum_n \gamma_{nk}$ (the effective number of points of
   component $k$)

$$\pi_k = \frac{N_k}{N} \qquad \mu_k = \frac{1}{N_k} \sum_n \gamma_{nk} x_n
\qquad \sigma^2_k = \frac{1}{N_k} \sum_n \gamma_{nk} (x_n - \mu_k)^2$$

3. Repeat until the log-likelihood stops increasing (as for the t-distribution
   weight)

The maximum likelihood always grows with $K$ (a mixture with $K$ components can
mimic one with $K-1$). For this reason, we need to penalise the number of
parameters $P(3K - 1\ in\ 1D)$ such that the lowest value wins.

$$BIC = -2\ln p(x | \theta_{ML}) + P \ln N$$
