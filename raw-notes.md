# Notes

Notes that need to be included in the rest of the book.

## Optimization

We want to find the optimum of a cost function.

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
(symmetric for smooth $f$).

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
