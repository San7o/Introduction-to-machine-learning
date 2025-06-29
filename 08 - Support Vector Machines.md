# Support Vector Machines

Linear classifiers assume the data can be linearly separable. If this
does not hold, the chosen hyperplane will keep adjusting at each
iteration without ever converging. Let's now discuss a particular type
of classifier: support vector machines (SVM).

## Large margin classifiers

We define the _margin_ of a classifier as the distance from the
classification hyperplane to the closest points of either class. We
call _large margin classifiers_ the classifiers that attempt to
maximize this measure.

We call the sets of "closest points" from the hyperplane with _support
vectors_. Note that for n dimensions, there will be at least $n+1$
support vectors; this is because a plane is identified by $n+1$ points in $n$ dimensions.

Therefore, large margin classifiers only use support vectors and
ignore all other training data.

Let's formalize the large margin classifier with algebra.

We uniquely identify the decision hyperplane by a normal vector w and a
scalar b which determines which of the planes perpendicular to w to
choose:

$$\vec{w}\vec{x} + b = 0$$

We are interested in maximizing the distance from the hyperplane to
the support vectors. Let us first find the distance from the
hyperplane to a point p. We know that the distance vector must be
perpendicular to the hyperplane, therefore It is parallel to the
normal vector w, which when normalized is $\frac{\vec{w}}{||\vec{w}||}$.

Therefore, the distance between the hyperplane and $p_1$, referred to as
$r$, is the difference between $p_1$ and the closest point in
the hyperplane $p_2$, which is the intersection between the line parallel
to the normal and point $p1$:
$$(1)\ r\frac{\vec{w}}{||\vec{w}||} = \vec{p_1} - \vec{p_2}$$

Moreover, since $p2$ lies on the hyperplane, It is ture that:
$$(2)\ \vec{w}\vec{p_2} + b = 0$$
We find $p2$ from $(1)$, $p_2 = p_1 - r\frac{\vec{w}}{||w||}$. We substitute
$p_2$ in $(2)$ and solve for r, the distance, and we get 
$$r = \frac{(\vec{w}\vec{p_1} + b) ||\vec{w}||}{\vec{w}\vec{w}}$$
Since $\vec{w}\vec{w}=||w||^2$, this can be easily demonstrated because
the modulo function uses the generalized pythagorean theorem, we get:
$$r = \frac{(\vec{w}\vec{p_1} + b)}{||\vec{w}||}$$

We just demonstrated the formula for finding the distance between a
point and a plane.  We now define the _classification function_ as
$f(x)= sign(\vec{w}\vec{x} + b)$, It will be $+1$ or $-1$ based on the sign of the equation. Intuitively, points on one side of the plane should have a positive output and points on the other side should have a negative one. 

We want to maximize this distance, but the value is positive or negative depending on where the point is, hence we multiply the hyperplane equation times the expected classification in roder to make the distance always positive (labels are either $-1$ or $1$). Indeed, If we assign the label $y_i=-1$ for the negative distance points, we will always get a positive output (negative times negative is negative, and positive times positive is positive). We define the _functional margin_ as double such quantity: this represents the distance between the two closest points from the plane with different labels:
$$margin = 2\frac{y_i(\vec{w}\vec{x_i}+b)}{||\vec{w}||}$$
For convenience, we require that the distance between the hyperplane and the support vectors is 1, therefore $y_i(\vec{w}\vec{x_i}+b) \ge 2 \forall i$ because negative distances will be multiplied by $y_i$ which is also negative, resulting in a positive value all the time, and a greater than $1$ because of the requirement.

For the support vectors, the following holds:
$$margin = \frac{1}{||\vec{w}||}$$

## Maximizing the margin

We want to maximize this margin, therefore we need to minimize $\frac{1}{2}||\vec{w}||$
with the constraint that $y_i(\vec{w}\vec{x_i}+b) \ge 1\ \forall i$. We
may as well minimize $\frac{1}{2}||\vec{w}||^2$ which is an example of a _quadratic
optimization problem_
$$ min_{w, b}\ \frac{1}{2} ||w||^2,\ y_i(\vec{w}\vec{x_i}) \ge 1 \forall i $$

## Soft margin classification

We may have data that is not perfect, for example we may have some
values that lay on one side of the hyperplane but they are classified
as the other, so $y_i(\vec{w}\vec{x_i}+b) \ge 1 \forall i$ does not
hold since you may have $\vec{w}\vec{x_i}+b \le 1$ and $y_i \ge 1$ or
the opposite.

To address this cases, we introduce _slack variabes_ which are
scalar values assigned for each $x_i$, and the _regularizaion
parameter_ $C > 0$:
$$ min_{w, b} ||\vec{w}||^2 + C \sum_{i} \zeta_i $$
subject to:
$$y_i(\vec{w}\vec{x_i} + b) \ge 1 - \zeta_i \forall i, \zeta_i \ge 0$$

with this correction, the values are "allowed to have mistakes", the
margin is reduced by $\zeta$ if $\zeta < 1$ or are moved to the
other side of the hyperplane if $\zeta > 1$.
$C$ determines how strong are those corrections, that is the "trade off"
between the slack variable penalty and the margin.

- small C allows constrains to be easily ignored in order to have a
  larger margin
- large C makes constraints hard to ignore in order to get a narrow margin
  - if C goes to infinity, we have hard margins.
  
We are now interested into calculating these slack variables. The
first observation we make is that if the constraint is already
satisfied, meaning that if $y_i(\vec{w}\vec{x_i} + b) \ge 1$ and the
value has not been misclassified, then we don't need any correction and
$\zeta = 0$. Otherwise, we want to correct the value by moving it on
the other side, plus the distance to the margin 1, so $1 - y_i(\vec{w}\vec{x_i}+b)$. We are subtracting because if the value is
misclassified, then It's distance from the hyperplane times It's label
is going to be a negative value (one of them nedds to be positive and
the other negative in order to have a misclassification).

Therefore:
$$\zeta = 0\ if\ y_i(\vec{w}\vec{x_i}+b) \ge 1$$
$$\zeta = 1 - y_i(\vec{w}\vec{x_i}+b)\ otherwise$$

which is the same as the following, using a notation introduced
in previous lessons:
$$\zeta = max(0, 1-y_i(\vec{w}\vec{x_i}+b)) = max(0, 1-yy')$$

If you recall from the lesson of Gradiente Descent, this is the hinge
loss function.

With this result, the objective is now to minimize the following:
$$min_{w, b} ||\vec{w}||^2 + C \sum_i max(0, 1 - y_i(\vec{w}\vec{x_i} +b))$$

