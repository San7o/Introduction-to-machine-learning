
# Linear Models

We have seen the KNN algorithm for classification. Said algorithm works everywhere as long as there is some spacial correlation in the data, which is often the case. We will now focus on a stricter set of problems, we will consider data that is _linearly separable_.

We say that labeled data is linearly separable if the classes can be binary separated by a line in 2 dimensions, or using hyperplanes in higher dimensions. In other words, a hyperplane defines a partition of the space and the data can be classified based on in which partition they live. This is a strong assumption to make and does not hold for many problems, we will see in a later chapter how we can circumvent this.

The equation of a plane in n dimensions looks like this

$$(1)\ b+\sum_{i=1}^{n}w_ix_i=0$$

Where $w_i$ are the values of the normal an $x_i$ are points in space in the dimension $i$.

We can classify a linear model by evaluating the equation $(1)$ with an input point and checking the sign of the output: positive values are classified as one thing, negative values as the other (remember that we are assuming binary classification).

More formally, given:

- an input space $X$
- an unknown distribution $D$ over $X \times \{ -1,+1 \}$
- a training set D sampled from $D$

Compute: A function $f$ minimizing $\mathbb{E}_{(x, y)\sim D}[f(x) \ne  y]$ 

## Training a linear model

Differently from KNN, linear models use _online learning_. The learning algorithm follows the following structure:

- the algorithm receives an unlabeled example $x_i$
	- For example: $w(1, 0),\ x_i(-1,1)$
- the algorithm predicts a classification of this example
	- Evaluate the sum equation $b+\sum_{i=1}^n w_ix_i$: $1*(-1)+0*1 = -1$ 
- the algorithm is them told the correct answer $y_i$ and It updates the model only if the answer is incorrect
	- check if the sign of the equation represents the right label, if not, add to each weight the current input times the label (assuming the label is either $+1$ or $-1$)

```
repeat until convergence (or some # of iterations):
	for each random training example (x1, x2, ..., xn, label):
		check if prediction is correct based on the current model
		if not correct, update all the weights:
			for each wi:
				wi = wi + xi*label
			b = b + label
```

The algorithm will converge only if the data can be linearly separated. The reason why $w_i + f_i\cdot label$ improves the solution will be obvious later when we will discuss gradient descent.

### Learning Rate

When the model makes an incorrect prediction, you hay have noticed that the correction of the weights are multiple of the label ($f_i\cdot label$ is added to the weight). This may make convergence difficult if we need more precise weights, so we multiply this by a _learning rate_.

$$w_i:=w_i+\lambda x_i*label$$


