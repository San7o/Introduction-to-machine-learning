
# The K-nearest neighbors algorithm

In this chapter we will discuss our first supervised learning algorithm: _K-nearest neighbors_, aka _K-NN_. Remember that supervised learning means that the algorithm is given data in an $n$-dimensional space with labels, we will now focus on the classification problem where the algorithm should predict the label of new unseen data.

If we make the only assumption that data with similar labels are somehow "cose" together in the space, a simple solution to this problem could be to look at the nearest known data and their label. We will discuss the concept distance later. Moreover, we can average over $k$ nearest neighbors to get a more representative result.

More precisely, to _classify_ an example $d$:

- find $k$ nearest neighbors of $d$
- choose as the label the majority label within the $k$ nearest neighbors

An example speudocode:

```
KNN(input, K, DATA) -> {1, ..., m}:
	int label_count[m] = {0}
	bool used[#DATA] = {false}
	for k=0 to K do:
		int max_distance = 0
		int max_input = 0
		for x_i, y_i in DATA do:
			d = distance(x_i, input)
			if d > max_distance and not used[x_i] then:
				max_distance = d
				max_input = x_i
		label_count[DATA[x_i]] += 1
		used[x_i] = true
	return max_index(label_count)
```

## How do we measure distance?

Measuring distance / similarity is a domain-specific problem and there are many different alternatives. We will now see different methods to model distance.

### Euclidean distance

Euclidean distance is measured in $n$ dimensions with the formula:

$$D(a, b) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2+...}$$

Note that here we are assuming that all the labels are comparable, meaning measured in the same range. A trivial example can be made to motivate this: a certain dimension may represent the cost of an item in euros and some other dimension may measure It in dollars, to find the difference we need to first convert everything to the same currency. Therefore data should be _standardized_, we will now see two common transformations:

- _Standardization or Z-score normalization_: Rescale the data so that the mean is 0 and the standard deviation from the mean is 1.
$$x_{norm}=\frac{x-\mu}{\sigma}$$
- _Min-Max scaling_: scale the data to a fixed range - between 0 and 1.
$$x_{norm}=\frac{x-x_{min}}{x_{max}-x_{min}}$$

### Manhatthan Distance

The Manhatthan distance is used to find the distance of two paths in a grid, It was originally invented for the city o Manatthan

$$D(a, b)=\sum_i |a_i - b_i|$$

### Checysher Distance

The checkysher distance is useful to calculate distances in the chess board, in particular the number of moved of the kind to reach a certain square.

$$D(a, b)=max_i|a_i-b_i|$$

### Minkowsky distance

All the three aforementioned distances can be generalized with the following formula (Minkowsky Distance) also called $p$-norm or $L_p$:

$$D(a, b) = (\sum_{k=1}^{n} |a_k - b_k|^p)^{\frac{1}{p}}$$

- $p=1$: Manhattan distance
- $p=2$: Euclidean distance
- $p\to\infty$: Checkysher distance

$L_1$ is popular because it tends to result is sparse solutions. However, It is not differentiable, so It only works for gradient descent solvers. $L_2$ is also popular because for some loss functions, it can be solved directly. $L_p$ is less popular since they don't tend to shrink the weights enough.

### Cosine Similarity

Given two different vectors $d_1$ and $d_2$, we can find the cosine $cos$ with the formula:

$$cos(d_1, d_2) = (d_1 \cdot d_2) / ||d_1|| ||d_2||$$

- where $\cdot$ is the dot product and $||d||$ is the length of vector $d$

In fact, this formula is derived from the definition of dot product: $d_1 \cdot d_2 = ||d_1||||d_2||cos(\theta)$

Cosine similarity does not depend on the magnitudes of the vectors, but only on their angle. For example, two proportional vectors have a cosine similarity of $+1$, two orthogonal vectors have a similarity of $0$, and two opposite vectors have a similarity of $-1$. 

For example, in information retrieval and text mining, each word is assigned a different coordinate and a document is represented by the vector of the numbers of occurrences of each word in the document. Cosine similarity then gives a useful measure of how similar two documents are likely to be, in terms of their subject matter, and independently of the length of the documents.

## Decision Boundaries

The KNN algorithm can be thought of as assigning a label to an object withing the space enclosed by a _decision boundary_. Decision boundaries are places in the features space where the classification of a point / example changes.

### Voronoi Diagram / Partitioning

A Voronoi diagram describes the areas that are nearest to any given point, given a set of data where each line segment is equidistant between two points. More formally, the Voronoi region $R_k$ associated with a subset of the points in the space $P_k$ is defined as:

$$R_k = \{ x\in X\ |\ d(x, P_K)\le d(x, P_j)\ for\ all\ j\ne k  \}$$

KNN does not explicitly compute decision boundaries, but form a subset of the Voronoi diagram for the training data.

## Choosing K

Some techniques to pick $k$ include:

- common heuristics
- use validation set
- use cross validatoin
- rule of thumb is k < sqrt(n) where n is the size of training examples

In general, bugger values of $k$ give a smoother decision boundary.

## Lazy Learner vs Eager Learner

k-NN belongs to the class of lazy learning algorithms.

- _lazy learning_: simply stores training data and operates when it is given a test example. Note that if the data is large, the machine may run out of memory.
- _eager learning_: given a training set, constructs a classification model before receiving new test data to classify

This means that k-NN is not really fast during inference, but no training is required.

## Curse of Dimensionality

There is a big problem in high dimensional data that may degrade the performance of the algorithm. That is, in high dimensions almost all points are far away from each other. I will now proceed to motivate this.

Suppose you have a space in $n$ dimensions where $m$ points are distributed uniformly. The volume of the space would be $S^n$ where $S$ is a measure of a side or the size of a domain in space, assuming all have the same domain. You could quantify a certain data density quantity as $\delta = \frac{m}{S^n}$. If we increase the dimensions by one, the density would decrease by a factor of $S$, hence we need to have $S$ times the original data size $m$ to get the same density: $\delta' = \frac{mS}{S^{n+1}}$. In general, $\delta''=\frac{mS^k}{S^{n+k}}$, for this reason we say that the size of the input has to grow exponencially with the dimensions. A less dense space means that all the data is more spread apart, hence all points are far away from each other.

The success of KNN is very dependent on having a dense data set since It requires points to be close in every dimension.

## Computational Cost

- Linear algorithm (no preprocessing) is $O(kN)$ to compute the distance for all N datapoints
- $O(klog(n))$ for tree-based data structures: pre-processing often using K-D tree
	- divide the space in regions. To check which region a point belongs to, simply traverse a binary tree.

k-NN variations: weighted k-NN where closest neighbors contribute the most.

