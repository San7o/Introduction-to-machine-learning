
# Clustering

Given unlabeled data, we want to group them into clusters. The first thing to think about is how should these clusters be represented. We may want to model a cluster as a circumference where all the data inside It belongs to the same cluster, or maybe an ellipse may be a better choice for some problems. Some other data may need a completely another structure. Ultmatively, how we represent the cluster will characterize the algorithm that we will develop. In this chapter we will discuss four techniques to achive this.

Some complications rise when discussing representation of data, how to measure similarity or distance, is it a flat clustering or hierarchical and is the number of clusters known a priori. 

Let's start by categorizing _hard clustering_ where each example belongs to exactly one cluster, and _soft clustering_ where an example can belong to more than one cluster (probabilistic).

## K-means clustering

First, we are assuming that we know the number of clusters. This is a strong assumption to make and may be impossible for some problems. Our goal is then to find an assignment of data points to clusters, as well as a set of vectors $\{u_k\}$, such that the sum of the squares of the distances of each data point to its closest vector $u_k$, is a minimum. We can think of $u_k$ as the center of a cluster $C$.

$$min_{C_1, ..., C_k}\sum_{j=1}^kV(C_j)$$

- the variation is typically given by $V(C_j) = \sum_{i\in C_j} ||x_i-u_j||^2$ which is the euclidean distance squared.
- the _centroid_ is computed with $u_j = \mathbb{E}_{c\in C}[x\in C] = \frac{1}{|C_j|}\sum_{i\in C_j} x_i$

### The algorithm

```
Initialization: Use some initialization strategy to get
some initial cluster centroids u1,...,uk

while clusters change, do
	Assign each datapoint to the closest centroid forming
	new clusters

		Cj = i in N such taht j = argmin_l(xi - ul)

	Compute cluster centroids u1,...,uk
```

The algorithm is guaranteed to converge, but not to find the global minimum.

### Initial Centroid Selection

- random selection
- points least similar to any existing center
- try multiple starting points

### Running time

- assignment: $O(kn)$ time
- centroid computation: $O(n)$ time

In this case, euclidian distance is not the best so we use _consine similarity_
$$sim(x, y)=\frac{x*y}{|x||y|}$$

## Expectation Maximization

If we restrain our problem by assuming that the clusters are formed as a mixture of Gaussians (elliptical data) and we assign data to a cluster with a certain probability (soft clustering) we can use the _Expectation Maximization_ (EM) algoritm. The algorithm proceeds as follows:

- Start with some initial cluster centers
- Iterate
	- Soft assign points to each cluster by calculating the probability of each point belonging to each cluster $p(\theta _c | x)$.
	- Train: Recalculate the cluster centers by calculating the maximum likelihood cluster centers given the current soft clustering $\theta _c$.

A Gaussian (ellipse) in 1 dimension is defined as:
$$f(x)=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma ^2}}$$
A Gaussian in $n$ dimensions is defined as:
$$N[x;\mu , \Sigma] = \frac{1}{(2\pi)^{d/2}\sqrt{det(\Sigma)}}exp[-\frac{1}{2}(x-\mu)^T\Sigma ^{-1}(x-\mu)]$$

We learn the means of each cluster (i.e. the center) and the covariance matrix (i.e. how spread out it is in any given direction).

Intuitively, each iteration of Expectation Maximization increases the likelihood of the data and is guaranteed to converge (though to a local optimum).

## Spectral Clustering

In case of non-gaussian data, we can use a technique called _spectral clustering_ where we group points based on links in a graph.

We first create a fully connected graph or a k-nearest neighbor graph (each node is only connected to its K closest neighbors). Alternatively, we can create a graph with some notion of similarity among nodes, for example using a gaussian kernel: $W(i, j)=exp[\frac{-|x_i-x_j|^2}{\sigma ^2}]$.

We then partition the graph by cutting the graph: we want to find a partition of the graph into two parts $A$ and $B$ where we minimize the sum of the weights of the set of edges that connect the two groups, aka $Cut(A, B)$. We observe that this problem can be formalized as a discrete optimization problem and then relaxed in the continuous domain and become a generalized eigenvalue problem.

## Hierarchical Clustering

We are interested in producing a set of nested clusters organized as a hierarchical tree, also called _dendrogram_.

The idea of the algorithm is to first mege very similar instances and then incrementally build larger clusters out of smaller clusters. The algorithm goes as follows:

- Initially, each instance has its own cluster
- Repeat
	- Pick the two closest clusters
	- Merge them into a new cluster
	- Stop when there is only one cluster left

