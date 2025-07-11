
# Beyond Binary Classification

## Multi-class classification

In multi-class classification problem, differently from binary classification, given:

- An input space $X$ and number of classes $K$
- An unknown distribution $D$ over $X \times [K]$
- A training set D sampled from $D$

Compute: A function $f$ minimizing $\mathbb{E}_{(x, y)\sim D}[f(x) \ne  y]$ 

- Idea: one line does not suffice, but we can combine more lines

There are two approaches to achieve multi-class classification which we will discuss:

- one vs all
- all vs all

## One vs All (OVA)

For each label $k=1, ..., K$ define a binary problem where:

- all examples with label $k$ are positive
- all other examples are negative

In practice, learn $K$ different classification models.

To classify we pick the most confident positive, in none vote positive, pick least confident negative.

```
OneVsAllTrain(D, BinaryTrain):
	for i=1 to K do
		D_1 = relabel D so class i is positive and \not i is negative
		f_i = BinaryTrain(D_1)
	end for
	return f_1,...,f_k 
```

```
OneVsAllTest(f_1,...,f_k, x):
	score = (0,...,0)
	for i=1 to K do
		y = f_1(x)
		score[i] = score[i] + y
	end for
	return max(score)
```

## All vs All (AVA)

All vs All, sometimes called _all pairs_, trains $K(K-1)/2$ classifiers:

- the classifier $F_{ij}$ receives all the examples of class $i$ as positive and all the examples of class $j$ as negative, for each pair $(i, j)$
- every time $F_{ij}$ predicts positive, the class $i$ gets a vote, otherwise, class $j$ gets a vote
- after running all $K(K-1)/2$ classifiers, the class with the most votes wins

Note: The teacher might ask to explain the algorithm in more detail

## Ova vs Ava

Train time:

- AVA learns more classifiers, however, they are trained on much smaller data this tends to make it faster if the labels are equally balanced

Test time:

- AVA has more classifiers, so often is slower

Error:

- AVA tests with more classifiers and therefore has more chances for errors

## Macroaveraging vs Microaveraging

- Microaveraging: average over examples
- Macroaveraging: calculate evaluation score (e.g. accuracy) for each label, then average over labels

