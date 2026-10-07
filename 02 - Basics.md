
# Machine Learning Basics

Machine Learning is the study of computer algorithms that improve automatically
through experience. The goal of this book is to make you understand how they
achieve such mystical behaviour. We will start or journey with a broad overview
of the types of machine learning, as well as defining some terminology and key
concepts. Don't worry, we will discuss these algorithms more in detail in the
future.

## The Learning pipeline

As we have said, in machine learning we want the computer to automatically
detect patterns and predict future data from other data. To do so, we perform a
series of steps which we call a _pipeline_. Some common steps are necessary in
most branches of machine learning and are relevant to our understanding, so they
are worth knowing. These are:

- _data acquisition_: collect the relevant data for the problems at hand.
- _preprocessing_: cleaning the data and preparing for analysis (normalizing the
  dataset, formatting it...). Techniques include _normalization_, _feature
  scaling_, _handling categorical variables_.
- _dimensionality reduction_: some methods can be applied to reduce the number
  of features while preserving the most important information (magic! but you
  will learn these tricks later).
- _model learning_: train the mode lon the preprocessed data.
- _model testing_: evaluated the model using a test set.

## Types of learning

We define a learning problem in the most general way as a function from _input_
space $X$ to _target_ space $Y$ (and you thought math was hard):

$$f: X \rightarrow Y$$

The input space $X$ is usually referred to with different names depending on
which phase of the pipeline we are: _training data_, _validation data_ and _test
data_.

A _model_ is a parametric family of functions $y(x;w)$, one for each choice of
parameters $w$. Choosing a model means choosing a _hypothesis space_:

$$H = \{ y(\cdot \ ; w): w \in \mathbb{R}^M \}$$

We discriminated between types of problems depending on the input and output
domains of the learn function, which determine the type of learning. Below are
described some of these which will be discussed in this book:

### Supervised Learning

Given labeled examples, create the model to predict the label, and test if the predicted label is correct.

  - _Classification_: there is a finite set of labels to predict.
    Given a training set $T = \{(x_1,y_1),...,(x_m,y_m)\}$ of size m,
	learn a function to predict $y$ given $x$:
$$f: \mathbb{R}^d \rightarrow \{1, 2, ..., k\}$$
	$x$ is generally multidimensional (multiple features).
	Applications include:
	- Face recognition
	- Character recognition
	- Spam detection (classical problem)
	- Medical diagnosis
	- Biometrics
  - _Regression_: the label is real value:
$$f: \mathbb{R}^d \rightarrow \mathbb{R}$$
  - _Ranking_: label indicates an order.

### Unsupervised Learning

The input is given with no labels. The main problems include:

- _Clustering_:  Given an input $T = \{x_1, ..., x_m\}$, output the hidden structure behind the $x$'s, which represents the clusters. Possible applications are:
	- social network analysis
	- genomics
	- image segmentation
	- anomaly detection
- _Dimensionality Reduction_: reduce the number of features under
	consideration by mapping data into another low dimensional space.
$$ f: \mathbb{R}^d \rightarrow \mathbb{R}^m, m << d $$
- _Density estimation_: find a probability distribution that fits
    the data.

### **Reinforcement learning**

The idea of reinforcement learning is that an _agent_ learns from the _environment_ by interacting with it and receiving _rewards_ for performing _actions_. This framework of acting is formally known as the _Markov Decision Process_. The agent needs to learn which actions to take given a state to maximize the overall reward collected.

### Other learning variations

- _semi-supervised_: some data have have labels, and some don't.
- _active learning_: the model learns a labeled dataset, and It
  interactively queries a human user to label new data points.
- _batch (offline) learning_: the model learns from the entire
  dataset in one go and is updated only after processing all the data
	- Once the model is trained, it does not change.
	- Typically used when the dataset fits into memory and can be processed efficiently as a whole.
- _online learning_: the model learns incrementally from each new
  data point or small batches of data.
	- Allows the model to adapt to changes in the data distribution over time.
	- Suitable for scenarios where data arrives sequentially and needs to be processed in real-time or where computational resources are limited. Examples include data streams, large-scale dataset and privacy-preserving applications.

## Loss function

Let $l(f, z)$ be a pointwise loss (a sum of point-losses) and $f(x_m ; w)$. The
error is computed from a function in an hypothesis space and a training set.

$$E(f, p) = \mathbb{E}_{z\sim pdata} [l(f, z)]$$

$$E(f, D) = \frac{1}{n}\sum_{i=1}^{n}l(f, t_i)$$

We want to minimize such error thought _optimisation_, such as through gradient
descent.

The loss function must be continuous and differentiable.


## Features

Features are the questions we can ask about the examples. They are
generally represented as _vectors_.

## Generalization

Machine learning is about generalization of data.

Generalization in machine learning refers to the ability of a trained
model to perform well on new, unseen data that was not used during the training process.

This can be done only if there is a correlation between inputs and
ouputs. More technically, we are going to use the _probabilistic
model_ of learning.

- there is some probability distribution over example / label pairs
  called the data generating distribution
- both the training data and the test set are generated based on the distribution

A data _generating distribution_ refers to the underlying probability
distribution that generates the observed data points in a dataset.
Understanding this distribution is crucial for building accurate and
generalizable machine learning models because It enables us to make
informed assumptions about the data and to make predictions or
decisions based on probabilistic reasoning.
This is valid for every kind of machine learning.

## Learning process

The steps to take in order to learn a model are:

1. Collect (annotated) data
2. Define a family of models for the classification task
3. Define an error function to measure how well a model fits the data
4. Find the model that minimized the error, aka train or learn a model

We define the following:

- _task_: a task represents the type of prediction being made to solve
  a problem on some data. $f: x \rightarrow y$
  - For example, in the classification case, $f: x \rightarrow \{c_1, ..., c_k \}$.
  - Similarly is clustering, where the output is a cluster index.
  - Regression: $f: \mathbb{R}^d \rightarrow \mathbb{R}$
  - Dimensionality reduction: $f: x \rightarrow y, dim(y) << dim(x)$
  - Density estimation: $f: x \rightarrow \Delta (x)$
- _data_: information about the problem to solve in the form of a
  distribution $p$ which is typically unknown.
  - training set: the failure of a machine learning algorithm is
	often caused by a bad selection of training samples.
  - validation set
  - test set
- _model hypotheses_: a model $Ftask$ is an implementation of a function $f$:
$$f \in Ftask$$
	A set of models forms an hypothesis space:
$$Hip \subseteq Ftask$$
	We use an hypothesis space to reduce the number of possible models in order to make our life easier.
- _learning algorithm_: the algorithm of your choice based on the problem
- _objective_: we want to minimize a (generalization) error function
	$E(f, p)$.
$$f* \in arg\ min\ E(f,p), f \in Ftask$$
	$Ftask$ is too big of a function space, we need an implementation
	(model hypotheses) so we define a model hypothesis space $Hip \in Ftask$ and seek a solution within that space.
$$f_{Hip}*(D) \in arg\ min_{f \in Hip_M} E(f, D)$$
	With $D=\{z_1, ..., z_n\}$ being the training data.

## Underfitting and Overfitting

- _Underfitting_: the error is very big, the output is very "far" from the ideal solution
- _Overfitting_: there is a large gap between the generalization
  (validation) and the training phase.

## How to improve generalization

Common techniques to improve generalization include:

- avoid attaining the minimum on training error.
- reduce model capacity.
- change the objective with a regularization term:
$$E_{reg}(f, D_n) = E(f, D_n)+\lambda \Omega (f)$$
	
	- $\lambda$ is the trade-off parameter
	- For example:
$$E_{reg}(f, D_n) = \frac{1}{n} \sum_{i=1}^{n} [f(x_i)-y_i]^2 + \frac{\lambda}{n} |w|^2$$

- inject noise in the learning algorithm.
- stop the learning algorithm before convergence.
- increase the amount of data:
$$E(f, D_N) \rightarrow E(f, p_{data}),\ n \rightarrow \inf$$
- augmenting the training set with transformations (rotate the image, change brightness...).
- combine predictions from multiple, decorrelated models (resembling).
	- train different models on different subsets of data, and we average the final solution between all of them

## Parametric vs Non-parametric Models

- _Parametric models_ have a finite number of parameters
	- linear regression, logistic regression, and linear support vector machines
- _Nonparametric model_: the number of parameters is (potentially) infinite
	- k-nearest neighbor, decision trees, RBF kernel SVMs

## Bias

The bias of a model is a measure of how strong the model assumptions are

- low-bias classifiers make minimal assumptions about the data
- high-bias classifiers make strong assumptions about the data


