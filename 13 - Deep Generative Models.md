
# Deep Generative Models

## Definitions

### Generative Models

_Generative models_ are statistical models of the data distribution $p_X$ or $p_{XY}$, depending on the availability of target data, where $Y$ is the label variable.

### Discriminative models

_Discriminative models_ are statistical models of the conditional distribution $p_{Y|X}$.

Note that a discriminative models can be constructed from a generative model via Bayes rule but not viceversa.

$$p_{Y|X}(y|x)=\frac{p_{XY}(x, y)}{\sum_{y'}p_{XY}(x, y') = p_X(x)}$$

## Density Estimation

Explicit problem: Find a probability distribution $f\in \Delta (Z)$ that fits the data, where $z\in Z$ is sampled from an unknown data distribution $p_{data}\in \Delta (Z)$.

Implicit problem: Find a function $f\in Z^{\Omega}$ that generates data $f(\omega)\in Z$ from an input $w$ sampled from some predefined distribution $p_\omega \in \Delta (\Omega)$ in a way that the distribution of generated samples fits the (unknown) data distribution $p_{data}\in \Delta (Z)$. 

In the case of supervised learning, $Z=X\times Y$ while in the case of unsupervised learning, $Z=X$. We will discuss this last case.

The objective is to define an hypothesis space $H\subset \Delta (Z)$ of models that can represent probability distributions and a divergence measure $d\in \mathbb{R}^{\Delta (z)\times \Delta (Z)}$ between probability distributions in $\Delta (Z)$. Then find the hypothesis $q*\in H$ that best fits the data distributed according to $p_{data}$:

$$q*\in arg\ min\ d_{q\in H}(p_{data}, q)$$

### KL-Divergence

Before continuing, It is worth taking some time to study KL-divergence since we will use this later. The Kullback–Leibler divergence, denoted as $d_{kl}(P|Q)$ is a type of statistical distance that measures how much a model or theory probability distribution $Q$ is different from a true probability distribution $P$. It is defined as:

$$d_{kl}(P|Q)=\sum_{x\in X}P(x)\log \frac{P(x)}{Q(x)}=\mathbb{E}[\log \frac{P(x)}{Q(x)}]$$

A simple interpretation of the KL divergence of $P$ from $Q$ is the expected excess surprise from using $Q$ as a model instead of $P$ when the actual distribution is $P$. If $P$ and $Q$ are the same distribution, the KL-divergence is $0$.

### Jensen's Inequality

Jensen's inequality generalizes the definition of convex function as a function that satisfies the following inequality:

$$f(\theta x_i + (1-\theta)x_2) \le \theta f(x_i)+(1-\theta)f(x_2)$$

In the context of probability theory, this is stated as:

$$\phi(\mathbb{E}[X])\le \mathbb{E}[\phi(X)]$$

where $\phi$ is a convex function.

## Autoencoders

The basic idea behind autoencoders is to encode information (as in compress) automatically, hence the name. Autoencoders are neural network with the smallest layer at the center, and are symmetrical on the left and right. We call the part of the network from the input to the center an _encoder_, we call the other part a _decoder_. The network can be trained with backpropagation by feeding input and setting the error to be the difference between the input and what came out.

The encoder can also be used for dimensionality reduction without the encoder, or as a supervised model. It will reduce the input to a much smaller latent space where another neural network can be connected to.

## Variational AutoEncoders (VAE)

Variational Autoencoders add a "probabilistic spin" to traditional Autoencoders by finding a probability distribution in the latent space and using this to decode another probabilistic distribution that best matches the input's distribution.

Let $\omega$ be the distribution in the _latent space_, $q_\theta(\omega|x)$ be an _encoder_ distribution and $q_\theta(x|\omega)$ be a _decoder_ distribution. Then the objective is:

$$\theta^* \in arg\ min_{\theta \in \Theta}\ d(p_{data}, q_{\theta})$$
$$(1)\ q_\theta (x)=\mathbb{E}_{\omega \sim p^{\omega}}[q_\theta (x|\omega)]$$

The two formulas above are really important. We want to find the best parameters that minimize the distance between the predicted $q_\theta$ distribution and the actual real distribution $p_{data}$. We find $q_\theta(x)$ by taking the expected value of the decoder function $q_\theta(x|\omega)$ where $\omega$ is the distribution in the latent space.

As the divergence measure we will use the KL-Divergence:

$$d_{KL}(p_{data}, q_{\theta})=\mathbb{E}_{x\sim p_{data}}[\log \frac{p_{data}(x)}{q_{\theta (x)}}]=-\mathbb{E}_{x\sim p_{data}}[\log\ q_{\theta}(x)]+const$$

$$=^{(1)} -\mathbb{E}_{x\sim p_{data}}[\log\ \mathbb{E}_{\omega \sim p_{\omega}}[q_{\theta (x|\omega)}]]+const$$

Let's focus on the argument of the outer expected value. Let $q_\psi(\omega | x)\in \Delta (\Omega)$ denote an encoding probability distribution with parameters $\psi$. We can change the expected value measure to this probability like this:

$$\log\ \mathbb{E}_{\omega \sim p_{\omega}}[q_{\theta} (x|\omega)]=\log\ \mathbb{E}_{\omega \sim q_\psi (\cdot|x)}[q_{\theta} (x|\omega)\frac{p_{\omega}(\omega)}{q_{\psi (\omega | x)}}]$$

Thanks to the Jensen's inequality we can write:

$$\ge \mathbb{E}_{\omega \sim q_\psi (\cdot|x)}[\log(q_{\theta}(x|\omega)\frac{p_{\omega}(\omega)}{q_{\psi (\omega | x)}})] =$$
$$=  \mathbb{E}_{\omega \sim q_\psi (\cdot|x)}[\log\ q_\theta (x|\omega)]-d_{KL}(q_\psi (\cdot|x), p_\omega)$$

Those two last terms are the _Reconstruction_ and the _Regularizer_. The first is still NP-hard to compute but we can estimate the gradients of the parameters, while the regularizer might have closed-form solution (for example, using Gaussian distribution).

In practice training is divided into several steps: first data samples are fed to an encoder which can be something like a normal neural network or a convolutional one. This encoder will reduce the dimensionality of the input into what's called a "latent space" which is composed of fewer nodes than the input layer. We are interested in the distribution of this latent space, so we assume that Its distribution follows a gaussian and we compute the mean and the covariance. We then feed this statistical model to the decoder which generates again some data. Finally, we compare the original samples with the generated data via a loss function and update the weights of both the encoder and decoder to minimize the difference.

## Issues with VAE

The problem with this approach is underfitting, since at initial stages the regularizer is too strong and tends to annihilate the model capacity, and blurry output data.

A modern approach to image generation is Vector Quantized VAE (VQ-VAE-2) which is a image synthesis model based on Variational Autoencoders. It produces images that are high quality by leveraging a _discrete latent space_.

### Conditional VAE

Variatonal Auto Encoders do not need lables. Conditional VAE are a variation of VAE that accept labels.

Assume we have side information $y\in Y$ (e.g. digit labels, attributes, etc) and we want to generate new data conditioned on the side information (e.g. generate digit 7, or generate a face with glasses).

Modify the encoder and decoder to take the side information in input obtaining $q_{\psi}(\omega | x, y)$ and $q_{\theta}(x|\omega, y)$.

Define priors conditioned on side information $p_{\omega}(\omega |y)$.

## Generative Adversial Networks (GAN)

VAE are able to find explicit densities, GAN enables the possibility to find implicit ones.

GAN models are composed by two "adversarial" submodels: a _generator_ and a _discriminator_. The term adversarial means that the two submodels are in competition and there is one winner (a zero sum game).

The generator is tasked to generate fake images, while the discriminator is tasked to recognize if an image is fake or not. The generator generates images for the discriminator to check, mixing real images with generated ones. If the generator fools the discriminator, than the latter needs to update Its weights, otherwise the opposite will happen.

GANs enable the possibility of estimating implicit densities. We assume to have a prior density $p_\omega \in \Delta (\Omega)$ given and a generator (or decoder) $g_\theta \in X^{\Omega}$ that generates data points in $X$ given a random element from $\Omega$.

The density induced by the prior $p_{\omega}$ and the generator $g_{\theta}$ is given by $q_{\theta}(x)=\mathbb{E}_{\omega \sim p^{\omega}}\delta [g_\theta (\omega)-x]$, where $\delta$ is the Dirac delta function. The Dirac function is a generalized function on the real numbers, whose value is zero everywhere except at zero, and whose integral over the entire real line is equal to one.

The (original) GAN objective is to find $\theta^*$ such that $q_{\theta^*}$ best fits the data distribution $p_{data}$ under the Jensen-Shannon divergence:

$$d_{JS}(P, Q)=\frac{1}{2}d_{kl}(P|M)+\frac{1}{2}d_{kl}(Q|M),\ M=\frac{1}{2}(P+Q)$$
$$\theta^* \in arg\ min_{\theta}\ d_{JS}(p_{data}, q_{\theta})$$

where

$$d_{JS}(p, q)=\frac{1}{2}d_{KL}(p, \frac{p+q}{2})+\frac{1}{2}d_{KL}(q, \frac{p+q}{2})$$
$$=\frac{1}{2}\mathbb{E}_{x\sim p}[\log\frac{2p(x)}{p(x)+q(x)}]+\frac{1}{2}\mathbb{E}_{x\sim q}[\log\frac{2q(x)}{p(x)+q(x)}]$$
$$=\frac{1}{2}\mathbb{E}_{x\sim p}[\log\frac{p(x)}{p(x)+q(x)}]+\frac{1}{2}\mathbb{E}_{x\sim q}[\log\frac{q(x)}{p(x)+q(x)}] + \log(2)$$
$$=\log(2)+\frac{1}{2}max_t\ \{ \mathbb{E}_{x\sim p}[\log\ t(x)] + \mathbb{E}_{x\sim q}[\log(1-t(x))] \}$$

The interpretation of the $JS$ divergence is that this mixes how the two distributions depend on one another, not just one over the other; It is useful when there is not a true reference distribution.

Let $t_\phi (x)$ be a binary classifier (or discriminator) for data point in the training set predicting whether $x$ came from $p$ or $q$, we get the following lower bound on our objective:

$$d_{JS}(p_{data}, q_\theta)=\log(2)+\frac{1}{2}max_t\ \{ \mathbb{E}_{x\sim p}[\log\ t(x)] + \mathbb{E}_{x\sim q}[\log(1-t(x))] \}$$
$$\ge \log(2)+\frac{1}{2}max_\phi \ \{ \mathbb{E}_{x\sim p}[\log\ t_\phi(x)] + \mathbb{E}_{x\sim q}[\log(1-t_\phi(x))] \}$$

Which is minimized to obtain the generator's parameters:

$$\theta^* \in argmin_\theta\ max_\phi \{ \mathbb{E}_{x\sim p}[\log\ t_\phi(x)] + \mathbb{E}_{x\sim q}[\log(1-t_\phi(g_{\theta}(x)))] \}$$

In practice, during training both real data and generated data are passed to a classifier that estimates if the data is real or generated with the $t\phi(x)$ function. The generator and the discriminators are opponents and this is modeled mathematically via the arg min-max function: the generator tries to minimize the parameters $\theta$ and the classifier tries to maximize the parameter $\phi$ to solve the equation.

