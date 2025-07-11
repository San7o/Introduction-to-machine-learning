
# Diffusion models

To recap, generative models train a generator $G$ from latent space to
data space and approximate the real data distribution. Variational
Autoencoders have the ability to generate new samples to regular
Autoencoders and use a probabilistic latent space (assumed to be a
multivariate Gaussian), while GAN have a generator that starts from
Gaussian noise and generates a data point in order to fool the
discriminator.

## Denoising diffusion

Denoising diffusion models consists of two processes:

- forward process to add noise
- reverse precess denoises to generate data

A _Markov chain_ or Markov process is a stochastic model describing a
sequence of possible events in which the probability of each event
depends only on the state attained in the previous event.

$$Pr(X_{n+1}=x|X_1=x_1, X_2 = x_2, ..., X_n = x_n)=Pr(X_{n+1}=x|X_{n}=x_n)$$

## Forward Process

A forward process gradually adds noise to the images over $T$ timesteps.

$$x_0\rightarrow x_1 \rightarrow x_2 \rightarrow ... \rightarrow x_T$$

The model will also be tasked to undo this noise called the "reverse" prorocess. Often the forward process is fixed and the reverse process needs to be trained.

A forward process has the following formulation:

$$q(x_t|x_{t-1})=N(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t I)\Rightarrow q(x_{1:T}|x_0)=\prod_{t=1}^Tq(x_t|x_{t-1})$$
Notice that we are multiplying the mean of the previous distribution times $\sqrt{1-\beta_t}$  and the variance by $\beta_t$, the two transformations are related in order to keep the distribution normalized. $\beta$ can be thought as the step size, It determins how aggressively the noise build up. Additionally, we are using $I$ as the base variance that is the identity matrix where all the values are completely independent for eachother.

Let $\alpha_t = (1-\beta_t)$, then

$$\bar{\alpha}_t=\prod_{s=1}^t(1-\beta_s)\Rightarrow q(x_t|x_0)=N(\sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t})I)$$

We can then sample directly at the desired timestep:

$$x_t=\sqrt{\bar{\alpha}_t}x_0 + \sqrt{(1-\bar{\alpha}_t)}\epsilon,\ \epsilon \sim N(0, I)$$

Formally we are applying a Gaussian convolution to the data at each timestep. Practically we are smoothening out the distribution to a Gaussian one.

A backward (denoising) process has the form:

$$p_\theta (x_{t-1}|x_t)=N(x_{t-1};\mu_\theta (x_t, t), \sigma_t^2I)$$
$$p_\theta (x_{0:T})=p(x_T)\prod_{t=1}^Tp_\theta (x_{t-1}|x_t)$$

## Generation Process

Given that $q(x_T)\approx N(0, I)$, a sample is $x_T\sim N(0, I)$ and iteratively $x_{t-1}=q(x_{t-1}|x_t)$. Through Bayes theorem, $q(x_{i-1}|x+t) \propto q(x_{t-1})q(x_t|x_{t-1})$ but we cannot solve this since we don't know $q(x|x_{t-1})$ . In other words, $p(x_{i-1}|x)$ depends on the entire Markov chain which we do not have. To solve this, we can approximate It by learning a Gaussian.

- $p(x_T)\sim N(0, I)$
- $p_{\theta}(x_{t-1}|x_t)\sim N(\mu_{\theta}(x_t, t), \sigma^2 I)$
- then $p_{\theta}(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_t)$

Hence, we need to find the parameters $\theta$ that approximate $p(x_{t-1}|x_t)$.

## Noising schedule

We can control the variance of the forward diffusion and reverse denoising processes respectively. Often a linear schedule is used for $\beta_t$ and $\sigma^2_t$ is set equal to $\beta_b$.

Kingma et al introduce a new parametrization of diffusion models using signal-to-noise ratio (SNR), and show how to learn the noise schedule by minimizing the variance of the training objective.

## Connection to VAE, GANs

- Latent variables have the same dimensionality of data.
- The same model is applied across different timesteps.
- The model is trained by revenging the variational bound.

## Training Parametrisation

We can train the model in a similar fashion as VAE, with a Variational Upper Bound:

$$L=\mathbb{E}_{q(x_0)}[-\log p_\theta (x_0)]\le \mathbb{E}_{q(x_0)q(x_{1:T}|x_0)}[-\log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}]$$

These can be divided into three terms:

$$L=\mathbb{E}_q[D_{KL}(q(x_T|x_0)||p(x_T))$$
$$+\sum_{t>1}D_{KL}(q(x_{t-1|x_t}, x_0)||p_{\theta}(x_{t-1}|x_t))-\log\ p_{\theta}(x_0|x_1)]$$

- the first term is fixed
- the second term is just summing gaussians

KL between Gaussians has a nice closed form, but Ho (with some math) proves the training can be simplified to a noise prediction problem, obraining a new loss:

$$L_{simple}=\mathbb{E}_{x_0\sim q(x_0), \epsilon \sim N(0, I), t\sim U(1, T)}[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{(1-\bar{\alpha}_t)}\epsilon, t)||]$$

Training Algorithm:

repeat

- $x_0 \sim q(x_0)$
- $t \sim Uniform(\{ 1, ..., T \})$
- $\epsilon \sim N(0, I)$
- Take gradient descent step on $\nabla_\theta || \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2$

until converged

Sampling algorithm:

- $x_T \sim N(0, 1)$
- for $t=T, ..., 1$ do
	- $x\sim N(0, I)$
	- $x_{t-1}=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta (x_t, t))+\sigma_t z$

- end for
- return $x_0$

The choice of the architecture is free. For images use U-NET.

## U-NET

The U-NET architecture contains two paths.

- The first path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The encoder extracts increasingly abstract features by applying convolutions and downsampling. At each level the spatial size decreases while the number of feature channels increases and allow the model to capture higher-level patterns.
- The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. The decoder begins to reconstruct the original image size through upsampling. At each level it combines decoder features with corresponding encoder features using skip connections to retain fine-grained spatial details.
- It is an end-to-end fully convolutional network, i.e. It only contains Convolutional layers and does not contain any Dense layer, for this reason it can accept image of any size.

## Generative Trilemma

Often fast sampling, mode coverage / diversity and high quality samples are difficult to coexist together.

- GAN have fast sampling with high quality samples but not mode coverage / diversity.
- Likelihood-based models (Variational Autoencoders and Normalizing flows) offer fast sampling and mode coverage/diversity but not high quality samples.
- Denoising diffusion models have mode coverage/diversity and high quality samples but not fast sampling.

## Diffusion GANs

Generative denoising diffusion models typically assume that the denoising distribution can be modeled by a Gaussian distribution. This assumption holds only for small denoising steps, which in practice translates to thousands of denoising steps in the synthesis process. In diffusion GANs, the denoising model is represented using multimodal and complex conditional GANs, enabling to efficiently generate data in a few steps. In other words, instead of working with Gaussians, we work with more complicated functions.

Compared to a one-shot GAN generator:

- Both generateor and discriminator are solving a much simpler problem
- Stronger mode coverage
- Better training stability

### Distillation

Distill a deterministic DDIM (Denoising Diffusion Implicit Model) sampler to the same model architecture. At each stage, a _student_ model is learned to distill two adjacent sampling steps of the _teacher_ model to one sampling step. At next stage, the "student" model from previous stage will serve as the new "teacher" model. This method allows to "skip" some stages in order to speedup computation.

## Latent-space diffusion models

The distribution of latent embeddings is close to Normal distribution, giving simpler denoising and faster synthesis. They allow augmented latent space and tailored autoencoders (graphs, text. 3D data, etc).

## Text-to-image: CLIP (OpenAI)

Jointly train a text encoder and an image encoder. Training by maximising the similarity between embeddings of (text, image) pairs. The resuling space has semantics for both images and text.

## Diffusion usages

- super resolution
- image-to-image (color a black and white image, extend and image's borders)
- semantic segmentation
- image editing (add something to a portion of the image)
- video generation
- 3d shape generation

