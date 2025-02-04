# RLHF for Flow-Based Diffusion Models

This repository explores how to apply **Reinforcement Learning from Human Feedback (RLHF)** techniques—specifically for **flow-based diffusion models**. It discusses the theory behind RLHF in the context of diffusion, highlights popular RLHF methods, and describes the challenges and modifications required to integrate RLHF with flow-matching schedulers.

---

## Table of Contents

1. [Introduction to RLHF for Diffusion](#introduction-to-rlhf-for-diffusion)
2. [Existing RLHF Techniques](#existing-rlhf-techniques)
3. [Challenges with Flow-Based Models](#challenges-with-flow-based-models)
4. [Implementation Steps](#implementation-steps)
5. [Next Steps](#next-steps)
6. [References](#references)

---

## Introduction to RLHF for Diffusion

**Reinforcement Learning from Human Feedback (RLHF)** has been widely adopted to refine large language models by aligning them with user preferences. In the context of **diffusion-based image generation**, RLHF follows similar principles:
- We collect **human preferences** (e.g., which of two images is better aligned with a prompt).
- We **fine-tune** a diffusion model so that it generates images more consistent with these user preferences.

Traditionally, diffusion models are trained with **score matching** or **forward noise schedules**, but RLHF adds a new feedback loop:
1. Generate multiple candidate images.
2. Ask humans (or a proxy) to label which image(s) they prefer.
3. Update the model using those preferences.

---

## Existing RLHF Techniques

Several RLHF approaches exist in the diffusion space:
- **D3PO**: A PPO-inspired method for diffusion, comparing the likelihood ratio of the new model vs. a reference model at each diffusion step.
- **Direct Preference Optimization**: Directly optimizes model outputs according to a preference or reward without needing a reward model.
- **PPO/ILQL**-style algorithms adapted to diffusion: Using policy-gradient methods but interpreting the model as a “policy” over latent transitions.

Most of these methods rely on calculating the **log-probabilities** of the diffusion steps so that one can compare “new” vs. “old” model likelihoods for user-preferred samples.

---

## Challenges with Flow-Based Models

1. **Deterministic Scheduler**  
   Flow-matching or Euler-style schedulers often use (near-)deterministic updates, making it tricky to define a meaningful log-prob distribution for each step.
2. **No Direct Gaussian Parameterization**  
   Standard methods (DDIM, DDPM) naturally produce a mean and variance for each step, allowing straightforward log-prob calculation. Flow-based transitions frequently lack an explicit variance term.
3. **Need to Introduce Variance**  
   To compute a \(\log p_{\theta}(x_{t+1} \mid x_t)\), we may need to artificially add small noise to the update step. This ensures the transition is treated as a probabilistic process instead of a delta function.

---

## Implementation Steps

1. **Dataset Preparation**  
   - **Capture Intermediate Latents:** Store \(x_t, x_{t+1}\), and the scheduler timesteps at each diffusion step.  
   - **Collect Human Preferences:** For each prompt, generate multiple final images; label which image a user prefers.

2. **Log-Prob Calculation**  
   - **Modify Scheduler:** Create a function (e.g. `flowmatch_step_with_logprob`) that:
     1. Computes the deterministic update (Euler/flow step).
     2. Introduces or assumes a small variance \(\sigma^2\).
     3. Returns the log-prob of the observed \(x_{t+1}\).
   - **Reference vs. New Model:** Keep a frozen copy of your flow-based model and compare new vs. reference log-probs at each step.

3. **Training Loop Adjustments**  
   - **Pairwise Comparisons:** For each batch, compare two final images. If image A is preferred, reward transitions that lead to A being more probable under the new model.  
   - **Ratio-Based Loss:** Compute  
     \[
       \text{ratio} = \exp(\log p_\theta(x_{t+1}\mid x_t) \;-\; \log p_{\text{ref}}(x_{t+1}\mid x_t))
     \]
     and optimize according to user preferences (like a PPO-style objective).

4. **Gradient Updates**  
   - **Accumulate Gradients Across Steps:** For each diffusion timestep, compute the ratio-based reward and backprop through the flow-based UNet/transformer.  
   - **Clip Gradients & Stabilize Training:** As with PPO or other RL methods, gradient clipping can help.

---

## Next Steps

- **Hyperparameter Tuning**: The artificial variance, learning rate, and batch sizes all play crucial roles.
- **Scaling Up**: Flow-based models can be large; distributed training and half-precision can be key.
- **User Interface**: Creating user-friendly annotation tools for collecting preferences can greatly improve data quality.
- **Benchmarking**: Compare RLHF-trained flow models to established methods (DDIM or DDPM with RLHF) to gauge performance gains.

---

## References

- **D3PO**: [yk7333/d3po GitHub Repo](https://github.com/yk7333/d3po/)
- **PPO**: [Proximal Policy Optimization paper](https://arxiv.org/abs/1707.06347)
- **Flow Matching**: [Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations” (NeurIPS 2020)](https://arxiv.org/abs/2011.13456)

---

**Questions or Contributions?**  
Feel free to open an issue or pull request if you have suggestions on improving RLHF for flow-based diffusion.
