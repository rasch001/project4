# Project 4 Overview

This repository contains three projects that build on one another. They are organized in numerical order as follows:

1. **project_4_file_1**  
2. **project_4_file_2**  
3. **project_4_file_3**

It is recommended that you read these files in numerical order to understand the progressive development of the ideas and techniques.

---

## File Descriptions

### project_4_file_1

This file contains the base implementation of a GPT-2 style model. The code demonstrates:
- **Transformer architecture:** Including token and positional embeddings, multi-head self-attention, and deep feedforward networks.
- **Next-token prediction training:** The model is trained to maximize the conditional probability of the next token given the previous tokens.

**Mathematical Background:**

The training objective is to maximize the likelihood of a sequence:
$$
P(x) = \prod_{t=1}^{n} P(x_t \mid x_{<t})
$$
The cross-entropy loss used is given by:
$$
\mathcal{L} = -\sum_{t=1}^{n} \log P(x_t \mid x_{<t})
$$

---

### project_4_file_2

This file extends the base GPT-2 model by incorporating a **Mixture-of-Experts (MoE)** mechanism known as **SwitchHead**. Key features include:
- **SwitchHead MOE for self-attention:** A gating mechanism routes tokens to different expert modules, reducing computation and memory requirements.
- **Efficiency improvements:** By computing fewer attention matrices, the model can maintain performance while being more resource-efficient.

**Mathematical Background:**

For each token, the gating mechanism computes a distribution over experts:
$$
g(x_i) = \text{softmax}(W_g x_i)
$$
where the chosen expert is:
$$
e_i = \arg\max g(x_i)
$$
This approach reduces the standard attention computation cost:
$$
\text{Cost}_{\text{SwitchHead}} \propto M \times L^2 \quad \text{(with } M \ll H\text{)}
$$

---

### project_4_file_3

This file implements a **Generative Adversarial Network (GAN)** that uses the GPT-2 MOE model as the generator. It introduces:
- **Discriminator network:** A simple LSTM-based binary classifier that distinguishes between real text and generated text.
- **Adversarial training:** The generator receives feedback (via a REINFORCE-style update) from the discriminator to improve its text generation.

**Mathematical Background:**

The discriminator minimizes the binary cross-entropy loss:
$$
\mathcal{L}_D = -\frac{1}{B} \sum_{i=1}^{B} \left[ \log D(x^{(i)}_{\text{real}}) + \log \left(1 - D(x^{(i)}_{\text{fake}})\right) \right]
$$
The generator loss using REINFORCE is:
$$
\mathcal{L}_G = -\frac{1}{B} \sum_{i=1}^{B} r^{(i)} \cdot \log p(x^{(i)}_{\text{fake}})
$$
where \( r^{(i)} \) is the reward from the discriminator.

---

## Reading Order

**Please read the files in numerical order:**

1. **project_4_file_1:** Start with the foundational GPT-2 implementation.
2. **project_4_file_2:** Next, explore the enhancements with the SwitchHead MOE mechanism.
3. **project_4_file_3:** Finally, study the GAN setup that integrates a discriminator to refine text generation.

This ordered progression will help you understand the evolution from a standard language model to an efficient, adversarially trained system.

*End of ReadMe*

