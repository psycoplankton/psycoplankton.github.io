---
layout: page
title: "Enriching Pre-Training Using Fuzzy Logic"
description: "Novel fuzzy logic pre-processing layer that models semantic uncertainty in graph representation learning (IEEE FUZZ-IEEE 2025)."
img: assets/img/projects/fuzzy_pretraining_fig1.png
importance: 1
category: work
related_publications: false
---

<div class="row mb-3">
  <div class="col-sm-12 text-center">
    <a href="/assets/pdf/IEEE_Fuzz_Enriching_Pre_Training_Using_Fuzzy_Logic.pdf" class="btn btn-sm z-depth-0 btn-outline-primary" role="button" target="_blank">
      <i class="fa-solid fa-file-pdf"></i> Read Paper (PDF)
    </a>
    <a href="https://github.com/psycoplankton/Fuzzy-Representation-Learning" class="btn btn-sm z-depth-0 btn-outline-primary" role="button" target="_blank">
      <i class="fa-brands fa-github"></i> GitHub Repository
    </a>
    <span class="badge badge-secondary p-2 ml-2" style="font-size: 0.85rem;">IEEE FUZZ-IEEE 2025</span>
  </div>
</div>

---

### Overview & Motivation

Graph Representation Learning (GRL) traditionally relies on structural and topological properties—such as random walks in **DeepWalk** and **Node2Vec** or neighborhood aggregation in Graph Neural Networks. However, purely structural embeddings often fail to capture **semantic ambiguity**, **latent uncertainty**, and **contextual nuances** inherent in real-world graphs (e.g., noisy citation networks, gene interactions, or social media graphs).

In this work, we propose a plug-and-play **fuzzy logic-based pre-processing layer** that enriches existing node representations before downstream tasks. By modeling uncertainty through fuzzy membership functions and defuzzifying them back into high-dimensional vector spaces, the framework bridges structural topology with semantic reasoning.

<div class="text-center my-4">
  <figure class="figure d-inline-block p-2 rounded" style="background-color: #ffffff; max-width: 100%;">
    <img src="/assets/img/projects/fuzzy_pretraining_fig1.png" class="figure-img img-fluid rounded mb-1" alt="Figure 1: Generation of new embedding set by introducing a fuzzy layer" style="max-height: 420px;">
    <figcaption class="figure-caption text-dark font-weight-bold">
      Figure 1: Generation of new embedding set by introducing the fuzzy pre-processing layer.
    </figcaption>
  </figure>
</div>

---

### Methodology & Architecture

The architecture consists of an end-to-end three-stage pipeline that integrates seamlessly with static or dynamic graph embedding algorithms:

<div class="text-center my-4">
  <figure class="figure d-inline-block p-2 rounded" style="background-color: #ffffff; max-width: 100%;">
    <img src="/assets/img/projects/fuzzy_pretraining_fig2_workflow.png" class="figure-img img-fluid rounded mb-1" alt="Figure 2: Training process workflow" style="max-height: 280px;">
    <figcaption class="figure-caption text-dark font-weight-bold">
      Figure 2: End-to-end training process workflow integrating the fuzzy pre-processing layer with GraphMamba.
    </figcaption>
  </figure>
</div>

1. **Initial Node Embedding**:
   - Baseline structural embeddings $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\} \subset \mathbb{R}^d$ are generated using random-walk algorithms (**Node2Vec** or **DeepWalk**).
2. **Fuzzy Pre-Processing Layer**:
   - **K-Means Node Clustering**: Groups embeddings in latent space into $K$ semantic clusters, determining cluster centroids $\boldsymbol{\mu}_k$ and variances $\boldsymbol{\sigma}_k^2$.
   - **Antecedent Computation**: Transforms each node vector dimension into a degree of membership using a Gaussian membership function:
     $$\mu_z(x_i) = \exp\left( -\frac{(x_i^z - \mu_m^z)^2}{2\sigma_z^2} \right)$$
     The dimension-wise memberships are averaged to yield a representative mean membership $\mu_i^{\text{mean}}$ across clusters.
   - **Consequent Phase (Defuzzification)**: Reconstructs crisp, semantically enriched embedding coordinates $y_i$ by inverting the Gaussian distribution:
     $$y_i = \sqrt{-\ln(\mu_i^{\text{mean}})} \cdot \sigma_k + \mu_k$$
3. **Downstream GraphMamba Architecture**:
   - Enriched embeddings $\mathbf{X}_{\kappa}^{\text{Fuzzy}}$ are fed into **GraphMamba** (selective state-space model equipped with 2 GraphGPS layers, 64 channels, and 4D positional encodings) for link prediction.
   - **Time Complexity**: The layer operates in $\mathcal{O}(N^2 \cdot K) + \mathcal{O}(K \cdot d \cdot N^2) + \mathcal{O}(d \cdot N)$, scaling cleanly for graph networks.

---

### Experimental Evaluation

The framework was evaluated on two benchmark datasets across varying degrees of synthetic Gaussian perturbation ($\kappa \in [0\%, 10\%]$) to simulate real-world noise:
- **CA-GrQc** (Arxiv Collaboration Network in General Relativity & Quantum Cosmology): $5{,}242$ nodes, $14{,}496$ edges ($K=5$ optimal clusters).
- **BIOGRID-Human** (Human Gene & Protein Interaction Network): $9{,}527$ nodes, $62{,}364$ edges ($K=5$ optimal clusters).

#### Quantitative Results (Table 1)

<div class="text-center my-3">
  <figure class="figure d-inline-block p-2 rounded" style="background-color: #ffffff; max-width: 100%;">
    <img src="/assets/img/projects/fuzzy_table1_results.png" class="figure-img img-fluid rounded mb-1" alt="Table 1: Evaluation Metrics across Noise Levels" style="max-height: 480px;">
    <figcaption class="figure-caption text-dark font-weight-bold">
      Table 1: Comprehensive evaluation metrics for DeepWalk and Node2Vec on CA-GrQc and BIOGRID-Human datasets under 0% to 10% Gaussian noise.
    </figcaption>
  </figure>
</div>

#### Performance Under Severe Noise

<div class="row my-4">
  <div class="col-sm-6 text-center mb-3">
    <figure class="figure d-inline-block p-2 rounded" style="background-color: #ffffff; width: 100%;">
      <img src="/assets/img/projects/fuzzy_perf_biogrid_deepwalk.png" class="figure-img img-fluid rounded mb-1" alt="BIOGRID-Human DeepWalk F1 Performance">
      <figcaption class="figure-caption text-dark font-weight-bold">
        BIOGRID-Human + DeepWalk Performance vs Noise
      </figcaption>
    </figure>
  </div>
  <div class="col-sm-6 text-center mb-3">
    <figure class="figure d-inline-block p-2 rounded" style="background-color: #ffffff; width: 100%;">
      <img src="/assets/img/projects/fuzzy_perf_cagrqc_node2vec.png" class="figure-img img-fluid rounded mb-1" alt="CA-GrQc Node2Vec F1 Performance">
      <figcaption class="figure-caption text-dark font-weight-bold">
        CA-GrQc + Node2Vec Performance vs Noise
      </figcaption>
    </figure>
  </div>
</div>

---

### Key Findings & Contributions

1. **Consistent Performance Gains**: Across both datasets, the fuzzy pre-processing layer delivers steady gains of **+2% to +2.5%** in Accuracy and F1 score under standard non-noisy regimes.
2. **Noise Resilience**: The layer demonstrates remarkable robustness at higher noise levels ($8\% - 10\%$). On the BIOGRID-Human dataset at 10% noise with Node2Vec, the fuzzy layer achieves **97.24% Accuracy** and **98.71% Recall** compared to **92.83%** and **88.82%** without fuzzy pre-training (**+4.4% accuracy boost** and **+9.9% recall boost**).
3. **Versatile & Model-Agnostic**: Can be directly inserted as a drop-in pre-processing module for any static or dynamic graph representation learning algorithm.

---

### Citation

```bibtex
@inproceedings{gupta2025fuzzy,
  author    = {Gupta, Vansh and Bharti, Vandana and Kumar, Abhinav and Sharma, Anshul and Singh, Sanjay Kumar},
  title     = {Enriching Pre-Training Using Fuzzy Logic},
  booktitle = {IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},
  year      = {2025}
}
```
