---
layout: page
title: "GraphGANs & Fuzzy Neural Networks"
description: "Visual Computing and Analytics Lab, IIT BHU: Incorporating fuzzy logic preprocessing into GraphGANs and node embeddings."
importance: 1
category: work
---

I worked at the **Visual Computing and Analytics Lab, IIT BHU**, for 9 months researching GraphGANs and Fuzzy Neural Networks.

### Problem Statement
GraphGANs use node embeddings as input to generate realistic graphs. The higher the quality and semantic richness of the embeddings, the better the generated graph topology. Real-world graph data is inherently noisy and semantically uncertain. Fuzzy logic incorporates these uncertainties and provides a natural formulation for learning robust representations.

### Solution
- Modeled a fuzzy pre-processing layer based on **TSK Fuzzy Logic Systems** integrated into node embedding generation algorithms.
- The layer assigns fuzzy membership values across embedding dimensions and performs defuzzification to output crisp embeddings.
- Evaluated on **Ca-GrQc** and **Biogrid-human** datasets using [Node2Vec](https://arxiv.org/abs/1607.00653), [DeepWalk](https://arxiv.org/abs/1403.6652), and [Struc2Vec](https://arxiv.org/abs/1704.03165).

