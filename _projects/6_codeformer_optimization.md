---
layout: page
title: "CodeFormer Inference Optimization"
description: "Machine Learning Engineer Intern at BingeClip.Ai: Accelerating face restoration inference via batching, PyTorch FX static quantization, and ONNX."
importance: 2
category: work
redirect: https://github.com/psycoplankton/CodeFormer_optimization
---

As a **Machine Learning Engineer Intern at [BingeClip.Ai](https://www.bingeclip.ai/)**, I optimized the inference pipeline for the [CodeFormer](https://shangchenzhou.com/projects/CodeFormer/) face restoration architecture.

### Optimization Strategies
1. **Batch Inference**: Refactored the 'one-image-at-a-time' pipeline into dynamic batching to maximize GPU compute utilization and eliminate redundant Python loop overhead.
2. **Post-Training Static Quantization**: Quantized the model weights using PyTorch's FX Graph Mode Quantization API, reducing memory footprint and boosting inference throughput with negligible quality degradation.
3. **ONNX Runtime Export**: Exported the transformer and neural modules into optimized ONNX computation graphs with graph kernel fusions.

- **Repository**: [github.com/psycoplankton/CodeFormer_optimization](https://github.com/psycoplankton/CodeFormer_optimization)

