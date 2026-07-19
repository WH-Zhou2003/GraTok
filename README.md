# GraTok
# GraTok: 3D Spatio-Temporal Graph Token Compression for Video MLLMs

Welcome to **GraTok**!

GraTok is a **training-free 3D spatio-temporal graph-based token compression method** designed for efficient video multimodal large language models (Video MLLMs). It reduces redundant visual tokens while preserving essential spatial and temporal information, enabling efficient video understanding with reduced computational cost.

## Overview

The overall pipeline of GraTok is illustrated below:

<p align="center">
  <img src="assets/gratok_overview.png" width="90%">
</p>

GraTok constructs a unified spatio-temporal graph over visual tokens and performs importance-aware iterative token merging. The proposed method can be directly applied to existing Video MLLMs without additional training.

---

# Environment

GraTok is tested under the following environment:

- Python >= 3.10.0
- PyTorch
- Transformers
- lmms-eval

This project relies on **lmms-eval** for evaluation.

For the complete evaluation framework, supported models, and benchmark implementation, please refer to:

https://github.com/EvolvingLMMs-Lab/lmms-eval

If you would like to use the complete evaluation pipeline, please follow the installation instructions provided by lmms-eval.

---

# Evaluation

## Run Evaluation

We provide evaluation scripts for different Video MLLM backbones.

### Qwen3-VL

```bash
bash examples/models/qwen3vl_compress.sh
