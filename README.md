# EchoX: Rethinking the Training Paradigm for Speech-to-Speech Large Language Models 

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](#environment-setup)
[![Framework](https://img.shields.io/badge/PyTorch-v1.10-orange)](#environment-setup)
[![Status](https://img.shields.io/badge/Status-WIP-yellow)]()

> Official implementation for the paper  
> **EchoX: Rethinking the Training Paradigm for Speech-to-Speech Large Language Models**  
> Yuhao Zhang, Yuhao Du, Zhanchen Dai, et al. — *Under review at ICLR 2026*  
> 📄 [Paper (to be added)](https://arxiv.org/abs/XXXX.XXXX)

---

## Table of Contents 📋
- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Performance](#performance)
- [Datasets and Models](#datasets-and-models)
  - [Dataset](#dataset)
  - [Model](#model)
- [Quickstart](#quickstart)
  - [Environment Setup](#environment-setup)
  - [Model Download](#model-download)
  - [Training](#training)
  - [Inference](#inference)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [License](#license)

---

## Highlights ⚡

![Figure 1: The training difference between three types of models.](asset/problem_figure.png)

- **Problem**: Current speech-to-speech large language models (SLLMs) suffer from degraded reasoning and knowledge due to the conflicting focus on acoustic learning.
- **Solution**: EchoX integrates both acoustic and semantic learning by dynamically generating speech tokens from semantic representations. This mitigates the degradation problem, preserving the reasoning abilities of LLMs.
- **Impact**: EchoX achieves state-of-the-art performance on knowledge-based question-answering tasks while reducing data requirements to just 10,000 hours of training data.
- **Performance**: EchoX outperforms existing models in various benchmarks, delivering superior performance with a more efficient training setup.

---

## Method Overview 🛠️

![Figure 1: The training difference between three types of models.](asset/method.png)

EchoX integrates both semantic and acoustic learning to address the challenges of current speech-to-speech models. We propose a novel three-stage training pipeline:

1. **Stage I: Speech-to-Text (S2T)**: Converts speech to text, enabling the model to capture semantic information from spoken inputs.
2. **Stage II: Text-to-Codec (T2C)**: Transforms text into speech tokens, bridging the gap between text-based LLMs and speech-based representations.
3. **Stage III: Echo Training**: Combines the outputs of S2T and T2C, training the model to generate speech from semantic understanding, preserving the core language model’s intelligence.

The integration of both speech and semantic learning ensures that EchoX can preserve the reasoning abilities of LLMs while improving speech-based tasks.

---

## Performance ⚡

![Figure 3: Model comparison on the knowledge QA benchmark.](asset/performance.png)

EchoX has demonstrated exceptional performance in knowledge-based question-answering tasks. The model achieves superior results with minimal training data, setting a new benchmark for efficiency.

---

## Datasets and Models 📚

### Dataset
EchoX is trained on carefully curated datasets for each stage of the pipeline, ensuring optimal performance across S2T, T2C, and S2S tasks. The datasets used are as follows:

| Dataset Type           | Description                                                              | Size                   | Download Link                                      |
| ---------------------- | ------------------------------------------------------------------------ | ---------------------- | -------------------------------------------------- |
| **Speech-to-Text (S2T)** | Multi-turn dialog datasets processed for speech-to-text tasks             | 810 hours              | [Link to S2T Dataset](https://huggingface.co/datasets/FreedomIntelligence/EchoX-Dataset) |
| **Text-to-Codec (T2C)** | Text-to-speech data with paired codec tokens                              | 40 hours               | [Link to T2C Dataset](https://huggingface.co/datasets/FreedomIntelligence/EchoX-Dataset) |
| **Speech-to-Speech (S2S)** | Conversational and reasoning-based speech datasets for generating responses | 150 hours              | [Link to S2S Dataset](https://huggingface.co/datasets/FreedomIntelligence/EchoX-Dataset) |

### Model
The following pre-trained models are available for download:

| Model     | Parameters | Training Data   | Download Link                                      |
| --------- | ---------- | --------------- | -------------------------------------------------- |
| **EchoX-3B** | 3 billion  | 10K hours       | [EchoX-3B Model](https://huggingface.co/FreedomIntelligence/EchoX) |
| **EchoX-8B** | 8 billion  | 10K hours       | [EchoX-8B Model](https://huggingface.co/FreedomIntelligence/EchoX) |

---

## Quickstart 🚀

### Environment Setup
To set up your environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/EchoX.git
   cd EchoX

2. **Create and activate a virtual environment**:

   ```bash
   conda create -n echox python=3.10
   conda activate echox
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Model Download

Download the pretrained models using the following command:

```bash
git lfs clone https://huggingface.co/your-model-repo
```

### Training

To start training, configure your settings and execute the following:

```bash
python train.py --config configs/train/echox_3B.yaml
```

For multi-GPU training:

```bash
torchrun --nproc_per_node 8 train.py --config configs/train/echox_3B.yaml
```

### Inference

Run inference on a test audio file with the following command:

```bash
python infer.py --checkpoint checkpoints/echox_3B.pt --input data/input.wav --output outputs/output.wav
```

---

## Acknowledgments 🙏

We acknowledge the contributions of the open-source community, especially for the datasets, model implementations, and frameworks used in EchoX. Special thanks to Hugging Face for hosting the models and datasets.

---

## Citation 📑

If you use EchoX in your research or projects, please cite our paper:

```bibtex
@inproceedings{chen2026echox,
  title={EchoX: Rethinking the Training Paradigm for Speech-to-Speech Large Language Models},
  author={Chen, Kai and Gou, Yunhao and Huang, Runhui and others},
  booktitle={Proceedings of ICLR 2026},
  year={2026},
  url={https://arxiv.org/abs/XXXX.XXXX}
}
```

---

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
