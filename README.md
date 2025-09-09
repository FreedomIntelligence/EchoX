# EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs

<!-- > **EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs**  
> Yuhao Zhang, Yuhao Du, Zhanchen Dai, et al. — *Under review at ICLR 2026*  
> 📄 [Paper (to be added)](https://arxiv.org/abs/XXXX.XXXX) -->

<p align="center">
   <a href="https://arxiv.org/abs/XXXX.XXXX">📄 Paper</a> |
   📦 Model Weights <a href="https://huggingface.co/FreedomIntelligence/EchoX-8B">8B</a> <a href="https://huggingface.co/FreedomIntelligence/EchoX-3B">3B</a> |
  <a href="https://huggingface.co/spaces/FreedomIntelligence/EchoX">🚀 HF Space</a> |
  <a href="https://freedomintelligence.github.io/EchoX">🌐 Web Demo</a>
</p>


## Contents
- [EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs](#echox-towards-mitigating-acoustic-semantic-gap-via-echo-training-for-speech-to-speech-llms)
  - [Contents](#contents)
  - [Highlights](#highlights)
  - [Method Overview](#method-overview)
  - [Performance](#performance)
  - [Datasets and Models](#datasets-and-models)
    - [Dataset](#dataset)
    - [Model](#model)
  - [Quickstart](#quickstart)
    - [Environment Setup](#environment-setup)
    - [Model Download](#model-download)
    - [Inference](#inference)
  - [Citation](#citation)
  - [License](#license)

## Highlights

<p align="center">
  <img src="asset/problem_figure.png" alt="Problem" style="width:90%;">
</div>

- **Problem**: Current speech-to-speech large language models (SLLMs) suffer from degraded reasoning and knowledge due to the conflicting focus on acoustic learning.
- **Solution**: EchoX integrates both acoustic and semantic learning by dynamically generating speech tokens from semantic representations. This mitigates the degradation problem, preserving the reasoning abilities of LLMs.
- **Impact**: EchoX achieves state-of-the-art performance on knowledge-based question-answering tasks while reducing data requirements to just 10k hours of training data.
- **Performance**: EchoX outperforms existing models in various benchmarks, delivering superior performance with a more efficient training setup.

## Method Overview

<p align="center">
  <img src="asset/method.png" alt="Method" style="width:80%;">
</div>

EchoX integrates both semantic and acoustic learning to address the challenges of current speech-to-speech models. We propose a novel three-stage training pipeline:

1. **Stage I: Speech-to-Text (S2T)**: Converts speech to text, enabling the model to capture semantic information from spoken inputs.
2. **Stage II: Text-to-Codec (T2C)**: Transforms text into speech tokens, bridging the gap between text-based LLMs and speech-based representations.
3. **Stage III: Echo Training**: Combines the outputs of S2T and T2C, training the model to generate speech from semantic understanding, preserving the core language model’s intelligence.

The integration of both speech and semantic learning ensures that EchoX can preserve the reasoning abilities of LLMs while improving speech-based tasks.

## Performance

<p align="center">
  <img src="asset/performance.png" alt="Performance" style="width:50%;">
</div>

EchoX has demonstrated exceptional performance in knowledge-based question-answering tasks. The model achieves superior results with minimal training data, setting a new benchmark for efficiency.

## Datasets and Models

### Dataset
EchoX is trained on carefully curated datasets for each stage of the pipeline, ensuring optimal performance across S2T, T2C, and S2S tasks. The datasets used are as follows:

| Dataset Type           | Description                                                              | Duration                   | Download Link                                      |
| ---------------------- | ------------------------------------------------------------------------ | ---------------------- | -------------------------------------------------- |
| **Speech-to-Text (S2T)** | Multi-turn dialog datasets processed for speech-to-text tasks             | 810 hours              | [Link to S2T Dataset](https://huggingface.co/datasets/FreedomIntelligence/EchoX-Dataset) |
| **Text-to-Codec (T2C)** | Text-to-speech data with paired codec tokens                              | 40 hours               | [Link to T2C Dataset](https://huggingface.co/datasets/FreedomIntelligence/EchoX-Dataset) |
| **Speech-to-Speech (S2S)** | Conversational and reasoning-based speech datasets for generating responses | 150 hours              | [Link to S2S Dataset](https://huggingface.co/datasets/FreedomIntelligence/EchoX-Dataset) |

### Model
The following pre-trained models are available for download:

| Model     | Parameters | Training Data Duration   | Download Link                                      |
| --------- | ---------- | --------------- | -------------------------------------------------- |
| **EchoX-3B** | 3 billion  | 10k hours       | [EchoX-3B Model](https://huggingface.co/FreedomIntelligence/EchoX-3B) |
| **EchoX-8B** | 8 billion  | 10k hours       | [EchoX-8B Model](https://huggingface.co/FreedomIntelligence/EchoX-8B) |

## Quickstart

### Environment Setup
To set up your environment, follow these steps:
```bash
git clone https://github.com/FreedomIntelligence/EchoX.git
cd EchoX
conda create -n echox python=3.10 pip=24.0
conda activate echox
pip install -r requirements.txt
```

### Model Download
Download the models in this repo directory with the following command:

```bash
pip install -U huggingface_hub
hf download --resume-download FreedomIntelligence/EchoX-8B --local-dir EchoX-8B
hf download --resume-download openai/whisper-large-v3 --local-dir whisper-large-v3
```

If they are downloaded elsewhere, directories in [inference/echox_stream.py](inference/echox_stream.py) of this two models should be modified.

### Inference
Run inference on a test case with the following command:
```bash
python demo.py
```

Or you can start Gradio web page with the following command:
```bash
python app.py
```

You can assign the specific GPU with the following command:
```bash
CUDA_VISIBLE_DEVICES=1 python app.py
```

## Citation
If you use EchoX in your research or projects, please cite our paper:

```bibtex
@inproceedings{zhang2026echox,
  title={EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs},
  author={Zhang, Yuhao and Du, Yuhao and Dai, Zhanchen and others},
  booktitle={Proceedings of ICLR 2026},
  year={2026},
  url={https://arxiv.org/abs/XXXX.XXXX}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
