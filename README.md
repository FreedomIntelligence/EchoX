# EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs

<!-- > **EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs**  
> Yuhao Zhang, Yuhao Du, Zhanchen Dai, et al. — *Under review at ICLR 2026*  
> 📄 [Paper (to be added)](https://arxiv.org/abs/XXXX.XXXX) -->

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache2.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/Model-8B%7C3B-orange.svg" alt="Model Size">
</p>

<p align="center">
   📄 <a href="https://arxiv.org/abs/2509.09174">Paper</a> |
   📦 <a href="https://huggingface.co/FreedomIntelligence/EchoX-8B">Model</a> | 
   🚀 <a href="https://huggingface.co/spaces/FreedomIntelligence/EchoX">HF Space</a> | 
   🌐 <a href="https://freedomintelligence.github.io/EchoX">Web Demo</a> | 
   📊 <a href="https://huggingface.co/datasets/FreedomIntelligence/EchoX-Dialougues">EchoX-Dialougues</a> | 
   📊 <a href="https://huggingface.co/datasets/KurtDu/EchoX-Dialogues-Plus">EchoX-Dialogues-Plus</a>
</p>



## Contents
- [EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs](#echox-towards-mitigating-acoustic-semantic-gap-via-echo-training-for-speech-to-speech-llms)
  - [Contents](#contents)
  - [Key Features](#key-features)
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

## Key Features
- Mitigates Acoustic-Semantic Gap in Speech-to-Speech LLMs
- Introduces Echo Training with a Novel Three-Stage Pipeline (S2T, T2C, Echo)
- Trained on Only 6k Hours of Curated Data, Ensuring Efficiency
- Achieves State-of-the-Art Performance in Knowledge-Based QA Benchmarks
- Preserves Reasoning and Knowledge Abilities for Interactive Speech Tasks

## Performance

<p align="center">
  <img src="asset/performance.png" alt="Performance" style="width:50%;">
</p>

EchoX demonstrates exceptional performance on knowledge-based question-answering tasks. The model achieves superior results with minimal training data, establishing a new benchmark for efficiency in speech-to-speech language models.

## Datasets and Models

### Dataset
EchoX is trained on carefully curated datasets for each stage of the pipeline, ensuring optimal performance across ASR, TTS, and SQA tasks. The datasets used are as follows:

| Task      | Data                | Size        | Duration(H) | Stage  | Download                                                                    |
| :-------- | :------------------ | :---------- | :---------- | :----- | :-------------------------------------------------------------------------- |
| ASR       | LibriSpeech         | 281,241     | 960         | I      | -                                                                           |
| ASR       | MLS                 | 723,636     | 3,000       | I      | -                                                                           |
| TTS       | AudioQA-1M          | 178,576     | 989         | II     | -                                                                           |
| TTS       | SpeechInstruct      | 31,563      | 84          | II     | -                                                                           |
| TTS       | HH-RLHF-Speech      | 124,945     | 656         | II     | -                                                                           |
| SQA       | sharechatx          | 43,223      | 178         | I, III | [Link](https://huggingface.co/datasets/KurtDu/EchoX-Dialogues) |
| SQA       | Magpie-Pro-Speech+   | 117,000     | 327         | I, III | [Link](https://huggingface.co/datasets/KurtDu/EchoX-Dialogues) |
| **Total** |                     | **1,500,184** | **6,194**   |        |                                                                             |

### Model
The following pre-trained models are available for download:

| Model        | Parameters | Training Data | Download Link                                      |
| ------------ | ---------- | ------------- | -------------------------------------------------- |
| **EchoX-3B** | 3 billion  | 6k hours     | [EchoX-3B Model](https://huggingface.co/FreedomIntelligence/EchoX-3B) |
| **EchoX-8B** | 8 billion  | 6k hours     | [EchoX-8B Model](https://huggingface.co/FreedomIntelligence/EchoX-8B) |

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
Download the models to this repository directory using the following commands:

```bash
pip install -U huggingface_hub
huggingface-cli download --resume-download FreedomIntelligence/EchoX-8B --local-dir EchoX-8B
huggingface-cli download --resume-download openai/whisper-large-v3 --local-dir whisper-large-v3
```

**Note**: If the models are downloaded to a different location or 3B version is used, please update the model directory paths in [inference/echox_stream.py](inference/echox_stream.py) and {your_EchoX_weight_directory}/config.json accordingly.

### Inference
Run inference on a test case:
```bash
python demo.py
```

Alternatively, start the Gradio web interface:
```bash
python app.py
```

To use a specific GPU:
```bash
CUDA_VISIBLE_DEVICES=1 python app.py
```

## Citation
If you use EchoX in your research or projects, please cite our paper:

```bibtex
@misc{zhang2025echoxmitigatingacousticsemanticgap,
      title={EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs}, 
      author={Yuhao Zhang and Yuhao Du and Zhanchen Dai and Xiangnan Ma and Kaiqi Kou and Benyou Wang and Haizhou Li},
      year={2025},
      eprint={2509.09174},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.09174}, 
}
```

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
