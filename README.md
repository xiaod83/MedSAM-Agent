
# MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning
[🤖 **Model**](https://huggingface.co/Saint-lsy/MedSAM-Agent-Qwen3-VL-8B-MedSAM2) | [**🤗 Dataset**](#) | [**📖 Paper**](https://arxiv.org/abs/2602.03320)
<p align="center">
  <img src="./assets/logo.png" alt="" width="120" height="140">
</p>

Shengyuan Liu<sup>1</sup> &emsp; Liuxin Bao<sup>1</sup> &emsp;
Qi Yang<sup>2,3</sup> &emsp; Wanting Geng<sup>2,4</sup> &emsp;
Boyun Zheng<sup>1</sup> &emsp; Chenxin Li<sup>1</sup> &emsp; 
Wenting Chen<sup>5</sup> Houwen Peng<sup>2✉</sup> 
Yixuan Yuan<sup>1✉</sup>

<sup>1</sup>Chinese University of Hong Kong &emsp; <sup>2</sup>Hunyuan Group, Tencent&emsp; <sup>3</sup>Institute of Automation, the Chinese Academy of Sciences &emsp;<sup>4</sup>Dalian University of Technology &emsp; <sup>5</sup>Stanford University&emsp;

<sup>✉</sup> Corresponding Author. 

## 🚀Overview
In this work, we propose MedSAM-Agent, a framework that reformulates interactive segmentation as a multi-step autonomous decision-making process. First, we introduce a hybrid prompting strategy for expert-curated trajectory generation, enabling the model to internalize human-like decision heuristics and adaptive refinement strategies. Furthermore, we develop a two-stage training pipeline that integrates multi-turn, end-to-end outcome verification with a clinical-fidelity process reward design to promote interaction parsimony and decision efficiency. 
![Main](assets/main.gif)

### ✨ Todo List
- [ ] Release the SFT and RL dataset for MedSAM-Agent.
- [ ] Release the code of trajectory generation.
- [x] Release the paper, model and the base code for MedSAM-Agent.

## Environment Setup
* We use python 3.11/CUDA 12.9/torch 2.8.0 for implementation.
* We train our models on 8 NVIDIA H20 GPUs with 96G memory.

```bash
# create environment
conda create -n msagent python=3.11 
conda activate msagent
pip install -r requirements.txt
```

## 📦Evaluation
### Model Download
This repository is configured for MedSAM2 segmentation. Please download MedSAM2 from:
- MedSAM2: [link](https://medsam2.github.io/)

Please also download the sam2 dependency repository and install it:
```bash
cd third_party/
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```
### Dataset Preparation
In this repo, our dataset is based on [BioMedParse](https://huggingface.co/datasets/microsoft/BiomedParseData) and [UniBioMed](https://huggingface.co/datasets/Luffy503/UniBiomed). We evaluate our model on 6 modalities and 21 datasets. Details of dataset split can be found in our paper.

We will release the SFT trajectory dataset and RL training dataset soon.
### Inference
- **Single sample (one image):**
Run the script in [infer/run_single_inference.py](infer/run_single_inference.py) with your paths:

```bash
cd infer
python run_single_inference.py \
  --img-path infer/demo/BTCV-0-106_CT_abdomen.png \
  --target-description "right kidney in abdomen CT" \
  --model-path /path/to/mllm_model \
  --seg-checkpoint /path/to/MedSAM2_latest.pt
```

- **Whole-dataset / multi-GPU:** Edit the variables at the top of [infer/run_batch_inference.sh](infer/run_batch_inference.sh): `MODEL_PATH` (local Qwen checkpoint or `gpt`), `MEDSAM2_CHECKPOINT`, `MEDSAM2_CONFIG`, `DATA_ROOT`, `DATASETS`, `SPLIT`, GPU topology (`N_GPUS`, `PROCESSES_PER_GPU`).

```bash
bash run_batch_inference.sh
```


## 🎈Acknowledgements
Greatly appreciate the tremendous effort for the following projects!
- [Verl](https://github.com/verl-project/verl)
- [Llama-Factory](https://github.com/hiyouga/LlamaFactory)
- [SAM2](https://github.com/facebookresearch/sam2)
- [MedSAM2](https://medsam2.github.io/)
- [UniBioMed](https://github.com/Luffy03/UniBiomed)
- [BioMedParse](https://github.com/microsoft/BiomedParse)
- [SegAgent](https://github.com/aim-uofa/SegAgent)

## 📜Citation
If you find this work helpful for your project, please consider citing our paper.
```
@misc{liu2026medsamagentempoweringinteractivemedical,
      title={MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning}, 
      author={Shengyuan Liu and Liuxin Bao and Qi Yang and Wanting Geng and Boyun Zheng and Chenxin Li and Wenting Chen and Houwen Peng and Yixuan Yuan},
      year={2026},
      eprint={2602.03320},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03320}, 
}
```

