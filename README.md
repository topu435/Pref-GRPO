<div align="center">
    <h1 align="center"> Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning
    </h1>




<a href="">
<img src='https://img.shields.io/badge/arXiv-Pref GRPO-blue' alt='Paper PDF'></a>

<a href="https://codegoat24.github.io/UnifiedReward/Pref-GRPO">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
<a href="https://github.com/CodeGoat24/UniGenBench">
<img src='https://img.shields.io/badge/Benchmark-UniGenBench-green' alt='Project Page'></a>

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/CodeGoat24/FLUX.1-dev-PrefGRPO) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-yellow)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard)

</div>

![pref_grpo_pipeline](/assets/pref_grpo_pipeline.png)



![pref_grpo_pipeline](/assets/pref_grpo_reward_hacking.png)



## 🔧 Environment Set Up
1. Clone this repository and navigate to the folder:
```bash
git clone https://github.com/CodeGoat24/UnifiedReward.git
cd UnifiedReward/Pref-GRPO
```

2. Install the training package:
```bash
conda create -n PrefGRPO python=3.12
conda activate PrefGRPO

bash env_setup.sh fastvideo

cd open_clip
pip install -e .
cd ..
```

3. Download Models
```bash
huggingface-cli download CodeGoat24/UnifiedReward-qwen-7b
huggingface-cli download CodeGoat24/UnifiedReward-Think-qwen-7b

wget https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/resolve/main/open_clip_pytorch_model.bin
```
## 💻 Training

#### 1. Deploy vLLM server

1. Install vLLM
```bash
pip install vllm==0.9.0.1 transformers==4.52.4
```
2. Start server
```bash
bash vllm_utils/vllm_server_UnifiedReward_Think.sh  
```
#### 2. Preprocess training Data 
we use training prompts in [UniGenBench](https://github.com/CodeGoat24/UniGenBench), as shown in ```"./data/unigenbench_train_data.txt"```.

```bash
bash fastvideo/data_preprocess/preprocess_flux_rl_embeddings.sh
```


#### 3. Train
```bash
bash finetune_prefgrpo_flux.sh
```

### 🚀 Inference and Evaluation
we use test prompts in [UniGenBench](https://github.com/CodeGoat24/UniGenBench), as shown in ```"./data/unigenbench_test_data.csv"```.
```bash
bash inference/flux_dist_infer.sh
```

Then, evaluate the outputs following [UniGenBench](https://github.com/CodeGoat24/UniGenBench).


## 📧 Contact
If you have any comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).


## 🤗 Acknowledgments
Our training code is based on [DanceGRPO](https://github.com/XueZeyue/DanceGRPO), [Flow-GRPO](https://github.com/yifan123/flow_grpo), and [FastVideo](https://github.com/hao-ai-lab/FastVideo).

We also use [UniGenBench](https://github.com/CodeGoat24/UniGenBench) for T2I model semantic consistency evaluation.

Thanks to all the contributors!


## ⭐ Citation
```bibtex

```
