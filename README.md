
# VidKV: Plug-and-Play _1.x-Bit_ KV Cache Quantization for Video Large Language Models

[Keda Tao](), [Haoxuan You](https://hxyou.github.io/), [Yang Sui](https://eclipsess.github.io/yangsui.github.io/), [Can Qin](https://canqin.tech/), [Huan Wang](https://huanwang.tech/), "Plug-and-Play _1.x-Bit_ KV Cache Quantization for Video Large Language Models"

[[Paper](https://arxiv.org/abs/2503.16257)]

#### 🔥🔥🔥 News

- **2025-3-21:** This repo is released.
- **2025-3-21**: The paper is released.

![overview](figures/method.png)


> **Abstract:** Video large language models (VideoLLMs) have demonstrated the capability to process longer video inputs and enable complex reasoning and analysis. However, due to the thousands of visual tokens from the video frames, key-value (KV) cache can significantly increase memory requirements, becoming a bottleneck for inference speed and memory usage. KV cache quantization is a widely used approach to address this problem. In this paper, we find that 2-bit KV quantization of VideoLLMs can hardly hurt the model performance, while the limit of KV cache quantization in even lower bits has not been investigated. To bridge this gap, we introduce VidKV, a plug-and-play KV cache quantization method to compress the KV cache to lower than 2 bits. Specifically, (1) for key, we propose a mixed-precision quantization strategy in the channel dimension, where we perform 2-bit quantization for anomalous channels and 1-bit quantization combined with FFT for normal channels; (2) for value, we implement 1.58-bit quantization while selectively filtering semantically salient visual tokens for targeted preservation, for a better trade-off between precision and model performance. Importantly, our findings suggest that the value cache of VideoLLMs should be quantized in a per-channel fashion instead of the per-token fashion proposed by prior KV cache quantization works for LLMs. Empirically, extensive results with LLaVA-OV-7B and Qwen2.5-VL-7B on six benchmarks show that VidKV effectively compresses the KV cache to 1.5-bit and 1.58-bit precision with almost no performance drop compared to the FP16 counterparts.
> 

## ⚒️ TODO

* [x] Release Paper 
* [x] Release code 
* [ ] Support more models

## Install
##### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/KD-TAO/VidKV.git
cd VidKV
```

##### 2. **Install the inference package:**
```bash
conda create -n vidkv python=3.10 -y
conda activate vidkv
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
# Transformers
bash env_setup.sh
```

## VidKV User Guidance

We implemented the VidKV quantization framework based on the **transformers** library. 
##### 1. **Meet one of the following two requirements:**
- Use our vidkv environment.
- Install the supplied version of the **transformers** library in your environment.
```bash
cd transformers
pip install .
cd ..
```
##### 2. **Parameter setting and generation:**

```python
# Firstly, you need the QUANTIZATION_CONFIG
### -> 2-bit quantization:
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheLM', 'nbits': 2, 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### -> 1.5-bit and 2-bit quantization: [1.5, 2] means K-1.5-bit & V-2-bit
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheLM', 'nbits': [1.5, 2], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### -> 1.5-bit and 1.58-bit quantization:
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheVLM', 'nbits': [1.5, 1.58], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### -> Quantization with STP: 
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheVLMSTP', 'nbits': [1.5, 1.58], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1, "vidkv_stp": 0.2}"
'''
# Now QuantizedCacheVLMSTP only supports the llava-onevision model. Other models we will support later

# Secondly, you can use during model generation
out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config=QUANTIZATION_CONFIG)
```

## Demo
- We provide a demo script for inference using the LLaVA-OV model:
```bash
python demo.py
```
- You can change the quantization Settings here:
```python
...
outputs = model.generate(
input_ids,
attention_mask=attention_mask, # Add attention mask
images=image_tensors,
image_sizes=image_sizes,
do_sample=False,
temperature=0,
max_new_tokens=1000,
modalities=["video"],
output_attentions=False,
use_cache = True,
return_dict_in_generate=True,
output_hidden_states=True,
cache_implementation="quantized",
# VidKV
cache_config={"backend": "QuantizedCacheVLM", "nbits": [1.5, 1.58], "q_group_size": 32, "residual_length": 128,"axis_key":-1, "axis_value":-1},
)
...
```

## Evaluation
The evaluation guidance is coming soon.
## Contact

If you have any questions, please feel free to contact with me at KD.TAO@outlook.com
## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{vidkv,
  title={Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models},
  author={Tao, Keda and You, Haoxuan and Sui, Yang and Qin, Can and Wang, Huan},
  journal={arXiv preprint arXiv:2503.16257},
  year={2025}
}
```