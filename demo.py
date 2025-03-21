from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import os
import torch
import cv2
import numpy as np
from PIL import Image
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import requests
import copy
import warnings
from decord import VideoReader, cpu
warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "llava-onevision-qwen2-7b-ov"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", torch_dtype="float16")

model.eval()

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Load and process video
video_path = "YOUR_VIDEO_PATH"
video_frames = load_video(video_path, 32)
print(video_frames.shape) # (16, 1024, 576, 3)

image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
CUDA_LAUNCH_BLOCKING=1
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()


input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# Create attention mask (1 for real tokens, 0 for padding)
attention_mask = torch.ones_like(input_ids)
image_sizes = [frame.size for frame in video_frames]

outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,  # Add attention mask
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
    cache_config={"backend": "QuantizedCacheVLM", "nbits": [1.5, 1.58], "q_group_size": 32, "residual_length": 128,"axis_key":-1, "axis_value":-1},
)

cont = outputs.sequences
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
