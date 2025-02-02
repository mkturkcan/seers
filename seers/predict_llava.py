import llm
import seers
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
import matplotlib
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download, snapshot_download
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load models
training_config = 'base' # one of 'tiny', 'small', 'dino'
if training_config == 'tiny':
    llm_tokenizer, llm_model = llm.get_llm(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device=device
    )
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', device=device, use_peft=False)
    llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model, device=device)
if training_config == 'base':
    llm_tokenizer, llm_model = llm.get_llm(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device=device
    )
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-large-patch32-384', device=device, use_peft=False)
    llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model, device=device)
    checkpoint_folder = "DeepSeek-LLaVA-Instruct"

# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model, device=device)

# Initialize MultiModalModel
multimodal_model = seers.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1),
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    prompt_text="Describe this image in one or two sentences.")

# Load the model

multimodal_model._load_model(checkpoint_folder, device=device)

# Dataset configuration
dataset_name = "AnyModal/flickr30k"
ds = vision.ImageDataset2(dataset_name, image_processor, split = 'test')

multimodal_model.eval()

os.makedirs("temp", exist_ok=True)

for _ in range(5):
    sample_idx = np.random.randint(len(ds))
    sample = ds[sample_idx]
    
    # save the image with the caption and the generated caption
    image = sample['image']
    caption = sample['text']
    generated_caption = multimodal_model.generate(sample['input'], max_new_tokens=2048)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"temp/image_{sample_idx}.png")

    with open(f"temp/image_{sample_idx}_caption.txt", "w") as f:
        f.write(f"Actual Caption: {caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")

imgs = [
    {
        'url': 'https://img.freepik.com/free-photo/people-posing-together-registration-day_23-2149096794.jpg',
        'meta': 'People'
    },
    {
        'url': 'https://discoverymood.com/wp-content/uploads/2019/01/iStock-629076332.jpg',
        'meta': 'Dog'
    }
]

for idx, cartoon in enumerate(imgs):
    # download the image
    response = requests.get(cartoon['url'])
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')

    # save the image
    img.save(f"web_image_{idx}.png")

    # process the image
    image = image_processor(img, return_tensors="pt")
    image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

    # generate the caption
    generated_caption = multimodal_model.generate(image, max_new_tokens=120)

    # save the caption
    with open(f"web_image_{idx}_caption.txt", "w") as f:
        f.write(f"Meta: {cartoon['meta']}\n")
        f.write(f"Generated Caption: {generated_caption}\n")

