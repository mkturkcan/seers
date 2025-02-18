import llm
import seers
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.amp import GradScaler

# DataLoader configuration
batch_size = 32

# Load models
training_config = 'femto' # one of 'femto', 'tiny', 'small', 'dino'
if training_config == 'femto':
    llm_tokenizer, llm_model = llm.get_llm(
        "mehmetkeremturkcan/SmollerLM2-100M-Instruct-sft"
    )
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('WinKawaks/vit-tiny-patch16-224', use_peft=True)
if training_config == 'tiny':
    llm_tokenizer, llm_model = llm.get_llm(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('WinKawaks/vit-tiny-patch16-224', use_peft=True)
if training_config == 'small':
    llm_tokenizer, llm_model = llm.get_llm(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('WinKawaks/vit-small-patch16-224', use_peft=True)
if training_config == 'dino':
    llm_tokenizer, llm_model = llm.get_llm(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('facebook/dino-vitb16', use_peft=True)

# Load the datasets
train_dataset1 = vision.ImageDatasetCauldron('HuggingFaceM4/the_cauldron', image_processor, name = 'localized_narratives', split = 'train')
train_dataset2 = vision.ImageDatasetCauldron('HuggingFaceM4/the_cauldron', image_processor, name = 'vqav2', split = 'train')
train_dataset3 = vision.ImageDatasetCauldron('HuggingFaceM4/the_cauldron', image_processor, name = 'cocoqa', split = 'train')
train_dataset4 = vision.ImageDataset2('AnyModal/flickr30k', image_processor, split = 'train')
train_dataset5 = vision.ImageDataset("openbmb/RLAIF-V-Dataset", image_processor, split = 'train')
train_dataset = vision.CombinedImageDatasets([train_dataset5, train_dataset4, train_dataset3, train_dataset2, train_dataset1])
val_dataset = vision.ImageDataset2('AnyModal/flickr30k', image_processor, split = 'validation')

train_size = len(train_dataset)
val_size = len(val_dataset)

llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(train_size//5)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(torch.utils.data.Subset(val_dataset, range(val_size//5)), batch_size=batch_size, shuffle=True)

train_size = len(train_loader)
val_size = len(val_loader)
print(f"Train size: {train_size}, Validation size: {val_size}")


# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

# Initialize MultiModalModel
multimodal_model = femtovision.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    lm_peft = llm.add_peft,
    prompt_text="Describe this image briefly.")

# multimodal_model.language_model = llm.add_peft(multimodal_model.language_model)

# Training configuration
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimodal_model = multimodal_model.to(device)
multimodal_model.train()

# Optimizer
optimizer = schedulefree.AdamWScheduleFree(multimodal_model.parameters(), lr=3e-4)
optimizer.train()

scaler = GradScaler()

os.makedirs("image_captioning_model_temp", exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    training_losses = []
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = multimodal_model(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_losses.append(loss.item())
    
    avg_train_loss = sum(training_losses) / len(training_losses)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    multimodal_model.eval()
    validation_losses = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False):
            logits, loss = multimodal_model(batch)
            validation_losses.append(loss.item())
        
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # Decode a random validation sample
        for _ in range(5):
            sample_idx = np.random.randint(len(val_dataset))
            sample = val_dataset[sample_idx]
            print("Actual Text: ", sample['text'])
            print("Generated Text: ", multimodal_model.generate(sample['input'], max_new_tokens=120))
            
    multimodal_model.train()
    # Save the model
    multimodal_model._save_model("image_captioning_model_temp")




multimodal_model.eval()

for _ in range(5):
    sample_idx = np.random.randint(len(val_dataset))
    sample = val_dataset[sample_idx]
    
    # save the image with the caption and the generated caption
    image = sample['image']
    caption = sample['text']
    generated_caption = multimodal_model.generate(sample['input'], max_new_tokens=120)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"image_{sample_idx}.png")

    with open(f"image_{sample_idx}_caption.txt", "w") as f:
        f.write(f"Actual Caption: {caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")
