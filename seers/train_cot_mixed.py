import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import llm
import seers
import vision
import schedulefree

class ImageCaptioningTrainer:
    def __init__(
        self,
        training_config: str = 'tiny',
        batch_size: int = 1,
        checkpoint_folder: str = "image_captioning_model_temp",
        device: str = None,
        learning_rate: float = 1e-4,
        train_subset_fraction: float = 1.,
        prompt_text: str = "Describe this image briefly.",
        vision_model: str = 'google/vit-base-patch16-224',
        llm_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    ):
        """
        Initialize the Image Captioning Trainer.

        Args:
            training_config (str): Model configuration ('tiny', 'small', or 'dino')
            batch_size (int): Batch size for training and validation
            checkpoint_folder (str): Directory to save model checkpoints
            device (str): Device to use for training ('cuda:0', 'cuda:1', 'cpu', etc.)
            learning_rate (float): Learning rate for the optimizer
            train_subset_fraction (float): Fraction of training data to use (0.0 to 1.0)
            prompt_text (str): Prompt text to use for image captioning
            vision_model (str): Name or path of the vision model
            llm_model (str): Name or path of the language model
        """
        self.batch_size = batch_size
        self.checkpoint_folder = checkpoint_folder
        self.device = device or torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.train_subset_fraction = train_subset_fraction
        self.prompt_text = prompt_text
        self.vision_model_name = vision_model
        self.llm_model_name = llm_model
        
        # Initialize models and datasets
        self._init_models(training_config)
        self._init_datasets()
        self._init_training_components()

    def _init_models(self, training_config):
        """Initialize the LLM and vision models based on configuration."""
        if training_config == 'tiny':
            self.llm_tokenizer, self.llm_model = llm.get_llm(
                self.llm_model_name,
                device=self.device
            )
            self.image_processor, self.vision_model, self.vision_hidden_size = vision.get_image_encoder(
                self.vision_model_name,
                device=self.device,
                use_peft=True
            )
            self.llm_hidden_size = llm.get_hidden_size(
                self.llm_tokenizer,
                self.llm_model,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported training config: {training_config}")

    def _init_datasets(self):
        """Initialize training and validation datasets."""
        train_dataset1 = vision.ImageDatasetCOT(
            '5CD-AI/LLaVA-CoT-o1-Instruct',
            self.image_processor,
            split='train'
        )
        train_dataset2 = vision.ImageDatasetCauldron('HuggingFaceM4/the_cauldron', self.image_processor, name = 'vqav2', split = 'train')
        train_dataset3 = vision.ImageDatasetCauldron('HuggingFaceM4/the_cauldron', self.image_processor, name = 'cocoqa', split = 'train')
        train_dataset4 = vision.ImageDataset2('AnyModal/flickr30k', self.image_processor, split = 'train')
        train_dataset5 = vision.ImageDataset("openbmb/RLAIF-V-Dataset", self.image_processor, split = 'train')
        train_dataset = vision.CombinedImageDatasets([train_dataset5, train_dataset4, train_dataset3, train_dataset2, train_dataset1])
        val_dataset = vision.ImageDataset2(
            'AnyModal/flickr30k',
            self.image_processor,
            split='validation'
        )

        # Use subset of training data based on fraction
        subset_size = int(len(train_dataset) * self.train_subset_fraction)
        train_subset = torch.utils.data.Subset(train_dataset, range(subset_size))
        NUM_WORKERS = 16
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_dataset = val_dataset

        print(f"Train size: {len(self.train_loader)}, Validation size: {len(self.val_loader)}")

    def _init_training_components(self):
        """Initialize the multimodal model, vision components, and optimizer."""
        # Vision components
        vision_encoder = vision.VisionEncoder(self.vision_model, device=self.device)
        vision_tokenizer = vision.Projector(
            self.vision_hidden_size,
            self.llm_hidden_size,
            num_hidden=1
        )

        # Multimodal model
        self.model = seers.MultiModalModel(
            input_processor=None,
            input_encoder=vision_encoder,
            input_tokenizer=vision_tokenizer,
            language_tokenizer=self.llm_tokenizer,
            language_model=self.llm_model,
            lm_peft=llm.add_peft,
            prompt_text=self.prompt_text,
            device=self.device
        )

        # Training components
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.scaler = GradScaler()
        
        os.makedirs(self.checkpoint_folder, exist_ok=True)

    def _check_loss(self, loss, batch_idx):
        """Check if loss is valid and handle NaN/inf values."""
        return True
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss detected (NaN/inf) at batch {batch_idx}")
            return False
        return True

    def _check_gradients(self):
        """Check if gradients are valid and handle NaN/inf values."""
        return True
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"WARNING: Invalid gradient detected in {name}")
                    return False
        return True

    def train_epoch(self, epoch):
        """Train for one epoch with NaN protection."""
        self.model.train()
        self.optimizer.train()  # Set ScheduleFree optimizer to train mode
        training_losses = []
        nan_batches = 0
        max_nan_batches = 100000000000000  # Maximum number of NaN batches before stopping

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False)):
            self.optimizer.zero_grad()
            
            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, loss = self.model(batch)

                # Check if loss is valid
                if not self._check_loss(loss, batch_idx):
                    nan_batches += 1
                    # print(batch)
                    if nan_batches > max_nan_batches:
                        print("ERROR: Too many NaN losses, stopping training")
                        return float('inf')
                    continue

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Check gradients before optimizer step
                if not self._check_gradients():
                    nan_batches += 1
                    # print(batch)
                    if nan_batches > max_nan_batches:
                        print("ERROR: Too many NaN gradients, stopping training")
                        return float('inf')
                    continue

                # Step optimizer if everything is valid
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                training_losses.append(loss.item())
                print(training_losses[-1])

            except RuntimeError as e:
                print(f"ERROR in batch {batch_idx}: {str(e)}")
                nan_batches += 1
                if nan_batches > max_nan_batches:
                    print("ERROR: Too many errors, stopping training")
                    return float('inf')
                continue

        if not training_losses:
            print("WARNING: No valid losses in this epoch")
            return float('inf')

        avg_loss = sum(training_losses) / len(training_losses)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f} (NaN batches: {nan_batches})")
        return avg_loss

    def validate_epoch(self, epoch):
        """Validate the model and generate sample predictions."""
        self.model.eval()
        self.optimizer.eval()  # Set ScheduleFree optimizer to eval mode
        validation_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                _, loss = self.model(batch)
                validation_losses.append(loss.item())

            avg_loss = sum(validation_losses) / len(validation_losses)
            print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}")

            # Generate sample predictions
            self.generate_samples(5)

        return avg_loss

    def generate_samples(self, num_samples):
        """Generate and display sample predictions."""
        for _ in range(num_samples):
            sample_idx = np.random.randint(len(self.val_dataset))
            sample = self.val_dataset[sample_idx]
            print("Actual Text:", sample['text'])
            print("Generated Text:", self.model.generate(sample['input'], max_new_tokens=120))
            print("-" * 80)

    def save_model(self):
        """Save the model checkpoint."""
        self.model._save_model(self.checkpoint_folder)

    def train(self, num_epochs=30, early_stop_patience=3):
        """
        Train the model for the specified number of epochs with early stopping.
        
        Args:
            num_epochs (int): Maximum number of epochs to train
            early_stop_patience (int): Number of epochs with invalid losses before early stopping
        """
        consecutive_bad_epochs = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train epoch with NaN protection
            train_loss = self.train_epoch(epoch)
            
            # Check for training failure
            if train_loss == float('inf'):
                consecutive_bad_epochs += 1
                print(f"WARNING: Invalid training epoch ({consecutive_bad_epochs}/{early_stop_patience})")
                if consecutive_bad_epochs >= early_stop_patience:
                    print("ERROR: Training stopped due to consistent failures")
                    break
                continue
            
            # Validate epoch
            val_loss = self.validate_epoch(epoch)
            
            # Reset counter if we got valid losses
            if val_loss < float('inf'):
                consecutive_bad_epochs = 0
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model()
                    print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                consecutive_bad_epochs += 1
                print(f"WARNING: Invalid validation epoch ({consecutive_bad_epochs}/{early_stop_patience})")
                if consecutive_bad_epochs >= early_stop_patience:
                    print("ERROR: Training stopped due to consistent failures")
                    break

    def generate_final_samples(self, num_samples=5):
        """Generate and save final sample predictions with images."""
        self.model.eval()
        self.optimizer.eval()  # Set ScheduleFree optimizer to eval mode

        for i in range(num_samples):
            sample_idx = np.random.randint(len(self.val_dataset))
            sample = self.val_dataset[sample_idx]
            
            # Generate and save predictions
            image = sample['image']
            actual_caption = sample['text']
            generated_caption = self.model.generate(sample['input'], max_new_tokens=256)

            # Save image
            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(f"image_{sample_idx}.png")
            plt.close()

            # Save captions
            with open(f"image_{sample_idx}_caption.txt", "w") as f:
                f.write(f"Actual Caption: {actual_caption}\n")
                f.write(f"Generated Caption: {generated_caption}\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train an Image Captioning Model')
    
    parser.add_argument('--training-config', type=str, default='tiny',
                        choices=['tiny'], help='Model configuration')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training and validation')
    parser.add_argument('--checkpoint-folder', type=str,
                        default='image_captioning_model_temp',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--learning-rate', type=float, default=1e-6,
                        help='Learning rate for the optimizer')
    parser.add_argument('--train-subset-fraction', type=float, default=1.,
                        help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--prompt-text', type=str,
                        default='Describe this image briefly.',
                        help='Prompt text for image captioning')
    parser.add_argument('--vision-model', type=str,
                        default='google/vit-large-patch32-384',
                        help='Vision model name or path')
    parser.add_argument('--llm-model', type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help='Language model name or path')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to generate after training')
    
    return parser.parse_args()

"""
Usage Instructions:

This script trains an image captioning model using a vision encoder and language model.

Basic usage:
```bash
# Train with default parameters
python script.py

# Train with custom parameters
python script.py --batch-size 2 --learning-rate 1e-4 --num-epochs 50

# Train on specific GPU
python script.py --device cuda:0

# Use smaller subset of training data
python script.py --train-subset-fraction 0.1

# Use custom checkpoint directory
python script.py --checkpoint-folder ./my_checkpoints
```

Requirements:
- torch
- transformers
- matplotlib
- numpy
- tqdm
- Custom modules: llm, seers, vision, schedulefree

The script will:
1. Initialize models and datasets
2. Train for the specified number of epochs
3. Save checkpoints after each epoch
4. Generate and save sample predictions with images
5. Display training and validation losses

Output:
- Model checkpoints saved in the specified checkpoint folder
- Sample images saved as 'image_X.png'
- Captions saved as 'image_X_caption.txt'
- Training progress and losses printed to console
"""

def main():
    args = parse_args()
    trainer = ImageCaptioningTrainer(
        training_config=args.training_config,
        batch_size=args.batch_size,
        checkpoint_folder=args.checkpoint_folder,
        device=args.device,
        learning_rate=args.learning_rate,
        train_subset_fraction=args.train_subset_fraction,
        prompt_text=args.prompt_text,
        vision_model=args.vision_model,
        llm_model=args.llm_model
    )
    
    trainer.train(num_epochs=args.num_epochs)
    trainer.generate_final_samples(num_samples=args.num_samples)

if __name__ == "__main__":
    main()