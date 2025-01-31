from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
from torch import nn
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
import datasets
from peft import get_peft_config, get_peft_model, LoraConfig
import numpy as np
import random

class ImageDatasetCOT(Dataset):
    """
    ImageDataset: A custom dataset to load images and corresponding text labels from a Hugging Face dataset.

    Attributes:
    - dataset: Hugging Face dataset split.
    - processor: Preprocessing function for images.
    """

    def __init__(self, dataset_name, processor, name = None, split='train'):
        """
        Initializes the dataset by loading a Hugging Face dataset and configuring an image processor.
        
        Parameters:
        - dataset_name: str, name of the Hugging Face dataset.
        - processor: Callable, processes image data into a format suitable for the model.
        - split: str, specifies the dataset split (default: 'train').
        """
        self.dataset = datasets.load_dataset(dataset_name, name)[split]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data item, including the processed image and corresponding text label.

        Parameters:
        - idx: int, index of the item to retrieve.

        Returns:
        - dict: Contains processed image and its text label.
        """
        item = self.dataset[idx]
        image = item['image']

        if not isinstance(image, Image.Image):
            image = Image.open(image.tobytesio())

        # Ensure the image is in RGB format
        image = image.convert('RGB')
        # get the rgb value of the image as np after resizing to 224x224
        rgb_val = np.array(image.resize((224, 224), Image.BICUBIC))
        image = self.processor(image, return_tensors="pt")
        image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension
        out = item['output']
        if '<REASONING>' in out:
            out = '<REASONING>' + out.split('<REASONING>')[1]
            out = out.replace('<REASONING>', '<think>')
            out = out.replace('</REASONING> <CONCLUSION>', '</think>')
            out = out.replace('</REASONING>', '</think>')
            out = out.replace('</CONCLUSION>', '')
            out = out.replace('<CONCLUSION>', '')
        out = out[:2048]
        return {
            'input': image,
            'text': out,
            'prompt': item['question'],
            'image': rgb_val
        }
        
        
class ImageDatasetCauldron(Dataset):
    """
    ImageDataset: A custom dataset to load images and corresponding text labels from a Hugging Face dataset.

    Attributes:
    - dataset: Hugging Face dataset split.
    - processor: Preprocessing function for images.
    """

    def __init__(self, dataset_name, processor, name = None, split='train'):
        """
        Initializes the dataset by loading a Hugging Face dataset and configuring an image processor.
        
        Parameters:
        - dataset_name: str, name of the Hugging Face dataset.
        - processor: Callable, processes image data into a format suitable for the model.
        - split: str, specifies the dataset split (default: 'train').
        """
        self.dataset = datasets.load_dataset(dataset_name, name)[split]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data item, including the processed image and corresponding text label.

        Parameters:
        - idx: int, index of the item to retrieve.

        Returns:
        - dict: Contains processed image and its text label.
        """
        item = self.dataset[idx]
        image = item['images'][0]

        if not isinstance(image, Image.Image):
            image = Image.open(image.tobytesio())

        # Ensure the image is in RGB format
        image = image.convert('RGB')
        # get the rgb value of the image as np after resizing to 224x224
        rgb_val = np.array(image.resize((224, 224), Image.BICUBIC))
        image = self.processor(image, return_tensors="pt")
        image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension
        qa_item = random.choice(item['texts'])
        prompt = qa_item['user']
        answer = qa_item['assistant']
        return {
            'input': image,
            'text': answer,
            'prompt': prompt,
            'image': rgb_val
        }

class ImageDataset(Dataset):
    """
    ImageDataset: A custom dataset to load images and corresponding text labels from a Hugging Face dataset.

    Attributes:
    - dataset: Hugging Face dataset split.
    - processor: Preprocessing function for images.
    """

    def __init__(self, dataset_name, processor, name = None, split='train'):
        """
        Initializes the dataset by loading a Hugging Face dataset and configuring an image processor.
        
        Parameters:
        - dataset_name: str, name of the Hugging Face dataset.
        - processor: Callable, processes image data into a format suitable for the model.
        - split: str, specifies the dataset split (default: 'train').
        """
        self.dataset = datasets.load_dataset(dataset_name, name)[split]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data item, including the processed image and corresponding text label.

        Parameters:
        - idx: int, index of the item to retrieve.

        Returns:
        - dict: Contains processed image and its text label.
        """
        item = self.dataset[idx]
        image = item['image']

        if not isinstance(image, Image.Image):
            image = Image.open(image.tobytesio())

        # Ensure the image is in RGB format
        image = image.convert('RGB')
        # get the rgb value of the image as np after resizing to 224x224
        rgb_val = np.array(image.resize((224, 224), Image.BICUBIC))
        image = self.processor(image, return_tensors="pt")
        image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

        return {
            'input': image,
            'text': item['chosen'],
            'prompt': item['question'],
            'image': rgb_val
        }

class ImageDataset2(Dataset):
    """
    ImageDataset: A custom dataset to load images and corresponding text labels from a Hugging Face dataset.

    Attributes:
    - dataset: Hugging Face dataset split.
    - processor: Preprocessing function for images.
    """

    def __init__(self, dataset_name, processor, name = None, split='train'):
        """
        Initializes the dataset by loading a Hugging Face dataset and configuring an image processor.
        
        Parameters:
        - dataset_name: str, name of the Hugging Face dataset.
        - processor: Callable, processes image data into a format suitable for the model.
        - split: str, specifies the dataset split (default: 'train').
        """
        self.dataset = datasets.load_dataset(dataset_name, name)[split]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data item, including the processed image and corresponding text label.

        Parameters:
        - idx: int, index of the item to retrieve.

        Returns:
        - dict: Contains processed image and its text label.
        """
        item = self.dataset[idx]
        image = item['image']

        if not isinstance(image, Image.Image):
            image = Image.open(image.tobytesio())

        # Ensure the image is in RGB format
        image = image.convert('RGB')
        # get the rgb value of the image as np after resizing to 224x224
        rgb_val = np.array(image.resize((224, 224), Image.BICUBIC))
        image = self.processor(image, return_tensors="pt")
        image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

        return {
            'input': image,
            'text': random.choice(item['original_alt_text']),
            'prompt': "Describe this image briefly.",
            'image': rgb_val
        }

import numpy as np

import numpy as np

class CombinedImageDatasets(Dataset):
    """
    A dataset class that combines multiple image datasets with shuffled indices for better sampling.
    """
    def __init__(self, datasets, seed=None):
        if not datasets:
            raise ValueError("At least one dataset must be provided")
            
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        
        # Calculate cumulative sizes
        self.cumulative_sizes = []
        current_size = 0
        for size in self.dataset_sizes:
            current_size += size
            self.cumulative_sizes.append(current_size)
            
        # Create and shuffle global indices
        if seed is not None:
            np.random.seed(seed)
            
        self.shuffled_indices = np.arange(self.cumulative_sizes[-1], dtype=np.int32)
        np.random.shuffle(self.shuffled_indices)
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def reshuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self.shuffled_indices)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
            
        # Get the actual index from shuffled indices and convert to Python int
        shuffled_idx = int(self.shuffled_indices[idx])  # Convert numpy.int64 to Python int
        
        # Binary search to find the correct dataset
        dataset_idx = self._find_dataset_index(shuffled_idx)
        
        # Calculate the local index within the selected dataset
        local_idx = shuffled_idx
        if dataset_idx > 0:
            local_idx = shuffled_idx - self.cumulative_sizes[dataset_idx - 1]
            
        return self.datasets[dataset_idx][int(local_idx)]  # Ensure local_idx is Python int
    
    def _find_dataset_index(self, idx):
        left, right = 0, len(self.cumulative_sizes)
        
        while left < right:
            mid = (left + right) // 2
            if mid == 0:
                if idx < self.cumulative_sizes[0]:
                    return 0
                left = mid + 1
            elif idx < self.cumulative_sizes[mid] and idx >= self.cumulative_sizes[mid-1]:
                return mid
            elif idx < self.cumulative_sizes[mid]:
                right = mid
            else:
                left = mid + 1
                
        return left

    def get_original_index(self, idx):
        shuffled_idx = int(self.shuffled_indices[idx])  # Convert to Python int
        dataset_idx = self._find_dataset_index(shuffled_idx)
        local_idx = shuffled_idx
        if dataset_idx > 0:
            local_idx = shuffled_idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, int(local_idx)  # Ensure local_idx is Python int

# Usage example:
# datasets = [
#     ImageDataset("dataset1_name", processor),
#     ImageDataset2("dataset2_name", processor),
#     ImageDataset("dataset3_name", processor)
# ]
# combined_dataset = CombinedImageDatasets(datasets, seed=42)  # Set seed for reproducibility
# 
# # Reshuffle if needed during training
# combined_dataset.reshuffle(seed=43)

class CombinedImageDataset(Dataset):
    """
    A dataset class that combines two different image datasets into a single dataset.
    Handles datasets with different column structures.
    
    Attributes:
        dataset1: First dataset instance
        dataset2: Second dataset instance
        dataset1_size: Length of first dataset
        dataset2_size: Length of second dataset
    """
    def __init__(self, dataset1, dataset2):
        """
        Initialize the combined dataset with two dataset instances.
        
        Parameters:
            dataset1: First dataset instance (ImageDataset)
            dataset2: Second dataset instance (ImageDataset2)
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset1_size = len(dataset1)
        self.dataset2_size = len(dataset2)
    
    def __len__(self):
        """Returns the total length of the combined dataset"""
        return self.dataset1_size + self.dataset2_size
    
    def __getitem__(self, idx):
        """
        Get an item from either dataset based on the index.
        Handles different column structures between datasets.
        
        Parameters:
            idx: Index of the item to retrieve
            
        Returns:
            dict: Standardized dictionary containing the item data
        """
        if idx < self.dataset1_size:
            # Get item from first dataset
            return self.dataset1[idx]
        else:
            # Get item from second dataset
            return self.dataset2[idx - self.dataset1_size]
            
class Projector(nn.Module):
    """
    Projector: A feedforward neural network for projecting feature embeddings to a target dimension.

    Attributes:
    - inp_layer: Input linear layer.
    - layers: Sequence of hidden layers.
    - dropout: Dropout applied between layers.
    - out_layer: Output linear layer.
    """

    def __init__(self, in_features, out_features, num_hidden=2):
        """
        Initializes the Projector.

        Parameters:
        - in_features: int, size of the input feature vector.
        - out_features: int, size of the output feature vector.
        - num_hidden: int, number of hidden layers (default: 2).
        """
        super(Projector, self).__init__()
        self.inp_layer = nn.Linear(in_features, out_features)
        self.layers = nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(num_hidden)])
        self.dropout = nn.Dropout(0.1)
        self.out_layer = nn.Linear(out_features, out_features)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: torch.Tensor, input tensor.

        Returns:
        - torch.Tensor, output tensor.
        """
        x = self.inp_layer(x)
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        return x

class VisionEncoder(nn.Module):
    """
    VisionEncoder: Wraps a vision model to extract hidden states as feature embeddings.

    Attributes:
    - model: Pre-trained vision model.
    - device: Torch device (GPU/CPU).
    """

    def __init__(self, model, device='cuda:0'):
        """
        Initializes the VisionEncoder.

        Parameters:
        - model: nn.Module, pre-trained vision model.
        """
        super(VisionEncoder, self).__init__()
        self.model = model
        if device is None:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
    def forward(self, inputs):
        """
        Forward pass to obtain feature embeddings.

        Parameters:
        - inputs: dict, preprocessed inputs compatible with the vision model.

        Returns:
        - torch.Tensor, last hidden state of the vision model.
        """
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]  # Extract last hidden state
        # return outputs.last_hidden_state

def get_image_encoder(model_name, device='cuda:0', use_peft=False):
    """
    Loads a vision model and its processor, optionally applying Parameter-Efficient Fine-Tuning (PEFT).
    
    Parameters:
    - model_name: str, name of the pre-trained vision model.
    - device: str or torch.device, device to place the model on (default: 'cuda:0')
    - use_peft: bool, whether to apply PEFT (default: False).
    
    Returns:
    - processor: Image processor for pre-processing.
    - model: Pre-trained vision model placed on the specified device.
    - hidden_size: int, size of the model's hidden layer.
    """
    # Convert device string to torch.device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load processor
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Load model and move to specified device
    model = ViTForImageClassification.from_pretrained(
        model_name,
        device_map=device  # Use device_map for optimal placement
    ).to(device)  # Ensure model is on the correct device
    
    hidden_size = model.config.hidden_size
    
    if use_peft:
        peft_config = LoraConfig(
            task_type=None, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules=['dense']
        )
        model = get_peft_model(model, peft_config)
        model = model.to(device)  # Ensure PEFT model is on correct device
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad = False
    
    return processor, model, hidden_size

if __name__ == '__main__':
    dataset_name = "Mozilla/flickr30k-transformed-captions"
    processor, model, hidden_size = get_image_encoder('google/vit-base-patch16-224')

    dataset = ImageDataset(dataset_name, processor)

    # Split dataset
    split_ratio = 0.8
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader setup
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize encoder and projector
    vision_encoder = VisionEncoder(model)
    vision_projector = Projector(hidden_size, 768)

    for batch in train_loader:
        vision_embeddings = vision_encoder(batch['input'])
        print(vision_embeddings.shape)
        vision_tokens = vision_projector(vision_embeddings)
        print(vision_tokens.shape)
        break
