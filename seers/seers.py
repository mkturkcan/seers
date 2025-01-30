import torch
import torch.nn as nn
import os

class MultiModalModel(nn.Module):
    """
    MultiModalModel: A flexible architecture for encoding non-text inputs and generating text outputs.
    
    This model supports a variety of input modalities through a customizable input processor and encoder.
    It integrates these modalities with a language model for text generation tasks.
    """

    def __init__(self,
                 input_processor,
                 input_encoder,
                 input_tokenizer,
                 language_tokenizer,
                 language_model,
                 input_start_token=None,
                 input_end_token=None,
                 lm_peft=None,
                 prompt_text="This input contains: ",
                 device='cuda:0'):
        """
        Initializes the MultiModalModel.
        
        Parameters:
        - input_processor: Callable, processes raw input data.
        - input_encoder: nn.Module, encodes processed input into a latent representation.
        - input_tokenizer: nn.Module, maps encoded input to token embeddings.
        - language_tokenizer: Tokenizer, converts text to tokens for the language model.
        - language_model: nn.Module, pre-trained language model for text generation.
        - input_start_token: str, special token marking the start of input.
        - input_end_token: str, special token marking the end of input.
        - lm_peft: a function that applies PEFT to the language model.
        - prompt_text: str, text prompt used as a context for generation.
        - device: str or torch.device, device to place the model on (default: 'cuda:0')
        """
        super().__init__()
        
        # Set up device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move the main module to specified device
        
        # Model components - move each to specified device
        self.input_processor = input_processor
        self.input_encoder = input_encoder.to(self.device)
        self.input_tokenizer = input_tokenizer.to(self.device)
        self.language_tokenizer = language_tokenizer
        self.language_model = language_model.to(self.device)

        # Special tokens and prompt
        self.input_start_token = input_start_token
        self.input_end_token = input_end_token
        self.prompt_text = prompt_text

        # Add special tokens to tokenizer and update embeddings
        self._add_special_tokens(lm_peft)

        # Loss function
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        # Precompute embeddings for special tokens and move to device
        self.start_embedding = self._embed_special_token(self.input_start_token)
        self.end_embedding = self._embed_special_token(self.input_end_token)
        self.prompt_embedding = self._embed_special_token(self.prompt_text)
        self.eos_embedding = self._embed_special_token(self.language_tokenizer.eos_token)

        # Embed the language prompt and move to device
        prompt_tokens = self.language_tokenizer.encode(self.prompt_text, return_tensors="pt").to(self.device)
        self.prompt_embedding = self._embed_tokens(prompt_tokens).squeeze(0)
        self.prompt_length = self.prompt_embedding.size(0)

    def _add_special_tokens(self, lm_peft):
        """
        Adds custom tokens to the tokenizer and resizes the language model's embeddings.
        """
        num_tokens_new = 0
        if self.input_start_token is not None:
            self.language_tokenizer.add_tokens([self.input_start_token], special_tokens=True)
            num_tokens_new += 1
        else:
            self.input_start_token = 'Image:'
        if self.input_end_token is not None:
            self.language_tokenizer.add_tokens([self.input_end_token], special_tokens=True)
            num_tokens_new += 1
        else:
            self.input_end_token = '.'
        if num_tokens_new > 0:
            self.language_model.resize_token_embeddings(len(self.language_tokenizer))

        if lm_peft is not None:
            self.language_model = lm_peft(self.language_model)
            self.language_model = self.language_model.to(self.device)

    def forward(self, batch):
        """
        Performs a forward pass with a batch of input and text data.
        
        Parameters:
        - batch: dict, contains 'input' and 'text'.
        
        Returns:
        - logits: torch.Tensor, model predictions.
        - loss: torch.Tensor, computed loss.
        """
        # Move batch data to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
                
        tokenized_input = self._encode_input(batch['input'])
        text_samples = batch['text']
        prompt_samples = batch['prompt']
        batch_size = len(text_samples)

        for i in range(batch_size):
            text_samples[i] = text_samples[i] + self.language_tokenizer.decode(self.language_model.config.eos_token_id)

        input_embeddings, target_labels, attention_masks = [], [], []
        max_sequence_length = 0

        for i in range(batch_size):
            tokenized_text = self.language_tokenizer(text_samples[i], return_tensors="pt")['input_ids'].to(self.device)
            tokenized_prompt = self._embed_tokens(self.language_tokenizer.encode(prompt_samples[i], return_tensors="pt").to(self.device)).squeeze(0)
            embedded_text = self._embed_tokens(tokenized_text)
            
            # Combine embeddings
            combined_input = torch.cat([
                self.start_embedding.squeeze(0),
                tokenized_input[i],
                self.end_embedding.squeeze(0),
                tokenized_prompt,
                embedded_text.squeeze(0)
            ], dim=0)

            label_sequence = torch.cat([
                torch.full((combined_input.shape[0] - tokenized_text.size(1),), -100, device=self.device),
                tokenized_text.squeeze(0)
            ], dim=0)

            attention_mask = torch.ones(combined_input.shape[0], device=self.device)
            
            input_embeddings.append(combined_input)
            target_labels.append(label_sequence)
            attention_masks.append(attention_mask)
            max_sequence_length = max(max_sequence_length, combined_input.shape[0])

        # Pad sequences to max length
        for i in range(batch_size):
            pad_length = max_sequence_length - input_embeddings[i].size(0)
            pad_token = torch.full((pad_length,), self.language_model.config.eos_token_id, dtype=torch.long, device=self.device)
            pad_embedding = self._embed_tokens(pad_token)
            input_embeddings[i] = torch.cat([input_embeddings[i], pad_embedding], dim=0)
            target_labels[i] = torch.cat([target_labels[i], torch.full((pad_length,), -100, dtype=torch.long, device=self.device)], dim=0)
            attention_masks[i] = torch.cat([attention_masks[i], torch.zeros(pad_length, device=self.device)], dim=0)

        # Stack tensors and ensure they're on the correct device
        input_embeddings = torch.stack(input_embeddings)
        target_labels = torch.stack(target_labels)
        attention_masks = torch.stack(attention_masks)

        outputs = self.language_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_masks,
            labels=target_labels
        )

        return outputs.logits, outputs.loss

    @torch.no_grad()
    def generate(self, input_data, max_new_tokens=100, prompt_text=None, **kwargs):
        """
        Generates text given input data.
        
        Parameters:
        - input_data: dict, raw input data.
        - max_new_tokens: int, maximum tokens to generate.
        
        Returns:
        - str, generated text.
        """
        # Move input data to device
        for key in input_data.keys():
            if torch.is_tensor(input_data[key]):
                input_data[key] = input_data[key].unsqueeze(0).to(self.device)
            
        if prompt_text is not None:
            self.prompt_text = prompt_text
            prompt_tokens = self.language_tokenizer.encode(self.prompt_text, return_tensors="pt").to(self.device)
            self.prompt_embedding = self._embed_tokens(prompt_tokens).squeeze(0)
            self.prompt_length = self.prompt_embedding.size(0)
        
        tokenized_input = self._encode_input(input_data)
        input_embeddings = torch.cat([
            self.start_embedding,
            tokenized_input,
            self.end_embedding,
            self.prompt_embedding.unsqueeze(0)
        ], dim=1)

        output_ids = self.language_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=torch.ones(input_embeddings.shape[:2], device=self.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.language_model.config.eos_token_id,
            **kwargs
        )

        return self.language_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _encode_input(self, modality_input):
        """
        Encodes the input modality using the processor, encoder, and tokenizer.
        """
        processed_input = self.input_processor(modality_input) if self.input_processor else modality_input
        encoded_input = self.input_encoder(processed_input)
        return self.input_tokenizer(encoded_input)

    def _embed_tokens(self, token_ids):
        """
        Embeds tokenized integers using the language model's embeddings.
        """
        return self.language_model.get_input_embeddings()(token_ids)

    def _embed_special_token(self, token):
        """
        Embeds a special token and returns its vector.
        """
        token_ids = self.language_tokenizer(token)['input_ids']
        token_ids = torch.tensor(token_ids, device=self.device)
        return self._embed_tokens(token_ids).unsqueeze(0)

    def _save_model(self, output_dir):
        """
        Saves the model to disk.
        """
        torch.save(self.input_tokenizer, f'{output_dir}/input_tokenizer.pt')
        self.input_encoder.model.save_pretrained(f'{output_dir}/input_encoder') 
        self.language_model.save_pretrained(f'{output_dir}/language_model')

    def _load_model(self, model_dir, device):
        """
        Loads the model from disk.
        """
        self.input_tokenizer = torch.load(f'{model_dir}/input_tokenizer.pt', map_location=device)
        self.language_model.load_adapter(f'{model_dir}/language_model')
        print(f'{model_dir}/input_encoder/')
        self.input_encoder.model.load_adapter(f'{model_dir}/input_encoder/')