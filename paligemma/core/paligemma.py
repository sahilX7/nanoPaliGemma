# Imports
import torch
from torch import nn
from paligemma.models.siglip import SiglipVisionConfig, SiglipVisionModel
from paligemma.models.gemma import GemmaConfig, KVCache, GemmaForCausalLM

class PaliGemmaConfig():
    def __init__(
        self,
        hidden_size,
        image_token_index,
        projection_dim,
        vision_config,
        text_config,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_token_index = image_token_index
        self.vision_config = SiglipVisionConfig(**vision_config)        
        self.vision_config.projection_dim = projection_dim
        self.text_config = GemmaConfig(**text_config)

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_embeds):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_embeds)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)

    def tie_weights(self):
        return self.language_model.tie_weights()
     
    def _merge_input_ids_with_image_embeddings(
        self, 
        image_embeds: torch.Tensor, 
        input_embeds: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        kv_cache: KVCache
    ):
        _, _, embed_dim = image_embeds.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_embeds = image_embeds / (self.config.hidden_size**0.5)
     
        # Combine the image and text embeddings
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index

        # Expand masks to embedding dimension or else it can't be used in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Insert text embeddings
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # Insert image embeddings. 
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_embeds) # We can't use torch.where because the sequence length of scaled_image_embeds is not equal to the sequence length of the final embedding

        # Create Attention mask
        q_len = input_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: Don't mask anything
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # When generating tokens, the query must be a single token
            assert q_len == 1

            kv_len = kv_cache.num_items() + q_len
            # In this case also, we don't need to mask anything, since query should be able to attend all previous tokens. 
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: KVCache
    ):
        # 1. Get input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids) # Converts input_ids to token embeddings using Gemma’s embedding matrix.

        # 2. Get image embeddings
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        image_embeds = self.vision_tower(pixel_values.to(input_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_embeds = self.multi_modal_projector(image_embeds)

        # 3. Merge image with input embeddings
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_embeddings(image_embeds, input_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs