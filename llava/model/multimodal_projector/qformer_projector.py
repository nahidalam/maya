import torch
import torch.nn as nn
from transformers import Blip2QFormerModel, Blip2QFormerConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerMultiHeadAttention
import math


class QFormerProjector(nn.Module):
    """
    Q-Former based projector for multimodal feature projection.
    Supports BLIP-2 and InstructBLIP architectures.
    """
    
    def __init__(self, config, qformer_type='blip2', num_query_tokens=64, pretrained_qformer_path=None):
        super().__init__()
        
        self.config = config
        self.qformer_type = qformer_type
        self.num_query_tokens = num_query_tokens
        self.vision_hidden_size = config.mm_hidden_size
        self.text_hidden_size = config.hidden_size
        
        # Create Q-Former configuration
        if qformer_type in ['blip2','instructblip']:
            self.qformer_config = Blip2QFormerConfig(
                vocab_size=30522,  # BERT vocab size
                hidden_size=768,   # Standard BERT hidden size
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                position_embedding_type="absolute",
                cross_attention_frequency=2,
                encoder_hidden_size=self.vision_hidden_size,
            )
        else:
            raise ValueError(f"Unsupported Q-Former type: {qformer_type}")
        
        # Initialize Q-Former model
        self.qformer = Blip2QFormerModel(self.qformer_config)
        
        # Create learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=self.qformer_config.initializer_range)
        
        # Final projection layer to match target hidden size
        self.projection = nn.Linear(self.qformer_config.hidden_size, self.text_hidden_size)
        
        # Load pretrained weights if specified
        if pretrained_qformer_path:
            self.load_pretrained_qformer(pretrained_qformer_path)
    
    def load_pretrained_qformer(self, pretrained_path):
        """Load pretrained Q-Former weights."""
        try:
            if pretrained_path.startswith('Salesforce/'):
                # Load from HuggingFace hub
                from transformers import Blip2ForConditionalGeneration
                pretrained_model = Blip2ForConditionalGeneration.from_pretrained(pretrained_path)
                qformer_state_dict = pretrained_model.qformer.state_dict()
                
                # Load Q-Former weights
                self.qformer.load_state_dict(qformer_state_dict, strict=False)
                
                # Load query tokens if available
                if hasattr(pretrained_model, 'query_tokens'):
                    query_tokens = pretrained_model.query_tokens
                    if query_tokens.size(1) == self.num_query_tokens:
                        self.query_tokens.data = query_tokens.data
                    else:
                        print(f"Warning: Pretrained query tokens size ({query_tokens.size(1)}) "
                              f"doesn't match configured size ({self.num_query_tokens}). "
                              f"Using random initialization.")
                
                print(f"Loaded pretrained Q-Former from {pretrained_path}")
            else:
                # Load from local path
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained Q-Former from {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained Q-Former from {pretrained_path}: {e}")
            print("Using random initialization instead.")
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: Tensor of shape (batch_size, num_patches, vision_hidden_size)
        
        Returns:
            projected_features: Tensor of shape (batch_size, num_query_tokens, text_hidden_size)
        """
        batch_size = vision_features.size(0)
        
        # Expand query tokens for the batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Apply Q-Former
        # The Q-Former expects:
        # - query_embeds: (batch_size, num_query_tokens, hidden_size)
        # - encoder_hidden_states: (batch_size, num_patches, encoder_hidden_size)
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=vision_features,
            return_dict=True,
        )
        
        # Get the query outputs
        query_outputs = qformer_outputs.last_hidden_state
        
        # Project to target hidden size
        projected_features = self.projection(query_outputs)
        
        return projected_features
    
    @property
    def config_dict(self):
        """Return configuration dictionary for saving/loading."""
        return {
            "mm_projector_type": f"qformer_{self.qformer_type}",
            "num_query_tokens": self.num_query_tokens,
            "qformer_type": self.qformer_type,
        }


class BlipQFormerProjector(QFormerProjector):
    """BLIP-2 Q-Former projector."""
    
    def __init__(self, config, num_query_tokens=64, pretrained_qformer_path=None):
        super().__init__(config, 'blip2', num_query_tokens, pretrained_qformer_path)


class InstructBlipQFormerProjector(QFormerProjector):
    """InstructBLIP Q-Former projector."""
    
    def __init__(self, config, num_query_tokens=64, pretrained_qformer_path=None):
        super().__init__(config, 'instructblip', num_query_tokens, pretrained_qformer_path)


def build_qformer_projector(config, qformer_type='blip2', num_query_tokens=64, pretrained_qformer_path=None):
    """Build Q-Former projector based on configuration."""
    if qformer_type == 'blip2':
        return BlipQFormerProjector(config, num_query_tokens, pretrained_qformer_path)
    elif qformer_type == 'instructblip':
        return InstructBlipQFormerProjector(config, num_query_tokens, pretrained_qformer_path)
    else:
        raise ValueError(f"Unsupported Q-Former type: {qformer_type}") 