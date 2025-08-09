import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    # Q-Former projector types
    if projector_type in ['qformer', 'qformer_blip2', 'qformer_instructblip']:
        from .qformer_projector import build_qformer_projector
        
        # Parse Q-Former type
        if projector_type in ['qformer','qformer_blip2']:
            qformer_type = 'blip2'  # default
        elif projector_type == 'qformer_instructblip':
            qformer_type = 'instructblip'
        else:
            raise ValueError(f'Unknown Q-Former type: {projector_type}')
        
        # Get Q-Former configuration parameters
        num_query_tokens = getattr(config, 'mm_qformer_num_query_tokens', 64)
        pretrained_qformer_path = getattr(config, 'mm_qformer_pretrained_path', None)
        
        return build_qformer_projector(
            config=config,
            qformer_type=qformer_type,
            num_query_tokens=num_query_tokens,
            pretrained_qformer_path=pretrained_qformer_path
        )

    raise ValueError(f'Unknown projector type: {projector_type}')
