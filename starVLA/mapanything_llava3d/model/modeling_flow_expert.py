import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GemmaModel, GemmaPreTrainedModel, GemmaConfig
from typing import Optional, Tuple, List

def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float32 
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    
    emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return emb

class FlowMatchingActionExpert(GemmaPreTrainedModel):
    """
    A Flow Matching Action Expert based on the Gemma architecture.
    Designed for 'Late Fusion' integration with LLaVA-3D.
    
    It accepts:
    1. Context Features (from VLM)
    2. Noisy Actions (Action trajectory at step t)
    3. Time Step (t)
    
    And predicts:
    - The velocity vector v_t to denoise the action.
    """
    config_class = GemmaConfig

    def __init__(self, config: GemmaConfig, action_dim: int = 32, action_horizon: int = 1, vlm_hidden_size: int = 4096):
        super().__init__(config)
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # 1. The Core Brain: Standard Gemma Model
        self.model = GemmaModel(config)
        
        # 2. Input Projections
        # A. Context Projector: Map VLM hidden dim (e.g. 4096) to Expert hidden dim (e.g. 2048)
        self.context_projector = nn.Linear(vlm_hidden_size, config.hidden_size)

        # B. Action Projection: Map raw action (dims) to Expert hidden dim
        self.action_in_proj = nn.Linear(action_dim, config.hidden_size)
        
        # C. Time Embedding & MLP
        self.time_mlp_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.time_mlp_out = nn.Linear(config.hidden_size, config.hidden_size)

        # 3. Output Projection
        # Map back from Hidden Dim -> Action Dim
        self.action_out_proj = nn.Linear(config.hidden_size, action_dim)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        context_features: torch.Tensor,
        actions: torch.Tensor,
        time: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            context_features: (B, Seq_Len, VLM_Hidden_Dim) - The 'Memory' from the VLM.
            actions: (B, Horizon, Action_Dim) - The noisy actions.
            time: (B, ) - The time steps [0, 1].
            context_mask: (B, Seq_Len) - Attention mask for context (1 for valid, 0 for pad).
        
        Returns:
            predicted_velocity: (B, Horizon, Action_Dim)
        """
        batch_size, seq_len, _ = context_features.shape
        device = context_features.device
        dtype = context_features.dtype

        # 1. Embed Inputs
        
        # A. Context Projection
        # (B, S, VLM_Dim) -> (B, S, Expert_Dim)
        context_embeds = self.context_projector(context_features)

        # B. Time
        # Create sinusoidal embedding
        time_embed = create_sinusoidal_pos_embedding(
            time, self.config.hidden_size, min_period=4e-3, max_period=4.0, device=device
        )
        time_embed = time_embed.to(dtype=dtype)
        
        # MLP Process Time
        time_embed = F.silu(self.time_mlp_in(time_embed))
        time_embed = self.time_mlp_out(time_embed) # (B, H)
        time_embed = time_embed.unsqueeze(1) # (B, 1, H)

        # C. Actions
        action_embeds = self.action_in_proj(actions) # (B, Horizon, H)
        
        # Concatenate: [Context, Time, Actions]
        inputs_embeds = torch.cat([context_embeds, time_embed, action_embeds], dim=1)

        # 2. Create Attention Masks
        # Handle Context Mask (padding from VLM)
        if context_mask is None:
            context_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        
        # Create mask for [Time (1) + Actions (Horizon)] -> All 1s
        # We assume full bidirectional attention for the action sequence generation
        expert_mask = torch.ones((batch_size, 1 + self.action_horizon), device=device, dtype=torch.long)
        
        attention_mask = torch.cat([context_mask, expert_mask], dim=1) # (B, Total_Len)

        # 3. Forward Pass through Gemma
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False, 
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # 4. Extract Output (Last 'Action_Horizon' tokens)
        action_output = last_hidden_state[:, -self.action_horizon :, :]
        
        # Project back to Action Space
        predicted_velocity = self.action_out_proj(action_output)
        
        return predicted_velocity

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def compute_loss(self, context_features, actions, context_mask=None):
        """
        Computes the Flow Matching MSE Loss.
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        # 1. Sample Time t ~ Uniform[0, 1]
        t = torch.rand((batch_size,), device=device)
        
        # 2. Sample Noise x_1
        noise = self.sample_noise(actions.shape, device)
        
        # 3. Compute Noisy Sample x_t
        # Flow Matching: x_t = (1 - t) * x_0 (Action) + t * x_1 (Noise) ?
        # Pi0 uses: x_t = time * noise + (1 - time) * actions
        # which means t=1 is Noise, t=0 is Action.
        # Target velocity u_t = noise - actions
        # d/dt (x_t) = noise - actions
        
        t_exp = t.view(batch_size, 1, 1)
        x_t = t_exp * noise + (1 - t_exp) * actions
        target_velocity = noise - actions
        
        # 4. Predict Velocity
        pred_velocity = self.forward(context_features, x_t, t, context_mask)
        
        # 5. MSE Loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample_actions(self, context_features, num_steps=10, context_mask=None):
        """
        Inference: Generate actions from noise using Euler integration.
        Goes from t=1 (Noise) to t=0 (Action).
        """
        batch_size = context_features.shape[0]
        device = context_features.device
        action_shape = (batch_size, self.action_horizon, self.action_dim)
        
        # Start from Noise (t=1)
        x_t = self.sample_noise(action_shape, device)
        
        # Time step size (negative because we go 1 -> 0)
        dt = -1.0 / num_steps
        
        # Euler Loop
        for step in range(num_steps):
            # t goes from 1.0 down to dt
            t_curr = 1.0 + step * dt 
            
            # Prepare t tensor
            t_tensor = torch.full((batch_size,), t_curr, device=device)
            
            # Predict velocity at current point
            v_t = self.forward(context_features, x_t, t_tensor, context_mask)
            
            # Update x_t
            x_t = x_t + v_t * dt
            
        return x_t