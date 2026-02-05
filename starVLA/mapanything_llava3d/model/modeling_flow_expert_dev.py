"""
Flow Matching Action Expert - Dev Version with LLaVA3D Deep Fusion

This version integrates with LLaVA3DWithActionExpertModel for deep fusion.
Instead of using Gemma or simple MLPs, it leverages the LLaVA3D transformer
for both prefix (vision+language) and suffix (state+action+time) processing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device: str = "cpu"
) -> torch.Tensor:
    """
    Create sinusoidal position embeddings for time encoding.
    
    Args:
        time: [B] tensor of time values (typically in [0, 1])
        dimension: embedding dimension (must be even)
        min_period: minimum period for sinusoidal encoding
        max_period: maximum period for sinusoidal encoding
        device: device to create tensors on
        
    Returns:
        [B, dimension] time embeddings
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time is expected to be of shape `(batch_size,)`")
    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return emb


class FlowMatchingActionExpert(nn.Module):
    """
    Flow Matching Action Expert with LLaVA3D Deep Fusion.
    
    This expert implements the Flow Matching algorithm for continuous action diffusion,
    but delegates the network forward to LLaVA3DWithActionExpertModel for deep fusion
    between prefix (vision+language) and suffix (state+action+time).
    
    Key Design:
    - Prefix: Image + Geometric + Text features (from wrapper)
    - Suffix: State + Noisy Actions + Time embeddings (constructed here)
    - Network: LLaVA3DWithActionExpertModel (deep fusion at every layer)
    - Algorithm: Flow Matching (t, noise, x_t, u_t, Euler ODE)
    """
    
    def __init__(
        self, 
        llava_with_expert_model,  # LLaVA3DWithActionExpertModel instance
        action_dim: int = 7,
        action_horizon: int = 10,
        state_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        use_state: bool = False,
        use_time_weight: bool = True,
    ):
        """
        Args:
            llava_with_expert_model: Instance of LLaVA3DWithActionExpertModel
            action_dim: Dimension of action space (e.g., 7 for robot arm)
            action_horizon: Number of action steps to predict
            state_dim: Dimension of proprioceptive state (optional)
            hidden_size: Hidden size for embeddings (inferred from model if None)
            use_state: Whether to include proprioceptive state in suffix
        """
        super().__init__()
        
        # Store model reference
        self.llava_with_expert = llava_with_expert_model
        
        # Action config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_state = use_state
        self.state_dim = state_dim if use_state else 0
        self.use_time_weight = use_time_weight
        
        # Infer hidden size from the model (align with expert width)
        if hidden_size is None:
            self.hidden_size = getattr(
                self.llava_with_expert,
                "expert_hidden_size",
                self.llava_with_expert.hidden_size,
            )
        else:
            self.hidden_size = hidden_size
        if hasattr(self.llava_with_expert, "expert_hidden_size"):
            assert (
                self.hidden_size == self.llava_with_expert.expert_hidden_size
            ), f"Flow expert hidden_size {self.hidden_size} must match expert_hidden_size {self.llava_with_expert.expert_hidden_size}"
        
        # Suffix embedding layers
        # State embedding (optional)
        if self.use_state and self.state_dim > 0:
            self.state_proj = nn.Linear(self.state_dim, self.hidden_size)
        else:
            self.state_proj = None
        
        # Action embedding: project each action step to hidden_size
        self.action_in_proj = nn.Linear(self.action_dim, self.hidden_size)
        
        # Time embedding: sinusoidal -> MLP
        self.time_mlp_in = nn.Linear(self.hidden_size, self.hidden_size)
        self.time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Action output projection: from suffix hidden to action velocity
        self.action_out_proj = nn.Linear(self.hidden_size, self.action_dim)
        
    def sample_noise(self, shape, device):
        """Sample Gaussian noise for flow matching."""
        return torch.randn(shape, device=device)

    def sample_time(self, batch_size: int, device, alpha: float = 1.5, beta: float = 1.0, eps: float = 1e-3):
        """Sample time t ~ Beta(alpha, beta) and keep it in (eps, 1-eps) for stability."""
        dist = torch.distributions.Beta(
            torch.as_tensor(alpha, device=device), torch.as_tensor(beta, device=device)
        )
        t = dist.sample((batch_size,))
        t = t * (1.0 - 2.0 * eps) + eps
        return t.to(dtype=torch.float32)
    
    def _construct_suffix_embeddings(
        self,
        actions: torch.Tensor,  # [B, H, action_dim] - noisy or clean actions
        time: torch.Tensor,     # [B] - time in [0, 1]
        state: Optional[torch.Tensor] = None,  # [B, state_dim] - proprioceptive state
    ) -> torch.Tensor:
        """
        Construct suffix embeddings from state, actions, and time.
        
        Suffix sequence structure:
        - If use_state: [state_token, action_token_1, ..., action_token_H, time_token]
        - Else: [action_token_1, ..., action_token_H, time_token]
        
        Args:
            actions: [B, H, action_dim] noisy or clean actions
            time: [B] time values in [0, 1]
            state: [B, state_dim] proprioceptive state (optional)
            
        Returns:
            suffix_embs: [B, suffix_seq_len, hidden_size]
        """
        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype
        
        suffix_tokens = []
        
        # 1. State token (optional)
        if self.use_state and self.state_proj is not None and state is not None:
            state_token = self.state_proj(state).unsqueeze(1)  # [B, 1, H]
            if not torch.isfinite(state_token).all():
                print(
                    "flow_debug state_token nan:",
                    torch.isnan(state_token).any().item(),
                    "inf:",
                    torch.isinf(state_token).any().item(),
                )
                print(
                    "flow_debug state_token range:",
                    state_token.min().item(),
                    state_token.max().item(),
                )
                raise ValueError("FlowMatchingActionExpert: state_token contains NaN or Inf")
            suffix_tokens.append(state_token)
        
        # 2. Action tokens: [B, H, action_dim] -> [B, H, hidden_size]
        action_tokens = self.action_in_proj(actions)  # [B, H, hidden_size]
        if not torch.isfinite(action_tokens).all():
            print(
                "flow_debug action_tokens nan:",
                torch.isnan(action_tokens).any().item(),
                "inf:",
                torch.isinf(action_tokens).any().item(),
            )
            print(
                "flow_debug action_tokens range:",
                action_tokens.min().item(),
                action_tokens.max().item(),
            )
            raise ValueError("FlowMatchingActionExpert: action_tokens contains NaN or Inf")
        suffix_tokens.append(action_tokens)
        
        # 3. Time token: [B] -> [B, 1, hidden_size]
        time_embed = create_sinusoidal_pos_embedding(
            time, 
            self.hidden_size, 
            min_period=4e-3, 
            max_period=4.0, 
            device=device
        )
        if not torch.isfinite(time_embed).all():
            print(
                "flow_debug time_embed_raw nan:",
                torch.isnan(time_embed).any().item(),
                "inf:",
                torch.isinf(time_embed).any().item(),
            )
            raise ValueError("FlowMatchingActionExpert: time_embed_raw contains NaN or Inf")
        time_embed = time_embed.to(dtype=dtype)
        time_embed = self.time_mlp_in(time_embed)
        time_embed = F.silu(time_embed)
        time_embed = self.time_mlp_out(time_embed)
        if not torch.isfinite(time_embed).all():
            print(
                "flow_debug time_embed_mlp nan:",
                torch.isnan(time_embed).any().item(),
                "inf:",
                torch.isinf(time_embed).any().item(),
            )
            print(
                "flow_debug time_embed_mlp range:",
                time_embed.min().item(),
                time_embed.max().item(),
            )
            raise ValueError("FlowMatchingActionExpert: time_embed_mlp contains NaN or Inf")
        time_token = time_embed.unsqueeze(1)  # [B, 1, hidden_size]
        suffix_tokens.append(time_token)
        
        # Concatenate all tokens
        suffix_embs = torch.cat(suffix_tokens, dim=1)  # [B, suffix_seq_len, hidden_size]
        # print("flow_debug suffix_embs shape:", tuple(suffix_embs.shape))
        return suffix_embs
    
    def forward(
        self,
        prefix_embs: torch.Tensor,  # [B, prefix_seq_len, hidden_size] from wrapper
        actions: torch.Tensor,       # [B, H, action_dim] - noisy or clean actions
        time: torch.Tensor,          # [B] - time in [0, 1]
        state: Optional[torch.Tensor] = None,  # [B, state_dim]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional = None,
    ) -> torch.Tensor:
        """
        Forward pass through deep fusion model.
        
        Args:
            prefix_embs: [B, L_p, H] prefix embeddings (image + geo + text)
            actions: [B, H, action_dim] noisy actions (x_t)
            time: [B] time values
            state: [B, state_dim] proprioceptive state (optional)
            attention_mask: [B, L_p + L_s] joint attention mask
            position_ids: [B, L_p + L_s] position ids
            
        Returns:
            pred_velocity: [B, H, action_dim] predicted velocity
        """
        suffix_embs = self._construct_suffix_embeddings(actions, time, state)
        if not torch.isfinite(suffix_embs).all():
            print("flow_debug suffix_embs nan:", torch.isnan(suffix_embs).any().item(), "inf:", torch.isinf(suffix_embs).any().item())
            print("flow_debug suffix_embs range:", suffix_embs.min().item(), suffix_embs.max().item())
            raise ValueError("FlowMatchingActionExpert: suffix_embs contains NaN or Inf")
        else:
            with torch.no_grad():
                s_min = suffix_embs.min().item()
                s_max = suffix_embs.max().item()
                # print("flow_debug suffix_embs ok range:", s_min, s_max)
        
        if prefix_embs is None and past_key_values is not None:
            outputs, _ = self.llava_with_expert(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
            )
        else:
            outputs, _ = self.llava_with_expert(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,  # No cache in training
            )
        
        prefix_output, suffix_output = outputs
        hook_model = self.llava_with_expert
        if getattr(hook_model, "prefix_feature_hook_enabled", False) and hasattr(hook_model, "_record_prefix_stats"):
            try:
                hook_model._record_prefix_stats(-1, prefix_output)
            except Exception:
                pass
        if not torch.isfinite(suffix_output).all():
            print("flow_debug suffix_output nan:", torch.isnan(suffix_output).any().item(), "inf:", torch.isinf(suffix_output).any().item())
            print("flow_debug suffix_output range:", suffix_output.min().item(), suffix_output.max().item())
            raise ValueError("FlowMatchingActionExpert: suffix_output contains NaN or Inf")
        
        # Extract action tokens from suffix output
        # suffix_output: [B, suffix_seq_len, hidden_size]
        # We need to extract the action tokens (skipping state and time tokens)
        
        if self.use_state and self.state_proj is not None:
            # Structure: [state_token, action_tokens, time_token]
            action_hidden = suffix_output[:, 1:1+self.action_horizon, :]  # [B, H, hidden_size]
        else:
            # Structure: [action_tokens, time_token]
            action_hidden = suffix_output[:, :self.action_horizon, :]  # [B, H, hidden_size]
        
        # Project to action velocity
        pred_velocity = self.action_out_proj(action_hidden)  # [B, H, action_dim]
        with torch.no_grad():
            self.last_forward_metrics = {
                "suffix_hidden_abs_mean": float(suffix_output.abs().mean()),
                "suffix_hidden_std": float(suffix_output.std()),
                "action_hidden_abs_mean": float(action_hidden.abs().mean()),
                "action_hidden_std": float(action_hidden.std()),
            }
        return pred_velocity
    
    def compute_loss(
        self,
        prefix_embs: torch.Tensor,
        actions: torch.Tensor,  # [B, H, action_dim] - ground truth clean actions
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Compute Flow Matching loss.
        
        Flow Matching formulation:
        - Sample t ~ Uniform(0, 1)
        - Sample noise ~ N(0, I)
        - Construct x_t = t * noise + (1 - t) * actions
        - Target velocity: u_t = noise - actions
        - Predict velocity: v_t = model(prefix, x_t, t)
        - Loss: MSE(v_t, u_t)
        
        Args:
            prefix_embs: [B, L_p, H] prefix embeddings
            actions: [B, H, action_dim] ground truth clean actions
            state: [B, state_dim] proprioceptive state (optional)
            attention_mask: [B, L_p + L_s] joint attention mask
            position_ids: [B, L_p + L_s] position ids
            
        Returns:
            loss: scalar tensor
        """
        actions_f32 = actions.to(torch.float32)
        batch_size = actions_f32.shape[0]
        device = actions_f32.device

        if not torch.isfinite(actions_f32).all():
            print("flow_debug actions_f32 nan:", torch.isnan(actions_f32).any().item(), "inf:", torch.isinf(actions_f32).any().item())
            print("flow_debug actions_f32 range:", actions_f32.min().item(), actions_f32.max().item())
            raise ValueError("FlowMatchingActionExpert: actions_f32 contains NaN or Inf")
        if not torch.isfinite(prefix_embs).all():
            print("flow_debug prefix_embs nan:", torch.isnan(prefix_embs).any().item(), "inf:", torch.isinf(prefix_embs).any().item())
            raise ValueError("FlowMatchingActionExpert: prefix_embs contains NaN or Inf")
        if state is not None and not torch.isfinite(state).all():
            print("flow_debug state nan:", torch.isnan(state).any().item(), "inf:", torch.isinf(state).any().item())
            raise ValueError("FlowMatchingActionExpert: state contains NaN or Inf")
        
        # Step 1: Sample time t ~ Beta(1.5, 1.0) (PI0 风格)，并做轻微裁剪保证数值稳定
        t = self.sample_time(batch_size=batch_size, device=device)
        
        # Step 2: Sample noise ~ N(0, I)
        noise = self.sample_noise(actions_f32.shape, device)
        if not torch.isfinite(noise).all():
            print("flow_debug noise nan:", torch.isnan(noise).any().item(), "inf:", torch.isinf(noise).any().item())
            raise ValueError("FlowMatchingActionExpert: noise contains NaN or Inf")
        
        # Step 3: Construct noisy actions x_t = t * noise + (1 - t) * actions
        t_exp = t.view(batch_size, 1, 1)  # [B, 1, 1]
        x_t = t_exp * noise + (1 - t_exp) * actions_f32
        if not torch.isfinite(x_t).all():
            print("flow_debug x_t nan:", torch.isnan(x_t).any().item(), "inf:", torch.isinf(x_t).any().item())
            print("flow_debug x_t range:", x_t.min().item(), x_t.max().item())
            raise ValueError("FlowMatchingActionExpert: x_t contains NaN or Inf")
        
        # Step 4: Target velocity u_t = noise - actions
        target_velocity = noise - actions_f32
        if not torch.isfinite(target_velocity).all():
            print("flow_debug target_velocity nan:", torch.isnan(target_velocity).any().item(), "inf:", torch.isinf(target_velocity).any().item())
            print("flow_debug target_velocity range:", target_velocity.min().item(), target_velocity.max().item())
            raise ValueError("FlowMatchingActionExpert: target_velocity contains NaN or Inf")
        
        # Step 5: Predict velocity v_t = model(prefix, x_t, t)
        pred_velocity = self.forward(
            prefix_embs=prefix_embs,
            actions=x_t,
            time=t,
            state=state,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if not torch.isfinite(pred_velocity).all():
            print("flow_debug pred_velocity nan:", torch.isnan(pred_velocity).any().item(), "inf:", torch.isinf(pred_velocity).any().item())
            print("flow_debug pred_velocity range:", pred_velocity.min().item(), pred_velocity.max().item())
            raise ValueError("FlowMatchingActionExpert: pred_velocity contains NaN or Inf")
        
        pred_velocity = pred_velocity.to(torch.float32)
        target_velocity = target_velocity.to(torch.float32)
        sq_err = (pred_velocity - target_velocity) ** 2
        if self.use_time_weight:
            weight = (1.0 - t).view(batch_size, 1, 1).to(dtype=sq_err.dtype)
            weighted_sq_err = sq_err * weight
            loss = weighted_sq_err.mean() / weight.mean()
            weight_mean = float(weight.mean())
            weighted_vel_mse = float(loss.detach())
        else:
            weight = torch.ones_like(sq_err)
            loss = sq_err.mean()
            weight_mean = 1.0
            weighted_vel_mse = float(loss.detach())
        with torch.no_grad():
            pred_flat = pred_velocity.view(batch_size, -1)
            target_flat = target_velocity.view(batch_size, -1)
            vel_cos = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1)
            metrics = {
                "t_mean": float(t.mean()),
                "t_min": float(t.min()),
                "t_max": float(t.max()),
                "actions_abs_mean": float(actions_f32.abs().mean()),
                "x_t_abs_mean": float(x_t.abs().mean()),
                "target_vel_abs_mean": float(target_velocity.abs().mean()),
                "pred_vel_abs_mean": float(pred_velocity.abs().mean()),
                "vel_mse": float(sq_err.mean()),
                "weight_mean": weight_mean,
                "weighted_vel_mse": weighted_vel_mse,
                "vel_cosine": float(vel_cos.mean()),
            }
            forward_metrics = getattr(self, "last_forward_metrics", None)
            if isinstance(forward_metrics, dict):
                metrics.update(forward_metrics)
            self.last_loss_metrics = metrics
        return loss
    
    @torch.no_grad()
    def sample_actions(
        self,
        prefix_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        prefix_past_key_values: Optional = None,
    ):
        """
        Sample actions using Euler ODE solver.
        
        ODE: dx/dt = v_t(x, t)
        Euler discretization: x_{t+dt} = x_t + v_t * dt
        
        We start from t=1 (pure noise) and integrate backwards to t=0 (clean action).
        
        Args:
            prefix_embs: [B, L_p, H] prefix embeddings
            state: [B, state_dim] proprioceptive state (optional)
            num_steps: number of Euler steps
            attention_mask: [B, L_p] prefix attention mask (suffix will be appended)
            position_ids: [B, L_p] prefix position ids (suffix will be appended)
            
        Returns:
            x_0: [B, H, action_dim] predicted clean actions
        """
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device
        
        # Initialize with pure noise at t=1
        action_shape = (batch_size, self.action_horizon, self.action_dim)
        x_t = self.sample_noise(action_shape, device)
        
        # Time step: dt = -1 / num_steps (going from t=1 to t=0)
        dt = -1.0 / num_steps
        
        # Euler integration loop
        for step in range(num_steps):
            # Current time: t = 1 + step * dt
            t_curr = 1.0 + step * dt
            t_tensor = torch.full((batch_size,), t_curr, device=device)
            
            # Predict velocity v_t at current x_t and t
            if prefix_past_key_values is not None:
                v_t = self.forward(
                    prefix_embs=None,
                    actions=x_t,
                    time=t_tensor,
                    state=state,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=prefix_past_key_values,
                )
            else:
                v_t = self.forward(
                    prefix_embs=prefix_embs,
                    actions=x_t,
                    time=t_tensor,
                    state=state,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            
            # Euler step: x_{t+dt} = x_t + v_t * dt
            x_t = x_t + v_t * dt
        
        return x_t
