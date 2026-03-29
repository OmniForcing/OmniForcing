"""
Loss functions for diffusion model distillation.

Standard loss functions for denoising diffusion models:
- X0Pred: MSE on predicted clean samples
- Velocity: MSE on predicted velocity (flow matching)
- Flow: MSE on predicted flow (same as velocity)
- Noise: MSE on predicted noise (DDPM-style)
- VPred: Weighted MSE with variance weighting
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


class DenosingLoss(ABC):
    """Base class for denoising loss functions."""

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: Optional[torch.Tensor],
        alphas_cumprod: Optional[torch.Tensor],
        timestep: torch.Tensor,
        flow_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute denoising loss.

        Args:
            x: Clean samples
            x_pred: Predicted clean samples
            noise: Original noise
            noise_pred: Predicted noise (optional)
            alphas_cumprod: Cumulative alpha products (for DDPM)
            timestep: Current timestep
            flow_pred: Predicted flow (for flow matching)

        Returns:
            Loss value
        """
        pass


class X0PredLoss(DenosingLoss):
    """
    X0 prediction loss: MSE between clean sample and predicted clean sample.

    Simple and direct loss for models that predict x0.
    """

    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: Optional[torch.Tensor],
        alphas_cumprod: Optional[torch.Tensor],
        timestep: torch.Tensor,
        flow_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.mse_loss(x, x_pred)


class VelocityPredLoss(DenosingLoss):
    """
    Velocity prediction loss for flow matching.

    Velocity is defined as: v = epsilon - x_0
    Loss: MSE(v_pred, v_true)
    """

    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: Optional[torch.Tensor],
        alphas_cumprod: Optional[torch.Tensor],
        timestep: torch.Tensor,
        flow_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute true velocity
        v_true = noise - x

        if flow_pred is not None:
            # Use provided flow prediction
            return F.mse_loss(flow_pred, v_true)
        else:
            # Derive velocity from x0 prediction: v = eps - x_0
            # We can compute this from x_pred
            # If x_pred is the predicted x0, then v_pred = noise - x_pred
            v_pred = noise - x_pred
            return F.mse_loss(v_pred, v_true)


class FlowPredLoss(DenosingLoss):
    """
    Flow matching prediction loss.

    Flow is defined as: flow = epsilon - x_0 (same as velocity)
    This is the standard loss used in flow matching models.
    """

    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: Optional[torch.Tensor],
        alphas_cumprod: Optional[torch.Tensor],
        timestep: torch.Tensor,
        flow_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # True flow: f = epsilon - x_0
        flow_true = noise - x

        if flow_pred is not None:
            return F.mse_loss(flow_pred, flow_true)
        else:
            # Compute predicted flow from x0 prediction
            # Since x_t = (1-t)*x_0 + t*eps, and we predict x_0
            # The flow should be eps - x_0
            flow_pred_computed = noise - x_pred
            return F.mse_loss(flow_pred_computed, flow_true)


class NoisePredLoss(DenosingLoss):
    """
    Noise prediction loss: MSE between noise and predicted noise.

    Traditional DDPM-style loss.
    """

    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: Optional[torch.Tensor],
        alphas_cumprod: Optional[torch.Tensor],
        timestep: torch.Tensor,
        flow_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise_pred is None:
            raise ValueError("noise_pred required for NoisePredLoss")
        return F.mse_loss(noise_pred, noise)


class VPredLoss(DenosingLoss):
    """
    V-prediction loss with variance weighting.

    Weighted MSE where weights depend on the noise schedule.
    """

    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: Optional[torch.Tensor],
        alphas_cumprod: Optional[torch.Tensor],
        timestep: torch.Tensor,
        flow_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if alphas_cumprod is None:
            # Fall back to simple MSE if no alpha schedule
            return F.mse_loss(x, x_pred)

        # Get alpha values for current timesteps
        alpha_t = alphas_cumprod[timestep.long()]

        # Reshape for broadcasting
        while alpha_t.dim() < x.dim():
            alpha_t = alpha_t.unsqueeze(-1)

        # Compute weights: 1 / (1 - alpha)
        weights = 1.0 / (1 - alpha_t + 1e-8)

        # Weighted MSE
        loss = weights * (x - x_pred) ** 2
        return loss.mean()


def get_denoising_loss(loss_type: str) -> type[DenosingLoss]:
    """
    Get denoising loss class by name.

    Args:
        loss_type: One of "x0", "velocity", "flow", "noise", "vpred"

    Returns:
        Loss class
    """
    loss_map = {
        "x0": X0PredLoss,
        "velocity": VelocityPredLoss,
        "flow": FlowPredLoss,
        "noise": NoisePredLoss,
        "vpred": VPredLoss,
    }

    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")

    return loss_map[loss_type]
