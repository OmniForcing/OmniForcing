import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class EMA_FSDP:
    """Exponential Moving Average for FSDP-wrapped modules.

    Stores full (unsharded) parameter copies on CPU in fp32.
    Uses FSDP.summon_full_params to gather sharded parameters for
    initialization and updates.
    """

    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1.0 - d)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
