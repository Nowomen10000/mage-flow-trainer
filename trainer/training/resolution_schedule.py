"""Step budgets for staged-resolution MOSS adapters; no epoch-based semantics."""
from dataclasses import dataclass, field
import math


@dataclass
class ResolutionStage:
    resolution: int
    share: float
    lr_mul: float = 1.0

    def __post_init__(self):
        if type(self.resolution) is not int or self.resolution <= 0:
            raise ValueError('Stage resolution must be a positive integer')
        if not math.isfinite(self.share) or self.share <= 0:
            raise ValueError('Stage share must be finite and positive')
        if isinstance(self.lr_mul, bool) or not math.isfinite(self.lr_mul) or not 0 < self.lr_mul <= 1:
            raise ValueError('Stage lr_mul must be finite and in (0, 1]')


@dataclass
class ResolutionSchedule:
    enabled: bool = False
    save_stage_end: bool = True
    stages: list = field(default_factory=list)

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.save_stage_end) is not bool:
            raise ValueError('Resolution schedule enabled/save_stage_end must be booleans')
        self.stages = [ResolutionStage(**s) if isinstance(s, dict) else s for s in self.stages]
        if not all(isinstance(s, ResolutionStage) for s in self.stages):
            raise ValueError('resolution_schedule.stages must contain resolution/share tables')
        if self.enabled:
            if not self.stages or not math.isclose(sum(s.share for s in self.stages), 1, abs_tol=1e-9):
                raise ValueError('Resolution stage shares must sum to 1')
            if len({s.resolution for s in self.stages}) != len(self.stages):
                raise ValueError('Each resolution must appear once in the stage plan')

    def budgets(self, steps):
        if type(steps) is not int or steps <= 0:
            raise ValueError('Staged resolution training requires a positive train.max_steps')
        exact = [steps * s.share / sum(p.share for p in self.stages) for s in self.stages]
        result = [math.floor(x) for x in exact]
        order = sorted(range(len(result)), key=lambda i: (-(exact[i] - result[i]), i))
        for i in order[:steps - sum(result)]:
            result[i] += 1
        if any(x < 1 for x in result):
            raise ValueError('Every resolution stage must receive at least one update')
        return result

    def validate(self, cfg):
        if not self.enabled:
            if cfg.train.pack_resolutions and isinstance(cfg.train.batch_size, dict):
                raise ValueError('Packed per-tier batch sizes require resolution_schedule.enabled=true')
            return
        from ..modeling.loader import resolve_model_family
        if resolve_model_family(cfg.train.model_path, cfg.train.model_family) != 'moss_image':
            raise ValueError('Staged resolution training currently supports MOSS only')
        if cfg.adapter.kind not in ('lora', 'lycoris_lora') or (cfg.adapter.kind == 'lycoris_lora' and cfg.adapter.lycoris_algo != 'locon'):
            raise ValueError('Staged resolution training requires PEFT LoRA or LyCORIS LoCon')
        self.budgets(cfg.train.max_steps)
        if not cfg.train.pack_resolutions or cfg.train.attention_backend not in ('torch_varlen', 'flash_attn_2'):
            raise ValueError('Staged resolution training requires packing and varlen attention')
        if not isinstance(cfg.train.batch_size, dict):
            raise ValueError('Staged resolution training requires per-tier train.batch_size')
        tiers = set(cfg.dataset.tiers)
        if {s.resolution for s in self.stages} != tiers or set(cfg.train.batch_size) != tiers:
            raise ValueError('Stage resolutions, dataset tiers and batch_size keys must match')
        if cfg.curriculum or cfg.flow.use_ot or cfg.self_flow.enabled or cfg.optimizer.gradient_release or cfg.perceptual.enabled or cfg.adversarial.enabled or cfg.preserve.enabled or cfg.rti.enabled:
            raise ValueError('Staged resolutions cannot combine with curriculum, OT, Self-Flow, gradient release, perceptual/adversarial loss, preservation or RTI')
        if cfg.dataset.source != 'latents':
            raise ValueError('Staged resolutions initially require cached latents for all tiers')
