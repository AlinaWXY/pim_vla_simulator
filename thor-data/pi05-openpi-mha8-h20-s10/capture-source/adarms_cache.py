"""Reversible inference-only time-conditioning cache for pinned OpenPI pi0.5.

The original sampler, denoiser, AdaRMS normalization and residual arithmetic
are unchanged. Cache construction uses the sampler's FP32 accumulated schedule,
not a linspace (which would produce slightly different timesteps).
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


class Pi05ConditioningCache:
    def __init__(self, model, num_steps=10, batch_size=1):
        from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

        if not model.pi05 or model.training:
            raise ValueError("Only eval-mode pi0.5 is supported")
        if num_steps < 1 or batch_size < 1:
            raise ValueError("Positive steps and batch size required")
        if torch.is_autocast_enabled():
            raise ValueError("Construct the cache outside autocast")
        self.model = model
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = model.time_mlp_in.weight.device
        self.native_sample = model.sample_actions
        self.native_embed = model.embed_suffix
        expert = model.paligemma_with_expert.gemma_expert.model
        self.norms = [(name, module) for name, module in expert.named_modules()
                      if getattr(module, "cond_dim", None) is not None
                      and getattr(module, "dense", None) is not None]
        if len(self.norms) != 2 * len(expert.layers) + 1:
            raise ValueError("Unexpected expert AdaRMS coverage")
        self.conditions = []
        self.times = []
        self.modulations = {name: {} for name, _ in self.norms}
        self.active = False
        self.installed = False
        self.step_index = 0
        self.dense_hits = 0
        self.requests = 0
        with torch.inference_mode():
            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=self.device)
            time = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            while time >= -dt / 2:
                timestep = time.expand(batch_size)
                self.times.append(timestep.clone())
                emb = create_sinusoidal_pos_embedding(
                    timestep, model.action_in_proj.out_features,
                    min_period=4e-3, max_period=4.0, device=timestep.device,
                ).type(dtype=timestep.dtype)
                cond = F.silu(model.time_mlp_out(F.silu(model.time_mlp_in(emb))))
                self.conditions.append(cond)
                for name, norm in self.norms:
                    self.modulations[name][id(cond)] = norm.dense(cond)
                time += dt
        if len(self.conditions) != num_steps:
            raise RuntimeError("Native accumulated schedule has unexpected step count")
        self.signature = self._signature()

    def _signature(self):
        parameters = list(self.model.time_mlp_in.parameters()) + list(self.model.time_mlp_out.parameters())
        for _, norm in self.norms:
            parameters.extend(norm.dense.parameters())
        return tuple((id(p), p._version, p.data_ptr(), p.dtype, p.device, tuple(p.shape))
                     for p in parameters)

    def _validate(self, observation, noise, num_steps):
        if not self.installed or self.active:
            raise RuntimeError("Cache must be installed and is not reentrant")
        if self.model.training or torch.is_grad_enabled() or torch.is_autocast_enabled():
            raise RuntimeError("Cached inference requires eval, no-grad and no autocast")
        if num_steps != self.num_steps or observation.state.shape[0] != self.batch_size:
            raise ValueError("Rebuild cache for a different timestep schedule or batch size")
        if observation.state.device != self.device or (noise is not None and noise.device != self.device):
            raise ValueError("Rebuild cache after changing device")
        if self._signature() != self.signature:
            raise RuntimeError("Conditioning parameters changed; rebuild cache")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        self._validate(observation, noise, num_steps)
        self.active = True
        self.step_index = 0
        self.dense_hits = 0
        try:
            result = self.native_sample(device, observation, noise=noise, num_steps=num_steps)
            if self.step_index != self.num_steps or self.dense_hits != self.num_steps * len(self.norms):
                raise RuntimeError("Cache did not cover exactly the native denoising schedule")
            self.requests += 1
            return result
        finally:
            self.active = False

    def embed_suffix(self, state, noisy_actions, timestep):
        if not self.active:
            return self.native_embed(state, noisy_actions, timestep)
        if self.step_index >= self.num_steps:
            raise RuntimeError("Unexpected extra denoising step")
        cond = self.conditions[self.step_index]
        self.step_index += 1
        # Preserve the pi0.5 suffix operations, omitting only time conditioning.
        action_emb = self.model._apply_checkpoint(self.model.action_in_proj, noisy_actions)
        embs = torch.cat([action_emb], dim=1)
        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks = torch.cat([action_time_mask], dim=1)
        masks = [1] + [0] * (self.model.config.action_horizon - 1)
        att_masks = torch.tensor(masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(masks))
        return embs, pad_masks, att_masks, cond

    @contextmanager
    def applied(self):
        if self.installed:
            raise RuntimeError("Cache already installed")
        restores = []

        def replace(obj, name, value):
            restores.append((obj, name, getattr(obj, name), name in obj.__dict__))
            setattr(obj, name, value)

        try:
            replace(self.model, "embed_suffix", self.embed_suffix)
            replace(self.model, "sample_actions", self.sample_actions)
            for name, norm in self.norms:
                original = norm.dense.forward
                table = self.modulations[name]

                def dense_forward(cond, original=original, table=table):
                    if not self.active:
                        return original(cond)
                    value = table.get(id(cond))
                    if value is None:
                        raise RuntimeError("Unexpected dynamic condition in cached denoising")
                    self.dense_hits += 1
                    return value

                replace(norm.dense, "forward", dense_forward)
            self.installed = True
            yield self
        finally:
            self.installed = False
            self.active = False
            for obj, name, original, had_local in reversed(restores):
                if had_local:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)

    def metadata(self):
        tensors = self.conditions + self.times
        tensors += [v for table in self.modulations.values() for v in table.values()]
        return {"steps": self.num_steps, "batch_size": self.batch_size,
                "norm_count": len(self.norms), "norm_names": [n for n, _ in self.norms],
                "condition_shape": list(self.conditions[0].shape),
                "modulation_shape": list(next(iter(self.modulations.values()))[id(self.conditions[0])].shape),
                "dtype": str(self.conditions[0].dtype),
                "bytes": sum(t.numel() * t.element_size() for t in tensors),
                "times": [t.tolist() for t in self.times], "completed_requests": self.requests,
                "last_request_dense_hits": self.dense_hits,
                "scope": "time embedding, time MLP, dense modulation outputs; not normalization or residuals"}
