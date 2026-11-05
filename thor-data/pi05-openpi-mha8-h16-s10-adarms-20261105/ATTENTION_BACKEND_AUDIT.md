# Attention Backend Audit

The raw capture's `dimensions.attention_implementation` field is a snapshot
taken immediately after synthetic MHA conversion, before the first call to
`sample_actions`. It reads `sdpa`. It must NOT be interpreted as the backend
used to obtain the latency results.

Read-only inspection of the same pinned deployed file on Thor confirms:

`runtime/openpi/src/openpi/models_pytorch/pi0_pytorch.py`:

- Line 392, in the sample_actions prefix path:
  `self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"`
- Line 448, in denoise_step:
  `self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"`

Its SHA256 is
`dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16`,
verified before both runs.

`runtime/overlay/transformers/models/gemma/modeling_gemma.py`, lines 312-314,
selects `eager_attention_forward` when the configuration is `eager`.
Its SHA256 is
`17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde`.

The measured AE trace independently contains separate QK BMM, softmax and
SV BMM kernels in all 50 steps. No fused attention kernel is presented as
separable softmax time. The original capture JSONs remain byte-for-byte
unchanged; this note corrects the interpretation of the early snapshot field.

This round preserves the previously accepted cached OpenPI implementation.
It is not a claim of fastest possible Thor deployment, nor a measurement of
vLLM-Omni, torch.compile or CUDA Graph inference. No backend was changed to
make profiling easier.
