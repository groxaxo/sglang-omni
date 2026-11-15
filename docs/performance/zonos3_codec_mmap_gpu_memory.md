# ZONOS3-Mini codec GPU-memory investigation — 16 November 2026

## Why a 3090 reports ~15.4 GiB

The released Zyphra source includes a 7,960,647,220-byte generator.pt and
a 4,012,134,189-byte full-mr iSTFT head checkpoint on disk. These archives
contain more than the active inference model, and are not 12 GiB of loaded
model parameters.

The actual SGLang-Omni Zonos3CodecDecoder module contains:

| Group | Parameter memory at runtime |
| --- | ---: |
| Transformer stages | 1.817 GiB |
| Quantizer | 0.052 GiB |
| iSTFT head | 0.055 GiB |
| All active decoder parameters | 1.924 GiB |

The old load_checkpoint_state function deserialized the multi-section
checkpoints directly into CUDA. Even after unused checkpoint tensors were
released, the PyTorch allocator retained a large cached pool, still counted
by NVIDIA NVML / nvidia-smi.

## Safe fix

- torch.load on CPU with mmap=True and weights_only=True
- Transfer only the requested section to its inference device as FP32
- Fold its weight norms on the inference device, preserving GPU arithmetic
- After decoder load and BF16 casting, empty_cache before CUDA-graph capture

PyTorch documentation confirms that empty_cache returns *unoccupied cached*
GPU memory to other applications without removing live tensors.

## Direct A/B results, same RTX 3090 model and source checkpoint

Both baseline and optimized SGLang-Omni used the same 11-layer GPTQ-Pro
INT4/BF16 language model, same original ZAC decoder and checkpoint, 0.40
GPU memory fraction, request concurrency 2 and decode graph batch sizes 1/2.
The baseline was stopped before starting the optimized worker on physical
GPU2, where a separate Parakeet process occupied ~2144 MiB throughout.

| Metric | Original loader | Optimized loader |
| --- | ---: | ---: |
| Worker NVML MiB, idle | 15,446 | 11,106 |
| Worker NVML MiB after speech requests | 15,542 | 11,310 |
| VRAM saved after inference | — | 4,232 MiB (~27.2%) |
| SGLang health | healthy | healthy |
| Prefill/decode CUDA graphs | pass | pass |
| End-to-end English & Spanish WAV | pass | pass |
| Two simultaneous requests | — | pass |

The optimized decoder logged:
allocated 5.638 GiB;
reserved before 12.635 GiB;
reserved after 8.186 GiB;
unused cached memory released **4.449 GiB**.

Repeated Spanish text+seed yielded identical WAV SHA256 across 3 requests on
each of the original and optimized servers, and 2/3 of the initial paired
utterances were byte-for-byte equal. One initial utterance length differed,
so this is not a blanket bit-exact proof across all possible sampling paths.

No speech quality or latency superiority is claimed from this tiny test.
The existing Haru-cascade running voice service was NOT modified or stopped.

## More GPTQ-Pro codec quantization potential

The active decoder includes 329 eligible transformer Linear matrices, using
approximately 1.816 GiB of BF16 weights. Their theoretical INT4 floor is
one quarter of current BF16 weight bytes (before metadata), so theoretical
maximum additional savings are ~1.36 GiB. For best quality, begin with FFN:
64 large stage-0 FFN matrices currently use 800 MiB and could save at most
600 MiB; remaining stage FFNs use 432 MiB, up to 324 MiB extra.
Preserve quantizer, convolution, normalization and iSTFT waveform head
until real speech calibration and listening/ASR/codec objective gates pass.

Unlike the cache-load fix, actual decoder GPTQ-Pro requires calibration from
real codebook streams, quantizing and packing weights with the GPTQ-Pro solver,
and new CUDA kernels/hooks for decoder stage nn.Linear layers. The current
SGLang-Omni decoder is still BF16 after this safe loader optimization.
