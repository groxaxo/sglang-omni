# ZONOS3-Mini GPTQ-Pro INT4 experts (opt-in)

A standalone GPTQ-Pro plugin at groxaxo/GPTQ-Pro/examples/zonos3_mini/sglang_omni_moe.py
makes SGLang-Omni's ZONOS3-Mini speech pipeline use its published packed
**GPTQ-Pro INT4/BF16** MoE expert weights instead of expanding every expert to BF16.

The integration is opt-in and does not change the model's attention, GDN,
sampling, conditioning or ZAC audio decoder. It requires the original
Zyphra BF16 checkpoint; the overlay is not standalone.

Example setup:

~~~bash
# SGLang 0.5.23 + SGLang-Omni + PyTorch/Triton CUDA environment
export PYTHONPATH=/path/to/GPTQ-Pro:/path/to/sglang-omni:$PYTHONPATH
export ZONOS3_GPTQPRO_BASE=/models/ZONOS3-Mini/lm/zonos3_mini
export ZONOS3_GPTQPRO_OVERLAY=groxaxo/ZONOS3-Mini-GPTQ-Pro-INT4

sgl-omni serve --model-path /models/ZONOS3-Mini \
  --host 127.0.0.1 --port 18082 \
  --tts_engine.gpu_memory_fraction 0.65 \
  --tts_engine.factory.max_running_requests 2 \
  --tts_engine.engine.cuda_graph_max_bs 2
~~~

The optional loader validates the source model SHA256, all 11 layer-shard
checksums, mixed-precision projection manifest and each expert's exact shape.
It keeps BF16-preserved experts in BF16, and runs the other 1005 expert
projections directly from packed int32 INT4 matrices. The SGLang expert IDs
are read **on the GPU**. There is no Python .item() during routing, so CUDA
graph capture and replay can change experts without host synchronization.
Tensor parallelism is not supported.

Missing plugin, incompatible file, wrong checkpoint or shape mismatch cause
startup to fail instead of silently running at a different precision. Unset
ZONOS3_GPTQPRO_OVERLAY for normal BF16 SGLang-Omni.

SGLang-Omni's kernel is a separate implementation from the original GPTQ-Pro
native C++ kernel and requires its own performance and speech-quality tests.
