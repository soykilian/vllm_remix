# vLLM-REMIX

> **REMIX** — **R**untime **E**xecution **M**apping for **I**nferencing on heterogeneous **X** (accelerators)

A fork of [vLLM](https://github.com/vllm-project/vllm) that adds a heterogeneous-device partitioning framework called **REMIX** on top of vLLM's `torch.compile`-based piecewise compilation backend. REMIX lets you split a single LLM graph across multiple device types — GPU, attention accelerators, PIM, custom NPUs — and route each subgraph to the backend that runs it most efficiently.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Partition Configuration](#partition-configuration)
- [Backends](#backends)
- [Timing & Profiling](#timing--profiling)
- [Supported Models](#supported-models)
- [Repository Layout](#repository-layout)
- [Citation / Upstream](#citation--upstream)

---

## Overview

Standard vLLM compiles the full model graph with a single backend (Inductor). REMIX intercepts the compiled `torch.fx` graph, applies module-level partition rules defined in a YAML file, splits the graph into subgraphs, and dispatches each subgraph to a pluggable `BaseBackend`. Device transfers between subgraphs are handled transparently.

Key additions over upstream vLLM:

| Feature | Description |
|---|---|
| YAML partition rules | Assign any named module (or glob pattern) to a named device |
| Multi-device dispatch | Each subgraph runs on its own backend with automatic host/device tensor transfers |
| Pluggable backends | `CUDAInductorBackend`, `AttAccBackend`, `TRACEBackend`, `PassthroughBackend` |
| Performance databases | CSV-based lookup tables for GEMM, attention, and MoE latencies on specific hardware |
| Timing collector | Per-subgraph latency instrumentation that writes JSON traces |
| Simulation mode | Estimate end-to-end latency from performance databases without running on real hardware |

---

## Installation

> TODO: fill in any additional prerequisites beyond standard vLLM

```bash
# 1. Clone this repo
git clone <this-repo-url> vLLM-REMIX
cd vLLM-REMIX

# 2. Install vLLM in editable mode (picks up the REMIX changes)
pip install -e .

# 3. (Optional) install TADS if using TRACEBackend
#    TADS must be importable on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/TADS
```

---

## Quick Start

```python
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
...
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=64))
```

> **TODO**: replace the `CompilationConfig` field name with the actual parameter once the API is finalised.

---

## Partition Configuration

Partitions are specified in a YAML file. Each line maps a **module glob pattern** to a **device name**. Rules are matched in order; the first match wins.

```yaml
# yaml/llama/remix_partition_attacc.yaml
layers.*.self_attn.qkv_proj: h100
layers.*.self_attn.o_proj:   h100
layers.*.mlp.gate_up_proj:   h100
layers.*.mlp.down_proj:      h100
layers.*.input_layernorm:    h100
```

Patterns support `*` (wildcard) and explicit layer indices (e.g. `layers.0`).


## Backends

All backends inherit from [`BaseBackend`](remix_backend/backends/base_backend.py).

| Class | File | Description |
|---|---|---|
| `CUDAInductorBackend` | `inductor_backend.py` | Compiles each subgraph with `torch._inductor.compile_fx` for GPU execution. Also exports Chrome traces for per-subgraph kernel profiling. |
| `AttAccBackend` | `attacc_backend.py` | Latency-lookup backend for an attention accelerator. Reads pre-profiled CSV data and returns simulated latency without executing on real hardware. |
| `TRACEBackend` | `trace_backend.py` | Integrates with the TADS compile framework for custom NPU backends. Applies graph rewriting patterns (FuseKQV, RMSNorm fusion, etc.) before dispatching. |

### Adding a custom backend

```python
from remix_backend.backends.base_backend import BaseBackend
import torch

class MyAcceleratorBackend(BaseBackend):
    def backend(self, graph_module, example_inputs):
        # compile or wrap graph_module for your hardware
        return compiled_fn

    def move_to_device(self, x):
        return x.to("my_device")

    def return_latency(self, profiler, meta_ctx) -> float:
        # return measured or estimated latency in seconds
        return 0.0
```

Register the backend by creating a [`Device`](remix_backend/device.py) and adding it to the device map used by `RemixAdaptor`.

---



---

## Timing & Profiling

[`TimingCollector`](remix_backend/timings.py) wraps each subgraph call and records per-partition latency:

```python
from remix_backend.timings import TimingCollector

collector = TimingCollector(path="profiler/my_run.json")
collector.is_active = True
# ... run inference ...
collector.summary()   # prints TTFT, TPOT, per-partition breakdown
```

`CUDAInductorBackend` additionally exports Chrome traces (`*.json`) for each subgraph that can be loaded in `chrome://tracing` or Perfetto.

---

## Supported Models

The following models have been tested with REMIX partitioning:

| Model | Partition YAMLs |
|---|---|
| LLaMA (3 / 3.1 / 3.2) | `yaml/llama/` |
| Bamba (Mamba2 hybrid) | `yaml/bamba/` |
| Mixtral (MoE) | `yaml/mixtral/` |

---

## Repository Layout

```
vLLM-REMIX/
├── remix_backend/              # Core REMIX library
│   ├── backends/               # Pluggable compute backends
│   │   ├── base_backend.py
│   │   ├── inductor_backend.py
│   │   ├── attacc_backend.py
│   │   ├── trace_backend.py
│   │   └── passthrough_backend.py
│   ├── database/               # Hardware performance databases
│   │   ├── base.py
│   │   ├── perf_database.py
│   │   ├── attacc_database.py
│   │   └── trace_database.py
│   ├── partitioning.py         # FX graph splitting logic
│   ├── device.py               # Device dataclass
│   ├── timings.py              # Timing instrumentation
│   └── utils.py                # Graph visualisation helpers
├── vllm/compilation/
│   ├── backends.py             # VllmBackend + RemixAdaptor (modified)
│   └── piecewise_backend.py    # Piecewise compile backend (modified)
├── yaml/                       # Example partition configurations
│   ├── llama/
│   ├── bamba/
│   └── mixtral/
├── tests/                      # Upstream vLLM tests
├── test_graph_split.py         # REMIX integration tests
└── README_REMIX.md             # This file
```

---

## Citation / Upstream


> **TODO**: add a citation for the REMIX work once published.
