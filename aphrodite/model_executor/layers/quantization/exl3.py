# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig

from aphrodite import _custom_ops as ops
from aphrodite.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
)

if TYPE_CHECKING:
    from aphrodite.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from aphrodite.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from aphrodite.model_executor.layers.quantization import QuantizationMethods
from aphrodite.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from aphrodite.model_executor.parameter import BaseAphroditeParameter
from aphrodite.model_executor.utils import set_weight_attrs
from aphrodite.platforms import current_platform

logger = init_logger(__name__)
_EXL3_MOE_MAX_TOKENS_PER_EXPERT = 128
_EXL3_MOE_MAX_EXPERTS_PER_TOKEN = 32
_EXL3_MOE_ACT_SILU = 0


def _get_exl3_moe_down_tuning(
    *,
    device: torch.device,
    k_down: int,
    intermediate_size: int,
    hidden_size: int,
) -> tuple[int, int]:
    """Return correctness-verified EXL3 MoE down-projection tuning.

    Keep this deliberately narrow: EXL3 MGEMM shape choices are not purely
    performance hints, and invalid choices can silently corrupt generation.
    Candidate additions should first pass
    benchmarks/kernels/bench_exl3_mgemm_correctness.py and an end-to-end
    generation sanity check.
    """
    device_props = torch.cuda.get_device_properties(device)
    if device_props.major == 12 and k_down == 4 and intermediate_size == 256 and hidden_size == 1024:
        return 2, 32
    return -1, 0


@torch.library.custom_op(
    "aphrodite::exl3_linear_one",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_linear_one(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    out_features = trellis.shape[1] * 16
    output = torch.empty(
        (x.shape[0], out_features),
        device=x.device,
        dtype=torch.float16,
    )
    x_had = torch.empty_like(x)

    if x.shape[0] <= 32:
        ops.exl3_gemm(
            x,
            trellis,
            output,
            suh,
            x_had,
            svh,
            -1,
            mcg,
            mul1,
            0,
        )
        return output

    weight = torch.empty(
        (trellis.shape[0] * 16, out_features),
        device=trellis.device,
        dtype=torch.float16,
    )
    ops.exl3_reconstruct(
        weight,
        trellis,
        # EXL3 reconstruct expects K where packed.shape[2] == 16 * K.
        trellis.shape[2] // 16,
        mcg,
        mul1,
    )
    ops.exl3_had_r_128(
        x,
        x_had,
        suh,
        None,
        1.0,
    )
    ops.exl3_hgemm(
        x_had,
        weight,
        output,
    )
    ops.exl3_had_r_128(
        output,
        output,
        None,
        svh,
        1.0,
    )
    return output


@_exl3_linear_one.register_fake
def _exl3_linear_one_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        device=x.device,
        dtype=torch.float16,
    )


@torch.library.custom_op(
    "aphrodite::exl3_gate_up",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_gate_up(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    gate_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_trellis: torch.Tensor,
    up_suh: torch.Tensor,
    up_svh: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    k: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    if x.shape[0] > 32:
        return torch.cat(
            [
                _exl3_linear_one(x, gate_trellis, gate_suh, gate_svh, mcg, mul1),
                _exl3_linear_one(x, up_trellis, up_suh, up_svh, mcg, mul1),
            ],
            dim=-1,
        )

    x_3d = x.view(1, x.shape[0], x.shape[1])
    out_features = gate_trellis.shape[1] * 16
    output = torch.empty(
        (2, x.shape[0], out_features),
        device=x.device,
        dtype=torch.float16,
    )
    x_had = torch.empty(
        (2, x.shape[0], x.shape[1]),
        device=x.device,
        dtype=torch.float16,
    )
    ops.exl3_mgemm(
        x_3d,
        ptrs_trellis,
        output,
        ptrs_suh,
        x_had,
        ptrs_svh,
        None,
        None,
        k,
        -1,
        mcg,
        mul1,
        -1,
        -1,
        0,
    )
    if x.shape[0] == 1:
        return output.view(1, out_features * 2)
    return torch.cat([output[0], output[1]], dim=-1)


@_exl3_gate_up.register_fake
def _exl3_gate_up_fake(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    gate_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_trellis: torch.Tensor,
    up_suh: torch.Tensor,
    up_svh: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    k: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del gate_suh, gate_svh, up_trellis, up_suh, up_svh
    del ptrs_trellis, ptrs_suh, ptrs_svh, k, mcg, mul1
    return torch.empty(
        (x.shape[0], gate_trellis.shape[1] * 32),
        device=x.device,
        dtype=torch.float16,
    )


@torch.library.custom_op(
    "aphrodite::exl3_qkv",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_qkv(
    x: torch.Tensor,
    q_trellis: torch.Tensor,
    q_suh: torch.Tensor,
    q_svh: torch.Tensor,
    k_trellis: torch.Tensor,
    k_suh: torch.Tensor,
    k_svh: torch.Tensor,
    v_trellis: torch.Tensor,
    v_suh: torch.Tensor,
    v_svh: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    k: int,
    q_mcg: bool,
    q_mul1: bool,
    kv_mcg: bool,
    kv_mul1: bool,
) -> torch.Tensor:
    q = _exl3_linear_one(x, q_trellis, q_suh, q_svh, q_mcg, q_mul1)
    if x.shape[0] > 32:
        return torch.cat(
            [
                q,
                _exl3_linear_one(x, k_trellis, k_suh, k_svh, kv_mcg, kv_mul1),
                _exl3_linear_one(x, v_trellis, v_suh, v_svh, kv_mcg, kv_mul1),
            ],
            dim=-1,
        )

    x_3d = x.view(1, x.shape[0], x.shape[1])
    out_features = k_trellis.shape[1] * 16
    output = torch.empty(
        (2, x.shape[0], out_features),
        device=x.device,
        dtype=torch.float16,
    )
    x_had = torch.empty(
        (2, x.shape[0], x.shape[1]),
        device=x.device,
        dtype=torch.float16,
    )
    ops.exl3_mgemm(
        x_3d,
        ptrs_trellis,
        output,
        ptrs_suh,
        x_had,
        ptrs_svh,
        None,
        None,
        k,
        -1,
        kv_mcg,
        kv_mul1,
        -1,
        -1,
        0,
    )
    return torch.cat([q, output[0], output[1]], dim=-1)


@_exl3_qkv.register_fake
def _exl3_qkv_fake(
    x: torch.Tensor,
    q_trellis: torch.Tensor,
    q_suh: torch.Tensor,
    q_svh: torch.Tensor,
    k_trellis: torch.Tensor,
    k_suh: torch.Tensor,
    k_svh: torch.Tensor,
    v_trellis: torch.Tensor,
    v_suh: torch.Tensor,
    v_svh: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    k: int,
    q_mcg: bool,
    q_mul1: bool,
    kv_mcg: bool,
    kv_mul1: bool,
) -> torch.Tensor:
    del q_suh, q_svh, k_suh, k_svh, v_suh, v_svh
    del ptrs_trellis, ptrs_suh, ptrs_svh, k
    del q_mcg, q_mul1, kv_mcg, kv_mul1
    out_features = q_trellis.shape[1] * 16 + k_trellis.shape[1] * 16 + v_trellis.shape[1] * 16
    return torch.empty(
        (x.shape[0], out_features),
        device=x.device,
        dtype=torch.float16,
    )


class Exl3Config(QuantizationConfig):
    """Config class for ExLlamaV3 EXL3 checkpoints.

    This implementation is inference-only and supports EXL3 dense linear
    layers plus a functional per-expert MoE path.
    """

    def __init__(
        self,
        bits: float | None = None,
        head_bits: float | None = None,
        tensor_storage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.tensor_storage = tensor_storage or {}

    def get_name(self) -> QuantizationMethods:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # EXL3 kernels operate on fp16 activations. bf16 models are accepted by
        # casting at the linear boundary and returning the original dtype.
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            tensor_storage=config.get("tensor_storage"),
        )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        del hf_config
        if self.tensor_storage:
            return

        if os.path.isdir(model_name):
            quant_config_path = os.path.join(model_name, "quantization_config.json")
            if not os.path.exists(quant_config_path):
                return
        else:
            try:
                quant_config_path = hf_hub_download(
                    repo_id=model_name,
                    filename="quantization_config.json",
                    revision=revision,
                )
            except Exception:
                logger.debug(
                    "Could not resolve EXL3 quantization_config.json for %s",
                    model_name,
                    exc_info=True,
                )
                return

        with open(quant_config_path) as f:
            config = json.load(f)
        self.tensor_storage = config.get("tensor_storage", {})

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        is_lm_head = layer.__class__.__name__ == "ParallelLMHead"
        if is_lm_head and not prefix:
            prefix = "lm_head"
        if isinstance(layer, LinearBase) or is_lm_head:
            if not self._linear_prefix_is_exl3(prefix):
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self)
        if isinstance(layer, RoutedExperts):
            if not self._moe_prefix_is_exl3(prefix):
                return None
            return Exl3MoEMethod(self, layer.moe_config)
        return None

    def _storage_entry(self, prefix: str) -> dict[str, Any] | None:
        candidates = [prefix]
        if prefix.startswith("model."):
            candidates.append(prefix.removeprefix("model."))
        else:
            candidates.append(f"model.{prefix}")

        parts = prefix.split(".")
        for idx in range(1, len(parts) - 1):
            if parts[idx] != "model":
                continue
            collapsed = ".".join(parts[:idx] + parts[idx + 1 :])
            candidates.append(collapsed)
            if collapsed.startswith("model."):
                candidates.append(collapsed.removeprefix("model."))
            else:
                candidates.append(f"model.{collapsed}")

        for candidate in candidates:
            entry = self.tensor_storage.get(candidate)
            if entry is not None:
                return entry
        return None

    def _is_exl3_prefix(self, prefix: str) -> bool:
        entry = self._storage_entry(prefix)
        return entry is not None and entry.get("quant_format") == "exl3"

    def _linear_prefix_is_exl3(self, prefix: str) -> bool:
        if prefix.endswith("lm_head") and self.head_bits is not None:
            return True

        if self._is_exl3_prefix(prefix):
            return True

        if prefix.endswith(".qkv_proj"):
            base = prefix.removesuffix(".qkv_proj")
            has_q = self._is_exl3_prefix(f"{base}.q_proj")
            has_k = self._is_exl3_prefix(f"{base}.k_proj")
            has_v = self._is_exl3_prefix(f"{base}.v_proj")
            # Gemma 4 full-attention layers can use K=V attention and store
            # only q_proj/k_proj tensors. The model loader duplicates K into
            # V, so the fused qkv_proj still needs EXL3 parameters.
            return has_q and has_k and (has_v or self._storage_entry(f"{base}.v_proj") is None)

        if prefix.endswith(".gate_up_proj"):
            base = prefix.removesuffix(".gate_up_proj")
            return all(self._is_exl3_prefix(f"{base}.{proj}") for proj in ("gate_proj", "up_proj"))

        if prefix.endswith(".in_proj_qkvz"):
            base = prefix.removesuffix(".in_proj_qkvz")
            return all(self._is_exl3_prefix(f"{base}.{proj}") for proj in ("in_proj_qkv", "in_proj_z"))

        return False

    def _moe_prefix_is_exl3(self, prefix: str) -> bool:
        expert_prefixes = (f"{prefix}.0", f"{prefix}.experts.0")
        return any(
            all(self._is_exl3_prefix(f"{expert_prefix}.{proj}") for proj in ("gate_proj", "up_proj", "down_proj"))
            for expert_prefix in expert_prefixes
        )

    def has_moe_tensors(self) -> bool:
        return any(
            ".experts." in prefix or ".mlp.experts" in prefix
            for prefix, entry in self.tensor_storage.items()
            if entry.get("quant_format") == "exl3"
        )

    def has_quantized_lm_head(self) -> bool:
        return self._is_exl3_prefix("lm_head")


class Exl3Parameter(BaseAphroditeParameter):
    """Small placeholder parameter that stores EXL3 tensors loaded from disk.

    Packed layers such as QKV and gate-up load multiple HF tensors into one
    Aphrodite module. The actual tensors can have different shapes, so a single
    dense Parameter cannot represent them. This parameter keeps the tensors in a
    shard dictionary keyed by the loader shard id.
    """

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[str | int | None, torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_id: str | int | None = None,
    ) -> None:
        self.exl3_tensors[shard_id] = loaded_weight.contiguous()


def _exl3_weight_loader(
    param: Exl3Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
) -> None:
    param.load_exl3_weight(loaded_weight, loaded_shard_id)


class Exl3MoEParameter(BaseAphroditeParameter):
    """Placeholder parameter storing EXL3 tensors by expert and shard."""

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[tuple[int, str], torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ) -> None:
        self.exl3_tensors[(expert_id, shard_id)] = loaded_weight.contiguous()


def _exl3_moe_weight_loader(
    param: Exl3MoEParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    del weight_name
    param.load_exl3_weight(loaded_weight, expert_id, shard_id)
    return True if return_success else None


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if not current_platform.is_cuda():
            raise NotImplementedError("EXL3 is only supported on CUDA")

        layer.exl3_tp_rank = get_tensor_model_parallel_rank()
        layer.exl3_tp_size = get_tensor_model_parallel_world_size()
        layer.exl3_input_size = input_size
        layer.input_size_per_partition = input_size_per_partition
        layer.exl3_output_size = output_size
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer.exl3_shard_ids = self._shard_ids_for_layer(layer, output_partition_sizes)
        layer.exl3_parallel_mode = "row" if input_size_per_partition != input_size else "column"

        for name in ("suh", "svh", "trellis", "mcg", "mul1"):
            layer.register_parameter(
                name,
                Exl3Parameter(weight_loader=_exl3_weight_loader),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        missing: list[str] = []
        for attr in ("suh", "svh", "trellis"):
            param = getattr(layer, attr)
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in param.exl3_tensors:
                    missing.append(f"{attr}[{shard_id!r}]")
        if missing:
            prefix = getattr(layer, "prefix", layer.__class__.__name__)
            raise ValueError(f"Missing EXL3 tensors for {prefix}: {', '.join(missing)}")

        for shard_id in layer.exl3_shard_ids:
            if shard_id in layer.mcg.exl3_tensors and shard_id in layer.mul1.exl3_tensors:
                prefix = getattr(layer, "prefix", layer.__class__.__name__)
                raise ValueError(f"EXL3 tensor {prefix}[{shard_id!r}] specifies both mcg and mul1")

        self._shard_tensors_for_tensor_parallel(layer)

        device = torch.device("cuda", torch.cuda.current_device())
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            param = getattr(layer, attr)
            for shard_id, tensor in list(param.exl3_tensors.items()):
                param.exl3_tensors[shard_id] = tensor.to(device=device, non_blocking=True)

        self._setup_mgemm_if_supported(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        else:
            x_2d = x_2d.contiguous()

        if getattr(layer, "exl3_can_mgemm", False):
            output = self._apply_fused_mgemm(layer, x_2d)
        else:
            outputs = [self._apply_one(layer, x_2d, shard_id) for shard_id in layer.exl3_shard_ids]
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output = output + bias
        output = output.reshape(*original_shape, output.shape[-1])
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    @staticmethod
    def _slice_exl3_tensor(
        tensor: torch.Tensor,
        *,
        dim: int,
        start: int,
        size: int,
    ) -> torch.Tensor:
        if size % 16 != 0 or start % 16 != 0:
            prefix = "output" if dim == 1 else "input"
            raise ValueError(f"EXL3 TP {prefix} shard must be 16-aligned, got start={start}, size={size}.")
        return tensor.narrow(dim, start // 16, size // 16).contiguous()

    @staticmethod
    def _output_shard_size(
        layer: torch.nn.Module,
        shard_id: str | int | tuple[int, ...] | None,
    ) -> int:
        if shard_id is None:
            shard_idx = 0
            return layer.exl3_output_partition_sizes[shard_idx]

        if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
            shard_idx = {"q": 0, "k": 1, "v": 2}[shard_id]
            return layer.exl3_output_partition_sizes[shard_idx]

        if isinstance(shard_id, tuple):
            return sum(layer.exl3_output_partition_sizes[idx] for idx in shard_id)

        if isinstance(shard_id, int):
            return layer.exl3_output_partition_sizes[shard_id]

        shard_idx = layer.exl3_shard_ids.index(shard_id)
        return layer.exl3_output_partition_sizes[shard_idx]

    @staticmethod
    def _qkv_output_start(
        layer: torch.nn.Module,
        shard_id: str | int | tuple[int, ...] | None,
        shard_size: int,
    ) -> int:
        if shard_id == "q":
            shard_rank = layer.exl3_tp_rank
        elif shard_id in ("k", "v"):
            shard_rank = layer.exl3_tp_rank // layer.num_kv_head_replicas
        else:
            shard_rank = layer.exl3_tp_rank
        return shard_rank * shard_size

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: torch.nn.Module) -> None:
        if layer.exl3_tp_size == 1:
            return

        if layer.exl3_parallel_mode == "row":
            start = layer.exl3_tp_rank * layer.input_size_per_partition
            size = layer.input_size_per_partition
            for shard_id in layer.exl3_shard_ids:
                layer.suh.exl3_tensors[shard_id] = layer.suh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
                layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[shard_id],
                    dim=0,
                    start=start,
                    size=size,
                )
            return

        already_sharded = cls._expand_tuple_output_shards_for_tensor_parallel(layer)

        for shard_id in layer.exl3_shard_ids:
            if shard_id in already_sharded:
                continue
            shard_size = cls._output_shard_size(layer, shard_id)
            start = cls._qkv_output_start(layer, shard_id, shard_size)
            layer.svh.exl3_tensors[shard_id] = (
                layer.svh.exl3_tensors[shard_id].narrow(0, start, shard_size).contiguous()
            )
            layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                layer.trellis.exl3_tensors[shard_id],
                dim=1,
                start=start,
                size=shard_size,
            )

    @classmethod
    def _expand_tuple_output_shards_for_tensor_parallel(
        cls,
        layer: torch.nn.Module,
    ) -> set[int]:
        tuple_shard_ids = [shard_id for shard_id in layer.exl3_shard_ids if isinstance(shard_id, tuple)]
        if not tuple_shard_ids:
            return set()

        expanded_shard_ids: list[str | int | None] = []
        expanded_component_ids: set[int] = set()
        for shard_id in layer.exl3_shard_ids:
            if isinstance(shard_id, tuple):
                expanded_shard_ids.extend(shard_id)
                expanded_component_ids.update(shard_id)
            else:
                expanded_shard_ids.append(shard_id)

        for tuple_shard_id in tuple_shard_ids:
            component_full_offsets: dict[int, int] = {}
            offset = 0
            for idx in tuple_shard_id:
                component_full_offsets[idx] = offset
                offset += layer.exl3_output_partition_sizes[idx] * layer.exl3_tp_size

            for idx in tuple_shard_id:
                local_size = layer.exl3_output_partition_sizes[idx]
                start = component_full_offsets[idx] + layer.exl3_tp_rank * local_size
                layer.suh.exl3_tensors[idx] = layer.suh.exl3_tensors[tuple_shard_id]
                layer.svh.exl3_tensors[idx] = (
                    layer.svh.exl3_tensors[tuple_shard_id].narrow(0, start, local_size).contiguous()
                )
                layer.trellis.exl3_tensors[idx] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[tuple_shard_id],
                    dim=1,
                    start=start,
                    size=local_size,
                )
                if tuple_shard_id in layer.mcg.exl3_tensors:
                    layer.mcg.exl3_tensors[idx] = layer.mcg.exl3_tensors[tuple_shard_id]
                if tuple_shard_id in layer.mul1.exl3_tensors:
                    layer.mul1.exl3_tensors[idx] = layer.mul1.exl3_tensors[tuple_shard_id]

            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                getattr(layer, attr).exl3_tensors.pop(tuple_shard_id, None)

        layer.exl3_shard_ids = expanded_shard_ids
        return expanded_component_ids

    def _apply_one(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        shard_id: str | int | None,
    ) -> torch.Tensor:
        suh = layer.suh.exl3_tensors[shard_id]
        svh = layer.svh.exl3_tensors[shard_id]
        trellis = layer.trellis.exl3_tensors[shard_id]
        mcg = shard_id in layer.mcg.exl3_tensors
        mul1 = shard_id in layer.mul1.exl3_tensors

        return _exl3_linear_one(x, trellis, suh, svh, mcg, mul1)

    def _apply_fused_mgemm(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(layer, "exl3_mgemm_mode", None) == "qkv_kv":
            return _exl3_qkv(
                x,
                layer.trellis.exl3_tensors["q"],
                layer.suh.exl3_tensors["q"],
                layer.svh.exl3_tensors["q"],
                layer.trellis.exl3_tensors["k"],
                layer.suh.exl3_tensors["k"],
                layer.svh.exl3_tensors["k"],
                layer.trellis.exl3_tensors["v"],
                layer.suh.exl3_tensors["v"],
                layer.svh.exl3_tensors["v"],
                layer.exl3_mgemm_ptrs_trellis,
                layer.exl3_mgemm_ptrs_suh,
                layer.exl3_mgemm_ptrs_svh,
                layer.exl3_mgemm_k,
                "q" in layer.mcg.exl3_tensors,
                "q" in layer.mul1.exl3_tensors,
                layer.exl3_mgemm_mcg,
                layer.exl3_mgemm_mul1,
            )

        if getattr(layer, "exl3_mgemm_mode", None) == "gate_up":
            gate_id, up_id = layer.exl3_shard_ids
            return _exl3_gate_up(
                x,
                layer.trellis.exl3_tensors[gate_id],
                layer.suh.exl3_tensors[gate_id],
                layer.svh.exl3_tensors[gate_id],
                layer.trellis.exl3_tensors[up_id],
                layer.suh.exl3_tensors[up_id],
                layer.svh.exl3_tensors[up_id],
                layer.exl3_mgemm_ptrs_trellis,
                layer.exl3_mgemm_ptrs_suh,
                layer.exl3_mgemm_ptrs_svh,
                layer.exl3_mgemm_k,
                layer.exl3_mgemm_mcg,
                layer.exl3_mgemm_mul1,
            )

        output = self._apply_mgemm(layer, x)
        return torch.cat([output[i] for i in range(output.shape[0])], dim=-1)

    def _apply_mgemm(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x_3d = x.view(1, x.shape[0], x.shape[1])
        output = torch.empty(
            (
                layer.exl3_mgemm_num_shards,
                x.shape[0],
                layer.exl3_mgemm_out_features,
            ),
            device=x.device,
            dtype=torch.float16,
        )
        x_had = torch.empty(
            (layer.exl3_mgemm_num_shards, x.shape[0], x.shape[1]),
            device=x.device,
            dtype=torch.float16,
        )
        ops.exl3_mgemm(
            x_3d,
            layer.exl3_mgemm_ptrs_trellis,
            output,
            layer.exl3_mgemm_ptrs_suh,
            x_had,
            layer.exl3_mgemm_ptrs_svh,
            None,
            None,
            layer.exl3_mgemm_k,
            -1,
            layer.exl3_mgemm_mcg,
            layer.exl3_mgemm_mul1,
            -1,
            -1,
            0,
        )
        return output

    @staticmethod
    def _setup_mgemm_if_supported(layer: torch.nn.Module) -> None:
        layer.exl3_can_mgemm = False

        prefix = getattr(layer, "prefix", "")
        if prefix.endswith("gate_up_proj") and len(layer.exl3_shard_ids) == 2:
            mgemm_shard_ids = layer.exl3_shard_ids
            layer.exl3_mgemm_mode = "gate_up"
        elif prefix.endswith("qkv_proj") and layer.exl3_shard_ids == ["q", "k", "v"]:
            mgemm_shard_ids = ["k", "v"]
            layer.exl3_mgemm_mode = "qkv_kv"
        else:
            return

        trellises = [layer.trellis.exl3_tensors[shard_id] for shard_id in mgemm_shard_ids]
        suhs = [layer.suh.exl3_tensors[shard_id] for shard_id in mgemm_shard_ids]
        svhs = [layer.svh.exl3_tensors[shard_id] for shard_id in mgemm_shard_ids]

        first_trellis = trellises[0]
        first_suh = suhs[0]
        first_svh = svhs[0]
        mcg = mgemm_shard_ids[0] in layer.mcg.exl3_tensors
        mul1 = mgemm_shard_ids[0] in layer.mul1.exl3_tensors
        if any(tensor.shape != first_trellis.shape for tensor in trellises[1:]):
            return
        if any(tensor.shape != first_suh.shape for tensor in suhs[1:]):
            return
        if any(tensor.shape != first_svh.shape for tensor in svhs[1:]):
            return
        if any((shard_id in layer.mcg.exl3_tensors) != mcg for shard_id in mgemm_shard_ids[1:]):
            return
        if any((shard_id in layer.mul1.exl3_tensors) != mul1 for shard_id in mgemm_shard_ids[1:]):
            return

        device = first_trellis.device
        layer.exl3_mgemm_ptrs_trellis = torch.tensor(
            [tensor.data_ptr() for tensor in trellises],
            dtype=torch.long,
            device=device,
        )
        layer.exl3_mgemm_ptrs_suh = torch.tensor(
            [tensor.data_ptr() for tensor in suhs],
            dtype=torch.long,
            device=device,
        )
        layer.exl3_mgemm_ptrs_svh = torch.tensor(
            [tensor.data_ptr() for tensor in svhs],
            dtype=torch.long,
            device=device,
        )
        layer.exl3_mgemm_k = first_trellis.shape[2] // 16
        layer.exl3_mgemm_out_features = first_trellis.shape[1] * 16
        layer.exl3_mgemm_num_shards = len(mgemm_shard_ids)
        layer.exl3_mgemm_mcg = mcg
        layer.exl3_mgemm_mul1 = mul1
        layer.exl3_can_mgemm = True

    @staticmethod
    def _shard_ids_for_layer(
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
    ) -> list[str | int | tuple[int, ...] | None]:
        if len(output_partition_sizes) == 1:
            return [None]

        prefix = getattr(layer, "prefix", "")
        if prefix.endswith("qkv_proj"):
            return ["q", "k", "v"]
        if prefix.endswith("gate_up_proj"):
            return [0, 1]
        if prefix.endswith("in_proj_qkvz"):
            return [(0, 1, 2), 3]

        return list(range(len(output_partition_sizes)))


class Exl3MoEMethod(FusedMoEMethodBase):
    """Functional EXL3 MoE method using per-expert EXL3 GEMM kernels."""

    def __init__(self, quant_config: Exl3Config, moe) -> None:
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del num_experts, params_dtype
        if not current_platform.is_cuda():
            raise NotImplementedError("EXL3 MoE is only supported on CUDA")
        if self.moe.moe_parallel_config.use_ep:
            raise NotImplementedError("EXL3 MoE currently does not support expert parallelism")

        layer.quant_config = self.quant_config
        layer.exl3_tp_rank = get_tensor_model_parallel_rank()
        layer.exl3_tp_size = get_tensor_model_parallel_world_size()
        layer.exl3_hidden_size = hidden_size
        layer.exl3_intermediate_size_per_partition = intermediate_size_per_partition
        extra_weight_attrs = dict(extra_weight_attrs)
        extra_weight_attrs.pop("weight_loader", None)

        for prefix in ("w13", "w2"):
            for suffix in ("suh", "svh", "trellis", "mcg", "mul1"):
                param = Exl3MoEParameter(weight_loader=_exl3_moe_weight_loader)
                layer.register_parameter(f"{prefix}_{suffix}", param)
                set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        missing: list[str] = []
        required = {
            "w13": ("w1", "w3"),
            "w2": ("w2",),
        }
        for prefix, shard_ids in required.items():
            for attr in ("suh", "svh", "trellis"):
                param = getattr(layer, f"{prefix}_{attr}")
                for expert_id in range(layer.local_num_experts):
                    for shard_id in shard_ids:
                        key = (expert_id, shard_id)
                        if key not in param.exl3_tensors:
                            missing.append(f"{prefix}_{attr}[{expert_id},{shard_id}]")
        if missing:
            raise ValueError(
                f"Missing EXL3 MoE tensors for {layer.layer_name}: "
                + ", ".join(missing[:32])
                + (" ..." if len(missing) > 32 else "")
            )

        self._shard_tensors_for_tensor_parallel(layer)

        device = torch.device("cuda", torch.cuda.current_device())
        for prefix in ("w13", "w2"):
            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                param = getattr(layer, f"{prefix}_{attr}")
                for key, tensor in list(param.exl3_tensors.items()):
                    param.exl3_tensors[key] = tensor.to(device=device, non_blocking=True)

        self._setup_fused_moe_kernel(layer, device)

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: RoutedExperts) -> None:
        if layer.exl3_tp_size == 1:
            return

        start = layer.exl3_tp_rank * layer.exl3_intermediate_size_per_partition
        size = layer.exl3_intermediate_size_per_partition
        for expert_id in range(layer.local_num_experts):
            for shard_id in ("w1", "w3"):
                key = (expert_id, shard_id)
                layer.w13_svh.exl3_tensors[key] = layer.w13_svh.exl3_tensors[key].narrow(0, start, size).contiguous()
                layer.w13_trellis.exl3_tensors[key] = Exl3LinearMethod._slice_exl3_tensor(
                    layer.w13_trellis.exl3_tensors[key],
                    dim=1,
                    start=start,
                    size=size,
                )

            key = (expert_id, "w2")
            layer.w2_suh.exl3_tensors[key] = layer.w2_suh.exl3_tensors[key].narrow(0, start, size).contiguous()
            layer.w2_trellis.exl3_tensors[key] = Exl3LinearMethod._slice_exl3_tensor(
                layer.w2_trellis.exl3_tensors[key],
                dim=0,
                start=start,
                size=size,
            )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        del layer
        return None

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.long

    @property
    def supports_shared_expert_overlap(self) -> bool:
        return False

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.activation != MoEActivation.SILU:
            raise NotImplementedError(f"EXL3 MoE only supports SiLU, got {layer.activation}.")
        if layer.expert_map is not None:
            raise NotImplementedError("EXL3 MoE does not support expert maps yet.")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError("EXL3 fused MoE does not support applying router weights on input.")

        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        else:
            x_2d = x_2d.contiguous()

        topk_ids = topk_ids.to(torch.long)
        topk_weights = topk_weights.to(torch.float16)
        total_assignments = x_2d.shape[0] * topk_ids.shape[-1]

        if x_2d.shape[0] == 1:
            return self._apply_single_token(
                layer,
                x_2d,
                topk_ids,
                topk_weights,
                original_dtype,
                x.shape[:-1],
            )

        if x_2d.shape[0] <= layer.exl3_small_batch_threshold:
            return self._apply_small_batch(
                layer,
                x_2d,
                topk_ids,
                topk_weights,
                original_dtype,
                x.shape[:-1],
            )

        output = torch.zeros(
            (x_2d.shape[0], layer.hidden_size),
            device=x_2d.device,
            dtype=torch.float32,
        )
        flat_expert = topk_ids.reshape(-1)
        # Route ids outside [0, local_num_experts) — e.g. padding tokens in
        # the startup memory-profiling dummy batch — into the sentinel bucket
        # at index local_num_experts instead of scatter_add_ing them out of
        # bounds below. Mirrors exllamav3's block_sparse_mlp.py reference
        # (flat_expert_local clamp via torch.where), which this fast/prefill
        # path lost when the kernels were last synced from upstream.
        _valid_expert = (flat_expert >= 0) & (flat_expert < layer.local_num_experts)
        flat_expert = torch.where(_valid_expert, flat_expert, torch.full_like(flat_expert, layer.local_num_experts))
        flat_weight = topk_weights.reshape(-1)
        flat_token = torch.arange(x_2d.shape[0], device=x_2d.device)
        flat_token = flat_token.repeat_interleave(topk_ids.shape[-1])
        order = flat_expert.argsort()
        expert_sorted = flat_expert[order]
        token_sorted = flat_token[order]
        weight_sorted = flat_weight[order]
        expert_count = torch.zeros(
            layer.local_num_experts + 1,
            device=x_2d.device,
            dtype=torch.long,
        )
        expert_count.scatter_add_(
            0,
            expert_sorted,
            torch.ones_like(expert_sorted, dtype=torch.long),
        )

        if not hasattr(torch.ops._C, "exl3_moe"):
            raise RuntimeError("EXL3 MoE kernel is not available. Rebuild Aphrodite.")

        ops.exl3_moe(
            x_2d,
            output,
            expert_count,
            token_sorted,
            weight_sorted,
            layer.exl3_temp_state_g,
            layer.exl3_temp_state_u,
            layer.exl3_temp_intermediate_g,
            layer.exl3_temp_intermediate_u,
            _EXL3_MOE_ACT_SILU,
            layer.exl3_moe_k_gate,
            layer.exl3_moe_k_up,
            layer.exl3_moe_k_down,
            layer.exl3_gate_ptrs_trellis,
            layer.exl3_gate_ptrs_suh,
            layer.exl3_gate_ptrs_svh,
            layer.exl3_up_ptrs_trellis,
            layer.exl3_up_ptrs_suh,
            layer.exl3_up_ptrs_svh,
            layer.exl3_down_ptrs_trellis,
            layer.exl3_down_ptrs_suh,
            layer.exl3_down_ptrs_svh,
            layer.exl3_gate_mcg,
            layer.exl3_gate_mul1,
            layer.exl3_up_mcg,
            layer.exl3_up_mul1,
            layer.exl3_down_mcg,
            layer.exl3_down_mul1,
            0.0,
        )

        if torch.cuda.is_current_stream_capturing():
            if output.dtype != original_dtype:
                output = output.to(original_dtype)
            return output.reshape(*x.shape[:-1], output.shape[-1])

        if total_assignments <= _EXL3_MOE_MAX_TOKENS_PER_EXPERT:
            if output.dtype != original_dtype:
                output = output.to(original_dtype)
            return output.reshape(*x.shape[:-1], output.shape[-1])

        needs_fallback = bool((expert_count[:-1] > _EXL3_MOE_MAX_TOKENS_PER_EXPERT).any().item())
        if not needs_fallback:
            if output.dtype != original_dtype:
                output = output.to(original_dtype)
            return output.reshape(*x.shape[:-1], output.shape[-1])

        expert_offsets = torch.empty(
            layer.local_num_experts + 2,
            device=x_2d.device,
            dtype=torch.long,
        )
        expert_offsets[0] = 0
        expert_offsets[1:] = expert_count.cumsum(0)
        expert_offsets_list = expert_offsets.cpu().tolist()

        for expert_id in range(layer.local_num_experts):
            start = expert_offsets_list[expert_id]
            end = expert_offsets_list[expert_id + 1]
            count = end - start
            if count <= _EXL3_MOE_MAX_TOKENS_PER_EXPERT:
                continue

            token_pos = token_sorted[start:end]
            route_weight = weight_sorted[start:end].unsqueeze(-1)
            expert_input = x_2d.index_select(0, token_pos)

            gate = self._apply_exl3(layer, "w13", expert_input, expert_id, "w1")
            up = self._apply_exl3(layer, "w13", expert_input, expert_id, "w3")
            intermediate = torch.nn.functional.silu(gate) * up
            expert_output = self._apply_exl3(layer, "w2", intermediate, expert_id, "w2")
            expert_output = expert_output * route_weight
            output.index_add_(0, token_pos, expert_output.to(torch.float32))

        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output.reshape(*x.shape[:-1], output.shape[-1])

    def _apply_single_token(
        self,
        layer: RoutedExperts,
        x_2d: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        original_dtype: torch.dtype,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        x_3d = x_2d.unsqueeze(0)

        if layer.exl3_fuse_gate_up:
            ops.make_gate_up_indices(
                layer.exl3_small_gate_up_ids,
                topk_ids,
                layer.local_num_experts,
            )
            ops.exl3_mgemm(
                x_3d,
                layer.exl3_gate_up_ptrs_trellis,
                layer.exl3_small_interm_gu,
                layer.exl3_gate_up_ptrs_suh,
                layer.exl3_small_yh_gu,
                layer.exl3_gate_up_ptrs_svh,
                layer.exl3_small_gate_up_ids,
                None,
                layer.exl3_moe_k_gate,
                -1,
                layer.exl3_gate_mcg,
                layer.exl3_gate_mul1,
                -1,
                -1,
                0,
            )
        else:
            ops.exl3_mgemm(
                x_3d,
                layer.exl3_gate_ptrs_trellis,
                layer.exl3_small_interm_g,
                layer.exl3_gate_ptrs_suh,
                layer.exl3_small_yh,
                layer.exl3_gate_ptrs_svh,
                topk_ids,
                None,
                layer.exl3_moe_k_gate,
                -1,
                layer.exl3_gate_mcg,
                layer.exl3_gate_mul1,
                -1,
                -1,
                0,
            )
            ops.exl3_mgemm(
                x_3d,
                layer.exl3_up_ptrs_trellis,
                layer.exl3_small_interm_u,
                layer.exl3_up_ptrs_suh,
                layer.exl3_small_yh,
                layer.exl3_up_ptrs_svh,
                topk_ids,
                None,
                layer.exl3_moe_k_up,
                -1,
                layer.exl3_up_mcg,
                layer.exl3_up_mul1,
                -1,
                -1,
                0,
            )
        ops.silu_mul(
            layer.exl3_small_interm_a,
            layer.exl3_small_interm_g,
            layer.exl3_small_interm_u,
        )
        ops.exl3_mgemm(
            layer.exl3_small_interm_a,
            layer.exl3_down_ptrs_trellis,
            layer.exl3_small_out_d,
            layer.exl3_down_ptrs_suh,
            layer.exl3_small_interm_a,
            layer.exl3_down_ptrs_svh,
            topk_ids,
            topk_weights,
            layer.exl3_moe_k_down,
            layer.exl3_down_force_shape,
            layer.exl3_down_mcg,
            layer.exl3_down_mul1,
            -1,
            -1,
            layer.exl3_down_force_num_sms,
        )
        output = layer.exl3_small_out_d[:1].reshape(*original_shape, layer.hidden_size)
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    def _apply_small_batch(
        self,
        layer: RoutedExperts,
        x_2d: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        original_dtype: torch.dtype,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        output = layer.exl3_small_batch_out[: x_2d.shape[0]]
        x_3d = x_2d.unsqueeze(1).unsqueeze(1)
        topk_ids_3d = topk_ids.unsqueeze(1)
        topk_weights_3d = topk_weights.unsqueeze(1)

        for i in range(x_2d.shape[0]):
            if layer.exl3_fuse_gate_up:
                ops.make_gate_up_indices(
                    layer.exl3_small_gate_up_ids,
                    topk_ids_3d[i],
                    layer.local_num_experts,
                )
                ops.exl3_mgemm(
                    x_3d[i],
                    layer.exl3_gate_up_ptrs_trellis,
                    layer.exl3_small_interm_gu,
                    layer.exl3_gate_up_ptrs_suh,
                    layer.exl3_small_yh_gu,
                    layer.exl3_gate_up_ptrs_svh,
                    layer.exl3_small_gate_up_ids,
                    None,
                    layer.exl3_moe_k_gate,
                    -1,
                    layer.exl3_gate_mcg,
                    layer.exl3_gate_mul1,
                    -1,
                    -1,
                    0,
                )
            else:
                ops.exl3_mgemm(
                    x_3d[i],
                    layer.exl3_gate_ptrs_trellis,
                    layer.exl3_small_interm_g,
                    layer.exl3_gate_ptrs_suh,
                    layer.exl3_small_yh,
                    layer.exl3_gate_ptrs_svh,
                    topk_ids_3d[i],
                    None,
                    layer.exl3_moe_k_gate,
                    -1,
                    layer.exl3_gate_mcg,
                    layer.exl3_gate_mul1,
                    -1,
                    -1,
                    0,
                )
                ops.exl3_mgemm(
                    x_3d[i],
                    layer.exl3_up_ptrs_trellis,
                    layer.exl3_small_interm_u,
                    layer.exl3_up_ptrs_suh,
                    layer.exl3_small_yh,
                    layer.exl3_up_ptrs_svh,
                    topk_ids_3d[i],
                    None,
                    layer.exl3_moe_k_up,
                    -1,
                    layer.exl3_up_mcg,
                    layer.exl3_up_mul1,
                    -1,
                    -1,
                    0,
                )
            ops.silu_mul(
                layer.exl3_small_interm_a,
                layer.exl3_small_interm_g,
                layer.exl3_small_interm_u,
            )
            ops.exl3_mgemm(
                layer.exl3_small_interm_a,
                layer.exl3_down_ptrs_trellis,
                layer.exl3_small_out_d,
                layer.exl3_down_ptrs_suh,
                layer.exl3_small_interm_a,
                layer.exl3_down_ptrs_svh,
                topk_ids_3d[i],
                topk_weights_3d[i],
                layer.exl3_moe_k_down,
                layer.exl3_down_force_shape,
                layer.exl3_down_mcg,
                layer.exl3_down_mul1,
                -1,
                -1,
                layer.exl3_down_force_num_sms,
            )
            output[i : i + 1] = layer.exl3_small_out_d[0]

        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output.reshape(*original_shape, output.shape[-1])

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("EXL3 MoE uses external routing.")

    @staticmethod
    def _setup_fused_moe_kernel(layer: RoutedExperts, device: torch.device) -> None:
        def tensor(prefix: str, attr: str, expert_id: int, shard_id: str):
            return getattr(layer, f"{prefix}_{attr}").exl3_tensors[(expert_id, shard_id)]

        def ptr_tensor(prefix: str, attr: str, shard_id: str):
            return torch.tensor(
                [tensor(prefix, attr, expert_id, shard_id).data_ptr() for expert_id in range(layer.local_num_experts)],
                dtype=torch.long,
                device=device,
            )

        layer.exl3_gate_ptrs_trellis = ptr_tensor("w13", "trellis", "w1")
        layer.exl3_gate_ptrs_suh = ptr_tensor("w13", "suh", "w1")
        layer.exl3_gate_ptrs_svh = ptr_tensor("w13", "svh", "w1")
        layer.exl3_up_ptrs_trellis = ptr_tensor("w13", "trellis", "w3")
        layer.exl3_up_ptrs_suh = ptr_tensor("w13", "suh", "w3")
        layer.exl3_up_ptrs_svh = ptr_tensor("w13", "svh", "w3")
        layer.exl3_gate_up_ptrs_trellis = torch.cat([layer.exl3_gate_ptrs_trellis, layer.exl3_up_ptrs_trellis])
        layer.exl3_gate_up_ptrs_suh = torch.cat([layer.exl3_gate_ptrs_suh, layer.exl3_up_ptrs_suh])
        layer.exl3_gate_up_ptrs_svh = torch.cat([layer.exl3_gate_ptrs_svh, layer.exl3_up_ptrs_svh])
        layer.exl3_down_ptrs_trellis = ptr_tensor("w2", "trellis", "w2")
        layer.exl3_down_ptrs_suh = ptr_tensor("w2", "suh", "w2")
        layer.exl3_down_ptrs_svh = ptr_tensor("w2", "svh", "w2")

        gate_trellis = tensor("w13", "trellis", 0, "w1")
        up_trellis = tensor("w13", "trellis", 0, "w3")
        down_trellis = tensor("w2", "trellis", 0, "w2")
        intermediate_size = gate_trellis.shape[1] * 16

        layer.exl3_moe_k_gate = gate_trellis.shape[2] // 16
        layer.exl3_moe_k_up = up_trellis.shape[2] // 16
        layer.exl3_moe_k_down = down_trellis.shape[2] // 16

        layer.exl3_gate_mcg = (0, "w1") in layer.w13_mcg.exl3_tensors
        layer.exl3_gate_mul1 = (0, "w1") in layer.w13_mul1.exl3_tensors
        layer.exl3_up_mcg = (0, "w3") in layer.w13_mcg.exl3_tensors
        layer.exl3_up_mul1 = (0, "w3") in layer.w13_mul1.exl3_tensors
        layer.exl3_down_mcg = (0, "w2") in layer.w2_mcg.exl3_tensors
        layer.exl3_down_mul1 = (0, "w2") in layer.w2_mul1.exl3_tensors
        layer.exl3_down_force_shape, layer.exl3_down_force_num_sms = _get_exl3_moe_down_tuning(
            device=device,
            k_down=layer.exl3_moe_k_down,
            intermediate_size=intermediate_size,
            hidden_size=layer.hidden_size,
        )
        layer.exl3_fuse_gate_up = (
            layer.exl3_moe_k_gate == layer.exl3_moe_k_up
            and layer.exl3_gate_mcg == layer.exl3_up_mcg
            and layer.exl3_gate_mul1 == layer.exl3_up_mul1
        )

        layer.exl3_small_batch_threshold = min(
            layer.local_num_experts // layer.top_k,
            _EXL3_MOE_MAX_EXPERTS_PER_TOKEN,
        )
        layer.exl3_small_yh_gu = torch.empty(
            (layer.top_k * 2, 1, layer.hidden_size), dtype=torch.float16, device=device
        )
        layer.exl3_small_interm_gu = torch.empty(
            (layer.top_k * 2, 1, intermediate_size), dtype=torch.float16, device=device
        )
        layer.exl3_small_yh = layer.exl3_small_yh_gu[: layer.top_k]
        layer.exl3_small_interm_g = layer.exl3_small_interm_gu[: layer.top_k]
        layer.exl3_small_interm_u = layer.exl3_small_interm_gu[layer.top_k :]
        layer.exl3_small_gate_up_ids = torch.empty((1, layer.top_k * 2), dtype=torch.long, device=device)
        layer.exl3_small_interm_a = torch.empty((layer.top_k, 1, intermediate_size), dtype=torch.float16, device=device)
        layer.exl3_small_out_d = torch.empty((layer.top_k, 1, layer.hidden_size), dtype=torch.float32, device=device)
        layer.exl3_small_batch_out = torch.empty(
            (layer.exl3_small_batch_threshold, layer.hidden_size),
            dtype=torch.float32,
            device=device,
        )

        concurrency = max(
            1,
            torch.cuda.get_device_properties(device).multi_processor_count // 12,
        )
        temp_shape_hidden = (
            concurrency,
            _EXL3_MOE_MAX_TOKENS_PER_EXPERT,
            layer.hidden_size,
        )
        temp_shape_intermediate = (
            concurrency,
            _EXL3_MOE_MAX_TOKENS_PER_EXPERT,
            intermediate_size,
        )
        layer.exl3_temp_state_g = torch.empty(temp_shape_hidden, dtype=torch.float16, device=device)
        layer.exl3_temp_state_u = torch.empty(temp_shape_hidden, dtype=torch.float16, device=device)
        layer.exl3_temp_intermediate_g = torch.empty(temp_shape_intermediate, dtype=torch.float16, device=device)
        layer.exl3_temp_intermediate_u = torch.empty(temp_shape_intermediate, dtype=torch.float16, device=device)

    @staticmethod
    def _apply_exl3(
        layer: RoutedExperts,
        prefix: str,
        x: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ) -> torch.Tensor:
        key = (expert_id, shard_id)
        suh = getattr(layer, f"{prefix}_suh").exl3_tensors[key]
        svh = getattr(layer, f"{prefix}_svh").exl3_tensors[key]
        trellis = getattr(layer, f"{prefix}_trellis").exl3_tensors[key]
        mcg = key in getattr(layer, f"{prefix}_mcg").exl3_tensors
        mul1 = key in getattr(layer, f"{prefix}_mul1").exl3_tensors

        output = torch.empty(
            (x.shape[0], trellis.shape[1] * 16),
            device=x.device,
            dtype=torch.float16,
        )
        x_had = torch.empty_like(x)

        if x.shape[0] <= 32:
            ops.exl3_gemm(
                x,
                trellis,
                output,
                suh,
                x_had,
                svh,
                -1,
                mcg,
                mul1,
                0,
            )
            return output

        weight = torch.empty(
            (trellis.shape[0] * 16, trellis.shape[1] * 16),
            device=trellis.device,
            dtype=torch.float16,
        )
        ops.exl3_reconstruct(
            weight,
            trellis,
            # EXL3 reconstruct expects K where packed.shape[2] == 16 * K.
            trellis.shape[2] // 16,
            mcg,
            mul1,
        )
        ops.exl3_had_r_128(x, x_had, suh, None, 1.0)
        ops.exl3_hgemm(x_had, weight, output)
        ops.exl3_had_r_128(output, output, None, svh, 1.0)
        return output
