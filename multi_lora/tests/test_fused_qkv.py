"""QKV 融合:合并 GEMM 的数值、权重共享、LoRA delta 三条路径."""

import torch
import torch.nn as nn

from multi_lora.backend.torch_backend import TorchLoRABackend
from multi_lora.batch_info import KernelInputMetaData
from multi_lora.layers import LinearWithLoRA, fuse_qkv_projections

PREFIX = "model.layers.0.self_attn."
OUT_DIMS = {"q_proj": 32, "k_proj": 8, "v_proj": 8}


def make_wrappers(in_dim=16, seed=0):
    """造一组 q/k/v,权重随机但确定性."""
    torch.manual_seed(seed)
    wrappers = {}
    for name, out_dim in OUT_DIMS.items():
        layer = nn.Linear(in_dim, out_dim, bias=False)
        wrappers[PREFIX + name] = LinearWithLoRA(layer)
    return wrappers


def attach_lora(wrappers, in_dim=16, rank=4, seed=1):
    """给三层挂同一个 Torch 后端和各自的 A/B 栈."""
    torch.manual_seed(seed)
    backend = TorchLoRABackend()
    backend.set_kernel_input_metadata(
        KernelInputMetaData(
            bs=2,
            seg_lens=torch.tensor([1, 1]),
            seg_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
            weight_indices=torch.tensor([0, 1]),
            lora_ranks=torch.tensor([rank, rank]),
            scalings=torch.tensor([1.0, 1.0]),
            max_len=1,
        )
    )
    for name, out_dim in OUT_DIMS.items():
        a_buf = torch.randn(2, rank, in_dim)
        b_buf = torch.randn(2, out_dim, rank) * 0.1
        wrappers[PREFIX + name].set_lora_info(backend, a_buf, b_buf)
    return wrappers


def test_fused_matches_separate():
    """decode 形状:融合后三层输出与独立 GEMM 逐位一致."""
    wrappers = make_wrappers()
    x = torch.randn(4, 1, 16)
    order = [PREFIX + n for n in OUT_DIMS]
    ref = [wrappers[k](x) for k in order]

    assert fuse_qkv_projections(wrappers) == 1

    got = [wrappers[k](x) for k in order]
    for name, g, r in zip(OUT_DIMS, got, ref):
        assert g.shape == r.shape, name
        assert torch.equal(g, r), name


def test_prefill_path_not_fused():
    """S>1 不合并,走三次独立 GEMM,结果仍一致."""
    wrappers = make_wrappers()
    x = torch.randn(2, 7, 16)
    order = [PREFIX + n for n in OUT_DIMS]
    ref = [wrappers[k](x) for k in order]

    fuse_qkv_projections(wrappers)

    got = [wrappers[k](x) for k in order]
    for g, r in zip(got, ref):
        assert torch.equal(g, r)


def test_weights_share_storage():
    """三层 weight 指向同一块 storage,且数值不变(不额外占显存)."""
    wrappers = make_wrappers()
    order = [PREFIX + n for n in OUT_DIMS]
    before = [wrappers[k].base_layer.weight.detach().clone() for k in order]

    fuse_qkv_projections(wrappers)

    weights = [wrappers[k].base_layer.weight for k in order]
    storages = {w.untyped_storage().data_ptr() for w in weights}
    assert len(storages) == 1
    for w, b in zip(weights, before):
        assert torch.equal(w.detach(), b)


def test_lora_delta_fused_matches_separate():
    """带 LoRA delta:融合后 = 合并基座 + 各自的 delta."""
    wrappers = attach_lora(make_wrappers())
    x = torch.randn(2, 1, 16)
    order = [PREFIX + n for n in OUT_DIMS]
    ref = [wrappers[k](x) for k in order]

    fuse_qkv_projections(wrappers)

    got = [wrappers[k](x) for k in order]
    tolerance = 1e-6
    for name, g, r in zip(OUT_DIMS, got, ref):
        max_diff = (g - r).abs().max().item()
        assert max_diff < tolerance, (name, max_diff)
