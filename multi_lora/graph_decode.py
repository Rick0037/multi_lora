"""把“一步真实 decode”录成 CUDA Graph 所需的常驻张量与录制入口.

decode 能被录下来的前提是:每一步碰到的张量,地址和形状都不变。
所以 KV 用 StaticCache 一次预分配,mask / position / 写指针也都是常驻张量,
在 graph 内原地推进,每步只改里面的数字。

用法两步:
    step = DecodeStep.new(mgr, batch, start, max_len)
    first_token = step.prefill(mgr, input_ids, attention_mask, names)
    graph, out = capture_graph(mgr, step)      # 之后每步 graph.replay()
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import StaticCache

WARMUP_ROUNDS = 3    # 录制前彩排几步,消耗掉 cuBLAS / triton 的首次准备


@dataclass
class DecodeStep:
    """一步 decode 要碰到的全部张量,建好后地址不再变."""

    kv: StaticCache             # 预分配 max_len 格的 KV
    input_ids: torch.Tensor     # (batch, 1) 这步喂进去的 token
    position_ids: torch.Tensor  # (batch, 1) rope 位置
    kv_pos: torch.Tensor        # (1,) KV 写到第几格
    attn_mask: torch.Tensor     # (batch, 1, 1, max_len) 哪些格子可见
    grid: torch.Tensor          # (max_len,) 0..max_len-1,算 attn_mask 用
    batch: int
    start: int                  # prefill 之后第一个位置,也是 rewind 的目标
    max_len: int                # KV 总格数

    @classmethod
    def new(cls, mgr, batch: int, start: int, max_len: int) -> "DecodeStep":
        """建一套:KV 全空,位置从 start 开始,输入 token 由调用方填."""
        device = mgr.pool.device
        return cls(
            kv=StaticCache(config=mgr.model.config, max_cache_len=max_len),
            input_ids=torch.zeros((batch, 1), device=device, dtype=torch.long),
            position_ids=torch.full(
                (batch, 1), start, device=device, dtype=torch.long),
            kv_pos=torch.tensor([start], device=device, dtype=torch.long),
            attn_mask=torch.zeros(
                (batch, 1, 1, max_len), device=device, dtype=torch.bool),
            grid=torch.arange(max_len, device=device, dtype=torch.long),
            batch=batch,
            start=start,
            max_len=max_len,
        )

    def like_new(self, mgr) -> "DecodeStep":
        """另一套同规格的:KV 全空、位置回到 start."""
        return DecodeStep.new(mgr, self.batch, self.start, self.max_len)

    def rewind(self) -> None:
        """写指针和位置退回 start.

        已写进 KV 的旧值不用清:从这里往前走,走到哪格就盖哪格。
        """
        for layer in self.kv.layers:
            layer.cumulative_length.fill_(self.start)
        self.position_ids.fill_(self.start)
        self.kv_pos.fill_(self.start)

    def prefill(self, mgr, input_ids, attention_mask, lora_names) -> torch.Tensor:
        """prefill 一遍,把 KV 前 start 格填满,返回下一个 token (batch, 1)."""
        out = mgr.forward_with_lora(
            input_ids=input_ids, attention_mask=attention_mask,
            lora_names=lora_names,
            seq_lens=[int(input_ids.shape[1])] * self.batch,
            past_key_values=self.kv, use_cache=True)
        logits = out.logits
        return logits[:, -1].argmax(dim=-1, keepdim=True)

    def run(self, mgr):
        """走一步真实 decode:刷 mask → forward(顺便写 KV) → 位置 +1."""
        self.attn_mask.copy_(
            (self.grid <= self.kv_pos).view(1, 1, 1, -1))
        out = mgr.inference_model(
            input_ids=self.input_ids, attention_mask=self.attn_mask,
            position_ids=self.position_ids, past_key_values=self.kv,
            cache_position=self.kv_pos, use_cache=True)
        self.position_ids.add_(1)
        self.kv_pos.add_(1)
        return out


def capture_graph(mgr, step: DecodeStep):
    """彩排几步再录一步,返回 (graph, 录下来的输出).

    彩排必须走另一条 stream、用另一套 KV:cuBLAS 开 workspace、triton 首次编译
    这类只做一次的动作录不进 graph,要先把它们消耗在彩排里。
    录制那次会真的跑一步,所以返回前 rewind,让调用方从干净状态开始。
    """
    warm = step.like_new(mgr)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_ROUNDS):
            warm.run(mgr)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step.run(mgr)
    step.rewind()
    return graph, out
