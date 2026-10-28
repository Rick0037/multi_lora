import argparse
import os
import torch


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    # 执行路径四挡,彼此互斥,取值直接进文件名,避免不同实现互相覆盖 trace
    p.add_argument(
        "--mode",
        choices=["eager", "compile", "cudagraph", "compile_cudagraph"],
        default="eager",
    )
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--trace_dir", default="./traces/01_matmul_add")
    return p.parse_args()


class CudaGraphRunner:
    """把 fn 捕获成一张 CUDA graph,调用时先写静态输入再 replay。

    graph 记录的是显存地址而不是张量句柄,所以每次调用都必须把新数据
    copy_ 进构造时那几张静态输入 buffer,否则 replay 出来的永远是捕获时的结果。
    """

    def __init__(self, fn, static_inputs, warmup_iters=5):
        # 记住静态输入引用，__call__ 时按位置把新数据写回这几张 buffer
        self.static_inputs = static_inputs

        # 在侧 stream 上热身,避免 warmup 的动作被一起捕获进 graph
        # 捕获只记录“当前 stream”上的操作，先把执行切到旁路 stream
        stream = torch.cuda.Stream()
        # 侧 stream 要等默认 stream 的既有工作做完，顺序才正确
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_iters):
                fn(*static_inputs)
        # 回到默认 stream：让默认 stream 等侧 stream 做完
        torch.cuda.current_stream().wait_stream(stream)
        # 硬同步：确保 kernel 全部结束、autotune/显存分配都已落地再开始捕获
        torch.cuda.synchronize()

        # output 必须长期持有引用,否则 graph pool 里的静态显存会被回收复用
        # 捕获：with 块内不真正执行 kernel，只把启动序列录进 graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = fn(*static_inputs)
            # ! L50 这次 fn(...) 的 kernel 并不真正确算，它的启动序列被逐条抄进 self.graph。真正执行是在 L57 的

    def __call__(self, *inputs):
        # 1) graph 认地址不认句柄，先把新数据拷进静态输入 buffer
        for dst, src in zip(self.static_inputs, inputs):
            dst.copy_(src)
        # 2) 一次性重放整条 kernel 序列，CPU 侧只付一次启动开销
        self.graph.replay()
        # 3) 返回固定地址的输出视图（内容已被本次 replay 覆写）
        return self.output


def prime_graph(runner, x, w, b):
    """把“捕获完 ~ 首次 replay 落地”这段脏数据窗口关掉。"""
    runner(x, w, b)            # copy_ 静态输入 + replay
    torch.cuda.synchronize()   # replay 是异步的，等它算完 output 才可信


def main():
    args = parse_arguments()

    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    x = torch.randn(args.size, args.size, device=device, dtype=dtype)
    w = torch.randn(args.size, args.size, device=device, dtype=dtype)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype)

    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    use_compile = "compile" in args.mode
    use_graph = "cudagraph" in args.mode
    # compile_cudagraph -> compile, cudagraph 两个都有

    # import pdb
    # pdb.set_trace()

    # ! no-cudagraphs：关掉 torch.compile 自带的 cudagraph,交给下面手动捕获
    if use_compile:
        fn = torch.compile(fn, mode="max-autotune-no-cudagraphs")

    # call 是被 kernel() 执行的唯一入口,换实现只改这里
    call = fn
    if use_graph:
        # clone 出地址固定的静态输入，捕获与 replay 只认这三张 buffer
        xs, ws, bs = x.clone(), w.clone(), b.clone()
        call = CudaGraphRunner(fn, (xs, ws, bs))
        prime_graph(call, x, w, b)

    def kernel():
        with torch.profiler.record_function("test_matmul_add"):
            # graph 模式下这里是 copy_ 进静态输入 + replay，x/w/b 本身不进 graph
            return call(x, w, b)

    if args.warmup:
        for _ in range(3):
            kernel()
        # 刷新队列,避免后续被 profile 的步骤把之前的工作也算进来
        torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    warmup_tag = "warm" if args.warmup else "cold"
    tag = f"{args.size}_{args.dtype}_{warmup_tag}_{args.mode}"

    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    # wait 跳过嘈杂的初始化,warmup 会走一遍 profiler 但不记录
    # （让缓存/自动调优稳定下来）,active 才是最终出现在
    # 表格/trace 里的部分。
    # ? ---------------------------------?
    # 末步靠循环里的 prof.step() 推进,iters 必须与 active 相等
    schedule = torch.profiler.schedule(
        wait=0, warmup=0, active=args.iters, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                    ],
        schedule=schedule,
        record_shapes=True,  # 会增加 CPU 开销
        # profile_memory=False,  # 会增加 CPU 开销
        # with_stack=False,
    ) as prof:
        for _ in range(args.iters):
            kernel()      # 不接收不打印：读 output 会隐式同步，抹掉 graph 的发射收益
            prof.step()
    torch.cuda.synchronize()

    print(f"saving traces ... {trace_path}")
    prof.export_chrome_trace(trace_path)

    with open(table_path, "w") as f:
        f.write(prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
