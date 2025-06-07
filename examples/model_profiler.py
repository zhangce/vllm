import argparse
from typing import List

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a simple MLP with torch.profiler")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trace-file", type=str, default=None,
                        help="Path to export Chrome trace JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mlp(args.input_dim, args.hidden_dim, args.output_dim, args.depth).to(device)
    model.eval()

    inputs = torch.randn(args.batch_size, args.input_dim, device=device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 with_stack=True) as prof:
        with record_function("model_inference"):
            with torch.inference_mode():
                model(inputs)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    if args.trace_file:
        prof.export_chrome_trace(args.trace_file)


if __name__ == "__main__":
    main()

