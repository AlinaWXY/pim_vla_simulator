import argparse

from .backend import run_trace
from .config import MemoryConfig
from .ffn import FFNConfig, generate_ffn, run_ffn
from .trace import generate_trace


def main():
    parser = argparse.ArgumentParser(description="LPDDR memory experiment interface")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run an LD/ST/PIM_MAC_AB/PIM_MAC_HB trace")
    run.add_argument("--config", required=True)
    run.add_argument("--trace", required=True)
    run.add_argument("--output", required=True)
    trace = sub.add_parser("trace", help="Generate a normal-memory smoke workload")
    trace.add_argument("--output", required=True)
    trace.add_argument("--requests", type=int, default=128)
    trace.add_argument("--pattern", choices=("read", "write", "mixed"), default="mixed")
    for command in ("ffn-trace", "ffn-run"):
        ffn = sub.add_parser(command, help="Generate or simulate a static output-major PIM FFN")
        ffn.add_argument("--output", required=True)
        ffn.add_argument("--packages", type=int, default=8)
        ffn.add_argument("--channels-per-package", type=int, default=2)
        ffn.add_argument("--ranks", type=int, default=1, help="Ranks per channel")
        ffn.add_argument("--horizon", type=int, default=50)
        ffn.add_argument("--reuse-tokens", type=int, default=1,
                         help="Input tokens computed per open-row visit; MAC timing unchanged")
        ffn.add_argument("--macab-tokens-per-command", type=int, default=1,
                         help="Experimental: token contexts served by one K-block MACAB")
        ffn.add_argument("--macab-issue-interval-ck", type=int, default=None,
                         help="Experimental nCCDAB override in integer memory CK")
        ffn.add_argument("--hidden-size", type=int, default=1024)
        ffn.add_argument("--intermediate-size", type=int, default=4096)
        ffn.add_argument("--model", choices=("lpddr5-6400", "lpddr5x-8533"),
                         default="lpddr5x-8533")
        ffn.add_argument("--density-gbit", type=int, default=16,
                         help="Effective capacity per channel/rank, NOT per package")
        ffn.add_argument("--no-refresh", action="store_true")
        ffn.add_argument("--input-buffer-bytes", type=int, default=None,
                         help="Per channel/rank input SRAM budget; default: size for a full tile")
        ffn.add_argument("--accumulator-buffer-bytes-per-bank", type=int, default=None,
                         help="Per bank FP32 accumulator budget; default: 4 bytes per tile token")
        ffn.add_argument("--nonlinear-cycles-per-token", type=int, default=None,
                         help="Shared GELU-product delay in memory CK; omitted means excluded")
        ffn.add_argument("--frontend-ratio", type=int, default=32)
        ffn.add_argument("--max-cycles", type=int, default=10_000_000)
    args = parser.parse_args()
    try:
        if args.command == "trace":
            print(generate_trace(args.output, requests=args.requests, pattern=args.pattern))
        elif args.command.startswith("ffn-"):
            fields = vars(args).copy()
            output = fields.pop("output")
            command = fields.pop("command")
            fields["refresh"] = not fields.pop("no_refresh")
            config = FFNConfig(**fields)
            if command == "ffn-trace":
                result = generate_ffn(config, output)
                print(f"{config.total_banks} banks; layout {result['layout_sha256']}")
                print(f"Trace: {output}/ffn.trace")
            else:
                result = run_ffn(config, output)
                print(f"{result['memory_cycles']} CK; {result['latency_ns']:.3f} ns")
                print(result["latency_scope"])
                print(f"Result: {output}/result.json")
        else:
            result = run_trace(MemoryConfig.from_yaml(args.config), args.trace, args.output)
            print(
                f"{result['completed_requests']} requests completed; "
                f"{result['memory_cycles']} cycles; {result['latency_ns']:.3f} ns"
            )
            print(f"Result: {args.output}/result.json")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
