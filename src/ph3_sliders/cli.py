"""Historical planner and explicit fixed-bank current014 product commands."""

import argparse
import json
import sys

from .contract import LiveOperationUnavailable, RequestPlan, refuse_live


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("bank-config", "bank-check", "bank-train"):
        from .execution import product_cli
        try:
            return product_cli(argv)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__, epilog="Fixed native bank: bank-config, bank-check, bank-train; see docs/FIXED-BANK-API.md")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print a request plan without runtime imports or file reads")
    plan.add_argument("--prompt", required=True)
    plan.add_argument("--width", type=int, default=2912)
    plan.add_argument("--height", type=int, default=2912)
    plan.add_argument("--reference", action="append", default=[])
    plan.add_argument("--seed", type=int, default=7)
    plan.add_argument("--strength", type=float, default=1)
    commands.add_parser("cpu-check", help="probe the shared routed projection on an unrelated CPU fixture")
    for op in ("train", "render", "download", "publish"):
        commands.add_parser(op, help="unavailable in this source-only revision")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = RequestPlan(args.prompt, args.width, args.height, tuple(args.reference),
                                 args.seed, args.strength).as_dict()
        elif args.command == "cpu-check":
            from .synthetic import check
            result = check()
        else:
            refuse_live(args.command)
    except (ValueError, LiveOperationUnavailable) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
