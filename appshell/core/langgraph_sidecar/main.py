from __future__ import annotations

import argparse
import pathlib
import sys


if __package__ in (None, ""):
    package_root = pathlib.Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from langgraph_sidecar.server import serve  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C LangGraph sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=58331)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
