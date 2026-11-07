"""Build an offline 2-D review pack, or retrack explicitly curated seeds."""

import argparse
from pathlib import Path

from ctdr_pipeline.motion_validation.review_bundle import build_review_bundle, retrack_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--validation-run", type=Path, required=True)
    build.add_argument("--assets-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    retrack = sub.add_parser("retrack")
    retrack.add_argument("--review", type=Path, required=True)
    retrack.add_argument("--bundle-dir", type=Path, required=True)
    retrack.add_argument("--assets-root", type=Path, required=True)
    retrack.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_review_bundle(args.validation_run, args.assets_root, args.output_dir)
        print(f"Research-only review pack: {args.output_dir / 'index.html'} | pack_id={result['pack_id']}")
    else:
        result = retrack_review(args.review, args.bundle_dir, args.assets_root, args.output_dir)
        print(f"{result['status']} | included={result['review_summary']['included_point_count']} | patient_3d_optimization_enabled=false")


if __name__ == "__main__":
    main()
