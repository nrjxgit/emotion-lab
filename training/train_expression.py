"""Train expression head (transfer learning). --dry-run for CI without data."""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data", default="")
    a = ap.parse_args()
    print(
        "dry-run ok"
        if a.dry_run or not a.data
        else f"training on {a.data} (implement per dataset loader)"
    )


if __name__ == "__main__":
    main()
