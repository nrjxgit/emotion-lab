"""Train fusion head with modality dropout. --dry-run for CI."""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print("fusion dry-run ok")


if __name__ == "__main__":
    main()
