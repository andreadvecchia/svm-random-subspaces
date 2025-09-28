#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run experiments for Regularized ERM on Random Subspaces.")
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset to use (synthetic/mnist/other).")
    parser.add_argument("--loss", type=str, default="hinge", help="Loss function (hinge/logistic/square).")
    parser.add_argument("--subspace", type=str, default="nystrom", help="Subspace method (nystrom/random/sketch).")
    args = parser.parse_args()

    print(f"Running experiment with dataset={args.dataset}, loss={args.loss}, subspace={args.subspace}")
    # TODO: integrate actual experiment code here

if __name__ == "__main__":
    main()
