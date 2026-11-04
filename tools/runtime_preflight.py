"""Verify the actual isolated modern runtime without GPU/model allocation."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('runtime preflight requires explicitly hidden CUDA devices')
    import torch
    from ph3_sliders.native_runtime import MODERN_COHORT, configure_numeric_runtime, runtime_identity

    configure_numeric_runtime()
    identity = runtime_identity(args.source_manifest, quantization='encoder_bf16', cohort=MODERN_COHORT)
    if torch.cuda.is_initialized():
        raise RuntimeError('runtime preflight unexpectedly initialized CUDA')
    print(json.dumps(dict(status='verified', cuda_initialized=False, runtime_identity=identity), allow_nan=False))


if __name__ == '__main__':
    main()
