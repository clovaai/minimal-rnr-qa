# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import os
import argparse
import glob
from distutils.util import strtobool

from minimal_rnr.utils.logger import get_logger
from minimal_rnr.utils.demo import run_app
from minimal_rnr.tfserving.inferencer import TFServingMinimalRnR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--tfserving_ip", default="127.0.0.1")
    parser.add_argument("--tfserving_port", default=8501)

    parser.add_argument("--resources_path", default="/resources")
    parser.add_argument("--use_faiss_index", default=True, type=strtobool)

    parser.add_argument("--demo_port", default=10001, type=int)
    parser.add_argument("--examples_path", default="/workspace/minimal_rnr/utils/static/examples.txt")
    args = parser.parse_args()
    return args


def main(args):
    logger = get_logger("minimal-rnr-qa")
    logger.info(vars(args))

    if args.use_faiss_index:
        assert glob.glob(os.path.join(args.resources_path, args.dataset, "*.index")), \
            f"Index file does not exist in the path: {os.path.join(args.resources_path, args.dataset)}"

    minimal_rnr = TFServingMinimalRnR(args)
    run_app(args, minimal_rnr)


if __name__ == "__main__":
    args = get_args()
    main(args)
