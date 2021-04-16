# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import sys
import logging


def get_logger(name):
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)

    # initialization
    formatter = logging.Formatter(fmt="[MinR&R %(asctime)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger