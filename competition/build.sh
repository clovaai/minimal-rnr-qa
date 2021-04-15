#!/bin/bash
# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

if [ "$#" -ne 1 ]; then
    echo "Usage: ./build.sh DATASET"
    echo "DATASET:    effqa | nq | trivia"
    exit 0
fi

DATASET=$1     # ["effqa", "nq", "trivia"]

docker build -f Dockerfile \
	--build-arg DATASET=$DATASET \
	. -t minimal-rnr-qa:$DATASET-competition

echo "minimal-rnr-qa:$DATASET"-competition