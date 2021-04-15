#!/bin/bash
# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0


if [ "$#" -ne 2 ]; then
    echo "Usage: ./build.sh DATASET MODEL_TYPE"
    echo "DATASET:    effqa     | nq              | trivia"
    echo "MODEL_TYPE: tfserving | tfserving_faiss | pytorch"
    exit 0
fi


DATASET=$1     # ["effqa", "nq", "trivia"]
MODEL_TYPE=$2  # ["tfserving", "tfserving_faiss", "pytorch"]

if [ "$MODEL_TYPE" = "pytorch" ]; then
    DOCKERFILE="Dockerfile-pytorch"
else
    DOCKERFILE="Dockerfile-tfserving"
fi


docker build -f configs/$DOCKERFILE \
	--build-arg DATASET=$DATASET \
	--build-arg MODEL_TYPE=$MODEL_TYPE \
	. -t minimal-rnr-qa:$DATASET-$MODEL_TYPE

echo "minimal-rnr-qa:$DATASET-$MODEL_TYPE"
