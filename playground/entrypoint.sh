#!/bin/bash
# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0


if [ -z "$MODE" ]; then
    echo "--env MODE must be defined: demo | file"
    exit 0
fi


# launch tfserving
if [ "$MODEL_TYPE" = "tfserving" ] || [ "$MODEL_TYPE" = "tfserving_faiss" ]; then
    /usr/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=minimal-rnr-qa --model_base_path=/models/minimal-rnr-qa &
fi


# set USE_FAISS
if [ "$MODEL_TYPE" = "tfserving" ]; then
    USE_FAISS=false
else
    USE_FAISS=true
fi


if [ "$MODE" = "demo" ]; then
    pip install --no-cache-dir flask tornado

    if [ "$MODEL_TYPE" = "tfserving" ] || [ "$MODEL_TYPE" = "tfserving_faiss" ]; then
        python -u run_tf_demo.py \
            --dataset $DATASET \
            `if [[ -n "${USE_FAISS}" ]]; then echo --use_faiss_index $USE_FAISS; fi`

    else
        python -u run_pt_demo.py \
            --dataset $DATASET

    fi

elif [ "$MODE" = "file" ]; then
    if [ "$MODEL_TYPE" = "tfserving" ] || [ "$MODEL_TYPE" = "tfserving_faiss" ]; then
        PYTHON_FILE="run_tf_inference.py"

    else
        PYTHON_FILE="run_pt_inference.py"
    fi

    python -u $PYTHON_FILE \
        --dataset $DATASET \
        `if [[ -n "${USE_FAISS}" ]]; then echo --use_faiss_index $USE_FAISS; fi` \
        `if [[ -n "${TOP_K}" ]]; then echo --top_k $TOP_K; fi` \
        `if [[ -n "${PASSAGE_W}" ]]; then echo --passage_score_weight $PASSAGE_W; fi` \
        --input_path $1 --output_path $2 \

else
    echo "--env MODE must be: demo | file"
    exit 0
fi
