#!/bin/bash
# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

URL="http://127.0.0.1:8501/v1/models/minimal-rnr-qa:predict"

# decompress tensorflow_model_server bzip2
bzip2 -d /usr/bin/tensorflow_model_server.bz2
chmod a+x /usr/bin/tensorflow_model_server

# decompress python
gzip -dr /usr/local/lib/python3.6

# decompress resources lzma
for f in /resources/*.lzma; do lzma -dc $f > "${f%.*}"; done
for f in /resources/*.lzma; do rm $f; done

# decompress models bzip2
bzip2 -d /models/minimal-rnr-qa/1/variables/variables.data-00000-of-00001.bz2
bzip2 -d /models/minimal-rnr-qa/1/saved_model.pb.bz2

# server
/usr/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=minimal-rnr-qa --model_base_path=/models/minimal-rnr-qa &

# client
cd /workspace
python -u infer.py --url $URL --resources_path /resources --input_path $1 --output_path $2 `if [[ -n "${TOP_K}" ]]; then echo --top_k $TOP_K; fi` `if [[ -n "${PASSAGE_W}" ]]; then echo --passage_score_weight $PASSAGE_W; fi`
