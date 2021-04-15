FROM tensorflow/serving:2.3.0 as build_image
FROM python:3.6.11-slim-buster

ARG DATASET

# Install TF Serving pkg.
COPY --from=build_image /usr/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Reduce size
RUN apt-get update && apt-get install -y --no-install-recommends lzma bzip2 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN bzip2 /usr/bin/tensorflow_model_server

# Install python packages.
RUN pip install --no-cache-dir absl-py requests
RUN pip uninstall pip -y
RUN gzip -r /usr/local/lib/python3.6

# Delete unnecessary files
RUN rm -rf /root/* && rm /usr/bin/perl* && rm -rf /usr/lib/x86_64-linux-gnu/perl* && rm -rf /var/cache

COPY models/$DATASET /models/minimal-rnr-qa/1
COPY resources/$DATASET /resources
COPY workspace /workspace
COPY submission.sh /

RUN chmod a+x /submission.sh

WORKDIR /
