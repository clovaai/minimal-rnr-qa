"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import logging
import os
from functools import wraps
from typing import Optional

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

USE_TF = True
USE_TORCH = False
_torch_available = False
_torch_tpu_available = False
_psutil_available = False
_py3nvml_available = False
_has_apex = False

try:
    import tensorflow as tf

    assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
    _tf_available = True  # pylint: disable=invalid-name
    logger.info("TensorFlow version {} available.".format(tf.__version__))
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
MODEL_CARD_NAME = "modelcard.json"


def is_tf_available():
    return _tf_available


def cached_path(
    url_or_filename,
) -> Optional[str]:
    if os.path.exists(url_or_filename):
        output_path = url_or_filename
    else:
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    return output_path


def tf_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_tf_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires TF.")

    return wrapper