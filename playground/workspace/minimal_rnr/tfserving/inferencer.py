# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0


import json
import os

import requests

from minimal_rnr.tfserving.bert_tokenizer import BertTokenizer
from minimal_rnr.utils.inferencer import MinimalRnR


class TFServingMinimalRnR(MinimalRnR):
    def __init__(self, args):
        super(TFServingMinimalRnR, self).__init__(args)

        self.url = f"http://{args.tfserving_ip}:{args.tfserving_port}/v1/models/minimal-rnr-qa:predict"
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "bert_tokenizer/mobilebert-uncased"))

    def get_question_encoding(self, question):
        return self._get_api_output(self._get_question_encoder_api_input(question))

    def get_retriever_output(self, question, top_k):
        retrieved_doc_ids = self._get_api_output(self._get_retrieve_api_input(question, top_k))
        return retrieved_doc_ids

    def get_reader_output(self, reader_input):
        reader_api_input = self._get_api_input("read", reader_input)
        reader_output = self._get_api_output(reader_api_input)

        if not isinstance(reader_output["relevance_logits"], (tuple, list)):  # for top_k=1
            reader_output["relevance_logits"] = [reader_output["relevance_logits"]]

        return reader_output

    def get_passage_score_weighted_answer_token_logits(self, token_logits, relevance_logits, attn_mask, passage_score_weight):
        weighted_logits = self._get_api_output(self._get_score_api_input(token_logits, relevance_logits, attn_mask, passage_score_weight))
        return weighted_logits

    def _get_question_encoder_api_input(self, question):
        input_ = self.tokenizer([question], max_length=self.max_retriever_input_len, truncation=True, return_token_type_ids=True)
        return self._get_api_input("encode", input_, preserve_token_type_ids=True)

    def _get_retrieve_api_input(self, question, top_k):
        input_ = self.tokenizer([question], max_length=self.max_retriever_input_len, truncation=True)
        input_["top_k"] = top_k

        return self._get_api_input("retrieve", input_)

    def _get_score_api_input(self, token_logits, relevace_logits, attn_mask, passage_score_weight):
        input_ = {
            "token_logits": token_logits,
            "relevance_logits": relevace_logits,
            "attn_mask": attn_mask,
            "passage_score_weight": passage_score_weight,
        }
        return json.dumps({
            "signature_name": "get_score",
            "inputs": input_,
        })

    def _get_api_input(self, signature_name, input_, preserve_token_type_ids=False):
        if not preserve_token_type_ids and "token_type_ids" in input_:
            del input_["token_type_ids"]

        if type(input_) != dict:
            input_ = dict(input_)

        return json.dumps({
            "signature_name": signature_name,
            "inputs": input_,
        })

    def _get_api_output(self, payload):
        response = requests.post(self.url, payload)
        if response.status_code != 200:
            response.raise_for_status()
        return response.json()["outputs"]