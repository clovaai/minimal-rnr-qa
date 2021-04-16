# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

from minimal_rnr.utils.inferencer import MinimalRnR
from minimal_rnr.pytorch.model import get_model_tokenizer_device


class TorchMinimalRnR(MinimalRnR):
    def __init__(self, args):
        super(TorchMinimalRnR, self).__init__(args)

        assert args.use_faiss_index, "PyTorch model cannot be run without  a faiss index."
        import torch
        self.torch = torch

        self.model, self.tokenizer, self.device = get_model_tokenizer_device(args)

    def get_question_encoding(self, question):
        input_ = self.tokenizer([question], max_length=self.max_retriever_input_len,
                                truncation=True, return_token_type_ids=True, return_tensors="pt")
        torch_input = self._add_prefix_and_to_device(input_, "retriever_")

        with self.torch.no_grad():
            question_encoding = self.model(**torch_input)
            np_question_encoding = question_encoding.cpu().numpy().astype("float32")

        return np_question_encoding

    def get_retriever_output(self, question, top_k):
        raise Exception("get_retriever_output is use for running without faiss index,",
                        "which is not supported for torch models")

    def get_reader_output(self, reader_input):
        torch_input = self._add_prefix_and_to_device(reader_input, "reader_", convert_to_tensor=True)

        with self.torch.no_grad():
            start_logits, end_logits, relevance_logits = self.model(**torch_input)

        relevance_logits = relevance_logits.squeeze()
        if relevance_logits.dim() == 0:  # for top_k=1
            relevance_logits = relevance_logits.unsqueeze(0)

        # returned as cuda tensors (on purpose)
        return {
            "start_logits": start_logits.squeeze(-1),
            "end_logits": end_logits.squeeze(-1),
            "relevance_logits": relevance_logits,
        }

    def get_passage_score_weighted_answer_token_logits(self, token_logits, relevance_logits, attn_mask, passage_score_weight):
        attn_mask = self.torch.tensor(attn_mask).float()

        relevance_logits = relevance_logits.unsqueeze(1)  # [M, 1]
        masked_token_logits = token_logits - 1e10 * (1.0 - attn_mask)
        log_span_prob = token_logits - masked_token_logits.logsumexp(dim=1, keepdim=True)      # [M, L] softmaxed over L
        log_passage_prob = relevance_logits - relevance_logits.logsumexp(dim=0, keepdim=True)  # [M, 1] softmaxed over M
        weighted_logits = log_span_prob * (1 - passage_score_weight) + log_passage_prob * passage_score_weight

        return weighted_logits

    def _add_prefix_and_to_device(self, data, prefix, convert_to_tensor=False):
        wrap = lambda x: self.torch.tensor(x) if convert_to_tensor else x
        data = {prefix + k: wrap(v).to(device=self.device) for k, v in data.items()}
        return data

    def maybe_tensor_to_list(self, tensor):
        if not isinstance(tensor, (list, tuple)):  # torch tensor
            return tensor.cpu().tolist()
        return tensor
