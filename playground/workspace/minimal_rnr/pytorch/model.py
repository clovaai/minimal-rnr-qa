#!/usr/bin/env python3
# ******************************************************************
# Copied and modified from https://github.com/facebookresearch/DPR *
# ******************************************************************
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from minimal_rnr.utils.inference import get_first_matched_file_path
from minimal_rnr.utils.logger import get_logger


def get_model_tokenizer_device(args):
    # not to import torch and transformers as default
    import torch
    from torch import Tensor as T
    from torch import nn
    from transformers import MobileBertModel, MobileBertConfig, AutoTokenizer


    def init_weights(modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


    class HFMobileBertEncoder(MobileBertModel):
        def __init__(self, config, project_dim: int = 0, ctx_bottleneck: bool = False):
            MobileBertModel.__init__(self, config)
            assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
            self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
            self.decode_proj = nn.Sequential(
                nn.Tanh(),
                nn.Linear(project_dim, (config.hidden_size + project_dim) // 2),
                nn.Tanh(),
                nn.Linear((config.hidden_size + project_dim) // 2, config.hidden_size),
            ) if ctx_bottleneck else None
            self.init_weights()

        @classmethod
        def init_encoder(cls, cfg_name: str) -> MobileBertModel:
            cfg = MobileBertConfig.from_pretrained(cfg_name)
            return cls.from_pretrained(cfg_name, config=cfg)

        def forward(self, input_ids: T, token_type_ids: T, attention_mask: T):
            if self.config.output_hidden_states:
                sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                                token_type_ids=token_type_ids,
                                                                                attention_mask=attention_mask)
            else:
                hidden_states = None
                sequence_output, pooled_output = super().forward(
                    input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            pooled_output = sequence_output[:, 0, :]
            return sequence_output, pooled_output, hidden_states

        def get_out_size(self):
            if self.encode_proj:
                return self.encode_proj.out_features
            return self.config.hidden_size


    class UnifiedRetrieverReader(nn.Module):
        def __init__(self, encoder: nn.Module):
            super(UnifiedRetrieverReader, self).__init__()

            self.emb_size = 128

            self.question_model = encoder
            hidden_size = encoder.config.hidden_size

            self.qa_outputs = nn.Linear(hidden_size, 2)
            self.qa_classifier = nn.Linear(hidden_size, 1)

            init_weights([self.qa_outputs, self.qa_classifier])

        @staticmethod
        def get_representation(sub_model: nn.Module, ids, segments, attn_mask):
            sequence_output = None
            pooled_output = None
            hidden_states = None
            if ids is not None:
                sequence_output, pooled_output, hidden_states = sub_model(ids, segments, attn_mask)

            return sequence_output, pooled_output, hidden_states

        def forward(
                self, retriever_input_ids=None, retriever_token_type_ids=None, retriever_attention_mask=None,
                reader_input_ids=None, reader_attention_mask=None, reader_token_type_ids=None):

            if retriever_input_ids is not None:
                _, encoding, _ = self.get_representation(
                    self.question_model, retriever_input_ids, retriever_token_type_ids, retriever_attention_mask)

                if self.emb_size is not None:
                    return encoding[:, :self.emb_size]
                return encoding

            if reader_input_ids is not None:
                start_logits, end_logits, relevance_logits = self._read(reader_input_ids, reader_token_type_ids, reader_attention_mask)
                return start_logits, end_logits, relevance_logits

        def _read(self, input_ids, token_type_ids, attention_mask):
            sequence_output, _pooled_output, _hidden_states = self.question_model(input_ids, token_type_ids, attention_mask)
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            qa_classifier_input = sequence_output[:, 0, :]
            relevance_logits = self.qa_classifier(qa_classifier_input)
            return start_logits, end_logits, relevance_logits


    cfg_name = "google/mobilebert-uncased"
    question_encoder = HFMobileBertEncoder.init_encoder(cfg_name)
    model = UnifiedRetrieverReader(question_encoder)
    tokenizer = AutoTokenizer.from_pretrained(cfg_name, do_lower_case=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = get_first_matched_file_path(args.model_path, args.dataset, "*.bin")
    logger = get_logger("minimal-rnr-qa")
    logger.info(f"Loading model from {model_file}...")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, device