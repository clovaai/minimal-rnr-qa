# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import abc
import json
from timeit import default_timer as timer

from minimal_rnr.utils.inference import read_txt, get_first_matched_file_path, read_through_qa_file, \
    extend_span_to_full_words
from minimal_rnr.utils.logger import get_logger


class MinimalRnR(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.logger = get_logger("minimal-rnr-qa")

        titles_file = get_first_matched_file_path(args.resources_path, args.dataset, "*.titles.txt")
        self.logger.info(f"Loading titles from {titles_file}...")
        self.all_titles = read_txt(titles_file)

        docs_file = get_first_matched_file_path(args.resources_path, args.dataset, "*.docs.txt")
        self.logger.info(f"Loading docs from {docs_file}...")
        self.all_docs = read_txt(docs_file)

        if args.use_faiss_index:
            import faiss
            index_file = get_first_matched_file_path(args.resources_path, args.dataset, "*.index")
            self.logger.info(f"Loading index from {index_file}...")
            self.index = faiss.read_index(index_file)

            import numpy as np
            self.np = np
        else:
            self.index = None

        self.tokenizer = None  # must be overriden

        self.max_retriever_input_len = 256
        self.max_reader_input_len = 350
        self.max_answer_len = 10
        self.num_contexts = 10
        self.num_passage_answer_candidates = 5

    @abc.abstractmethod
    def get_question_encoding(self, question):
        pass

    @abc.abstractmethod
    def get_retriever_output(self, question, top_k):
        pass

    @abc.abstractmethod
    def get_reader_output(self, reader_input):
        pass

    @abc.abstractmethod
    def get_passage_score_weighted_answer_token_logits(self, token_logits, relevance_logits, attn_mask, passage_score_weight):
        pass

    def maybe_tensor_to_list(self, tensor):
        return tensor

    def get_inference_api(self):
        def api(question, top_k, passage_score_weight):
            return self.predict_answer(question, top_k, passage_score_weight, return_context=True)

        return api

    def inference_on_file(self, input_path, output_path, top_k=50, passage_score_weight=None):
        num_correct = 0
        total = 0

        evaluation_start = timer()
        with open(output_path, "w", encoding="utf-8") as f:
            for qa_dict in read_through_qa_file(input_path):
                start = timer()
                question = qa_dict["question"]
                prediction = self.predict_answer(question, top_k, passage_score_weight)
                qa_dict["prediction"] = prediction

                if "answers" in qa_dict:
                    from minimal_rnr.utils.evaluation import ems
                    correct = ems(prediction, qa_dict["answers"])
                    qa_dict["correct"] = correct

                    total += 1
                    num_correct += int(correct)

                f.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
                self.logger.info(str(qa_dict) + " (%.4fs)" % (timer() - start))

        if total > 0:
            self.logger.info(f"EM: {100 * num_correct / total:.4f} ({num_correct} / {total})")
            self.logger.info(f"Evaluation took {timer() - evaluation_start:.4f}s.")

    def predict_answer(self, question, top_k=50, passage_score_weight=None, return_context=False):
        # retrieve
        start = timer()
        if self.index:
            retrieved_doc_ids = self._get_retriever_output_from_faiss_index(question, top_k)
        else:
            retrieved_doc_ids = self.get_retriever_output(question, top_k)

        self.logger.info(f"  retrieve: {timer() - start:.4f}s")

        start = timer()
        title_doc_dict = self._get_title_doc_dict(retrieved_doc_ids)
        reader_input = self._get_reader_input(question, title_doc_dict)
        self.logger.info(f"  convert: {timer() - start:.4f}s")

        # read
        start = timer()
        reader_output = self.get_reader_output(reader_input)
        self.logger.info(f"  read: {timer() - start:.4f}s")

        start = timer()
        if passage_score_weight is not None:
            answer = self._get_answer_deep(reader_input, reader_output, title_doc_dict["titles"], passage_score_weight, return_context=return_context)
        else:
            answer = self._get_answer_greedy(reader_input, reader_output, title_doc_dict["titles"], return_context=return_context)
        self.logger.info(f"  search: {timer() - start:.4f}s")
        return answer

    def _get_retriever_output_from_faiss_index(self, question, top_k):
        start = timer()
        question_encoding = self.get_question_encoding(question)
        self.logger.info(f"  * encode: {timer() - start:.4f}s")

        start = timer()
        if not isinstance(question_encoding, self.np.ndarray):  # tfserving_faiss
            question_encoding = self.np.asarray(question_encoding, dtype=self.np.float32)
        _, np_retrieved_doc_ids = self.index.search(question_encoding, top_k)
        self.logger.info(f"  * faiss search: {timer() - start:.4f}s")
        retrieved_doc_ids = np_retrieved_doc_ids[0]
        return retrieved_doc_ids

    def _get_title_doc_dict(self, retrieved_doc_ids):
        retrieved_titles = []
        retrieved_docs = []

        for i in retrieved_doc_ids:
            retrieved_titles.append(self.all_titles[i])
            retrieved_docs.append(self.all_docs[i])

        return {
            "titles": retrieved_titles,
            "docs": retrieved_docs,
        }

    def _get_reader_input(self, question_str, title_doc_dict):
        input_ids = []
        attention_mask = []

        retrieved_titles = title_doc_dict["titles"]
        retrieved_docs = title_doc_dict["docs"]

        question = self.tokenizer.encode(question_str, max_length=self.max_retriever_input_len, truncation=True)

        # concat inputs
        for title, doc in zip(retrieved_titles, retrieved_docs):
            concat = question + title + [self.tokenizer.sep_token_id] + doc
            concat = concat[:self.max_reader_input_len]
            input_ids.append(concat)
        max_len = max(len(ids) for ids in input_ids)

        # pad inputs
        for i in range(len(input_ids)):
            padding = [self.tokenizer.pad_token_id] * (max_len - len(input_ids[i]))
            attention_mask.append([1] * len(input_ids[i]) + padding)
            input_ids[i] = input_ids[i] + padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _get_answer_deep(self, reader_input, reader_output, retrieved_titles, passage_score_weight=None, return_context=False):
        input_ids = reader_input["input_ids"]
        attn_mask = reader_input["attention_mask"]

        _start_logits = reader_output["start_logits"]
        _end_logits = reader_output["end_logits"]
        _relevance_logits = reader_output["relevance_logits"]

        # weighted scores
        start_logits = self.get_passage_score_weighted_answer_token_logits(_start_logits, _relevance_logits, attn_mask, passage_score_weight)
        end_logits = self.get_passage_score_weighted_answer_token_logits(_end_logits, _relevance_logits, attn_mask, passage_score_weight)

        start_logits = self.maybe_tensor_to_list(start_logits)
        end_logits = self.maybe_tensor_to_list(end_logits)

        candidate_answers = []
        candidate_contexts = []

        for passage_idx in range(len(input_ids)):
            sequence_len = sum(id_ != 0 for id_ in input_ids[passage_idx])
            passage_offset = input_ids[passage_idx].index(self.tokenizer.sep_token_id) + 1
            title_passage_ids = input_ids[passage_idx][passage_offset:sequence_len]

            p_start_logits = start_logits[passage_idx][passage_offset:sequence_len]
            p_end_logits = end_logits[passage_idx][passage_offset:sequence_len]

            scores = self._get_spans_sorted_with_scores(p_start_logits, p_end_logits)

            chosen_span_intervals = []
            p_candidate_answers = []
            p_candidate_contexts = []

            for (start_index, end_index), score in scores:
                ret = self._get_answer_and_passage(start_index, end_index, chosen_span_intervals, title_passage_ids)
                if not ret:
                    continue
                else:
                    answer, passage, start_index, ent_index = ret
                    title = retrieved_titles[passage_idx]

                if not return_context:
                    p_candidate_answers.append((answer, score))
                else:
                    context = (self.tokenizer.decode(passage[:start_index])
                               + " <strong>" + answer + "</strong> "
                               + self.tokenizer.decode(passage[end_index + 1:]))
                    p_candidate_contexts.append(({"title": self.tokenizer.decode(title),
                                                  "context": context}, score))

                if max(len(p_candidate_answers), len(p_candidate_contexts)) == self.num_passage_answer_candidates:
                    break

            if p_candidate_answers:
                candidate_answers.extend(p_candidate_answers)
            if p_candidate_contexts:
                candidate_contexts.extend(p_candidate_contexts)

        if not return_context:
            sorted_candidate_answers = sorted(candidate_answers, key=lambda x: x[1], reverse=True)
            return sorted_candidate_answers[0][0].strip()
        else:
            sorted_candidate_contexts = sorted(candidate_contexts, key=lambda x: x[1], reverse=True)
            return [context[0] for context in sorted_candidate_contexts][:self.num_contexts]

    def _get_answer_greedy(self, reader_input, reader_output, retrieved_titles, return_context=False):
        start_logits = reader_output["start_logits"]
        end_logits = reader_output["end_logits"]
        relevance_logits = reader_output["relevance_logits"]

        start_logits = self.maybe_tensor_to_list(start_logits)
        end_logits = self.maybe_tensor_to_list(end_logits)
        relevance_logits = self.maybe_tensor_to_list(relevance_logits)

        max_answer_length = 10
        input_ids = reader_input["input_ids"]

        top_passage_idx = max(enumerate(relevance_logits), key=lambda x: x[1])[0]

        sequence_len = sum(id_ != 0 for id_ in input_ids[top_passage_idx])
        passage_offset = input_ids[top_passage_idx].index(self.tokenizer.sep_token_id) + 1
        title_passage_ids = input_ids[top_passage_idx][passage_offset:sequence_len]
        p_start_logits = start_logits[top_passage_idx][passage_offset:sequence_len]
        p_end_logits = end_logits[top_passage_idx][passage_offset:sequence_len]

        scores = self._get_spans_sorted_with_scores(p_start_logits, p_end_logits)
        chosen_span_intervals = []

        for (start_index, end_index), score in scores:
            assert start_index <= end_index
            length = end_index - start_index + 1
            assert length <= max_answer_length

            ret = self._get_answer_and_passage(start_index, end_index, chosen_span_intervals, title_passage_ids)
            if not ret:
                continue
            answer, passage, start_index, end_index = ret
            title = retrieved_titles[top_passage_idx]

            if not return_context:
                return answer.strip()
            else:
                context = (self.tokenizer.decode(passage[:start_index])
                           + " <strong>" + answer + "</strong> "
                           + self.tokenizer.decode(passage[end_index + 1:]))
                return [{"title": self.tokenizer.decode(title), "context": context}]

    def _get_spans_sorted_with_scores(self, p_start_logits, p_end_logits):
        scores = []
        for (i, s) in enumerate(p_start_logits):
            for (j, e) in enumerate(p_end_logits[i:i + self.max_answer_len]):
                scores.append(((i, i + j), s + e))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores

    def _get_answer_and_passage(self, start_index, end_index, chosen_span_intervals, title_passage_ids):
        assert start_index <= end_index
        length = end_index - start_index + 1
        assert length <= self.max_answer_len

        if any([start_index <= prev_start_index <= prev_end_index <= end_index or
                prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals]):
            return

        start_index, end_index = extend_span_to_full_words(self.tokenizer, title_passage_ids, (start_index, end_index))

        title_offset = title_passage_ids.index(self.tokenizer.sep_token_id) + 1

        passage = title_passage_ids[title_offset:]
        start_index -= title_offset
        end_index -= title_offset

        answer = self.tokenizer.decode(passage[start_index:end_index + 1])

        return answer, passage, start_index, end_index
