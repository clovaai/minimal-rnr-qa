# coding=utf-8
# minimal-rnr-qa
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import os
import glob
import json
import requests
from argparse import ArgumentParser
from timeit import default_timer as timer

from bert_tokenizer import BertTokenizer


def read_txt(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            ids = [int(x) for x in line.rstrip().split(" ")]
            data.append(ids)
    return data


def get_api_input(signature_name, input_):
    if "token_type_ids" in input_:
        del input_["token_type_ids"]

    if type(input_) != dict:
        input_ = dict(input_)

    return json.dumps({
        "signature_name": signature_name,
        "inputs": input_,
    })


def get_output(url, data):
    return requests.post(url, data).json()["outputs"]


def get_score_input(token_logits, relevace_logits, attn_mask, passage_score_weight):
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


def get_retriever_input(question_str, tokenizer, top_k):
    input_ = tokenizer([question_str], max_length=256, truncation=True)
    input_.update({"top_k": top_k})
    return get_api_input("retrieve", input_)


def get_reader_input(question_str, tokenizer, retriever_output):
    input_ids = []
    attention_mask = []

    retrieved_titles = retriever_output["titles"]
    retrieved_docs = retriever_output["docs"]

    question = tokenizer.encode(question_str, max_length=256, truncation=True)
    for title, doc in zip(retrieved_titles, retrieved_docs):
        concat = question + title + [tokenizer.sep_token_id] + doc
        concat = concat[:350]
        input_ids.append(concat)
    max_len = max(len(ids) for ids in input_ids)
    for i in range(len(input_ids)):
        padding = [0] * (max_len - len(input_ids[i]))
        attention_mask.append([1] * len(input_ids[i]) + padding)
        input_ids[i] = input_ids[i] + padding
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def is_sub_word_id(tokenizer, token_id):
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    return token.startswith("##") or token.startswith(" ##")


def extend_span_to_full_words(tokenizer, tokens, span):
    start_index, end_index = span
    max_len = len(tokens)
    while start_index > 0 and is_sub_word_id(tokenizer, tokens[start_index]):
        start_index -= 1

    while end_index < max_len - 1 and is_sub_word_id(tokenizer, tokens[end_index + 1]):
        end_index += 1

    return start_index, end_index


def get_answer_greedy(tokenizer, reader_input, reader_output):
    max_answer_length = 10
    input_ids = reader_input["input_ids"]
    relevance_logits = reader_output["relevance_logits"]
    if not isinstance(relevance_logits, (tuple, list)):
        relevance_logits = [relevance_logits]

    top_doc_idx = max(enumerate(relevance_logits), key=lambda x: x[1])[0]

    sequence_len = sum(id_ != 0 for id_ in input_ids[top_doc_idx])
    passage_offset = input_ids[top_doc_idx].index(tokenizer.sep_token_id) + 1
    ctx_ids = input_ids[top_doc_idx][passage_offset:sequence_len]
    p_start_logits = reader_output["start_logits"][top_doc_idx][passage_offset:sequence_len]
    p_end_logits = reader_output["end_logits"][top_doc_idx][passage_offset:sequence_len]

    scores = []
    for (i, s) in enumerate(p_start_logits):
        for (j, e) in enumerate(p_end_logits[i:i + max_answer_length]):
            scores.append(((i, i + j), s + e))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    chosen_span_intervals = []

    answer = ""
    for (start_index, end_index), score in scores:
        assert start_index <= end_index
        length = end_index - start_index + 1
        assert length <= max_answer_length

        if any([start_index <= prev_start_index <= prev_end_index <= end_index or
                prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals]):
            continue

        start_index, end_index = extend_span_to_full_words(tokenizer, ctx_ids, (start_index, end_index))
        answer = tokenizer.decode(ctx_ids[start_index:end_index + 1], skip_special_tokens=True)
        break
    return answer


def get_answer_deep(tokenizer, reader_input, reader_output, url, passage_score_weight):
    max_answer_length = 10
    input_ids = reader_input["input_ids"]
    attn_mask = reader_input["attention_mask"]
    relevance_logits = reader_output["relevance_logits"]
    if not isinstance(relevance_logits, (tuple, list)):
        relevance_logits = [relevance_logits]

    start_logits = get_output(url, get_score_input(reader_output["start_logits"], relevance_logits, attn_mask, passage_score_weight))
    end_logits = get_output(url, get_score_input(reader_output["end_logits"], relevance_logits, attn_mask, passage_score_weight))

    nbest = []
    for passage_idx in range(len(input_ids)):
        sequence_len = sum(id_ != 0 for id_ in input_ids[passage_idx])
        passage_offset = input_ids[passage_idx].index(tokenizer.sep_token_id) + 1
        ctx_ids = input_ids[passage_idx][passage_offset:sequence_len]

        p_start_logits = start_logits[passage_idx][passage_offset:sequence_len]
        p_end_logits = end_logits[passage_idx][passage_offset:sequence_len]

        scores = []
        for (i, s) in enumerate(p_start_logits):
            for (j, e) in enumerate(p_end_logits[i:i + max_answer_length]):
                scores.append(((i, i + j), s + e))
                
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        chosen_span_intervals = []
        best_spans = []

        for (start_index, end_index), score in scores:
            assert start_index <= end_index
            length = end_index - start_index + 1
            assert length <= max_answer_length

            if any([start_index <= prev_start_index <= prev_end_index <= end_index or
                    prev_start_index <= start_index <= end_index <= prev_end_index
                    for (prev_start_index, prev_end_index) in chosen_span_intervals]):
                continue

            start_index, end_index = extend_span_to_full_words(tokenizer, ctx_ids, (start_index, end_index))

            title_offset = ctx_ids.index(tokenizer.sep_token_id) + 1

            context = ctx_ids[title_offset:]
            start_index -= title_offset
            end_index -= title_offset

            answer = tokenizer.decode(context[start_index:end_index + 1])

            best_spans.append((answer, score))
            if len(best_spans) == 5:
                break
        nbest.extend(best_spans)
        
    nbest = sorted(nbest, key=lambda x: x[1], reverse=True)
    return nbest[0][0]


def get_retriever_output(retrieved_doc_ids, titles, docs):
    retrieved_titles = [titles[i] for i in retrieved_doc_ids]
    retrieved_docs = [docs[i] for i in retrieved_doc_ids]
    return {
        "titles": retrieved_titles,
        "docs": retrieved_docs,
    }


def predict(url, tokenizer, titles, docs, question_str, top_k, passage_score_weight):
    question_str = question_str.lower()
    if question_str.endswith("?"):
        question_str = question_str[:-1]

    retrieved_doc_ids = get_output(url, get_retriever_input(question_str, tokenizer, top_k))
    retriever_output = get_retriever_output(retrieved_doc_ids, titles, docs)
    reader_input = get_reader_input(question_str, tokenizer, retriever_output)

    reader_output = get_output(url, get_api_input("read", reader_input))
    if passage_score_weight is not None:
        answer = get_answer_deep(tokenizer, reader_input, reader_output, url, passage_score_weight)
    else:
        answer = get_answer_greedy(tokenizer, reader_input, reader_output)
    return answer


def main(args):
    print(args)

    url = args.url
    titles_path = sorted(glob.glob(os.path.join(args.resources_path, "*.titles.txt")))[0]
    docs_path = sorted(glob.glob(os.path.join(args.resources_path, "*.docs.txt")))[0]

    print(titles_path)
    print(docs_path)

    titles = read_txt(titles_path)
    docs = read_txt(docs_path)

    tokenizer = BertTokenizer.from_pretrained("bert_tokenizer/mobilebert-uncased")

    with open(args.input_path, "r", encoding="utf-8") as input_file, open(args.output_path, "w", encoding="utf-8") as output_file:
        for line in input_file:
            start = timer()
            question_str = json.loads(line.strip())["question"]
            answer = predict(url, tokenizer, titles, docs, question_str, args.top_k, args.passage_score_weight)
            out = {"question": question_str, "prediction": answer.strip()}
            output_file.write(json.dumps(out) + "\n")
            print(str(out) + " (%.4fs)" % (timer() - start))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--url",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resources_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.txt",
        required=True,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--passage_score_weight",
        type=float,
        default=None,
    )

    args = parser.parse_args()
    main(args)
