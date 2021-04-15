import csv
import glob
import json
import os
import regex


class SimpleTokenizer(object):
    # Copied and modified from https://github.com/facebookresearch/FiD

    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def extend_span_to_full_words(tokenizer, tokens, span):
    # Copied and modified from https://github.com/facebookresearch/DPR

    def is_sub_word_id(tokenizer, token_id):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    start_index, end_index = span
    max_len = len(tokens)
    while start_index > 0 and is_sub_word_id(tokenizer, tokens[start_index]):
        start_index -= 1

    while end_index < max_len - 1 and is_sub_word_id(tokenizer, tokens[end_index + 1]):
        end_index += 1

    return start_index, end_index


def read_txt(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            ids = [int(x) for x in line.rstrip().split(" ")]
            data.append(ids)
    return data


def get_first_matched_file_path(directory, dataset, file_pattern):
    return sorted(glob.glob(os.path.join(directory, dataset, file_pattern)))[0]


def normalize_question(question):
    question = question.lower()
    if question.endswith("?"):
        question = question[:-1]
    return question


def _read_through_csv_qa_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            question = normalize_question(row[0])
            answers = None if len(row) < 2 else row[1]

            if answers is None:
                qa_dict = {"question": question}
            else:
                qa_dict = {"question": question, "answers": answers}
            yield qa_dict


def _read_through_jsonl_qa_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            qa_dict = json.loads(line.strip())
            qa_dict["question"] = normalize_question(qa_dict["question"])

            if "answer" in qa_dict:
                qa_dict["answers"] = qa_dict["answer"]
                del qa_dict["answer"]

            yield qa_dict


def read_through_qa_file(file_path):
    if file_path.endswith(".csv"):
        return _read_through_csv_qa_file(file_path)
    elif file_path.endswith(".jsonl") or file_path.endswith("jl"):
        return _read_through_jsonl_qa_file(file_path)
    else:
        raise ValueError(f"File {file_path} must be either csv or jsonlines file")