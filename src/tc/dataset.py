import re

import torch
from torch.utils import data as D
from transformers import AutoTokenizer

BATCH_SIZE = 8
NUM_CLASSES = 14


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.additional_special_tokens = ["[BOP]", "[EOP]"]
    return tokenizer


def get_spans_with_context(data, max_length=256):
    text = data["text"]
    examples = []

    start = data["technique_classification"]["start_char_offset"]
    end = data["technique_classification"]["end_char_offset"]
    for i in range(len(start)):
        span = text[start[i] : end[i]]
        span_words = span.split(" ")
        span_length = len(span_words)

        before_context = text[: start[i]].split(" ")
        after_context = text[end[i] + 1 :].split(" ")
        before_words = len(before_context)
        after_words = len(after_context)

        pre_buffer, post_buffer = (0, 0)
        remaining_length = max_length - (span_length * 2)

        before_length = min(before_words, remaining_length // 2)
        if before_length < remaining_length // 2:
            pre_buffer = remaining_length // 2 - before_length

        after_length = min(after_words, remaining_length // 2)
        if after_length < remaining_length // 2:
            post_buffer = remaining_length // 2 - after_length

        if pre_buffer > 0 and after_words > after_length:
            buffer = min(pre_buffer, after_words - after_length)
            after_length += buffer

        if post_buffer > 0 and before_words > before_length:
            buffer = min(post_buffer, before_words - before_length)
            before_length += buffer

        assert (
            before_length + after_length + (span_length * 2)
        ) <= max_length, f"Span length is {span_length}, before length is {before_length}, after length is {after_length}, total length is {(before_length + after_length + (span_length *2))}"

        context = (
            before_context[-1 - before_length : -1]
            + span_words
            + after_context[:after_length]
        )
        context = " ".join(context)
        context = re.sub(r"\n+", "\n", context).strip()

        example = {
            "article_id": data["article_id"],
            "context": context,
            "span": span,
            "span_start": start[i],
            "span_end": end[i],
            "text": text,
            "technique": data["technique_classification"].get(
                "technique", [-1] * len(start)
            )[i],
        }
        examples.append(example)

    return examples


def get_spans_in_context(data, max_length=256):
    text = data["text"]
    examples = []

    start = data["technique_classification"]["start_char_offset"]
    end = data["technique_classification"]["end_char_offset"]
    for i in range(len(start)):
        span = text[start[i] : end[i]]
        span_words = span.split(" ")
        span_length = len(span_words)

        before_context = text[: start[i]].split(" ")
        after_context = text[end[i] + 1 :].split(" ")
        before_words = len(before_context)
        after_words = len(after_context)

        pre_buffer, post_buffer = (0, 0)
        remaining_length = max_length - span_length - 2

        before_length = min(before_words, remaining_length // 2)
        if before_length < remaining_length // 2:
            pre_buffer = remaining_length // 2 - before_length

        after_length = min(after_words, remaining_length // 2)
        if after_length < remaining_length // 2:
            post_buffer = remaining_length // 2 - after_length

        if pre_buffer > 0 and after_words > after_length:
            buffer = min(pre_buffer, after_words - after_length)
            after_length += buffer

        if post_buffer > 0 and before_words > before_length:
            buffer = min(post_buffer, before_words - before_length)
            before_length += buffer

        assert (
            before_length + after_length + span_length + 2
        ) <= max_length, f"Span length is {span_length}, before length is {before_length}, after length is {after_length}, total length is {(before_length + after_length + (span_length *2))}"

        context = (
            before_context[-1 - before_length : -1]
            + ["[BOP]"]
            + span_words
            + ["[EOP]"]
            + after_context[:after_length]
        )
        context = " ".join(context)
        context = re.sub(r"\n+", "\n", context).strip()

        example = {
            "article_id": data["article_id"],
            "context": context,
            "span": span,
            "span_start": start[i],
            "span_end": end[i],
            "text": text,
            "technique": data["technique_classification"].get(
                "technique", [-1] * len(start)
            )[i],
        }
        examples.append(example)

    return examples


def prepare_data(dataset, max_length=256, prepare_fn=get_spans_with_context):
    train_data = []
    for data in dataset["train"]:
        train_data.extend(prepare_fn(data, max_length=max_length))

    dev_data = []
    for data in dataset["validation"]:
        dev_data.extend(prepare_fn(data, max_length=max_length))

    return train_data, dev_data


def label_counts(data, labels):
    label_counts = [0] * len(labels)
    for d in data:
        label_counts[d["technique"]] += 1

    return label_counts


class PTCTechniqueClassificationDataset(D.Dataset):
    def __init__(self, data, tokenizer, num_classes=NUM_CLASSES):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        if "[BOP]" in data[0] and "[EOP]" in data[0]:
            self.mode = "spans_in_context"
        else:
            self.mode = "spans_with_context"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_item = self.data[i]
        if self.mode == "spans_in_context":
            encodings = self.tokenizer(data_item["context"], truncation=True)
        else:
            encodings = self.tokenizer(
                data_item["span"], data_item["context"], truncation=True
            )
        label = torch.tensor(data_item["technique"])
        encodings["labels"] = label

        return encodings