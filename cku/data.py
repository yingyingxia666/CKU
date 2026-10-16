"""Small, explicit prompt/response readers and answer-only causal-LM batches."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    prompt: str
    response: str


def load_records(path: str | Path) -> list[Record]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        raise ValueError("Data must be a .json, .jsonl or .csv file")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty list of records")
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        if "instruction" in row:
            instruction, context = row["instruction"], row.get("input", "")
            if not isinstance(instruction, str) or not isinstance(context, str):
                raise ValueError(f"{path}: instruction/input must be strings at record {index}")
            prompt = instruction + ("\n" + context if context else "")
            response = row.get("output")
        elif "prompt" in row and "response" in row:
            prompt, response = row["prompt"], row["response"]
        else:
            prompt, response = row.get("input"), row.get("output")
        if not all(isinstance(value, str) and value.strip() for value in (prompt, response)):
            raise ValueError(f"{path}: nonempty prompt/response required at record {index}")
        records.append(Record(prompt=prompt, response=response))
    return records


def encode_record(record: Record, tokenizer, max_length: int, prompt_format: str) -> dict:
    if prompt_format == "chat":
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template; choose --prompt-format plain")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": record.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    elif prompt_format == "plain":
        prompt = f"User: {record.prompt}\nAssistant: "
    else:
        raise ValueError("prompt_format must be 'plain' or 'chat'")
    # Tokenize segments without injecting a second BOS at the answer boundary.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None and (
        not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id
    ):
        prompt_ids.insert(0, tokenizer.bos_token_id)
    answer_ids = tokenizer.encode(record.response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None and (
        not answer_ids or answer_ids[-1] != tokenizer.eos_token_id
    ):
        answer_ids.append(tokenizer.eos_token_id)
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    if not any(value != -100 for value in labels[1:]):
        raise ValueError(
            "Truncation leaves no answer tokens; increase max_length or shorten prompt"
        )
    return {"input_ids": input_ids, "labels": labels}


class AnswerDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int, prompt_format: str):
        self.examples = [
            encode_record(record, tokenizer, max_length, prompt_format) for record in records
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PairedDataset(Dataset):
    def __init__(self, forget: AnswerDataset, retain: AnswerDataset | None = None):
        self.forget = forget
        self.retain = retain

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, index):
        item = {"forget": self.forget[index]}
        if self.retain is not None:
            item["retain"] = self.retain[index % len(self.retain)]
        return item


class AnswerCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        length = max(len(example["input_ids"]) for example in examples)
        ids, labels, attention = [], [], []
        for example in examples:
            size = len(example["input_ids"])
            pad = length - size
            ids.append(example["input_ids"] + [self.pad_token_id] * pad)
            labels.append(example["labels"] + [-100] * pad)
            attention.append([1] * size + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def paired(self, examples):
        return {key: self([example[key] for example in examples]) for key in examples[0]}
