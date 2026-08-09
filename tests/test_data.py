import json

from minimind_kd.training.data import PackedPretrainDataset, SFTDataset


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(character) % 20) for character in text]


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_pretrain_and_sft_label_masks(tmp_path):
    tokenizer = FakeTokenizer()
    pretrain_path = tmp_path / "pretrain.jsonl"
    write_jsonl(pretrain_path, [{"text": "abc"}, {"text": "def"}])
    pretrain = PackedPretrainDataset(pretrain_path, tokenizer, sequence_length=8)
    sample = pretrain[0]
    assert sample["input_ids"].shape == (8,)
    assert sample["attention_mask"].sum() > 0

    sft_path = tmp_path / "sft.jsonl"
    write_jsonl(sft_path, [{"prompt": "Question: ", "response": "answer"}])
    sft = SFTDataset(sft_path, tokenizer, sequence_length=32)
    labels = sft[0]["labels"]
    assert (labels == -100).any()
    assert (labels >= 0).any()
