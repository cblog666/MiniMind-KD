from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimind_kd.config import load_yaml_config
from minimind_kd.modeling.model import MiniMindKDForCausalLM
from minimind_kd.training.data import PromptDataset
from minimind_kd.training.opd import (
    MultiTeacherOnPolicyDistiller,
    OPDConfig,
    TeacherSpec,
)
from minimind_kd.training.optim import build_optimizer
from minimind_kd.training.supervised import resolve_device, seed_everything

from .common import load_model, load_tokenizer, synchronize_vocabulary


def _load_manifest(path: str | Path) -> list[dict]:
    manifest_path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to load the teacher manifest") from exc
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    teachers = raw.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("Teacher manifest needs a non-empty 'teachers' list")
    for item in teachers:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            item["checkpoint"] = str((manifest_path.parent / checkpoint).resolve())
    return teachers


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-teacher on-policy distillation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teachers", required=True, help="Local YAML teacher manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    args = parser.parse_args()

    model_config, train_config, extra = load_yaml_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    synchronize_vocabulary(model_config, tokenizer)
    device = resolve_device(train_config.device)
    seed_everything(train_config.seed)
    student = load_model(model_config, args.student).to(device)
    specs: list[TeacherSpec] = []
    for item in _load_manifest(args.teachers):
        teacher_model = MiniMindKDForCausalLM.from_pretrained(item["checkpoint"]).to(device)
        specs.append(
            TeacherSpec(
                name=str(item["name"]),
                model=teacher_model,
                domains=set(item.get("domains", ["*"])),
                weight=float(item.get("weight", 1.0)),
            )
        )
    optimizer = build_optimizer(student, train_config)
    distiller = MultiTeacherOnPolicyDistiller(
        student,
        specs,
        tokenizer,
        optimizer,
        OPDConfig(**extra.get("opd", {})),
        device,
    )
    dataset = PromptDataset(args.data)
    records = [dataset[index] for index in range(len(dataset))]
    if not records:
        raise ValueError("OPD prompt dataset is empty")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.steps):
            start = (step * args.prompts_per_step) % len(records)
            batch = [records[(start + offset) % len(records)] for offset in range(args.prompts_per_step)]
            metrics = distiller.step(batch)
            metrics["step"] = step + 1
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)
    student.save_pretrained(output / "final")


if __name__ == "__main__":
    main()
