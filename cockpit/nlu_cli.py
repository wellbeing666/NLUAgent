"""模型准备、阈值校准与独立评测：python -m cockpit.nlu_cli --help。"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from cockpit.nlu import mask_temperature
from cockpit.semantic_nlu import (
    DATA_DIR, MODEL_DIR, MODEL_ID, MODEL_REVISION, ROOT,
    LocalEncoder, NLURouter, NLUStartupError, SemanticNLU, read_json,
)
from cockpit.tools import ToolService


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    """只有显式prepare命令联网；正常运行和评测不下载。"""
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    from huggingface_hub import snapshot_download

    print(f"下载 {MODEL_ID}，固定版本 {MODEL_REVISION}，目标 {MODEL_DIR}", flush=True)
    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=str(MODEL_DIR), cache_dir=str(ROOT / ".hf-cache"),
        allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/config.json"],
    )
    required = ["config.json", "model.safetensors", "modules.json", "tokenizer.json", "1_Pooling/config.json"]
    if any(not (MODEL_DIR / name).is_file() for name in required):
        raise NLUStartupError("下载未完成，缺少必要模型文件，请重试prepare。")
    write_json(MODEL_DIR / "nlu-model.json", {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "device": "cpu",
        "weights_sha256": fingerprint(MODEL_DIR / "model.safetensors"),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print("模型已准备好。下一步：python -m cockpit.nlu_cli calibrate", flush=True)


def predicted_label(decision) -> str:
    if decision.action == "execute":
        return decision.intent.name
    return "clarify" if decision.action == "clarify" else "fallback"


def validate_splits() -> None:
    examples = read_json(DATA_DIR / "examples.json")
    calibration = read_json(DATA_DIR / "calibration.json")
    test = read_json(DATA_DIR / "test.json")
    sets = [{s for rows in examples.values() for s in rows},
            {r["text"] for r in calibration}, {r["text"] for r in test}]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("示例、校准、测试包含重复文本")
    if {r["group"] for r in calibration} & {r["group"] for r in test}:
        raise ValueError("校准和测试的模板分组有交叉")
    masked_examples = {mask_temperature(text) for text in sets[0]}
    cal_templates = {mask_temperature(r["text"]) for r in calibration if not r.get("boundary")}
    test_templates = {mask_temperature(r["text"]) for r in test if not r.get("boundary")}
    if masked_examples & (cal_templates | test_templates) or cal_templates & test_templates:
        raise ValueError("去除温度数值后，示例、校准、测试模板有重复；边界回归案例应单独标注boundary")
    if min(map(len, examples.values())) < 12 or min(len(calibration), len(test)) < 40:
        raise ValueError("数据量不足：每个意图至少12个示例，校准与测试各至少40条")


def calibrate() -> None:
    validate_splits()
    start = perf_counter()
    semantic = SemanticNLU(LocalEncoder(), thresholds=read_json(DATA_DIR / "thresholds.json"))
    router = NLURouter(semantic)
    rows = read_json(DATA_DIR / "calibration.json")
    candidates = semantic.score_many([r["text"] for r in rows])
    selected = {}
    # 每个意图分别选阈值。优先最少误接受，其次最多正确本地接受，再选择更严格阈值。
    for intent in ("set_temperature", "query_vehicle"):
        relevant = [(r, c) for r, c in zip(rows, candidates) if c.intent.name == intent]
        best = None
        for score_int in range(50, 102):
            for gap_int in range(0, 22):
                threshold = {"similarity": score_int / 100, "margin": gap_int / 100}
                semantic.thresholds[intent] = threshold
                false_accept = correct_accept = 0
                for row, candidate in relevant:
                    decision = router.recognize(row["text"], candidate)
                    prediction = predicted_label(decision)
                    false_accept += prediction != "fallback" and prediction != row["label"]
                    correct_accept += prediction != "fallback" and prediction == row["label"]
                key = (false_accept, -correct_accept, -threshold["similarity"], -threshold["margin"])
                if best is None or key < best[0]:
                    best = key, threshold.copy(), false_accept, correct_accept
        selected[intent] = best[1]
        semantic.thresholds[intent] = best[1]
        print(f"{intent}: {best[1]}, 校准误接受={best[2]}, 正确本地接受={best[3]}", flush=True)
    metadata = {
        "calibrated": True, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "examples_sha256": fingerprint(DATA_DIR / "examples.json"),
        "calibration_sha256": fingerprint(DATA_DIR / "calibration.json"),
        "calibration_count": len(rows), "used_test_for_selection": False,
        "objective": "min_false_accept_then_max_correct_accept_then_stricter_thresholds",
        "note": "similarity和margin均不是概率；1.01表示该意图语义匹配被关闭。",
    }
    write_json(DATA_DIR / "thresholds.json", {**selected, "metadata": metadata})
    report = {"thresholds": selected, "metadata": metadata,
              "elapsed_seconds": round(perf_counter() - start, 3)}
    write_json(ROOT / "reports" / "nlu-threshold-selection.json", report)
    print("阈值已写入nlu_data/thresholds.json；未使用保留测试集调参。", flush=True)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 3)


def metrics(records: list[dict]) -> dict:
    labels = ["set_temperature", "query_vehicle", "clarify", "fallback"]
    confusion = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for record in records:
        confusion[record["expected"]][record["predicted"]] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        per_class[label] = {"f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0}
    total = len(records)
    accepted = [r for r in records if r["predicted"] != "fallback"]
    correct_local = sum(r["predicted"] == r["expected"] for r in accepted)
    unsafe = [r for r in records if r["expected"] != "set_temperature"]
    mistakes = [r for r in records if r.get("misexecuted")]
    local_positive = [r for r in records if r["expected"] in {"set_temperature", "query_vehicle"}]
    semantic_times = [r["latency_ms"] for r in records if r["decision"]["source"] == "semantic_nlu"]
    return {
        "total": total, "accuracy": round(sum(r["predicted"] == r["expected"] for r in records) / total, 4),
        "macro_f1": round(sum(x["f1"] for x in per_class.values()) / len(labels), 4),
        "intent_macro_f1": round(sum(per_class[k]["f1"] for k in labels[:2]) / 2, 4),
        "local_correct_coverage": round(correct_local / total, 4),
        "in_scope_intent_coverage": round(sum(r["predicted"] == r["expected"] for r in local_positive) / len(local_positive), 4),
        "accepted_precision": round(correct_local / len(accepted), 4) if accepted else 0,
        "false_accept_count": sum(r["predicted"] != r["expected"] for r in accepted),
        "rejection_rate": round(sum(r["predicted"] == "fallback" for r in records) / total, 4),
        "misexecution_count": len(mistakes),
        "misexecution_rate_on_non_write_samples": round(len(mistakes) / len(unsafe), 4) if unsafe else 0,
        "warmed_pipeline_p50_ms": percentile([r["latency_ms"] for r in records], .5),
        "warmed_pipeline_p95_ms": percentile([r["latency_ms"] for r in records], .95),
        "warmed_semantic_p50_ms": percentile(semantic_times, .5),
        "warmed_semantic_p95_ms": percentile(semantic_times, .95),
        "per_class": per_class, "confusion": confusion,
    }


def evaluate(split: str, output: Path, rules_only: bool = False) -> dict:
    validate_splits()
    start = perf_counter()
    router = NLURouter() if rules_only else NLURouter.hybrid()
    cold_start_ms = (perf_counter() - start) * 1000
    router.recognize("查询当前车辆电量")  # 单次预热，不纳入延迟统计。
    records = []
    for row in read_json(DATA_DIR / f"{split}.json"):
        start = perf_counter()
        decision = router.recognize(row["text"])
        latency_ms = (perf_counter() - start) * 1000
        service = ToolService()  # 每条评测使用独立初始车辆状态。
        result = None
        if decision.action == "execute":
            result = service.execute(decision.intent.tool, decision.intent.arguments,
                                     allow_write=decision.intent.name == "set_temperature")
        misexecuted = bool(result and result["ok"] and decision.intent.name == "set_temperature"
                           and row["label"] != "set_temperature")
        records.append({"text": row["text"], "group": row["group"], "expected": row["label"],
                        "predicted": predicted_label(decision), "latency_ms": round(latency_ms, 3),
                        "decision": decision.trace_data(), "tool_result": result, "misexecuted": misexecuted})
    report = {"split": split, "backend": "rules" if rules_only else "hybrid_cpu",
              "model_id": None if rules_only else MODEL_ID, "model_revision": None if rules_only else MODEL_REVISION,
              "cold_start_ms": round(cold_start_ms, 3), "thresholds": None if rules_only else router.semantic.thresholds,
              "data_sha256": fingerprint(DATA_DIR / f"{split}.json"),
              "summary": metrics(records), "records": records}
    write_json(output, report)
    print(json.dumps({"report": str(output), "cold_start_ms": report["cold_start_ms"],
                      **report["summary"]}, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="本地BGE NLU：准备模型、校准阈值、独立评测")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="显式联网下载固定版本模型")
    sub.add_parser("calibrate", help="仅使用校准集选择阈值")
    evaluation = sub.add_parser("evaluate", help="评测真实本地模型，不访问网络")
    evaluation.add_argument("--split", choices=["calibration", "test"], default="test")
    evaluation.add_argument("--rules-only", action="store_true")
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "calibrate":
            calibrate()
        else:
            output = args.output or ROOT / "reports" / f"nlu-{args.split}{'-rules' if args.rules_only else ''}.json"
            evaluate(args.split, output, args.rules_only)
    except (NLUStartupError, ImportError, OSError, ValueError) as exc:
        print(f"NLU命令失败：{type(exc).__name__}。{exc if isinstance(exc, (NLUStartupError, ValueError)) else '请检查依赖、模型文件和网络（仅prepare需要联网）。'}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
