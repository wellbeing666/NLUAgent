"""本地语义NLU：示例匹配提供候选，确定性检查决定是否能执行。"""

import json
import hashlib
import math
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Protocol

from cockpit.nlu import Intent, mask_temperature, non_local_reason, normalize_text, parse_intent, temperature_request_check

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
DATA_DIR = ROOT / "nlu_data"
PREPARE_HINT = "先安装 requirements-nlu.txt，再运行：python -m cockpit.nlu_cli prepare"


class NLUStartupError(RuntimeError):
    pass


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


@lru_cache(maxsize=1)
def _load_model(path: str):
    # 在导入HF依赖之前关闭联网和遥测；运行时绝不隐式下载。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise NLUStartupError("缺少本地NLU依赖。" + PREPARE_HINT) from exc
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        return SentenceTransformer(path, device="cpu", local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise NLUStartupError("本地模型加载失败，请检查文件完整性及依赖版本。" + PREPARE_HINT) from exc


class LocalEncoder:
    def __init__(self, model_dir: Path = MODEL_DIR):
        required = ["config.json", "model.safetensors", "modules.json", "tokenizer.json", "nlu-model.json"]
        if any(not (model_dir / name).is_file() for name in required):
            raise NLUStartupError("本地BGE模型尚未准备完整。" + PREPARE_HINT)
        manifest = json.loads((model_dir / "nlu-model.json").read_text(encoding="utf-8"))
        if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
            raise NLUStartupError("本地模型清单版本不匹配。" + PREPARE_HINT)
        self.model = _load_model(str(model_dir.resolve()))

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=32,
        ).tolist()


@dataclass(frozen=True)
class NLUDecision:
    action: str = "fallback"
    source: str = "semantic_nlu"
    intent: Intent | None = None
    reason: str = ""
    similarity: float | None = None
    margin: float | None = None
    matched_example: str | None = None
    elapsed_ms: float = 0.0

    def trace_data(self) -> dict:
        return {"source": self.source, "action": self.action,
                "intent": self.intent.name if self.intent else None,
                "arguments": self.intent.arguments if self.intent else {}, "reason": self.reason,
                "similarity": round(self.similarity, 4) if self.similarity is not None else None,
                "margin": round(self.margin, 4) if self.margin is not None else None,
                "matched_example": self.matched_example, "elapsed_ms": round(self.elapsed_ms, 2)}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _unit(vector) -> list[float]:
    vector = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in vector))
    if not norm or not math.isfinite(norm):
        raise ValueError("编码器返回了无效向量")
    return [v / norm for v in vector]


class SemanticNLU:
    def __init__(self, encoder: Encoder, examples: dict | None = None, thresholds: dict | None = None):
        self.encoder = encoder
        self.examples = examples if examples is not None else read_json(DATA_DIR / "examples.json")
        self.thresholds = thresholds if thresholds is not None else read_json(DATA_DIR / "thresholds.json")
        if not isinstance(self.examples, dict) or set(self.examples) != {"set_temperature", "query_vehicle"}:
            raise ValueError("示例库必须包含set_temperature和query_vehicle两个意图")
        if any(not isinstance(rows, list) or not rows or any(not isinstance(x, str) or not x.strip() for x in rows)
               for rows in self.examples.values()):
            raise ValueError("每个意图的示例必须是非空文本列表")
        if not isinstance(self.thresholds, dict):
            raise ValueError("阈值配置必须是对象")
        if thresholds is None and self.thresholds.get("metadata", {}).get("calibrated"):
            metadata = self.thresholds["metadata"]
            digest = hashlib.sha256((DATA_DIR / "examples.json").read_bytes()).hexdigest()
            if metadata.get("examples_sha256") != digest or metadata.get("model_revision") != MODEL_REVISION:
                raise NLUStartupError("示例库或模型版本已变化，请先运行：python -m cockpit.nlu_cli calibrate")
        for name in ("set_temperature", "query_vehicle"):
            config = self.thresholds.get(name, {})
            if not isinstance(config, dict) or any(not isinstance(config.get(key), (int, float)) for key in ["similarity", "margin"]):
                raise ValueError("每个意图需要数值型similarity和margin阈值")
            if not (0 <= config["similarity"] <= 1.01 and 0 <= config["margin"] <= 2):
                raise ValueError("NLU阈值超出允许范围")
        self.rows = [(name, sentence) for name, sentences in self.examples.items() for sentence in sentences]
        self.vectors = [_unit(v) for v in encoder.encode([mask_temperature(text) for _, text in self.rows])]
        if len(self.vectors) != len(self.rows) or len({len(v) for v in self.vectors}) != 1:
            raise ValueError("示例向量数量或维度不一致")

    def score_many(self, texts: list[str]) -> list[NLUDecision]:
        vectors = self.encoder.encode([mask_temperature(text) for text in texts])
        if len(vectors) != len(texts):
            raise ValueError("查询向量数量不一致")
        results = []
        for vector in vectors:
            vector = _unit(vector)
            if len(vector) != len(self.vectors[0]):
                raise ValueError("查询和示例向量维度不一致")
            best = {}
            for (name, example), sample in zip(self.rows, self.vectors):
                score = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, sample))))
                if name not in best or score > best[name][0]:
                    best[name] = score, example
            ranking = sorted(best, key=lambda name: (-best[name][0], name))
            top, second = ranking
            results.append(NLUDecision(
                intent=Intent(top, "set_temperature" if top == "set_temperature" else "get_vehicle_state", {}),
                similarity=best[top][0], margin=best[top][0] - best[second][0], matched_example=best[top][1],
            ))
        return results

    def decide(self, text: str, candidate: NLUDecision) -> NLUDecision:
        reason = non_local_reason(text)
        if reason:
            return replace(candidate, reason=reason)
        name = candidate.intent.name
        config = self.thresholds[name]
        if candidate.similarity < config["similarity"] or candidate.margin < config["margin"]:
            return replace(candidate, reason="below_similarity_or_margin_threshold")
        if name == "set_temperature":
            action, value, reason = temperature_request_check(text)
            args = {"temperature": value} if value is not None else {}
            return replace(candidate, action=action, reason=reason, intent=Intent(name, "set_temperature", args))
        return replace(candidate, action="execute", reason="semantic_vehicle_query")

    def recognize(self, text: str) -> NLUDecision:
        start = perf_counter()
        result = self.decide(text, self.score_many([text])[0])
        return replace(result, elapsed_ms=(perf_counter() - start) * 1000)


@lru_cache(maxsize=1)
def _default_semantic() -> SemanticNLU:
    # CLI中多个应用共享模型与示例向量；修改示例或阈值后需重启进程。
    return SemanticNLU(LocalEncoder())


class NLURouter:
    def __init__(self, semantic: SemanticNLU | None = None):
        self.semantic = semantic

    @classmethod
    def hybrid(cls):
        return cls(_default_semantic())

    def recognize(self, text: str, candidate: NLUDecision | None = None) -> NLUDecision:
        start = perf_counter()
        if len(text) > 512:
            return NLUDecision(source="nlu_guard", reason="input_too_long")
        intent = parse_intent(text)
        if intent:
            return NLUDecision(action="execute", source="rule_nlu", intent=intent, reason="exact_rule_match",
                               elapsed_ms=(perf_counter() - start) * 1000)
        # 向量库仍只有温度/状态两类，不能把新增续航/记忆请求误吸收成普通状态查询。
        if re.search(r'续航|能.*(?:跑|行驶|开).*(?:远|公里)|行驶记录|行驶历史|历史能耗|历史.*耗电', normalize_text(text)):
            return NLUDecision(source='nlu_guard', reason='range_or_memory_request')
        if self.semantic is None:
            return NLUDecision(source="rule_nlu", reason="no_rule_match")
        if non_local_reason(text):
            return NLUDecision(source="nlu_guard", reason=non_local_reason(text))
        # 常见的不完整温度请求明确澄清，不用相似度猜一个温度。
        normalized = normalize_text(text).rstrip("。！!")
        if re.fullmatch(r"(?:我|车里|车内|这里|现在|感觉|觉得|有点儿?|有些|太|很|好|真|比较|稍微)*(?:热|冷)(?:了|啊|呀)?", normalized):
            return NLUDecision(action="clarify", source="nlu_guard", reason="missing_temperature")
        status, _, reason = temperature_request_check(text)
        command = re.sub(r"空调|温度|车内|车里|座舱|驾驶室", "", normalized)
        if status == "clarify" and (
            (re.search(r"空调|温度|车内|车里|座舱|驾驶室", normalized) and re.search(r"设|调|改|弄|升|降", command))
            or re.match(r"再(?:高|低|升|降|调)", normalized)
        ):
            return NLUDecision(action="clarify", source="nlu_guard", reason=reason)
        if candidate is not None:
            return self.semantic.decide(text, candidate)
        return self.semantic.recognize(text)
