# 模块说明：
# - 旧 external RAG 的 embedding 检索器。
# - 与 bundled skill 内置检索是两套体系。

"""Semantic retrieval for case-based RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    AIHUBMIX_API_KEY,
    AIHUBMIX_BASE_URL,
    DATA_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_CORPUS_DIR,
    RETRIEVAL_DB_DIR,
)
from src.retriever.retrieval_text_builder import (
    build_case_retrieval_artifacts,
    build_query_retrieval_text,
    conversation_to_user_only_text,
)
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path


@dataclass
class RetrievalResult:
    sample_id: str
    score: float
    conversation: List[Dict[str, str]]
    is_minor: bool
    icbo_features: Dict[str, Any]
    user_persona: Dict[str, Any]
    source: str


class SemanticRetriever:
    """Retrieve similar dialogue cases with embedding vectors."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        embedding_model: str = EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        base_url: str = AIHUBMIX_BASE_URL,
    ):
        self.index_path = Path(index_path) if index_path else RETRIEVAL_DB_DIR / "index.pkl"
        self.corpus_path = Path(corpus_path) if corpus_path else RETRIEVAL_CORPUS_DIR / "cases_v1.jsonl"
        self.manifest_path = Path(manifest_path) if manifest_path else RETRIEVAL_DB_DIR / "manifest.json"
        self.embedding_model = embedding_model
        self.api_key = api_key or AIHUBMIX_API_KEY
        self.base_url = base_url.rstrip("/")
        self.text_template_version = "user_only_with_time_scene_v2"
        self.preprocess_version = "basic_whitespace_time_tags_v2"

        self.embeddings: Optional[np.ndarray] = None
        self.samples: List[Dict[str, Any]] = []
        self.manifest: Dict[str, Any] = {}

        if self.index_path.exists():
            self.load_index()

    def _get_embedding(self, text: str) -> np.ndarray:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": text,
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        embedding = data["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings: List[List[float]] = []
        i = 0
        current_batch_size = max(1, batch_size)

        while i < len(texts):
            batch = texts[i:i + current_batch_size]
            batch_embeddings = self._request_embeddings_with_retry(batch, start_index=i)
            all_embeddings.extend(batch_embeddings)
            i += len(batch)
            print(f"  Embedded {min(i, len(texts))}/{len(texts)}")

        return np.array(all_embeddings, dtype=np.float32)

    def _request_embeddings_with_retry(
        self,
        batch: List[str],
        start_index: int = 0,
        max_retries: int = 3,
    ) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            payload = {
                "model": self.embedding_model,
                "input": batch,
            }
            try:
                with httpx.Client(timeout=90.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
            except (httpx.HTTPError, httpx.RemoteProtocolError, ValueError) as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    wait_time = 1.5 * (2 ** attempt)
                    print(
                        f"  Embedding batch retry {attempt + 1}/{max_retries - 1} "
                        f"(batch={len(batch)}, offset={start_index}) in {wait_time:.1f}s: {exc}"
                    )
                    time.sleep(wait_time)

        if len(batch) > 1:
            mid = len(batch) // 2
            print(f"  Split embedding batch {len(batch)} -> {mid}+{len(batch) - mid} at offset={start_index}")
            left = self._request_embeddings_with_retry(
                batch[:mid],
                start_index=start_index,
                max_retries=max_retries,
            )
            right = self._request_embeddings_with_retry(
                batch[mid:],
                start_index=start_index + mid,
                max_retries=max_retries,
            )
            return left + right

        raise RuntimeError(f"Embedding request failed at offset={start_index}: {last_error}")

    def _conversation_to_text(self, conversation: List[Dict[str, str]]) -> str:
        return conversation_to_user_only_text(conversation)

    def _build_case_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        conversation = sample.get("conversation", [])
        user_persona = sample.get("user_persona", {})
        retrieval_artifacts = build_case_retrieval_artifacts(sample)
        user_turn_count = sum(1 for turn in conversation if turn.get("role") == "user")
        return {
            "source": sample.get("source", "unknown"),
            "is_minor": sample.get("is_minor"),
            "user_turn_count": user_turn_count,
            "total_turn_count": len(conversation),
            "age": user_persona.get("age"),
            "age_range": user_persona.get("age_range"),
            "education_stage": user_persona.get("education_stage"),
            "identity_markers": user_persona.get("identity_markers", []),
            "raw_time_hint": retrieval_artifacts["raw_time_hint"],
            "time_features": retrieval_artifacts["time_features"],
        }

    def _build_case_record(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(sample)
        retrieval_artifacts = build_case_retrieval_artifacts(sample)
        record["_embedding_text"] = retrieval_artifacts["embedding_text"]
        record["_time_features"] = retrieval_artifacts["time_features"]
        record["_raw_time_hint"] = retrieval_artifacts["raw_time_hint"]
        record["_scene_tags"] = retrieval_artifacts["scene_tags"]
        record["_metadata"] = self._build_case_metadata(sample)
        return record

    def _compute_dataset_hash(self, records: List[Dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for record in records:
            digest.update(str(record.get("sample_id", "")).encode("utf-8"))
            digest.update(b"\n")
            digest.update(record.get("_embedding_text", "").encode("utf-8"))
            digest.update(b"\n")
            digest.update(str(record.get("is_minor", "")).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _write_corpus(self, records: List[Dict[str, Any]]):
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for record in records:
                corpus_entry = {
                    "sample_id": record.get("sample_id"),
                    "text": record.get("_embedding_text", ""),
                    "metadata": record.get("_metadata", {}),
                    "payload": {
                        "source": record.get("source", "unknown"),
                        "is_minor": record.get("is_minor"),
                        "conversation": record.get("conversation", []),
                        "icbo_features": record.get("icbo_features", {}),
                        "time_features": record.get("_time_features", {}),
                        "user_persona": record.get("user_persona", {}),
                    },
                }
                f.write(json.dumps(corpus_entry, ensure_ascii=False) + "\n")

    def _build_manifest(self, source_path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        manifest_dir = self.manifest_path.parent
        return {
            "built_at": datetime.now().isoformat(),
            "source_data_path": to_relative_posix_path(source_path, manifest_dir),
            "index_path": to_relative_posix_path(self.index_path, manifest_dir),
            "corpus_path": to_relative_posix_path(self.corpus_path, manifest_dir),
            "embedding_model": self.embedding_model,
            "text_template_version": self.text_template_version,
            "preprocess_version": self.preprocess_version,
            "sample_count": len(records),
            "dataset_hash": self._compute_dataset_hash(records),
        }

    def build_index(
        self,
        data_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        save: bool = True,
    ) -> int:
        if data_path is None:
            data_path = DATA_DIR / "benchmark" / "train.jsonl"
        source_path = Path(data_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Missing retrieval source: {source_path}")

        print(f"Loading retrieval source: {source_path}")

        records: List[Dict[str, Any]] = []
        texts: List[str] = []

        with open(source_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)
                record = self._build_case_record(sample)
                if not record["_embedding_text"]:
                    continue

                records.append(record)
                texts.append(record["_embedding_text"])

        print(f"Loaded {len(records)} retrieval cases")

        if not records:
            self.embeddings = None
            self.samples = []
            self.manifest = self._build_manifest(source_path, records)
            if save:
                self._write_corpus(records)
                self.save_index()
            return 0

        print(f"Generating embeddings (model: {self.embedding_model})...")
        self.embeddings = self._get_embeddings_batch(texts)
        self.samples = records
        self.manifest = self._build_manifest(source_path, records)

        print(f"Index built: {self.embeddings.shape}")

        if save:
            self._write_corpus(records)
            self.save_index()

        return len(records)

    def save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        index_data = {
            "embeddings": self.embeddings,
            "samples": self.samples,
            "embedding_model": self.embedding_model,
            "manifest": self.manifest,
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(index_data, f)

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                normalize_project_paths(self.manifest, project_root=DATA_DIR.parent, start=self.manifest_path.parent),
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved retrieval index: {self.index_path}")
        print(f"Saved retrieval manifest: {self.manifest_path}")

    def load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing retrieval index: {self.index_path}")

        with open(self.index_path, "rb") as f:
            index_data = pickle.load(f)

        self.embeddings = index_data["embeddings"]
        self.samples = index_data["samples"]
        self.manifest = index_data.get("manifest", {})

        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        saved_model = index_data.get("embedding_model")
        if saved_model and saved_model != self.embedding_model:
            print(
                f"Warning: index built with embedding model {saved_model}, "
                f"current config uses {self.embedding_model}"
            )

        print(f"Loaded retrieval index with {len(self.samples)} cases")

    def retrieve(
        self,
        conversation: List[Dict[str, str]],
        top_k: int = 3,
        threshold: float = 0.0,
        raw_time_hint: str = "",
        time_features: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if self.embeddings is None or len(self.samples) == 0:
            print("Warning: retrieval index is empty")
            return []

        query_text = build_query_retrieval_text(
            conversation,
            raw_time_hint=raw_time_hint,
            time_features=time_features,
        )
        if not query_text:
            return []

        query_embedding = self._get_embedding(query_text)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        index_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(index_norms, query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < threshold:
                continue

            sample = self.samples[idx]
            result = RetrievalResult(
                sample_id=sample.get("sample_id", f"sample_{idx}"),
                score=score,
                conversation=sample.get("conversation", []),
                is_minor=sample.get("is_minor", True),
                icbo_features=sample.get("icbo_features", {}),
                user_persona=sample.get("user_persona", {}),
                source=sample.get("source", "unknown"),
            )
            results.append(result)

        return results

    def format_for_prompt(
        self,
        results: List[RetrievalResult],
        include_icbo: bool = True,
        include_label: bool = True,
    ) -> str:
        if not results:
            return ""

        parts = ["## Retrieved Similar Cases\n"]

        if include_label:
            minor_count = sum(1 for r in results if r.is_minor)
            adult_count = len(results) - minor_count
            ages = []
            for result in results:
                age = result.user_persona.get("age")
                if isinstance(age, (int, float)):
                    ages.append(int(age))
                elif isinstance(age, str) and age.isdigit():
                    ages.append(int(age))

            parts.append(f"Retrieved {len(results)} similar cases")
            parts.append(f"- Minor cases: {minor_count} | Adult cases: {adult_count}")
            if ages:
                age_min, age_max = min(ages), max(ages)
                if age_min == age_max:
                    parts.append(f"- Ages seen: {age_min}")
                else:
                    parts.append(f"- Ages seen: {age_min}-{age_max}")

            if minor_count > adult_count:
                parts.append("")
                parts.append(
                    "> Similar expressions are more often from minor users. "
                    "Use this as weak evidence only and still judge from the full context."
                )
            elif adult_count > minor_count:
                parts.append("")
                parts.append(
                    "> Similar expressions are more often from adult users. "
                    "Use this as weak evidence only and still judge from the full context."
                )
            else:
                parts.append("")
                parts.append(
                    "> Similar expressions appear in both minor and adult cases. "
                    "Do not over-trust retrieval alone."
                )
            parts.append("")

        parts.append("**Case Details**:")
        for i, result in enumerate(results, 1):
            parts.append(f"### Case {i} (score: {result.score:.3f})")
            conv_preview = result.conversation[:4] if len(result.conversation) > 4 else result.conversation
            conv_text = "\n".join(
                (
                    f"  {turn.get('role', 'user')}: {turn.get('content', '')[:100]}..."
                    if len(turn.get("content", "")) > 100
                    else f"  {turn.get('role', 'user')}: {turn.get('content', '')}"
                )
                for turn in conv_preview
            )
            parts.append(f"Conversation snippet:\n{conv_text}")

            if include_icbo and result.icbo_features:
                icbo_text = ", ".join(
                    (
                        f"{key}: {str(value)[:50]}..."
                        if len(str(value)) > 50
                        else f"{key}: {value}"
                    )
                    for key, value in result.icbo_features.items()
                )
                parts.append(f"ICBO features: {icbo_text}")

            parts.append("")

        return "\n".join(parts)


def build_retrieval_index(max_samples: Optional[int] = None):
    retriever = SemanticRetriever()
    train_path = DATA_DIR / "benchmark" / "train.jsonl"
    if not train_path.exists():
        print("Warning: benchmark train set is missing. Run scripts/prepare_data.py first.")
        return None

    retriever.build_index(str(train_path), max_samples=max_samples)
    return retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or test the retrieval index")
    parser.add_argument("--build", action="store_true", help="Build the retrieval index")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit indexed samples")
    parser.add_argument("--test", action="store_true", help="Run a simple retrieval smoke test")
    args = parser.parse_args()

    if args.build:
        print("Building RAG retrieval index...")
        retriever = build_retrieval_index(max_samples=args.max_samples)
        if retriever:
            print("Retrieval index build finished")

    if args.test:
        print("\nTesting retrieval...")
        retriever = SemanticRetriever()
        test_conversation = [
            {"role": "user", "content": "我大三了，最近一直在准备考研，408 复习压力很大。"},
            {"role": "assistant", "content": "能具体说说你最近的学习安排吗？"},
            {"role": "user", "content": "室友都在找实习，我还在刷题。"},
        ]

        results = retriever.retrieve(test_conversation, top_k=3)
        print(f"\nRetrieved {len(results)} cases")
        for result in results:
            print(f"  - {result.sample_id}: score={result.score:.3f}, is_minor={result.is_minor}")

        formatted = retriever.format_for_prompt(results)
        print("\nFormatted prompt context:")
        print(formatted)
