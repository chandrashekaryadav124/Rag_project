import os
from functools import lru_cache
from typing import Optional

import httpx
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
HF_INFERENCE_URL = (
    f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"
    "/pipeline/feature-extraction"
)

_hf_client: Optional[httpx.Client] = None


def _get_hf_client() -> httpx.Client:
    global _hf_client
    if _hf_client is None:
        _hf_client = httpx.Client(timeout=10.0)
    return _hf_client


def _l2_normalize(vector: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector = vector / norm
    return vector.astype(np.float32).tolist()


@lru_cache(maxsize=1)
def _load_onnx_components() -> tuple[ort.InferenceSession, Tokenizer]:
    model_path = hf_hub_download(
        repo_id=EMBEDDING_MODEL,
        filename="onnx/model.onnx",
    )
    tokenizer_path = hf_hub_download(
        repo_id=EMBEDDING_MODEL,
        filename="tokenizer.json",
    )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(
        os.getenv("ONNX_NUM_THREADS", "1")
    )
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
    tokenizer.no_padding()

    return session, tokenizer


def preload_embeddings() -> None:
    """Warm up local ONNX embedding model when HF Inference API is not configured."""
    if os.getenv("HF_TOKEN", "").strip():
        return
    _load_onnx_components()


def _embed_via_hf_api(text: str, hf_token: str) -> list[float]:
    response = _get_hf_client().post(
        HF_INFERENCE_URL,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text},
    )
    response.raise_for_status()
    payload = response.json()

    array = np.asarray(payload, dtype=np.float32)
    if array.ndim == 2:
        array = array[0]
    elif array.ndim == 3:
        array = array[0].mean(axis=0)

    if array.ndim != 1 or array.shape[0] != 384:
        raise ValueError(f"Unexpected embedding shape from HF API: {array.shape}")

    return _l2_normalize(array)


def _embed_via_onnx(text: str) -> list[float]:
    session, tokenizer = _load_onnx_components()
    encoding = tokenizer.encode(text)

    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoding.type_ids], dtype=np.int64)

    last_hidden_state = session.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]

    mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    summed = np.sum(last_hidden_state * mask, axis=1)
    counts = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
    pooled = (summed / counts)[0]

    return _l2_normalize(pooled)


def embed_query(text: str) -> list[float]:
    """Generate a normalized 384-dim all-MiniLM-L6-v2 embedding for a query."""
    cleaned = text.strip() if text else ""
    if not cleaned:
        raise ValueError("Cannot embed empty query text.")

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        try:
            return _embed_via_hf_api(cleaned, hf_token)
        except Exception as error:
            print("HF Inference API embedding failed, falling back to ONNX:", str(error))

    return _embed_via_onnx(cleaned)
