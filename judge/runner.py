from __future__ import annotations

import math
import multiprocessing as mp
import time
import traceback
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from judge.problems import PROBLEMS, get_problem_effective


DEFAULT_TIMEOUT_SEC = 20
ATOL = 1e-5
RTOL = 1e-4
MAX_CUSTOM_CASES = 5


def run_problem(
    problem_id: str,
    user_code: str,
    mode: str,
    custom_cases: list[dict[str, Any]] | None = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    if problem_id not in PROBLEMS:
        return {"ok": False, "error": f"Unknown problem_id: {problem_id}"}

    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_worker,
        args=(queue, problem_id, user_code, mode, custom_cases),
        daemon=True,
    )

    process.start()
    process.join(timeout=timeout_sec)

    if process.is_alive():
        process.terminate()
        process.join(1)
        return {
            "ok": True,
            "result": {
                "passed": False,
                "summary": {"total": 0, "passed": 0},
                "cases": [],
                "compile_error": None,
                "runtime_error": f"Execution timed out after {timeout_sec}s.",
            },
        }

    if queue.empty():
        return {"ok": False, "error": "No result returned from worker process."}

    return queue.get()


def _worker(
    queue: mp.Queue,
    problem_id: str,
    user_code: str,
    mode: str,
    custom_cases: list[dict[str, Any]] | None,
) -> None:
    try:
        torch.set_num_threads(1)
        result = _run_judge(problem_id, user_code, mode, custom_cases)
        queue.put({"ok": True, "result": result})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _run_judge(
    problem_id: str,
    user_code: str,
    mode: str,
    custom_cases: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if mode not in {"test", "submit"}:
        raise ValueError("mode must be 'test' or 'submit'")

    problem = get_problem_effective(problem_id)
    if problem is None:
        raise ValueError(f"Unknown problem_id: {problem_id}")
    started = time.perf_counter()

    base_globals: dict[str, Any] = {
        "__builtins__": __builtins__,
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Optional": Optional,
        "List": List,
        "Tuple": Tuple,
    }

    ref_ns = dict(base_globals)
    usr_ns = dict(base_globals)

    exec(problem["reference_code"], ref_ns)
    try:
        exec(user_code, usr_ns)
    except Exception:
        return {
            "passed": False,
            "summary": {"total": 0, "passed": 0},
            "cases": [],
            "compile_error": traceback.format_exc(),
            "runtime_error": None,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    entry_name = problem["entry_name"]
    if entry_name not in usr_ns:
        return {
            "passed": False,
            "summary": {"total": 0, "passed": 0},
            "cases": [],
            "compile_error": f"Cannot find required entry: `{entry_name}`",
            "runtime_error": None,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    ref_target = ref_ns[entry_name]
    usr_target = usr_ns[entry_name]

    cases = _select_cases(problem_id, mode, custom_cases)
    case_results: list[dict[str, Any]] = []
    passed_count = 0

    for case in cases:
        case_name = case["name"]
        case_type = case["case_type"]
        spec = case["spec"]

        try:
            expected, actual = _dispatch_run(problem_id, ref_target, usr_target, spec)
            cmp = _compare_outputs(actual, expected, atol=ATOL, rtol=RTOL)
            passed = cmp["passed"]
            if passed:
                passed_count += 1

            case_result: dict[str, Any] = {
                "name": case_name,
                "type": case_type,
                "passed": passed,
                "message": cmp["message"],
                "max_abs_diff": cmp["max_abs_diff"],
            }
            if mode == "test":
                case_result["expected"] = _preview(expected)
                case_result["actual"] = _preview(actual)
            case_results.append(case_result)
        except Exception:
            case_results.append(
                {
                    "name": case_name,
                    "type": case_type,
                    "passed": False,
                    "message": "Runtime error in this case.",
                    "max_abs_diff": None,
                    "traceback": traceback.format_exc(),
                }
            )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    total = len(cases)
    return {
        "passed": passed_count == total and total > 0,
        "summary": {"total": total, "passed": passed_count},
        "cases": case_results,
        "compile_error": None,
        "runtime_error": None,
        "elapsed_ms": elapsed_ms,
    }


def _select_cases(
    problem_id: str, mode: str, custom_cases: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    problem = PROBLEMS[problem_id]
    cases: list[dict[str, Any]] = []

    if mode == "test":
        for case in problem["visible_cases"]:
            cases.append(
                {
                    "name": case["name"],
                    "spec": case["spec"],
                    "case_type": "visible",
                }
            )

        if custom_cases:
            if len(custom_cases) > MAX_CUSTOM_CASES:
                raise ValueError(f"Too many custom cases. Max={MAX_CUSTOM_CASES}")
            for idx, custom_spec in enumerate(custom_cases, start=1):
                sanitized = _sanitize_custom_case(problem_id, custom_spec)
                cases.append(
                    {
                        "name": f"custom-{idx}",
                        "spec": sanitized,
                        "case_type": "custom",
                    }
                )
    else:
        for case in problem["hidden_cases"]:
            cases.append(
                {
                    "name": case["name"],
                    "spec": case["spec"],
                    "case_type": "hidden",
                }
            )

    return cases


def _sanitize_custom_case(problem_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("Each custom case must be a JSON object.")

    if problem_id == "scaled-dot-product-attention":
        spec = {
            "batch": _as_int(raw.get("batch", 1), min_value=1),
            "heads": _as_int(raw.get("heads", 4), min_value=1),
            "q_len": _as_int(raw.get("q_len", 8), min_value=1),
            "k_len": _as_int(raw.get("k_len", 8), min_value=1),
            "head_dim": _as_int(raw.get("head_dim", 16), min_value=1),
            "use_mask": bool(raw.get("use_mask", True)),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }
        return spec

    if problem_id == "multi-head-self-attention-forward":
        d_model = _as_int(raw.get("d_model", 32), min_value=1)
        num_heads = _as_int(raw.get("num_heads", 4), min_value=1)
        if d_model % num_heads != 0:
            raise ValueError("num_heads must divide d_model.")
        spec = {
            "batch": _as_int(raw.get("batch", 1), min_value=1),
            "seq_len": _as_int(raw.get("seq_len", 8), min_value=1),
            "d_model": d_model,
            "num_heads": num_heads,
            "dropout": _as_float(raw.get("dropout", 0.0), min_value=0.0, max_value=1.0),
            "use_mask": bool(raw.get("use_mask", True)),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }
        return spec

    if problem_id == "transformer-encoder-block-forward":
        d_model = _as_int(raw.get("d_model", 32), min_value=1)
        num_heads = _as_int(raw.get("num_heads", 4), min_value=1)
        if d_model % num_heads != 0:
            raise ValueError("num_heads must divide d_model.")
        spec = {
            "batch": _as_int(raw.get("batch", 1), min_value=1),
            "seq_len": _as_int(raw.get("seq_len", 8), min_value=1),
            "d_model": d_model,
            "num_heads": num_heads,
            "mlp_ratio": _as_int(raw.get("mlp_ratio", 4), min_value=1),
            "dropout": _as_float(raw.get("dropout", 0.0), min_value=0.0, max_value=1.0),
            "use_mask": bool(raw.get("use_mask", True)),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }
        return spec

    if problem_id == "lenet5-forward":
        return {
            "batch": _as_int(raw.get("batch", 2), min_value=1),
            "num_classes": _as_int(raw.get("num_classes", 10), min_value=1),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }

    if problem_id == "rmsnorm-forward":
        return {
            "batch": _as_int(raw.get("batch", 2), min_value=1),
            "seq_len": _as_int(raw.get("seq_len", 8), min_value=1),
            "dim": _as_int(raw.get("dim", 32), min_value=1),
            "eps": _as_float(raw.get("eps", 1e-6), min_value=1e-12, max_value=1e-1),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }

    if problem_id == "apply-rope":
        dim = _as_int(raw.get("dim", 16), min_value=2)
        if dim % 2 != 0:
            raise ValueError("dim must be even for RoPE.")
        style = str(raw.get("broadcast_style", "2d")).lower()
        if style not in {"2d", "4d"}:
            raise ValueError("broadcast_style must be '2d' or '4d'.")
        return {
            "batch": _as_int(raw.get("batch", 1), min_value=1),
            "heads": _as_int(raw.get("heads", 4), min_value=1),
            "seq_len": _as_int(raw.get("seq_len", 8), min_value=1),
            "dim": dim,
            "broadcast_style": style,
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }

    if problem_id == "layernorm-forward":
        return {
            "batch": _as_int(raw.get("batch", 2), min_value=1),
            "seq_len": _as_int(raw.get("seq_len", 8), min_value=1),
            "dim": _as_int(raw.get("dim", 32), min_value=1),
            "eps": _as_float(raw.get("eps", 1e-5), min_value=1e-12, max_value=1e-1),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }

    if problem_id == "swiglu-ffn-forward":
        return {
            "batch": _as_int(raw.get("batch", 2), min_value=1),
            "seq_len": _as_int(raw.get("seq_len", 8), min_value=1),
            "d_model": _as_int(raw.get("d_model", 32), min_value=1),
            "hidden_dim": _as_int(raw.get("hidden_dim", 128), min_value=1),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }

    if problem_id == "lora-linear-forward":
        rank = _as_int(raw.get("rank", 4), min_value=1)
        return {
            "batch": _as_int(raw.get("batch", 2), min_value=1),
            "seq_len": _as_int(raw.get("seq_len", 8), min_value=1),
            "in_features": _as_int(raw.get("in_features", 32), min_value=1),
            "out_features": _as_int(raw.get("out_features", 32), min_value=1),
            "rank": rank,
            "alpha": _as_float(raw.get("alpha", 8.0), min_value=1e-6, max_value=1e6),
            "bias": bool(raw.get("bias", True)),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }

    if problem_id == "decode-step-attention-with-kv-cache":
        return {
            "batch": _as_int(raw.get("batch", 1), min_value=1),
            "heads": _as_int(raw.get("heads", 4), min_value=1),
            "cache_len": _as_int(raw.get("cache_len", 8), min_value=0),
            "head_dim": _as_int(raw.get("head_dim", 16), min_value=1),
            "seed": _as_int(raw.get("seed", 0), min_value=0),
        }

    raise ValueError(f"Unsupported problem_id: {problem_id}")


def _dispatch_run(
    problem_id: str,
    ref_target: Any,
    usr_target: Any,
    spec: dict[str, Any],
) -> tuple[Any, Any]:
    if problem_id == "scaled-dot-product-attention":
        return _run_scaled_dot_case(ref_target, usr_target, spec)
    if problem_id == "multi-head-self-attention-forward":
        return _run_mhsa_case(ref_target, usr_target, spec)
    if problem_id == "transformer-encoder-block-forward":
        return _run_transformer_block_case(ref_target, usr_target, spec)
    if problem_id == "lenet5-forward":
        return _run_lenet_case(ref_target, usr_target, spec)
    if problem_id == "rmsnorm-forward":
        return _run_rmsnorm_case(ref_target, usr_target, spec)
    if problem_id == "apply-rope":
        return _run_rope_case(ref_target, usr_target, spec)
    if problem_id == "layernorm-forward":
        return _run_layernorm_case(ref_target, usr_target, spec)
    if problem_id == "swiglu-ffn-forward":
        return _run_swiglu_case(ref_target, usr_target, spec)
    if problem_id == "lora-linear-forward":
        return _run_lora_linear_case(ref_target, usr_target, spec)
    if problem_id == "decode-step-attention-with-kv-cache":
        return _run_decode_step_attn_case(ref_target, usr_target, spec)

    raise ValueError(f"Unsupported problem_id: {problem_id}")


def _run_scaled_dot_case(ref_fn: Any, usr_fn: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    g = _make_generator(spec["seed"])
    q = torch.randn(spec["batch"], spec["heads"], spec["q_len"], spec["head_dim"], generator=g)
    k = torch.randn(spec["batch"], spec["heads"], spec["k_len"], spec["head_dim"], generator=g)
    v = torch.randn(spec["batch"], spec["heads"], spec["k_len"], spec["head_dim"], generator=g)
    mask = (
        _make_attention_mask(spec["batch"], spec["q_len"], spec["k_len"], spec["seed"] + 999)
        if spec["use_mask"]
        else None
    )

    with torch.no_grad():
        expected = ref_fn(q.clone(), k.clone(), v.clone(), None if mask is None else mask.clone())
        actual = usr_fn(q.clone(), k.clone(), v.clone(), None if mask is None else mask.clone())
    return expected, actual


def _run_mhsa_case(ref_cls: Any, usr_cls: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    ref_model = ref_cls(spec["d_model"], spec["num_heads"], spec["dropout"])
    usr_model = usr_cls(spec["d_model"], spec["num_heads"], spec["dropout"])
    _sync_state_dict(usr_model, ref_model)

    g = _make_generator(spec["seed"] + 17)
    x = torch.randn(spec["batch"], spec["seq_len"], spec["d_model"], generator=g)
    mask = (
        _make_attention_mask(spec["batch"], spec["seq_len"], spec["seq_len"], spec["seed"] + 777)
        if spec["use_mask"]
        else None
    )

    ref_model.eval()
    usr_model.eval()
    with torch.no_grad():
        expected = ref_model(x.clone(), None if mask is None else mask.clone())
        actual = usr_model(x.clone(), None if mask is None else mask.clone())
    return expected, actual


def _run_transformer_block_case(ref_cls: Any, usr_cls: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    ref_model = ref_cls(spec["d_model"], spec["num_heads"], spec["mlp_ratio"], spec["dropout"])
    usr_model = usr_cls(spec["d_model"], spec["num_heads"], spec["mlp_ratio"], spec["dropout"])
    _sync_state_dict(usr_model, ref_model)

    g = _make_generator(spec["seed"] + 33)
    x = torch.randn(spec["batch"], spec["seq_len"], spec["d_model"], generator=g)
    mask = (
        _make_attention_mask(spec["batch"], spec["seq_len"], spec["seq_len"], spec["seed"] + 111)
        if spec["use_mask"]
        else None
    )

    ref_model.eval()
    usr_model.eval()
    with torch.no_grad():
        expected = ref_model(x.clone(), None if mask is None else mask.clone())
        actual = usr_model(x.clone(), None if mask is None else mask.clone())
    return expected, actual


def _run_lenet_case(ref_cls: Any, usr_cls: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    ref_model = ref_cls(spec["num_classes"])
    usr_model = usr_cls(spec["num_classes"])
    _sync_state_dict(usr_model, ref_model)

    g = _make_generator(spec["seed"] + 1)
    x = torch.randn(spec["batch"], 1, 32, 32, generator=g)

    ref_model.eval()
    usr_model.eval()
    with torch.no_grad():
        expected = ref_model(x.clone())
        actual = usr_model(x.clone())
    return expected, actual


def _run_rmsnorm_case(ref_cls: Any, usr_cls: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    ref_model = ref_cls(spec["dim"], spec["eps"])
    usr_model = usr_cls(spec["dim"], spec["eps"])
    _sync_state_dict(usr_model, ref_model)

    g = _make_generator(spec["seed"] + 5)
    x = torch.randn(spec["batch"], spec["seq_len"], spec["dim"], generator=g)

    ref_model.eval()
    usr_model.eval()
    with torch.no_grad():
        expected = ref_model(x.clone())
        actual = usr_model(x.clone())
    return expected, actual


def _run_rope_case(ref_fn: Any, usr_fn: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    g = _make_generator(spec["seed"] + 2)
    x = torch.randn(spec["batch"], spec["heads"], spec["seq_len"], spec["dim"], generator=g)
    half = spec["dim"] // 2
    angles = torch.randn(spec["seq_len"], half, generator=g)
    cos = angles.cos()
    sin = angles.sin()

    if spec["broadcast_style"] == "4d":
        cos = cos.unsqueeze(0).unsqueeze(0).expand(spec["batch"], spec["heads"], -1, -1).clone()
        sin = sin.unsqueeze(0).unsqueeze(0).expand(spec["batch"], spec["heads"], -1, -1).clone()

    with torch.no_grad():
        expected = ref_fn(x.clone(), cos.clone(), sin.clone())
        actual = usr_fn(x.clone(), cos.clone(), sin.clone())
    return expected, actual


def _run_layernorm_case(ref_cls: Any, usr_cls: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    ref_model = ref_cls(spec["dim"], spec["eps"])
    usr_model = usr_cls(spec["dim"], spec["eps"])
    _sync_state_dict(usr_model, ref_model)

    g = _make_generator(spec["seed"] + 41)
    x = torch.randn(spec["batch"], spec["seq_len"], spec["dim"], generator=g)

    ref_model.eval()
    usr_model.eval()
    with torch.no_grad():
        expected = ref_model(x.clone())
        actual = usr_model(x.clone())
    return expected, actual


def _run_swiglu_case(ref_cls: Any, usr_cls: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    ref_model = ref_cls(spec["d_model"], spec["hidden_dim"])
    usr_model = usr_cls(spec["d_model"], spec["hidden_dim"])
    _sync_state_dict(usr_model, ref_model)

    g = _make_generator(spec["seed"] + 53)
    x = torch.randn(spec["batch"], spec["seq_len"], spec["d_model"], generator=g)

    ref_model.eval()
    usr_model.eval()
    with torch.no_grad():
        expected = ref_model(x.clone())
        actual = usr_model(x.clone())
    return expected, actual


def _run_lora_linear_case(ref_cls: Any, usr_cls: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    ref_model = ref_cls(
        spec["in_features"],
        spec["out_features"],
        spec["rank"],
        spec["alpha"],
        spec["bias"],
    )
    usr_model = usr_cls(
        spec["in_features"],
        spec["out_features"],
        spec["rank"],
        spec["alpha"],
        spec["bias"],
    )
    _sync_state_dict(usr_model, ref_model)

    g = _make_generator(spec["seed"] + 67)
    x = torch.randn(spec["batch"], spec["seq_len"], spec["in_features"], generator=g)

    ref_model.eval()
    usr_model.eval()
    with torch.no_grad():
        expected = ref_model(x.clone())
        actual = usr_model(x.clone())
    return expected, actual


def _run_decode_step_attn_case(ref_fn: Any, usr_fn: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    g = _make_generator(spec["seed"] + 79)
    b = spec["batch"]
    h = spec["heads"]
    t = spec["cache_len"]
    d = spec["head_dim"]

    q_t = torch.randn(b, h, 1, d, generator=g)
    k_cache = torch.randn(b, h, t, d, generator=g)
    v_cache = torch.randn(b, h, t, d, generator=g)
    k_t = torch.randn(b, h, 1, d, generator=g)
    v_t = torch.randn(b, h, 1, d, generator=g)

    with torch.no_grad():
        expected = ref_fn(q_t.clone(), k_cache.clone(), v_cache.clone(), k_t.clone(), v_t.clone())
        actual = usr_fn(q_t.clone(), k_cache.clone(), v_cache.clone(), k_t.clone(), v_t.clone())
    return expected, actual


def _sync_state_dict(user_model: nn.Module, ref_model: nn.Module) -> None:
    user_state = user_model.state_dict()
    ref_state = ref_model.state_dict()

    if set(user_state.keys()) != set(ref_state.keys()):
        missing = sorted(set(ref_state.keys()) - set(user_state.keys()))
        extra = sorted(set(user_state.keys()) - set(ref_state.keys()))
        raise ValueError(
            "state_dict keys mismatch. "
            f"missing={missing[:5]} extra={extra[:5]} (showing up to 5 each)"
        )

    for key, ref_value in ref_state.items():
        user_value = user_state[key]
        if tuple(user_value.shape) != tuple(ref_value.shape):
            raise ValueError(
                f"Parameter shape mismatch for key '{key}': "
                f"user={tuple(user_value.shape)} ref={tuple(ref_value.shape)}"
            )

    user_model.load_state_dict(ref_state, strict=True)


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _make_attention_mask(batch: int, q_len: int, k_len: int, seed: int) -> torch.Tensor:
    g = _make_generator(seed)
    mask = torch.rand(batch, 1, q_len, k_len, generator=g) > 0.2

    if q_len == k_len:
        causal = torch.tril(torch.ones(q_len, k_len, dtype=torch.bool))
        mask = mask & causal.view(1, 1, q_len, k_len)

    # Ensure each query position has at least one valid key.
    mask[..., 0] = True
    return mask


def _compare_outputs(actual: Any, expected: Any, atol: float, rtol: float) -> dict[str, Any]:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return {
                "passed": False,
                "message": f"type mismatch: expected tensor, got {type(actual).__name__}",
                "max_abs_diff": None,
            }
        if expected.shape != actual.shape:
            return {
                "passed": False,
                "message": f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}",
                "max_abs_diff": None,
            }
        if expected.dtype != actual.dtype:
            actual = actual.to(expected.dtype)

        close = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        diff = (actual - expected).abs().max().item() if expected.numel() > 0 else 0.0
        return {
            "passed": bool(close),
            "message": "ok" if close else f"tensor mismatch (atol={atol}, rtol={rtol})",
            "max_abs_diff": float(diff),
        }

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)):
            return {
                "passed": False,
                "message": f"type mismatch: expected {type(expected).__name__}, got {type(actual).__name__}",
                "max_abs_diff": None,
            }
        if len(expected) != len(actual):
            return {
                "passed": False,
                "message": f"length mismatch: expected {len(expected)}, got {len(actual)}",
                "max_abs_diff": None,
            }
        max_diff = 0.0
        for i, (a_i, e_i) in enumerate(zip(actual, expected)):
            cmp = _compare_outputs(a_i, e_i, atol, rtol)
            if not cmp["passed"]:
                return {
                    "passed": False,
                    "message": f"item {i}: {cmp['message']}",
                    "max_abs_diff": cmp["max_abs_diff"],
                }
            if cmp["max_abs_diff"] is not None:
                max_diff = max(max_diff, float(cmp["max_abs_diff"]))
        return {"passed": True, "message": "ok", "max_abs_diff": max_diff}

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return {
                "passed": False,
                "message": f"type mismatch: expected dict, got {type(actual).__name__}",
                "max_abs_diff": None,
            }
        if set(expected.keys()) != set(actual.keys()):
            return {
                "passed": False,
                "message": "dict keys mismatch",
                "max_abs_diff": None,
            }
        max_diff = 0.0
        for key in expected:
            cmp = _compare_outputs(actual[key], expected[key], atol, rtol)
            if not cmp["passed"]:
                return {
                    "passed": False,
                    "message": f"key '{key}': {cmp['message']}",
                    "max_abs_diff": cmp["max_abs_diff"],
                }
            if cmp["max_abs_diff"] is not None:
                max_diff = max(max_diff, float(cmp["max_abs_diff"]))
        return {"passed": True, "message": "ok", "max_abs_diff": max_diff}

    if isinstance(expected, float):
        if not isinstance(actual, (float, int)):
            return {
                "passed": False,
                "message": f"type mismatch: expected float, got {type(actual).__name__}",
                "max_abs_diff": None,
            }
        diff = abs(float(actual) - expected)
        ok = diff <= (atol + rtol * abs(expected))
        return {"passed": ok, "message": "ok" if ok else "float mismatch", "max_abs_diff": diff}

    ok = actual == expected
    return {"passed": ok, "message": "ok" if ok else "value mismatch", "max_abs_diff": None}


def _preview(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1).float()
        take = min(6, flat.numel())
        sample = flat[:take].tolist() if take > 0 else []
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sample": [round(float(x), 6) for x in sample],
        }
    if isinstance(value, (list, tuple)):
        return [_preview(v) for v in value]
    if isinstance(value, dict):
        return {k: _preview(v) for k, v in value.items()}
    return value


def _as_int(value: Any, min_value: int | None = None, max_value: int | None = None) -> int:
    ivalue = int(value)
    if min_value is not None and ivalue < min_value:
        raise ValueError(f"value must be >= {min_value}")
    if max_value is not None and ivalue > max_value:
        raise ValueError(f"value must be <= {max_value}")
    return ivalue


def _as_float(value: Any, min_value: float | None = None, max_value: float | None = None) -> float:
    fvalue = float(value)
    if min_value is not None and fvalue < min_value:
        raise ValueError(f"value must be >= {min_value}")
    if max_value is not None and fvalue > max_value:
        raise ValueError(f"value must be <= {max_value}")
    return fvalue
