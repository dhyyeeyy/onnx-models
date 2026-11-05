#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow.parquet as pq
import torch
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT.parent / "FINRL" / "dataset"
OUTPUT_DIR = ROOT / "reports" / "chronos2_benchmark_2026_jan_to_may01_covariates"

TIMEFRAME_SECONDS = {
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

SPECS = {
    "V03H_F002_SELL": {
        "timeframe": "M15",
        "contexts": ["H1", "H4", "D1"],
        "horizon": 32,
        "take_profit_atr": 2.0,
        "stop_loss_atr": 1.0,
        "direction": "SELL",
        "stride": 1,
        "eval_start": "2026-01-01T00:00:00Z",
        "eval_end": "2026-05-01T23:59:59Z",
        "model": ROOT / "models" / "v03h" / "final_rf_sell_model.onnx",
        "features": ROOT / "models" / "v03h" / "feature_order.json",
        "cutoff": 0.5933100533976955,
    },
    "V05H_BUY": {
        "timeframe": "M5",
        "contexts": ["M15", "H1", "H4"],
        "horizon": 6,
        "take_profit_atr": 2.0,
        "stop_loss_atr": 1.5,
        "direction": "BUY",
        "stride": 4,
        "eval_start": "2026-01-01T00:00:00Z",
        "eval_end": "2026-05-01T23:59:59Z",
        "model": ROOT / "models" / "v05h" / "final_rf_buy_model.onnx",
        "features": ROOT / "models" / "v05h" / "feature_order.json",
        "cutoff": 0.9391089677810669,
    },
}


def load_feature_order(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"Invalid feature-order file: {path}")
    return [str(value) for value in values]


def _tick_part_to_minutes(path: Path) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=["timestamp", "bid_price", "ask_price"],
    )
    ticks = table.to_pandas()
    timestamps = pd.to_datetime(ticks["timestamp"], utc=True)
    if not timestamps.is_monotonic_increasing:
        ticks = ticks.assign(timestamp=timestamps).sort_values("timestamp")
        timestamps = ticks["timestamp"]

    bid = pd.to_numeric(ticks["bid_price"], errors="coerce")
    ask = pd.to_numeric(ticks["ask_price"], errors="coerce")
    valid = (
        timestamps.notna()
        & bid.gt(0)
        & ask.gt(0)
        & np.isfinite(bid)
        & np.isfinite(ask)
    )
    frame = pd.DataFrame({
        "timestamp": timestamps[valid],
        "mid": ((bid[valid] + ask[valid]) / 2.0).to_numpy(),
    })
    frame["minute"] = frame["timestamp"].dt.floor("min")
    return (
        frame.groupby("minute", sort=True)["mid"]
        .agg(open="first", high="max", low="min", close="last", volume="size")
    )


def load_minute_bars(dataset_root: Path, years: set[int]) -> pd.DataFrame:
    files = sorted(
        path
        for path in dataset_root.glob("year=*/*/*.parquet")
        if int(path.parts[-3].split("=")[1]) in years
    )
    if not files:
        raise FileNotFoundError(
            f"No year/month Parquet files found under {dataset_root} for {sorted(years)}"
        )

    print(f"Aggregating {len(files)} tick files for {sorted(years)} ...", flush=True)
    partials = []
    for index, path in enumerate(files, start=1):
        partials.append(_tick_part_to_minutes(path))
        if index % 10 == 0 or index == len(files):
            print(f"  tick files: {index}/{len(files)}", flush=True)

    minutes = pd.concat(partials).groupby(level=0, sort=True).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    minutes.index.name = "timestamp"
    return minutes.reset_index()


def resample_bars(minutes: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = minutes.set_index("timestamp")
    grouped = frame.resample(
        f"{TIMEFRAME_SECONDS[timeframe]}s",
        origin="start_day",
        closed="left",
        label="left",
    )
    bars = grouped.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    bars["symbol"] = "XAUUSD"
    bars["timeframe"] = timeframe
    return bars.reset_index()


def build_feature_frame(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    import p3ab_build_core_features as p3ab
    import p3f_advanced_indicator_pack as p3f
    import p3h_market_structure_features as p3h
    import p3h2_live_safe_confirmed_structure as p3h2
    from v05h2_m5_buy_dry_run_scorer import add_live_gap_safety_features

    frame = p3ab.optimize_dtypes(p3ab.add_core_features(bars.copy()))
    frame, _ = p3f.add_advanced_indicators(frame)
    frame, _ = p3h.add_structure_features(frame, timeframe)
    frame, _ = p3h2.add_confirmed_structure(frame, timeframe)
    frame = add_live_gap_safety_features(frame, timeframe)

    # These training features are computed with shift(-1), so their historical
    # values reveal the next bar. Use the live-safe value (unknown -> zero).
    for column in (
        "pre_gap_risk_bars_remaining",
        "pre_gap_risk_score",
        "pre_gap_risk_score_linear",
        "pre_gap_risk_score_exp",
        "is_pre_large_gap_risk",
    ):
        if column in frame:
            frame[column] = 0

    return frame.replace([np.inf, -np.inf], np.nan)


def align_context(
    base: pd.DataFrame,
    context_frames: dict[str, pd.DataFrame],
    context_tfs: list[str],
    feature_order: list[str],
    base_tf: str,
) -> pd.DataFrame:
    decision_time = base["timestamp"] + pd.to_timedelta(
        TIMEFRAME_SECONDS[base_tf],
        unit="s",
    )
    out = base.assign(_decision_time=decision_time).sort_values("_decision_time")

    for timeframe in context_tfs:
        source = context_frames[timeframe]
        prefix = f"{timeframe}_"
        required = {
            name[len(prefix):]
            for name in feature_order
            if name.startswith(prefix)
        }
        required.update({
            "confirmed_last_swing_high",
            "confirmed_last_swing_low",
            "confirmed_structure_trend_code",
            "confirmed_bos_bull",
            "confirmed_bos_bear",
            "confirmed_choch_bull",
            "confirmed_choch_bear",
            "is_feature_row_safe",
            "confirmed_structure_row_safe",
        })
        columns = ["timestamp"] + sorted(
            name for name in required if name in source.columns
        )
        htf = source[columns].copy()
        htf["_available_at"] = htf["timestamp"] + pd.to_timedelta(
            TIMEFRAME_SECONDS[timeframe],
            unit="s",
        )
        htf = htf.drop(columns="timestamp")
        available_at = f"{timeframe}_available_at"
        htf = htf.rename(columns={"_available_at": available_at})
        rename = {
            name: f"{timeframe}_{name}"
            for name in columns
            if name != "timestamp"
        }
        htf = htf.rename(columns=rename).sort_values(available_at)

        out = pd.merge_asof(
            out.sort_values("_decision_time"),
            htf,
            left_on="_decision_time",
            right_on=available_at,
            direction="backward",
            allow_exact_matches=True,
        )
        if (out[available_at] > out["_decision_time"]).fillna(False).any():
            raise RuntimeError(f"{timeframe} context alignment leaked future bars.")
        context_available = out[available_at].notna().astype("int8")
        out[f"{timeframe}_context_fresh"] = context_available
        out[f"{timeframe}_context_feature_safe"] = context_available
        out[f"{timeframe}_context_confirmed_structure_safe"] = context_available

    out["mtf_context_row_safe"] = 1
    return out.sort_values("timestamp").reset_index(drop=True)


def build_model_frame(
    frames: dict[str, pd.DataFrame],
    spec: dict[str, Any],
    feature_order: list[str],
) -> pd.DataFrame:
    from scripts.p3k_mtf_swing_confluence_zones import compute_confluence
    from scripts.p3l_candlestick_pattern_scores import compute_patterns

    base_tf = spec["timeframe"]
    aligned = align_context(
        frames[base_tf],
        frames,
        spec["contexts"],
        feature_order,
        base_tf,
    )
    aligned, _ = compute_confluence(
        aligned,
        base_tf,
        spec["contexts"],
    )
    aligned, _ = compute_patterns(
        aligned,
        base_tf,
    )
    if base_tf == "M15":
        for price, confirmed in (
            ("confirmed_swing_high_price", "confirmed_last_swing_high"),
            ("confirmed_swing_low_price", "confirmed_last_swing_low"),
        ):
            aligned[price] = aligned[price].fillna(aligned[confirmed])
        for column in (
            "mtf_nearest_resistance_distance",
            "mtf_nearest_resistance_distance_atr",
            "mtf_nearest_support_distance",
            "mtf_nearest_support_distance_atr",
        ):
            aligned[column] = aligned[column].fillna(0.0)

    missing = [feature for feature in feature_order if feature not in aligned.columns]
    if missing:
        raise ValueError(
            f"{base_tf} feature parity failed; {len(missing)} columns missing: {missing[:30]}"
        )
    return aligned


def barrier_labels(
    bars: pd.DataFrame,
    horizon: int,
    take_profit_atr: float,
    stop_loss_atr: float,
    direction: str,
) -> np.ndarray:
    close = bars["close"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    atr = bars["atr_14"].to_numpy(dtype=np.float64)
    labels = np.full(len(bars), np.nan, dtype=np.float32)

    for index in range(len(bars) - horizon):
        if not (
            np.isfinite(close[index])
            and np.isfinite(atr[index])
            and atr[index] > 0
        ):
            continue
        if direction == "BUY":
            target = close[index] + take_profit_atr * atr[index]
            stop = close[index] - stop_loss_atr * atr[index]
        else:
            target = close[index] - take_profit_atr * atr[index]
            stop = close[index] + stop_loss_atr * atr[index]

        labels[index] = 0.0
        for future in range(index + 1, index + horizon + 1):
            if direction == "BUY":
                target_hit = high[future] >= target
                stop_hit = low[future] <= stop
            else:
                target_hit = low[future] <= target
                stop_hit = high[future] >= stop

            if target_hit or stop_hit:
                # If both barriers occur in one OHLC bar, assume the stop came first.
                labels[index] = float(target_hit and not stop_hit)
                break

    return labels


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end

    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid]
    labels = labels[valid].astype(np.int8)
    if not len(labels):
        return {"rows": 0}

    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    positives = int(labels.sum())
    cumulative_precision = np.cumsum(ranked_labels) / np.arange(1, len(labels) + 1)
    average_precision = (
        float(cumulative_precision[ranked_labels == 1].sum() / positives)
        if positives
        else None
    )

    top = {}
    for fraction in (0.001, 0.01, 0.03):
        count = max(1, math.ceil(len(labels) * fraction))
        top[f"precision_at_{fraction:g}"] = float(ranked_labels[:count].mean())
        top[f"signals_at_{fraction:g}"] = count

    return {
        "rows": int(len(labels)),
        "positive_rows": positives,
        "positive_rate": float(labels.mean()),
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision,
        **top,
    }


def evaluation_origins(
    timestamp: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
    stride: int,
) -> np.ndarray:
    candidates = np.flatnonzero((timestamp >= start) & (timestamp <= end))
    future_indices = candidates + horizon
    in_data = future_indices < len(timestamp)
    candidates = candidates[in_data]
    future_indices = future_indices[in_data]
    candidates = candidates[(timestamp.iloc[future_indices] <= end).to_numpy()]
    return candidates[::stride]


def onnx_positive_probability(session: ort.InferenceSession, values: np.ndarray) -> np.ndarray:
    model_input = session.get_inputs()[0]
    dtype = np.float64 if "double" in model_input.type else np.float32
    outputs = session.run(None, {model_input.name: values.astype(dtype, copy=False)})

    for output in reversed(outputs):
        if isinstance(output, list) and output and isinstance(output[0], dict):
            for key in (1, "1", True):
                if key in output[0]:
                    return np.asarray([float(row[key]) for row in output])
        array = np.asarray(output)
        if array.ndim == 2 and array.shape[1] >= 2 and array.dtype.kind in "fiu":
            return array[:, 1].astype(np.float64)

    raise RuntimeError("Could not identify positive-class ONNX probabilities.")


def chronos_forecasts(
    pipeline: Chronos2Pipeline,
    closes: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    origins: np.ndarray,
    horizon: int,
    context_length: int,
    batch_size: int,
    origin_batch_size: int,
) -> np.ndarray:
    quantiles: list[np.ndarray] = []
    if feature_values.shape != (len(closes), len(feature_names)):
        raise ValueError(
            "Chronos covariate matrix shape does not match the close series and feature names."
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Chronos covariate feature names must be unique.")
    valid_history = np.isfinite(closes) & np.isfinite(feature_values).all(axis=1)

    for start in range(0, len(origins), origin_batch_size):
        group = origins[start:start + origin_batch_size]
        contexts = []
        for origin_value in group:
            origin = int(origin_value)
            context_start = max(0, origin - context_length + 1)
            invalid = np.flatnonzero(~valid_history[context_start:origin + 1])
            if invalid.size:
                context_start += int(invalid[-1]) + 1
            context_end = origin + 1
            contexts.append({
                "target": closes[context_start:context_end].astype(np.float32),
                "past_covariates": {
                    name: feature_values[context_start:context_end, index].astype(
                        np.float32,
                        copy=False,
                    )
                    for index, name in enumerate(feature_names)
                },
            })
        predicted, _ = pipeline.predict_quantiles(
            contexts,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=batch_size,
            context_length=context_length,
        )
        quantiles.extend(
            forecast[0, :, -1].detach().cpu().numpy()
            for forecast in predicted
        )
        print(
            f"  Chronos forecasts: {min(start + len(group), len(origins))}/{len(origins)} "
            f"(past covariates: {len(feature_names)})",
            flush=True,
        )
    return np.asarray(quantiles, dtype=np.float64)


def evaluate_strategy(
    name: str,
    spec: dict[str, Any],
    aligned: pd.DataFrame,
    feature_order: list[str],
    pipeline: Chronos2Pipeline,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    timestamp = pd.to_datetime(aligned["timestamp"], utc=True)
    start = pd.Timestamp(spec["eval_start"])
    end = pd.Timestamp(spec["eval_end"])
    candidates = evaluation_origins(
        timestamp,
        start,
        end,
        spec["horizon"],
        spec["stride"],
    )
    evaluation_candidates = candidates.copy()

    values = aligned[feature_order].apply(pd.to_numeric, errors="coerce")
    feature_values = values.to_numpy(dtype=np.float64)
    finite = np.isfinite(feature_values).all(axis=1)
    candidates = candidates[finite[candidates]]
    feature_valid_count = len(candidates)
    if args.max_origins:
        candidates = candidates[:args.max_origins]

    labels_all = barrier_labels(
        aligned,
        spec["horizon"],
        spec["take_profit_atr"],
        spec["stop_loss_atr"],
        spec["direction"],
    )
    label_valid = np.isfinite(labels_all[candidates])
    label_valid_count = int(label_valid.sum())
    candidates = candidates[label_valid]
    if not len(candidates):
        missing = (
            values.iloc[evaluation_candidates]
            .isna()
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        raise RuntimeError(
            f"{name}: no valid evaluation origins "
            f"(evaluation={len(evaluation_candidates)}, "
            f"finite_features={feature_valid_count}, "
            f"finite_labels={label_valid_count}); "
            f"most-missing features={missing[missing > 0].to_dict()}"
        )

    close = aligned["close"].to_numpy(dtype=np.float64)
    atr = aligned["atr_14"].to_numpy(dtype=np.float64)
    q = chronos_forecasts(
        pipeline,
        close,
        feature_values,
        feature_order,
        candidates,
        spec["horizon"],
        args.context_length,
        args.batch_size,
        args.origin_batch_size,
    )
    median = q[:, 1]
    forecast_delta_atr = (median - close[candidates]) / atr[candidates]
    chronos_score = forecast_delta_atr if spec["direction"] == "BUY" else -forecast_delta_atr
    forecast_spread_atr = (q[:, 2] - q[:, 0]) / atr[candidates]

    session = ort.InferenceSession(
        str(spec["model"]),
        providers=["CPUExecutionProvider"],
    )
    model_scores = onnx_positive_probability(
        session,
        values.iloc[candidates].to_numpy(dtype=np.float32),
    )
    labels = labels_all[candidates].astype(np.int8)
    direction = spec["direction"]
    chronos_metrics = ranking_metrics(chronos_score, labels)
    model_metrics = ranking_metrics(model_scores, labels)

    published_cutoff_signals = model_scores >= spec["cutoff"]
    published_cutoff_precision = (
        float(labels[published_cutoff_signals].mean())
        if published_cutoff_signals.any()
        else None
    )

    actual_index = candidates + spec["horizon"]
    actual_close = close[actual_index]
    actual_change = actual_close - close[candidates]
    point_error = median - actual_close
    point_mae = float(np.mean(np.abs(point_error)))
    naive_mae = float(np.mean(np.abs(close[candidates] - actual_close)))

    summary = {
        "strategy": name,
        "timeframe": spec["timeframe"],
        "direction": direction,
        "horizon_bars": spec["horizon"],
        "evaluation_start": str(timestamp.iloc[candidates[0]]),
        "evaluation_end": str(timestamp.iloc[candidates[-1]]),
        "sampling_stride_bars": spec["stride"],
        "rows_scored": int(len(candidates)),
        "barrier_label": {
            "target_atr": spec["take_profit_atr"],
            "stop_atr": spec["stop_loss_atr"],
            "successes": int(labels.sum()),
            "success_rate": float(labels.mean()),
            "same_bar_target_stop_treated_as": "stop",
        },
        "chronos2": {
            **chronos_metrics,
            "input": "close target plus model feature_order as past-only covariates",
            "past_covariate_count": len(feature_order),
            "spread_atr_as_success_ranker": ranking_metrics(forecast_spread_atr, labels),
            "median_forecast_mae_price": point_mae,
            "naive_last_close_mae_price": naive_mae,
            "mae_improvement_vs_naive": (
                float(1 - point_mae / naive_mae) if naive_mae else None
            ),
            "median_forecast_directional_accuracy": float(
                np.mean(np.sign(median - close[candidates]) == np.sign(actual_change))
            ),
            "q10_q90_endpoint_coverage": float(
                np.mean((actual_close >= q[:, 0]) & (actual_close <= q[:, 2]))
            ),
        },
        "onnx_model": {
            **model_metrics,
            "fixed_policy_cutoff": spec["cutoff"],
            "fixed_policy_signal_count": int(published_cutoff_signals.sum()),
            "fixed_policy_precision": published_cutoff_precision,
        },
    }

    result = pd.DataFrame({
        "timestamp": timestamp.iloc[candidates].to_numpy(),
        "close": close[candidates],
        "atr_14": atr[candidates],
        "barrier_success": labels,
        "chronos_q10_at_horizon": q[:, 0],
        "chronos_q50_at_horizon": q[:, 1],
        "chronos_q90_at_horizon": q[:, 2],
        "chronos_directional_score_atr": chronos_score,
        "chronos_forecast_spread_atr": forecast_spread_atr,
        "onnx_score": model_scores,
        "onnx_fixed_policy_signal": published_cutoff_signals,
    })
    return summary, result


def self_test() -> None:
    sample = pd.DataFrame({
        "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        "high": [100.0, 103.0, 100.0, 103.0, 100.0],
        "low": [100.0, 100.0, 97.0, 97.0, 100.0],
        "atr_14": [1.0] * 5,
    })
    buy = barrier_labels(sample, 2, 2.0, 1.0, "BUY")
    assert buy[0] == 1.0
    assert buy[1] == 0.0
    assert buy[2] == 0.0
    assert np.isnan(buy[-2])
    assert ranking_metrics(np.array([0.1, 0.9]), np.array([0, 1]))["roc_auc"] == 1.0
    times = pd.Series(pd.date_range("2026-05-01", periods=5, freq="15min", tz="UTC"))
    assert evaluation_origins(
        times,
        times.iloc[0],
        times.iloc[3],
        horizon=1,
        stride=1,
    ).tolist() == [0, 1, 2]

    class StubPipeline:
        def predict_quantiles(
            self,
            inputs: list[dict[str, Any]],
            prediction_length: int,
            quantile_levels: list[float],
            batch_size: int,
            context_length: int,
        ):
            assert list(inputs[0]["past_covariates"]) == ["feature_a", "feature_b"]
            assert inputs[0]["past_covariates"]["feature_a"].tolist() == [1.0, 2.0]
            assert prediction_length == 2
            assert quantile_levels == [0.1, 0.5, 0.9]
            assert batch_size == 16
            assert context_length == 2
            forecast = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
            return [forecast], []

    forecasts = chronos_forecasts(
        StubPipeline(),  # type: ignore[arg-type]
        np.array([100.0, 101.0, 102.0]),
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ["feature_a", "feature_b"],
        np.array([1]),
        horizon=2,
        context_length=2,
        batch_size=16,
        origin_batch_size=1,
    )
    assert forecasts.tolist() == [[11.0, 21.0, 31.0]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Research-only Chronos-2 vs V03H/V05H XAUUSD benchmark."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--origin-batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-origins", type=int)
    args = parser.parse_args()

    self_test()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available()."
        )
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if args.batch_size is None:
        args.batch_size = 16 if device == "cuda" else 128
    if args.batch_size < 1 or args.origin_batch_size < 1:
        raise ValueError("Batch sizes must be positive integers.")
    for spec in SPECS.values():
        if not spec["model"].is_file():
            raise FileNotFoundError(f"Missing ONNX baseline: {spec['model']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    minute_bars = load_minute_bars(args.dataset, {2025, 2026})
    bars = {
        timeframe: resample_bars(minute_bars, timeframe)
        for timeframe in TIMEFRAME_SECONDS
    }
    del minute_bars

    print("Building locked-contract features from aggregated mid-price bars ...", flush=True)
    frames = {
        timeframe: build_feature_frame(frame, timeframe)
        for timeframe, frame in bars.items()
    }

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
    )
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(
        f"Chronos-2 device={device}"
        + (f" ({gpu_name})" if gpu_name else "")
        + f", batch_size={args.batch_size}",
        flush=True,
    )
    results = []
    for name, spec in SPECS.items():
        order = load_feature_order(spec["features"])
        print(f"Preparing {name} ({spec['timeframe']}, {len(order)} features) ...", flush=True)
        aligned = build_model_frame(frames, spec, order)
        summary, predictions = evaluate_strategy(
            name,
            spec,
            aligned,
            order,
            pipeline,
            args,
        )
        results.append(summary)
        predictions.to_parquet(
            args.output_dir / f"{name.lower()}_predictions.parquet",
            index=False,
        )
        print(json.dumps(summary, indent=2), flush=True)

    output = {
        "model": "amazon/chronos-2",
        "dataset": str(args.dataset.resolve()),
        "dataset_years_used": [2025, 2026],
        "warmup_years": [2025],
        "evaluation_years": [2026],
        "evaluation_period": {
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-05-01T23:59:59Z",
            "end_inclusive": True,
            "labels_must_resolve_by_end": True,
        },
        "device": device,
        "gpu_name": gpu_name,
        "batch_size": args.batch_size,
        "price_construction": "midpoint of tick bid/ask; OHLC resampled from ticks",
        "volume_proxy": "tick count per bar",
        "costs_included": False,
        "chronos_sampling": {
            "V03H_M15": "every bar",
            "V05H_M5": "every 4th bar to keep CPU inference practical",
            "context_length_bars": args.context_length,
        },
        "notes": [
            "Both ONNX baselines and Chronos-2 are scored only on 2026 origins from January 1 through May 1.",
            "2025 is included only to warm up rolling/context features; no 2025 rows are scored.",
            "Origins are excluded if their full label horizon would extend beyond the evaluation end.",
            "Chronos forecast-spread/ATR ranking is reported separately from the signed median-drift score.",
            "Chronos-2 receives each model's full locked feature_order as historical covariates; no future features are supplied.",
            "Barrier labels use each strategy's stated ATR target, stop, and hold horizon.",
            "Same-bar target and stop touches count as a stop.",
            "V05H pre_gap_* inputs are fixed to zero because their historical implementation uses the next bar.",
            "This rebuild uses the supplied tick feed and current feature scripts; it may differ from the original broker/training data.",
            "Published evaluation numbers are not directly comparable because their source datasets and row filters differ.",
            "Research only: no signals are written and MT5 is not contacted.",
        ],
        "strategies": results,
    }
    summary_path = args.output_dir / "chronos2_benchmark_summary.json"
    summary_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    pd.json_normalize(results).to_csv(
        args.output_dir / "chronos2_benchmark_metrics.csv",
        index=False,
    )
    print(f"\nBenchmark summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
