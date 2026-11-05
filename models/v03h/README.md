---
tags:
- finance
- trading
- xauusd
- gold
- machine-learning
- onnx
- random-forest
- time-series
library_name: onnxruntime
---

# XAUUSD V03H SELL

Experimental machine-learning SELL model for XAUUSD / Gold.

## Model information

- Model ID: `V03H_WF2025_RF_SELL`
- Strategy ID: `V03H_F002_SELL`
- Instrument: `XAUUSD`
- Direction: `SELL`
- Timeframe: `M15`
- Model family: `Random Forest`
- Inference format: `ONNX`
- Required features: `300`

## Decision rule

The model produces a SELL probability. The runtime converts the score
into a historical percentile rank using `score_distribution.parquet`.

Current operational rule:

    rank_pct >= 0.90

Original locked research threshold:

    rank_pct >= 0.99

## Trade contract

    Direction: SELL
    Timeframe: M15
    Stop Loss: 1.0 ATR
    Take Profit: 2.0 ATR
    Maximum Hold: 32 M15 bars
    Magic Number: 7300201

## Included files

    final_rf_sell_model.onnx
    feature_order.json
    score_distribution.parquet
    policy_config.json
    model_registry.json

## Feature contract

Inputs must use exactly the features and ordering defined in:

    feature_order.json

The live runtime validates:

    required features = 300
    missing features = 0
    null features = 0
    infinite features = 0

before inference.

## Runtime flow

    MT5 closed bars
        -> feature engineering
        -> 300-feature vector
        -> V03H ONNX model
        -> SELL probability
        -> historical percentile rank
        -> rank threshold
        -> SELL signal or HOLD

## Timestamp requirement

Live market timestamps must use consistent UTC semantics.

Incorrect broker-server time handling can alter session and time-derived
features even when other indicators appear correct.

## Intended use

This model is intended to operate as one independent strategy inside a
multi-strategy XAUUSD portfolio framework.

Other BUY or SELL models do not need to agree before this strategy
generates its own signal.

## Risk notice

This repository contains experimental trading research software.

Historical, validation, or backtest performance does not guarantee
future profitability. Leveraged trading can result in substantial
losses.

Use a demo or paper environment first and independently validate model
behavior, execution, spread, commission, slippage, and risk controls.
