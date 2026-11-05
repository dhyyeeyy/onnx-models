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

# XAUUSD V05H BUY

Experimental machine-learning BUY model for XAUUSD / Gold.

## Model information

- Model ID: `V05H_WF2025_RF_BUY`
- Strategy ID: `V05H_BUY`
- Instrument: `XAUUSD`
- Direction: `BUY`
- Timeframe: `M5`
- Model family: `Random Forest`
- Inference format: `ONNX`
- Required features: `830`

## Decision rule

Current BUY probability rule:

    prob_buy >= 0.9391089677810669

If the probability is below the threshold, the strategy returns HOLD.

## Trade contract

    Direction: BUY
    Timeframe: M5
    Stop Loss: 1.5 ATR
    Take Profit: 2.0 ATR
    Maximum Hold: 6 M5 bars
    Magic Number: 7300501

## Included files

    final_rf_buy_model.onnx
    feature_order.json
    score_distribution.parquet
    policy_config.json
    model_registry.json

## Feature contract

Inputs must follow exactly the feature order defined in:

    feature_order.json

Current live validation requires:

    expected features = 830
    present features = 830
    missing features = 0
    null after fill = 0
    infinite after fill = 0

before inference.

## Runtime flow

    MT5 M5/M15/H1/H4 data
        -> multi-timeframe feature engineering
        -> structure / confluence / pattern features
        -> 830-feature vector
        -> V05H ONNX model
        -> BUY probability
        -> probability threshold
        -> BUY signal or HOLD

## Timestamp requirement

Live market timestamps must use consistent UTC semantics.

Broker/server time should not simply be labeled as UTC because session
and time-dependent features can otherwise shift.

## Intended use

This model is designed to operate independently inside the
multi-strategy XAUUSD portfolio framework.

For example, an M15 SELL strategy and this M5 BUY strategy may both
generate their own independent positions on an MT5 hedging account.

## Risk notice

This repository contains experimental trading research software.

Historical, validation, or backtest performance does not guarantee
future profitability. Leveraged trading can result in substantial
losses.

Use a demo or paper environment first and independently validate model
behavior, execution, spread, commission, slippage, and risk controls.
