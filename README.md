<div align="center">

# Steam Account Price Prediction

<img src="https://img.shields.io/badge/R²-0.916-success?style=for-the-badge" alt="R2">
<img src="https://img.shields.io/badge/Pearson-0.957-success?style=for-the-badge" alt="Pearson">
<img src="https://img.shields.io/badge/deployment-ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black" alt="CatBoost">
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="sklearn">
<img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas">

</div>

Production ML system for pricing gaming accounts on a large account marketplace (built as the public part of commercial work for LZT Market). Category-specific regression models trained on marketplace history, exported to **ONNX** for lightweight deployment.

## Pipeline

```mermaid
flowchart LR
    D[(🗄 Marketplace history)] --> FE[Feature pipeline]
    FE --> T["Per-category CatBoost training"]
    T --> V["Validation — R² 0.916 · Pearson 0.957"]
    V --> O[ONNX export]
    O --> S[🚀 Marketplace backend inference]

    style V fill:#1a7f37,color:#fff
    style S fill:#0969da,color:#fff
```

## Results

| Metric | Value |
|---|---|
| R² | **0.916** |
| Pearson correlation | **0.957** |
| MAE | 16.4 (price units) |

Best model: `models/trained/category_1_model.onnx`

## Why category-specific models

Account prices are driven by different features in different categories (inventory value, level, badges, ban history…). One model per category outperformed a single global model and keeps every model small enough for cheap ONNX inference inside the marketplace backend.

## Project structure

```
ml_price_predictor/
├── src/
│   ├── models/            # ML model implementations
│   ├── data_processing/   # Feature pipeline
│   └── inference/         # Training and inference scripts
├── tests/                 # Test suite
└── models/
    ├── trained/           # CatBoost models
    └── onnx/              # ONNX exports for deployment
```

## Stack

CatBoost · ONNX / onnxruntime · pandas · scikit-learn
