# Steam Account Price Prediction

Production ML system for pricing gaming accounts on a large account marketplace (built as the public part of commercial work for LZT Market). Category-specific regression models trained on marketplace history, exported to **ONNX** for lightweight deployment.

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
