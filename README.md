# Enterprise Retail Demand Forecasting Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

An end-to-end, production-grade demand forecasting and inventory optimization engine built for multi-store enterprise retail environments (e.g., Walmart, Amazon, Target).

---

## 📌 Executive Summary & Financial Impact

Traditional retail demand models trained strictly on sales transactions under-forecast true demand due to **stockout truncation** and fail to optimize working capital. This engine models **latent demand** incorporating price elasticity, calendar Fourier harmonics, and leak-free lag structures.

### Key Business Results:
* **Financial Loss Reduction:** LightGBM reduced total inventory mismatch costs (holding costs + lost stockout margins) by **~28.7%** compared to standard moving average baselines.
* **Forecast Accuracy:** Achieved a **36.4% MAE improvement** across 50 SKUs and 10 retail stores.
* **Operational Value:** Directly feeds safety stock and reorder point (ROP) calculations to balance working capital.

---

## 🛠 System Architecture & Data Flow

[ Synthetic Enterprise Generator ] --> [ Leak-Free Feature Pipeline ] --> [ Temporal Walk-Forward Split ]
│                                   │                                    │
▼                                   ▼                                    ▼
(Stockout Truncation & Elasticity)    (Fourier, Lags, Rolling Stats)        (LightGBM / XGBoost / Prophet)
│
▼
[ Financial Loss Evaluator ] <------------------------------------------- [ Model Inference Engine ]

---

## 📂 Repository Architecture
