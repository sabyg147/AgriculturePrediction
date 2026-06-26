# 🌾 AgriCast — GPU-Accelerated Agricultural Price Forecasting

AgriCast is a real-time agricultural commodity price forecasting system built on NVIDIA RAPIDS, Dask, XGBoost, and FastAPI. It ingests historical market price data, trains a GPU-accelerated XGBoost model, and serves daily price predictions through a REST API and an interactive web dashboard.

---

## 📸 Screenshots

> Dashboard running on Google Colab with T4 GPU via ngrok tunnel.

---

## 🧠 What Does This Project Do?

AgriCast allows users to:
- Select a **State**, **District**, and **Commodity** (e.g. Tamil Nadu → Chengalpattu → Tomato)
- Select a **forecast month** (e.g. March 2026)
- Get a **day-by-day price forecast** for that commodity in that location
- Compare **2025 actual prices** vs **2026 predicted prices** on an interactive chart
- View **price statistics**: mean, std deviation, min/max, price range
- See a **Historical Price Distribution** and **Day-by-Day Comparison table**

---

## ⚙️ Tech Stack

### GPU / ML
| Tool | Purpose |
|---|---|
| NVIDIA CUDA (T4 GPU) | Hardware acceleration |
| RAPIDS cuDF | GPU DataFrame operations |
| Dask + Dask-cuDF | Distributed GPU DataFrame processing |
| XGBoost (CUDA) | Price prediction model |
| CuPy | GPU array math |

### Backend
| Tool | Purpose |
|---|---|
| FastAPI | REST API server |
| Uvicorn | ASGI web server |
| pyngrok | Public tunnel from Colab to browser |

### Frontend
| Tool | Purpose |
|---|---|
| HTML + CSS + JavaScript | Interactive dashboard UI |
| Chart.js | Price charts and graphs |

---

## 🗂️ Project Structure

```
AgriculturePrediction/
├── agri.ipynb          # Main Colab notebook (run this)
├── app.py              # FastAPI backend with ML pipeline
├── launcher.py         # Subprocess launcher for Dask + Uvicorn
├── templates/
│   └── index.html      # Frontend dashboard
└── README.md
```

---

## 🚀 How to Run

> ⚠️ This project requires an NVIDIA GPU. It is designed to run on **Google Colab with T4 GPU**. It will NOT work on a regular laptop.

### Step 1 — Open in Colab
Open `agri.ipynb` from your Google Drive in Google Colab. Make sure runtime is set to **T4 GPU**:
`Runtime → Change runtime type → T4 GPU`

### Step 2 — Upload Dataset
Upload your `2025.csv` dataset to Colab local storage. The CSV must have these columns:
```
State, District Name, Commodity, Price Date, Modal Price (Rs./Quintal), Min Price (Rs./Quintal), Max Price (Rs./Quintal)
```

### Step 3 — Set ngrok Token
Get your free authtoken from [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken) and set it in **Colab Secrets** (the 🔑 icon in the left sidebar):
- Name: `NGROK_AUTH_TOKEN`
- Value: your token

### Step 4 — Run All Cells
Click `Runtime → Run all`. Wait for:
```
🌾 AgriCast is LIVE at: https://xxxx.ngrok-free.dev
```

### Step 5 — Open the Dashboard
Click the ngrok URL printed in the output. The dashboard will open in your browser.

---

## 🔌 API Endpoints

### `GET /api/states`
Returns all available states in the dataset.

**Response:**
```json
["Andhra Pradesh", "Gujarat", "Kerala", ...]
```

---

### `GET /api/districts?state=Kerala`
Returns all districts for the selected state.

**Response:**
```json
["Kannur", "Kozhikode", "Thrissur", ...]
```

---

### `GET /api/commodities?state=Kerala&district=Kannur`
Returns all commodities available for the selected state and district.

**Response:**
```json
["Bhindi(Ladies Finger)", "Tomato", "Onion", ...]
```

---

### `GET /api/predict?state=Kerala&district=Kannur&commodity=Tomato&month=3`
Returns the price forecast for the selected combination.

**Response:**
```json
{
  "meta": {
    "state": "Kerala",
    "district": "Kannur",
    "commodity": "Tomato",
    "month": "March",
    "rows_trained": 741,
    "gpu": "Tesla T4",
    "rapids_version": "26.2"
  },
  "stats": {
    "mean": 3878.2,
    "std": 876.19,
    "min": 2900.0,
    "max": 4600.0
  },
  "actual_2025": {
    "days": [8, 9, 10, ...],
    "prices": [3500.0, 3400.0, ...]
  },
  "forecast_2026": {
    "days": [1, 2, 3, ...],
    "prices": [3824.92, 3824.38, ...]
  }
}
```

---

## 🧮 Feature Engineering

The model is trained on these engineered features:

| Feature | Description |
|---|---|
| `month` | Month number (1–12) |
| `day` | Day of month |
| `day_of_year` | Day number in the year (1–365) |
| `sin_month` | Cyclic sine encoding of month |
| `cos_month` | Cyclic cosine encoding of month |
| `lag_1` | Modal price 1 day ago |
| `lag_2` | Modal price 2 days ago |
| `lag_3` | Modal price 3 days ago |
| `lag_7` | Modal price 7 days ago |

Cyclic encoding is used for `month` so the model understands that December (12) and January (1) are close in time — a regular number wouldn't capture this.

---

## 🔥 GPU Acceleration — Why It's Fast

Traditional pandas/sklearn runs on CPU. AgriCast uses:

- **cuDF** instead of pandas → DataFrame ops run on GPU cores
- **Dask-cuDF** → Distributed multi-partition GPU processing
- **XGBoost with `device="cuda"`** → Trees trained on GPU
- **LocalCUDACluster** → Manages GPU workers via Dask

This makes preprocessing and training significantly faster on large datasets compared to CPU-only pipelines.

---

## 📊 Dataset

The project uses historical agricultural market price data from [agmarknet.gov.in](https://agmarknet.gov.in).

Required columns:
```
Sl no., District Name, Market Name, Commodity, Variety, Grade,
Min Price (Rs./Quintal), Max Price (Rs./Quintal), Modal Price (Rs./Quintal),
Price Date, State
```

> The dataset is not included in this repo due to size. Download it from agmarknet.gov.in or Kaggle (search "agmarknet price data").

---

## ❓ FAQ

**Q: Why can't I run this locally?**
A: The project uses NVIDIA RAPIDS (cuDF, cuML, Dask-CUDA) which requires an NVIDIA GPU with CUDA. Most laptops don't have this. Use Google Colab with T4 GPU.

**Q: Why does the app use ngrok?**
A: Google Colab runs on a remote server. ngrok creates a public URL that tunnels traffic from your browser to the Colab server running on port 8000.

**Q: Why is a subprocess (launcher.py) used instead of running FastAPI directly?**
A: LocalCUDACluster and Uvicorn both need to boot without an existing asyncio event loop. Running them in a subprocess avoids conflicts with Colab's own event loop.

**Q: The dataset isn't in the repo — where do I get it?**
A: Download from [agmarknet.gov.in](https://agmarknet.gov.in) → Price & Arrival Reports, or search Kaggle for "agmarknet vegetable prices India". Rename your file to `2025.csv`.

**Q: What does "Trained on X rows" mean in the dashboard?**
A: The model filters data to only your selected State + District + Commodity combination. "Trained on 741 rows" means 741 historical price records were used to train the model for that specific selection.

**Q: Can I deploy this to the cloud?**
A: Yes, but you'll need a cloud instance with an NVIDIA GPU (e.g. AWS p3, GCP A100). Docker support and cloud deployment are planned future improvements.

**Q: What if a commodity has very few rows?**
A: The model may produce less accurate predictions. A minimum of ~100 rows is recommended for reliable forecasting.

---

## 🔮 Future Improvements

- LSTM / Transformer-based forecasting
- Weather data integration
- Crop production and demand analysis
- Docker containerization
- Cloud deployment (AWS / GCP)
- Multi-GPU support

---

## 👥 Authors

Developed as a collaborative academic project demonstrating GPU-accelerated machine learning for agricultural price forecasting using NVIDIA RAPIDS and XGBoost.
