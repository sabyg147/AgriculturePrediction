
import cudf
import cupy as cp
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import dask
import xgboost as xgb
from xgboost import dask as dxgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os

FILE_PATH   = "2025_fixed.csv"
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
MONTH_DAYS  = {1:31,2:28,3:31,4:30,5:31,6:30,
               7:31,8:31,9:30,10:31,11:30,12:31}

app     = FastAPI(title="AgriCast")
client  = None
ddf_all = None


def init_cluster():
    # Boot cluster + load CSV. Called from launcher.py BEFORE uvicorn starts.
    global client, ddf_all
    print(">> Initialising LocalCUDACluster ...")
    cluster = LocalCUDACluster()
    client  = Client(cluster)
    print(f"   Dask ready -> {client.dashboard_link}")

    if not os.path.exists(FILE_PATH):
        raise RuntimeError(f"{FILE_PATH} not found.")

    print(f">> Loading {FILE_PATH} ...")
    ddf_all = dask_cudf.read_csv(FILE_PATH)
    ddf_all = ddf_all.rename(columns={
        'Price Date': 'Arrival_Date',
        'District Name': 'District',
        'Modal Price (Rs./Quintal)': 'Modal_Price',
        'Min Price (Rs./Quintal)': 'Min_Price',
        'Max Price (Rs./Quintal)': 'Max_Price',
        'Market Name': 'Market'
    })
    ddf_all.columns = [c.strip() for c in ddf_all.columns]
    ddf_all["Arrival_Date"] = ddf_all["Arrival_Date"].map_partitions(
        cudf.to_datetime, format="%d-%m-%Y",
        meta=cudf.Series([], dtype="datetime64[ns]")
    )
    print(">> Data ready.")


def prepare_features(ddf):
    ddf["month"]        = ddf["Arrival_Date"].dt.month
    ddf["day"]          = ddf["Arrival_Date"].dt.day
    ddf["day_of_year"]  = ddf["Arrival_Date"].dt.dayofyear
    ddf["week_of_year"] = (ddf["day_of_year"] // 7) + 1

    def cyclic(df):
        df["sin_day"] = cp.sin(2 * cp.pi * df["day_of_year"].values / 365.0)
        df["cos_day"] = cp.cos(2 * cp.pi * df["day_of_year"].values / 365.0)
        return df

    ddf = ddf.map_partitions(cyclic)
    for lag in [1, 2, 3, 7]:
        ddf[f"lag_{lag}"] = ddf["Modal_Price"].shift(lag)
    return ddf.dropna()


FEATURES = ["month","day_of_year","week_of_year","sin_day","cos_day",
            "lag_1","lag_2","lag_3","lag_7"]


@app.get("/api/states")
async def get_states():
    states = sorted(ddf_all["State"].unique().compute().to_arrow().to_pylist())
    return JSONResponse({"states": states})


@app.get("/api/districts")
async def get_districts(state: str):
    mask = ddf_all["State"] == state
    districts = sorted(ddf_all[mask]["District"].unique().compute().to_arrow().to_pylist())
    return JSONResponse({"districts": districts})


@app.get("/api/commodities")
async def get_commodities(state: str, district: str):
    mask = (ddf_all["State"] == state) & (ddf_all["District"] == district)
    commodities = sorted(ddf_all[mask]["Commodity"].unique().compute().to_arrow().to_pylist())
    return JSONResponse({"commodities": commodities})


@app.get("/api/predict")
async def predict(state: str, district: str, commodity: str, month: int):
    global ddf_all, client
    if month < 1 or month > 12:
        raise HTTPException(400, "month must be 1-12")

    mask     = ((ddf_all["State"]     == state) &
                (ddf_all["District"]  == district) &
                (ddf_all["Commodity"] == commodity))
    hist_ddf = ddf_all[mask].repartition(npartitions=1)
    row_count = len(hist_ddf)
    if row_count < 10:
        raise HTTPException(400, f"Not enough data: only {row_count} rows.")

    ddf_proc = prepare_features(hist_ddf)
    X = ddf_proc[FEATURES].astype("float32")
    y = ddf_proc["Modal_Price"].astype("float32")
    X, y = dask.persist(X, y)
    wait([X, y])

    month_mask  = ddf_proc["month"] == month
    hist_month  = ddf_proc[month_mask][["day","Modal_Price","Min_Price","Max_Price"]].compute()
    days_2025   = hist_month["day"].to_arrow().to_pylist()
    actual_2025 = hist_month["Modal_Price"].to_arrow().to_pylist()

    all_prices = ddf_proc["Modal_Price"].compute()
    price_mean = float(all_prices.mean())
    price_std  = float(all_prices.std())
    price_min  = float(all_prices.min())
    price_max  = float(all_prices.max())

    dtrain = dxgb.DaskDMatrix(client, X, y)
    output = dxgb.train(
        client,
        {"tree_method":"hist","device":"cuda","objective":"reg:squarederror",
         "eta":0.05,"max_depth":6,"subsample":0.8},
        dtrain, num_boost_round=150
    )

    days       = list(range(1, MONTH_DAYS[month] + 1))
    date_strs  = [f"2026-{month:02d}-{d:02d}" for d in days]
    seed_price = float(hist_ddf["Modal_Price"].mean().compute())

    future_cdf = cudf.DataFrame({
        "Arrival_Date": cudf.to_datetime(cudf.Series(date_strs), format="%Y-%m-%d"),
        "Modal_Price" : cudf.Series([seed_price] * len(days), dtype="float32"),
        "Min_Price"   : cudf.Series([seed_price] * len(days), dtype="float32"),
        "Max_Price"   : cudf.Series([seed_price] * len(days), dtype="float32"),
    })
    future_ddf  = dask_cudf.from_cudf(future_cdf, npartitions=1)
    future_proc = prepare_features(future_ddf)
    X_future    = future_proc[FEATURES].astype("float32")
    preds_raw   = dxgb.predict(client, output, X_future).compute()
    days_2026   = future_proc["day"].compute().to_arrow().to_pylist()
    preds_2026  = [round(float(v), 2) for v in cp.asnumpy(preds_raw)]

    device     = cp.cuda.Device(0)
    gpu_name   = cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode()
    rapids_ver = cudf.__version__

    return JSONResponse({
        "meta": {"state": state, "district": district, "commodity": commodity,
                 "month": MONTH_NAMES[month-1], "rows_trained": row_count,
                 "gpu": gpu_name, "rapids_version": rapids_ver},
        "stats": {"mean": round(price_mean,2), "std": round(price_std,2),
                  "min":  round(price_min,2),  "max": round(price_max,2)},
        "actual_2025":   {"days": days_2025,  "prices": [round(float(p),2) for p in actual_2025]},
        "forecast_2026": {"days": days_2026,  "prices": preds_2026}
    })


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html") as f:
        return f.read()

