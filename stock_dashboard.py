import pandas as pd
import panel as pn
import hvplot.pandas
from holoviews import opts
import os
from pathlib import Path

pn.extension("tabulator")
pn.config.sizing_mode = "stretch_width"
pn.extension(css_files=["https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"])

PORT = int(os.environ.get("PORT", 5007))

css = """
body {
    font-family: 'Roboto', sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 0;
}
.card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    padding: 20px;
    margin-bottom: 20px;
}
.header {
    background: linear-gradient(135deg, #3498db, #2c3e50);
    color: white;
    padding: 30px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    text-align: center;
}
.header h2 {
    margin: 0;
}
.header p {
    margin: 10px 0 0;
    opacity: 0.8;
}
.select-widget select {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Roboto', sans-serif;
    transition: all 0.3s;
}
.select-widget select:focus {
    border-color: #3498db;
    box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
}
.tabs {
    margin-top: 10px;
}
.tabs button {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 10px 20px;
    font-family: 'Roboto', sans-serif;
}
.tabs button.active {
    background-color: white;
    border-bottom: 2px solid #3498db;
}
.metric-badge {
    display: inline-block;
    padding: 4px 10px;
    background-color: #3498db;
    color: white;
    border-radius: 20px;
}
.info-value {
    color: #3498db;
}
.metric-card {
    border-left: 4px solid #3498db;
    padding-left: 15px;
    margin-bottom: 15px;
}
.metric-name {
    color: #2c3e50;
}
.metric-desc {
    color: #7f8c8d;
}
"""
pn.extension(raw_css=[css])

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

model_files = {
    "LSTM Model": "data/LSTM/LSTM_pred_true_records.csv",
    "AREWMA Model": "data/AREWMA_Model/predictions.csv",
    "LSTM+GARCH Hybrid Model": "data/Hybrid_Model/hybrid_predictions.csv",
    "EGARCH Model": "data/EGARCH/test_garch_predicted.csv",
    "XGBoost Model": "data/XGBoost_Model/XGBoost_predictions.csv", 
}

metrics_files = {
    "LSTM Model": "data/LSTM/LSTM_new_weekly_metrics.csv",
    "AREWMA Model": "data/AREWMA_Model/metrics.csv",
    "LSTM+GARCH Hybrid Model": "data/Hybrid_Model/hybrid_metrics.csv",
    "EGARCH Model": "data/EGARCH/egarch_metrics.csv",
    "XGBoost Model": "data/XGBoost_Model/XGBoost_metrics.csv",
}

def safe_read_csv(filepath, model_name):
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(1)
            return None
    except Exception:
        print(2)
        return None

def standardize_df(df: pd.DataFrame, model: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    
    df_copy = df.copy()

    try:
        if model == "AREWMA Model":
            df_copy = df_copy.rename(columns={"actual_volatility": "true", "ar_ewma_prediction": "pred", "time_seconds": "_time_tmp"})
            df_copy = df_copy.sort_values(["stock_id", "week", "_time_tmp"])
            df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            df_copy = df_copy.drop(columns=["_time_tmp"])

        elif model == "EGARCH Model":
            df_copy = df_copy.rename(columns={"realized_volatility": "true", "predicted_volatility": "pred"})
            if "date" in df_copy.columns:
                df_copy = df_copy.sort_values(["stock_id", "date", "bucket_start"])
                df_copy["bucket"] = df_copy.groupby(["stock_id", "date"]).cumcount()
                if "week" not in df_copy.columns:
                    try:
                        df_copy["week"] = pd.to_datetime(df_copy["date"]).dt.isocalendar().week.astype(int)
                    except Exception:
                        df_copy["week"] = 0
            else:
                df_copy = df_copy.sort_values(["stock_id", "week", "bucket_start"])
                df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            df_copy["cluster"] = df_copy.get("cluster", None)

        elif model == "LSTM+GARCH Hybrid Model":
            df_copy = df_copy.rename(columns={"actual": "true", "pred_hybrid": "pred", "time": "_time_tmp", "stock": "stock_id"})
            df_copy = df_copy.sort_values(["stock_id", "week", "_time_tmp"])
            df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            df_copy = df_copy.drop(columns=["_time_tmp"])

        elif model == "LSTM Model":
            df_copy = df_copy.rename(columns={"true_value": "true", "predicted_value": "pred"})
            if "time" in df_copy.columns:
                df_copy = df_copy.sort_values(["stock_id", "week", "time"])
                df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            elif "bucket" not in df_copy.columns:
                if "week" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
                else:
                    df_copy["bucket"] = df_copy.groupby("stock_id").cumcount()

        elif model == "XGBoost Model":
            df_copy = df_copy.rename(columns={"stock": "stock_id", "actual": "true"})
            if "time" in df_copy.columns:
                df_copy = df_copy.sort_values(["stock_id", "week", "time"])
                df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            elif "bucket" not in df_copy.columns:
                if "week" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
                else:
                    df_copy["bucket"] = df_copy.groupby("stock_id").cumcount()
            if "stock_id" not in df_copy.columns:
                print(3)
                return pd.DataFrame()

        df_copy["stock_id"] = df_copy["stock_id"].apply(lambda x: str(int(float(str(x)))))
        if "week" in df_copy.columns:
            df_copy["week"] = df_copy["week"].astype(int)
        else:
            df_copy["week"] = 0

        required = {"stock_id", "week", "bucket", "true", "pred"}
        missing = required - set(df_copy.columns)
        if missing:
            print(4)
            return pd.DataFrame()

        if "cluster" not in df_copy.columns:
            df_copy["cluster"] = None

        return df_copy[list(required) + ["cluster"]]
    except Exception:
        print(5)
        return pd.DataFrame()

model_dfs = {}
successfully_loaded_models = []

print(6)
for model, path in model_files.items():
    print(7)
    raw_df = safe_read_csv(path, model)
    if raw_df is not None:
        std_df = standardize_df(raw_df, model)
        if not std_df.empty:
            std_df["model"] = model
            model_dfs[model] = std_df
            successfully_loaded_models.append(model)
            print(8)
        else:
            print(9)
    else:
        print(10)



if not model_dfs:
    full_df = pd.DataFrame(columns=["stock_id", "week", "bucket", "true", "pred", "cluster", "model"])
    print(11)
else:
    print(12)
    if len(model_dfs) > 1:
        valid_dfs_for_intersection = [
            df for df in model_dfs.values() 
            if not df.empty and "stock_id" in df.columns and "week" in df.columns
        ]
        if valid_dfs_for_intersection:
            sets = [set(zip(df["stock_id"], df["week"])) for df in valid_dfs_for_intersection]
            common_pairs = set.intersection(*sets) if sets else set()
        else:
            common_pairs = set()
    elif model_dfs:
        single_df = next(iter(model_dfs.values()))
        if not single_df.empty and "stock_id" in single_df.columns and "week" in single_df.columns:
            common_pairs = set(zip(single_df["stock_id"], single_df["week"]))
        else:
            common_pairs = set()
    else:
        common_pairs = set()

    if common_pairs:
        dfs_to_concat = []
        for model_name, df_val in model_dfs.items():
            if not df_val.empty and "stock_id" in df_val.columns and "week" in df_val.columns:
                dfs_to_concat.append(df_val[df_val[["stock_id", "week"]].apply(tuple, axis=1).isin(common_pairs)])
        if dfs_to_concat:
            full_df = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            full_df = pd.DataFrame(columns=["stock_id", "week", "bucket", "true", "pred", "cluster", "model"])
    elif model_dfs:
        valid_dfs_to_concat = [df for df in model_dfs.values() if not df.empty]
        if valid_dfs_to_concat:
            full_df = pd.concat(valid_dfs_to_concat, ignore_index=True)
        else:
            full_df = pd.DataFrame(columns=["stock_id", "week", "bucket", "true", "pred", "cluster", "model"])
    else:
        full_df = pd.DataFrame(columns=["stock_id", "week", "bucket", "true", "pred", "cluster", "model"])

unique_stocks = sorted(full_df["stock_id"].unique()) if not full_df.empty and "stock_id" in full_df.columns else []
model_options = successfully_loaded_models if successfully_loaded_models else ["No available model"]

print(13)
print(14)

model_select = pn.widgets.Select(
    name="Model", 
    options=model_options, 
    value=model_options[0] if model_options else None,
    css_classes=['select-widget']
)
stock_select = pn.widgets.Select(
    name="Stock", 
    options=unique_stocks, 
    value=unique_stocks[0] if unique_stocks else None,
    css_classes=['select-widget']
)
week_select = pn.widgets.Select(
    name="Week", 
    options=[], 
    css_classes=['select-widget']
)

controls = pn.Column(
    pn.pane.Markdown("### Forecast Parameters"),
    pn.Row(model_select, stock_select, week_select, align="center"),
    css_classes=['card']
)

@pn.depends(stock=stock_select.param.value, model=model_select.param.value, watch=True)
def update_weeks(stock, model):
    if not (stock and model and model != "No available model" and not full_df.empty and "week" in full_df.columns):
        week_select.options = []
        week_select.value = None
        return
    if "stock_id" in full_df.columns and "model" in full_df.columns:
        wks = sorted(full_df[(full_df.stock_id == stock) & (full_df.model == model)]["week"].unique())
        week_select.options = wks
        week_select.value = wks[0] if wks else None
    else:
        week_select.options = []
        week_select.value = None

if unique_stocks:
    update_weeks(stock_select.value, model_select.value)
elif not full_df.empty and "stock_id" in full_df.columns and "model" in full_df.columns:
    if model_select.value:
        update_weeks(None, model_select.value)

metrics_df_list = []
for name, path in metrics_files.items():
    try:
        dfm = safe_read_csv(path, name)
        if dfm is not None and not dfm.empty:
            if name == "AREWMA Model":
                for col in dfm.columns:
                    if dfm[col].astype(str).str.contains('AR-EWMA').any():
                        dfm = dfm[dfm[col] == 'AR-EWMA']
                        break
            if name == "XGBoost Model":
                if 'model' in dfm.columns:
                    dfm = dfm[dfm['model'] == 'XGB']
            dfm["model"] = name
            if 'stock_id' in dfm.columns and 'stock' not in dfm.columns:
                dfm['stock'] = dfm['stock_id']
            metrics_df_list.append(dfm)
    except Exception:
        print(15)

all_metrics_df = pd.concat(metrics_df_list, ignore_index=True) if metrics_df_list else pd.DataFrame()

mapping_file = "data/EGARCH/stock_week_cluster_mapping.csv"
try:
    map_df = safe_read_csv(mapping_file, "Cluster Mapping")
    if map_df is not None and not map_df.empty:
        if 'stock_id' in map_df.columns:
            map_df["stock"] = map_df["stock_id"].astype(int)
        elif 'stock' in map_df.columns:
            map_df["stock"] = map_df["stock"].astype(int)
        else:
            raise ValueError("No stock column found")
        map_df["week"] = map_df["week"].astype(int)
        if "cluster" not in map_df.columns:
            raise ValueError("No cluster column found")
        clusters = sorted(map_df["cluster"].unique())
        if not clusters:
            clusters = [0]
        print(16)
    else:
        raise ValueError("Empty mapping file")
except Exception:
    print(17)
    map_df = pd.DataFrame(columns=["stock", "week", "cluster"])
    clusters = [0]

metrics_model_options = list(metrics_files.keys()) if metrics_files else ["No Metrics Model"]
metrics_cluster_options = sorted([int(c) for c in clusters if pd.notna(c)]) if clusters else [0]

metrics_model_sel = pn.widgets.Select(
    name="Model", 
    options=metrics_model_options, 
    value=metrics_model_options[0] if metrics_model_options else None,
    css_classes=['select-widget']
)
metrics_cluster_sel = pn.widgets.Select(
    name="Cluster", 
    options=metrics_cluster_options, 
    value=metrics_cluster_options[0] if metrics_cluster_options else None,
    css_classes=['select-widget']
)

metrics_controls = pn.Column(
    pn.pane.Markdown("### Performance Metrics by Cluster"),
    pn.Row(metrics_model_sel, metrics_cluster_sel, align="center"),
    css_classes=['card']
)

metrics_display_area = pn.Column(sizing_mode="stretch_width", min_height=700)

@pn.depends(model=metrics_model_sel.param.value, cluster=metrics_cluster_sel.param.value, watch=True)
def update_metrics_display_area(model, cluster):
    content = _generate_metrics_display_content(model, cluster)
    metrics_display_area.objects = [content]

header = pn.pane.HTML(
    """
    <div class='header'>
        <h2>Financial Volatility Forecast Dashboard</h2>
        <p>Predictive Analytics for Market Volatility</p>
    </div>
    """,
    sizing_mode="stretch_width",
)

alert_pane = None
if not model_dfs:
    alert_pane = pn.pane.Alert(
        "Error: No model data loaded. Dashboard functionality will be limited.", 
        alert_type="danger",
        css_classes=['card']
    )

tabs = pn.Tabs(
    ("Forecast Chart", pn.Column(controls, make_plot, sizing_mode="stretch_width")),
    ("Model Metrics", pn.Column(metrics_controls, metrics_display_area, sizing_mode="stretch_width")),
    css_classes=['tabs']
)

dashboard_description = pn.pane.Markdown(
    """
    ### About This Dashboard

    This dashboard presents volatility forecasting models for financial market analysis. The system uses multiple predictive models including LSTM, EGARCH, and hybrid approaches to forecast stock volatility.

    **Key features:**
    - **Forecast Chart**: Compare predicted vs. actual volatility values  
    - **Model Metrics**: Analyze performance metrics by stock clusters
    """,
    sizing_mode="stretch_width"
)

layout_items = [header]
if alert_pane:
    layout_items.append(alert_pane)
layout_items.extend([dashboard_description, tabs])

layout = pn.Column(*layout_items, sizing_mode="stretch_width")

update_metrics_display_area(metrics_model_sel.value, metrics_cluster_sel.value)
tabs.active = 0

footer = pn.pane.Markdown(
    f"""
    <div style="text-align:center; margin-top:20px; padding:15px; background:#f5f5f5; border-radius:8px;">
        <p style="color:#7f8c8d; margin:0;">
            Financial Volatility Analysis Dashboard • Running on Port {PORT} • Cloud Deployed ☁️
        </p>
    </div>
    """,
    sizing_mode="stretch_width"
)

layout.append(footer)

def create_app():
    return layout

layout.servable()

if __name__ == "__main__":
    print(18)
    print(19)
    print(20)

    allowed_origins = [f"localhost:{PORT}"]

    if "RAILWAY_STATIC_URL" in os.environ:
        allowed_origins.append(os.environ["RAILWAY_STATIC_URL"])
    elif "RENDER_EXTERNAL_HOSTNAME" in os.environ:
        allowed_origins.append(os.environ["RENDER_EXTERNAL_HOSTNAME"])

    pn.serve(
        layout,
        port=PORT,
        allow_websocket_origin=allowed_origins,
        show=True,
        title="Financial Volatility Forecast Dashboard",
        autoreload=False
    )
