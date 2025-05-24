import pandas as pd
import panel as pn
import hvplot.pandas
from holoviews import opts
import os
from pathlib import Path

# 云部署配置
pn.extension("tabulator")
pn.config.sizing_mode = "stretch_width"
pn.extension(css_files=["https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"])

# 获取端口（云平台会设置环境变量）
PORT = int(os.environ.get("PORT", 5007))

# CSS样式（保持原有样式）
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

# 设置相对路径 - 适配云环境
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# 更新文件路径为相对路径
model_files = {
    "LSTM Model": "data/LSTM/LSTM_pred_true_records.csv",
    "AREWMA Model": "data/AREWMA_Model/predictions.csv",
    "LSTM+GARCH Hybrid Model": "data/Hybrid_Model/hybrid_predictions.csv",
    "EGARCH Model": "data/EGARCH/EGARCH_pred_true_records.csv",
    "XGBoost Model": "data/XGBoost_Model/XGBoost_predictions.csv", 
}

metrics_files = {
    "LSTM Model": "data/LSTM/LSTM_new_weekly_metrics.csv",
    "AREWMA Model": "data/AREWMA_Model/metrics.csv",
    "LSTM+GARCH Hybrid Model": "data/Hybrid_Model/hybrid_metrics.csv",
    "EGARCH Model": "data/EGARCH/EGARCH_new_weekly_metrics.csv",
    "XGBoost Model": "data/XGBoost_Model/XGBoost_metrics.csv",
}
mapping_file = "data/EGARCH/stock_week_cluster_mapping.csv"
# 改进的错误处理
def safe_read_csv(filepath, model_name):
    """安全读取CSV文件，包含详细错误信息"""
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"Warning: File not found for {model_name}: {filepath}")
            return None
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        return None

def standardize_df(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """标准化数据框结构"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df_copy = df.copy()
    

    try:
        
        if model == "AREWMA Model":
            df_copy = df_copy.rename(columns={"actual_volatility": "true", "ar_ewma_prediction": "pred", "time_seconds": "_time_tmp"})
            if not all(col in df_copy.columns for col in ["stock_id", "week", "_time_tmp"]):
                raise ValueError(f"Missing required columns for {model}")
            df_copy = df_copy.sort_values(["stock_id", "week", "_time_tmp"])
            df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            df_copy = df_copy.drop(columns=["_time_tmp"])

        elif model == "EGARCH Model":
            df_copy = df_copy.rename(columns={"realized_volatility": "true", "predicted_volatility": "pred"})
            if "time" in df_copy.columns and "week" in df_copy.columns and "stock_id" in df_copy.columns:
                df_copy = df_copy.sort_values(["stock_id", "week", "time"])
                df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            elif "bucket" not in df_copy.columns:
                if "week" in df_copy.columns and "stock_id" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
                elif "stock_id" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby("stock_id").cumcount()
                else:
                    raise ValueError(f"Missing required columns for {model}")


        elif model == "LSTM+GARCH Hybrid Model":
            df_copy = df_copy.rename(columns={"actual": "true", "pred_hybrid": "pred", "time": "_time_tmp", "stock": "stock_id"})
            if not all(col in df_copy.columns for col in ["stock_id", "week", "_time_tmp"]):
                raise ValueError(f"Missing required columns for {model}")
            df_copy = df_copy.sort_values(["stock_id", "week", "_time_tmp"])
            df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            df_copy = df_copy.drop(columns=["_time_tmp"])

        elif model == "LSTM Model":
            df_copy = df_copy.rename(columns={"true_value": "true", "predicted_value": "pred"})
            if "time" in df_copy.columns and "week" in df_copy.columns and "stock_id" in df_copy.columns:
                df_copy = df_copy.sort_values(["stock_id", "week", "time"])
                df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            elif "bucket" not in df_copy.columns:
                if "week" in df_copy.columns and "stock_id" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
                elif "stock_id" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby("stock_id").cumcount()
                else:
                    raise ValueError(f"Missing required columns for {model}")
                    
        elif model == "XGBoost Model":
            df_copy = df_copy.rename(columns={"stock": "stock_id", "actual": "true"})
            if "time" in df_copy.columns and "week" in df_copy.columns and "stock_id" in df_copy.columns:
                df_copy = df_copy.sort_values(["stock_id", "week", "time"])
                df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
            elif "bucket" not in df_copy.columns:
                if "week" in df_copy.columns and "stock_id" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby(["stock_id", "week"]).cumcount()
                elif "stock_id" in df_copy.columns:
                    df_copy["bucket"] = df_copy.groupby("stock_id").cumcount()
                else:
                    raise ValueError(f"Missing required columns for {model}")
            if "stock_id" not in df_copy.columns:
                raise ValueError(f"Missing stock_id column for {model}")

        # 标准化数据类型
        df_copy["stock_id"] = df_copy["stock_id"].apply(lambda x: str(int(float(str(x)))))
        if "week" in df_copy.columns:
            df_copy["week"] = df_copy["week"].astype(int)
        else:
            df_copy["week"] = 0

        # 检查必需列
        required = {"stock_id", "week", "bucket", "true", "pred"}
        missing = required - set(df_copy.columns)
        if missing:
            raise ValueError(f"Missing final required columns for {model}: {missing}")

        # 从CSV文件读取cluster映射信息
        cluster_mapping_path = "data/EGARCH/stock_week_cluster_mapping.csv"
        try:
            import os
            if os.path.exists(cluster_mapping_path):
                cluster_df = pd.read_csv(cluster_mapping_path)
                
                # 标准化cluster_df中的数据类型以匹配df_copy
                cluster_df["stock_id"] = cluster_df["stock_id"].apply(lambda x: str(int(float(str(x)))))
                cluster_df["week"] = cluster_df["week"].astype(int)
                
                # 合并数据获取cluster信息
                df_copy = df_copy.merge(
                    cluster_df[["stock_id", "week", "cluster"]], 
                    on=["stock_id", "week"], 
                    how="left"
                )
                
                print(f"Successfully loaded cluster mapping from {cluster_mapping_path}")
            else:
                print(f"Warning: Cluster mapping file not found at {cluster_mapping_path}")
                df_copy["cluster"] = None
                
        except Exception as e:
            print(f"Error loading cluster mapping: {str(e)}")
            df_copy["cluster"] = None

        # 如果cluster列仍然不存在，设置为None
        if "cluster" not in df_copy.columns:
            df_copy["cluster"] = None

        return df_copy[list(required) + ["cluster"]]
    
    except Exception as e:
        print(f"Error standardizing {model}: {str(e)}")
        return pd.DataFrame()
# 加载模型数据
model_dfs = {}
successfully_loaded_models = []

print("Loading model data...")
for model, path in model_files.items():
    print(f"Attempting to load {model} from {path}")
    raw_df = safe_read_csv(path, model)
    if raw_df is not None:
        std_df = standardize_df(raw_df, model)
        if not std_df.empty:
            std_df["model"] = model
            model_dfs[model] = std_df
            successfully_loaded_models.append(model)
            print(f"✓ Successfully loaded {model}")
        else:
            print(f"✗ Failed to standardize {model}")
    else:
        print(f"✗ Failed to load {model}")

# 处理数据合并逻辑（保持原有逻辑）
if not model_dfs:
    full_df = pd.DataFrame(columns=["stock_id", "week", "bucket", "true", "pred", "cluster", "model"])
    print("Warning: No model data loaded!")
else:
    print(f"Loaded {len(model_dfs)} models successfully")
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

# 设置选择器选项
unique_stocks = sorted(full_df["stock_id"].unique()) if not full_df.empty and "stock_id" in full_df.columns else []
model_options = successfully_loaded_models if successfully_loaded_models else ["No available model"]

print(f"Available stocks: {len(unique_stocks)}")
print(f"Available models: {model_options}")

# 创建UI组件
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
    pn.pane.HTML("<h3>Forecast Parameters</h3>"),
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

# 初始化week选择器
if unique_stocks:
    update_weeks(stock_select.value, model_select.value)
elif not full_df.empty and "stock_id" in full_df.columns and "model" in full_df.columns:
    if model_select.value:
        update_weeks(None, model_select.value)

@pn.depends(model=model_select.param.value, stock=stock_select.param.value, week=week_select.param.value)
def make_plot(model, stock, week):
    if not (model and stock and week is not None and model != "No available model"):
        return pn.pane.HTML(
            """
            <div class="card" style="text-align:center; padding:40px;">
                <img src="https://cdnjs.cloudflare.com/ajax/libs/ionicons/5.5.0/collection/components/icon/svg/trending-up.svg" width="64" height="64" style="opacity:0.2; margin-bottom:20px;">
                <h3 style="color:#7f8c8d;">Please select model, stock, and week to view forecast.</h3>
            </div>
            """
        )
    try:
        week_int = int(week)
    except (ValueError, TypeError):
        return pn.pane.HTML("<div class='card'><p>Invalid week selection</p></div>")

    if full_df.empty or not ({'model', 'stock_id', 'week', 'bucket', 'true', 'pred'}.issubset(full_df.columns)):
        return pn.pane.HTML("<div class='card'><p>No data available</p></div>")

    df_slice = full_df[
        (full_df.model == model) & (full_df.stock_id == stock) & (full_df.week == week_int)
    ]
    if df_slice.empty:
        return pn.pane.HTML(f"<div class='card'><p>No data for {model}, Stock {stock}, Week {week_int}</p></div>")

    y_cols = [c for c in ["true", "pred"] if df_slice[c].notna().any()]
    if not y_cols:
        return pn.pane.HTML("<div class='card'><p>No valid prediction data</p></div>")

    opts.defaults(
        opts.Curve(
            line_width=3, 
            tools=['hover'], 
            toolbar='above',
            fontscale=1.2,
            show_grid=True
        )
    )

    import holoviews as hv
    import hvplot.pandas
    color_cycle = hv.Cycle(['#3498db', '#e74c3c'])
    plot = df_slice.hvplot(
        x="bucket",
        y=y_cols,
        xlabel="Time Bucket",
        ylabel="Volatility",
        width=900,
        height=400,
        title="Volatility Forecast",
        legend="top_right",
        line_width=2.5,
        responsive=True,
        grid=True,
        fontscale=1.2,
        cmap=['#3498db', '#e74c3c']
    ).opts(
        tools=["pan", "box_zoom", "wheel_zoom", "reset", "save"],
        active_tools=[], 
        shared_axes=False,
        framewise=True,
        bgcolor='white'
    )

    cluster_html = ""
    if "EGARCH Model" in model_dfs and not model_dfs["EGARCH Model"].empty:
        eg_df = model_dfs["EGARCH Model"]
        if "stock_id" in eg_df.columns and "week" in eg_df.columns and "cluster" in eg_df.columns:
            row = eg_df[(eg_df["stock_id"] == str(stock)) & (eg_df["week"] == week_int)]
            if not row.empty:
                cluster_val = row["cluster"].iloc[0]
                if pd.notna(cluster_val):
                    cluster_html = f"""
                    <div style='text-align:center; margin-top:15px;'>
                        <span class='metric-badge'>Cluster {int(cluster_val)}</span>
                    </div>
                    """

    info_card = pn.pane.HTML(f"""
    <div style='text-align:center; margin-bottom:15px;'>
        <div style='font-size:0.9em; color:#7f8c8d;'>CURRENT SELECTION</div>
        <div style='font-size:1.2em; margin:10px 0;'>
            Model: <span class='info-value'>{model}</span> • 
            Stock: <span class='info-value'>{stock}</span> • 
            Week: <span class='info-value'>{week_int}</span>
        </div>
        {cluster_html}
    </div>
    """)

    return pn.Column(
        info_card,
        pn.pane.HoloViews(plot, sizing_mode="stretch_width"),
        css_classes=['card']
    )

# 加载metrics数据 - 保持原有逻辑
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
    except Exception as e:
        print(f"Error loading metrics for {name}: {str(e)}")

all_metrics_df = pd.concat(metrics_df_list, ignore_index=True) if metrics_df_list else pd.DataFrame()

# 加载mapping数据 - 修改为相对路径
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
        print("✓ Successfully loaded cluster mapping")
    else:
        raise ValueError("Empty mapping file")
except Exception as e:
    print(f"Warning: Failed to load cluster mapping: {str(e)}")
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
    pn.pane.HTML("<h3>Performance Metrics by Cluster</h3>"),
    pn.Row(metrics_model_sel, metrics_cluster_sel, align="center"),
    css_classes=['card']
)

def _generate_metrics_display_content(model, cluster_val):
    if model == "No Metrics Model" or model is None or cluster_val is None:
         return pn.pane.HTML(
            """
            <div class="card" style="text-align:center; padding:40px;">
                <img src="https://cdnjs.cloudflare.com/ajax/libs/ionicons/5.5.0/collection/components/icon/svg/stats-chart.svg" width="64" height="64" style="opacity:0.2; margin-bottom:20px;">
                <h3 style="color:#7f8c8d;">Please select a model and cluster to view metrics.</h3>
            </div>
            """
        )

    if all_metrics_df.empty or map_df.empty:
        return pn.pane.HTML("<div class='card'><p>No metrics or mapping data available</p></div>")

    try:
        cluster_int = int(cluster_val)
    except (ValueError, TypeError):
        return pn.pane.HTML("<div class='card'><p>Invalid cluster selection</p></div>")

    if not ({'stock', 'week', 'cluster'}.issubset(map_df.columns)):
        return pn.pane.HTML("<div class='card'><p>Invalid mapping data structure</p></div>")
        
    sel = map_df[map_df.cluster == cluster_int]
    if sel.empty:
        return pn.pane.HTML(f"<div class='card'><p>No data for cluster {cluster_int}</p></div>")

    if not ({'stock', 'week', 'model'}.issubset(all_metrics_df.columns)):
        return pn.pane.HTML("<div class='card'><p>Invalid metrics data structure</p></div>")

    pairs = set(zip(sel.stock, sel.week))
    
    dfm = all_metrics_df[
        (all_metrics_df.model == model)
        & all_metrics_df[["stock", "week"]].apply(tuple, axis=1).isin(pairs)
    ]

    if dfm.empty:
        return pn.pane.HTML(f"<div class='card'><p>No metrics data for {model} in cluster {cluster_int}</p></div>")

    metric_cols = ["R2", "RMSE", "MAE", "MedAE", "MAPE(%)", "SMAPE(%)", "QLIKE"]
    plots = []
    
    box_opts = dict(
        box_fill_color="#f8d7da",
        box_line_color="#e74c3c",
        whisker_line_color="#e74c3c",
        outlier_fill_color="#3498db",
        outlier_line_color="#3498db",
        width=300,
        height=300,
        toolbar="right",
        tools=["hover", "box_zoom", "reset"],
        active_tools=[],
        fontscale=1.2
    )
    
    for col in metric_cols:
        if col in dfm.columns and dfm[col].notna().any():
            q1 = dfm[col].quantile(0.25)
            q3 = dfm[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr 
            lower = q1 - 1.5 * iqr
            clean = dfm[(dfm[col] <= upper) & (dfm[col] >= lower) & (dfm[col].notna())]
            
            if clean.empty or clean[col].nunique() < 1:
                 p_obj = pd.DataFrame({col: []}).hvplot.box(y=col, title=f"{col} (No data or no variance)").opts(**box_opts)
            else:
                p_obj = clean.hvplot.box(y=col, title=col).opts(**box_opts)
            
            plots.append(p_obj)

    if not plots:
        return pn.pane.HTML("<div class='card'><p>No valid metrics to display</p></div>")

    rows = []
    if len(plots) >= 4:
        top_row_plots = plots[:4]
        rows.append(pn.Row(
            pn.Spacer(width=20),
            *top_row_plots,
            pn.Spacer(width=20),
            sizing_mode="stretch_width"
        ))
        if len(plots) > 4:
            remaining_plots = plots[4:]
            spacer_width = (4 - len(remaining_plots)) * 150
            rows.append(pn.Row(
                pn.Spacer(width=spacer_width),
                *remaining_plots,
                pn.Spacer(width=spacer_width),
                sizing_mode="stretch_width"
            ))
    else:
        spacer_width = (4 - len(plots)) * 75
        rows.append(pn.Row(
            pn.Spacer(width=spacer_width),
            *plots,
            pn.Spacer(width=spacer_width),
            sizing_mode="stretch_width"
        ))

    title_text = f"""
    <div style='text-align:center; margin-bottom:20px;'>
        <h3 style='margin:0;'>Performance Metrics: <span class='info-value'>{model}</span></h3>
        <div style='margin-top:10px;'>
            <span class='metric-badge'>Cluster {cluster_int}</span>
            <span style='margin-left:10px; color:#7f8c8d;'>{len(dfm)} data points</span>
        </div>
    </div>
    """
    metrics_explanation = pn.pane.HTML("""
    <div style="margin-top:20px;">
        <h4 style="border-bottom:1px solid #e0e0e0; padding-bottom:8px;">Metrics Definitions</h4>
        <div class="metric-card">
            <div class="metric-name">R² (Coefficient of Determination)</div>
            <div class="metric-desc">Proportion of variance explained by the model; higher is better (max=1.0)</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">RMSE (Root Mean Square Error)</div>
            <div class="metric-desc">Square root of mean squared errors; lower is better</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">MAE (Mean Absolute Error)</div>
            <div class="metric-desc">Average of absolute differences; lower is better</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">MedAE (Median Absolute Error)</div>
            <div class="metric-desc">Median of absolute differences; lower is better, robust to outliers</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">MAPE & SMAPE (Percentage Errors)</div>
            <div class="metric-desc">Error in percentage terms; lower is better</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">QLIKE (Quasi-Likelihood)</div>
            <div class="metric-desc">Loss function for volatility forecasting; lower is better</div>
        </div>
    </div>
    """)

    title = pn.pane.HTML(title_text)
    
    return pn.Column(
        title,
        *rows,
        metrics_explanation,
        css_classes=['card'],
        sizing_mode="stretch_width"
    )

metrics_display_area = pn.Column(sizing_mode="stretch_width", min_height=700)

@pn.depends(model=metrics_model_sel.param.value, cluster=metrics_cluster_sel.param.value, watch=True)
def update_metrics_display_area(model, cluster):
    content = _generate_metrics_display_content(model, cluster)
    metrics_display_area.objects = [content]

# 创建页面布局
header = pn.pane.HTML(
    """
    <div class='header'>
        <h2>Financial Volatility Forecast Dashboard</h2>
        <p>Predictive Analytics for Market Volatility - Cloud Deployed</p>
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

dashboard_description = pn.pane.HTML(
    """
    <div class='card'>
        <h3>About This Dashboard</h3>
        <p>This dashboard presents volatility forecasting models for financial market analysis. The system uses multiple predictive models including LSTM, EGARCH, and hybrid approaches to forecast stock volatility.</p>
        <p>Key features:</p>
        <ul>
            <li><strong>Forecast Chart</strong> - Compare predicted vs. actual volatility values</li>
            <li><strong>Model Metrics</strong> - Analyze performance metrics by stock clusters</li>
        </ul>
    </div>
    """,
    sizing_mode="stretch_width"
)

layout_items = [header]
if alert_pane:
    layout_items.append(alert_pane)
layout_items.extend([dashboard_description, tabs])

layout = pn.Column(
    *layout_items,
    sizing_mode="stretch_width"
)

update_metrics_display_area(metrics_model_sel.value, metrics_cluster_sel.value)
tabs.active = 0

footer = pn.pane.HTML(
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
    """创建Panel应用"""
    return layout

# 使应用可服务
layout.servable()

# 本地运行时的入口点
if __name__ == "__main__":
    print(f"Starting Panel server on port {PORT}")
    print(f"Loaded models: {successfully_loaded_models}")
    print(f"Available clusters: {metrics_cluster_options}")
    
    # 确定允许的来源
    allowed_origins = [f"localhost:{PORT}"]
    
    # 如果在云环境中，添加云域名
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