"""
Basic Predictive Maintenance App
AI4I 2020 Dataset — Machine Failure Prediction
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Predictive Maintenance of Milling Machine ")
st.markdown("**AI4I 2020 Dataset** · XGBoost / Random Forest · SMOTE balanced")
st.divider()

# ── Load artifacts ────────────────────────────────────────────────────────────
MODEL_DIR = "model_artifacts"

@st.cache_resource
def load_artifacts():
    model      = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le         = joblib.load(os.path.join(MODEL_DIR, "label_encoder_type.joblib"))
    feat_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, scaler, le, feat_names

@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df["Temp_diff"]   = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power"]       = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df

try:
    model, scaler, le, feat_names = load_artifacts()
    df = load_data()
except Exception as e:
    st.error(f"❌ Could not load model artifacts: {e}")
    st.info("Run `Predictive_Maintenance_ML_Pipeline.py` first to generate the model.")
    st.stop()

# ── Safety ranges (from AI4I paper) ──────────────────────────────────────────
SAFETY = {
    "Air temperature [K]":     {"safe_lo": 296.5, "safe_hi": 302.5, "unit": "K"},
    "Process temperature [K]": {"safe_lo": 307.0, "safe_hi": 312.0, "unit": "K"},
    "Rotational speed [rpm]":  {"safe_lo": 1300,  "safe_hi": 2600,  "unit": "rpm"},
    "Torque [Nm]":             {"safe_lo": 10.0,  "safe_hi": 60.0,  "unit": "Nm"},
    "Tool wear [min]":         {"safe_lo": 0,     "safe_hi": 200,   "unit": "min"},
}

FAILURE_INFO = {
    "TWF": ("🔧 Tool Wear Failure",        "Tool Wear > 200 min"),
    "HDF": ("🌡️ Heat Dissipation Failure", "ΔTemp < 8.6 K & RPM < 1380"),
    "PWF": ("⚡ Power Failure",            "Power outside [3500–9000] W"),
    "OSF": ("💥 Overstrain Failure",       "Wear × Torque > type limit"),
    "RNF": ("🎲 Random Failure",           "0.1% random probability"),
}

# ── Helper: build feature vector ──────────────────────────────────────────────
def make_features(air_temp, proc_temp, rpm, torque, wear, ptype, le, feat_names):
    temp_diff  = proc_temp - air_temp
    power      = torque * rpm
    tsr        = torque / (rpm + 1e-9)
    wear_torq  = wear * torque
    enc        = le.transform([ptype])[0]
    fd = {
        "Air temperature _K":     air_temp,
        "Process temperature _K": proc_temp,
        "Rotational speed _rpm":  rpm,
        "Torque _Nm":             torque,
        "Tool wear _min":         wear,
        "Temp_diff":              temp_diff,
        "Power":                  power,
        "Torque_speed_ratio":     tsr,
        "Wear_torque":            wear_torq,
        "Type_encoded":           enc,
        "Type_H": 1 if ptype == "H" else 0,
        "Type_L": 1 if ptype == "L" else 0,
        "Type_M": 1 if ptype == "M" else 0,
    }
    X = pd.DataFrame([fd])[feat_names]
    return X, fd

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Failure", "🛡️ Safety Check", "📊 Feature Analysis", "🔬 Failure Types", "📈 Dataset Overview"],
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT FAILURE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔮 Predict Failure":
    st.header("🔮 Predict Machine Failure")
    st.markdown("Enter the current operating parameters and click **Predict** to get the model output.")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("⚙️ Input Parameters")
        ptype    = st.selectbox("Product Type", ["L", "M", "H"],
                                index=1, help="L=Low · M=Medium · H=High grade")
        air_temp  = st.slider("🌡️ Air Temperature (K)",   295.0, 304.0, 300.0, 0.1)
        proc_temp = st.slider("🔥 Process Temperature (K)", 305.0, 314.0, 310.0, 0.1)
        rpm       = st.slider("🔄 Rotational Speed (rpm)", 1168, 2886, 1500, 10)
        torque    = st.slider("💪 Torque (Nm)",            3.8, 76.6, 40.0, 0.1)
        wear      = st.slider("🔧 Tool Wear (min)",        0, 253, 100, 1)
        threshold = st.slider("⚖️ Decision Threshold",    0.10, 0.90, 0.50, 0.01,
                               help="Lower → catch more failures (higher recall)")
        predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")

    with col_output:
        st.subheader("📋 Prediction Result")
        if predict_btn:
            X, fd = make_features(air_temp, proc_temp, rpm, torque, wear, ptype, le, feat_names)
            X_sc  = scaler.transform(X)
            probs = model.predict_proba(X_sc)[0]
            p_fail = float(probs[1])
            pred   = 1 if p_fail >= threshold else 0

            # Result box
            if pred == 1:
                st.error(f"## ⚠️ FAILURE PREDICTED\n**Failure Probability: {p_fail*100:.1f}%**")
            else:
                st.success(f"## ✅ No Failure\n**Safe Probability: {(1-p_fail)*100:.1f}%**")

            # Probability bar
            st.metric("Failure Probability", f"{p_fail*100:.2f}%",
                      delta=f"{(p_fail - threshold)*100:+.1f}% vs threshold")
            st.progress(p_fail)

            # Derived values
            st.divider()
            st.markdown("**📐 Derived Values**")
            power_w = torque * rpm * 2 * np.pi / 60
            st.write(f"- Temperature Difference: **{proc_temp - air_temp:.1f} K**")
            st.write(f"- Power (Torque × RPM × 2π/60): **{power_w:,.0f} W**")
            st.write(f"- Wear × Torque: **{wear * torque:,.1f}**")

            # Rule-based failure type checks
            st.divider()
            st.markdown("**🔍 Rule-Based Failure Type Flags**")
            temp_diff = proc_temp - air_temp
            power_w   = torque * rpm * 2 * np.pi / 60   # convert to actual Watts
            wt_val    = wear * torque
            thresh_map = {"L": 11000, "M": 12000, "H": 13000}

            flags = {
                "TWF": wear > 200,
                "HDF": temp_diff < 8.6 and rpm < 1380,
                "PWF": power_w < 3500 or power_w > 9000,  # AI4I threshold in real Watts
                "OSF": wt_val > thresh_map.get(ptype, 12000),
                "RNF": False,
            }
            for code, triggered in flags.items():
                name, trigger = FAILURE_INFO[code]
                icon = "🔴" if triggered else "🟢"
                st.write(f"{icon} **{name}** — _{trigger}_")
        else:
            st.info("👈 Set parameters on the left and click **Predict**.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SAFETY CHECK
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🛡️ Safety Check":
    st.header("🛡️ Safety Operating Parameter Check")
    st.markdown("Enter live machine readings to instantly check if they are within the safe operating envelope.")

    st.subheader("📥 Current Machine Readings")
    c1, c2, c3 = st.columns(3)
    v_air  = c1.number_input("Air Temp (K)",      value=300.0, step=0.1, format="%.1f")
    v_proc = c1.number_input("Process Temp (K)",  value=310.0, step=0.1, format="%.1f")
    v_rpm  = c2.number_input("Speed (rpm)",       value=1500,  step=10)
    v_torq = c2.number_input("Torque (Nm)",       value=40.0,  step=0.1, format="%.1f")
    v_wear = c3.number_input("Tool Wear (min)",   value=100,   step=1)

    readings = {
        "Air temperature [K]":     v_air,
        "Process temperature [K]": v_proc,
        "Rotational speed [rpm]":  v_rpm,
        "Torque [Nm]":             v_torq,
        "Tool wear [min]":         v_wear,
    }

    st.divider()
    st.subheader("📊 Safety Status")

    all_safe = True
    for param, val in readings.items():
        s = SAFETY[param]
        in_safe = s["safe_lo"] <= val <= s["safe_hi"]
        if not in_safe:
            all_safe = False
        status  = "✅ Safe" if in_safe else "🔴 Out of Range"
        pct     = (val - s["safe_lo"]) / (s["safe_hi"] - s["safe_lo"])
        pct_clamp = max(0.0, min(1.0, pct))

        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
        col_a.write(f"**{param}**")
        col_b.write(f"`{val:.1f} {s['unit']}`")
        col_c.write(f"Safe: `{s['safe_lo']} – {s['safe_hi']}`")
        col_d.write(status)
        st.progress(pct_clamp)

    st.divider()
    if all_safe:
        st.success("✅ **All parameters are within safe operating range.** Machine is healthy.")
    else:
        st.error("⚠️ **One or more parameters are out of range! Immediate attention required.**")

    # Historical context
    st.subheader("📉 Historical Distributions")
    st.caption("Green = No Failure · Red = Failure · Dashed line = your current value · Shaded band = safe zone")

    params_list = list(readings.keys())
    vals_list   = list(readings.values())

    # 2-row grid: 3 on top, 2 on bottom (last axis hidden)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.ravel()

    for i, (param, val) in enumerate(zip(params_list, vals_list)):
        ax = axes_flat[i]
        s = SAFETY[param]

        safe_data = df[df["Machine failure"] == 0][param]
        fail_data = df[df["Machine failure"] == 1][param]

        # Histograms
        ax.hist(safe_data, bins=35, alpha=0.55, color="#22c55e",
                label="No Failure", density=True, edgecolor="white", linewidth=0.4)
        ax.hist(fail_data, bins=35, alpha=0.65, color="#ef4444",
                label="Failure",    density=True, edgecolor="white", linewidth=0.4)

        # Safe-zone shading
        ax.axvspan(s["safe_lo"], s["safe_hi"], alpha=0.12, color="#22c55e",
                   label="Safe zone")

        # Safe-zone boundary lines
        ax.axvline(s["safe_lo"], color="#15803d", linewidth=1.2,
                   linestyle=":", alpha=0.8)
        ax.axvline(s["safe_hi"], color="#15803d", linewidth=1.2,
                   linestyle=":", alpha=0.8)

        # Current value line
        ax.axvline(val, color="#1e293b", linewidth=2.5,
                   linestyle="--", label=f"Current: {val:.1f} {s['unit']}")

        # Labels & styling
        short_name = param.split("[")[0].strip()
        ax.set_title(short_name, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel(f"Value ({s['unit']})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide the unused 6th subplot
    axes_flat[-1].set_visible(False)

    plt.suptitle("Historical Distributions — No Failure vs Failure",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Feature Analysis":
    st.header("📊 Feature Analysis")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlation", "🎯 Feature vs Failure"])

    num_cols = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                "Temp_diff", "Power", "Wear_torque"]

    with tab1:
        feat = st.selectbox("Select Feature", num_cols, key="dist")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        for label, color, name in [(0, "#22c55e", "No Failure"), (1, "#ef4444", "Failure")]:
            sub = df[df["Machine failure"] == label][feat]
            axes[0].hist(sub, bins=40, alpha=0.6, color=color, label=name, density=True)
        axes[0].set_title(f"Distribution: {feat}")
        axes[0].set_xlabel(feat)
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # Box plot
        df_box = df[[feat, "Machine failure"]].copy()
        df_box["Status"] = df_box["Machine failure"].map({0: "No Failure", 1: "Failure"})
        bp_groups = [df_box[df_box["Status"] == s][feat].values
                     for s in ["No Failure", "Failure"]]
        bp = axes[1].boxplot(bp_groups, patch_artist=True, notch=True,
                              labels=["No Failure", "Failure"])
        bp["boxes"][0].set_facecolor("#22c55e")
        bp["boxes"][1].set_facecolor("#ef4444")
        for box in bp["boxes"]: box.set_alpha(0.7)
        axes[1].set_title(f"Box Plot: {feat}")
        axes[1].set_ylabel(feat)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Stats table
        st.markdown("**Descriptive Statistics**")
        stats = df.groupby("Machine failure")[feat].describe().round(3)
        stats.index = ["No Failure (0)", "Failure (1)"]
        st.dataframe(stats, use_container_width=True)

    with tab2:
        corr_cols = num_cols + ["Machine failure"]
        corr = df[corr_cols].corr().round(3)

        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top correlates with failure
        st.markdown("**Top Features Correlated with Machine Failure**")
        corr_fail = (corr["Machine failure"]
                     .drop("Machine failure")
                     .abs()
                     .sort_values(ascending=False)
                     .reset_index())
        corr_fail.columns = ["Feature", "|Correlation|"]
        st.dataframe(corr_fail, use_container_width=True, hide_index=True)

    with tab3:
        feat2 = st.selectbox("Select Feature", num_cols, key="vf")
        n_bins = st.slider("Number of bins", 5, 30, 15)

        df_vf = df.copy()
        df_vf["bin"] = pd.qcut(df_vf[feat2], q=n_bins, duplicates="drop")
        grp = (df_vf.groupby("bin", observed=True)["Machine failure"]
                    .agg(["mean", "count"])
                    .reset_index())
        grp["mid"] = grp["bin"].apply(lambda x: (x.left + x.right) / 2)
        grp["fail_pct"] = grp["mean"] * 100

        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax2 = ax1.twinx()
        ax1.bar(range(len(grp)), grp["count"], color="#bfdbfe", alpha=0.8, label="Count")
        ax2.plot(range(len(grp)), grp["fail_pct"], color="#ef4444",
                 marker="o", linewidth=2, label="Failure Rate (%)")
        ax1.set_xticks(range(len(grp)))
        ax1.set_xticklabels([f"{v:.1f}" for v in grp["mid"]], rotation=45, fontsize=8)
        ax1.set_ylabel("Sample Count", color="#3b82f6")
        ax2.set_ylabel("Failure Rate (%)", color="#ef4444")
        ax1.set_title(f"{feat2} vs Failure Rate", fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — FAILURE TYPES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Failure Types":
    st.header("🔬 Failure Type Guide")
    st.markdown("Understand each failure mode, how often it occurs, and what triggers it.")

    ft_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    counts  = df[ft_cols].sum()
    rates   = (df[ft_cols].mean() * 100).round(3)

    # Summary table
    summary = pd.DataFrame({
        "Failure Type": [FAILURE_INFO[c][0] for c in ft_cols],
        "Code": ft_cols,
        "Count": counts.values,
        "Rate (%)": rates.values,
        "Trigger": [FAILURE_INFO[c][1] for c in ft_cols],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # Horizontal bar chart — avoids long label overlap
    fig, ax = plt.subplots(figsize=(10, 4))
    ft_colors = ["#f97316", "#ef4444", "#8b5cf6", "#d946ef", "#64748b"]
    short_labels = [FAILURE_INFO[c][0] for c in ft_cols]
    bars = ax.barh(short_labels, counts.values, color=ft_colors,
                   edgecolor="white", height=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(int(v)), va="center", fontweight="bold", fontsize=11)
    ax.set_title("Failure Type Frequency", fontweight="bold", fontsize=13)
    ax.set_xlabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Per-type feature violin
    st.divider()
    st.subheader("📉 Feature Distribution by Failure Type")
    sel_ft  = st.selectbox("Failure Type", ft_cols)
    sel_col = st.selectbox("Feature", ["Air temperature [K]",
                                        "Process temperature [K]",
                                        "Rotational speed [rpm]",
                                        "Torque [Nm]", "Tool wear [min]"])

    df_ft = df.copy()
    df_ft["has_failure"] = df_ft[sel_ft].map({1: f"Has {sel_ft}", 0: f"No {sel_ft}"})

    vp_colors = ["#22c55e", "#ef4444"]
    vp_group_labels = [f"No {sel_ft}", f"Has {sel_ft}"]
    vp_groups = [df_ft[df_ft["has_failure"] == g][sel_col].dropna().values
                 for g in vp_group_labels]

    # Guard: skip if a group is empty (e.g. RNF has very few positives)
    if all(len(g) > 1 for g in vp_groups):
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        vp = ax2.violinplot(vp_groups, showmedians=True)
        for i, pc in enumerate(vp["bodies"]):
            pc.set_facecolor(vp_colors[i])
            pc.set_alpha(0.7)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(vp_group_labels)
        ax2.set_ylabel(sel_col)
        ax2.set_title(f"{sel_col} — {sel_ft} vs No {sel_ft}", fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    else:
        st.warning(f"⚠️ Not enough samples for '{sel_ft}' to plot a violin chart. Try a different failure type.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Dataset Overview":
    st.header("📈 Dataset Overview")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Samples",    f"{len(df):,}")
    col2.metric("Total Failures",   f"{df['Machine failure'].sum():,}")
    col3.metric("Failure Rate",     f"{df['Machine failure'].mean()*100:.2f}%")
    col4.metric("Features (model)", str(len(feat_names)))
    col5.metric("Missing Values",   str(df.isnull().sum().sum()))

    st.divider()

    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("🗂️ Failure Class Distribution")
        # Sort by index so 0=No Failure is always first, 1=Failure always second
        counts = df["Machine failure"].value_counts().sort_index()
        labels = ["No Failure", "Failure"]
        bar_colors = ["#22c55e", "#ef4444"]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, counts.values,
                      color=bar_colors, edgecolor="white", width=0.45)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{v:,}", ha="center", fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title("Machine Failure Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c_right:
        st.subheader("🏷️ Failure Rate by Product Type")
        fail_type = df.groupby("Type")["Machine failure"].mean().sort_values() * 100
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        type_colors = {"L": "#3b82f6", "M": "#f97316", "H": "#ef4444"}
        bars2 = ax2.bar(fail_type.index,
                        fail_type.values,
                        color=[type_colors.get(t, "#94a3b8") for t in fail_type.index],
                        edgecolor="white", width=0.45)
        for bar, val in zip(bars2, fail_type.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}%", ha="center", fontweight="bold")
        ax2.set_ylabel("Failure Rate (%)")
        ax2.set_title("Failure Rate by Product Type")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Descriptive stats
    st.subheader("📋 Descriptive Statistics")
    num_cols_raw = ["Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    st.dataframe(df[num_cols_raw].describe().round(3), use_container_width=True)

    # Correlation with failure
    st.subheader("🔗 Feature Correlation with Machine Failure")
    corr_vals = df[num_cols_raw + ["Machine failure"]].corr()["Machine failure"].drop("Machine failure")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    colors3 = ["#ef4444" if v > 0 else "#3b82f6" for v in corr_vals.values]
    ax3.barh(corr_vals.index, corr_vals.values, color=colors3, edgecolor="white")
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("Pearson Correlation")
    ax3.set_title("Feature Correlation with Machine Failure", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Failure types summary
    st.subheader("⚠️ Failure Type Counts")
    ft_df = df[["TWF","HDF","PWF","OSF","RNF"]].sum().reset_index()
    ft_df.columns = ["Failure Type", "Count"]
    ft_df["Rate (%)"] = (ft_df["Count"] / len(df) * 100).round(3)
    st.dataframe(ft_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("📁 Dataset: AI4I 2020\n🤖 Model: model_artifacts/\n📊 10,000 samples · 5 failure types")
