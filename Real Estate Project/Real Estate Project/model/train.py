import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
EDA_DIR    = os.path.join(ROOT_DIR, 'eda_output')
MODEL_DIR  = BASE_DIR
os.makedirs(EDA_DIR, exist_ok=True)

sns.set_theme(style='darkgrid', palette='muted')
ACCENT = '#e94560'
PALETTE = ['#e94560', '#0f3460', '#533483', '#06d6a0', '#ffd166']

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. LOAD DATA
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("=" * 60)
print("  HOUSE PRICE PREDICTION â€“ ML PIPELINE")
print("=" * 60)

print("\n[1/7] Loading California Housing dataset â€¦")
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df.rename(columns={'MedHouseVal': 'Price'}, inplace=True)
print(f"      Shape: {df.shape}")
print(df.head())

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. DATA CLEANING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n[2/7] Data Cleaning â€¦")
print(f"      Missing values:\n{df.isnull().sum()}")
print(f"      Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print(f"      Shape after cleaning: {df.shape}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. EXPLORATORY DATA ANALYSIS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n[3/7] Exploratory Data Analysis â€¦")

# â”€â”€ 3a. Price Distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

axes[0].hist(df['Price'], bins=50, color=ACCENT, edgecolor='#1a1a2e', alpha=0.9)
axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Median House Value ($100k)')
axes[0].set_ylabel('Frequency')

axes[1].hist(np.log1p(df['Price']), bins=50, color='#06d6a0', edgecolor='#1a1a2e', alpha=0.9)
axes[1].set_title('Log(Price) Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('log(Median House Value)')
axes[1].set_ylabel('Frequency')

plt.suptitle('House Price Distribution Analysis', color='white', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'price_distribution.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("      âœ“ Saved: price_distribution.png")

# â”€â”€ 3b. Correlation Heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt='.2f',
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'color': 'white', 'size': 9})
ax.set_title('Feature Correlation Heatmap', color='white', fontsize=16, fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'correlation_heatmap.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("      âœ“ Saved: correlation_heatmap.png")

# â”€â”€ 3c. Feature vs Price Scatter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
features = [c for c in df.columns if c != 'Price']
n = len(features)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
fig.patch.set_facecolor('#1a1a2e')
axes_flat = axes.flatten()
for i, feat in enumerate(features):
    ax = axes_flat[i]
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.scatter(df[feat], df['Price'], alpha=0.3, s=5, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(feat)
    ax.set_ylabel('Price ($100k)')
    ax.set_title(f'{feat} vs Price', fontsize=10, fontweight='bold')
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.suptitle('Feature vs Price Scatter Plots', color='white', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'feature_vs_price.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("      âœ“ Saved: feature_vs_price.png")

# â”€â”€ 3d. Geographic Price Map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
sc = ax.scatter(df['Longitude'], df['Latitude'], c=df['Price'],
                cmap='plasma', s=3, alpha=0.6)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Median House Value ($100k)', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
ax.set_xlabel('Longitude', color='white')
ax.set_ylabel('Latitude', color='white')
ax.set_title('California House Prices by Location', color='white', fontsize=16, fontweight='bold')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'geo_price_map.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("      âœ“ Saved: geo_price_map.png")

# â”€â”€ 3e. Box Plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.patch.set_facecolor('#1a1a2e')
axes_flat = axes.flatten()
for i, feat in enumerate(features):
    ax = axes_flat[i]
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    bp = ax.boxplot(df[feat].dropna(), patch_artist=True,
                    medianprops=dict(color=ACCENT, linewidth=2),
                    boxprops=dict(facecolor='#0f3460', color='white'),
                    whiskerprops=dict(color='white'),
                    capprops=dict(color='white'),
                    flierprops=dict(marker='o', color=ACCENT, alpha=0.3, markersize=2))
    ax.set_title(feat, fontsize=10, fontweight='bold')
plt.suptitle('Feature Box Plots', color='white', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'box_plots.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("      âœ“ Saved: box_plots.png")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. OUTLIER REMOVAL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n[4/7] Removing Outliers â€¦")
shape_before = df.shape[0]

# IQR method on Price
Q1, Q3 = df['Price'].quantile(0.25), df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)]

# Z-score on numerical features (|z| > 3.5)
z_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
z_scores = np.abs(stats.zscore(df[z_cols]))
df = df[(z_scores < 3.5).all(axis=1)]

shape_after = df.shape[0]
print(f"      Removed {shape_before - shape_after} outlier rows")
print(f"      Shape after outlier removal: {df.shape}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. FEATURE ENGINEERING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n[5/7] Feature Engineering â€¦")
df['RoomsPerHousehold']     = df['AveRooms']   / (df['HouseAge'] + 1)
df['BedroomsPerRoom']       = df['AveBedrms']  / (df['AveRooms'] + 1e-5)
df['PopulationPerHousehold']= df['Population'] / (df['AveOccup'] + 1e-5)
print(f"      Added 3 engineered features. Final shape: {df.shape}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6. TRAIN / TEST SPLIT & SCALING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
FEATURE_COLS = [c for c in df.columns if c != 'Price']
X = df[FEATURE_COLS]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 7. MODEL TRAINING & EVALUATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n[6/7] Training & Evaluating Models â€¦")

models = {
    'Linear Regression':     LinearRegression(),
    'Ridge Regression':      Ridge(alpha=1.0),
    'Random Forest':         RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':     GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    'SVR':                   SVR(kernel='rbf', C=10, epsilon=0.1),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Model':<25} {'RÂ²':>8} {'MAE':>8} {'RMSE':>8} {'CV RÂ²':>8}")
print("  " + "-" * 60)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(model, X_train_s, y_train, cv=kf, scoring='r2', n_jobs=-1).mean()
    results[name] = {'model': model, 'r2': r2, 'mae': mae, 'rmse': rmse, 'cv': cv, 'y_pred': y_pred}
    print(f"  {name:<25} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f} {cv:>8.4f}")

# â”€â”€ Model Comparison Plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a2e')
metrics = ['r2', 'mae', 'rmse']
metric_labels = ['RÂ² Score', 'MAE', 'RMSE']
model_names = list(results.keys())
colors = PALETTE

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=colors, edgecolor='#1a1a2e', linewidth=0.5)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=8)

plt.suptitle('Model Performance Comparison', color='white', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'model_comparison.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("\n      âœ“ Saved: model_comparison.png")

# â”€â”€ Select Best Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best_name = max(results, key=lambda k: results[k]['r2'])
best      = results[best_name]
print(f"\n[7/7] Best Model: {best_name}")
print(f"      RÂ²={best['r2']:.4f}  MAE={best['mae']:.4f}  RMSE={best['rmse']:.4f}")

# â”€â”€ Actual vs Predicted Plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.scatter(y_test, best['y_pred'], alpha=0.4, s=15, color=ACCENT, label='Predictions')
lims = [min(y_test.min(), best['y_pred'].min()), max(y_test.max(), best['y_pred'].max())]
ax.plot(lims, lims, 'w--', linewidth=1.5, label='Perfect Fit')
ax.set_xlabel('Actual Price ($100k)', color='white')
ax.set_ylabel('Predicted Price ($100k)', color='white')
ax.set_title(f'Actual vs Predicted â€“ {best_name}', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white')
ax.legend(facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'actual_vs_predicted.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("      âœ“ Saved: actual_vs_predicted.png")

# â”€â”€ Feature Importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if hasattr(best['model'], 'feature_importances_'):
    importances = best['model'].feature_importances_
    feat_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances})
    feat_df.sort_values('Importance', ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(feat_df))],
                   edgecolor='#1a1a2e')
    ax.set_title(f'Feature Importances â€“ {best_name}', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    for bar, val in zip(bars, feat_df['Importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='white', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'feature_importance.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("      âœ“ Saved: feature_importance.png")

# â”€â”€ Residuals Plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
residuals = y_test.values - best['y_pred']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

axes[0].scatter(best['y_pred'], residuals, alpha=0.4, s=10, color='#ffd166')
axes[0].axhline(0, color=ACCENT, linewidth=1.5, linestyle='--')
axes[0].set_xlabel('Predicted Value')
axes[0].set_ylabel('Residual')
axes[0].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')

axes[1].hist(residuals, bins=50, color='#533483', edgecolor='#1a1a2e', alpha=0.9)
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')

plt.suptitle('Residual Analysis', color='white', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, 'residuals.png'), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("      âœ“ Saved: residuals.png")

# â”€â”€ Save Model & Metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_data = {
    'model':        best['model'],
    'scaler':       scaler,
    'feature_cols': FEATURE_COLS,
    'best_name':    best_name,
    'metrics': {
        'r2':   round(best['r2'],  4),
        'mae':  round(best['mae'], 4),
        'rmse': round(best['rmse'],4),
        'cv':   round(best['cv'],  4),
    },
    'all_results': {
        name: {'r2': round(v['r2'],4), 'mae': round(v['mae'],4),
               'rmse': round(v['rmse'],4), 'cv': round(v['cv'],4)}
        for name, v in results.items()
    },
}
out_path = os.path.join(MODEL_DIR, 'house_model.pkl')
joblib.dump(model_data, out_path)
print(f"\nâœ… Model saved â†’ {out_path}")
print("\nðŸŽ‰ Pipeline complete! All plots saved to eda_output/\n")

