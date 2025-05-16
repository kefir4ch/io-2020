import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Optional imports for advanced regression and interactivity
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

try:
    import mplcursors
    _HAS_MPLCURSORS = True
except ImportError:
    _HAS_MPLCURSORS = False


def load_and_clean(
    csv_path: str,
    row_label_map: dict[str, str],
    sector_selector: dict = None,
    price_scale: float = 1.0,
    labor_scale: float = 1.0,
    omit_sectors: list[str] | None = None
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='latin1')
    df.set_index(df.columns[0], inplace=True)
    if sector_selector is None:
        sector_cols = df.columns[:19]
    elif 'cols' in sector_selector:
        sector_cols = sector_selector['cols']
    elif 'regex' in sector_selector:
        sector_cols = df.filter(regex=sector_selector['regex']).columns
    elif 'range' in sector_selector:
        start, end = sector_selector['range']
        sector_cols = df.columns[start:end]
    else:
        raise ValueError("sector_selector must have 'cols','regex','range'.")
    sector_codes = [c.strip()[0] for c in sector_cols]
    omit_sectors = omit_sectors or []
    invalid = set(omit_sectors) - set(sector_codes)
    if invalid:
        warnings.warn(f"Omit sectors not found and ignored: {invalid}")
    for key, label in row_label_map.items():
        if label not in df.index:
            raise KeyError(f"Row label '{label}' for '{key}' not in table.")
    prices = df.loc[row_label_map['price'], sector_cols].astype(float) * price_scale
    labor = df.loc[row_label_map['hours'], sector_cols].astype(float) * labor_scale
    interm = df.loc[df.index[:len(sector_cols)], sector_cols].apply(pd.to_numeric, errors='coerce')
    mat_in = interm.sum(axis=0)
    cap = df.loc[row_label_map['fixed_capital'], sector_cols].astype(float)
    wages = df.loc[row_label_map['wages'], sector_cols].astype(float)
    occ = (mat_in + cap) / wages
    data = pd.DataFrame({'Sector': sector_codes, 'Price': prices.values, 'Labor': labor.values, 'OCC': occ.values})
    data = data[~data['Sector'].isin(omit_sectors)].reset_index(drop=True)
    mask = (data['Price'] > 0) & (data['Labor'] > 0)
    if not mask.all():
        warnings.warn(f"Dropping {(~mask).sum()} nonpositive entries.")
    return data[mask].copy()


def compute_log_relationship(
    data: pd.DataFrame,
    log_base: float = 10,
    normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, float, float, float, bool]:
    log_fn = np.log10 if log_base == 10 else (np.log if log_base == np.e else (lambda x: np.log(x)/np.log(log_base)))
    log_p = log_fn(data['Price'])
    log_l = log_fn(data['Labor'])
    if normalize:
        x = log_l / log_l.max()
        y = log_p / log_p.max()
    else:
        x, y = log_l.to_numpy(), log_p.to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    r2 = r2_score(y, slope*x + intercept)
    return x, y, slope, intercept, r2, normalize


def regression_stats(
    data: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray
):
    if sm is None:
        warnings.warn("statsmodels not installed; skipping detailed stats.")
        return None
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.summary()


def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
    r2: float,
    data: pd.DataFrame,
    normalize: bool,
    annotate: bool = True,
    colormap: str = 'viridis',
    figsize: tuple[int, int] = (10, 6),
    show_interactive: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    scat = ax.scatter(x, y, c=data['OCC'], cmap=colormap,
                      s=100, edgecolors='face', alpha=0.8)
    fig.colorbar(scat, ax=ax, label='OCC')
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope*xs + intercept, 'b-',
            label=f'y={slope:.2f}x+{intercept:.2f} (R²={r2:.2f})')
    if annotate:
        for i, row in data.iterrows():
            ax.annotate(row['Sector'], (x[i], y[i]), xytext=(0,5), textcoords='offset points', ha='center')
    ax.set_xlabel('Normalized log labor' if normalize else 'Log labor')
    ax.set_ylabel('Normalized log price' if normalize else 'Log price')
    ax.set_title('Log–Log Price vs. Labor')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if show_interactive and _HAS_MPLCURSORS:
        mplcursors.cursor(scat, hover=True)
    return fig, ax


def analyze_price_value_table(
    csv_path: str = 'iotable.csv',
    row_label_map: dict[str, str] | None = None,
    sector_selector: dict | None = None,
    omit_sectors: list[str] | None = None,
    price_scale: float = 1.0,
    labor_scale: float = 1.0,
    log_base: float = 10,
    normalize: bool = True,
    annotate: bool = True,
    colormap: str = 'viridis',
    figsize: tuple[int, int] = (10, 6),
    show_interactive: bool = False
) -> tuple[pd.DataFrame, float, float, float, plt.Figure, plt.Axes]:
    default_map = {
        'price': 'Gross_Output', 'hours': 'Working_Hours',
        'fixed_capital': 'Poraba stalnega kapitala', 'wages': 'Total_Wages'
    }
    if row_label_map is None:
        row_label_map = default_map
    data = load_and_clean(
        csv_path, row_label_map, sector_selector,
        price_scale, labor_scale, omit_sectors
    )
    x, y, slope, intercept, r2, used_norm = compute_log_relationship(
        data, log_base, normalize
    )
    fig, ax = plot_results(
        x, y, slope, intercept, r2, data,
        used_norm, annotate, colormap, figsize, show_interactive
    )
    if sm:
        summary = regression_stats(data, x, y)
        print(summary)
    return data, slope, intercept, r2, fig, ax


if __name__ == '__main__':
    df_res, m, b, r2val, fig, ax = analyze_price_value_table(
        'iotable.csv', omit_sectors=['L'], price_scale=1.0,
        labor_scale=1.0, normalize=True, show_interactive=False
    )
    print(f"Fit: y={m:.4f}x+{b:.4f}, R²={r2val:.4f}")
    print("Comparison table (Sector | Price | Labor | OCC):")
    print(df_res[['Sector', 'Price', 'Labor', 'OCC']].to_string(index=False))
    plt.show()
    # Print comparison table of Sector, Price, Labor, and OCC
    
