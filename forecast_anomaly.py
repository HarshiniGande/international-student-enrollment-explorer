"""
International Student Enrollment: Forecasting & Anomaly Detection
Extends analysis.py. Reads its CSV outputs and writes three new ones:

  output/13_forecast_total_enrollment.csv   2025-2027 forecast with 95% intervals
  output/14_forecast_backtest.csv           holdout test (train 2000-2021, test 2022-2024)
  output/15_enrollment_anomalies.csv        unusual country-year swings

Run after analysis.py:  python forecast_anomaly.py
Only needs numpy, pandas, and scipy.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "output")

COVID_YEARS = [2020, 2021]
HORIZON = [2025, 2026, 2027]
BACKTEST_SPLIT = 2021          # train on <= 2021, test on 2022-2024


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
trend = pd.read_csv(os.path.join(OUT, "01_total_enrollment_trend.csv"))
trend = trend[["year", "total_enrollment"]].dropna().sort_values("year").reset_index(drop=True)
countries = pd.read_csv(os.path.join(OUT, "12_all_countries_long.csv"))

print("=" * 55)
print(f"Loaded {len(trend)} years of totals ({trend.year.min()}-{trend.year.max()})")
print(f"Loaded {countries['place'].nunique()} countries, {len(countries)} country-year rows")


# ─────────────────────────────────────────────
# 2. MODEL A: LINEAR TREND + COVID INDICATOR (OLS)
# ─────────────────────────────────────────────
def fit_ols(df):
    """y = b0 + b1*year_idx + b2*covid. Returns a dict with what we need to predict."""
    x0 = df["year"].min()
    X = np.column_stack([
        np.ones(len(df)),
        df["year"] - x0,
        df["year"].isin(COVID_YEARS).astype(int),
    ])
    y = df["total_enrollment"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    sigma2 = resid @ resid / (n - p)
    XtX_inv = np.linalg.inv(X.T @ X)
    return {"beta": beta, "x0": x0, "sigma2": sigma2, "XtX_inv": XtX_inv, "dof": n - p,
            "r2": 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)}


def predict_ols(model, years, level=0.95):
    Xf = np.column_stack([
        np.ones(len(years)),
        np.array(years) - model["x0"],
        np.isin(years, COVID_YEARS).astype(int),
    ])
    mean = Xf @ model["beta"]
    # prediction interval for a new observation
    se = np.sqrt(model["sigma2"] * (1 + np.sum(Xf @ model["XtX_inv"] * Xf, axis=1)))
    t = stats.t.ppf(0.5 + level / 2, model["dof"])
    return mean, mean - t * se, mean + t * se


# ─────────────────────────────────────────────
# 3. MODEL B: HOLT'S LINEAR EXPONENTIAL SMOOTHING
# ─────────────────────────────────────────────
def holt_sse(y, alpha, beta):
    level, trend_ = y[0], y[1] - y[0]
    sse, errors = 0.0, []
    for t in range(1, len(y)):
        fcst = level + trend_
        err = y[t] - fcst
        errors.append(err)
        sse += err ** 2
        new_level = alpha * y[t] + (1 - alpha) * (level + trend_)
        trend_ = beta * (new_level - level) + (1 - beta) * trend_
        level = new_level
    return sse, level, trend_, np.array(errors)


def fit_holt(df):
    """Grid-search alpha and beta to minimise one-step-ahead squared error."""
    y = df["total_enrollment"].values.astype(float)
    best = None
    for a in np.linspace(0.05, 0.95, 19):
        for b in np.linspace(0.05, 0.95, 19):
            sse, level, tr, errs = holt_sse(y, a, b)
            if best is None or sse < best["sse"]:
                best = {"alpha": a, "beta": b, "sse": sse, "level": level,
                        "trend": tr, "sigma": errs.std(ddof=1)}
    best["last_year"] = int(df["year"].max())
    return best


def predict_holt(model, years, level=0.95):
    h = np.array(years) - model["last_year"]
    mean = model["level"] + h * model["trend"]
    z = stats.norm.ppf(0.5 + level / 2)
    # approximate interval: one-step error grows with the square root of the horizon
    half = z * model["sigma"] * np.sqrt(h)
    return mean, mean - half, mean + half


# ─────────────────────────────────────────────
# 4. BACKTEST: WHICH MODEL WOULD HAVE PREDICTED 2022-2024 BETTER?
# ─────────────────────────────────────────────
print("\nSTEP 1: Backtest (train 2000-2021, test 2022-2024)")
train = trend[trend.year <= BACKTEST_SPLIT]
test = trend[trend.year > BACKTEST_SPLIT]
test_years = test["year"].tolist()
actual = test["total_enrollment"].values

ols_bt = predict_ols(fit_ols(train), test_years)[0]
holt_bt = predict_holt(fit_holt(train), test_years)[0]


def mape(a, f):
    return float(np.mean(np.abs((a - f) / a)) * 100)


scores = {"OLS trend + COVID indicator": mape(actual, ols_bt),
          "Holt exponential smoothing": mape(actual, holt_bt)}
for name, s in scores.items():
    print(f"  {name:30s} MAPE = {s:.1f}%")
best_name = min(scores, key=scores.get)
print(f"  Best on holdout: {best_name}")

backtest = pd.DataFrame({
    "year": test_years,
    "actual": actual,
    "ols_forecast": ols_bt.round(0),
    "holt_forecast": holt_bt.round(0),
})
backtest["ols_pct_error"] = ((backtest.ols_forecast - backtest.actual) / backtest.actual * 100).round(2)
backtest["holt_pct_error"] = ((backtest.holt_forecast - backtest.actual) / backtest.actual * 100).round(2)
backtest.to_csv(os.path.join(OUT, "14_forecast_backtest.csv"), index=False)


# ─────────────────────────────────────────────
# 5. FINAL FORECAST 2025-2027 (refit on all data)
# ─────────────────────────────────────────────
print("\nSTEP 2: Forecast 2025-2027 (refit on 2000-2024)")
ols_full = fit_ols(trend)
holt_full = fit_holt(trend)
o_mean, o_lo, o_hi = predict_ols(ols_full, HORIZON)
h_mean, h_lo, h_hi = predict_holt(holt_full, HORIZON)

forecast = pd.DataFrame({
    "year": HORIZON,
    "ols_forecast": o_mean.round(0), "ols_lower_95": o_lo.round(0), "ols_upper_95": o_hi.round(0),
    "holt_forecast": h_mean.round(0), "holt_lower_95": h_lo.round(0), "holt_upper_95": h_hi.round(0),
})
forecast["recommended_model"] = best_name
forecast.to_csv(os.path.join(OUT, "13_forecast_total_enrollment.csv"), index=False)
print(f"  OLS R² on 2000-2024: {ols_full['r2']:.3f}")
print(f"  Holt alpha={holt_full['alpha']:.2f}, beta={holt_full['beta']:.2f}")
print(forecast.to_string(index=False))


# ─────────────────────────────────────────────
# 6. ANOMALY DETECTION: UNUSUAL COUNTRY-YEAR SWINGS
# ─────────────────────────────────────────────
# For each country, compare each year-over-year % change with that country's own
# typical change, using a robust z-score (median and MAD), so one huge year does
# not hide the others. Small senders are skipped because tiny counts swing wildly.
print("\nSTEP 3: Anomaly detection (robust z-score on year-over-year change)")
Z_CUTOFF = 3.5
MIN_PRIOR = 1000        # prior-year students needed to be considered
MIN_ABS_CHANGE = 500    # ignore swings smaller than this many students

c = countries.dropna(subset=["enrollment"]).sort_values(["place", "year"]).copy()
c["prior_year"] = c.groupby("place")["year"].shift(1)
c["prior_enrollment"] = c.groupby("place")["enrollment"].shift(1)
c = c[c["year"] - c["prior_year"] == 1]          # only consecutive years
c["abs_change"] = c["enrollment"] - c["prior_enrollment"]
c["pct_change"] = c["abs_change"] / c["prior_enrollment"] * 100


def robust_z(x):
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


c["robust_z"] = c.groupby("place")["pct_change"].transform(robust_z)
flags = c[(c["robust_z"].abs() >= Z_CUTOFF)
          & (c["prior_enrollment"] >= MIN_PRIOR)
          & (c["abs_change"].abs() >= MIN_ABS_CHANGE)].copy()
flags["direction"] = np.where(flags["pct_change"] > 0, "spike", "drop")
flags["covid_year"] = flags["year"].isin(COVID_YEARS)
flags = flags.sort_values("robust_z", key=lambda s: s.abs(), ascending=False)

cols = ["place", "year", "prior_enrollment", "enrollment", "abs_change",
        "pct_change", "robust_z", "direction", "covid_year"]
flags[cols].round(2).to_csv(os.path.join(OUT, "15_enrollment_anomalies.csv"), index=False)

print(f"  Country-years checked: {len(c):,}")
print(f"  Anomalies flagged:     {len(flags)} "
      f"({int(flags['covid_year'].sum())} in COVID years, {int((~flags['covid_year']).sum())} outside)")
print(flags[cols].head(10).round(1).to_string(index=False))

print("\n" + "=" * 55)
print("Wrote output/13_forecast_total_enrollment.csv, 14_forecast_backtest.csv, 15_enrollment_anomalies.csv")
