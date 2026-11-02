# International Student Enrollment Explorer

An interactive analysis of 25 years of international student enrollment in US higher education (2000-2024), built from the public **IIE Open Doors 2025** census data.

**Live demo:** https://harshinigande.github.io/international-student-enrollment-explorer/

![Python](https://img.shields.io/badge/Python-pandas%20%7C%20scikit--learn-blue) ![D3.js](https://img.shields.io/badge/D3.js-v7-orange) ![Tableau](https://img.shields.io/badge/Tableau-workbook-lightgrey)

## Why I built this

I work with international student data day to day and wanted to see the national picture: which countries send the most students, who is growing, who is shrinking, and how much COVID changed the trend. Nobody assigned this project; I built it to answer those questions for myself and to practice turning messy public data into something a non-technical reader can explore.

## What's inside

The web page has 8 interactive sections:

| # | View | What it shows |
|---|------|---------------|
| 01 | Time series | Total enrollment 2000-2024, annotated with the COVID drop |
| 02 | Animated bar race | Top 15 countries by year (India overtakes China in 2023) |
| 03 | K-Means clusters | Countries grouped into 4 growth patterns |
| 04 | Field of study | Stacked area chart of the top 8 fields |
| 05 | Regional share | Share of enrollment by world region |
| 06 | Regression | Actual vs. predicted enrollment with a COVID dummy variable |
| 07 | Geospatial map | Choropleth of enrollment by country for 2000, 2010, 2019, 2024 and 24-year growth (CAGR), with zoom and tooltips |
| 08 | Forecast and anomalies | 2025-2027 forecast with 95% intervals, a backtest of two models, and flagged country-year anomalies |

## Key findings

- **1.18M** international students in 2024, a new record after recovering from COVID.
- Enrollment fell about **15% in 2020**, the sharpest single-year decline in the data.
- A linear trend with a COVID dummy explains **92% of the variance** (R² = 0.92), about **+30,700 students per year** on trend.
- **India** became the largest sender in 2023 and reached **363,019** students in 2024.
- **South Asia** overtook East Asia as the largest sending region in 2024 (36.1% vs. 33.5% share).
- Among countries with 1,000+ students, **Vietnam** grew fastest since 2000 (about +11% per year); **30 countries** sent fewer students in 2024 than in 2000.
- The trend model forecasts about **1.30M students by 2027** (95% interval 1.15M to 1.46M). It beat Holt exponential smoothing on a 2022-2024 holdout (MAPE 10.7% vs. 26.3%).
- Anomaly detection flags swings such as **Brazil +78% in 2014** and **-32% in 2016**, and **Saudi Arabia +129% in 2006**, which line up with large government scholarship programs starting and ending.

## Methods

1. **Cleaning:** parsed two messy IIE Excel files (merged headers, dash placeholders, footnote digits in field names, region subtotal rows mixed with countries) with `openpyxl` and `pandas`.
2. **Reshaping:** converted wide year columns to long format; separated countries from region subtotals.
3. **Metrics:** CAGR (2000-2024 and post-COVID), peak year, averages, regional shares, STEM vs. non-STEM split.
4. **Statistics:** Pearson correlations, OLS regression with a COVID indicator, K-Means clustering (k = 4, chosen with the elbow method) on standardized features.
5. **Outputs:** 12 analysis-ready CSVs for Tableau and D3.js.
6. **Forecasting** (`forecast_anomaly.py`): OLS trend with a COVID indicator vs. Holt linear exponential smoothing, compared on a 2022-2024 holdout (MAPE), then refit on all years to forecast 2025-2027 with 95% prediction intervals.
7. **Anomaly detection:** robust z-scores (median and MAD) of each country's year-over-year % change against its own history; flags |z| >= 3.5 for countries with 1,000+ students and swings of 500+ students.
8. **Visualization:** D3.js v7 web page (including a d3-geo choropleth on Natural Earth boundaries) and a Tableau workbook.

## Repository structure

```
index.html                 Interactive D3.js page (open in a browser)
analysis.py                Full Python pipeline: cleaning, metrics, models, CSV export
forecast_anomaly.py        Forecasting (backtested) and anomaly detection, writes CSVs 13-15
tableau/                   Tableau workbook (.twbx)
data/                      Put the IIE Excel files here (not included)
output/                    Cleaned, analysis-ready CSVs (01-12) plus forecast and anomaly outputs (13-15)
requirements.txt
```

## How to run

**View the page:** open `index.html` in a browser (internet needed for D3 and map boundaries), or use the live demo link above.

**Re-run the analysis:**

1. Download these two files from [IIE Open Doors](https://opendoorsdata.org/) and place them in `data/`:
   - `Census_All-Places-of-Origin-Selected-Years_OD25_Website.xlsx`
   - `Census_Field-of-Study_OD25_Website.xlsx`
2. Install dependencies and run:

```bash
pip install -r requirements.txt
python analysis.py
python forecast_anomaly.py
```

The CSVs are written to `output/`. The cleaned outputs are already included there, so you can explore them without rerunning the pipeline.

## Data source

Institute of International Education (IIE), *Open Doors 2025*. Map boundaries: Natural Earth via the `world-atlas` package. This is an independent analysis and is not affiliated with IIE.

## Author

**Harshini Gande** · [LinkedIn](https://www.linkedin.com/in/harshini-gande-61572423b) · [GitHub](https://github.com/HarshiniGande)
