"""
International Student Enrollment — Full Analysis Pipeline
IIE Open Doors 2025 Data
Outputs: cleaned CSVs ready for Tableau + D3.js
"""

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ─────────────────────────────────────────────
# 1. PARSE: PLACES OF ORIGIN
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Parsing Places of Origin...")

wb = load_workbook(
    os.path.join(DATA_DIR, "Census_All-Places-of-Origin-Selected-Years_OD25_Website.xlsx"),
    read_only=True
)
ws = wb.active
all_rows = list(ws.iter_rows(values_only=True))

year_labels = [c for c in all_rows[2][3:39] if c is not None]

records = []
for row in all_rows[4:]:
    code = row[0]
    name = row[1]
    if code is None or name is None:
        continue
    if not isinstance(code, (int, float)):
        continue
    values = row[3:39]
    cleaned = []
    for v in values:
        if v is None or v == '-' or v == '\u2500' or str(v).strip() in ['-', '\u2500', '']:
            cleaned.append(np.nan)
        else:
            try:
                cleaned.append(float(v))
            except:
                cleaned.append(np.nan)

    code_int = int(code)
    is_region = (code_int % 100 == 0)

    records.append({
        "code": code_int,
        "place": name.strip(),
        "is_region": is_region,
        **{year_labels[i]: cleaned[i] for i in range(len(year_labels))}
    })

df_origin_wide = pd.DataFrame(records)
wb.close()

df_countries_wide = df_origin_wide[~df_origin_wide["is_region"]].copy()
df_regions_wide = df_origin_wide[df_origin_wide["is_region"]].copy()

print(f"  Total places parsed: {len(df_origin_wide)}")
print(f"  Countries: {len(df_countries_wide)}")
print(f"  Regions: {len(df_regions_wide)}")

# ─────────────────────────────────────────────
# 2. PARSE: FIELD OF STUDY
# ─────────────────────────────────────────────
print("\nSTEP 2: Parsing Field of Study...")

wb2 = load_workbook(
    os.path.join(DATA_DIR, "Census_Field-of-Study_OD25_Website.xlsx"),
    read_only=True
)
ws2 = wb2.active
all_rows2 = list(ws2.iter_rows(values_only=True))

fos_years = [c for c in all_rows2[2][2:29] if c is not None]

fos_records = []
seen_fields = set()
for row in all_rows2[4:]:
    field = row[0]
    if field is None or str(field).strip() == '':
        continue
    field = str(field).strip().rstrip("0123456789")
    values = row[2:29]
    cleaned = []
    for v in values:
        if v is None or v == '-':
            cleaned.append(np.nan)
        else:
            try:
                cleaned.append(float(v))
            except:
                cleaned.append(np.nan)

    if all(np.isnan(x) for x in cleaned):
        continue
    if field in seen_fields:
        continue
    seen_fields.add(field)

    fos_records.append({
        "field": field,
        **{fos_years[i]: cleaned[i] for i in range(min(len(fos_years), len(cleaned)))}
    })

df_fos_wide = pd.DataFrame(fos_records)
wb2.close()
print(f"  Fields of study parsed: {len(df_fos_wide)}")

# ─────────────────────────────────────────────
# 3. RESHAPE TO LONG FORMAT
# ─────────────────────────────────────────────
print("\nSTEP 3: Reshaping to long format...")

df_origin_long = df_origin_wide.melt(
    id_vars=["code", "place", "is_region"],
    value_vars=year_labels,
    var_name="year_label",
    value_name="enrollment"
)
df_origin_long["year"] = df_origin_long["year_label"].str[:4].astype(int)
df_origin_long = df_origin_long.dropna(subset=["enrollment"])

df_countries_long = df_origin_long[~df_origin_long["is_region"]].copy()
df_regions_long = df_origin_long[df_origin_long["is_region"]].copy()

df_fos_long = df_fos_wide.melt(
    id_vars=["field"],
    value_vars=fos_years,
    var_name="year_label",
    value_name="enrollment"
)
df_fos_long["year"] = df_fos_long["year_label"].str[:4].astype(int)
df_fos_long = df_fos_long.dropna(subset=["enrollment"])

print(f"  Country-year records: {len(df_countries_long)}")
print(f"  Field-year records: {len(df_fos_long)}")

# ─────────────────────────────────────────────
# 4. US TOTAL ENROLLMENT
# ─────────────────────────────────────────────
total_by_year = df_countries_long.groupby("year")["enrollment"].sum().reset_index()
total_by_year.columns = ["year", "total_enrollment"]
total_by_year = total_by_year.sort_values("year")

modern_years = list(range(2000, 2025))
total_modern = total_by_year[total_by_year["year"].isin(modern_years)].copy()
total_modern["yoy_growth"] = total_modern["total_enrollment"].pct_change() * 100
total_modern["covid_flag"] = total_modern["year"].isin([2020, 2021])

print("\n  Total US enrollment (recent years):")
print(total_modern[total_modern["year"] >= 2015].to_string(index=False))

# ─────────────────────────────────────────────
# 5. TOP COUNTRIES ANALYSIS
# ─────────────────────────────────────────────
print("\nSTEP 4: Top countries analysis...")

countries_modern = df_countries_long[df_countries_long["year"].isin(modern_years)].copy()
latest_year = countries_modern["year"].max()
top20_latest = (
    countries_modern[countries_modern["year"] == latest_year]
    .nlargest(20, "enrollment")
    [["place", "enrollment"]]
)
print(f"\n  Top 10 countries in {latest_year}:")
print(top20_latest.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 6. REGIONAL SHARE ANALYSIS
# ─────────────────────────────────────────────
print("\nSTEP 5: Regional share analysis...")

def map_region_by_name(place):
    place_upper = place.upper()
    if any(x in place_upper for x in ["CHINA", "JAPAN", "KOREA", "TAIWAN", "VIETNAM", "THAILAND", "MALAYSIA", "INDONESIA", "PHILIPPINES", "MYANMAR", "CAMBODIA", "LAOS"]):
        return "East Asia & Pacific"
    elif any(x in place_upper for x in ["INDIA", "PAKISTAN", "BANGLADESH", "SRI LANKA", "NEPAL", "BHUTAN"]):
        return "South Asia"
    elif any(x in place_upper for x in ["SAUDI", "IRAN", "TURKEY", "ISRAEL", "JORDAN", "EGYPT", "MOROCCO", "TUNISIA", "ALGERIA", "IRAQ", "KUWAIT", "UAE", "QATAR", "BAHRAIN", "OMAN", "YEMEN", "LEBANON", "SYRIA"]):
        return "Middle East & North Africa"
    elif any(x in place_upper for x in ["NIGERIA", "KENYA", "GHANA", "ETHIOPIA", "CAMEROON", "TANZANIA", "UGANDA", "RWANDA", "CONGO", "AFRICA"]):
        return "Sub-Saharan Africa"
    elif any(x in place_upper for x in ["BRAZIL", "MEXICO", "COLOMBIA", "VENEZUELA", "PERU", "CHILE", "ARGENTINA", "ECUADOR", "BOLIVIA", "PARAGUAY", "URUGUAY", "CUBA", "JAMAICA", "TRINIDAD", "GUATEMALA", "HONDURAS", "EL SALVADOR", "NICARAGUA", "COSTA RICA", "PANAMA", "DOMINICAN"]):
        return "Latin America & Caribbean"
    elif any(x in place_upper for x in ["GERMANY", "FRANCE", "UNITED KINGDOM", "SPAIN", "ITALY", "NETHERLANDS", "SWEDEN", "NORWAY", "DENMARK", "FINLAND", "POLAND", "RUSSIA", "UKRAINE", "ROMANIA", "CZECH", "AUSTRIA", "SWITZERLAND", "BELGIUM", "PORTUGAL", "GREECE", "HUNGARY", "BULGARIA"]):
        return "Europe"
    elif any(x in place_upper for x in ["CANADA", "AUSTRALIA", "NEW ZEALAND"]):
        return "Other Developed"
    elif any(x in place_upper for x in ["KAZAKHSTAN", "UZBEKISTAN", "KYRGYZSTAN", "TAJIKISTAN", "TURKMENISTAN", "AZERBAIJAN", "ARMENIA", "GEORGIA"]):
        return "Central Asia"
    else:
        return "Other"

countries_modern["region"] = countries_modern["place"].apply(map_region_by_name)

regional_by_year = countries_modern.groupby(["year", "region"])["enrollment"].sum().reset_index()
total_by_year_ref = countries_modern.groupby("year")["enrollment"].sum().reset_index()
total_by_year_ref.columns = ["year", "total"]
regional_by_year = regional_by_year.merge(total_by_year_ref, on="year")
regional_by_year["share_pct"] = (regional_by_year["enrollment"] / regional_by_year["total"] * 100).round(2)

# ─────────────────────────────────────────────
# 7. CAGR & GROWTH METRICS PER COUNTRY
# ─────────────────────────────────────────────
print("\nSTEP 6: CAGR and growth metrics...")

def compute_cagr(start_val, end_val, years):
    if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0 or years == 0:
        return np.nan
    return ((end_val / start_val) ** (1 / years) - 1) * 100

pivot_countries = countries_modern.pivot_table(
    index="place", columns="year", values="enrollment", aggfunc="first"
)

metrics = []
for place in pivot_countries.index:
    row = pivot_countries.loc[place]
    v2000 = row.get(2000, np.nan)
    v2010 = row.get(2010, np.nan)
    v2019 = row.get(2019, np.nan)
    v2024 = row.get(2024, np.nan)
    avg = row.dropna().mean()
    peak = row.dropna().max()
    peak_year = row.dropna().idxmax() if not row.dropna().empty else np.nan

    cagr_full = compute_cagr(v2000, v2024, 24)
    cagr_post = compute_cagr(v2019, v2024, 5)

    metrics.append({
        "place": place,
        "enrollment_2000": v2000,
        "enrollment_2010": v2010,
        "enrollment_2019": v2019,
        "enrollment_2024": v2024,
        "cagr_2000_2024": round(cagr_full, 2) if not np.isnan(cagr_full) else np.nan,
        "cagr_post_covid": round(cagr_post, 2) if not np.isnan(cagr_post) else np.nan,
        "avg_enrollment": round(avg, 0),
        "peak_enrollment": peak,
        "peak_year": peak_year
    })

df_metrics = pd.DataFrame(metrics)

# ─────────────────────────────────────────────
# 8. CORRELATION ANALYSIS
# ─────────────────────────────────────────────
print("\nSTEP 7: Correlation analysis...")

df_corr = df_metrics.dropna(subset=["enrollment_2024", "cagr_2000_2024", "avg_enrollment"])
corr_enroll_cagr, p_enroll_cagr = pearsonr(df_corr["enrollment_2024"], df_corr["cagr_2000_2024"])
corr_enroll_avg,  p_enroll_avg  = pearsonr(df_corr["enrollment_2024"], df_corr["avg_enrollment"])
print(f"  Enrollment 2024 vs CAGR:     r = {corr_enroll_cagr:.3f}, p = {p_enroll_cagr:.4f}")
print(f"  Enrollment 2024 vs Avg:      r = {corr_enroll_avg:.3f}, p = {p_enroll_avg:.4f}")

fos_pivot = df_fos_long.pivot_table(index="year", columns="field", values="enrollment", aggfunc="sum")
fos_corr_matrix = fos_pivot.corr()
print(f"  FoS correlation matrix shape: {fos_corr_matrix.shape}")

# ─────────────────────────────────────────────
# 9. LINEAR REGRESSION
# ─────────────────────────────────────────────
print("\nSTEP 8: Regression...")

reg_df = total_modern.dropna(subset=["total_enrollment"]).copy()
reg_df["year_idx"] = reg_df["year"] - reg_df["year"].min()
reg_df["is_covid"] = reg_df["year"].isin([2020, 2021]).astype(int)

X = reg_df[["year_idx", "is_covid"]].values
y = reg_df["total_enrollment"].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"  R² = {r2:.4f}")
print(f"  Coefficient (year trend): {model.coef_[0]:,.0f} students/year")
print(f"  Coefficient (COVID penalty): {model.coef_[1]:,.0f} students")

reg_df["predicted"] = y_pred
reg_df["residual"] = y - y_pred

# ─────────────────────────────────────────────
# 10. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
print("\nSTEP 9: K-Means clustering...")

cluster_df = df_metrics.dropna(subset=["enrollment_2000", "enrollment_2024", "cagr_2000_2024"]).copy()
cluster_df = cluster_df[cluster_df["enrollment_2024"] >= 500]

features = ["enrollment_2024", "cagr_2000_2024", "avg_enrollment"]
X_cluster = cluster_df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

inertias = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
print("  Inertias k=2..7:", [round(x, 1) for x in inertias])

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_df["cluster"] = km4.fit_predict(X_scaled)

cluster_summary = cluster_df.groupby("cluster").agg(
    countries=("place", "count"),
    avg_2024=("enrollment_2024", "mean"),
    avg_cagr=("cagr_2000_2024", "mean")
).round(1)
print("\n  Cluster summary:")
print(cluster_summary)

cluster_labels = {}
for c in cluster_df["cluster"].unique():
    subset = cluster_summary.loc[c]
    if subset["avg_2024"] > 150000:
        cluster_labels[c] = "Dominant (100k+)"
    elif subset["avg_cagr"] > 8:
        cluster_labels[c] = "Rising Stars"
    elif subset["avg_cagr"] < 0:
        cluster_labels[c] = "Declining"
    else:
        cluster_labels[c] = "Stable Mid-Tier"

cluster_df["cluster_label"] = cluster_df["cluster"].map(cluster_labels)
print("\n  Top countries per cluster:")
for label in cluster_df["cluster_label"].unique():
    top = cluster_df[cluster_df["cluster_label"] == label].nlargest(5, "enrollment_2024")["place"].tolist()
    print(f"    {label}: {', '.join(top)}")

# ─────────────────────────────────────────────
# 11. STEM vs NON-STEM
# ─────────────────────────────────────────────
print("\nSTEP 10: STEM classification...")

STEM_FIELDS = [
    "Engineering", "Mathematics and Computer Science", "Physical and Life Sciences",
    "Natural Resources and Conservation", "Agriculture", "Health Professions",
    "Engineering Technologies/Technicians", "Computer and Information Sciences"
]
df_fos_long["is_stem"] = df_fos_long["field"].apply(
    lambda f: any(s.lower() in f.lower() for s in STEM_FIELDS)
)
stem_by_year = df_fos_long.groupby(["year", "is_stem"])["enrollment"].sum().reset_index()
stem_by_year["stem_label"] = stem_by_year["is_stem"].map({True: "STEM", False: "Non-STEM"})

# ─────────────────────────────────────────────
# 12. EXPORT ALL CSVs
# ─────────────────────────────────────────────
print("\nSTEP 11: Exporting CSVs...")

out = OUTPUT_DIR
os.makedirs(out, exist_ok=True)

reg_df[["year", "total_enrollment", "yoy_growth", "covid_flag", "predicted", "residual"]].to_csv(f"{out}/01_total_enrollment_trend.csv", index=False)

top_countries_list = top20_latest["place"].tolist()
top_countries_ts = countries_modern[countries_modern["place"].isin(top_countries_list)].copy()
top_countries_ts[["place", "year", "enrollment", "region"]].to_csv(f"{out}/02_top_countries_timeseries.csv", index=False)

cluster_export = cluster_df[["place", "enrollment_2024", "enrollment_2019",
                              "enrollment_2010", "enrollment_2000",
                              "cagr_2000_2024", "cagr_post_covid",
                              "avg_enrollment", "peak_year",
                              "cluster", "cluster_label"]].copy()
cluster_export["region"] = cluster_export["place"].apply(map_region_by_name)
cluster_export.to_csv(f"{out}/03_countries_clustered.csv", index=False)

df_fos_long[["field", "year", "enrollment", "is_stem"]].to_csv(f"{out}/04_field_of_study.csv", index=False)
regional_by_year.to_csv(f"{out}/05_regional_share.csv", index=False)

bar_race_countries = (
    countries_modern[countries_modern["year"] == latest_year]
    .nlargest(15, "enrollment")["place"].tolist()
)
bar_race_df = countries_modern[countries_modern["place"].isin(bar_race_countries)].copy()
bar_race_df[["place", "year", "enrollment", "region"]].to_csv(f"{out}/06_bar_race_data.csv", index=False)

iso3_lookup = {
    "China": "CHN", "India": "IND", "South Korea": "KOR", "Saudi Arabia": "SAU",
    "Canada": "CAN", "Vietnam": "VNM", "Taiwan": "TWN", "Japan": "JPN",
    "Brazil": "BRA", "Iran": "IRN", "Mexico": "MEX", "Nigeria": "NGA",
    "Germany": "DEU", "United Kingdom": "GBR", "France": "FRA",
    "Bangladesh": "BGD", "Turkey": "TUR", "Pakistan": "PAK",
    "Indonesia": "IDN", "Thailand": "THA", "Malaysia": "MYS",
    "Nepal": "NPL", "Hong Kong": "HKG", "Colombia": "COL",
    "Ghana": "GHA", "Kenya": "KEN", "Egypt": "EGY", "Ethiopia": "ETH",
    "Sri Lanka": "LKA", "Russia": "RUS", "Ukraine": "UKR", "Spain": "ESP",
    "Italy": "ITA", "Sweden": "SWE", "Australia": "AUS", "New Zealand": "NZL",
    "Norway": "NOR", "Netherlands": "NLD", "Israel": "ISR", "Jordan": "JOR",
    "Kuwait": "KWT", "Morocco": "MAR", "Algeria": "DZA", "Tunisia": "TUN",
    "Peru": "PER", "Chile": "CHL", "Argentina": "ARG", "Ecuador": "ECU",
    "Venezuela": "VEN", "Guatemala": "GTM", "El Salvador": "SLV",
    "Honduras": "HND", "Costa Rica": "CRI", "Panama": "PAN",
    "Dominican Republic": "DOM", "Jamaica": "JAM", "Trinidad and Tobago": "TTO",
    "Cuba": "CUB", "Cameroon": "CMR", "Tanzania": "TZA", "Uganda": "UGA",
    "Rwanda": "RWA", "Senegal": "SEN", "Mozambique": "MOZ", "Zimbabwe": "ZWE",
    "Zambia": "ZMB", "Ivory Coast": "CIV",
    "Congo, Dem. Rep. of the (Kinshasa)": "COD", "Congo, Republic of the (Brazzaville)": "COG",
    "Sudan": "SDN", "Somalia": "SOM", "Libya": "LBY", "Angola": "AGO",
    "Malawi": "MWI", "Burkina Faso": "BFA", "Mali": "MLI", "Niger": "NER",
    "Togo": "TGO", "Benin": "BEN", "Sierra Leone": "SLE", "Liberia": "LBR",
    "Gambia": "GMB", "Mauritius": "MUS", "Philippines": "PHL",
    "Myanmar": "MMR", "Cambodia": "KHM", "Laos": "LAO", "Singapore": "SGP",
    "Mongolia": "MNG", "Kazakhstan": "KAZ", "Uzbekistan": "UZB",
    "Kyrgyzstan": "KGZ", "Tajikistan": "TJK", "Turkmenistan": "TKM",
    "Afghanistan": "AFG", "Iraq": "IRQ", "Syria": "SYR", "Yemen": "YEM",
    "Lebanon": "LBN", "Qatar": "QAT", "United Arab Emirates": "ARE",
    "Bahrain": "BHR", "Oman": "OMN", "Poland": "POL", "Czech Republic": "CZE",
    "Hungary": "HUN", "Romania": "ROU", "Bulgaria": "BGR", "Austria": "AUT",
    "Switzerland": "CHE", "Belgium": "BEL", "Portugal": "PRT", "Greece": "GRC",
    "Denmark": "DNK", "Finland": "FIN", "Serbia": "SRB", "Croatia": "HRV",
    "Albania": "ALB", "Belarus": "BLR", "Lithuania": "LTU", "Latvia": "LVA",
    "Estonia": "EST", "Azerbaijan": "AZE", "Armenia": "ARM", "Georgia": "GEO",
    "Bolivia": "BOL", "Paraguay": "PRY", "Uruguay": "URY", "Nicaragua": "NIC",
    "Haiti": "HTI", "Guyana": "GUY", "Fiji": "FJI", "Bhutan": "BTN", "Maldives": "MDV"
}

choropleth_df = countries_modern[["place", "year", "enrollment"]].copy()
choropleth_df["iso3"] = choropleth_df["place"].map(iso3_lookup)
choropleth_df.to_csv(f"{out}/07_choropleth_data.csv", index=False)

reg_df.to_csv(f"{out}/08_regression_output.csv", index=False)
stem_by_year.to_csv(f"{out}/09_stem_vs_nonstem.csv", index=False)

top_fields = df_fos_long.groupby("field")["enrollment"].sum().nlargest(12).index.tolist()
fos_corr_top = fos_pivot[top_fields].corr().round(3)
fos_corr_top.to_csv(f"{out}/10_fos_correlation_matrix.csv")

elbow_df = pd.DataFrame({"k": range(2, 8), "inertia": inertias})
elbow_df.to_csv(f"{out}/11_kmeans_elbow.csv", index=False)

df_countries_long[["place", "year", "enrollment"]].to_csv(f"{out}/12_all_countries_long.csv", index=False)

print("\n  CSVs exported:")
for f in sorted(os.listdir(out)):
    size = os.path.getsize(f"{out}/{f}")
    print(f"    {f}  ({size:,} bytes)")

print("\n" + "=" * 55)
print("ANALYSIS COMPLETE")
print("=" * 55)
print(f"  Countries tracked: {len(cluster_df)}")
print(f"  Regression R²: {r2:.3f}")
print(f"  Largest country (2024): {top20_latest.iloc[0]['place']} ({top20_latest.iloc[0]['enrollment']:,.0f})")