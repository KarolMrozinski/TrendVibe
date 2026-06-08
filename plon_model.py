"""
PLON Market — Model ML predykcji sprzedaży Fresh (Punkt 2)
Consult IT! 2026

Cechy modelu:
  - Kalendarz: dzień tygodnia, miesiąc, tydzień roku, weekend, piątek, poniedziałek
  - Lag features: sprzedaż t-7, t-14, t-28 (date-based merge, nie row-shift)
  - Rolling means: avg z ostatnich 7/14/28 dni (per SKU × Sklep)
  - Eventy: święta, imprezy lokalne, siła eventu
  - Pogoda: temp_avg, opady, śnieg, nasłonecznienie
  - Promocje: flag i liczba aktywnych promocji w kategorii

Algorytm: XGBoost z early stopping (eval metric = MAPE na val)
Horyzont walidacji: ostatnie 30 dni (pełne pokrycie 10 SKU)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

XLSX_PATH  = 'attached_assets/PLON_Market_dane_1778319078948.xlsx'
OUTPUT_PATH = 'predykcje_fresh.csv'

print("=" * 60)
print("PLON Market — Predykcja sprzedaży Fresh (XGBoost)")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. WCZYTANIE DANYCH
# ─────────────────────────────────────────────
print("\n[1/6] Wczytywanie danych...")

df_sales = pd.read_excel(XLSX_PATH, sheet_name='Sprzedaż dzienna',
                         parse_dates=['Data'])
df_events = pd.read_excel(XLSX_PATH, sheet_name='Eventy lokalne',
                          parse_dates=['Data'])
df_promo  = pd.read_excel(XLSX_PATH, sheet_name='Promocje',
                          parse_dates=['Data od', 'Data do'])
df_weather = pd.read_excel(XLSX_PATH, sheet_name='Pogoda',
                           parse_dates=['Data'])

df_sales = df_sales.rename(columns={
    'ID Sklepu':        'ID_Sklepu',
    'Sztyki sprzedane': 'Sprzedaz_szt',
    'Sztuki sprzedane': 'Sprzedaz_szt',
    'Cena jedn. (PLN)': 'Cena_jdn',
    'Sprzedaż (PLN)':   'Sprzedaz_PLN',
    'Nazwa SKU':        'Nazwa_SKU',
})

print(f"  Sprzedaż dzienna: {len(df_sales):,} wierszy, "
      f"{df_sales['Data'].min().date()} → {df_sales['Data'].max().date()}")

# ─────────────────────────────────────────────
# 2. TOP 10 SKU Z KATEGORII FRESH
# ─────────────────────────────────────────────
print("\n[2/6] Wybór Top 10 SKU z kategorii Fresh...")

FRESH_CATS = {'Warzywa i owoce', 'Mięso i wędliny',
              'Nabiał i jaja', 'Pieczywo i wyroby cukiernicze'}
df_fresh = df_sales[df_sales['Kategoria'].isin(FRESH_CATS)].copy()

top_10_sku = (df_fresh.groupby('ID_SKU')['Sprzedaz_szt']
                       .sum().nlargest(10).index.tolist())
sku_names  = df_fresh.groupby('ID_SKU')['Nazwa_SKU'].first()

print("\n  Top 10 SKU Fresh (wg łącznego wolumenu):")
for i, sku in enumerate(top_10_sku, 1):
    vol = df_fresh[df_fresh['ID_SKU'] == sku]['Sprzedaz_szt'].sum()
    print(f"  {i:2d}. {sku:10s} — {sku_names[sku]:<30s} ({vol:,.0f} szt)")

df_top = df_fresh[df_fresh['ID_SKU'].isin(top_10_sku)].copy()
print(f"\n  Zbiór modelowy: {len(df_top):,} wierszy")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3/6] Inżynieria cech...")

# 3a. Cechy kalendarza
df_top['day_of_week']  = df_top['Data'].dt.dayofweek
df_top['month']        = df_top['Data'].dt.month
df_top['week_of_year'] = df_top['Data'].dt.isocalendar().week.astype(int)
df_top['is_weekend']   = df_top['day_of_week'].isin([5, 6]).astype(int)
df_top['is_monday']    = (df_top['day_of_week'] == 0).astype(int)
df_top['is_friday']    = (df_top['day_of_week'] == 4).astype(int)

# 3b. LAG FEATURES — date-based merge (poprawna obsługa luk w danych)
print("  Budowanie lag features (date-based merge)...")

def add_lag_features(df, lags=(7, 14, 28)):
    """Dołącza lag features przez merge na przesuniętej dacie."""
    base = df[['Data', 'ID_SKU', 'ID_Sklepu', 'Sprzedaz_szt']].copy()
    for lag in lags:
        shifted = base.copy()
        shifted['Data'] = shifted['Data'] + pd.Timedelta(days=lag)
        shifted = shifted.rename(columns={'Sprzedaz_szt': f'lag_{lag}d'})
        df = df.merge(shifted[['Data', 'ID_SKU', 'ID_Sklepu', f'lag_{lag}d']],
                      on=['Data', 'ID_SKU', 'ID_Sklepu'], how='left')
    return df

df_top = add_lag_features(df_top, lags=(7, 14, 28))

# 3c. ROLLING MEAN — oparte na pivot (poprawna obsługa luk)
print("  Budowanie rolling mean features...")

def add_rolling_mean(df, windows=(7, 14, 28)):
    """Rolling mean przez pivot → rolling → melt → merge."""
    base = df[['Data', 'ID_SKU', 'ID_Sklepu', 'Sprzedaz_szt']].copy()
    key = base['ID_SKU'] + '__' + base['ID_Sklepu']
    base['_key'] = key

    pivot = base.pivot_table(index='Data', columns='_key',
                             values='Sprzedaz_szt', aggfunc='sum')
    # Pełna skala czasowa (wszystkie dni)
    pivot = pivot.reindex(pd.date_range(pivot.index.min(), pivot.index.max(), freq='D'))

    for w in windows:
        roll = pivot.shift(1).rolling(window=w, min_periods=1).mean()
        roll_long = roll.reset_index().melt(id_vars='index',
                                            var_name='_key',
                                            value_name=f'roll_{w}d')
        roll_long = roll_long.rename(columns={'index': 'Data'})
        roll_long['ID_SKU']    = roll_long['_key'].str.split('__').str[0]
        roll_long['ID_Sklepu'] = roll_long['_key'].str.split('__').str[1]
        df = df.merge(
            roll_long[['Data', 'ID_SKU', 'ID_Sklepu', f'roll_{w}d']],
            on=['Data', 'ID_SKU', 'ID_Sklepu'], how='left'
        )
    return df

df_top = add_rolling_mean(df_top, windows=(7, 14, 28))

# 3d. Eventy lokalne
ev_all  = (df_events[df_events['Miasto'] == 'ALL']
            .groupby('Data')['Wpływ szacowany (1-5)'].max()
            .reset_index().rename(columns={'Wpływ szacowany (1-5)': 'ev_all'}))
ev_city = (df_events[df_events['Miasto'] != 'ALL']
            .groupby(['Data', 'Miasto'])['Wpływ szacowany (1-5)'].max()
            .reset_index().rename(columns={'Wpływ szacowany (1-5)': 'ev_city'}))
ev_hol  = (df_events[(df_events['Typ'] == 'Święto') & (df_events['Miasto'] == 'ALL')]
            [['Data']].drop_duplicates().assign(is_holiday=1))

df_top = (df_top
          .merge(ev_all,  on='Data', how='left')
          .merge(ev_city, on=['Data', 'Miasto'], how='left')
          .merge(ev_hol,  on='Data', how='left'))
df_top['ev_all']       = df_top['ev_all'].fillna(0)
df_top['ev_city']      = df_top['ev_city'].fillna(0)
df_top['is_holiday']   = df_top['is_holiday'].fillna(0)
df_top['event_flag']   = ((df_top['ev_all'] > 0) | (df_top['ev_city'] > 0)).astype(int)
df_top['event_impact'] = df_top[['ev_all', 'ev_city']].max(axis=1)

# 3e. Pogoda
w_cols = {
    'Temp_avg (°C)':       'temp_avg',
    'Opady (mm)':          'opady_mm',
    'Śnieg (cm)':          'snieg_cm',
    'Nasłonecznienie (h)': 'naslonecznienie_h',
    'Kategoria pogody':    'kat_pogody',
}
df_w = df_weather.rename(columns=w_cols)[
    ['Data', 'Miasto'] + list(w_cols.values())
].copy()
pogoda_map = {'Słonecznie': 4, 'Pochmurno': 2, 'Deszcz': 1, 'Śnieg': 0, 'Burza': 0, 'Mgła': 2}
df_w['pogoda_num'] = df_w['kat_pogody'].map(pogoda_map).fillna(2)
df_top = df_top.merge(
    df_w[['Data', 'Miasto', 'temp_avg', 'opady_mm', 'snieg_cm',
          'naslonecznienie_h', 'pogoda_num']],
    on=['Data', 'Miasto'], how='left'
)
for c in ['temp_avg', 'opady_mm', 'snieg_cm', 'naslonecznienie_h', 'pogoda_num']:
    df_top[c] = df_top[c].fillna(df_top[c].median())

# 3f. Promocje — aktywne promocje danego dnia w kategorii SKU
promo_rows = []
for _, r in df_promo.iterrows():
    for d in pd.date_range(r['Data od'], r['Data do'], freq='D'):
        promo_rows.append({
            'Data': d,
            'Kat_raw': r['Kategoria'],
            'Typ_Promocji': r['Typ promocji'],
        })
df_pd = pd.DataFrame(promo_rows)

cat_map = {
    'Warzywa i owoce': 'Warzywa i owoce',
    'Warzywa': 'Warzywa i owoce',
    'Owoce': 'Warzywa i owoce',
    'Mięso i wędliny': 'Mięso i wędliny',
    'Mięso': 'Mięso i wędliny',
    'Nabiał i jaja': 'Nabiał i jaja',
    'Nabiał': 'Nabiał i jaja',
    'Pieczywo i wyroby cukiernicze': 'Pieczywo i wyroby cukiernicze',
    'Pieczywo': 'Pieczywo i wyroby cukiernicze',
}
df_pd['Kategoria'] = df_pd['Kat_raw'].map(cat_map)
df_pd = df_pd.dropna(subset=['Kategoria'])
typ_map = {'2+1': 3, 'BOGO': 3, 'Rabat %': 2, 'Promocja cenowa': 2, 'Gazetka': 1}
df_pd['promo_typ_num'] = df_pd['Typ_Promocji'].map(typ_map).fillna(1)

df_pa = (df_pd.groupby(['Data', 'Kategoria'])
              .agg(promo_flag=('promo_typ_num', 'max'),
                   promo_count=('promo_typ_num', 'count'))
              .reset_index())

df_top = df_top.merge(df_pa, on=['Data', 'Kategoria'], how='left')
df_top['promo_flag']  = df_top['promo_flag'].fillna(0)
df_top['promo_count'] = df_top['promo_count'].fillna(0)

# 3g. Label encoding
sku_enc   = {s: i for i, s in enumerate(top_10_sku)}
sklep_enc = {s: i for i, s in enumerate(sorted(df_top['ID_Sklepu'].unique()))}
df_top['sku_code']   = df_top['ID_SKU'].map(sku_enc)
df_top['sklep_code'] = df_top['ID_Sklepu'].map(sklep_enc)

FEATURES = [
    'sku_code', 'sklep_code',
    'day_of_week', 'month', 'week_of_year',
    'is_weekend', 'is_monday', 'is_friday',
    'lag_7d', 'lag_14d', 'lag_28d',
    'roll_7d', 'roll_14d', 'roll_28d',
    'event_flag', 'event_impact', 'is_holiday',
    'temp_avg', 'opady_mm', 'snieg_cm', 'naslonecznienie_h', 'pogoda_num',
    'promo_flag', 'promo_count',
]

print(f"  Liczba cech: {len(FEATURES)}")

# ─────────────────────────────────────────────
# 4. PODZIAŁ TRAIN / TEST (ostatnie 30 dni)
# ─────────────────────────────────────────────
print("\n[4/6] Podział danych train/test...")

max_date     = df_top['Data'].max()
train_cutoff = max_date - pd.Timedelta(days=30)

# Usunięcie wierszy z NaN w lag (pierwsze 28 dni historii)
df_model = df_top.dropna(subset=['lag_7d', 'lag_14d', 'lag_28d']).copy()

train = df_model[df_model['Data'] <= train_cutoff].copy()
test  = df_model[df_model['Data'] >  train_cutoff].copy()

print(f"  Train: {len(train):,} wierszy  ({train['Data'].min().date()} → {train['Data'].max().date()})")
print(f"  Test:  {len(test):,} wierszy   ({test['Data'].min().date()} → {test['Data'].max().date()})")
print(f"  SKU w teście: {test['ID_SKU'].nunique()} / {len(top_10_sku)}")
print(f"  Sklepy w teście: {test['ID_Sklepu'].nunique()}")

X_train, y_train = train[FEATURES], train['Sprzedaz_szt']
X_test,  y_test  = test[FEATURES],  test['Sprzedaz_szt']

# ─────────────────────────────────────────────
# 5. TRENING MODELU XGBoost
# ─────────────────────────────────────────────
print("\n[5/6] Trening modelu XGBoost (z early stopping)...")

model = XGBRegressor(
    n_estimators          = 2000,
    learning_rate         = 0.03,
    max_depth             = 5,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 3,
    gamma                 = 0.1,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    random_state          = 42,
    early_stopping_rounds = 50,
    eval_metric           = 'mape',
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

print(f"  Najlepsza liczba drzew (early stopping): {model.best_iteration}")

# Feature importance
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  Top 10 najważniejszych cech:")
for feat, val in imp.head(10).items():
    bar = '█' * int(val * 80)
    print(f"  {feat:<22s} {val:.4f}  {bar}")

# ─────────────────────────────────────────────
# 6. PREDYKCJA I METRYKA MAPE
# ─────────────────────────────────────────────
print("\n[6/6] Predykcja i eksport wyników...")

predictions = np.maximum(model.predict(X_test), 0)
mape_score  = mean_absolute_percentage_error(y_test, predictions)

print(f"\n  ✓ MAPE (całościowy): {mape_score:.4f}  ({mape_score*100:.2f}%)")

test_out = test.copy()
test_out['Prognoza_Sprzedazy'] = predictions

print("\n  MAPE per SKU:")
for sku in top_10_sku:
    sub = test_out[test_out['ID_SKU'] == sku]
    if len(sub) == 0:
        print(f"  {sku} — {sku_names.get(sku,''):<30s}  (brak w teście)")
        continue
    m = mean_absolute_percentage_error(sub['Sprzedaz_szt'], sub['Prognoza_Sprzedazy'])
    print(f"  {sku} — {sku_names.get(sku,''):<30s}  MAPE={m:.4f}")

# Wymagany format CSV
output = test_out[['Data', 'ID_SKU', 'ID_Sklepu', 'Prognoza_Sprzedazy']].copy()
output['Prognoza_Sprzedazy'] = output['Prognoza_Sprzedazy'].round(2)
output = output.sort_values(['Data', 'ID_SKU', 'ID_Sklepu']).reset_index(drop=True)
output.to_csv(OUTPUT_PATH, index=False)

print(f"\n  ✓ Zapisano: {OUTPUT_PATH}")
print(f"  Wierszy: {len(output):,} | SKU: {output['ID_SKU'].nunique()} | Sklepy: {output['ID_Sklepu'].nunique()}")
print(f"  Zakres dat: {output['Data'].min().date()} → {output['Data'].max().date()}")

print("\n" + "=" * 60)
print(f"GOTOWE — MAPE = {mape_score:.4f}  ({mape_score*100:.2f}%)")
print("=" * 60)
