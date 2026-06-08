# TrendVibe

# PLON Market — Model ML Predykcji Sprzedaży Fresh

Projekt zrealizowany w ramach zadania konkursowego **Consult IT! 2026**. Celem modelu jest prognozowanie dziennej sprzedaży (w sztukach) dla 10 najpopularniejszych produktów (SKU) z kategorii Fresh.

## 🚀 Cel projektu
* **Predykcja:** Dzienna sprzedaż (sztuki) dla Top 10 SKU Fresh.
* **Horyzont:** 30 dni (dane testowe: marzec 2026).
* **Metryka:** MAPE (Mean Absolute Percentage Error) oraz WMAPE (Weighted MAPE).
* **Wejście:** Sprzedaż historyczna, eventy lokalne, promocje, dane pogodowe.

## 🛠 Użyta technologia
* **Język:** Python 3.11
* **Model:** XGBoost (Gradient Boosting)
* **Biblioteki:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`

## 📊 Inżynieria cech
Model wykorzystuje zaawansowane podejście do tworzenia cech, aby uniknąć *data leakage* i poradzić sobie z lukami w danych:
* **Kalendarz:** Dzień tygodnia, miesiąc, weekendy, święta.
* **Lagi:** Sprzedaż z $t-7, t-14, t-28$ (metoda *date-based merge*).
* **Rolling Mean:** Średnie kroczące z 7, 14 i 28 dni (obliczane przez pivot dla zapewnienia ciągłości).
* **Dane zewnętrzne:** Wpływ eventów, parametry pogodowe (temperatura, opady), oraz flaga i intensywność promocji.

## 📂 Struktura repozytorium
* `PLON_Market_model_ML.ipynb` – Jupyter Notebook z pełną eksploracją, wizualizacją i logiką modelu.
* `plon_model.py` – Skrypt produkcyjny (wersja .py) do uruchomienia modelu.
* `predykcje_fresh.csv` – Wygenerowany plik wynikowy (przykładowy).

## ⚙️ Jak uruchomić?
1. Upewnij się, że masz zainstalowane zależności:
   ```bash
   pip install pandas numpy xgboost scikit-learn matplotlib openpyxl
