# 🏆 WC 2026 Match Predictor

> **Predicting the 2026 FIFA World Cup — One Match at a Time**
> An end-to-end machine learning pipeline that combines ELO ratings, squad strength, recent form, head-to-head history, and player market values to forecast match outcomes in real time.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost_%2B_Poisson-green)](https://xgboost.readthedocs.io)
[![Data](https://img.shields.io/badge/Data-Live_from_GitHub-orange)](https://github.com/martj42/international_results)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wc2026-match-predictor-worldcup.streamlit.app/)

👉 **Live App:** [(https://wc2026-match-predictor-worldcup.streamlit.app/)]

---

## 🚀 Live Demo

🌍 **Try the app here:**
(https://wc2026-match-predictor-worldcup.streamlit.app/)

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <b>Tariq Elnaggar</b><br>
      Data Scientist & ML Engineer<br><br>
      <a href="https://github.com/tito644">
        <img src="https://img.shields.io/badge/GitHub-tito644-181717?logo=github&style=for-the-badge"/>
      </a><br><br>
      <a href="https://www.linkedin.com/in/tarek-mohamed-el-naggar/">
        <img src="https://img.shields.io/badge/LinkedIn-Tarek_Elnaggar-0A66C2?logo=linkedin&style=for-the-badge"/>
      </a><br><br>
      <a href="https://www.kaggle.com/tarekelnaggar">
        <img src="https://img.shields.io/badge/Kaggle-tarekelnaggar-20BEFF?logo=kaggle&style=for-the-badge"/>
      </a>
    </td>
  </tr>
</table>

---

## 🎯 Project Overview

The 2026 FIFA World Cup is the biggest sporting event on the planet — 48 teams, 104 matches, 3 host nations.
This project builds a **live-updating prediction engine** that answers one question:

> *Who will win — and by how much?*

Rather than relying on a single metric like FIFA rankings, the model combines **9 categories of features** spanning historical data from 1872 to today, squad quality, psychological pressure, and even penalty shootout history.

---

## 🧠 What Makes This Different

Most football prediction models stop at ELO ratings and recent form.
This one goes further:

| Feature Category | What It Captures |
|-----------------|-----------------|
| ⚡ **ELO Rating (Live)** | Dynamic team strength — updates after every match |
| 📈 **Recent Form** | Last 5 & 10 official matches (weighted by tournament importance) |
| ⚔️ **Head-to-Head** | Historical matchups + World Cup encounters specifically |
| 👥 **Squad Strength** | Average EAFC 26 Overall Rating of top 15 players |
| 💰 **Market Value** | Total Transfermarkt squad value (€) |
| 🎯 **Penalty Shootouts** | Historical shootout win rate — the "clutch factor" ⭐ |
| 🔥 **Motivation Flag** | Knockout = must-win pressure encoded as a feature ⭐ |
| 🏆 **WC Experience** | Average international caps per squad |
| 🤝 **Upset Potential** | Detects when a weaker team is in better form — flags upsets ⭐ |

> ⭐ = Features not commonly found in other prediction models

---

## 🏗️ Project Architecture

```
wc2026_predictor/
│
├── 📄 README.md
├── 🖥️  app.py                          # Streamlit dashboard (6 pages)
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📓 01_data_collection.ipynb        # Data pipeline — 49K+ matches from 1872
├── 📓 02_feature_engineering.ipynb    # Building all 22 features
├── 📓 04_live_update_engine.ipynb     # Live tournament state from GitHub
├── 📓 03_model_training.ipynb         # XGBoost + Poisson training & evaluation
│
├── 📁 models/
│   ├── xgb_model_balanced.pkl         # XGBoost classifier
│   ├── poisson_home.pkl               # Poisson goals model (home)
│   ├── poisson_away.pkl               # Poisson goals model (away)
│   ├── scaler.pkl                     # RobustScaler
│   └── feature_cols.pkl               # Feature column names
│
└── 📁 data/
    ├── raw/                           # Source datasets (not tracked by git)
    └── processed/                     # Engineered features (not tracked by git)
```

---

## 📊 Data Sources

| # | Dataset | Source | Content |
|---|---------|--------|---------|
| DS-1 | International Football Results | [martj42/GitHub](https://github.com/martj42/international_results) | 49,477 matches (1872–2026) — updates daily |
| DS-2 | FIFA World Rankings | [Kaggle](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Monthly rankings 1992–2024 |
| DS-3 | WC 2026 Live Dataset | [Kaggle](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset) | 1,248 players + live match stats |
| DS-4 | EAFC 26 Player Database | [Kaggle](https://www.kaggle.com/datasets/flynn28/eafc26-player-database) | 16,228 players with full attributes |
| DS-5 | Transfermarkt Dataset | [Kaggle](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets/) | Market values + injury history |

---

## 🤖 Modeling Approach

### Why Two Models?

**XGBoost Classifier** → Predicts *Win / Draw / Loss* with probability
**Poisson Regression** → Predicts *exact expected goals* for each team

The final prediction is an **ensemble** of both:

```python
Final P(Home Win) = (XGBoost_P + Poisson_P) / 2
```

This approach captures both the *outcome probability* and the *goal distribution*, giving richer and more calibrated predictions.

### The ELO Dominance Problem — and How We Solved It

Early versions of the model suffered from **ELO Dominance**: the model learned that "the stronger team always wins" and ignored all other signals.

The fix: convert raw ELO difference to a **win probability** using the original ELO formula:

```python
def elo_to_win_prob(elo_diff, neutral=True):
    return 1 / (1 + 10 ** (-(elo_diff) / 400))
```

This means even a team with +300 ELO advantage only gets 88% — not 100%.
Combined with a **±200 cap** on raw ELO difference and three new **upset features**, the model became significantly more balanced.

### Tournament Weighting

Not all matches are equal:

```python
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup"    : 4,   # Maximum weight
    "UEFA Euro"         : 3,
    "Copa América"      : 3,
    "WC Qualification"  : 2,
    "Friendly"          : 1,   # Minimum weight
}
```

### Temporal Split — No Data Leakage

Training: matches **before 2018** | Testing: **2018 onwards**

Every feature is computed using only data available *before* that match was played.

---

## ⚡ Live Update Engine

The tournament is happening **right now** — so the model updates itself automatically.

Every hour, a live pipeline pulls the latest results from GitHub:

```python
@st.cache_data(ttl=3600)
def get_live_tournament_state():
    # Fetches from martj42/international_results (updates daily)
    # Detects: qualified teams, eliminated teams, updated ELO
    # Smart fallback if GitHub is unavailable
```

**Smart elimination logic:**
- Teams not in the Round of 32 bracket → eliminated from groups
- Teams that lost a KO match → eliminated
- Penalty shootout losers → manual override (auto-removes once GitHub updates)

---

## 🖥️ Dashboard — 6 Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Live stats + strongest teams (ELO) + all 12 groups |
| ⚔️ **Predict Match** | Head-to-head prediction + detailed comparison table |
| 🔵 **Group Stage** | Simulate any group's final standings (Monte Carlo) |
| 📅 **Next Round** | All upcoming matches with win/draw/loss probabilities |
| 🏆 **Tournament Winner** | Monte Carlo simulation — who lifts the trophy? |
| 📄 **PDF Report** | Download full tournament prediction report |

**Extra features:**
- 🌙 Dark / ☀️ Light mode
- 🇸🇦 Arabic / 🇬🇧 English language toggle
- 🔄 One-click live refresh from GitHub

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/tito644/wc2026-match-predictor.git
cd wc2026-match-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
Place these files in `data/raw/`:

| File | Source |
|------|--------|
| `squads_and_players.csv` | [Kaggle DS-3](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset) |
| `match_team_stats.csv` | [Kaggle DS-3](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset) |
| `matches.csv` | [Kaggle DS-3](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset) |
| `teams.csv` | [Kaggle DS-3](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset) |
| `EAFC26-Men.csv` | [Kaggle DS-4](https://www.kaggle.com/datasets/flynn28/eafc26-player-database) |

> `results.csv` and `shootouts.csv` are downloaded **automatically** from GitHub — no manual download needed.

### 4. Run the notebooks in order
```bash
# Step 1 — Collect & clean data
01_data_collection.ipynb

# Step 2 — Build features
02_feature_engineering.ipynb

# Step 3 — Live update (run BEFORE training!)
04_live_update_engine.ipynb

# Step 4 — Train models
03_model_training.ipynb
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Cross-Validation Accuracy | ~58–62% |
| Poisson MAE (Home Goals) | < 1.0 |
| Poisson MAE (Away Goals) | < 1.0 |
| Training Data | 1,018 World Cup matches (1930–2026) |
| Features | 22 engineered features |

> **Note:** 58–62% is the realistic ceiling for football prediction.
> The sport's inherent randomness means no model can do much better.
> What matters is *calibrated probabilities*, not just raw accuracy.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
streamlit
joblib
scipy
reportlab
tqdm
```

---

## 🗺️ Roadmap

- [ ] xG integration from live WC 2026 data
- [ ] Injury & suspension data
- [ ] Coach win rate as a feature
- [ ] Deploy on Streamlit Cloud
- [ ] Betting odds comparison

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.

---

<div align="center">

**Built with ❤️ during FIFA World Cup 2026**

*"Football is unpredictable. That's exactly what makes predicting it so fascinating."*

⭐ If you found this useful, please give it a star!

[![GitHub](https://img.shields.io/badge/GitHub-tito644-181717?logo=github&style=flat-square)](https://github.com/tito644)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tarek_Elnaggar-0A66C2?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/tarek-mohamed-el-naggar/)
[![Kaggle](https://img.shields.io/badge/Kaggle-tarekelnaggar-20BEFF?logo=kaggle&style=flat-square)](https://www.kaggle.com/tarekelnaggar)

</div>
