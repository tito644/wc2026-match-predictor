"""
🏆 WC 2026 Match Predictor — Professional Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, random, warnings
from datetime import datetime
from scipy.stats import poisson as sp_poisson
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# 1. PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 2. CONSTANTS — Groups & Teams
# ═══════════════════════════════════════════════════════════
WC2026_GROUPS = {
    "A": ["Mexico","South Africa","South Korea","Czechia"],
    "B": ["Switzerland","Canada","Bosnia and Herzegovina","Qatar"],
    "C": ["Brazil","Morocco","Scotland","Haiti"],
    "D": ["United States","Australia","Paraguay","Turkey"],
    "E": ["Germany","Ivory Coast","Ecuador","Curaçao"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Egypt","Iran","New Zealand"],
    "H": ["Spain","Cape Verde","Uruguay","Saudi Arabia"],
    "I": ["France","Norway","Senegal","Iraq"],
    "J": ["Argentina","Austria","Algeria","Jordan"],
    "K": ["Colombia","Portugal","DR Congo","Uzbekistan"],
    "L": ["England","Croatia","Ghana","Panama"],
}
ALL_WC_TEAMS = sorted({t for v in WC2026_GROUPS.values() for t in v})

TEAM_NAME_MAP = {
    "Holland":"Netherlands","Korea Republic":"South Korea",
    "IR Iran":"Iran","USA":"United States",
    "Côte d'Ivoire":"Ivory Coast","Czechia":"Czech Republic",
}
def normalize_team(name):
    if not name or str(name)=="nan": return name
    return TEAM_NAME_MAP.get(str(name).strip(), str(name).strip())

MANUAL_ELIMINATED = {"Germany","Netherlands","Colombia","Australia"} # خرجوا بركلات الترجيح في دور الـ32

# ═══════════════════════════════════════════════════════════
# 3. TRANSLATIONS
# ═══════════════════════════════════════════════════════════
TR = {
    "ar": {
        "nav_home":"🏠  الرئيسية","nav_predict":"⚔️  توقع مباراة",
        "nav_groups":"🔵  دور المجموعات","nav_next":"📅  مباريات الدور القادم",
        "nav_champ":"🏆  الفائز بالكأس","nav_pdf":"📄  تقرير PDF",
        "team_a":"🔴 الفريق الأول","team_b":"🔵 الفريق الثاني",
        "predict_btn":"🔮 توقع المباراة","ko_toggle":"⚡ مباراة إقصائية",
        "neutral_toggle":"🏟️ ملعب محايد","draw":"تعادل",
        "exp_score":"النتيجة المتوقعة","final_pred":"التوقع النهائي",
        "sim_btn":"🔮 توقع ترتيب المجموعة","mc_btn":"🎲 ابدأ المحاكاة",
        "pdf_btn":"📄 توليد التقرير","dl_pdf":"⬇️ تحميل التقرير PDF",
        "refresh":"🔄 تحديث","group_sel":"اختار المجموعة",
        "sim_speed":"دقة المحاكاة","mc_speed":"دقة Monte Carlo",
        "fast":"سريع (100)","medium":"متوسط (300)","accurate":"دقيق (500)",
        "mc_fast":"500 (سريع)","mc_med":"1000 (متوسط)","mc_acc":"2000 (دقيق)",
        "include_elim":"تضمين الفرق المُقصاة",
        "alive_hdr":"✅ لسه في المنافسة","elim_hdr":"❌ خرجوا",
        "compare":"📊 مقارنة تفصيلية","strongest":"⚡ أقوى الفرق دلوقتي",
        "groups_title":"🔵 المجموعات الـ 12","full_table":"📊 الترتيب الكامل",
        "champ_title":"🏆 من سيفوز بكأس العالم 2026؟",
        "upset":"⚠️ تحذير: upset محتمل!","elo":"ELO Rating",
        "fifa_rank":"FIFA Ranking","form10":"فورم آخر 10",
        "goals_pg":"أهداف/مباراة","fc26":"متوسط FC26",
        "market":"القيمة السوقية","shootout":"ركلات الترجيح",
        "qualified":"✅ متأهل","ko_out":"❌ خرج من KO",
        "grp_out":"⬛ خرج من المجموعات","direct_q":"✅ متأهل مباشرة",
        "best3":"⚠️ مرشح أحسن توالت","out_lbl":"❌ خارج",
        "standings":"🏆 الترتيب المتوقع — المجموعة","points":"نقطة",
        "grp_matches":"⚔️ توقع نتائج مباريات المجموعة",
        "next_title":"📅 مباريات الدور القادم — التوقعات الكاملة",
        "pdf_title":"📄 تقرير PDF شامل للبطولة",
        "pdf_info":"التقرير بيحتوي على: نسب الفوز بالكأس + توقعات كل المباريات + ترتيب الفرق",
        "pdf_ready":"✅ التقرير جاهز!","sims_on":"محاكاة Monte Carlo",
        "analysis_on":"📊 التحليل على","teams_lbl":"فريق متأهل فعلياً",
        "gen_report":"📄 توليد التقرير",
        "predict_page":"⚔️ توقع مباراة بين أي فريقين",
        "groups_page":"🔵 توقع ترتيب دور المجموعات",
    },
    "en": {
        "nav_home":"🏠  Home","nav_predict":"⚔️  Predict Match",
        "nav_groups":"🔵  Group Stage","nav_next":"📅  Next Round Matches",
        "nav_champ":"🏆  Tournament Winner","nav_pdf":"📄  PDF Report",
        "team_a":"🔴 Team A","team_b":"🔵 Team B",
        "predict_btn":"🔮 Predict Match","ko_toggle":"⚡ Knockout Match",
        "neutral_toggle":"🏟️ Neutral Venue","draw":"Draw",
        "exp_score":"Expected Score","final_pred":"Prediction",
        "sim_btn":"🔮 Simulate Standings","mc_btn":"🎲 Run Simulation",
        "pdf_btn":"📄 Generate Report","dl_pdf":"⬇️ Download PDF Report",
        "refresh":"🔄 Refresh","group_sel":"Select Group",
        "sim_speed":"Simulation Accuracy","mc_speed":"Monte Carlo Accuracy",
        "fast":"Fast (100)","medium":"Medium (300)","accurate":"Accurate (500)",
        "mc_fast":"500 (Fast)","mc_med":"1000 (Medium)","mc_acc":"2000 (Accurate)",
        "include_elim":"Include eliminated teams",
        "alive_hdr":"✅ Still Competing","elim_hdr":"❌ Eliminated",
        "compare":"📊 Detailed Comparison","strongest":"⚡ Strongest Teams (ELO)",
        "groups_title":"🔵 The 12 Groups","full_table":"📊 Full Table",
        "champ_title":"🏆 Who will win WC 2026?",
        "upset":"⚠️ Warning: Upset possible!","elo":"ELO Rating",
        "fifa_rank":"FIFA Ranking","form10":"Form (Last 10)",
        "goals_pg":"Goals/Match","fc26":"Avg FC26 Rating",
        "market":"Market Value","shootout":"Penalty Shootouts",
        "qualified":"✅ Qualified","ko_out":"❌ Out (KO)",
        "grp_out":"⬛ Out (Groups)","direct_q":"✅ Qualified",
        "best3":"⚠️ Best 3rd candidate","out_lbl":"❌ Out",
        "standings":"🏆 Expected Standings — Group","points":"pts",
        "grp_matches":"⚔️ Group Match Predictions",
        "next_title":"📅 Next Round Matches — Full Predictions",
        "pdf_title":"📄 Comprehensive PDF Tournament Report",
        "pdf_info":"Report includes: Win probabilities + Match predictions + Team rankings",
        "pdf_ready":"✅ Report Ready!","sims_on":"Monte Carlo simulations",
        "analysis_on":"📊 Analysis on","teams_lbl":"qualified teams",
        "gen_report":"📄 Generate Report",
        "predict_page":"⚔️ Predict Any Match",
        "groups_page":"🔵 Predict Group Stage Standings",
    }
}

# ═══════════════════════════════════════════════════════════
# 4. SESSION STATE  (after TRANSLATIONS)
# ═══════════════════════════════════════════════════════════
if "theme" not in st.session_state: st.session_state["theme"] = "dark"
if "lang"  not in st.session_state: st.session_state["lang"]  = "ar"

def T(key):
    return TR[st.session_state["lang"]].get(key, key)

# ═══════════════════════════════════════════════════════════
# 5. DYNAMIC CSS  (after session_state)
# ═══════════════════════════════════════════════════════════
D = st.session_state["theme"] == "dark"

BG    = "#06111e"   if D else "#f0f4f8"
CARD  = "#0f2847"   if D else "#ffffff"
SIDE  = "#0a1a2e"   if D else "#1a2f5a"
BDR   = "#1e3a5f"   if D else "#e2e8f0"
TTXT  = "#ffffff"   if D else "#1a2f5a"
STXT  = "rgba(255,255,255,0.5)" if D else "#64748b"
HERO  = "linear-gradient(135deg,#0a1628,#0f2847,#091d38)" if D else "linear-gradient(135deg,#1a3060,#0d3b6e,#1a4a8a)"
SVAL  = "#FFD700"   if D else "#1a3a5c"
SHDR  = "#FFD700"   if D else "#1a3a5c"
SBDR  = "rgba(255,215,0,0.2)" if D else "rgba(26,58,92,0.2)"
MVAL  = "#FFD700"   if D else "#1a3a5c"
STATB = "linear-gradient(135deg,#0f2847,#091d38)" if D else "linear-gradient(135deg,#ffffff,#f8fafc)"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{{font-family:'Inter',sans-serif!important}}
.stApp{{background:{BG}!important}}
section[data-testid="stSidebar"]{{background:{SIDE}!important;border-right:1px solid {BDR}}}
section[data-testid="stSidebar"] *{{color:white!important}}
.hero-wrap{{background:{HERO};border:1px solid rgba(255,215,0,0.25);border-radius:20px;padding:2.5rem 2rem 2rem;text-align:center;margin-bottom:1.5rem;position:relative;overflow:hidden}}
.hero-wrap::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at center,rgba(255,215,0,0.05) 0%,transparent 60%);pointer-events:none}}
.hero-title{{font-size:2.6rem;font-weight:800;color:#FFD700;letter-spacing:-1.5px;margin:0;text-shadow:0 0 40px rgba(255,215,0,0.3)}}
.hero-sub{{color:rgba(255,255,255,0.7);font-size:1rem;margin-top:0.4rem;font-weight:300}}
.hero-badges{{margin-top:1rem;display:flex;gap:8px;justify-content:center;flex-wrap:wrap}}
.hbadge{{background:rgba(255,215,0,0.1);border:1px solid rgba(255,215,0,0.35);color:#FFD700;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:500}}
.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem}}
.stat-card{{background:{STATB};border:1px solid {BDR};border-radius:14px;padding:1.2rem;text-align:center;transition:border-color 0.2s}}
.stat-card:hover{{border-color:rgba(255,215,0,0.4)}}
.stat-val{{font-size:2.2rem;font-weight:700;color:{SVAL};line-height:1}}
.stat-lbl{{font-size:0.75rem;color:{STXT};margin-top:4px}}
.sec-hdr{{font-size:1.3rem;font-weight:700;color:{SHDR};border-bottom:2px solid {SBDR};padding-bottom:8px;margin:1.5rem 0 1rem;display:flex;align-items:center;gap:8px}}
.match-card{{background:{CARD};border:1px solid {BDR};border-radius:16px;padding:1.5rem;margin:0.5rem 0}}
.match-teams{{font-size:1.8rem;font-weight:700;color:{TTXT};text-align:center;letter-spacing:-0.5px}}
.match-score{{font-size:1.1rem;color:#FFD700;text-align:center;margin:0.3rem 0}}
.winner-pill{{display:inline-block;background:linear-gradient(90deg,#FFD700,#FFA500);color:#06111e;font-weight:700;padding:6px 20px;border-radius:30px;font-size:1rem;margin-top:0.8rem}}
.upset-warn{{background:rgba(255,100,0,0.15);border:1px solid rgba(255,100,0,0.4);color:#ffa94d;border-radius:8px;padding:6px 12px;font-size:0.82rem;margin-top:8px;display:inline-block}}
.cmp-table{{width:100%;border-collapse:collapse}}
.cmp-table th{{padding:8px 12px;font-size:0.8rem;color:{STXT};background:rgba(255,255,255,0.04)}}
.cmp-table td{{padding:8px 12px;font-size:0.87rem;color:{TTXT};border-bottom:1px solid rgba(255,255,255,0.06);text-align:center}}
.cmp-better{{color:#51cf66!important;font-weight:600}}
.cmp-worse{{color:{STXT}!important}}
.podium-card{{border-radius:16px;padding:1.5rem;text-align:center;border:2px solid;transition:transform 0.2s;background:{CARD}}}
.podium-card:hover{{transform:translateY(-4px)}}
.podium-medal{{font-size:2.8rem}}
.podium-team{{font-size:1.1rem;font-weight:700;color:{TTXT};margin:0.5rem 0}}
.podium-pct{{font-size:2rem;font-weight:800}}
.podium-sub{{font-size:0.75rem;color:{STXT};margin-top:3px}}
.upcoming-card{{background:{CARD};border:1px solid {BDR};border-radius:12px;padding:1rem;margin:6px 0;display:flex;align-items:center;justify-content:space-between}}
.uc-teams{{font-weight:600;color:{TTXT};font-size:0.95rem}}
.uc-date{{color:{STXT};font-size:0.78rem;margin-top:2px}}
.uc-winner{{color:#FFD700;font-weight:600;font-size:0.88rem}}
.uc-score{{color:{STXT};font-size:0.78rem}}
div.stButton>button{{background:linear-gradient(90deg,#FFD700,#FFA500)!important;color:#06111e!important;font-weight:700!important;border:none!important;border-radius:10px!important;padding:0.6rem 1.5rem!important;transition:opacity 0.2s!important}}
div.stButton>button:hover{{opacity:0.9!important}}
div[data-testid="stMetric"]{{background:{CARD};border-radius:10px;padding:0.8rem;border:1px solid {BDR}}}
div[data-testid="stMetric"] label{{color:{STXT}!important;font-size:0.78rem!important}}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{{color:{MVAL}!important;font-size:1.5rem!important;font-weight:700!important}}
.alive-team{{color:#51cf66!important;font-size:0.78rem;padding:1px 0}}
.elim-team{{color:#ff6b6b!important;font-size:0.78rem;text-decoration:line-through;padding:1px 0}}
.footer{{text-align:center;padding:2rem 0 1rem;color:{STXT};font-size:0.72rem;border-top:1px solid rgba(255,255,255,0.05);margin-top:3rem}}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 6. LIVE TOURNAMENT STATE
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def get_live_state():
    try:
        BASE = "https://raw.githubusercontent.com/martj42/international_results/master/"
        res  = pd.read_csv(BASE+"results.csv", parse_dates=["date"])  # cached for 1hr
        for col in ["home_team","away_team"]:
            res[col] = res[col].apply(normalize_team)
        wc26      = res[(res["tournament"]=="FIFA World Cup")&(res["date"]>="2026-01-01")].copy().sort_values("date")
        wc_done   = wc26.dropna(subset=["home_score","away_score"])
        KO        = pd.Timestamp("2026-06-28")
        ko_all    = wc26[wc26["date"]>=KO]
        ko_done   = ko_all.dropna(subset=["home_score","away_score"])
        ko_pend   = ko_all[ko_all["home_score"].isna()]
        q32       = set(ko_all["home_team"].dropna())|set(ko_all["away_team"].dropna())
        all_teams = set(t for v in WC2026_GROUPS.values() for t in v)
        elim_grp  = all_teams - q32
        elim_ko   = set()
        for _,m in ko_done.iterrows():
            if m["home_score"]!=m["away_score"]:
                elim_ko.add(m["away_team"] if m["home_score"]>m["away_score"] else m["home_team"])
        upcoming  = set(ko_pend["home_team"].dropna())|set(ko_pend["away_team"].dropna())
        act_manual= MANUAL_ELIMINATED - elim_ko - upcoming
        all_elim  = elim_grp|elim_ko|act_manual
        alive     = q32 - all_elim
        pending   = [(str(m["home_team"]),str(m["away_team"]),str(m["date"].date()))
                     for _,m in ko_pend.sort_values("date").iterrows()]
        return dict(alive=alive, all_elim=all_elim, elim_ko=elim_ko,
                    elim_grp=elim_grp, act_manual=act_manual, pending=pending,
                    last_match=str(wc_done["date"].max().date()) if len(wc_done)>0 else "N/A",
                    updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    source="GitHub Live ⚡", error=None)
    except Exception as e:
        fb_alive = {"Algeria","Argentina","Australia","Austria","Belgium","Brazil",
                    "Canada","Cape Verde","Colombia","Egypt","England","France",
                    "Ghana","Mexico","Morocco","Norway","Paraguay",
                    "Portugal","Spain","Switzerland","United States"}
        fb_pend  = [("Spain","Austria","2026-07-02"),("Portugal","Croatia","2026-07-02"),
                    ("Switzerland","Algeria","2026-07-02"),("Argentina","Cape Verde","2026-07-03"),
                    ("Australia","Egypt","2026-07-03"),("Colombia","Ghana","2026-07-03"),
                    ("Canada","Morocco","2026-07-04"),("Paraguay","France","2026-07-04"),
                    ("Mexico","England","2026-07-05"),("Brazil","Norway","2026-07-05"),
                    ("United States","Belgium","2026-07-06")]
        return dict(alive=fb_alive, all_elim=set(), elim_ko=set(), elim_grp=set(),
                    act_manual=MANUAL_ELIMINATED, pending=fb_pend,
                    last_match="N/A", updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    source="Fallback ⚠️", error=str(e))

_live        = get_live_state()
STILL_ALIVE  = _live["alive"]
ALL_ELIM     = _live["all_elim"]
ELIM_KO      = _live["elim_ko"]
PENDING      = _live["pending"]
LIVE_SRC     = _live["source"]
LIVE_UPD     = _live["updated"]
LIVE_MATCH   = _live["last_match"]
LIVE_ERR     = _live["error"]
ACT_MANUAL   = _live["act_manual"]

# ═══════════════════════════════════════════════════════════
# 7. LOAD MODELS & DATA
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    paths = {"xgb":"models/xgb_model_balanced.pkl","p_home":"models/poisson_home.pkl",
             "p_away":"models/poisson_away.pkl","scaler":"models/scaler.pkl",
             "feat_cols":"models/feature_cols.pkl"}
    m = {}
    for k,p in paths.items():
        if not os.path.exists(p): st.error(f"❌ Missing: {p}"); return None
        m[k] = joblib.load(p)
    return m

@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
    d = {}
    elo_path = "data/processed/elo_live_updated.csv" if os.path.exists("data/processed/elo_live_updated.csv") else "data/processed/elo_current.csv"
    files = {"results":"data/processed/results_clean.csv","elo":elo_path,
             "shootouts":"data/processed/shootout_stats.csv",
             "squad_summary":"data/processed/squad_summary.csv",
             "team_ratings":"data/processed/team_ratings_fc26.csv",
             "teams":"data/raw/teams.csv","fifa_rank":"data/processed/fifa_rank_latest.csv"}
    for k,p in files.items():
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"] if k=="results" else False)
            for col in df.columns:
                if any(x in col.lower() for x in ["team","country"]):
                    df[col] = df[col].apply(normalize_team)
            if k=="results":
                for col in ["home_team","away_team"]:
                    if col in df.columns: df[col]=df[col].apply(normalize_team)
            d[k] = df
    d["elo_path"] = elo_path
    return d

# ── Loading screen احترافي ────────────────────────────────
_load_placeholder = st.empty()
_load_placeholder.markdown(f"""
<div style="background:{'#0f2847' if D else '#ffffff'};border:1px solid {'#1e3a5f' if D else '#e2e8f0'};
            border-radius:16px;padding:2rem;text-align:center;margin:2rem 0">
    <div style="font-size:2rem;margin-bottom:0.5rem">⏳</div>
    <div style="color:#FFD700;font-size:1.1rem;font-weight:600;margin-bottom:0.3rem">
        {'جاري تحميل النماذج والبيانات...' if st.session_state['lang']=='ar' else 'Loading models and data...'}
    </div>
    <div style="color:rgba(255,255,255,0.5);font-size:0.85rem">
        {'ده بيحصل مرة واحدة بس ثم بيتكاش' if st.session_state['lang']=='ar' else 'This happens once then gets cached'}
    </div>
</div>
""", unsafe_allow_html=True)

MDL  = load_models()
DATA = load_data()

_load_placeholder.empty()  # تمسح loading screen لما الداتا تتحمّل

# ═══════════════════════════════════════════════════════════
# 8. PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════
def elo_prob(diff, neutral=True):
    return 1/(1+10**(-(diff+(0 if neutral else 50))/400))

def team_elo(team):
    df  = DATA.get("elo", pd.DataFrame())
    col = "elo_live" if "elo_live" in df.columns else "elo"
    r   = df[df["team"]==team]
    return float(r[col].values[0]) if len(r)>0 else 1500.0

def team_stats(team):
    s = {"elo": team_elo(team)}
    tr = DATA.get("teams",pd.DataFrame())
    row = tr[tr["team_name"]==team] if len(tr)>0 else pd.DataFrame()
    s["fifa_rank"]    = int(row["fifa_ranking_pre_tournament"].values[0]) if len(row)>0 else 50
    s["elo_official"] = float(row["elo_rating"].values[0]) if len(row)>0 else s["elo"]
    res = DATA.get("results", pd.DataFrame())
    for n in [5,10]:
        hg=res[res["home_team"]==team].copy() if len(res)>0 else pd.DataFrame()
        ag=res[res["away_team"]==team].copy() if len(res)>0 else pd.DataFrame()
        if len(hg)>0:
            hg["sc"]=hg["home_score"]; hg["cn"]=hg["away_score"]
            hg["pts"]=hg["result"].map({"home_win":3,"draw":1,"away_win":0}) if "result" in hg.columns else pd.Series([1]*len(hg))
        if len(ag)>0:
            ag["sc"]=ag["away_score"]; ag["cn"]=ag["home_score"]
            ag["pts"]=ag["result"].map({"away_win":3,"draw":1,"home_win":0}) if "result" in ag.columns else pd.Series([1]*len(ag))
        all_g=pd.concat([hg,ag]).sort_values("date",ascending=False).head(n) if len(res)>0 else pd.DataFrame()
        if len(all_g)>0:
            s[f"f{n}_wr"]=round((all_g["pts"]==3).mean(),3)
            s[f"f{n}_gf"]=round(all_g["sc"].mean(),2)
            s[f"f{n}_ga"]=round(all_g["cn"].mean(),2)
            s[f"f{n}_gd"]=round((all_g["sc"]-all_g["cn"]).mean(),2)
            s[f"f{n}_pts"]=int(all_g["pts"].sum())
        else:
            s[f"f{n}_wr"]=0.33;s[f"f{n}_gf"]=1.0;s[f"f{n}_ga"]=1.0;s[f"f{n}_gd"]=0.0;s[f"f{n}_pts"]=5
    fc=DATA.get("team_ratings",pd.DataFrame())
    fr=fc[fc["team"]==team] if len(fc)>0 else pd.DataFrame()
    s["avg_ovr"]=float(fr["avg_overall"].values[0]) if len(fr)>0 else 75.0
    s["max_ovr"]=float(fr["max_overall"].values[0]) if len(fr)>0 else 80.0
    sq=DATA.get("squad_summary",pd.DataFrame())
    sr=sq[sq["team"]==team] if len(sq)>0 else pd.DataFrame()
    s["total_mv"]=float(sr["total_market_value"].values[0]) if len(sr)>0 else 0.0
    s["avg_mv"]  =float(sr["avg_market_value"].values[0])   if len(sr)>0 else 0.0
    s["avg_caps"]=float(sr["avg_caps"].values[0])            if len(sr)>0 else 30.0
    so=DATA.get("shootouts",pd.DataFrame())
    sor=so[so["team"]==team] if len(so)>0 else pd.DataFrame()
    s["so_wr"]=float(sor["shootout_win_rate"].values[0]) if len(sor)>0 else 0.5
    return s

def build_mf(ta,tb,is_ko=False,neutral=True):
    fa=team_stats(ta); fb=team_stats(tb)
    res=DATA.get("results",pd.DataFrame())
    h2h=pd.DataFrame(); h2wc=pd.DataFrame()
    if len(res)>0:
        mask=((res["home_team"]==ta)&(res["away_team"]==tb))|((res["home_team"]==tb)&(res["away_team"]==ta))
        h2h=res[mask]; h2wc=h2h[h2h["tournament"]=="FIFA World Cup"] if "tournament" in h2h.columns else pd.DataFrame()
    def h2h_wr(df,a):
        if len(df)==0: return 0.333
        w=len(df[((df["home_team"]==a)&(df.get("result","")=="home_win"))|((df["away_team"]==a)&(df.get("result","")=="away_win"))])
        return round(w/len(df),3)
    ed=fa["elo"]-fb["elo"]; fd=fa["f10_wr"]-fb["f10_wr"]
    sa=0.35*(fa["elo"]/2200)+0.25*fa["f10_wr"]+0.20*(fa["total_mv"]/1.5e9)+0.20*(fa["avg_ovr"]/90)
    sb=0.35*(fb["elo"]/2200)+0.25*fb["f10_wr"]+0.20*(fb["total_mv"]/1.5e9)+0.20*(fb["avg_ovr"]/90)
    return {"elo_diff":ed,"elo_win_prob":elo_prob(ed,neutral),"elo_diff_capped":np.clip(ed,-200,200),
            "elo_official_diff":fa["elo_official"]-fb["elo_official"],
            "fifa_rank_diff":fb["fifa_rank"]-fa["fifa_rank"],
            "form5_win_rate_diff":fa["f5_wr"]-fb["f5_wr"],
            "form10_win_rate_diff":fd,"form10_gd_diff":fa["f10_gd"]-fb["f10_gd"],
            "form10_gf_diff":fa["f10_gf"]-fb["f10_gf"],
            "avg_overall_diff":fa["avg_ovr"]-fb["avg_ovr"],
            "max_overall_diff":fa["max_ovr"]-fb["max_ovr"],
            "market_value_diff":fa["total_mv"]-fb["total_mv"],
            "h2h_win_rate_a":h2h_wr(h2h,ta),"h2h_wc_win_rate_a":h2h_wr(h2wc,ta),
            "shootout_win_rate_diff":fa["so_wr"]-fb["so_wr"],
            "avg_caps_diff":fa["avg_caps"]-fb["avg_caps"],
            "is_knockout":int(is_ko),"neutral_venue":int(neutral),
            "strength_score_diff":sa-sb,
            "upset_potential":int(ed<-50 and fd>0.15),
            "ko_upset_risk":int(is_ko)*(1-elo_prob(ed)),
            "form_elo_interaction":fd*(1-elo_prob(ed)),
            "_fa":fa,"_fb":fb}

def predict(ta,tb,is_ko=False,neutral=True):
    if MDL is None: return None
    mf=build_mf(ta,tb,is_ko,neutral)
    fa=mf.pop("_fa"); fb=mf.pop("_fb")
    x=np.array([[mf.get(f,0) for f in MDL["feat_cols"]]])
    xs=MDL["scaler"].transform(x)
    pr=MDL["xgb"].predict_proba(xs)[0]
    pb_x,pd_x,pa_x=pr[0]*100,pr[1]*100,pr[2]*100
    lh=max(0.3,MDL["p_home"].predict(xs)[0]); la=max(0.3,MDL["p_away"].predict(xs)[0])
    ph=pd_=pa=0.0
    for h in range(10):
        for a in range(10):
            p=sp_poisson.pmf(h,lh)*sp_poisson.pmf(a,la)
            if h>a: ph+=p
            elif h==a: pd_+=p
            else: pa+=p
    return {"p_home":round((pa_x+ph*100)/2,1),"p_draw":round((pd_x+pd_*100)/2,1),
            "p_away":round((pb_x+pa*100)/2,1),"exp_h":round(lh,2),"exp_a":round(la,2),
            "fa":fa,"fb":fb,"upset":bool(mf.get("upset_potential",0))}

def sim_once(ta,tb):
    if MDL is None: return ta
    mf=build_mf(ta,tb,is_ko=True,neutral=True)
    fa=mf.pop("_fa"); fb=mf.pop("_fb")
    x=np.array([[mf.get(f,0) for f in MDL["feat_cols"]]])
    xs=MDL["scaler"].transform(x)
    lh=max(0.3,MDL["p_home"].predict(xs)[0]); la=max(0.3,MDL["p_away"].predict(xs)[0])
    gh=np.random.poisson(lh); ga=np.random.poisson(la)
    if gh>ga: return ta
    elif ga>gh: return tb
    so=DATA.get("shootouts",pd.DataFrame())
    def sor(t): r=so[so["team"]==t]; return float(r["shootout_win_rate"].values[0]) if len(r)>0 else 0.5
    sa=sor(ta); sb=sor(tb); tot=sa+sb or 1
    return ta if random.random()<sa/tot else tb

def sim_group(teams,n=300):
    if MDL is None: return [{"team":t,"pts":0.0,"gd":0.0,"gf":0.0} for t in teams]
    pts_a={t:[] for t in teams}; gd_a={t:[] for t in teams}; gf_a={t:[] for t in teams}
    for _ in range(n):
        pts={t:0 for t in teams}; gd={t:0 for t in teams}; gf={t:0 for t in teams}
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                a,b=teams[i],teams[j]
                mf=build_mf(a,b,is_ko=False,neutral=True); mf.pop("_fa"); mf.pop("_fb")
                x=np.array([[mf.get(f,0) for f in MDL["feat_cols"]]])
                xs=MDL["scaler"].transform(x)
                lh=max(0.3,MDL["p_home"].predict(xs)[0]); la=max(0.3,MDL["p_away"].predict(xs)[0])
                gh=np.random.poisson(lh); ga=np.random.poisson(la)
                gf[a]+=gh; gf[b]+=ga; gd[a]+=(gh-ga); gd[b]+=(ga-gh)
                if gh>ga: pts[a]+=3
                elif ga>gh: pts[b]+=3
                else: pts[a]+=1; pts[b]+=1
        for t in teams: pts_a[t].append(pts[t]); gd_a[t].append(gd[t]); gf_a[t].append(gf[t])
    return sorted([{"team":t,"pts":round(np.mean(pts_a[t]),1),"gd":round(np.mean(gd_a[t]),1),"gf":round(np.mean(gf_a[t]),1)} for t in teams],key=lambda x:(-x["pts"],-x["gd"],-x["gf"]))

def run_mc(teams,n=1000,prog=None):
    tl=list(teams); wins={t:0 for t in tl}; finals={t:0 for t in tl}
    for i in range(n):
        rem=tl.copy(); random.shuffle(rem)
        while len(rem)>1:
            nxt=[]
            for k in range(0,len(rem)-1,2):
                nxt.append(sim_once(rem[k],rem[k+1]))
            if len(rem)%2==1: nxt.append(rem[-1])
            if len(rem)<=4:
                for t in nxt: finals[t]=finals.get(t,0)+1
            rem=nxt
        wins[rem[0]]+=1
        if prog and i%max(1,n//20)==0: prog.progress(min((i+1)/n,1.0))
    return pd.DataFrame([{"الفريق":t,"win_pct":round(w/n*100,1),"final_pct":round(finals.get(t,0)/n*100,1)} for t,w in wins.items()]).sort_values("win_pct",ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════
# 9. PDF
# ═══════════════════════════════════════════════════════════
def make_pdf(mc_df, preds):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,HRFlowable
    from reportlab.lib.units import cm
    import io
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    GOLD=colors.HexColor("#FFD700"); DARK=colors.HexColor("#0a1628"); GRAY=colors.HexColor("#888888")
    s=getSampleStyleSheet()
    T1=ParagraphStyle("t1",parent=s["Title"],textColor=GOLD,fontSize=20,alignment=1,fontName="Helvetica-Bold",spaceAfter=4)
    T2=ParagraphStyle("t2",parent=s["Normal"],textColor=GRAY,fontSize=9,alignment=1,spaceAfter=2)
    T3=ParagraphStyle("t3",parent=s["Heading2"],textColor=GOLD,fontSize=12,fontName="Helvetica-Bold",spaceBefore=14,spaceAfter=6)
    W=colors.white
    story=[Paragraph("🏆 WC 2026 — Match Predictor Report",T1),
           Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | XGBoost + Poisson Ensemble",T2),
           HRFlowable(width="100%",thickness=1,color=GOLD,spaceAfter=12)]
    story.append(Paragraph("🏆 Tournament Winner Probabilities (Monte Carlo)",T3))
    td=[["#","Team","Win %","Final %","Bar"]]
    medals=["🥇","🥈","🥉"]+[str(i) for i in range(4,17)]
    for idx,row in mc_df.head(16).iterrows():
        td.append([medals[idx] if idx<3 else str(idx+1),row["الفريق"],f"{row['win_pct']}%",f"{row['final_pct']}%","█"*int(row["win_pct"]/2)])
    t=Table(td,colWidths=[1.5*cm,5*cm,2.5*cm,2.5*cm,6*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),DARK),("TEXTCOLOR",(0,0),(-1,0),GOLD),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),9),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(1,0),(1,-1),"LEFT"),
        ("FONTSIZE",(0,1),(-1,-1),8),("TEXTCOLOR",(0,1),(-1,-1),W),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#0d0d0d"),colors.HexColor("#0a0a0a")]),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#333333")),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.append(t)
    story.append(Spacer(1,12))
    story.append(Paragraph("⚔️ Upcoming Match Predictions",T3))
    ud=[["Date","Match","Score","Winner","H%","D%","A%"]]
    for p in preds:
        ud.append([p["date"],f"{p['ta']} vs {p['tb']}",f"{p['exp_h']:.1f}—{p['exp_a']:.1f}",p["winner"],f"{p['p_home']}%",f"{p['p_draw']}%",f"{p['p_away']}%"])
    t2=Table(ud,colWidths=[2*cm,5.5*cm,2.5*cm,3*cm,1.5*cm,1.5*cm,1.5*cm])
    t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),DARK),("TEXTCOLOR",(0,0),(-1,0),GOLD),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8),
        ("FONTSIZE",(0,1),(-1,-1),7.5),("TEXTCOLOR",(0,1),(-1,-1),W),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#0d0d0d"),colors.HexColor("#0a0a0a")]),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#333333")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(1,0),(1,-1),"LEFT"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(t2)
    story.append(Spacer(1,20))
    story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY))
    story.append(Paragraph("Model: XGBoost + Poisson | Features: ELO, Form, Squad Strength, H2H, Market Value | Data: martj42 + Kaggle",ParagraphStyle("ft",parent=s["Normal"],textColor=GRAY,fontSize=7,alignment=1,spaceBefore=6)))
    doc.build(story); buf.seek(0); return buf

# ═══════════════════════════════════════════════════════════
# 10. SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='text-align:center;color:#FFD700;font-size:1.05rem;font-weight:700;padding:0.8rem 0 0.5rem'>🏆 WC 2026 Predictor</div>", unsafe_allow_html=True)

    # Theme + Language buttons
    st.markdown("""<style>
    div[data-testid="stSidebar"] div.stButton>button{
        background:rgba(255,215,0,0.12)!important;
        color:#FFD700!important;
        border:1px solid rgba(255,215,0,0.35)!important;
        border-radius:8px!important;
        font-size:0.8rem!important;
        font-weight:600!important;
        height:38px!important;
        min-height:38px!important;
        padding:0!important;
        width:100%!important;
        display:flex!important;
        align-items:center!important;
        justify-content:center!important;
        line-height:1!important;
    }
    div[data-testid="stSidebar"] div.stButton>button:hover{
        background:rgba(255,215,0,0.25)!important;
    }
    div[data-testid="stSidebar"] div.stButton>button p{
        margin:0!important;
        font-size:0.8rem!important;
        line-height:1!important;
    }
    </style>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        lbl = "☀️ Light" if D else "🌙 Dark"
        if st.button(lbl, key="theme_btn", use_container_width=True):
            st.session_state["theme"] = "light" if D else "dark"
            st.rerun()
    with c2:
        lbl2 = "🇬🇧 EN" if st.session_state["lang"]=="ar" else "🇸🇦 AR"
        if st.button(lbl2, key="lang_btn", use_container_width=True):
            st.session_state["lang"] = "en" if st.session_state["lang"]=="ar" else "ar"
            st.rerun()

    st.markdown("---")
    page = st.radio("", [T("nav_home"),T("nav_predict"),T("nav_groups"),
                         T("nav_next"),T("nav_champ"),T("nav_pdf")],
                    label_visibility="collapsed")
    st.markdown("---")

    st.markdown(f"<div style='color:#FFD700;font-size:0.78rem;font-weight:600;margin-bottom:5px'>{T('alive_hdr')}</div>", unsafe_allow_html=True)
    for t in sorted(STILL_ALIVE):
        st.markdown(f"<div class='alive-team'>✅ {t}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='color:#ff6b6b;font-size:0.78rem;font-weight:600;margin:8px 0 5px'>{T('elim_hdr')}</div>", unsafe_allow_html=True)
    for t in sorted(ELIM_KO|ACT_MANUAL)[:8]:
        st.markdown(f"<div class='elim-team'>❌ {t}</div>", unsafe_allow_html=True)

    st.markdown("---")
    src_color = "#51cf66" if "Live" in LIVE_SRC else "#ffa94d"
    st.markdown(f"<div style='font-size:0.68rem;color:rgba(255,255,255,0.35);line-height:1.6'><span style='color:{src_color}'>⚡ {LIVE_SRC}</span><br>Updated: {LIVE_UPD}<br>Last match: {LIVE_MATCH}</div>", unsafe_allow_html=True)
    if LIVE_ERR:
        st.markdown(f"<div style='color:#ffa94d;font-size:0.65rem'>⚠️ Fallback</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:rgba(255,255,255,0.2);font-size:0.62rem;margin-top:4px'>{'التحميل الأول أبطأ — بعدها بيكون سريع' if st.session_state['lang']=='ar' else 'First load is slower — cached after'}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 11. HERO
# ═══════════════════════════════════════════════════════════
ch, cr = st.columns([6,1])
with ch:
    st.markdown(f"""<div class="hero-wrap">
    <div class="hero-title">🏆 WC 2026 Match Predictor</div>
    <div class="hero-sub">نموذج توقع احترافي مبني على XGBoost + Poisson Ensemble</div>
    <div class="hero-badges">
        <span class="hbadge">{LIVE_SRC}</span>
        <span class="hbadge">🤖 ML Model</span>
        <span class="hbadge">📊 {len(STILL_ALIVE)} Teams Alive</span>
        <span class="hbadge">🎲 Monte Carlo</span>
        <span class="hbadge">📄 PDF</span>
    </div></div>""", unsafe_allow_html=True)
with cr:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button(T("refresh"), help="Refresh from GitHub"):
        st.cache_data.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════
# 12. PAGES
# ═══════════════════════════════════════════════════════════

# ── HOME ────────────────────────────────────────────────────
if T("nav_home") == page:
    st.markdown(f"""<div class="stat-grid">
    <div class="stat-card"><div class="stat-val">48</div><div class="stat-lbl">{'فريق مشارك' if st.session_state['lang']=='ar' else 'Teams'}</div></div>
    <div class="stat-card"><div class="stat-val">104</div><div class="stat-lbl">{'مباراة' if st.session_state['lang']=='ar' else 'Matches'}</div></div>
    <div class="stat-card"><div class="stat-val">{len(STILL_ALIVE)}</div><div class="stat-lbl">{T('alive_hdr')}</div></div>
    <div class="stat-card"><div class="stat-val">{len(PENDING)}</div><div class="stat-lbl">{'مباريات قادمة' if st.session_state['lang']=='ar' else 'Upcoming'}</div></div>
    </div>""", unsafe_allow_html=True)
    if LIVE_ERR:
        st.warning(f"⚠️ GitHub unavailable — using fallback data")

    st.markdown(f'<div class="sec-hdr">{T("strongest")}</div>', unsafe_allow_html=True)
    elo_df  = DATA.get("elo", pd.DataFrame()).copy()
    elo_col = "elo_live" if "elo_live" in elo_df.columns else "elo"
    elo_df  = elo_df.rename(columns={elo_col:"elo"})
    if "team" in elo_df.columns:
        alive_elo = elo_df[elo_df["team"].isin(STILL_ALIVE)].sort_values("elo",ascending=False).head(10).reset_index(drop=True)
        c1,c2 = st.columns(2)
        for idx,row in alive_elo.iterrows():
            col = c1 if idx%2==0 else c2
            with col:
                bw = int((row["elo"]-1700)/600*100)
                st.markdown(f"""<div style="background:{CARD};border:1px solid {BDR};border-radius:10px;padding:10px 14px;margin:4px 0;display:flex;align-items:center;gap:12px">
                <div style="color:#FFD700;font-weight:700;font-size:1.1rem;min-width:24px">{idx+1}</div>
                <div style="flex:1"><div style="color:{TTXT};font-weight:600;font-size:0.9rem">{row['team']}</div>
                <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:6px;margin-top:4px">
                <div style="background:linear-gradient(90deg,#FFD700,#FFA500);height:100%;width:{bw}%;border-radius:4px"></div></div></div>
                <div style="color:#FFD700;font-weight:700;font-size:0.95rem">{int(row['elo'])}</div></div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sec-hdr">{T("groups_title")}</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for gidx,(group,teams) in enumerate(WC2026_GROUPS.items()):
        with cols[gidx%4]:
            rows_html=""
            for t in teams:
                if t in STILL_ALIVE:
                    rows_html+=f"<div style='color:#51cf66;font-size:0.78rem;padding:2px 0'>✅ {t}</div>"
                elif t in (ELIM_KO|ACT_MANUAL):
                    rows_html+=f"<div style='color:#ff6b6b;font-size:0.78rem;text-decoration:line-through;padding:2px 0'>❌ {t}</div>"
                else:
                    rows_html+=f"<div style='color:rgba(255,255,255,0.35);font-size:0.78rem;padding:2px 0'>— {t}</div>"
            st.markdown(f"""<div style="background:{CARD};border:1px solid rgba(255,215,0,0.2);border-radius:12px;padding:12px;margin-bottom:10px">
            <div style="color:#FFD700;font-weight:700;font-size:0.92rem;margin-bottom:6px">{'المجموعة' if st.session_state['lang']=='ar' else 'Group'} {group}</div>
            {rows_html}</div>""", unsafe_allow_html=True)

# ── PREDICT MATCH ───────────────────────────────────────────
elif T("nav_predict") == page:
    st.markdown(f'<div class="sec-hdr">{T("predict_page")}</div>', unsafe_allow_html=True)
    alive_list = sorted(STILL_ALIVE)
    all_list   = sorted(ALL_WC_TEAMS)
    inc_elim   = st.toggle(T("include_elim"), value=False)
    opts       = all_list if inc_elim else alive_list
    c1,cv,c2   = st.columns([5,1,5])
    with c1: ta = st.selectbox(T("team_a"), opts, index=0)
    with cv: st.markdown(f"<div style='text-align:center;padding-top:1.8rem;font-size:1.8rem'>🆚</div>", unsafe_allow_html=True)
    with c2: tb = st.selectbox(T("team_b"), opts, index=min(1,len(opts)-1))
    cc1,cc2 = st.columns(2)
    with cc1: is_ko   = st.toggle(T("ko_toggle"), value=True)
    with cc2: neutral = st.toggle(T("neutral_toggle"), value=True)
    if st.button(T("predict_btn"), type="primary", use_container_width=True):
        if ta==tb: st.error("⚠️ Choose two different teams!")
        elif MDL is None: st.error("Models not loaded!")
        else:
            with st.spinner("Analyzing..."):
                r = predict(ta,tb,is_ko,neutral)
            if r["p_home"]>=r["p_away"] and r["p_home"]>=r["p_draw"]:
                winner=ta; wc="#FFD700"; wtc="#06111e"
            elif r["p_away"]>=r["p_home"] and r["p_away"]>=r["p_draw"]:
                winner=tb; wc="#4dabf7"; wtc="#06111e"
            else:
                winner=T("draw"); wc="#868e96"; wtc="white"
            upset_html=f'<div class="upset-warn">{T("upset")}</div>' if r["upset"] else ""
            st.markdown(f"""<div class="match-card" style="text-align:center">
            <div style="color:rgba(255,255,255,0.4);font-size:0.8rem;margin-bottom:8px">
            {"⚡ Knockout" if is_ko else "🔵 Group Stage"} &nbsp;•&nbsp; {T("exp_score")}: <b style="color:#FFD700">{r['exp_h']:.1f} — {r['exp_a']:.1f}</b></div>
            <div class="match-teams">{ta}  🆚  {tb}</div>
            <span class="winner-pill" style="background:linear-gradient(90deg,{wc},{wc}cc);color:{wtc}">🏆 {winner}</span>
            {upset_html}</div>""", unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1: st.metric(f"{T('win')} {ta}", f"{r['p_home']}%"); st.progress(r["p_home"]/100)
            with c2: st.metric(T("draw"), f"{r['p_draw']}%"); st.progress(r["p_draw"]/100)
            with c3: st.metric(f"{T('win')} {tb}", f"{r['p_away']}%"); st.progress(r["p_away"]/100)
            st.markdown(f'<div class="sec-hdr">{T("compare")}</div>', unsafe_allow_html=True)
            fa,fb=r["fa"],r["fb"]
            rows=[(T("elo"),int(fa["elo"]),int(fb["elo"]),"high"),
                  (T("fifa_rank"),f"#{fa['fifa_rank']}",f"#{fb['fifa_rank']}",'low'),
                  (T("form10"),f"{fa['f10_wr']*100:.0f}%",f"{fb['f10_wr']*100:.0f}%","high"),
                  (T("goals_pg"),f"{fa['f10_gf']:.2f}",f"{fb['f10_gf']:.2f}","high"),
                  (T("fc26"),f"{fa['avg_ovr']:.1f}",f"{fb['avg_ovr']:.1f}","high"),
                  (T("market"),f"€{fa['total_mv']/1e6:.0f}M",f"€{fb['total_mv']/1e6:.0f}M","high"),
                  (T("shootout"),f"{fa['so_wr']*100:.0f}%",f"{fb['so_wr']*100:.0f}%","high")]
            hdr=f"<tr><th style='text-align:left'>{'' }</th><th>{ta}</th><th>{tb}</th></tr>"
            brows=""
            for lbl,va,vb,d in rows:
                try:
                    vaf=float(str(va).replace("%","").replace("€","").replace("M","").replace("#",""))
                    vbf=float(str(vb).replace("%","").replace("€","").replace("M","").replace("#",""))
                    ab=vaf>=vbf if d=="high" else vaf<=vbf
                except: ab=None
                ca="cmp-better" if ab==True else ("cmp-worse" if ab==False else "")
                cb="cmp-better" if ab==False else ("cmp-worse" if ab==True else "")
                brows+=f"<tr><td style='text-align:left'>{lbl}</td><td class='{ca}'>{va}</td><td class='{cb}'>{vb}</td></tr>"
            st.markdown(f'<table class="cmp-table"><thead>{hdr}</thead><tbody>{brows}</tbody></table>', unsafe_allow_html=True)

# ── GROUP STAGE ─────────────────────────────────────────────
elif T("nav_groups") == page:
    st.markdown(f'<div class="sec-hdr">{T("groups_page")}</div>', unsafe_allow_html=True)
    grp_lbl = f"{'المجموعة' if st.session_state['lang']=='ar' else 'Group'}"
    gl = st.selectbox(T("group_sel"), [f"{grp_lbl} {g}" for g in WC2026_GROUPS.keys()]).split()[-1]
    gt = WC2026_GROUPS[gl]
    gcols = st.columns(4)
    for i,t in enumerate(gt):
        with gcols[i]:
            ev = int(team_elo(t))
            if t in STILL_ALIVE: st2=T("qualified"); sc="#51cf66"
            elif t in (ELIM_KO|ACT_MANUAL): st2=T("ko_out"); sc="#ff6b6b"
            else: st2=T("grp_out"); sc="#868e96"
            st.markdown(f"""<div style="background:{CARD};border:1px solid rgba(255,215,0,0.2);border-radius:12px;padding:14px;text-align:center">
            <div style="font-size:1rem;font-weight:700;color:{TTXT};margin-bottom:4px">{t}</div>
            <div style="color:#FFD700;font-size:0.85rem">ELO: {ev}</div>
            <div style="color:{sc};font-size:0.75rem;margin-top:4px">{st2}</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    n_map = {T("fast"):100,T("medium"):300,T("accurate"):500}
    nsim  = n_map[st.select_slider(T("sim_speed"),[T("fast"),T("medium"),T("accurate")],value=T("medium"))]
    if st.button(T("sim_btn"), type="primary", use_container_width=True):
        if MDL is None: st.error("Models not loaded!")
        else:
            with st.spinner(f"Simulating {nsim} scenarios..."):
                standings = sim_group(gt, n=nsim)
            st.markdown(f'<div class="sec-hdr">{T("standings")} {gl}</div>', unsafe_allow_html=True)
            medals=["🥇","🥈","🥉","4️⃣"]
            ql=[T("direct_q"),T("direct_q"),T("best3"),T("out_lbl")]
            qc=["#51cf66","#51cf66","#ffa94d","#ff6b6b"]
            for idx,row in enumerate(standings):
                t=row["team"]; is_e=t in (ELIM_KO|ACT_MANUAL)
                c1,c2,c3,c4,c5,c6=st.columns([1,4,2,2,2,3])
                with c1: st.markdown(f"<div style='text-align:center;font-size:1.4rem'>{medals[idx]}</div>",unsafe_allow_html=True)
                with c2:
                    sty="text-decoration:line-through;opacity:0.5;" if is_e else ""
                    st.markdown(f"<div style='color:{TTXT};font-weight:600;padding-top:4px;{sty}'>{t}{'  ❌' if is_e else ''}</div>",unsafe_allow_html=True)
                with c3: st.markdown(f"<div style='color:#FFD700;font-weight:700;text-align:center;padding-top:4px'>{row['pts']} {T('points')}</div>",unsafe_allow_html=True)
                with c4:
                    gc="#51cf66" if row["gd"]>=0 else "#ff6b6b"
                    st.markdown(f"<div style='color:{gc};text-align:center;padding-top:4px'>GD: {row['gd']:+.1f}</div>",unsafe_allow_html=True)
                with c5: st.markdown(f"<div style='color:{STXT};text-align:center;padding-top:4px'>GF: {row['gf']:.1f}</div>",unsafe_allow_html=True)
                with c6: st.markdown(f"<div style='color:{qc[idx]};font-size:0.82rem;padding-top:6px'>{ql[idx]}</div>",unsafe_allow_html=True)
                st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:4px 0'>",unsafe_allow_html=True)
            st.markdown(f'<div class="sec-hdr">{T("grp_matches")}</div>', unsafe_allow_html=True)
            for i in range(len(gt)):
                for j in range(i+1,len(gt)):
                    a,b=gt[i],gt[j]
                    r=predict(a,b,is_ko=False,neutral=True)
                    if r:
                        if r["p_home"]>=r["p_away"] and r["p_home"]>=r["p_draw"]: pred=f"🏆 {a}"
                        elif r["p_away"]>=r["p_home"]: pred=f"🏆 {b}"
                        else: pred=f"🤝 {T('draw')}"
                        st.markdown(f"""<div class="upcoming-card">
                        <div><div class="uc-teams">{a} 🆚 {b}</div><div class="uc-date">Group {gl}</div></div>
                        <div class="uc-pred"><div class="uc-winner">{pred}</div>
                        <div class="uc-score">{r['exp_h']:.1f}—{r['exp_a']:.1f} | {r['p_home']}%/{r['p_draw']}%/{r['p_away']}%</div></div></div>""",unsafe_allow_html=True)

# ── NEXT ROUND ──────────────────────────────────────────────
elif T("nav_next") == page:
    st.markdown(f'<div class="sec-hdr">{T("next_title")}</div>', unsafe_allow_html=True)
    st.info(f"📊 {'لسه في المنافسة' if st.session_state['lang']=='ar' else 'Still competing'}: **{len(STILL_ALIVE)}** | {'مباريات قادمة' if st.session_state['lang']=='ar' else 'Upcoming'}: **{len(PENDING)}**")
    if not PENDING:
        st.info("🎉 All R32 matches completed!")
    elif MDL is None:
        st.error("Models not loaded!")
    else:
        with st.spinner("Calculating predictions..."):
            all_p=[]
            for ta,tb,date in PENDING:
                r=predict(ta,tb,is_ko=True,neutral=True)
                if r:
                    if r["p_home"]>=r["p_away"] and r["p_home"]>=r["p_draw"]: pred=ta
                    elif r["p_away"]>=r["p_home"]: pred=tb
                    else: pred=T("draw")
                    all_p.append({**r,"ta":ta,"tb":tb,"date":date,"winner":pred})
        dates=sorted(set(p["date"] for p in all_p))
        for d in dates:
            dm=datetime.strptime(d,"%Y-%m-%d")
            st.markdown(f"<div style='color:#FFD700;font-size:0.85rem;font-weight:600;margin:14px 0 6px'>📅 {dm.strftime('%A, %d %B %Y')}</div>",unsafe_allow_html=True)
            for p in [x for x in all_p if x["date"]==d]:
                a,b=p["ta"],p["tb"]
                if p["p_home"]>=p["p_away"] and p["p_home"]>=p["p_draw"]:
                    fav=a; fp=p["p_home"]; fc="#FFD700"
                elif p["p_away"]>=p["p_home"]:
                    fav=b; fp=p["p_away"]; fc="#4dabf7"
                else:
                    fav=T("draw"); fp=p["p_draw"]; fc="#868e96"
                ub=f'<span style="background:rgba(255,100,0,0.2);border:1px solid rgba(255,100,0,0.4);color:#ffa94d;font-size:0.7rem;padding:2px 8px;border-radius:10px;margin-left:8px">⚠️ Upset Risk</span>' if p["upset"] else ""
                st.markdown(f"""<div style="background:{CARD};border:1px solid {BDR};border-radius:14px;padding:14px 18px;margin:6px 0">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">
                <div><div style="font-size:1.05rem;font-weight:700;color:{TTXT}">{a} 🆚 {b} {ub}</div>
                <div style="color:{STXT};font-size:0.78rem;margin-top:3px">⚡ Knockout • {T('exp_score')}: {p['exp_h']:.1f}—{p['exp_a']:.1f}</div></div>
                <div style="text-align:right"><div style="color:{fc};font-weight:700;font-size:0.95rem">🏆 {fav}</div>
                <div style="color:{STXT};font-size:0.75rem">{fp:.0f}%</div></div></div>
                <div style="margin-top:10px;display:flex;gap:6px">
                <div style="flex:1"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{STXT};margin-bottom:2px"><span>{a}</span><span>{p['p_home']}%</span></div>
                <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:8px"><div style="background:#FFD700;height:100%;width:{p['p_home']}%;border-radius:4px"></div></div></div>
                <div style="flex:0.6"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{STXT};margin-bottom:2px"><span>{T('draw')}</span><span>{p['p_draw']}%</span></div>
                <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:8px"><div style="background:#868e96;height:100%;width:{p['p_draw']}%;border-radius:4px"></div></div></div>
                <div style="flex:1"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{STXT};margin-bottom:2px"><span>{b}</span><span>{p['p_away']}%</span></div>
                <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:8px"><div style="background:#4dabf7;height:100%;width:{p['p_away']}%;border-radius:4px"></div></div></div>
                </div></div>""",unsafe_allow_html=True)

# ── TOURNAMENT WINNER ───────────────────────────────────────
elif T("nav_champ") == page:
    st.markdown(f'<div class="sec-hdr">{T("champ_title")}</div>', unsafe_allow_html=True)
    st.info(f"{T('analysis_on')} **{len(STILL_ALIVE)}** {T('teams_lbl')}")
    nm_map={T("mc_fast"):500,T("mc_med"):1000,T("mc_acc"):2000}
    nv=nm_map[st.select_slider(T("mc_speed"),[T("mc_fast"),T("mc_med"),T("mc_acc")],value=T("mc_med"))]
    if st.button(T("mc_btn"), type="primary", use_container_width=True):
        if MDL is None: st.error("Models not loaded!")
        else:
            prog=st.progress(0,text="🎲 Running simulation...")
            mc=run_mc(STILL_ALIVE,n=nv,prog=prog)
            prog.empty()
            st.session_state["mc_df"]=mc; st.session_state["mc_n"]=nv

    if "mc_df" in st.session_state:
        mc=st.session_state["mc_df"]; nv2=st.session_state.get("mc_n",1000)
        top3=mc.head(3)
        st.markdown("<br>",unsafe_allow_html=True)
        c2,c1,c3=st.columns(3)
        pod=[(c1,0,"🥇","#FFD700","#1a1400"),(c2,1,"🥈","#C0C0C0","#141414"),(c3,2,"🥉","#CD7F32","#120900")]
        for col,idx,medal,color,bg in pod:
            if idx<len(top3):
                row=top3.iloc[idx]
                with col:
                    st.markdown(f"""<div class="podium-card" style="background:{bg};border-color:{color}">
                    <div class="podium-medal">{medal}</div>
                    <div class="podium-team">{row['الفريق']}</div>
                    <div class="podium-pct" style="color:{color}">{row['win_pct']}%</div>
                    <div class="podium-sub">{'احتمال الفوز بالكأس' if st.session_state['lang']=='ar' else 'Win probability'}</div>
                    <div class="podium-sub">{'نهائي' if st.session_state['lang']=='ar' else 'Final'}: {row['final_pct']}%</div></div>""",unsafe_allow_html=True)
        st.markdown("<br>")
        st.markdown(f'<div class="sec-hdr">{T("full_table")}</div>',unsafe_allow_html=True)
        mx=mc["win_pct"].max() or 1
        for idx,row in mc.iterrows():
            if row["win_pct"]==0: continue
            bw=int(row["win_pct"]/mx*100)
            m=["🥇","🥈","🥉"][idx] if idx<3 else f"{idx+1}."
            st.markdown(f"""<div style="background:{CARD};border:1px solid {BDR};border-radius:10px;padding:10px 14px;margin:3px 0;display:flex;align-items:center;gap:12px">
            <div style="min-width:28px;text-align:center;font-size:0.95rem">{m}</div>
            <div style="min-width:150px;color:{TTXT};font-weight:600;font-size:0.9rem">{row['الفريق']}</div>
            <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:5px;height:10px;overflow:hidden">
            <div style="background:linear-gradient(90deg,#FFD700,#FFA500);height:100%;width:{bw}%;border-radius:5px"></div></div>
            <div style="min-width:50px;text-align:right;color:#FFD700;font-weight:700;font-size:0.9rem">{row['win_pct']}%</div>
            <div style="min-width:80px;text-align:right;color:{STXT};font-size:0.76rem">{'نهائي' if st.session_state['lang']=='ar' else 'Final'}: {row['final_pct']}%</div></div>""",unsafe_allow_html=True)
        st.markdown(f"<div style='color:{STXT};font-size:0.7rem;margin-top:6px'>Based on {nv2:,} {T('sims_on')}</div>",unsafe_allow_html=True)

# ── PDF ─────────────────────────────────────────────────────
elif T("nav_pdf") == page:
    st.markdown(f'<div class="sec-hdr">{T("pdf_title")}</div>',unsafe_allow_html=True)
    st.info(T("pdf_info"))
    c1,c2=st.columns(2)
    with c1:
        nm_map2={T("mc_fast"):500,T("mc_med"):1000,T("mc_acc"):2000}
        np_val=nm_map2[st.select_slider(T("mc_speed"),[T("mc_fast"),T("mc_med"),T("mc_acc")],value=T("mc_med"),key="pdf_mc")]
    with c2:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"<div style='color:{STXT};font-size:0.85rem'>{'تقرير PDF احترافي جاهز للطباعة والمشاركة' if st.session_state['lang']=='ar' else 'Professional PDF ready for print & sharing'}</div>",unsafe_allow_html=True)
    if st.button(T("gen_report"), type="primary", use_container_width=True):
        if MDL is None: st.error("Models not loaded!")
        else:
            prog=st.progress(0,text="🎲 Running Monte Carlo...")
            mc=run_mc(STILL_ALIVE,n=np_val,prog=prog)
            prog.progress(0.65,text="⚔️ Calculating match predictions...")
            preds=[]
            for ta,tb,date in PENDING:
                r=predict(ta,tb,is_ko=True,neutral=True)
                if r:
                    if r["p_home"]>=r["p_away"] and r["p_home"]>=r["p_draw"]: winner=ta
                    elif r["p_away"]>=r["p_home"]: winner=tb
                    else: winner="Draw"
                    preds.append({**r,"ta":ta,"tb":tb,"date":date,"winner":winner})
            prog.progress(0.9,text="📄 Building PDF...")
            buf=make_pdf(mc,preds)
            prog.empty()
            st.success(T("pdf_ready"))
            st.download_button(label=T("dl_pdf"),data=buf,
                file_name=f"WC2026_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",use_container_width=True)

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""<div class="footer">
🏆 WC 2026 Match Predictor &nbsp;•&nbsp; XGBoost + Poisson Ensemble &nbsp;•&nbsp;
Data: martj42 (GitHub) + Kaggle &nbsp;•&nbsp; Built with ❤️ By Tariq Elnaggar &nbsp;•&nbsp;
{datetime.now().strftime('%Y-%m-%d')}
</div>""", unsafe_allow_html=True)
