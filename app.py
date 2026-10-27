# ==============================================================================
# PULSE AI — People Intelligence & Retention Studio | ONE Google Colab cell
# Understand the patterns. Improve the workplace.
# ==============================================================================
import subprocess, sys, os, re, json, zipfile, html, hashlib, warnings
warnings.filterwarnings("ignore")

def _deps():
    try:
        from importlib.metadata import version
        if int(version("gradio").split(".")[0]) not in (4, 5): raise RuntimeError("gradio")
        for p in ("plotly", "scikit-learn", "scipy", "requests", "pandas", "numpy"): version(p)
    except Exception:
        print("Installing dependencies (about a minute)...")
        try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scikit-learn>=1.2", "scipy", "requests", "pandas", "numpy"]); print("Dependencies installed.")
        except Exception as e: print("Install warning:", type(e).__name__, "- check internet access and re-run.")
_deps()
try:
    import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr, requests
    from scipy import stats
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
    from sklearn.inspection import permutation_importance
except Exception as e:
    raise SystemExit("Libraries could not be imported (" + type(e).__name__ + "). Restart the runtime and re-run this cell.")

C = dict(bg="#151C27", sf="#242F3B", mi="#7BE0C3", la="#D3B5FF", am="#F1BE78", tx="#F2F5F7", mu="#A7B5C5", co="#F18C91")
G = {}; TESTS = []; _TS = []; _SM = {}
BLANK = ["", "nan", "none", "n/a", "na", "-", "null", "nat"]
esc = lambda x: html.escape(str(x))
norm = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
money = lambda v: "—" if nul(v) else ("-" if v < 0 else "") + format(abs(v), ",.0f")
pc0 = lambda v: "—" if nul(v) else format(v, ".0%")
pc1 = lambda v: "—" if nul(v) else format(v, ".1%")
fx = lambda v, d=2: "—" if nul(v) else format(v, "." + str(d) + "f")
nf = lambda v: "—" if nul(v) else format(v, ",.0f")
pp = lambda v: "—" if nul(v) else "%+.1f pp" % (v * 100)
pretty = lambda f: str(f).replace("_", " ").capitalize()
TG = dict(obs=("OBSERVED", C["mi"]), pred=("MODEL PREDICTION", C["la"]), hyp=("HYPOTHESIS", C["am"]), scn=("SCENARIO", C["mu"]))
def tag(k): return "<span class='tag' style='color:%s;border-color:%s'>%s</span>" % (TG[k][1], TG[k][1], TG[k][0])
def note(t, k="info", tg=None):
    col = {"info": C["mu"], "ok": C["mi"], "warn": C["am"], "err": C["co"], "ai": C["la"]}[k]
    return "<div class='note' style='border-left:3px solid %s'>%s%s</div>" % (col, (tag(tg) + " ") if tg else "", t)
def card(lab, val, desc, col=C["mi"], tg=None): return "<div class='kc'><div class='kl'>%s</div><div class='kv' style='color:%s'>%s</div><div class='kd'>%s%s</div></div>" % (lab, col, val, (tag(tg) + " ") if tg else "", desc)
H = lambda t: "<div class='sec'>%s</div>" % t
def tbl(d, cols, fmt=None):
    fmt = fmt or {}; h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows(): h += "<tr>" + "".join("<td>%s</td>" % (fmt[k](r[k]) if k in fmt else ("—" if nul(r[k]) else esc(r[k]))) for k, _n in cols) + "</tr>"
    return h + "</table></div>"
def sty(f, h=380, t=""):
    f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["tx"]), height=h, title=dict(text=t, x=.02), margin=dict(l=50, r=20, t=55, b=50), legend=dict(orientation="h", y=-.25), colorway=[C["mi"], C["la"], C["am"], C["co"]])
    return f
def empty(msg):
    f = go.Figure(); f.add_annotation(text=msg, showarrow=False, font=dict(color=C["mu"], size=14)); f.update_xaxes(visible=False); f.update_yaxes(visible=False); return sty(f, 300)
NOP = lambda n: tuple(empty("No data") for _ in range(n))
def wilson(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - a) / d, (c + a) / d)
def clean_str(s):
    t = s.astype(str).str.strip(); return t.where(~(s.isna() | t.str.lower().isin(BLANK)), np.nan)
def to_num(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[$,%₹€£\s]", "", regex=True), errors="coerce")
def to_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes", "y", "t"])
def to_dt(s):
    try: return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception: return pd.to_datetime(s, errors="coerce")
def parse_flag(s):
    m = {"1": 1, "0": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "left": 1, "resigned": 1, "terminated": 1, "active": 0, "stayed": 0, "current": 0}
    out = s.astype(str).str.strip().str.lower().map(m); num = pd.to_numeric(s, errors="coerce"); return out.fillna(num.where(num.isin([0, 1]))).astype(float).where(s.notna())
def read_csv_any(file):
    path = file if isinstance(file, str) else getattr(file, "name", file)
    try: df = pd.read_csv(path)
    except Exception: df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    if df.empty: raise ValueError("The file contains no rows.")
    if len(df) > 1000000: raise ValueError("File too large (limit 1,000,000 rows).")
    return df
def parse_map(text):
    ov = {}
    for p in re.split(r"[,;\n]", text or ""):
        if "=" in p: k, v = p.split("=", 1); ov[norm(k)] = norm(v)
    return ov
def pseudo(s, salt):
    mp = {x: "P-" + hashlib.sha256((salt + str(x)).encode()).hexdigest()[:10] for x in s.dropna().unique()}; return s.map(mp)

# ------------------------------ FICTIONAL synthetic workforce ------------------------------
DEPTS = ["Engineering", "Sales", "Customer Support", "Operations", "Finance", "HR", "Marketing"]
ROLES = {"Engineering": ["Software Engineer", "QA Analyst", "DevOps Engineer"], "Sales": ["Account Executive", "Sales Development Rep"], "Customer Support": ["Support Agent", "Support Lead"], "Operations": ["Operations Analyst", "Logistics Coordinator"],
         "Finance": ["Financial Analyst", "Accountant"], "HR": ["HR Partner", "Recruiter"], "Marketing": ["Marketing Specialist", "Content Strategist"]}
def make_synth(n=2200, seed=11, M=12):
    r = np.random.default_rng(seed); dates = pd.date_range("2025-07-01", periods=M, freq="MS")
    dp = r.choice(len(DEPTS), n, p=[.24, .20, .18, .16, .08, .06, .08]); role = np.array([r.choice(ROLES[DEPTS[x]]) for x in dp]); lvl = r.choice([1, 2, 3, 4], n, p=[.4, .3, .2, .1])
    ten0 = np.where(r.random(n) < .18, r.integers(0, 6, n), np.clip(r.gamma(2.2, 22, n), 1, 240)).astype(int); compa = np.clip(r.normal(1, .09, n), .7, 1.3)
    sal = np.round(np.array([4200, 3800, 2900, 3300, 4000, 3600, 3500])[dp] * (1 + .28 * (lvl - 1)) * compa); otb = np.clip(np.array([9, 11, 7, 12, 5, 3, 6])[dp] + r.normal(0, 3, n), 0, 40)
    mg = r.integers(0, 45, n); meff = r.normal(0, .35, 45); eng = np.clip(70 - .9 * (otb - 8) + 25 * (compa - 1) - 6 * meff[mg] + r.normal(0, 9, n), 15, 98); trn = r.gamma(3, 1.4, n)
    promos = r.poisson(ten0 / 48); msp0 = np.where(promos > 0, np.minimum(ten0, r.integers(3, 60, n)), ten0)
    z = (.045 * (otb - 8) - .035 * (eng - 68) - 2.2 * (compa - 1) + .012 * np.maximum(msp0 - 24, 0) + .6 * (ten0 < 12) - .12 * (trn - 4) + np.array([.15, .1, .3, .2, -.2, -.25, 0])[dp] + meff[mg] + r.normal(0, .35, n))
    lo, hi = 1e-5, .3
    for _ in range(50):
        s = (lo + hi) / 2
        if (1 - (1 - np.minimum(s * np.exp(z), .99)) ** 12).mean() > .18: hi = s
        else: lo = s
    h = np.minimum(s * np.exp(z), .99); ex = r.random((n, M)) < h[:, None]; has = ex.any(1); first = np.where(has, ex.argmax(1), M)
    t = np.arange(M)[None, :]; fi = first[:, None]; keep = t <= np.where(has, first, M)[:, None]; pre = has[:, None] & (t >= fi - 3) & (t <= fi); ramp = np.where(pre, t - (fi - 3) + 1, 0)
    engm = np.clip(eng[:, None] + r.normal(0, 4, (n, M)) - 2.5 * ramp, 0, 100); otm = np.clip(otb[:, None] + r.normal(0, 3, (n, M)) + 1.0 * ramp, 0, 60)
    ab = r.poisson((.5 + .04 * np.maximum(otb - 8, 0) + .3 * (eng < 55))[:, None] * (1 + .15 * ramp)); js = np.clip(np.round(1 + engm / 100 * 4 + r.normal(0, .5, (n, M))), 1, 5)
    ms = np.clip(np.round(3.3 - 2.0 * meff[mg][:, None] + r.normal(0, .8, (n, M))), 1, 5); wl = np.clip(np.round(4.2 - .09 * otm + r.normal(0, .6, (n, M))), 1, 5)
    th = r.poisson(trn[:, None] * np.ones((1, M))); ts = r.poisson(.5, (n, M)); lv = r.poisson(1.2, (n, M)); perf = np.clip(np.round(r.normal(3.2, .8, n)), 1, 5).astype(int)
    scp = r.choice([0, 2, 3, 5, 8], n); mob = (r.random(n) < .06).astype(int); rc = np.round(sal * r.uniform(.8, 1.4, n)); erc = np.round(sal * 12 * r.uniform(.35, .6, n))
    gen = r.choice(["Female", "Male", "Non-binary"], n, p=[.47, .5, .03]); age = r.choice(["<30", "30-39", "40-49", "50+"], n, p=[.25, .35, .25, .15]); hire = (dates[0] - pd.to_timedelta(ten0 * 30.44, unit="D")).normalize()
    ii, tt = np.where(keep); is_exit = has[ii] & (tt == first[ii])
    df = pd.DataFrame(dict(employee_id=np.array(["E%05d" % (i + 1) for i in range(n)])[ii], observation_date=dates[tt].strftime("%Y-%m-%d"), hire_date=hire[ii].strftime("%Y-%m-%d"),
        exit_date=np.where(is_exit, (dates[tt] + pd.Timedelta(days=14)).strftime("%Y-%m-%d"), None), attrition=is_exit.astype(int), employment_status=np.where(is_exit, "Left", "Active"), department=np.array(DEPTS)[dp][ii],
        job_role=role[ii], job_level=lvl[ii], manager_id=np.array(["M%02d" % (x + 1) for x in range(45)])[mg][ii], performance_rating=perf[ii], workload_hours=np.round(165 + otm[ii, tt] + r.normal(0, 4, len(ii)), 1),
        overtime_hours=np.round(otm[ii, tt], 1), overtime_flag=(otm[ii, tt] > 10).astype(int), absenteeism_days=ab[ii, tt], leave_days=lv[ii, tt], monthly_salary=sal[ii], salary_band=np.array(["L%d" % x for x in lvl])[ii],
        salary_change_percentage=scp[ii], engagement_score=np.round(engm[ii, tt], 1), job_satisfaction=js[ii, tt], manager_satisfaction=ms[ii, tt], work_life_balance=wl[ii, tt], training_hours=th[ii, tt], training_sessions=ts[ii, tt],
        promotion_count=promos[ii], months_since_last_promotion=msp0[ii] + tt, tenure_months=ten0[ii] + tt, internal_mobility=mob[ii], recruitment_cost=rc[ii], estimated_replacement_cost=erc[ii], gender=gen[ii], age_band=age[ii]))
    for c, fr in [("engagement_score", .03), ("overtime_hours", .015), ("training_hours", .02)]: df.loc[r.random(len(df)) < fr, c] = np.nan
    return pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)
SAMPLE = make_synth()

# ------------------------------ ingestion & validation ------------------------------
SYN = {"employee_id": ["emp_id", "employeeid", "staff_id", "worker_id", "id"], "observation_date": ["date", "snapshot_date", "month", "period", "as_of_date"], "hire_date": ["start_date", "date_of_joining", "joining_date"],
       "exit_date": ["termination_date", "separation_date", "leaving_date", "last_working_day"], "attrition": ["left", "turnover", "is_leaver", "resigned", "exited"], "employment_status": ["status", "emp_status"],
       "department": ["dept", "business_unit", "function"], "job_role": ["role", "job_title", "position", "title"], "job_level": ["level", "grade"], "manager_id": ["manager", "supervisor_id", "line_manager"],
       "performance_rating": ["performance", "perf_rating", "rating"], "workload_hours": ["hours_worked", "monthly_hours", "working_hours"], "overtime_hours": ["overtime", "ot_hours"], "overtime_flag": ["overtime_yn", "works_overtime"],
       "absenteeism_days": ["absence_days", "absences", "sick_days"], "leave_days": ["leave_taken"], "monthly_salary": ["salary", "monthly_income", "monthly_pay", "pay"], "salary_band": ["pay_band"],
       "salary_change_percentage": ["salary_hike", "pct_salary_hike", "raise_pct"], "engagement_score": ["engagement", "engagement_index"], "job_satisfaction": ["satisfaction", "job_sat"],
       "manager_satisfaction": ["manager_sat", "relationship_satisfaction"], "work_life_balance": ["wlb", "worklife_balance"], "training_hours": ["training_time", "learning_hours"], "training_sessions": ["trainings"],
       "promotion_count": ["promotions", "num_promotions"], "months_since_last_promotion": ["time_since_promotion"], "tenure_months": ["tenure", "months_at_company"], "internal_mobility": ["internal_transfer", "transferred"],
       "recruitment_cost": ["hiring_cost"], "estimated_replacement_cost": ["replacement_cost", "cost_to_replace"], "gender": ["sex"], "age_band": ["age_group", "age"]}
NUMC = ["job_level", "performance_rating", "workload_hours", "overtime_hours", "absenteeism_days", "leave_days", "monthly_salary", "salary_change_percentage", "engagement_score", "job_satisfaction", "manager_satisfaction", "work_life_balance",
        "training_hours", "training_sessions", "promotion_count", "months_since_last_promotion", "tenure_months", "recruitment_cost", "estimated_replacement_cost"]
CATC = ["department", "job_role", "manager_id", "salary_band", "gender", "age_band"]
MEAN = ["workload_hours", "overtime_hours", "absenteeism_days", "leave_days", "engagement_score", "job_satisfaction", "manager_satisfaction", "work_life_balance", "training_hours", "training_sessions"]
SENS = {"race", "ethnicity", "religion", "nationality", "disability", "marital", "pregnancy", "orientation", "caste"}
TB = ["0-6", "7-12", "13-24", "25-48", "49+"]
def ingest(raw, mapping="", salt=None):
    W = []; dq = {}; df = raw.copy(); df.columns = [norm(c) for c in df.columns]; ov = parse_map(mapping)
    bad = [k + "=" + v for k, v in ov.items() if k not in SYN or v not in df.columns]
    if bad: W.append("Mapping ignored (unknown field or column): " + ", ".join(bad) + ".")
    ren = {}
    for k, al in SYN.items():
        c = ov[k] if ov.get(k) in df.columns else next((a for a in [k] + al if a in df.columns and a not in ren), None)
        if c is None or c in ren: continue
        if c != k and k in df.columns: df = df.drop(columns=[k])
        ren[c] = k
    df = df.rename(columns=ren); df = df.loc[:, ~df.columns.duplicated()]
    if "employee_id" not in df.columns: raise ValueError("Mandatory field missing: employee_id. Columns found: " + ", ".join(map(str, raw.columns)) + ". Use the mapping box, e.g. employee_id=emp_no.")
    extra = [c for c in df.columns if c not in SYN and set(c.split("_")) & SENS]
    if extra: W.append("Potentially sensitive columns (" + ", ".join(extra) + ") are ignored and never used.")
    col = lambda k: df[k] if k in df.columns else pd.Series(np.nan, index=df.index)
    d = pd.DataFrame({"employee_id": col("employee_id").astype(str).str.strip()}); dq["rows_in"] = len(raw)
    for c in NUMC:
        s = col(c)
        if c == "job_level" and s.notna().any() and not pd.api.types.is_numeric_dtype(s): s = s.astype(str).str.extract(r"(\d+)")[0]
        d[c] = to_num(s) if s.notna().any() else np.nan
        nb = int((d[c].isna() & s.notna() & ~s.astype(str).str.strip().str.lower().isin(BLANK)).sum())
        if nb: W.append("%d non-numeric values in '%s' treated as missing." % (nb, c))
        m = d[c] < 0
        if m.any(): d.loc[m, c] = np.nan; W.append("%d negative values in '%s' set to missing." % (int(m.sum()), c))
    for c in ["overtime_flag", "internal_mobility"]:
        s = col(c); d[c] = to_bool(s).astype(float).where(s.notna()) if s.notna().any() else np.nan
    for c in ["observation_date", "hire_date", "exit_date"]:
        s = col(c); d[c] = to_dt(s) if s.notna().any() else pd.NaT
        nb = int((d[c].isna() & s.notna() & ~s.astype(str).str.strip().str.lower().isin(BLANK)).sum())
        if nb: W.append("%d unparseable dates in '%s' treated as missing." % (nb, c))
    for c in CATC:
        s = col(c); d[c] = clean_str(s) if s.notna().any() else np.nan
    st = col("employment_status"); d["status_left"] = st.astype(str).str.lower().str.contains(r"left|term|resign|exit|separat|inactive|quit|former") & st.notna(); d["flag"] = parse_flag(col("attrition"))
    has_out = bool(d["flag"].notna().any() or st.notna().any() or d["exit_date"].notna().any()); miss = d["employee_id"].str.lower().isin(BLANK)
    if miss.any(): W.append("%d rows without an employee_id were removed." % int(miss.sum())); d = d[~miss]
    if d["monthly_salary"].median() > 100000: W.append("Median monthly_salary is %s: this looks like an annual figure. Check the units before using compensation and cost results." % nf(d["monthly_salary"].median()))
    if d["workload_hours"].max() > 400: W.append("workload_hours above 400 per month found: check whether the field is monthly.")
    n_ot = int((d["overtime_hours"] > d["workload_hours"]).sum())
    if n_ot: W.append("%d rows have overtime_hours above workload_hours (check definitions)." % n_ot)
    if (d["performance_rating"] > 10).any(): W.append("performance_rating values above 10 found: check the scale.")
    n0 = len(d); d = d.drop_duplicates(); dq["dup_exact"] = n0 - len(d)
    if dq["dup_exact"]: W.append("%d exact duplicate rows removed." % dq["dup_exact"])
    key = ["employee_id", "observation_date"] if d["observation_date"].notna().any() else ["employee_id"]; n1 = len(d); d = d.sort_values(key, kind="mergesort").drop_duplicates(key, keep="last"); dq["dup_key"] = n1 - len(d)
    if dq["dup_key"]: W.append("%d rows repeated the same employee%s and the last one was kept." % (dq["dup_key"], " and date" if len(key) == 2 else ""))
    if salt: d["employee_id"] = pseudo(d["employee_id"], salt)
    panel = bool(d["observation_date"].notna().any() and d.groupby("employee_id").size().median() > 1.5); d["left_row"] = (d["flag"] == 1) | d["status_left"] | d["exit_date"].notna()
    ex0 = d.groupby("employee_id")["exit_date"].min(); ex = ex0.copy()
    if panel: ex = ex.fillna(d[d["left_row"]].groupby("employee_id")["observation_date"].min())
    dq["derived_exit"] = int((ex0.isna() & ex.notna()).sum()); dq["post_exit"] = 0
    if dq["derived_exit"]: W.append("%d leavers had no exit_date: the first observation flagged as left was used." % dq["derived_exit"])
    if panel:
        post = d["observation_date"] > d["employee_id"].map(ex) + pd.Timedelta(days=31); dq["post_exit"] = int(post.sum())
        if post.any(): W.append("%d observations recorded more than 31 days after an employee's exit were removed (never used)." % int(post.sum())); d = d[~post]
    bh = int((ex.reindex(d["employee_id"]).values < d["hire_date"].values).sum()) if d["hire_date"].notna().any() else 0
    if bh: W.append("%d rows have an exit date before the hire date (check records)." % bh)
    g = d.groupby("employee_id", sort=False); E = d.drop_duplicates("employee_id", keep="last").set_index("employee_id"); mm = g[MEAN].mean()
    for c in MEAN: E[c] = mm[c].reindex(E.index)
    E["left"] = g["left_row"].any().reindex(E.index).astype(bool) if has_out else False; E["exit_date"] = ex.reindex(E.index); E["first_obs"] = g["observation_date"].min().reindex(E.index); E["last_obs"] = g["observation_date"].max().reindex(E.index)
    ref = E["exit_date"].fillna(E["last_obs"]); der = (ref - E["hire_date"]).dt.days / 30.44; m = E["tenure_months"].isna() & der.notna(); E.loc[m, "tenure_months"] = der[m].clip(lower=0)
    try:
        gm = E.groupby(["department", "job_level"])["monthly_salary"]; E["compa_ratio"] = E["monthly_salary"] / gm.transform("median").where(gm.transform("count") >= 5)
    except Exception: E["compa_ratio"] = np.nan
    E["tenure_band"] = pd.cut(E["tenure_months"], [-.01, 6, 12, 24, 48, np.inf], labels=TB).astype(object); E["lf"] = E["left"]
    if not has_out: W.append("No attrition outcome (attrition flag, employment status or exit date) found: attrition rates, drivers and the model are unavailable; only descriptive workforce metrics are shown.")
    if not panel: W.append("Snapshot data (one record per employee): trends, pre-exit changes and a forward-looking model split are unavailable. Features may have been recorded at or after the exit decision, so treat results as associations only.")
    sens_cols = [c for c in ["gender", "age_band"] if d[c].notna().any()]; return dict(d=d.reset_index(drop=True), E=E, W=W, dq=dq, panel=panel, has_out=has_out, sens=sens_cols, raw_n=len(raw), key=key)
def resolve_cost(E, pct=.4):
    if E["estimated_replacement_cost"].notna().any(): return E["estimated_replacement_cost"], "recorded estimated_replacement_cost"
    if E["recruitment_cost"].notna().any(): return E["recruitment_cost"], "recorded recruitment_cost only (excludes onboarding, vacancy and productivity costs)"
    if E["monthly_salary"].notna().any(): return E["monthly_salary"] * 12 * pct, "ASSUMPTION: %.0f%% of annual salary (no cost column supplied)" % (pct * 100)
    return pd.Series(np.nan, index=E.index), "unavailable (no cost or salary data)"
def filt(E, depts=None, roles=None, bands=None, d1="", d2=""):
    X = E.copy()
    if depts: X = X[X["department"].isin(depts)]
    if roles: X = X[X["job_role"].isin(roles)]
    if bands: X = X[X["tenure_band"].isin(bands)]
    D1 = pd.to_datetime(d1, errors="coerce") if d1 else pd.NaT; D2 = pd.to_datetime(d2, errors="coerce") if d2 else pd.NaT; lf = X["left"].copy()
    if pd.notna(D2): X = X[X["first_obs"] <= D2]
    if pd.notna(D1): X = X[~(X["left"] & (X["exit_date"] < D1))]
    lf = X["left"].copy()
    if pd.notna(D1): lf &= X["exit_date"] >= D1
    if pd.notna(D2): lf &= X["exit_date"] <= D2
    X["lf"] = lf.astype(bool); return X

# ------------------------------ analytics ------------------------------
def rate_table(X, col):
    g = X.dropna(subset=[col]).groupby(col, observed=True)["lf"].agg(["sum", "count"]).reset_index(); g.columns = [col, "exits", "n"]; g["rate"] = g["exits"] / g["n"]
    ci = [wilson(a, b) for a, b in zip(g["exits"], g["n"])]; g["lo"] = [c[0] for c in ci]; g["hi"] = [c[1] for c in ci]; ov = X["lf"].mean(); g["pp"] = g["rate"] - ov; g["flag"] = np.where(g["n"] < 30, "small n", "")
    if col == "tenure_band": g["_o"] = g[col].map({b: i for i, b in enumerate(TB)}); g = g.sort_values("_o").drop(columns="_o")
    return g
def group_table(X, col, cost):
    Y = X.dropna(subset=[col]).copy(); Y["_c"] = cost.reindex(Y.index).where(Y["lf"])
    g = Y.groupby(col, observed=True).agg(n=("lf", "size"), exits=("lf", "sum"), ot=("overtime_hours", "mean"), wl=("workload_hours", "mean"), eng=("engagement_score", "mean"), msp=("months_since_last_promotion", "mean"), sal=("monthly_salary", "mean"),
                                         trn=("training_hours", "mean"), absn=("absenteeism_days", "mean"), repl=("_c", lambda s: s.sum(min_count=1))).reset_index()
    g["rate"] = g["exits"] / g["n"]; ci = [wilson(a, b) for a, b in zip(g["exits"], g["n"])]; g["lo"] = [c[0] for c in ci]; g["hi"] = [c[1] for c in ci]; ov = X["lf"].mean(); g["pp"] = g["rate"] - ov
    g["status"] = np.where(g["n"] < 30, "small n", np.where(g["lo"] > ov, "elevated", np.where(g["pp"] >= .03, "watch", ""))); return g.sort_values("rate", ascending=False)
DIRN = {"overtime_hours": "high", "workload_hours": "high", "absenteeism_days": "high", "leave_days": "high", "months_since_last_promotion": "high"}
DRV_NUMS = ["overtime_hours", "workload_hours", "absenteeism_days", "leave_days", "monthly_salary", "compa_ratio", "salary_change_percentage", "engagement_score", "job_satisfaction", "manager_satisfaction", "work_life_balance", "training_hours",
            "training_sessions", "promotion_count", "months_since_last_promotion", "tenure_months", "performance_rating", "job_level"]
DCOLS = ["f", "n_ex", "n_st", "mean_st", "mean_ex", "auc", "p", "p_adj", "rate_lo", "rate_hi", "pp", "overlap", "strength"]
def avail_nums(X): return [f for f in DRV_NUMS if f in X.columns and X[f].notna().mean() >= .5]
def drivers(X, nums):
    rows = []; k = max(len(nums), 1)
    for f in nums:
        s = X[f]; a = s[~X["lf"]].dropna(); b = s[X["lf"]].dropna()
        if len(a) < 30 or len(b) < 15 or s.nunique() < 3: continue
        u, p = stats.mannwhitneyu(b, a, alternative="two-sided"); auc = u / (len(a) * len(b))
        try: q = pd.qcut(s, 3, duplicates="drop"); cd = pd.Series(q.cat.codes, index=s.index).where(s.notna(), -1)
        except Exception: continue
        if cd.max() < 1: continue
        lm, hm = cd == 0, cd == cd.max(); nl, nh = int(lm.sum()), int(hm.sum()); kl, kh = int(X.loc[lm, "lf"].sum()), int(X.loc[hm, "lf"].sum())
        if nl == 0 or nh == 0: continue
        cl, ch = wilson(kl, nl), wilson(kh, nh); ov = not (cl[0] > ch[1] or ch[0] > cl[1]); d_ = kh / nh - kl / nl; padj = min(1.0, p * k)
        stg = "Strong" if (padj < .05 and abs(d_) >= .04 and not ov and len(b) >= 30) else ("Tentative" if (p < .05 or abs(d_) >= .05) else "Weak")
        rows.append(dict(f=f, n_ex=len(b), n_st=len(a), mean_st=a.mean(), mean_ex=b.mean(), auc=auc, p=p, p_adj=padj, rate_lo=kl / nl, rate_hi=kh / nh, pp=d_, overlap=ov, strength=stg))
    d = pd.DataFrame(rows, columns=DCOLS)
    return d.assign(_s=(d["auc"] - .5).abs()).sort_values("_s", ascending=False).drop(columns="_s") if len(d) else d
def inter_tbl(X, a, b):
    if a not in X.columns or b not in X.columns or a == b: return pd.DataFrame()
    A = X[a] >= X[a].median(); B = X[b] >= X[b].median(); ok = X[a].notna() & X[b].notna(); rows = []
    for la, ma in (("high " + pretty(a), A), ("low " + pretty(a), ~A)):
        for lb, mb in (("high " + pretty(b), B), ("low " + pretty(b), ~B)):
            m = ok & ma & mb; n = int(m.sum()); k = int(X.loc[m, "lf"].sum()); lo, hi = wilson(k, n); rows.append(dict(a=la, b=lb, n=n, exits=k, rate=k / n if n else np.nan, lo=lo, hi=hi, flag="small n" if n < 30 else ""))
    return pd.DataFrame(rows)
def pre_exit(d, X):
    L = d[d["employee_id"].isin(X.index)].sort_values(["employee_id", "observation_date"], kind="mergesort").copy(); L["rk"] = L.groupby("employee_id").cumcount(ascending=False); rows = []
    for f in ["engagement_score", "overtime_hours", "absenteeism_days", "job_satisfaction"]:
        if L[f].notna().sum() < 50: continue
        rec = L[L["rk"] < 3].groupby("employee_id")[f].mean(); pri = L[L["rk"] >= 3]; cnt = pri.groupby("employee_id")[f].count(); pm = pri.groupby("employee_id")[f].mean().where(cnt >= 2); ch = (rec - pm).dropna()
        lf = X["lf"].reindex(ch.index).astype(bool); a, b = ch[lf], ch[~lf]
        if len(a) < 10 or len(b) < 10: continue
        rows.append(dict(f=f, n_l=len(a), n_s=len(b), ch_l=a.mean(), ch_s=b.mean(), diff=a.mean() - b.mean(), p=stats.mannwhitneyu(a, b).pvalue))
    return pd.DataFrame(rows, columns=["f", "n_l", "n_s", "ch_l", "ch_s", "diff", "p"])
def eta2(y, g):
    y = np.asarray(y, float); g = pd.Series(g).astype(str).values; m = ~np.isnan(y); y, g = y[m], g[m]
    if len(y) < 30 or np.var(y) == 0: return np.nan
    mu = y.mean(); return sum(len(y[g == k]) * (y[g == k].mean() - mu) ** 2 for k in np.unique(g)) / ((y - mu) ** 2).sum()

# ------------------------------ intervention engine ------------------------------
RUL = [dict(id="workload", feats=["overtime_hours", "workload_hours"], adv="high", name="Workload redistribution and overtime review", hyp="Sustained heavy workload may reduce recovery time and raise exit risk.", out="Lower sustained overtime and workload peaks in the affected teams.", res="Manager time, possible backfill or hiring for peak load, workload tooling (cost: HR to estimate).", tm="2-4 months to redistribute work; review at 6 and 12 months.",
             lead="Average overtime hours, share of staff above the overtime threshold", kpi="Voluntary exits in the cohort", guard="Output and quality metrics, customer service levels", ev="Staggered rollout across teams with comparison teams; compare exits and overtime at 6 and 12 months.", risk="Overtime may reflect staffing levels or seasonality; high performers may carry more overtime."),
       dict(id="engage", feats=["engagement_score", "job_satisfaction", "work_life_balance"], adv="low", name="Structured stay interviews and employee listening (with flexible-work review where work-life balance is low)", hyp="Low engagement or satisfaction may reflect unmet needs that managers have not yet heard.", out="Earlier detection of team-level concerns and fewer avoidable exits.", res="Trained interviewers, interview guide, follow-up action tracking.", tm="Pilot in 1-2 months; run quarterly.",
             lead="Interview completion rate, number of actions closed", kpi="Engagement and satisfaction scores, voluntary exits", guard="Confidentiality complaints, survey response rate", ev="Pilot teams vs matched comparison teams, same survey and exit definitions.", risk="Low engagement can also follow exit intentions (reverse direction); surveys may be biased by response patterns."),
       dict(id="manager", feats=["manager_satisfaction"], adv="low", name="Manager coaching and team communication", hyp="Weaker manager relationships may contribute to exits in some teams.", out="Better manager-employee relationships in teams with low scores.", res="Coaching programme, facilitator time.", tm="3-6 months.", lead="Manager satisfaction pulse, 1:1 frequency",
             kpi="Team-level exits", guard="Manager workload, team performance", ev="Randomise or stagger coaching by team; compare exits and satisfaction against waiting-list teams.", risk="Manager effects are confounded with team workload and role; small teams give noisy rates."),
       dict(id="comp", feats=["compa_ratio", "salary_change_percentage"], adv="low", name="Compensation benchmarking and pay-equity review", hyp="Pay below peers in the same department and level may raise exit risk.", out="Pay positions that are competitive and consistent for comparable roles.", res="Market data, HR analyst time, budget for adjustments if justified.", tm="2-3 months for review; adjustments in the next cycle.",
             lead="Share of employees below the pay-band midpoint", kpi="Exits among under-benchmarked roles", guard="Pay-equity gaps, total payroll cost", ev="Where adjustments are made, compare against eligible but unadjusted groups; do not adjust pay on a model score alone.", risk="Pay is confounded with level, tenure and performance; adjustments are costly and may not change exits."),
       dict(id="career", feats=["months_since_last_promotion"], adv="high", name="Career-path clarity and promotion-process review", hyp="Long gaps since promotion may signal limited visible progression.", out="Transparent promotion criteria and timelines.", res="HRBP and manager time, career framework.", tm="3-6 months.", lead="Share with a documented development plan",
             kpi="Exits among long-tenure, long-since-promotion cohorts", guard="Promotion fairness and cost", ev="Compare cohorts before and after the process change using the same definitions.", risk="Promotions may lag because of role structure rather than employer practice."),
       dict(id="train", feats=["training_hours", "training_sessions"], adv="low", name="Training and upskilling programme", hyp="Limited development opportunities may reduce attachment.", out="More employees using development opportunities linked to career paths.", res="Learning budget, course catalogue, manager time.", tm="3-6 months.", lead="Training participation and completion",
             kpi="Exits among low-training cohorts", guard="Productivity during training, cost", ev="Offer to randomly selected eligible employees first and compare with the waiting group.", risk="Training uptake may reflect motivation or workload rather than opportunity."),
       dict(id="absence", feats=["absenteeism_days"], adv="high", name="Review of absenteeism patterns and workplace support", hyp="Higher absence may reflect workload strain or lack of support in some teams (do not infer individual circumstances).", out="Better workplace support where absence clusters.", res="HR analysis, wellbeing resources.", tm="2-4 months.", lead="Team-level absence trends",
             kpi="Team-level exits", guard="Employee privacy; avoid any individual-level surveillance", ev="Review aggregated team patterns and compare after support changes.", risk="Absence has many causes (health, caring duties); never treat it as a performance signal.")]
HINT = {"overtime_hours": "ot", "workload_hours": "wl", "engagement_score": "eng", "months_since_last_promotion": "msp", "training_hours": "trn", "absenteeism_days": "absn"}
def target_cohort(X, tab, f):
    m = HINT.get(f); 
    if not m or tab is None or m not in tab.columns: return "organisation-wide (no department-level data for this measure)"
    sd = X[f].std(); ov = X[f].mean(); hot = tab[tab["status"].isin(["elevated", "watch"])]
    if not len(hot) or not sd or sd == 0: return "organisation-wide (no department stands out on attrition)"
    adv = [r["department"] for _, r in hot.iterrows() if (r[m] - ov) / sd * (1 if DIRN.get(f) == "high" else -1) >= .2]
    return ", ".join(adv[:3]) + " (elevated attrition and adverse %s)" % pretty(f).lower() if adv else "organisation-wide (elevated departments do not show an adverse level on this measure)"
def build_playbook(X, drv, tab):
    rows = []
    for rl in RUL:
        best = None
        for f in rl["feats"]:
            r = drv[drv["f"] == f]
            if not len(r): continue
            r = r.iloc[0]; ok = (r["pp"] > 0 and r["auc"] > .5) if rl["adv"] == "high" else (r["pp"] < 0 and r["auc"] < .5)
            if ok and r["strength"] in ("Strong", "Tentative") and (best is None or abs(r["auc"] - .5) > abs(best["auc"] - .5)): best = r
        if best is None: continue
        f = best["f"]; adv_txt = "highest" if rl["adv"] == "high" else "lowest"
        ev = "%s: attrition %s among the %s third vs %s among the opposite third (%s; %s exits; adjusted p = %s). Association only." % (pretty(f), pc1(best["rate_hi"] if rl["adv"] == "high" else best["rate_lo"]), adv_txt, pc1(best["rate_lo"] if rl["adv"] == "high" else best["rate_hi"]), pp(abs(best["pp"])), nf(best["n_ex"]), "<0.001" if best["p_adj"] < .001 else fx(best["p_adj"], 3))
        rows.append(dict(name=rl["name"], level=("Strong association" if best["strength"] == "Strong" else "Tentative association"), ev=ev, hyp=rl["hyp"], out=rl["out"], cohort=target_cohort(X, tab, f), res=rl["res"], tm=rl["tm"], lead=rl["lead"], kpi=rl["kpi"], guard=rl["guard"], evald=rl["ev"], risk=rl["risk"], f=f))
    if "tenure_band" in X.columns and X["tenure_band"].notna().any():
        t = rate_table(X, "tenure_band"); e = t[(t["tenure_band"].isin(["0-6", "7-12"])) & (t["n"] >= 40) & (t["pp"] >= .05)]
        if len(e): rows.append(dict(name="Onboarding support for early-tenure employees", level="Tentative association" if e["lo"].max() <= X["lf"].mean() else "Strong association", ev="; ".join("tenure %s: attrition %s (%s vs overall; n=%s)" % (r["tenure_band"], pc1(r["rate"]), pp(r["pp"]), nf(r["n"])) for _, r in e.iterrows()), hyp="Early-tenure exits may reflect onboarding, role-fit or expectation gaps.", out="Smoother first-year experience.", cohort="employees in their first 12 months", res="Buddy programme, structured 30-60-90 day plans.", tm="Immediate; review at 6 months.", lead="Onboarding checklist completion, 90-day pulse", kpi="12-month retention of new hires", guard="Manager workload, new-hire productivity", evald="Compare cohorts hired before and after, or randomise buddy assignment.", risk="Hiring-quality differences between cohorts can mimic onboarding effects.", f="tenure_band"))
    if not rows: rows.append(dict(name="Structured stay interviews and employee listening", level="Generic best practice", ev="No organisational pattern in this data reached the evidence threshold; this is a general, low-risk practice rather than a data-driven finding.", hyp="Unknown: the data does not point to a specific cause.", out="Earlier visibility of concerns.", cohort="organisation-wide", res="Interview guide, trained interviewers.", tm="Pilot in 1-2 months.", lead="Interview completion", kpi="Voluntary exits", guard="Confidentiality", evald="Pilot teams vs comparison teams.", risk="No evidence of benefit from this dataset.", f=""))
    P = pd.DataFrame(rows); P.insert(0, "#", range(1, len(P) + 1)); return P
PBC = [("name", "Intervention"), ("level", "Evidence level"), ("ev", "Observed evidence"), ("hyp", "Concern / hypothesis"), ("out", "Intended outcome"), ("cohort", "Target cohort"), ("res", "Resources and cost (HR to estimate)"), ("tm", "Timeframe"), ("lead", "Leading indicators"), ("kpi", "Outcome KPI"), ("guard", "Guardrail"), ("evald", "Evaluation design"), ("risk", "Risks and alternative explanations")]

# ------------------------------ ROI simulator ------------------------------
def roi_calc(n, cov, base, months, eff, cpe, rec, onb, vac):
    exits0 = n * base * months / 12; avoided = n * cov * base * months / 12 * eff; cpx = rec + onb + vac; gross = avoided * cpx; prog = n * cov * cpe; net = gross - prog
    den = n * cov * base * months / 12 * cpx; return dict(exits0=exits0, exits1=exits0 - avoided, avoided=avoided, gross=gross, prog=prog, net=net, roi=(net / prog if prog > 0 else np.nan), be=(prog / den if den > 0 else np.nan), cpx=cpx)
SCN_COLS = ["Intervention", "Coverage %", "Assumed relative reduction %", "Program cost per employee"]
def roi_defaults():
    if "E" not in G: return (1000, 15.0, 12, 0.0, 0.0, 0.0, pd.DataFrame(columns=SCN_COLS))
    E = G["E"]; n = int((~E["left"]).sum()); X = G["E"]; rate = X["lf"].mean() if G["B"]["has_out"] else np.nan
    mths = (G["d"]["observation_date"].max() - G["d"]["observation_date"].min()).days / 30.44 + 1 if G["B"]["panel"] else 12; base = rate * 12 / mths if pd.notna(rate) and mths > 0 else np.nan
    c = G["cost"][E["left"]].mean(); names = G["PB"]["name"].tolist()[:4] if "PB" in G else ["Intervention A"]; df = pd.DataFrame([[nm, 50.0, 10.0, 150.0] for nm in names], columns=SCN_COLS)
    return (n, round(float(base) * 100, 1) if pd.notna(base) else 15.0, 12, round(float(c)) if pd.notna(c) else 0.0, 0.0, 0.0, df)
def roi_run(n, base, months, rec, onb, vac, lowm, highm, tab):
    d = pd.DataFrame(tab).copy()
    if d.shape[1] != 4: raise ValueError("The scenario table was altered; press 'Reset to data-derived defaults'.")
    d.columns = SCN_COLS; num = SCN_COLS[1:]; d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    if d[num].isna().any().any() or (d[num] < 0).any().any(): raise ValueError("Scenario table needs non-negative numbers in every cell.")
    if (d["Coverage %"] > 100).any() or (d["Assumed relative reduction %"] > 100).any(): raise ValueError("Coverage and reduction must be at most 100%.")
    if min(n, base, months, rec, onb, vac) < 0 or base > 100: raise ValueError("Global assumptions must be non-negative; baseline attrition at most 100%.")
    out = []
    for _, r in d.iterrows():
        for sc, m in (("Low", lowm), ("Base", 1.0), ("High", highm)): out.append(dict(name=r["Intervention"], sc=sc, **roi_calc(n, r["Coverage %"] / 100, base / 100, months, min(1.0, r["Assumed relative reduction %"] / 100 * m), r["Program cost per employee"], rec, onb, vac)))
    return d, pd.DataFrame(out)
def render_roi(n, base, months, rec, onb, vac, lowm, highm, tab):
    if "E" not in G: return note("Load data first.", "info"), empty("No data"), empty("No data"), None
    try: d, R = roi_run(float(n or 0), float(base or 0), float(months or 0), float(rec or 0), float(onb or 0), float(vac or 0), float(lowm), float(highm), tab)
    except Exception as e: return note("Invalid scenario input: " + esc(e), "err"), empty("Invalid input"), empty("Invalid input"), None
    h = note("<b>Scenario estimates, not measured savings or proven causal effects.</b> Not every exit is preventable and not every replacement cost is avoidable. HR must validate the assumptions before any budget use. Overlapping programmes reach the same employees: <b>do not add rows together</b>.", "warn", "scn")
    h += note("<b>Formulas.</b> baseline exits = employees x baseline annual rate x months/12; exits avoided = employees x coverage x baseline x months/12 x assumed relative reduction; avoided cost = exits avoided x (recruitment + onboarding/training + vacancy cost per exit); programme cost = employees x coverage x cost per employee; net benefit = avoided cost - programme cost; ROI = net / programme cost (only if cost &gt; 0); break-even reduction = programme cost / (employees x coverage x baseline x months/12 x cost per exit).", "info")
    if (rec + onb + vac) <= 0: h += note("Cost per exit is zero: enter recruitment, onboarding and vacancy costs, otherwise every scenario shows only costs.", "warn")
    B = R[R["sc"] == "Base"]; h += H("Base case by intervention") + tbl(B, [("name", "Intervention"), ("exits0", "Baseline exits"), ("exits1", "Scenario exits"), ("avoided", "Exits avoided (hypothetical)"), ("gross", "Avoided replacement cost"), ("prog", "Programme cost"), ("net", "Net estimated benefit"), ("roi", "ROI"), ("be", "Break-even reduction")],
                                                                      dict(exits0=lambda v: fx(v, 1), exits1=lambda v: fx(v, 1), avoided=lambda v: fx(v, 1), gross=money, prog=money, net=lambda v: "<span style='color:%s'>%s</span>" % (C["mi"] if v >= 0 else C["co"], money(v)), roi=lambda v: "—" if nul(v) else "%.0f%%" % (v * 100),
                                                                           be=lambda v: "—" if nul(v) else ("above 100% (cannot break even)" if v > 1 else "%.1f%%" % (v * 100))))
    L = R.pivot(index="name", columns="sc", values="net").reset_index(); h += H("Net benefit: low / base / high (user-defined effect multipliers)") + tbl(L, [("name", "Intervention"), ("Low", "Low"), ("Base", "Base"), ("High", "High")], dict(Low=money, Base=money, High=money))
    f1 = go.Figure([go.Bar(x=L["name"], y=L[s], name=s, marker_color=c) for s, c in (("Low", C["mu"]), ("Base", C["mi"]), ("High", C["la"]))]); f1.update_layout(barmode="group"); f1 = sty(f1, 380, "Estimated net benefit by scenario (illustrative)")
    r0 = d.iloc[0]; xs = np.linspace(0, .4, 17); ys = [roi_calc(float(n), r0["Coverage %"] / 100, float(base) / 100, float(months), e, r0["Program cost per employee"], float(rec), float(onb), float(vac))["net"] for e in xs]
    f2 = sty(go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C["la"], width=3))), 340, "Sensitivity of net benefit to the assumed effect: " + str(r0["Intervention"])[:50]); f2.update_xaxes(title="Assumed relative attrition reduction", tickformat=".0%"); f2.update_yaxes(title="Net benefit")
    p = "pulse_scenarios.csv"; R.drop(columns=[]).to_csv(p, index=False); G["scn_R"] = R; return h, f1, f2, p

# ------------------------------ ML pipeline ------------------------------
FN = ["tenure_months", "job_level", "performance_rating", "workload_hours", "overtime_hours", "absenteeism_days", "leave_days", "salary_change_percentage", "compa_ratio", "engagement_score", "job_satisfaction", "manager_satisfaction", "work_life_balance",
      "training_hours", "training_sessions", "promotion_count", "months_since_last_promotion", "internal_mobility", "d_engagement", "d_overtime"]
FC = ["department", "job_role"]; OBJ = {"Balanced (F1)": 1.0, "Recall-weighted (F2)": 4.0, "Precision-weighted (F0.5)": .25}
def pick_feats(X):
    nums = [c for c in FN if c in X.columns and X[c].notna().mean() >= .6 and X[c].nunique() >= 2]; cats = [c for c in FC if c in X.columns and 2 <= X[c].nunique() <= 30 and X[c].notna().mean() >= .6]; return nums, cats
def prep_X(df, nums, cats):
    X = pd.DataFrame(index=df.index)
    for c in nums: X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cats: X[c] = df[c].astype(object).where(df[c].notna(), "Unknown").astype(str)
    return X
def frame_at(d, E, S, hz, target=True):
    L = d[d["observation_date"] <= S].sort_values(["employee_id", "observation_date"], kind="mergesort").copy(); L["rk"] = L.groupby("employee_id").cumcount(ascending=False); X = L[L["rk"] == 0].set_index("employee_id").copy()
    rm = L[L["rk"] < 3].groupby("employee_id")[MEAN].mean()
    for c in MEAN: X[c] = rm[c].reindex(X.index)
    for f, nm in (("engagement_score", "d_engagement"), ("overtime_hours", "d_overtime")):
        rec = L[L["rk"] < 3].groupby("employee_id")[f].mean(); pri = L[(L["rk"] >= 3) & (L["rk"] < 9)]; cnt = pri.groupby("employee_id")[f].count(); pm = pri.groupby("employee_id")[f].mean().where(cnt >= 2); X[nm] = (rec - pm).reindex(X.index)
    try:
        gm = X.groupby(["department", "job_level"])["monthly_salary"]; X["compa_ratio"] = X["monthly_salary"] / gm.transform("median").where(gm.transform("count") >= 5)
    except Exception: X["compa_ratio"] = np.nan
    exd = E["exit_date"].reindex(X.index); X = X[~(exd <= S)]; exd = exd.reindex(X.index)
    if not target: return X, None
    y = ((exd > S) & (exd <= S + pd.DateOffset(months=hz) + pd.Timedelta(days=31))).astype(int); return X, y
def make_ohe():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)
def zoo(nums, cats):
    def pre(sc):
        t = []
        if nums: t.append(("n", Pipeline([("i", SimpleImputer(strategy="median"))] + ([("s", StandardScaler())] if sc else [])), nums))
        if cats: t.append(("c", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")), ("o", make_ohe())]), cats))
        return ColumnTransformer(t, remainder="drop")
    return {"Logistic Regression": Pipeline([("p", pre(True)), ("m", LogisticRegression(max_iter=2000, C=.5, class_weight="balanced"))]), "Random Forest": Pipeline([("p", pre(False)), ("m", RandomForestClassifier(n_estimators=120, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
            "Gradient Boosting": Pipeline([("p", pre(False)), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=.06, max_iter=120, l2_regularization=1.0, random_state=42))])}
def calibrated(p):
    try: return CalibratedClassifierCV(estimator=clone(p), method="sigmoid", cv=3)
    except TypeError: return CalibratedClassifierCV(base_estimator=clone(p), method="sigmoid", cv=3)
def oof(m, X, y, n=5): return cross_val_predict(clone(m), X, y, cv=StratifiedKFold(n_splits=n, shuffle=True, random_state=42), method="predict_proba")[:, 1]
def pick_threshold(y, p, b2):
    y = np.asarray(y).astype(int); p = np.asarray(p); g = np.linspace(.02, .95, 94); P = p[:, None] >= g[None, :]; tp = (P & (y[:, None] == 1)).sum(0).astype(float); fp = (P & (y[:, None] == 0)).sum(0).astype(float); fn = (y == 1).sum() - tp
    return float(g[int(np.argmax((1 + b2) * tp / np.maximum((1 + b2) * tp + b2 * fn + fp, 1e-9)))])
def fbeta(y, p, t, b2):
    pr = (np.asarray(p) >= t).astype(int); y = np.asarray(y); tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum(); return (1 + b2) * tp / max((1 + b2) * tp + b2 * fn + fp, 1e-9)
def mets(y, p, t):
    pr = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return dict(roc=roc_auc_score(y, p), pr=average_precision_score(y, p), prec=precision_score(y, pr, zero_division=0), rec=recall_score(y, pr, zero_division=0), f1=f1_score(y, pr, zero_division=0), brier=brier_score_loss(y, p), thr=t, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
def contributions(model, X, ref):
    base = model.predict_proba(X)[:, 1]; Cm = np.zeros((len(X), X.shape[1]))
    for j, f in enumerate(X.columns): Xm = X.copy(); Xm[f] = ref[f]; Cm[:, j] = base - model.predict_proba(Xm)[:, 1]
    return Cm, base
def train_model(X, y, nums, cats, b2):
    M = dict(ok=False, why="", warn=[]); y = np.asarray(y).astype(int)
    if len(y) < 300 or y.sum() < 30 or (len(y) - y.sum()) < 30: M["why"] = "training needs at least 300 eligible employees with 30 or more exits and 30 or more stayers (found %d employees, %d exits)." % (len(y), int(y.sum())); return M
    tr, te = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42); assert set(X.index[tr]).isdisjoint(set(X.index[te])), "employee overlap between train and test"
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]; R = {}
    for nm, pipe in zoo(nums, cats).items():
        try: o = oof(pipe, Xtr, ytr); m = clone(pipe).fit(Xtr, ytr); R[nm] = dict(pipe=pipe, model=m, p=m.predict_proba(Xte)[:, 1], cv=float(roc_auc_score(ytr, o)), thr=pick_threshold(ytr, o, b2))
        except Exception as e: M["warn"].append("%s could not be trained (%s)." % (nm, type(e).__name__))
    if not R: M["why"] = "all candidate models failed to train."; return M
    for r_ in R.values(): r_["test"] = mets(yte, r_["p"], r_["thr"])
    bp = np.full(len(yte), ytr.mean())
    if "department" in cats: rm = pd.Series(ytr, index=Xtr.index).groupby(Xtr["department"]).mean(); bp = Xte["department"].map(rm).fillna(ytr.mean()).values
    base_auc = float(roc_auc_score(yte, bp)) if len(np.unique(bp)) > 1 else .5; best = max(R, key=lambda k: R[k]["cv"]); cal = calibrated(R[best]["pipe"]); oc = oof(cal, Xtr, ytr); thr = pick_threshold(ytr, oc, b2)
    pc = clone(cal).fit(Xtr, ytr).predict_proba(Xte)[:, 1]; final = clone(cal).fit(X, y); imp = None
    try:
        pi = permutation_importance(R[best]["model"], Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=1); imp = pd.DataFrame(dict(f=list(Xte.columns), imp=pi.importances_mean, sd=pi.importances_std)).sort_values("imp", ascending=False)
    except Exception: pass
    rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        i = rng.integers(0, len(yte), len(yte))
        if yte[i].min() != yte[i].max(): bs.append(roc_auc_score(yte[i], pc[i]))
    ref = {f: float(X[f].median()) for f in nums}; ref.update({f: X[f].mode().iat[0] for f in cats}); fb, ob = calibration_curve(yte, pc, n_bins=5, strategy="quantile")
    M.update(ok=True, R=R, best=best, thr=thr, pc=pc, yte=yte, Xte=Xte, final=final, imp=imp, ref=ref, base_auc=base_auc, base_rate=float(y.mean()), n_train=len(tr), n_test=len(te), tr_ids=list(X.index[tr]), te_ids=list(X.index[te]), test=mets(yte, pc, thr),
             ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan, np.nan), calib=pd.DataFrame(dict(pred=ob, obs=fb)), feats=nums + cats, nums=nums, cats=cats, b2=b2); return M

# ------------------------------ core pipeline ------------------------------
def core(raw, mapping="", hz=6, b2=1.0, pct=.4, salt=None):
    B = ingest(raw, mapping, salt); E, d = B["E"], B["d"]; cost, csrc = resolve_cost(E, pct); E["rep_cost"] = cost; R = dict(B=B, E=E, d=d, cost=cost, csrc=csrc, hz=hz, pct=pct, W=list(B["W"]))
    nums = avail_nums(E); R["nums"] = nums; R["drv"] = drivers(E, nums) if B["has_out"] else pd.DataFrame(columns=DCOLS); R["tab"] = group_table(E, "department", cost) if E["department"].notna().any() else None
    R["PB"] = build_playbook(E, R["drv"], R["tab"]) if B["has_out"] else pd.DataFrame(columns=["#"] + [k for k, _ in PBC]); M = dict(ok=False, why="no attrition outcome in the data.", warn=[]); R["now"] = None
    if B["has_out"]:
        try:
            if B["panel"]:
                end = d["observation_date"].max(); S = end - pd.DateOffset(months=hz)
                if S < d["observation_date"].min() + pd.DateOffset(months=3): M["why"] = "the observation window is too short for a %d-month horizon with at least 3 months of history." % hz
                else:
                    Xf, y = frame_at(d, E, S, hz, True); nm_, ct_ = pick_feats(Xf); M = train_model(Xf[nm_ + ct_], y.loc[Xf.index].values, nm_, ct_, b2); M["S"] = S; M["design"] = "Features measured as of %s (last 3 months averaged); outcome = exit within the next %d months; split by employee (25%% held out)." % (S.date(), hz)
                    if M["ok"]:
                        Xn, _ = frame_at(d, E, end, hz, False); Xn = Xn[~E["left"].reindex(Xn.index).fillna(False).astype(bool)]; Xn = Xn[M["feats"]]; pn = M["final"].predict_proba(Xn)[:, 1]
                        R["now"] = pd.DataFrame(dict(p=pn, band=np.where(pn >= M["thr"], "Elevated", np.where(pn >= M["base_rate"], "Moderate", "Lower")), department=E["department"].reindex(Xn.index).values), index=Xn.index); R["Xn"] = Xn
            else:
                Xf = E.copy(); nm_, ct_ = pick_feats(Xf); M = train_model(Xf[nm_ + ct_], Xf["left"].astype(int).values, nm_, ct_, b2); M["design"] = "Snapshot data: one record per employee; outcome = left; random stratified split by employee."
                if M["ok"]: Xn = Xf[~Xf["left"]][M["feats"]]; pn = M["final"].predict_proba(Xn)[:, 1]; R["now"] = pd.DataFrame(dict(p=pn, band=np.where(pn >= M["thr"], "Elevated", np.where(pn >= M["base_rate"], "Moderate", "Lower")), department=E["department"].reindex(Xn.index).values), index=Xn.index); R["Xn"] = Xn
        except Exception as e: M = dict(ok=False, why="model training failed (%s: %s)." % (type(e).__name__, str(e)[:80]), warn=[])
    R["M"] = M; R["W"] += M.get("warn", []); return R
def run_all(file, mapping, hz, obj, pct, pseud_on):
    prev = dict(G); G.clear()
    try:
        demo = file is None; raw = SAMPLE.copy() if demo else read_csv_any(file); salt = os.urandom(6).hex() if (pseud_on and not demo) else None
        R = core(raw, mapping, int(hz or 6), OBJ.get(obj, 1.0), float(pct or 40) / 100, salt); G.update(R); G.update(demo=demo, pseud=bool(salt), obj=obj); M = R["M"]; B = R["B"]; E = R["E"]
        msg = ("Fictional demo data analysed. " if demo else "Your data analysed. ") + "%s employees (%s mode)." % (nf(len(E)), "longitudinal" if B["panel"] else "snapshot")
        msg += (" Attrition model: %s (calibrated), held-out ROC-AUC %.3f vs department baseline %.3f." % (M["best"], M["test"]["roc"], M["base_auc"])) if M["ok"] else " Prediction unavailable: " + M["why"]
        return note(esc(msg), "ok" if M["ok"] else "warn")
    except Exception as e: G.clear(); G.update(prev) if False else None; return note("Could not complete the analysis: " + esc(e), "err")
def refresh():
    if "E" not in G: return tuple(gr.update() for _ in range(8))
    E = G["E"]; nums = G["nums"]; ids = G["now"].sort_values("p", ascending=False).head(30).index.tolist() if G.get("now") is not None else []
    return (gr.update(choices=sorted(E["department"].dropna().unique().tolist()), value=[]), gr.update(choices=sorted(E["job_role"].dropna().unique().tolist()), value=[]), gr.update(choices=[t for t in TB if t in set(E["tenure_band"].dropna())], value=[]),
            gr.update(choices=nums, value=nums[0] if nums else None), gr.update(choices=nums, value=nums[0] if nums else None), gr.update(choices=nums, value=nums[1] if len(nums) > 1 else None), gr.update(choices=ids, value=None), gr.update(value=roi_defaults()[-1]))

# ------------------------------ renderers ------------------------------
def view(dep, rol, ten, d1, d2):
    X = filt(G["E"], dep, rol, ten, d1, d2); return X
def render_overview(dep, rol, ten, d1, d2):
    if "E" not in G: return (note("Load data first.", "info"),) + NOP(3)
    B = G["B"]; X = view(dep, rol, ten, d1, d2); M = G["M"]
    if X.empty: return (note("No employees match the filters.", "warn"),) + NOP(3)
    n = len(X); ex = int(X["lf"].sum()); out = B["has_out"]; rate = ex / n if out else np.nan; lo, hi = wilson(ex, n) if out else (np.nan, np.nan); rep = G["cost"].reindex(X.index)[X["lf"]].sum(min_count=1) if out else np.nan
    h = ""
    if G["demo"]: h += note("<b>Illustrative synthetic data.</b> Fictional employees for demonstration; nothing here describes a real organisation.", "warn")
    h += "<div class='kg'>" + card("Employees in view", nf(n), "unique employees (each counted once)", tg="obs") + card("Still employed", nf(int((~X["left"]).sum())), "headcount in view without a recorded exit", tg="obs") + \
         card("Historical attrition", pc1(rate), "exits ÷ unique employees in view (95%% CI %s-%s)" % (pc1(lo), pc1(hi)) if out else "no outcome column", C["co"], "obs") + card("Exits in selected period", nf(ex) if out else "—", "exit_date or flag based", C["co"], "obs") + \
         card("Replacement cost of exits", money(rep), G["csrc"], C["am"], "obs" if "recorded" in G["csrc"] else "scn") + "</div>"
    drv = drivers(X, avail_nums(X)) if out else pd.DataFrame(columns=DCOLS); top = drv[drv["strength"] != "Weak"].head(3)
    h += H("Highest-priority organisational patterns (associations)") + (tbl(top, [("f", "Factor"), ("pp", "Attrition: high vs low third"), ("auc", "Separation (AUC)"), ("p_adj", "Adj. p"), ("strength", "Evidence")], dict(f=pretty, pp=pp, auc=fx, p_adj=lambda v: fx(v, 3))) if len(top) else note("No pattern reached the evidence threshold in this view.", "info"))
    h += note("Factors are associated with exits; they are not proven causes.", "warn", "hyp")
    h += H("Model evaluation summary") + (note("Best model %s (calibrated): held-out ROC-AUC %s (95%% CI %s-%s), PR-AUC %s, versus a department-average baseline AUC of %s. Decision-support signal only; metrics come from %s." % (esc(M["best"]), fx(M["test"]["roc"], 3), fx(M["ci"][0]), fx(M["ci"][1]), fx(M["test"]["pr"], 3), fx(M["base_auc"], 3), "synthetic data" if G["demo"] else "your data"), "ai", "pred") if M["ok"] else note("Prediction unavailable: " + esc(M["why"]), "warn"))
    if len(G["PB"]): h += H("Retention intervention priorities") + tbl(G["PB"].head(3), [("name", "Intervention"), ("level", "Evidence level"), ("cohort", "Target cohort")])
    f1 = empty("Needs department and outcome data")
    if out and X["department"].notna().any():
        t = rate_table(X, "department").sort_values("rate", ascending=False); f1 = go.Figure(go.Bar(x=t["department"], y=t["rate"], marker_color=[C["co"] if v > rate else C["mi"] for v in t["rate"]], text=["n=%d" % v for v in t["n"]], error_y=dict(type="data", symmetric=False, array=(t["hi"] - t["rate"]).values, arrayminus=(t["rate"] - t["lo"]).values, color=C["mu"])))
        f1 = sty(f1, 360, "Attrition rate by department (95% Wilson interval)"); f1.update_yaxes(tickformat=".0%")
    f2 = f3 = empty("Monthly trends need longitudinal data")
    if B["panel"] and out:
        L = G["d"][G["d"]["employee_id"].isin(X.index)]; mo = L.assign(m=L["observation_date"].dt.to_period("M").dt.to_timestamp()).groupby("m").agg(hc=("employee_id", "nunique"), ot=("overtime_hours", "mean"), eng=("engagement_score", "mean"))
        exm = X[X["lf"]].assign(m=lambda x: x["exit_date"].dt.to_period("M").dt.to_timestamp()).groupby("m").size(); mo["rate"] = exm.reindex(mo.index).fillna(0) / mo["hc"]
        if len(mo) >= 3:
            f2 = sty(go.Figure(go.Scatter(x=mo.index, y=mo["rate"], mode="lines+markers", line=dict(color=C["co"], width=3))), 320, "Monthly turnover (exits ÷ employees observed that month)"); f2.update_yaxes(tickformat=".1%")
            f3 = go.Figure([go.Scatter(x=mo.index, y=mo["ot"], name="Avg overtime hours", line=dict(color=C["am"], width=3)), go.Scatter(x=mo.index, y=mo["eng"], name="Avg engagement", yaxis="y2", line=dict(color=C["mi"], width=3))]); f3 = sty(f3, 320, "Overtime and engagement trend (current filters; period dates not applied)"); f3.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    return h, f1, f2, f3
def render_explorer(dep, rol, ten, d1, d2, by, minn):
    if "E" not in G: return (note("Load data first.", "info"),) + NOP(2)
    X = view(dep, rol, ten, d1, d2); by = by or "department"
    if by not in X.columns or X[by].notna().sum() == 0: return (note("The selected grouping is not available in this data.", "warn"),) + NOP(2)
    t = group_table(X, by, G["cost"]); t = t[t["n"] >= (minn or 10)]
    if t.empty: return (note("No group reaches the minimum size.", "warn"),) + NOP(2)
    ov = X["lf"].mean(); h = note("Aggregated view; groups below the minimum size are hidden. Rates for small groups are unstable (wide intervals). Replacement cost is %s." % esc(G["csrc"]), "info", "obs")
    h += tbl(t, [(by, pretty(by)), ("n", "Employees"), ("exits", "Exits"), ("rate", "Attrition"), ("lo", "CI low"), ("hi", "CI high"), ("pp", "vs overall"), ("ot", "Overtime h"), ("wl", "Workload h"), ("eng", "Engagement"), ("msp", "Months since promotion"), ("sal", "Avg salary"), ("repl", "Replacement cost"), ("status", "Flag")],
             dict(n=nf, exits=nf, rate=pc1, lo=pc1, hi=pc1, pp=pp, ot=fx, wl=lambda v: fx(v, 0), eng=lambda v: fx(v, 1), msp=lambda v: fx(v, 1), sal=money, repl=money))
    f1 = empty("Needs workload and engagement data"); tt = t.dropna(subset=["wl", "eng"])
    if len(tt): f1 = go.Figure(go.Scatter(x=tt["wl"], y=tt["eng"], mode="markers+text", text=tt[by].astype(str), textposition="top center", marker=dict(size=10 + 40 * tt["n"] / tt["n"].max(), color=tt["rate"], colorscale=[[0, C["mi"]], [1, C["co"]]], showscale=True, colorbar=dict(title="Attrition")), hovertemplate="%{text}<br>workload %{x:.0f}h<br>engagement %{y:.1f}<extra></extra>")); f1 = sty(f1, 400, "Workload vs engagement (bubble = size, colour = attrition)"); f1.update_xaxes(title="Avg monthly workload hours"); f1.update_yaxes(title="Avg engagement")
    plan = []; sd = {k: X[c].std() for k, c in (("ot", "overtime_hours"), ("eng", "engagement_score"), ("msp", "months_since_last_promotion"), ("trn", "training_hours"), ("absn", "absenteeism_days"))}; mn = {k: X[c].mean() for k, c in (("ot", "overtime_hours"), ("eng", "engagement_score"), ("msp", "months_since_last_promotion"), ("trn", "training_hours"), ("absn", "absenteeism_days"))}
    NM = dict(ot="Workload redistribution and overtime review", eng="Stay interviews and employee listening", msp="Career-path clarity and promotion review", trn="Training and upskilling", absn="Absenteeism pattern and support review"); SG = dict(ot=1, eng=-1, msp=1, trn=-1, absn=1)
    for _, r in t[t["status"].isin(["elevated", "watch"])].iterrows():
        ad = [(k, r[k]) for k in NM if pd.notna(r[k]) and sd.get(k) and (r[k] - mn[k]) / sd[k] * SG[k] >= .2]; plan.append(dict(g=r[by], rate=r["rate"], st=r["status"], ev="; ".join("%s %s vs org %s" % (NM[k].split(" ")[0].lower() if False else k, fx(v, 1), fx(mn[k], 1)) for k, v in ad) or "no adverse level on the measured conditions", act="; ".join(NM[k] for k, _ in ad) or "Investigate with stay interviews (generic)"))
    h += H("Prioritised organisational action plan (hypotheses to investigate)") + (tbl(pd.DataFrame(plan), [("g", pretty(by)), ("rate", "Attrition"), ("st", "Flag"), ("ev", "Adverse conditions (overtime h / engagement / months since promotion / training h / absence days)"), ("act", "Investigate / intervene")], dict(rate=pc1)) if plan else note("No group shows elevated attrition at this minimum size.", "info"))
    return h, f1
def band_fig(X, f):
    s = X[f]; q = pd.qcut(s, 3, duplicates="drop"); cd = pd.Series(q.cat.codes, index=s.index).where(s.notna(), -1); rows = []
    for c in sorted(cd[cd >= 0].unique()): m = cd == c; k = int(X.loc[m, "lf"].sum()); n = int(m.sum()); lo, hi = wilson(k, n); rows.append((["Lowest third", "Middle third", "Highest third"][min(c, 2)] if cd.max() == 2 else "band %d" % c, k / n, lo, hi, n))
    d = pd.DataFrame(rows, columns=["b", "r", "lo", "hi", "n"]); fig = go.Figure(go.Bar(x=d["b"], y=d["r"], marker_color=C["la"], text=["n=%d" % v for v in d["n"]], error_y=dict(type="data", symmetric=False, array=(d["hi"] - d["r"]).values, arrayminus=(d["r"] - d["lo"]).values, color=C["mu"])))
    fig = sty(fig, 340, "Attrition by %s (95%% interval)" % pretty(f).lower()); fig.update_yaxes(tickformat=".0%"); return fig
def render_drivers(dep, rol, ten, d1, d2, fsel, ia, ib):
    if "E" not in G: return (note("Load data first.", "info"),) + NOP(2)
    if not G["B"]["has_out"]: return (note("Driver analysis needs an attrition outcome.", "warn"),) + NOP(2)
    X = view(dep, rol, ten, d1, d2); drv = drivers(X, avail_nums(X)); ov = X["lf"].mean()
    h = note("Overall attrition in view: <b>%s</b> (%s exits of %s). <b>Associations, not causes.</b> Differences compare the highest and lowest third of each measure. Ranking is by effect size (AUC distance from 0.5), not by raw correlation; 'Strong' needs adjusted p &lt; 0.05, a gap of at least 4 pp, non-overlapping intervals and at least 30 exits. Confounding (for example department, level, tenure) is not removed here." % (pc1(ov), nf(X["lf"].sum()), nf(len(X))), "warn", "obs")
    h += H("Ranked organisational patterns") + (tbl(drv, [("f", "Factor"), ("n_ex", "Exits"), ("n_st", "Stayers"), ("mean_ex", "Mean (exits)"), ("mean_st", "Mean (stayers)"), ("rate_lo", "Attrition, lowest third"), ("rate_hi", "Attrition, highest third"), ("pp", "Difference"), ("overlap", "CIs overlap"), ("auc", "AUC"), ("p_adj", "Adj. p (Bonferroni)"), ("strength", "Evidence")],
                                                  dict(f=pretty, mean_ex=fx, mean_st=fx, rate_lo=pc1, rate_hi=pc1, pp=pp, overlap=lambda v: "yes" if v else "no", auc=fx, p_adj=lambda v: "<0.001" if v < .001 else fx(v, 3))) if len(drv) else note("Too few exits or too little variation for driver analysis in this view (needs at least 15 exits and 30 stayers).", "warn"))
    for c in ["department", "job_level", "tenure_band", "job_role"]:
        if X[c].notna().sum() > 50:
            t = rate_table(X, c); t = t[t["n"] >= 10]; h += H("Attrition by " + pretty(c).lower()) + tbl(t, [(c, pretty(c)), ("n", "Employees"), ("exits", "Exits"), ("rate", "Rate"), ("lo", "CI low"), ("hi", "CI high"), ("pp", "vs overall"), ("flag", "Flag")], dict(n=nf, exits=nf, rate=pc1, lo=pc1, hi=pc1, pp=pp))
    if len(drv) and G["B"]["panel"]:
        pe = pre_exit(G["d"], X)
        if len(pe): h += H("Do conditions change before exits? (change over the last 3 observations vs earlier)") + tbl(pe, [("f", "Measure"), ("n_l", "Leavers"), ("n_s", "Stayers"), ("ch_l", "Change, leavers"), ("ch_s", "Change, stayers"), ("diff", "Difference"), ("p", "p")], dict(f=pretty, ch_l=lambda v: "%+.2f" % v, ch_s=lambda v: "%+.2f" % v, diff=lambda v: "%+.2f" % v, p=lambda v: "<0.001" if v < .001 else fx(v, 3))) + note("A change that precedes the exit is stronger <b>temporal</b> evidence than a cross-section, but it is still not proof of cause (for example, employees who plan to leave may disengage first).", "info", "hyp")
    it = inter_tbl(X, ia, ib)
    if len(it): h += H("Interaction explorer: %s x %s (median splits)" % (pretty(ia), pretty(ib))) + tbl(it, [("a", pretty(ia)), ("b", pretty(ib)), ("n", "Employees"), ("exits", "Exits"), ("rate", "Attrition"), ("lo", "CI low"), ("hi", "CI high"), ("flag", "Flag")], dict(n=nf, exits=nf, rate=pc1, lo=pc1, hi=pc1))
    f1 = empty("No ranked patterns")
    if len(drv): d = drv.head(12).iloc[::-1]; f1 = sty(go.Figure(go.Bar(y=[pretty(x) for x in d["f"]], x=d["auc"] - .5, orientation="h", marker_color=[C["co"] if v > 0 else C["mi"] for v in d["auc"] - .5])), 400, "Effect size: AUC - 0.5 (right = higher among leavers)")
    f2 = band_fig(X, fsel) if fsel in X.columns and X[fsel].notna().sum() > 60 else empty("Select a factor"); return h, f1, f2

def render_model():
    if "M" not in G: return (note("Load data first.", "info"),) + NOP(6)
    M = G["M"]
    if not M["ok"]: return (note("Prediction is disabled: " + esc(M["why"]) + " The descriptive analytics remain fully available.", "warn"),) + NOP(6)
    R, t, yte, pc, thr = M["R"], M["test"], M["yte"], M["pc"], M["thr"]; rows = [dict(m=n + (" *" if n == M["best"] else ""), k="raw score (class-weighted)", cv=r["cv"], roc=r["test"]["roc"], pr=r["test"]["pr"], prec=r["test"]["prec"], rec=r["test"]["rec"], f1=r["test"]["f1"], thr=r["thr"]) for n, r in R.items()]
    rows.append(dict(m=M["best"] + " + sigmoid calibration (SELECTED)", k="calibrated probability", cv=np.nan, roc=t["roc"], pr=t["pr"], prec=t["prec"], rec=t["rec"], f1=t["f1"], thr=thr)); rows.append(dict(m="Baseline: department average attrition", k="simple rate", cv=np.nan, roc=M["base_auc"], pr=np.nan, prec=np.nan, rec=np.nan, f1=np.nan, thr=np.nan))
    h = note("<b>Design.</b> " + esc(M["design"]) + " Employees never appear in both train and test. Models and the threshold were chosen with cross-validation on the training set only; every number below is from the %d held-out employees (%.1f%% left). The output is a decision-support signal, not a judgement about any individual's intentions, and must never be the sole basis for an employment decision." % (M["n_test"], 100 * yte.mean()), "ai", "pred")
    h += tbl(pd.DataFrame(rows), [("m", "Model"), ("k", "Score type"), ("cv", "Train CV AUC"), ("roc", "Test ROC-AUC"), ("pr", "Test PR-AUC"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1"), ("thr", "Threshold")], dict(cv=lambda v: fx(v, 3), roc=lambda v: fx(v, 3), pr=lambda v: fx(v, 3), prec=lambda v: fx(v, 3), rec=lambda v: fx(v, 3), f1=lambda v: fx(v, 3), thr=fx))
    gain = t["roc"] - M["base_auc"]; h += note("Test ROC-AUC 95%% bootstrap interval %s-%s. Improvement over the department baseline: <b>%+.3f</b> AUC%s. PR-AUC base rate (random) is %s. Threshold %.2f chosen by '%s' on training data: the model flags %d of %d test employees, catching %d of %d leavers (recall %.0f%%, precision %.0f%%). A lower threshold catches more leavers but flags more employees who would have stayed." %
              (fx(M["ci"][0], 3), fx(M["ci"][1], 3), gain, "" if gain > .02 else " (little or no improvement: do not rely on individual scores)", fx(M["base_rate"], 3), thr, esc(G.get("obj", "")), t["tp"] + t["fp"], len(yte), t["tp"], t["tp"] + t["fn"], t["rec"] * 100, t["prec"] * 100), "info", "pred")
    h += note("Metrics come from " + ("<b>synthetic data</b> and say nothing about real-world accuracy." if G["demo"] else "your data and can change with new data.") + " Calibration (below) checks whether predicted probabilities match observed rates. Importance and explanations show model behaviour, <b>not</b> what would happen if a factor were changed.", "warn")
    f1 = go.Figure()
    for n, r in R.items(): a, b, _ = roc_curve(yte, r["p"]); f1.add_trace(go.Scatter(x=a, y=b, name="%s (%.2f)" % (n, r["test"]["roc"])))
    a, b, _ = roc_curve(yte, pc); f1.add_trace(go.Scatter(x=a, y=b, name="Selected calibrated (%.2f)" % t["roc"], line=dict(width=4, color=C["mi"]))); f1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash", color=C["mu"]))); f1 = sty(f1, 360, "ROC curves (held-out)")
    pr_, rc_, _ = precision_recall_curve(yte, pc); f2 = go.Figure(go.Scatter(x=rc_, y=pr_, line=dict(color=C["mi"], width=3), name="Selected")); f2.add_hline(y=float(yte.mean()), line_dash="dash", line_color=C["mu"], annotation_text="base rate"); f2 = sty(f2, 360, "Precision-recall (held-out)"); f2.update_xaxes(title="Recall"); f2.update_yaxes(title="Precision")
    f3 = go.Figure(go.Heatmap(z=[[t["tn"], t["fp"]], [t["fn"], t["tp"]]], x=["Predicted stay", "Predicted exit"], y=["Actually stayed", "Actually left"], colorscale=[[0, C["sf"]], [1, C["mi"]]], text=[[t["tn"], t["fp"]], [t["fn"], t["tp"]]], texttemplate="%{text}", showscale=False)); f3 = sty(f3, 320, "Confusion matrix at threshold %.2f" % thr); f3.update_yaxes(autorange="reversed")
    cb = M["calib"]; f4 = go.Figure([go.Scatter(x=cb["pred"], y=cb["obs"], mode="lines+markers", name="Calibrated", line=dict(color=C["mi"])), go.Scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash", color=C["mu"]))]); f4 = sty(f4, 320, "Calibration (held-out, quantile bins)"); f4.update_xaxes(title="Mean predicted"); f4.update_yaxes(title="Observed rate")
    g = np.linspace(.05, .9, 35); P = pc[:, None] >= g[None, :]; tp = (P & (yte[:, None] == 1)).sum(0); fp = (P & (yte[:, None] == 0)).sum(0); prc = tp / np.maximum(tp + fp, 1); rcl = tp / max(int(yte.sum()), 1)
    f5 = go.Figure([go.Scatter(x=g, y=prc, name="Precision"), go.Scatter(x=g, y=rcl, name="Recall"), go.Scatter(x=g, y=2 * prc * rcl / np.maximum(prc + rcl, 1e-9), name="F1")]); f5.add_vline(x=thr, line_dash="dash", line_color=C["am"], annotation_text="chosen"); f5 = sty(f5, 320, "Threshold trade-offs (held-out)")
    imp = M["imp"]; f6 = empty("Importance unavailable")
    if imp is not None: d = imp.head(12).iloc[::-1]; f6 = go.Figure(go.Bar(y=[pretty(x) for x in d["f"]], x=d["imp"], orientation="h", error_x=dict(type="data", array=d["sd"].values), marker_color=C["la"])); f6 = sty(f6, 380, "Permutation importance on held-out data: drop in AUC (not causal)")
    E = G["E"]; Xte = M["Xte"]; gp = E.reindex(Xte.index); rows = []
    for c in ["department", "job_level"] + G["B"]["sens"]:
        if c not in gp.columns or gp[c].notna().sum() < 50: continue
        for lv in sorted(gp[c].dropna().unique(), key=str):
            m = (gp[c] == lv).values; n = int(m.sum()); yy = yte[m]; ok = n >= 40 and yy.sum() >= 8 and (n - yy.sum()) >= 8; fr = (pc[m] >= thr).mean() if n else np.nan
            rows.append(dict(grp=pretty(c) + ": " + str(lv), n=n, rate=yy.mean() if n else np.nan, auc=roc_auc_score(yy, pc[m]) if ok else np.nan, flag=fr, ratio=fr / ((pc >= thr).mean()) if (pc >= thr).mean() > 0 else np.nan, small="small subgroup" if n < 40 else ""))
    if rows:
        S = pd.DataFrame(rows); S["rv"] = np.where((S["ratio"] > 1.25) | (S["ratio"] < .8), np.where(S["n"] >= 30, "review", ""), ""); h += H("Subgroup performance and flag rates (held-out)") + tbl(S, [("grp", "Group"), ("n", "Test n"), ("rate", "Actual exit rate"), ("auc", "Test AUC"), ("flag", "Flagged at threshold"), ("ratio", "Flag rate vs overall"), ("rv", "Human review"), ("small", "")], dict(n=nf, rate=pc1, auc=fx, flag=pc1, ratio=lambda v: fx(v) + "x"))
        h += note("Subgroup AUC is shown only where n ≥ 40 with at least 8 of each outcome. This is a screening check, not a legal fairness test.", "info")
    now = G.get("now")
    if now is not None: c = now["band"].value_counts().reindex(["Elevated", "Moderate", "Lower"]).fillna(0); h += H("Current employees by modelled risk band (aggregate)") + tbl(pd.DataFrame(dict(b=c.index, n=c.values, s=c.values / c.sum())), [("b", "Band"), ("n", "Employees"), ("s", "Share")], dict(n=nf, s=pc1)) + note("Elevated = calibrated probability ≥ threshold; Moderate = at least the base rate. Bands describe groups for planning, not individuals' intentions.", "info", "pred")
    return h, f1, f2, f3, f4, f5, f6
def restricted(ok, eid):
    if not ok: return note("Employee-level view is restricted. Tick the authorisation box (appropriate role, lawful purpose) to continue. Aggregated analysis above needs no authorisation.", "warn"), empty("Not authorised")
    if G.get("now") is None or not eid or eid not in G["now"].index: return note("Select a pseudonymous employee ID.", "info"), empty("No employee selected")
    M = G["M"]; x = G["Xn"].loc[[eid]]; Cm, base = contributions(M["final"], x, M["ref"]); cm = pd.Series(Cm[0], index=x.columns); top = cm.reindex(cm.abs().sort_values(ascending=False).index)[:8].iloc[::-1]; p = float(base[0]); band = G["now"].loc[eid, "band"]
    fig = sty(go.Figure(go.Bar(y=[pretty(f) for f in top.index], x=top.values * 100, orientation="h", marker_color=[C["co"] if v > 0 else C["mi"] for v in top.values])), 360, "Model drivers (what-if vs typical value, percentage points)")
    pos = [pretty(f).lower() for f in cm[cm >= .01].sort_values(ascending=False).index[:3]]
    return note("Group-planning signal for <b>%s</b>: modelled probability %s (band %s). Factors raising the score relative to a typical employee: %s. <br><b>Use:</b> human review only; never as the sole basis for any adverse decision; this does not describe the person's intentions or circumstances; employees may contest inaccurate records through HR." % (esc(eid), pc1(p), band, esc(", ".join(pos) or "none stand out")), "ai", "pred"), fig
def render_playbook():
    if "PB" not in G: return note("Load data first.", "info"), pd.DataFrame()
    P = G["PB"]; h = note("Each recommendation maps observed organisational evidence to a hypothesis and an evaluation plan. 'Generic best practice' means no pattern in this data supports it. Interventions target working conditions and systems, not individual employees. Cost cells are for HR to estimate; nothing here is a measured saving.", "ai")
    for _, r in P.iterrows(): h += "<div class='tw'><table><tr><th colspan='2'>%d. %s %s</th></tr>" % (r["#"], esc(r["name"]), tag("obs" if r["level"].startswith("Strong") else "hyp")) + "".join("<tr><td style='width:200px'>%s</td><td>%s</td></tr>" % (lab, esc(r[k]) if k != "ev" else r[k]) for k, lab in PBC[1:]) + "</table></div>"
    return h, P[[k for k, _ in PBC]].rename(columns=dict(PBC))
def render_rai():
    if "E" not in G: return note("Load data first.", "info")
    B, M, E = G["B"], G["M"], G["E"]; h = note("<b>Responsible use.</b> Attrition predictions must not be the sole basis for dismissal, promotion or pay denial, discipline or other adverse decisions. The platform does not infer sensitive circumstances or intentions. Employee-level views are restricted and pseudonymous. External LLMs receive only aggregated tool outputs. Model explanations and importance do not establish causation. Employees must have a documented route to contest inaccurate records, and every action based on a score needs human review.", "warn")
    h += note("Sensitive attributes (gender, age band) are <b>never model features</b>; they are used only to audit performance and flag rates. Other sensitive-looking columns are ignored. Data used here: " + ("<b>synthetic</b>" if G["demo"] else "uploaded (IDs %s)" % ("pseudonymised" if G["pseud"] else "as supplied")) + ".", "info")
    if B["sens"]:
        rows = []
        for c in B["sens"]:
            for f in [x for x in (G["M"].get("nums", []) if M["ok"] else G["nums"]) if x in E.columns]:
                e = eta2(E[f], E[c]); rows.append(dict(c=pretty(c), f=pretty(f), e=e, flag="possible proxy" if pd.notna(e) and e > .05 else ""))
        S = pd.DataFrame(rows).sort_values("e", ascending=False).head(8); h += H("Proxy check: how strongly model inputs differ by group (share of variance explained)") + tbl(S, [("c", "Group"), ("f", "Model input"), ("e", "Variance explained"), ("flag", "Flag")], dict(e=pc1)) + note("Values above 5% would suggest an input may act as a proxy for the group. Small values here do not prove the absence of disparate impact.", "info")
        t = rate_table(E, B["sens"][0]); h += H("Observed attrition by %s (descriptive audit only)" % pretty(B["sens"][0]).lower()) + tbl(t, [(B["sens"][0], pretty(B["sens"][0])), ("n", "Employees"), ("exits", "Exits"), ("rate", "Rate"), ("lo", "CI low"), ("hi", "CI high"), ("flag", "Flag")], dict(n=nf, exits=nf, rate=pc1, lo=pc1, hi=pc1)) + note("Group differences in observed attrition are not intrinsic characteristics of employees; they can reflect job mix, manager, pay or workplace conditions.", "warn")
    h += H("Limitations") + "<ul class='ins'><li>Association is not causation; omitted variables (for example manager quality, market conditions) may drive both factors and exits.</li><li>The model predicts a historical pattern; it may not transfer to new conditions.</li><li>Missing engagement or overtime values are never treated as zero or as low engagement.</li><li>Replacement costs: %s.</li><li>Not implemented: drift monitoring over time, causal estimation, full fairness audits.</li></ul>" % esc(G["csrc"])
    return h
def render_dq():
    if "E" not in G: return note("Load data first.", "info"), ""
    B, d = G["B"], G["d"]; dq = B["dq"]; h = note("Rows read: <b>%s</b>; employees: <b>%s</b>; mode: <b>%s</b>. " % (nf(B["raw_n"]), nf(len(G["E"])), "longitudinal (employee-month records)" if B["panel"] else "snapshot (one record per employee)") + ("<b>Illustrative synthetic data.</b>" if G["demo"] else "Uploaded data; never written into the project ZIP."), "ok")
    h += "".join(note(esc(w), "warn") for w in G["W"]); cov = [(k, d[k].notna().mean() if k in d.columns else 0.0) for k in SYN]
    h += H("Field availability (share of rows with a value)") + tbl(pd.DataFrame(cov, columns=["k", "c"]), [("k", "Field"), ("c", "Coverage")], dict(c=pc0))
    un = [n for n, ks in (("Overtime and workload analysis", ["overtime_hours"]), ("Compensation analysis", ["monthly_salary"]), ("Engagement analysis", ["engagement_score"]), ("Training and career analysis", ["training_hours", "months_since_last_promotion"]), ("Replacement-cost analysis", ["monthly_salary"]), ("Trends and pre-exit change", ["observation_date"])) if not all(d[k].notna().any() for k in ks if k in d.columns) or not all(k in d.columns for k in ks)]
    h += note("Unavailable with the current data: " + (", ".join(un) if un else "none: all supported analyses can run."), "info") + note("Duplicates removed: %s exact, %s repeated employee-date. Observations after exit removed: %s. Exit dates derived: %s." % (nf(dq.get("dup_exact")), nf(dq.get("dup_key")), nf(dq.get("post_exit")), nf(dq.get("derived_exit"))), "info")
    return h, H("Self-test results (run at start-up)") + "".join(note(("PASS " if ok else "FAIL ") + esc(g + ": " + n) + " — " + esc(x), "ok" if ok else "err") for g, n, ok, x in TESTS)

# ------------------------------ strategist (tools) ------------------------------
def dtxt(f):
    r = G["drv"][G["drv"]["f"] == f]
    if not len(r): return "%s: not enough data for an analysis." % pretty(f)
    r = r.iloc[0]; return "%s: attrition %s in the lowest third vs %s in the highest third (%s; %s exits; adj. p %s; evidence %s)" % (pretty(f), pc1(r["rate_lo"]), pc1(r["rate_hi"]), pp(r["pp"]), nf(r["n_ex"]), fx(r["p_adj"], 3), r["strength"])
def t_summary():
    E = G["E"]; ex = int(E["left"].sum()); return "%s employees in view (each counted once); %s exits; historical attrition %s; replacement cost of exits %s (%s)." % (nf(len(E)), nf(ex), pc1(ex / len(E)) if G["B"]["has_out"] else "unavailable", money(G["cost"][E["left"]].sum(min_count=1)), G["csrc"])
def t_patterns():
    d = G["drv"][G["drv"]["strength"] != "Weak"].head(3); return "Top associations: " + ("; ".join(dtxt(f) for f in d["f"]) if len(d) else "none reach the evidence threshold") + ". These are associations, not causes."
def t_dept():
    t = G["tab"]
    if t is None: return "Department data is not available."
    return "Departments by attrition: " + "; ".join("%s %s (n=%s, %s, replacement cost %s)" % (r["department"], pc1(r["rate"]), nf(r["n"]), r["status"] or "typical", money(r["repl"])) for _, r in t.head(4).iterrows())
def t_model():
    M = G["M"]
    if not M["ok"]: return "Prediction is unavailable: " + M["why"]
    return "Held-out ROC-AUC %s vs department baseline %s; PR-AUC %s; recall %s at precision %s (threshold %.2f). Top importances: %s. Importance is not causal." % (fx(M["test"]["roc"], 3), fx(M["base_auc"], 3), fx(M["test"]["pr"], 3), pc0(M["test"]["rec"]), pc0(M["test"]["prec"]), M["thr"], ", ".join(pretty(x) for x in M["imp"]["f"].head(4)) if M["imp"] is not None else "n/a")
def t_interv(): P = G["PB"]; return "Suggested interventions: " + "; ".join("%s [%s]: %s" % (r["name"], r["level"], r["ev"]) for _, r in P.iterrows())
def t_scn():
    n, b, m, rc, ob, vc, tab = roi_defaults()
    try: d, R = roi_run(n, b, m, rc, ob, vc, .5, 1.5, tab); B = R[R["sc"] == "Base"].iloc[0]; return "Illustrative base scenario for '%s' (coverage %s%%, assumed %s%% relative reduction, placeholder cost per employee 150): exits avoided %s, avoided cost %s, programme cost %s, net %s. These are assumptions HR must validate, not measured savings." % (B["name"], "50", "10", fx(B["avoided"], 1), money(B["gross"]), money(B["prog"]), money(B["net"]))
    except Exception as e: return "Scenario unavailable: " + str(e)
def t_eval(): return "To test whether a programme works: assign teams or employees to treatment and comparison groups (randomly or staggered), define exits and the observation window beforehand, track leading indicators (for example overtime, engagement) and guardrails, and compare exit rates at 6 and 12 months with confidence intervals. Observational patterns alone cannot show that a programme caused the change."
def t_limits(): return "Limitations: associations are not causes; missing values are not treated as zero; %s; replacement costs: %s; model scores are decision-support only. Data notes: %s" % ("longitudinal data" if G["B"]["panel"] else "snapshot data limits trend analysis", G["csrc"], " | ".join(G["W"][:3]))
def get_key(key, provider, env=True):
    k = (key or "").strip()
    if k or not env: return k or None
    nm = "ANTHROPIC_API_KEY" if provider == "Anthropic" else "OPENAI_API_KEY"; k = os.environ.get(nm)
    if not k:
        try:
            from google.colab import userdata; k = userdata.get(nm)
        except Exception: k = None
    return k
def llm(system, user, key, model, provider, base):
    k = get_key(key, provider)
    if provider == "Anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": k, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json={"model": (model or "claude-sonnet-5-5").strip(), "max_tokens": 1200, "system": system, "messages": [{"role": "user", "content": user}]}, timeout=90); r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))
    r = requests.post((base or "https://api.openai.com/v1").rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + k, "content-type": "application/json"}, json={"model": (model or "gpt-4o-mini").strip(), "max_tokens": 1200, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}, timeout=90); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
def check_numbers(ans, ctx):
    cn = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ctx.replace(",", ""))]; bad = []
    for s in re.findall(r"\d+(?:\.\d+)?", ans.replace(",", "")):
        v = float(s)
        if v <= 10 or 1990 <= v <= 2100: continue
        if not any(abs(v - c) <= max(.06, .015 * abs(c)) for c in cn): bad.append(s)
    return sorted(set(bad))
def ask(q, provider="Anthropic", model="", base="", key="", use_llm=True):
    if "E" not in G: return "Load data first."
    q = (q or "").strip() or "Which organisational patterns are most concerning?"; ql = q.lower(); parts = []
    if re.search(r"concern|pattern|summary|overview|workforce", ql): parts += [("workforce_summary", t_summary()), ("pattern_analysis", t_patterns())]
    if re.search(r"overtime|workload", ql): parts.append(("workload_analysis", dtxt("overtime_hours") + "; " + dtxt("workload_hours")))
    if re.search(r"department|role|turnover|cost|replacement", ql): parts.append(("department_comparison", t_dept()))
    if re.search(r"compens|salary|pay", ql): parts.append(("compensation_analysis", dtxt("compa_ratio") + "; " + dtxt("salary_change_percentage")))
    if re.search(r"engage|absen|satisf", ql): parts.append(("engagement_absence_analysis", dtxt("engagement_score") + "; " + dtxt("absenteeism_days")))
    if re.search(r"train|career|promot|progress", ql): parts.append(("training_career_analysis", dtxt("training_hours") + "; " + dtxt("months_since_last_promotion")))
    if re.search(r"model|predict|explain|importance", ql): parts.append(("model_explanation", t_model()))
    if re.search(r"intervention|recommend|address|evidence", ql): parts.append(("intervention_playbook", t_interv()))
    if re.search(r"evaluate|measure|work\b|experiment|programme|program ", ql): parts.append(("evaluation_guidance", t_eval()))
    if re.search(r"scenario|roi|saving|budget|cost-benefit", ql): parts.append(("scenario_calculation", t_scn()))
    if re.search(r"limit|gap|caveat|bias", ql): parts.append(("limitations", t_limits()))
    if "report" in ql: parts.append(("report", "Use 'Generate executive summary' in the Data Quality and Exports tab."))
    if not parts: parts = [("workforce_summary", t_summary()), ("pattern_analysis", t_patterns()), ("department_comparison", t_dept())]
    ctx = "\n".join("[%s] %s" % (n, t) for n, t in parts); ba = "### Answer (deterministic, computed by analytical tools)\n" + "\n".join("- **%s**: %s" % (n, t) for n, t in parts) + "\n\n*Tools used: " + ", ".join(n for n, _ in parts) + ". Every figure comes from the current dataset; associations are not causes.*"
    if not use_llm: return ba
    if not get_key(key, provider): return ba + "\n\n*No API key set: the transparent rule-based strategist answered. A key (session only) enables natural-language synthesis; only these aggregated tool results would be sent, never employee-level data.*"
    try:
        ans = llm("You are an HR analytics assistant. Answer ONLY from the TOOL RESULTS. Never invent numbers, causes, savings or employee circumstances. Label hypotheses as hypotheses. Be concise.", "TOOL RESULTS:\n" + ctx + "\n\nQUESTION: " + q, key, model, provider, base); bad = check_numbers(ans, ctx)
        return "### LLM-written answer (grounded in tool results; aggregated data only)\n" + ans + (("\n\nNumbers not found in tool results: " + ", ".join(bad) + ". Treat as unverified.") if bad else "\n\nNumeric check passed: all figures match tool results.") + "\n\n---\n" + ba
    except Exception as e: return ba + "\n\n*LLM call failed (%s); the deterministic answer is shown.*" % type(e).__name__

# ------------------------------ exports and report ------------------------------
def playbook_md(P):
    out = ["# Retention intervention playbook (PULSE AI)", "", "_Hypotheses and associations, not validated causal effects._", ""]
    for _, r in pd.DataFrame(P).iterrows(): out += ["## %s" % r.get("Intervention", r.get("name", ""))] + ["- **%s:** %s" % (c, r[c]) for c in P.columns if c not in ("Intervention", "name", "#")] + [""]
    return "\n".join(out)
def export_all(pb=None, scn=None):
    if "E" not in G: return None, note("Load data first.", "info")
    out = []; G["drv"].to_csv("pulse_driver_analysis.csv", index=False); out.append("pulse_driver_analysis.csv")
    for c, fn in (("department", "pulse_department_summary.csv"), ("job_role", "pulse_role_summary.csv")):
        if G["E"][c].notna().any(): group_table(G["E"], c, G["cost"]).to_csv(fn, index=False); out.append(fn)
    M = G["M"]
    if M["ok"]:
        pd.DataFrame([dict(model=n, test_roc_auc=r["test"]["roc"], test_pr_auc=r["test"]["pr"], precision=r["test"]["prec"], recall=r["test"]["rec"], f1=r["test"]["f1"], threshold=r["thr"]) for n, r in M["R"].items()] + [dict(model="selected calibrated " + M["best"], test_roc_auc=M["test"]["roc"], test_pr_auc=M["test"]["pr"], precision=M["test"]["prec"], recall=M["test"]["rec"], f1=M["test"]["f1"], threshold=M["thr"]), dict(model="department baseline", test_roc_auc=M["base_auc"])]).to_csv("pulse_model_metrics.csv", index=False); out.append("pulse_model_metrics.csv")
        if M["imp"] is not None: M["imp"].to_csv("pulse_feature_importance.csv", index=False); out.append("pulse_feature_importance.csv")
    P = pd.DataFrame(pb) if pb is not None and len(pd.DataFrame(pb)) else G["PB"][[k for k, _ in PBC]].rename(columns=dict(PBC)); P.to_csv("pulse_playbook.csv", index=False); open("pulse_playbook.md", "w", encoding="utf-8").write(playbook_md(P)); out += ["pulse_playbook.csv", "pulse_playbook.md"]
    try: n, b, m, rc, ob, vc, tab = roi_defaults(); d, R = roi_run(n, b, m, rc, ob, vc, .5, 1.5, scn if scn is not None and len(pd.DataFrame(scn)) else tab); R.to_csv("pulse_scenarios.csv", index=False); out.append("pulse_scenarios.csv")
    except Exception: pass
    open("pulse_dq_responsible_ai.md", "w", encoding="utf-8").write("# Data quality and responsible AI summary\n\n" + "\n".join("- " + w for w in G["W"]) + "\n\n" + RAI_MD + "\n\nExports are aggregated and contain no employee-level records or identifiers.\n"); out.append("pulse_dq_responsible_ai.md")
    return out, note("Exported %d aggregated files (no employee-level data)." % len(out), "ok")
def make_report():
    if "E" not in G: return None, note("Load data first.", "info")
    M = G["M"]; h = lambda t: "<h2>" + t + "</h2>"; p = ["<h1>PULSE AI — Executive Summary</h1><p class='m'>Understand the patterns. Improve the workplace.</p>" + ("<p class='w'>Illustrative synthetic data: fictional employees; not real results.</p>" if G["demo"] else "")]
    p.append(h("Summary (observed)") + "<p>%s</p><p>%s</p><p>%s</p>" % (esc(t_summary()), esc(t_patterns()), esc(t_dept()))); p.append(h("Data coverage and quality") + "<ul>" + "".join("<li>%s</li>" % esc(w) for w in G["W"]) + "</ul>")
    p.append(h("Model (prediction, decision support only)") + "<p>%s</p>" % esc(t_model())); p.append(h("Recommended interventions (hypotheses)") + tbl(G["PB"], [("name", "Intervention"), ("level", "Evidence level"), ("cohort", "Target cohort"), ("kpi", "Outcome KPI"), ("evald", "Evaluation design")]))
    p.append(h("Scenario (illustrative)") + "<p>%s</p>" % esc(t_scn())); p.append(h("Responsible AI") + "<p>%s</p>" % esc(RAI_MD))
    css = "body{font-family:Inter,Arial,sans-serif;background:#151C27;color:#F2F5F7;max-width:1000px;margin:30px auto;padding:0 18px}h1{color:#7BE0C3}h2{color:#D3B5FF;border-left:3px solid #7BE0C3;padding-left:10px;margin-top:28px}.m{color:#A7B5C5}.w{background:#F1BE7822;border-left:3px solid #F1BE78;padding:8px 12px}table{border-collapse:collapse;width:100%;background:#242F3B;margin:8px 0}th{background:#2d3a49;color:#7BE0C3;text-align:left;padding:8px}td{padding:7px 8px;border-top:1px solid #ffffff14}.tw{overflow-x:auto}"
    path = "pulse_executive_summary.html"; open(path, "w", encoding="utf-8").write("<html><head><meta charset='utf-8'><title>PULSE AI</title><style>" + css + "</style></head><body>" + "".join(p) + "</body></html>"); return path, note("Executive summary generated (aggregated figures only).", "ok")

# ------------------------------ self-tests ------------------------------
def _mini():
    r = []
    for e in "ABCD":
        for m in range(4):
            dt = "2026-%02d-01" % (m + 1); ex = None; at = 0; stt = "Active"
            if e == "B" and m == 0: ex = "2026-01-10"; at = 1; stt = "Left"
            if e == "C" and m == 3: at = 1; stt = "Left"
            r.append(dict(employee_id=e, observation_date=dt, department="X" if e in "AB" else "Y", attrition=at, exit_date=ex, employment_status=stt, engagement_score=60.0 + m))
    return pd.DataFrame(r)
def _small():
    if "s" not in _SM: _SM["s"] = core(make_synth(1200, seed=3), "", 6, 1.0, .4)
    return _SM["s"]
def run_selftests(verbose=True):
    if _TS: return _TS
    T = []; saved = dict(G)
    def chk(g, n, fn):
        try: ok, det = fn(); T.append((g, n, bool(ok), det))
        except Exception as e: T.append((g, n, False, type(e).__name__ + ": " + str(e)[:90]))
    def t_cols():
        try: ingest(pd.DataFrame({"x": [1]})); return False, "no error"
        except ValueError as e: return "employee_id" in str(e), "clear error for a missing mandatory field"
    def t_dup(): m = _mini(); m = pd.concat([m, m.iloc[[0]], m.iloc[[1]].assign(engagement_score=99.0)]); B = ingest(m); return B["dq"]["dup_exact"] == 1 and B["dq"]["dup_key"] == 1, "exact duplicates %d, repeated employee-date %d" % (B["dq"]["dup_exact"], B["dq"]["dup_key"])
    def t_inv():
        m = _mini(); m["monthly_salary"] = 500000.0; m.loc[0, "engagement_score"] = -5; B = ingest(m); return pd.isna(B["d"]["engagement_score"]).sum() >= 1 and any("annual" in w for w in B["W"]), "negative value set to missing; salary unit warning raised"
    def t_head(): B = ingest(_mini()); E = B["E"]; return len(E) == 4 and int(E["left"].sum()) == 2 and abs(E["left"].mean() - .5) < 1e-9 and B["panel"], "4 employees counted once from 16 monthly rows; attrition 50%"
    def t_exit(): B = ingest(_mini()); return B["dq"]["post_exit"] == 2 and B["dq"]["derived_exit"] == 1 and pd.Timestamp("2026-04-01") == B["E"].loc["C", "exit_date"], "2 post-exit rows removed; 1 missing exit date derived"
    def t_win(): E = ingest(_mini())["E"]; X = filt(E, d1="2026-03-01"); return len(X) == 3 and int(X["lf"].sum()) == 1 and "B" not in X.index, "leaver before the window excluded; window rate 1 of 3"
    def t_miss():
        B = ingest(_mini().drop(columns=["engagement_score"])); E = B["E"]; t = group_table(E.assign(lf=E["left"]), "department", resolve_cost(E)[0]); return E["engagement_score"].isna().all() and t["eng"].isna().all(), "missing engagement stays missing (not zero)"
    def t_rate(): X = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50, "lf": [True] * 10 + [False] * 40 + [True] * 25 + [False] * 25}); t = rate_table(X, "g").set_index("g"); return abs(t.loc["a", "rate"] - .2) < 1e-9 and abs(t.loc["b", "rate"] - .5) < 1e-9 and abs(t.loc["a", "pp"] - (-.15)) < 1e-9, "rates 20% and 50%; difference vs overall -15 pp"
    def t_drv():
        r = np.random.default_rng(0); lf = r.random(600) < .25; X = pd.DataFrame({"lf": lf, "f": r.normal(0, 1, 600) + 1.2 * lf}); d = drivers(X, ["f"]); X2 = X.copy(); X2["lf"] = False; X2.iloc[:10, 0] = True
        return len(d) == 1 and d.iloc[0]["auc"] > .6 and d.iloc[0]["pp"] > 0 and len(drivers(X2, ["f"])) == 0, "association detected in the right direction; too few exits skipped"
    def t_leak(): R = _small(); M = R["M"]; bad = {"exit_date", "attrition", "employee_id", "gender", "age_band", "employment_status", "estimated_replacement_cost", "recruitment_cost", "manager_id"}; return M["ok"] and not (set(M["feats"]) & bad) and set(M["tr_ids"]).isdisjoint(set(M["te_ids"])), "%d features, no outcome, identifier, demographic or cost fields; train and test employees disjoint" % len(M["feats"])
    def t_small(): X = pd.DataFrame({"a": np.arange(100.0)}); M = train_model(X, np.array([0] * 90 + [1] * 10), ["a"], [], 1.0); return not M["ok"] and "needs" in M["why"], "small / imbalanced data disables prediction with an explanation"
    def t_thr():
        r = np.random.default_rng(1); y = np.r_[np.zeros(400), np.ones(100)].astype(int); p = np.clip(np.r_[r.normal(.3, .15, 400), r.normal(.55, .2, 100)], 0, 1); out = []
        for b2 in (1.0, 4.0): t = pick_threshold(y, p, b2); out.append(fbeta(y, p, t, b2) >= fbeta(y, p, .5, b2) - 1e-12 and .02 <= t <= .95)
        return all(out), "threshold maximises the chosen F-score and beats a fixed 0.5"
    def t_e2e(): R = _small(); M = R["M"]; return M["ok"] and M["test"]["roc"] > .5 and abs(M["n_test"] / (M["n_test"] + M["n_train"]) - .25) < .01, "held-out ROC-AUC %.3f (baseline %.3f) on %d held-out employees" % (M["test"]["roc"], M["base_auc"], M["n_test"])
    def t_expl(): R = _small(); M = R["M"]; Cm, b = contributions(M["final"], R["Xn"].iloc[:15], M["ref"]); return Cm.shape == (15, len(M["feats"])) and np.isfinite(Cm).all() and ((b >= 0) & (b <= 1)).all(), "explanation matrix finite with valid probabilities"
    def t_int():
        R = _small(); X = R["E"].copy(); X["overtime_hours"] = np.nan; X["workload_hours"] = np.nan; X["absenteeism_days"] = np.nan; drv = drivers(X, avail_nums(X)); P = build_playbook(X, drv, None); bad = P["name"].str.contains("overtime|absenteeism", case=False).any(); return not bad and P["ev"].str.len().min() > 10, "no workload or absence recommendation without that evidence; every row carries evidence text"
    def t_roi():
        r = roi_calc(1000, .5, .2, 12, .1, 100, 10000, 0, 0); return abs(r["avoided"] - 10) < 1e-9 and abs(r["gross"] - 100000) < 1e-6 and abs(r["prog"] - 50000) < 1e-6 and abs(r["net"] - 50000) < 1e-6 and abs(r["roi"] - 1) < 1e-9 and abs(r["be"] - .05) < 1e-9, "exits avoided 10, net 50,000, ROI 100%, break-even reduction 5%"
    def t_zero(): r = roi_calc(1000, .5, .2, 12, .1, 0, 10000, 0, 0); return pd.isna(r["roi"]) and r["be"] == 0, "zero programme cost: ROI undefined, break-even 0"
    def t_cost(): E = pd.DataFrame({"estimated_replacement_cost": [np.nan], "recruitment_cost": [np.nan], "monthly_salary": [1000.0]}); c, s = resolve_cost(E, .5); return abs(c.iloc[0] - 6000) < 1e-9 and "ASSUMPTION" in s, "assumed cost 6,000 and labelled as an assumption"
    def t_exp():
        R = _small(); G.clear(); G.update(R); G.update(demo=True, pseud=False, obj="Balanced (F1)"); files, _ = export_all(); bad = [f for f in files if f.endswith(".csv") and "employee_id" in pd.read_csv(f).columns]; G.clear(); G.update(saved); return len(files) >= 5 and not bad, "%d aggregated exports, none with employee identifiers" % len(files)
    def t_llm():
        R = _small(); G.clear(); G.update(R); G.update(demo=True, pseud=False, obj="Balanced (F1)"); a = ask("what are the limitations?", use_llm=False); k = get_key("", "Anthropic", env=False); G.clear(); G.update(saved); return k is None and "deterministic" in a, "no key: deterministic tool-grounded answer returned"
    for g, n, f in [("data", "missing mandatory field", t_cols), ("data", "duplicate records", t_dup), ("data", "invalid values and units", t_inv), ("data", "privacy-safe exports", t_exp), ("workforce", "employees counted once", t_head), ("workforce", "exit dates and rows after exit", t_exit), ("workforce", "observation window", t_win),
                    ("workforce", "missing values stay missing", t_miss), ("drivers", "attrition rate and differences", t_rate), ("drivers", "driver direction and minimum size", t_drv), ("model", "leakage prevention and disjoint split", t_leak), ("model", "small or imbalanced data", t_small), ("model", "threshold selection", t_thr),
                    ("model", "held-out evaluation", t_e2e), ("model", "explanation validity", t_expl), ("interventions", "recommendations grounded in evidence", t_int), ("interventions", "LLM fallback", t_llm), ("roi", "scenario arithmetic", t_roi), ("roi", "zero-cost edge case", t_zero), ("roi", "replacement-cost calculation", t_cost)]: chk(g, n, f)
    G.clear(); G.update(saved); _TS[:] = T
    if verbose:
        for g, n, ok, x in T: print(("PASS " if ok else "FAIL ") + g + ": " + n + " :: " + x)
    return T

# ------------------------------ project export ------------------------------
RAI_MD = "Responsible use: predictions are decision-support signals, never the sole basis for adverse employment decisions; no inference of sensitive circumstances or intentions; gender and age are audit-only; restricted pseudonymous employee view; aggregated data only for external LLMs; human review and a route to contest records are required; explanations do not show causation."
METH = """# Methodology
- **Unit of analysis:** employees. Longitudinal files (employee-month rows) are collapsed to one record per employee, so headcount-based rates never count an employee twice. Attrition rate = exits / unique employees in view; filters by department, role, tenure band and exit-date window apply.
- **Cleaning:** exact duplicates removed; repeated employee-date rows keep the last; observations more than 31 days after an exit are dropped; missing exit dates are derived from the first 'left' observation; negative values become missing; missing values are never treated as zero.
- **Drivers:** per factor, attrition in the lowest vs highest third with Wilson intervals, Mann-Whitney AUC effect size, Bonferroni-adjusted p-values, minimum sample sizes. Ranking is by effect size. Associations only.
- **Pre-exit change (longitudinal):** change over the last 3 observations vs earlier ones, leavers vs stayers.
- **Model:** features measured at a snapshot date (last 3 months averaged), outcome = exit in the following horizon; employees split 75/25 (stratified); Logistic Regression, Random Forest, Gradient Boosting compared by cross-validated AUC on training data; best model sigmoid-calibrated; threshold chosen on training cross-validation by F1, F2 or F0.5; baseline = department average attrition. Permutation importance and what-if explanations are model behaviour, not causal.
- **Interventions:** rule-based mapping from evidence (strong/tentative associations) to hypotheses, KPIs, guardrails and evaluation designs; generic practices are labelled generic.
- **ROI:** exits avoided = employees x coverage x baseline x months/12 x assumed reduction; avoided cost = exits avoided x cost per exit; net = avoided cost - programme cost; ROI = net / cost if cost > 0; break-even reduction = cost / (employees x coverage x baseline x months/12 x cost per exit).
"""
DICT = "# Data dictionary\nemployee_id; observation_date; hire_date; exit_date; attrition (0/1); employment_status; department; job_role; job_level; manager_id; performance_rating; workload_hours; overtime_hours; overtime_flag; absenteeism_days; leave_days; monthly_salary; salary_band; salary_change_percentage; engagement_score; job_satisfaction; manager_satisfaction; work_life_balance; training_hours; training_sessions; promotion_count; months_since_last_promotion; tenure_months; internal_mobility; recruitment_cost; estimated_replacement_cost; gender; age_band (audit only).\nAll fields are optional except employee_id; columns can be remapped in the app. sample_data.csv is fully synthetic and fictional.\n"
LIMITS = "# Limitations\nAssociations are not causes. Predictions are decision-support signals with uncertainty. Replacement-cost estimates may be assumptions. Subgroup and fairness checks are partial. No drift monitoring, causal estimation or legal fairness testing is implemented. Intervention effects must be measured with experiments.\n"
RAI_DOC = "# Responsible AI\n" + RAI_MD + "\n"
LICENSE = "MIT License\n\nCopyright (c) 2026 PULSE AI contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
README = """# PULSE AI — People Intelligence & Retention Studio
Understand the patterns. Improve the workplace.

## Business problem and features
Employee turnover brings recruitment costs, lost knowledge and disruption. PULSE AI integrates workload, compensation, engagement, absence, training and career data to find organisational patterns associated with exits, compare retention interventions and estimate scenario costs and benefits. Bundled data is **fictional synthetic data**.
Features: CSV upload with column mapping and validation, headcount-safe attrition analytics, driver analysis with intervals and effect sizes, optional calibrated attrition model with baseline comparison, permutation importance, restricted employee-level view, evidence-linked intervention playbook, ROI simulator, rule-based strategist with optional LLM, responsible-AI audit, exports and report.

## Architecture
@@@mermaid
flowchart LR
  A[CSV upload or synthetic data] --> B[Validation + mapping]
  B --> C[Employee table + monthly records]
  C --> D[Workforce analytics]
  C --> E[Driver analysis]
  C --> F[Snapshot model + baseline]
  E --> G[Intervention playbook]
  F --> H[Explainability + audit]
  G --> I[ROI simulator]
  D --> J[Strategist tools]
  E --> J
  G --> J
  I --> J
  H --> J
  J --> K[Gradio UI, exports, report]
@@@

## Installation and launch
@@@
pip install -r requirements.txt
python app.py
@@@
Google Colab: paste `app.py` into one cell and run it.

## Data, preprocessing and methodology
See `docs/data_dictionary.md` and `docs/methodology.md`. If auto-detection fails, map columns such as `employee_id=emp_no, overtime_hours=ot`.

## Model, explainability and limitations
Calibrated model with a department-average baseline; permutation importance and what-if explanations are not causal. See `docs/limitations.md`.

## LLM configuration
Optional: Anthropic or an OpenAI-compatible endpoint; enter a key in the app (session only) or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Without a key a deterministic assistant is used. Only aggregated tool results are sent.

## Intervention logic and ROI
Evidence-to-intervention rules and the ROI formulas are in `docs/methodology.md`. Scenario outputs are assumptions, not savings.

## Privacy and responsible AI
See `docs/responsible_ai.md`. Never commit real employee data or keys. Exports are aggregated.

## Tests and deployment
@@@
pip install pytest
pytest tests
@@@
Deploy `app.py` as a Hugging Face Space (SDK: Gradio) with your own credentials. Add screenshots to `screenshots/`.
""".replace("@@@", "`" * 3)
TFILE = "from app import run_selftests\n\n\ndef test_%s():\n    res = [r for r in run_selftests(verbose=False) if r[0] == '%s']\n    assert res and all(r[2] for r in res), res\n"
def _src():
    try: return open(__file__, encoding="utf-8").read()
    except Exception: pass
    try:
        for s in list(get_ipython().user_ns.get("In") or [])[::-1]:
            if "PULSE AI — People Intelligence" in s and "def main" in s: return s
    except Exception: pass
    return None
def build_project():
    root = "pulse-ai-retention-intelligence"
    for d in ["tests", "docs", "screenshots"]: os.makedirs(os.path.join(root, d), exist_ok=True)
    src = _src()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    SAMPLE.to_csv(root + "/sample_data.csv", index=False)
    for fn, tx in [("requirements.txt", "gradio>=4.44,<6\nplotly\nscikit-learn>=1.2\nscipy\nrequests\npandas\nnumpy\n"), ("README.md", README), ("LICENSE", LICENSE), (".gitignore", ".env\n*.key\nsecrets*\n__pycache__/\n.ipynb_checkpoints/\n*.zip\npulse_*.csv\npulse_*.md\npulse_*.html\n"),
                   ("docs/methodology.md", METH), ("docs/data_dictionary.md", DICT), ("docs/responsible_ai.md", RAI_DOC), ("docs/limitations.md", LIMITS), ("screenshots/README.md", "# Screenshots\nAdd your own screenshots of the running app here. None are included.\n"),
                   ("tests/test_data_validation.py", TFILE % ("data_validation", "data")), ("tests/test_workforce_metrics.py", TFILE % ("workforce_metrics", "workforce")), ("tests/test_driver_analysis.py", TFILE % ("driver_analysis", "drivers")),
                   ("tests/test_model_pipeline.py", TFILE % ("model_pipeline", "model")), ("tests/test_interventions.py", TFILE % ("interventions", "interventions")), ("tests/test_roi_simulator.py", TFILE % ("roi_simulator", "roi"))]: open(os.path.join(root, fn), "w", encoding="utf-8").write(tx)
    z = "pulse_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

# ------------------------------ UI ------------------------------
CSS = """
.gradio-container{background:#151C27!important;color:#F2F5F7!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1320px!important}
.hdr{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:14px;padding:28px 30px;border-radius:14px;background:#242F3B;border:1px solid #ffffff1a;border-left:4px solid #7BE0C3;margin-bottom:12px}
.brand{display:flex;gap:16px;align-items:center}.logo{width:52px;height:52px;border-radius:10px;border:2px solid #7BE0C3;color:#7BE0C3;font-weight:800;font-size:22px;display:flex;align-items:center;justify-content:center}
.hdr h1{margin:0;font-size:30px;letter-spacing:3px;color:#F2F5F7;font-weight:700}.hdr h1 span{color:#7BE0C3}.hdr .sub{color:#D3B5FF;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin-top:2px}.hdr p{margin:6px 0 0;color:#A7B5C5;font-size:15px;font-style:italic}
.badges span{display:inline-block;margin:3px;padding:4px 11px;border-radius:6px;background:#ffffff0d;border:1px solid #ffffff1a;color:#A7B5C5;font-size:12px}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:10px 0}.kc{background:#242F3B;border:1px solid #ffffff14;border-radius:10px;padding:14px 16px}
.kl{color:#A7B5C5;font-size:11px;text-transform:uppercase;letter-spacing:1.2px}.kv{font-size:24px;font-weight:700;margin:6px 0 4px}.kd{color:#A7B5C5;font-size:12px}
.tag{display:inline-block;border:1px solid;border-radius:4px;padding:0 6px;font-size:10px;letter-spacing:.8px;font-weight:700;margin-right:4px}
.note{background:#242F3B;padding:10px 14px;border-radius:8px;margin:8px 0;color:#F2F5F7;font-size:14px;line-height:1.5}
.sec{font-size:15px;font-weight:700;color:#D3B5FF;margin:18px 0 6px;text-transform:uppercase;letter-spacing:1px}.ins{background:#242F3B;border-radius:8px;padding:12px 12px 12px 30px;font-size:14px;line-height:1.6}
.tw{overflow-x:auto;border-radius:8px;border:1px solid #ffffff14;margin:8px 0}table{border-collapse:collapse;width:100%;background:#242F3B;font-size:13px}
th{background:#2d3a49;color:#7BE0C3;text-align:left;padding:9px 10px;font-weight:600}td{padding:8px 10px;border-top:1px solid #ffffff10;color:#F2F5F7}tr:hover td{background:#7BE0C30d}
button.primary{background:#7BE0C3!important;color:#151C27!important;border:none!important;font-weight:700!important}
@media(max-width:640px){.hdr h1{font-size:22px}.kv{font-size:20px}.hdr{padding:18px}}
"""
HEADER = ("<div class='hdr'><div class='brand'><div class='logo'>PU</div><div><h1>PULSE <span>AI</span></h1><div class='sub'>People Intelligence &amp; Retention Studio</div><p>Understand the patterns. Improve the workplace.</p></div></div>"
          "<div class='badges'><span>Associations, not causes</span><span>Decision support, never sole basis</span><span>Core analytics need no API key</span></div></div>")
def export_pb(df): d = pd.DataFrame(df); d.to_csv("pulse_playbook_edited.csv", index=False); open("pulse_playbook_edited.md", "w", encoding="utf-8").write(playbook_md(d)); return ["pulse_playbook_edited.csv", "pulse_playbook_edited.md"]
def roi_reset(): n, b, m, rc, ob, vc, t = roi_defaults(); return n, b, m, rc, ob, vc, t
def build_ui(zip_path, sample_path):
    with gr.Blocks(css=CSS, title="PULSE AI", theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("Data and settings (demo data is FICTIONAL and illustrative)", open=True):
            with gr.Row():
                up = gr.File(label="Upload employee CSV (leave empty for the demo)", file_types=[".csv"])
                with gr.Column():
                    mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="employee_id=emp_no, overtime_hours=ot, attrition=left", lines=2)
                    gr.File(value=sample_path, label="Sample dataset (synthetic)", interactive=False)
            with gr.Row():
                hz = gr.Slider(3, 12, 6, step=1, label="Prediction horizon (months)"); obj = gr.Radio(list(OBJ), value="Balanced (F1)", label="Decision-threshold objective"); pct = gr.Slider(10, 100, 40, step=5, label="Replacement cost if none supplied (% of annual salary, ASSUMPTION)"); pse = gr.Checkbox(True, label="Pseudonymise employee IDs (uploads)")
            run_btn = gr.Button("Analyse workforce", variant="primary"); status = gr.HTML(note("Loading the fictional demo workforce and training models (about 1-2 minutes)...", "info"))
        with gr.Accordion("Filters (apply to Overview, Explorer and Driver Analysis)", open=False):
            with gr.Row():
                fl_dep = gr.Dropdown([], multiselect=True, label="Department"); fl_rol = gr.Dropdown([], multiselect=True, label="Job role"); fl_ten = gr.Dropdown([], multiselect=True, label="Tenure band (months)")
            with gr.Row():
                fl_d1 = gr.Textbox(label="Exit period from (YYYY-MM-DD)"); fl_d2 = gr.Textbox(label="Exit period to (YYYY-MM-DD)"); fl_btn = gr.Button("Apply filters", variant="primary")
        with gr.Tabs():
            with gr.Tab("Executive Overview"):
                ov_html = gr.HTML()
                with gr.Row():
                    ov1 = gr.Plot(); ov2 = gr.Plot()
                ov3 = gr.Plot()
            with gr.Tab("Workforce Explorer"):
                with gr.Row():
                    by = gr.Dropdown(["department", "job_role", "tenure_band", "job_level", "manager_id"], value="department", label="Group by"); minn = gr.Slider(5, 100, 15, step=5, label="Minimum group size"); ex_btn = gr.Button("Update", variant="primary")
                ex_html = gr.HTML(); ex_fig = gr.Plot()
            with gr.Tab("Driver Analysis"):
                with gr.Row():
                    dr_f = gr.Dropdown([], label="Factor to chart"); ia = gr.Dropdown([], label="Interaction factor A"); ib = gr.Dropdown([], label="Interaction factor B"); dr_btn = gr.Button("Update", variant="primary")
                dr_html = gr.HTML()
                with gr.Row():
                    dr1 = gr.Plot(); dr2 = gr.Plot()
            with gr.Tab("Attrition Model and Explainability"):
                md_html = gr.HTML()
                with gr.Row():
                    m1 = gr.Plot(); m2 = gr.Plot()
                with gr.Row():
                    m3 = gr.Plot(); m4 = gr.Plot()
                with gr.Row():
                    m5 = gr.Plot(); m6 = gr.Plot()
                with gr.Accordion("Restricted: employee-level view (authorised HR use only)", open=False):
                    auth = gr.Checkbox(False, label="I am authorised to view pseudonymous employee-level decision-support signals for a lawful purpose and will not use them as the sole basis for any employment decision")
                    rs_id = gr.Dropdown([], label="Pseudonymous employee ID (highest modelled risk first)", allow_custom_value=True); rs_btn = gr.Button("Show", variant="primary"); rs_html = gr.HTML(); rs_fig = gr.Plot()
            with gr.Tab("Retention Interventions"):
                pb_html = gr.HTML(); pb_df = gr.Dataframe(interactive=True, wrap=True, label="Editable playbook"); pb_btn = gr.Button("Export edited playbook (Markdown + CSV)", variant="primary"); pb_file = gr.File(label="Playbook files", file_count="multiple", interactive=False)
            with gr.Tab("ROI Simulator"):
                gr.Markdown("All outputs are **scenario estimates**, not measured savings. HR must validate assumptions before budgeting.")
                with gr.Row():
                    r_n = gr.Number(1000, label="Eligible employees"); r_b = gr.Number(15, label="Baseline annual attrition %"); r_m = gr.Number(12, label="Evaluation period (months)")
                with gr.Row():
                    r_rec = gr.Number(0, label="Recruitment cost per replacement"); r_onb = gr.Number(0, label="Onboarding and training cost per replacement"); r_vac = gr.Number(0, label="Vacancy / productivity cost per replacement")
                with gr.Row():
                    lowm = gr.Slider(.1, 1, .5, step=.05, label="Low scenario: effect multiplier"); highm = gr.Slider(1, 3, 1.5, step=.05, label="High scenario: effect multiplier"); r_rst = gr.Button("Reset to data-derived defaults")
                scn = gr.Dataframe(value=pd.DataFrame(columns=SCN_COLS), interactive=True, wrap=True, label="Interventions: editable assumptions (cost per employee is a placeholder; replace with your own)")
                r_btn = gr.Button("Run simulation", variant="primary"); r_html = gr.HTML()
                with gr.Row():
                    r_f1 = gr.Plot(); r_f2 = gr.Plot()
                r_file = gr.File(label="Scenario results (CSV)", interactive=False)
            with gr.Tab("PULSE Strategist"):
                gr.Markdown("Questions are answered by analytical tools over the current dataset. Without an API key a rule-based assistant answers. With a key, an LLM phrases the answer from **aggregated tool outputs only**; no employee-level data is sent. Check your organisation's policy before enabling external processing.")
                ag_q = gr.Textbox(label="Ask a question", placeholder="Which organisational patterns are most concerning? / Is overtime associated with attrition? / How should HR evaluate a retention programme?")
                with gr.Row():
                    prov = gr.Radio(["Anthropic", "OpenAI-compatible"], value="Anthropic", label="LLM provider (optional)"); mdl = gr.Textbox(label="Model", value="claude-sonnet-5-5"); base = gr.Textbox(label="Base URL (OpenAI-compatible only)"); key = gr.Textbox(label="API key (optional, session only)", type="password")
                ag_btn = gr.Button("Ask", variant="primary"); ag_out = gr.Markdown()
            with gr.Tab("Responsible AI"):
                rai_html = gr.HTML()
            with gr.Tab("Data Quality and Exports"):
                dq_html = gr.HTML(); t_html = gr.HTML(); gr.Markdown(METH)
                gr.Markdown("### Export and publish\nDownload the ZIP, unzip it, create an empty GitHub repository and upload the contents (Add file, then Upload files). Exports are aggregated; never upload real employee data or API keys.")
                gr.File(value=zip_path, label="Download GitHub-ready project (ZIP)", interactive=False)
                with gr.Row():
                    rep_btn = gr.Button("Generate executive summary", variant="primary"); exp_btn = gr.Button("Export analysis files (aggregated)", variant="primary")
                rep_msg = gr.HTML(); rep_file = gr.File(label="Executive summary (HTML)", interactive=False); exp_msg = gr.HTML(); exp_files = gr.File(label="Exports", file_count="multiple", interactive=False)
        INF = [fl_dep, fl_rol, fl_ten, fl_d1, fl_d2]; EXI = INF + [by, minn]; DRI = INF + [dr_f, ia, ib]; ROI_IN = [r_n, r_b, r_m, r_rec, r_onb, r_vac, lowm, highm, scn]; ROI_OUT = [r_html, r_f1, r_f2, r_file]
        def chain(ev):
            e = ev.then(refresh, None, [fl_dep, fl_rol, fl_ten, dr_f, ia, ib, rs_id, scn]).then(render_overview, INF, [ov_html, ov1, ov2, ov3]).then(render_explorer, EXI, [ex_html, ex_fig]).then(render_drivers, DRI, [dr_html, dr1, dr2])
            e = e.then(render_model, None, [md_html, m1, m2, m3, m4, m5, m6]).then(render_playbook, None, [pb_html, pb_df]).then(roi_reset, None, [r_n, r_b, r_m, r_rec, r_onb, r_vac, scn]).then(render_roi, ROI_IN, ROI_OUT)
            return e.then(render_rai, None, [rai_html]).then(render_dq, None, [dq_html, t_html])
        chain(run_btn.click(run_all, [up, mp, hz, obj, pct, pse], [status], show_progress="full")); chain(demo.load(run_all, [up, mp, hz, obj, pct, pse], [status]))
        fl_btn.click(render_overview, INF, [ov_html, ov1, ov2, ov3]).then(render_explorer, EXI, [ex_html, ex_fig]).then(render_drivers, DRI, [dr_html, dr1, dr2])
        ex_btn.click(render_explorer, EXI, [ex_html, ex_fig]); dr_btn.click(render_drivers, DRI, [dr_html, dr1, dr2]); dr_f.change(render_drivers, DRI, [dr_html, dr1, dr2]); rs_btn.click(restricted, [auth, rs_id], [rs_html, rs_fig])
        pb_btn.click(export_pb, pb_df, pb_file); r_btn.click(render_roi, ROI_IN, ROI_OUT, show_progress="full"); r_rst.click(roi_reset, None, [r_n, r_b, r_m, r_rec, r_onb, r_vac, scn]).then(render_roi, ROI_IN, ROI_OUT)
        ag_btn.click(ask, [ag_q, prov, mdl, base, key], ag_out, show_progress="full"); ag_q.submit(ask, [ag_q, prov, mdl, base, key], ag_out); rep_btn.click(make_report, None, [rep_file, rep_msg]); exp_btn.click(export_all, pb_df, [exp_files, exp_msg])
    return demo
def main():
    zp, saved = build_project(); sp = "pulse_sample_data.csv"; SAMPLE.to_csv(sp, index=False)
    print("=" * 78); print("PULSE AI starting. Demo data is FICTIONAL and illustrative. Core analytics need no API key."); print("Running self-tests (about 30-60 seconds):")
    TESTS[:] = run_selftests(verbose=True); demo = build_ui(zp, sp)
    print("Project ZIP:", zp, "| standalone app.py included:", "yes" if saved else "NO (create app.py yourself by pasting this cell into it)")
    print("The demo analyses automatically (1-2 minutes); tabs fill in when it finishes. Keys: use Colab Secrets (ANTHROPIC_API_KEY / OPENAI_API_KEY). The public Gradio link is open to anyone: do not load real employee data unless you restrict access.")
    print("=" * 78); demo.queue().launch(debug=False, show_error=True)
if __name__ == "__main__" or "google.colab" in sys.modules:
    main()