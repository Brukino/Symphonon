"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYMPHONON FV — Fault Injection + Gold Standard Comparison                 ║
║                                                                              ║
║  Protocollo:                                                                ║
║    1. FAULT INJECTION su dati reali HKUST                                  ║
║       - 3 tipi di fault × 4 severità × 20 stazioni = 240 esperimenti      ║
║       - Fault iniettati su serie temporali reali (contesto autentico)       ║
║                                                                              ║
║    2. GOLD STANDARD: PR con soglia mobile                                  ║
║       - Allarme se PR < α × EMA_30(PR)  con α ∈ {0.80, 0.85, 0.90}       ║
║       - Standard industriale diffuso (IEC 61724-2 inspired)               ║
║       - Calcolato in parallelo su ogni esperimento                          ║
║                                                                              ║
║    3. CONFRONTO SIDE-BY-SIDE                                               ║
║       - Detection rate, advance warning, FA                                ║
║       - Su stessa serie → confronto controllato                             ║
║       - ROC-like curve: FA vs detection al variare della soglia            ║
║                                                                              ║
║  Tipi di fault iniettati:                                                   ║
║    F1: Soiling progressivo — drift lineare lento del PR                    ║
║        (riduzione 5-25% su T=60-120 giorni)                                ║
║    F2: Degrado inverter progressivo — drift + rumore crescente             ║
║        (efficienza cala, varianza aumenta = signature coupling)            ║
║    F3: Fault episodico — cali brevi e ricorrenti del PR                   ║
║        (convertitore intermittente, pre-fault recurrent drops)             ║
║                                                                              ║
║  ZERO TUNING: tutti i parametri Symphonon FV identici a run HKUST         ║
╚══════════════════════════════════════════════════════════════════════════════╝

ESECUZIONE:
  python symphonon_fv_fault_injection.py --data "D:\\...\\Dataset\\Dataset" --out output_fi
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, sys, warnings, json, argparse, re
from pathlib import Path
from itertools import product

warnings.filterwarnings('ignore')

# ── Parametri Symphonon FV (identici alla run principale) ────────────────────
WIN=24; STEP=6; RW=7; RW_PR=14; THR_P=0.72
L_PERS_A=8; THR_PERS_A=0.12; THR_VEL_AMP=0.08
L_VEL_SMOOTH=4; VEL_SHORT=2; VEL_LONG=14
L_PERS_B=20; THR_PERS_BASE=0.62; PERS_FLOOR=0.10; ALPHA=0.80; L_VEL=30
WARMUP_WIN=60; SEASON_EMA=90; SEASON_INIT_WIN=60
K_CONFIRM=4; MIN_OFF_WIN=2; FLEET_MIN=3
G_NIGHT=50.; G_LOW=200.; G_HIGH=600.
CLOUD_ALPHA=0.20; CLOUD_EMA=90; CLOUD_SMOOTH=12; CLOUD_CLIP=2.; THR_B_CEIL=0.85
DARK_BG='#0d0d0d'
PV_GEN   = "Time series dataset/PV generation dataset"
NO_OPT   = "PV stations without panel level optimizer/Site level dataset"
WITH_OPT = "PV stations with panel level optimizer/Inverter level dataset"

# ── Protocollo fault injection ────────────────────────────────────────────────
FAULT_TYPES = {
    'F1_soiling': {
        'desc': 'Soiling progressivo — drift lineare lento del PR',
        'severities': [0.05, 0.10, 0.15, 0.25],   # riduzione finale PR
        'duration_days': [60, 90, 120, 90],         # durata del drift
    },
    'F2_inverter': {
        'desc': 'Degrado inverter — drift + rumore crescente',
        'severities': [0.08, 0.12, 0.18, 0.25],
        'duration_days': [45, 60, 75, 60],
    },
    'F3_episodic': {
        'desc': 'Fault episodico — cali brevi ricorrenti',
        'severities': [0.10, 0.15, 0.20, 0.30],    # profondità del calo
        'duration_days': [45, 60, 90, 75],
    },
}

# Gold standard thresholds (PR < α × EMA_30)
GS_ALPHAS = [0.80, 0.85, 0.90]

# N stazioni per esperimento (usa stazioni con dati completi)
N_STATIONS_EXP = 20
ADV_WINDOW     = 30   # giorni — finestra advance warning
RANDOM_SEED    = 42


# ══════════════════════════════════════════════════════════════════════════════
#  CARICAMENTO DATI (identico al runner principale)
# ══════════════════════════════════════════════════════════════════════════════

def find_pv_csvs(data_dir):
    base=Path(data_dir); found=[]
    for target in ['Site level dataset','Inverter level dataset']:
        for d in base.rglob(target):
            found.extend(sorted(d.glob('*.csv')))
    if not found:
        found=sorted([f for f in base.rglob('*.csv')
                      if not any(k in f.name.lower()
                                 for k in ['irrad','weather','meteo','readme'])])
    return found

def load_pv_csv(path):
    try:
        df=pd.read_csv(str(path),low_memory=False)
        if df.empty or len(df.columns)<2: return None
        tc=df.columns[0]
        df[tc]=pd.to_datetime(df[tc],errors='coerce')
        df=df.dropna(subset=[tc]).set_index(tc).sort_index()
        pc=next((c for c in df.columns if 'power' in c.lower()),None)
        if pc is None:
            pc=next((c for c in df.columns if 'gen' in c.lower() or 'kwh' in c.lower()),None)
        if pc is None:
            num=df.select_dtypes(include=[np.number]).columns
            if not len(num): return None
            pc=num[0]
        power=pd.to_numeric(df[pc],errors='coerce').clip(0)
        if 'kwh' in pc.lower(): power=power*1000.
        p99=power.quantile(0.999)
        if p99>0: power=power.clip(upper=p99*1.5)
        s=power.resample('1h').mean().dropna()
        return s if len(s)>=WIN*3 else None
    except: return None

def find_irradiance(data_dir):
    for f in Path(data_dir).rglob('*.csv'):
        if any(k in f.name.lower() for k in ['irrad','weather','meteo','solar']): return f
    for f in sorted(Path(data_dir).rglob('*.csv'))[:15]:
        try:
            h=pd.read_csv(str(f),nrows=0).columns.tolist()
            if any('irrad' in c.lower() for c in h): return f
        except: pass
    return None

def load_irradiance(path,t_index):
    if path is None: return None
    try:
        df=pd.read_csv(str(path),low_memory=False)
        tc=df.columns[0]
        df[tc]=pd.to_datetime(df[tc],errors='coerce')
        df=df.dropna(subset=[tc]).set_index(tc).sort_index()
        gc=next((c for c in df.columns if any(k in c.lower() for k in ['irrad','ghi','solar','rad'])),df.columns[0])
        G=pd.to_numeric(df[gc],errors='coerce').clip(0,1500)
        G_1h=G.resample('1h').mean()
        G_aln=G_1h.reindex(t_index,method='nearest',tolerance=pd.Timedelta('90min'))
        return G_aln if G_aln.notna().mean()>0.2 else None
    except: return None


# ══════════════════════════════════════════════════════════════════════════════
#  FAULT INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def inject_fault(power_series, G_series, fault_type, severity, duration_days,
                 inject_start_frac=0.5, seed=42):
    """
    Inietta un fault sintetico su power_series reale.

    inject_start_frac: dove inizia il fault (0.5 = metà della serie)
    Ritorna:
        power_injected: pd.Series con fault
        fault_start: Timestamp inizio fault
        fault_end: Timestamp fine fault (= inject_start + duration_days)
    """
    rng = np.random.default_rng(seed)
    P = power_series.copy()
    T = len(P)

    # Trova indice inizio fault (evita warmup e fine serie)
    inject_idx = int(T * inject_start_frac)
    inject_idx = max(WARMUP_WIN + WIN, min(inject_idx, T - duration_days * 24 - 10))
    fault_start = P.index[inject_idx]
    inject_end  = min(inject_idx + duration_days * 24, T - 1)
    fault_end   = P.index[inject_end]

    P_arr = P.values.copy()

    # Maschera diurna per applicare il fault solo quando c'è produzione
    if G_series is not None:
        G_aln = G_series.reindex(P.index, method='nearest',
                                  tolerance=pd.Timedelta('90min')).fillna(0).values
        day_mask = G_aln > G_NIGHT
    else:
        h = pd.DatetimeIndex(P.index).hour.values
        day_mask = (h >= 6) & (h <= 19)

    if fault_type == 'F1_soiling':
        # Drift lineare: PR scende linearmente di 'severity' su duration_days
        n_fault = inject_end - inject_idx
        ramp = np.linspace(0, severity, n_fault)
        for i in range(n_fault):
            idx = inject_idx + i
            if day_mask[idx] and P_arr[idx] > 0:
                P_arr[idx] *= (1 - ramp[i])

    elif fault_type == 'F2_inverter':
        # Drift + rumore crescente
        n_fault = inject_end - inject_idx
        ramp = np.linspace(0, severity, n_fault)
        noise_scale = np.linspace(0, severity * 0.5, n_fault)
        for i in range(n_fault):
            idx = inject_idx + i
            if day_mask[idx] and P_arr[idx] > 0:
                noise = rng.normal(0, noise_scale[i] * P_arr[idx])
                P_arr[idx] = max(0, P_arr[idx] * (1 - ramp[i]) + noise)

    elif fault_type == 'F3_episodic':
        # Cali brevi ricorrenti: ogni 3-7 giorni, durata 2-6 ore
        t = inject_idx
        while t < inject_end:
            if day_mask[min(t, T-1)] and P_arr[min(t, T-1)] > 0:
                # Durata evento: 2-8 ore
                event_dur = int(rng.integers(2, 9))
                for j in range(event_dur):
                    if t + j < inject_end and day_mask[t+j] and P_arr[t+j] > 0:
                        P_arr[t+j] *= (1 - severity * rng.uniform(0.5, 1.0))
            # Intervallo fino al prossimo evento: 2-7 giorni
            t += int(rng.integers(48, 168))

    P_inj = pd.Series(P_arr, index=P.index)
    return P_inj, fault_start, fault_end


# ══════════════════════════════════════════════════════════════════════════════
#  GOLD STANDARD — PR con soglia mobile
# ══════════════════════════════════════════════════════════════════════════════

def gold_standard_pr(power_series, G_series, alpha=0.85):
    """
    Gold standard industriale: allarme se PR_t < alpha × EMA_30d(PR_t)

    Parameters:
        alpha: soglia (0.80 = allarme se PR scende del 20%, ecc.)

    Ritorna:
        pd.Series(bool) — allarme per ogni campione orario
    """
    P = power_series.values.copy()

    if G_series is not None:
        G = G_series.reindex(power_series.index, method='nearest',
                              tolerance=pd.Timedelta('90min')).fillna(0).values
    else:
        h = pd.DatetimeIndex(power_series.index).hour.values
        G = np.where((h >= 6) & (h <= 19), 500., 0.)

    G = np.clip(G, 0, 1500)

    # Performance Ratio istantaneo
    P_peak = np.nanpercentile(P[G > G_NIGHT], 98) if (G > G_NIGHT).any() else 1.
    if P_peak < 1e-3: P_peak = 1.

    PR = np.where(G > G_NIGHT,
                  np.clip(P / (G / 1000. * P_peak + 1e-6), 0, 1.5),
                  np.nan)

    # EMA 30 giorni (720h)
    w = 720; a = 2 / (w + 1)
    PR_ema = np.zeros(len(PR)); PR_ema[0] = PR[0] if not np.isnan(PR[0]) else 0.8
    for i in range(1, len(PR)):
        val = PR[i] if not np.isnan(PR[i]) else PR_ema[i-1]
        PR_ema[i] = a * val + (1 - a) * PR_ema[i-1]

    # Allarme: PR < alpha × EMA_30  (solo ore diurne)
    alarm = np.zeros(len(PR), bool)
    for i in range(w, len(PR)):
        if G[i] > G_NIGHT and not np.isnan(PR[i]):
            if PR[i] < alpha * PR_ema[i]:
                alarm[i] = True

    # Conferma: allarme solo se persiste per K_CONFIRM ore consecutive
    confirmed = np.zeros(len(alarm), bool)
    for i in range(K_CONFIRM, len(alarm)):
        if all(alarm[i-K_CONFIRM:i+1]):
            confirmed[i] = True

    return pd.Series(confirmed, index=power_series.index), \
           pd.Series(PR, index=power_series.index), \
           pd.Series(PR_ema, index=power_series.index)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE SYMPHONON FV (invarianti — compatta)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(a,w):
    a=np.asarray(a,float); alpha=2/(w+1); o=np.zeros_like(a); o[0]=a[0]
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o

def _rsig(a,w):
    o=np.zeros(len(a))
    for i in range(len(a)):
        s=a[max(0,i-w):i+1]; o[i]=1/(1+np.exp(-(a[i]-s.mean())/(s.std()+1e-8)))
    return o

def _pc1(W):
    try:
        e=np.abs(np.linalg.eigvalsh(np.cov(W.T))); s=e.sum()
        return float(e[-1]/s) if s>1e-10 else 0.
    except: return 0.

def _pr(W):
    try:
        e=np.abs(np.linalg.eigvalsh(np.cov(W.T))); s=e.sum()
        pr=(s**2)/((e**2).sum()+1e-12)
        return float(np.clip((pr-1)/(W.shape[1]-1+1e-8),0,1))
    except: return 0.5

def _nd(W,b): return float(np.clip(np.nanmean(np.abs(np.nanmean(W,0)-b)),0,3)/3)

def _ci(W,reg):
    best=0.
    for r in [0,1,2]:
        m=reg==r
        if m.sum()<5: continue
        try:
            e=np.abs(np.linalg.eigvalsh(np.cov(W[m].T))); s=e.sum()
            best=max(best,float(e[-1]/s) if s>1e-10 else 0)
        except: pass
    return best

def _rdet(n,reg,ew=SEASON_EMA,iw=SEASON_INIT_WIN):
    T=len(n); alpha=2/(ew+1); bl={}
    for r in [0,1,2]:
        m=reg[:iw]==r; bl[r]=float(n[:iw][m].mean() if m.any() else n[:iw].mean())
    out=np.zeros(T)
    for i in range(T):
        r=int(reg[i]) if reg[i] in (0,1,2) else 1
        if i>=iw: bl[r]=alpha*n[i]+(1-alpha)*bl[r]
        out[i]=max(n[i]-bl[r],0)
    return out

def _vamp(n):
    v=np.clip(_ema(n,VEL_SHORT)-_ema(n,VEL_LONG),0,None)
    vs=_ema(v,L_VEL_SMOOTH); p=np.percentile(vs,95)+1e-8
    return np.clip(vs/p,0,1)

def _svp(n):
    v=(_ema(n,VEL_SHORT)-_ema(n,VEL_LONG)>0).astype(float)
    o=np.zeros(len(n))
    for i in range(len(n)): o[i]=v[max(0,i-L_VEL):i+1].mean()
    return o

def _pers(pb,L=None,thr=THR_P):
    if L is None: L=L_PERS_B
    ab=(pb>thr).astype(float); o=np.zeros(len(ab))
    for i in range(len(ab)): o[i]=ab[max(0,i-L):i+1].mean()
    return o

def _hdet(pb,gate,wu=WARMUP_WIN):
    T=len(pb); al=np.zeros(T,bool); st=np.zeros(T,int)
    cc=0; oc=0; ia=False
    for i in range(T):
        if i<wu: continue
        cand=(pb[i]>THR_P) and (gate[i]>0)
        if cand: cc+=1; oc=0
        else: oc+=1; cc=0
        if cc>=K_CONFIRM: st[i]=3; al[i]=True; ia=True
        elif cc>0: st[i]=2 if cc>=K_CONFIRM//2 else 1
        elif ia and oc<MIN_OFF_WIN: st[i]=2; al[i]=True
        else:
            if oc>=MIN_OFF_WIN: ia=False; st[i]=0
    return al,st


def run_symphonon_fv(power_series, G_series, fleet_n_raw=None, cloud_z_arr=None):
    """
    Esegue la pipeline Symphonon FV su una stazione con eventuale fleet deviation.
    fleet_n_raw: se disponibile, array già calcolato per la fleet deviation
    """
    P=power_series.values.copy()
    t_index=power_series.index

    if G_series is not None:
        G=G_series.reindex(t_index,method='nearest',
                            tolerance=pd.Timedelta('90min')).fillna(0).clip(0,1500).values
    else:
        h=pd.DatetimeIndex(t_index).hour.values
        G=np.where((h>=6)&(h<=19),500.,0.)

    day_mask=G>G_NIGHT
    P_peak=np.nanpercentile(P[day_mask],98) if day_mask.any() else 1.
    if P_peak<1e-3: P_peak=1.

    G_safe=np.where(G>G_NIGHT,G,np.nan)
    PR=np.where(G>G_NIGHT,np.clip(P/(G_safe/1000.*P_peak+1e-6),0,1.5),np.nan)
    P_norm=np.clip(P/(P_peak+1e-6),0,1.2)
    PR_f=np.where(np.isnan(PR),0.,PR)
    a_=2/25; PR_ema=np.zeros_like(PR_f); PR_ema[0]=PR_f[0]
    for i in range(1,len(PR_f)): PR_ema[i]=a_*PR_f[i]+(1-a_)*PR_ema[i-1]

    arr=np.column_stack([PR,P_norm,PR_ema])
    di=np.where(day_mask)[0]
    if len(di)<WIN*3: return None,None

    mu=arr[di].mean(0); std=arr[di].std(0)+1e-8; az=(arr-mu)/std
    reg=np.where(G<G_LOW,0,np.where(G<G_HIGH,1,2)); reg[G<G_NIGHT]=-1
    ac=az.copy()
    for r in [0,1,2]:
        m=reg==r
        if m.sum()>10: ac[m]-=az[m].mean(0)

    # Invarianti
    bl=arr[di[:WIN*3]].mean(0)
    ti=[]; nr=[]; kr=[]; cr=[]; prr=[]; cir=[]; rw=[]
    T_=len(ac)
    for s in range(0,T_-WIN,STEP):
        W=ac[s:s+WIN]; rs=reg[s:s+WIN]; dm=day_mask[s:s+WIN]
        if dm.mean()<0.25: continue
        Wd=W[dm]; rd=rs[dm]
        if len(Wd)<5 or np.isnan(Wd).mean()>0.4: continue
        ti.append(s+WIN//2)
        ts=np.nanstd(Wd,0); kap=float(np.clip(1-np.nanmean(ts)/(np.nanstd(ts)+1e-8),0,1))
        nr.append(_nd(Wd,bl)); kr.append(kap)
        cr.append(_pc1(Wd)); prr.append(_pr(Wd)); cir.append(_ci(Wd,rd))
        rw.append(int(reg[s+WIN//2]) if reg[s+WIN//2]>=0 else 1)
    if len(ti)<10: return None,None

    t_arr=np.array(ti); tdt=t_index[t_arr.clip(0,len(t_index)-1)]
    n=np.array(nr); pr_=np.array(prr); ci_=np.array(cir); rw_=np.array(rw)

    nrs=_rsig(n,RW); prrs=_rsig(_ema(pr_,3),RW_PR); cirs=_rsig(ci_,RW)
    Pb=0.45*nrs+0.30*prrs+0.25*cirs

    fdev=np.zeros(len(t_arr)) if fleet_n_raw is None else fleet_n_raw
    na=_rdet(n,rw_); ps=_pers(Pb,L=L_PERS_A); va=_vamp(na)
    gA=((ps>THR_PERS_A)&(va>THR_VEL_AMP)).astype(float)
    pl=_pers(Pb,L=L_PERS_B); vs=_svp(fdev)
    czz=np.zeros(len(vs))
    if cloud_z_arr is not None and len(cloud_z_arr)==len(vs): czz=cloud_z_arr
    thr=np.minimum(np.maximum(THR_PERS_BASE-ALPHA*vs+CLOUD_ALPHA*czz,PERS_FLOOR),THR_B_CEIL)
    gB=(pl>thr).astype(float)
    al,st=_hdet(Pb,np.clip(gA+gB,0,1))

    alarm_series=pd.Series(False,index=t_index)
    for i,ti_ in enumerate(t_arr):
        if ti_<len(t_index): alarm_series.iloc[ti_]=bool(al[i])

    return alarm_series, tdt


# ══════════════════════════════════════════════════════════════════════════════
#  METRICHE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(alarm_series, fault_start, fault_end, t_index):
    """
    Valuta detection e advance warning.

    Detection: allarme alzato nella finestra [fault_start - ADV_WINDOW*24h, fault_end]
    Advance warning: giorni tra il primo allarme e fault_start
    False alarms: allarmi FUORI dalla finestra [fault_start-ADV_WINDOW*24h, fault_end+24h]
    """
    t_start_window = fault_start - pd.Timedelta(days=ADV_WINDOW)
    t_end_window   = fault_end   + pd.Timedelta(hours=24)

    # Campioni nella finestra detection
    in_window = (alarm_series.index >= t_start_window) & \
                (alarm_series.index <= t_end_window)
    out_window= (alarm_series.index < t_start_window - pd.Timedelta(days=7)) | \
                (alarm_series.index > t_end_window)

    detected = bool(alarm_series[in_window].any())
    advance_days = None
    if detected:
        first_alarm = alarm_series[in_window & alarm_series].index[0]
        advance_days = max(0, (fault_start - first_alarm).total_seconds() / 86400)

    # Conta FA (eventi distinti fuori finestra)
    fa_series = alarm_series[out_window]
    n_fa = 0; prev = False
    for a in fa_series:
        if a and not prev: n_fa += 1
        prev = a

    # Mesi di osservazione fuori finestra
    months_obs = max(1, out_window.sum() / (24 * 30.44))
    fa_rate = n_fa / months_obs

    return dict(detected=detected, advance_days=advance_days,
                n_fa=n_fa, fa_rate_mo=fa_rate)


# ══════════════════════════════════════════════════════════════════════════════
#  ESPERIMENTO COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(power_series, G_series, fault_type, severity, duration_days,
                   station_name, exp_id, seed=42):
    """
    Esegue un singolo esperimento di fault injection + confronto.
    Ritorna dict con risultati Symphonon FV e Gold Standard.
    """
    # Inietta fault
    P_inj, fault_start, fault_end = inject_fault(
        power_series, G_series, fault_type, severity, duration_days,
        inject_start_frac=0.5, seed=seed)

    results = {
        'station': station_name, 'fault_type': fault_type,
        'severity': severity, 'duration_days': duration_days,
        'fault_start': str(fault_start), 'fault_end': str(fault_end),
    }

    # ── Symphonon FV ──────────────────────────────────────────────────────────
    alarm_fv, _ = run_symphonon_fv(P_inj, G_series)
    if alarm_fv is not None:
        m = evaluate(alarm_fv, fault_start, fault_end, P_inj.index)
        results['fv'] = m
    else:
        results['fv'] = dict(detected=False, advance_days=None, n_fa=0, fa_rate_mo=0.)

    # ── Gold Standard (3 soglie) ───────────────────────────────────────────────
    results['gs'] = {}
    for alpha in GS_ALPHAS:
        alarm_gs, _, _ = gold_standard_pr(P_inj, G_series, alpha=alpha)
        m = evaluate(alarm_gs, fault_start, fault_end, P_inj.index)
        results['gs'][f'alpha_{alpha:.2f}'] = m

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════

def _sty(ax):
    ax.set_facecolor('#111'); ax.tick_params(colors='#888',labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#333')

def plot_roc_curve(all_results, out_dir):
    """ROC-like: detection rate vs FA rate per tipo di fault e soglia GS."""
    fig,axs=plt.subplots(1,3,figsize=(18,5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axs: _sty(ax)
    fig.suptitle('Symphonon FV vs Gold Standard — Detection Rate vs FA Rate',
                  color='#ddd',fontsize=11)

    for ax_i,(ftype,ax) in enumerate(zip(FAULT_TYPES.keys(),axs)):
        exp_f=[r for r in all_results if r['fault_type']==ftype]

        # Symphonon FV: un punto (media su severità)
        fv_dr=np.mean([r['fv']['detected'] for r in exp_f])
        fv_fa=np.mean([r['fv']['fa_rate_mo'] for r in exp_f])
        ax.scatter([fv_fa],[fv_dr],color='#00c853',s=120,zorder=5,
                   label=f'Symphonon FV  DR={fv_dr:.0%} FA={fv_fa:.2f}',marker='*')

        # Gold Standard per soglia
        colors_gs=['#ff8844','#44ccff','#ff66cc']
        for ai,(alpha,col) in enumerate(zip(GS_ALPHAS,colors_gs)):
            key=f'alpha_{alpha:.2f}'
            gs_dr=np.mean([r['gs'][key]['detected'] for r in exp_f])
            gs_fa=np.mean([r['gs'][key]['fa_rate_mo'] for r in exp_f])
            ax.scatter([gs_fa],[gs_dr],color=col,s=80,zorder=4,
                       label=f'GS α={alpha}  DR={gs_dr:.0%} FA={gs_fa:.2f}',marker='o')

        ax.set_xlim(-0.05,3.0); ax.set_ylim(-0.05,1.1)
        ax.axhline(0.5,color='#444',lw=0.5,ls=':'); ax.axvline(0.5,color='#444',lw=0.5,ls=':')
        ax.set_xlabel('FA rate (allarmi/mese)',color='#888',fontsize=9)
        ax.set_ylabel('Detection rate',color='#888',fontsize=9)
        ax.set_title(FAULT_TYPES[ftype]['desc'],color='#ddd',fontsize=9)
        ax.legend(fontsize=7,labelcolor='#ccc',framealpha=0.2,loc='lower right')

    plt.tight_layout()
    plt.savefig(out_dir/'fig_roc_comparison.png',dpi=150,bbox_inches='tight',facecolor=DARK_BG)
    plt.close(); print('  fig_roc_comparison.png')


def plot_advance_warning(all_results, out_dir):
    """Advance warning vs severità per metodo."""
    fig,axs=plt.subplots(1,3,figsize=(18,5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axs: _sty(ax)
    fig.suptitle('Advance Warning (giorni) vs Severità Fault',color='#ddd',fontsize=11)

    for ax_i,(ftype,ax) in enumerate(zip(FAULT_TYPES.keys(),axs)):
        sevs=sorted(set(r['severity'] for r in all_results if r['fault_type']==ftype))
        fv_adv=[np.nanmean([r['fv']['advance_days'] or 0
                            for r in all_results
                            if r['fault_type']==ftype and r['severity']==s])
                for s in sevs]
        ax.plot(sevs,fv_adv,color='#00c853',lw=2,marker='*',ms=10,label='Symphonon FV')

        colors_gs=['#ff8844','#44ccff','#ff66cc']
        for alpha,col in zip(GS_ALPHAS,colors_gs):
            key=f'alpha_{alpha:.2f}'
            gs_adv=[np.nanmean([r['gs'][key]['advance_days'] or 0
                                for r in all_results
                                if r['fault_type']==ftype and r['severity']==s])
                    for s in sevs]
            ax.plot(sevs,gs_adv,color=col,lw=1.5,marker='o',ms=7,label=f'GS α={alpha}')

        ax.set_xlabel('Severità fault (riduzione PR)',color='#888')
        ax.set_ylabel('Advance warning (giorni)',color='#888')
        ax.set_title(FAULT_TYPES[ftype]['desc'],color='#ddd',fontsize=9)
        ax.legend(fontsize=7,labelcolor='#ccc',framealpha=0.2)
        ax.axhline(30,color='#666',lw=0.7,ls=':',alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_dir/'fig_advance_warning.png',dpi=150,bbox_inches='tight',facecolor=DARK_BG)
    plt.close(); print('  fig_advance_warning.png')


def plot_example_fault(power_clean, power_injected, alarm_fv, alarm_gs,
                       pr_gs, pr_ema_gs, fault_start, fault_end,
                       station_name, fault_type, severity, out_dir):
    """Plot esempio singolo esperimento."""
    fig,axs=plt.subplots(3,1,figsize=(16,8),gridspec_kw={'height_ratios':[1.2,1,0.8]})
    fig.patch.set_facecolor(DARK_BG)
    for ax in axs: _sty(ax)

    # PR clean vs injected
    axs[0].plot(power_clean.index,power_clean.values,color='#446688',
                lw=0.7,alpha=0.5,label='P originale')
    axs[0].plot(power_injected.index,power_injected.values,color='#ff8844',
                lw=0.8,label='P con fault iniettato')
    axs[0].axvline(fault_start,color='#ff4444',lw=1.5,ls='--',label=f'fault start (sev={severity})')
    axs[0].axvline(fault_end,color='#ff4444',lw=1.,ls=':')
    axs[0].set_ylabel('Power (W)',color='#888',fontsize=8)
    axs[0].legend(fontsize=7,labelcolor='#ccc',framealpha=0.2,ncol=3)

    # PR e soglia GS
    pr_clean=pr_gs.values.copy(); pr_clean[np.isnan(pr_clean)]=0
    axs[1].plot(power_injected.index,pr_clean,color='#44ccff',lw=0.9,label='PR')
    axs[1].plot(power_injected.index,pr_ema_gs.values*0.85,color='#ffaa00',
                lw=1.,ls='--',alpha=0.8,label='GS soglia α=0.85')
    axs[1].axvline(fault_start,color='#ff4444',lw=1.5,ls='--')
    axs[1].set_ylim(0,1.5); axs[1].set_ylabel('Performance Ratio',color='#888',fontsize=8)
    axs[1].legend(fontsize=7,labelcolor='#ccc',framealpha=0.2,ncol=3)

    # Allarmi
    tv=power_injected.index
    for i in range(len(tv)-1):
        if alarm_fv is not None and i<len(alarm_fv) and alarm_fv.iloc[i]:
            axs[2].axvspan(tv[i],tv[i+1],color='#00c853',alpha=0.7)
        if i<len(alarm_gs) and alarm_gs.iloc[i]:
            axs[2].axvspan(tv[i],tv[i+1],color='#ff8844',alpha=0.4)
    axs[2].axvline(fault_start,color='#ff4444',lw=1.5,ls='--')
    from matplotlib.patches import Patch
    axs[2].legend(handles=[Patch(color='#00c853',label='Symphonon FV alarm'),
                            Patch(color='#ff8844',label='GS alarm (α=0.85)')],
                  fontsize=7,labelcolor='#ccc',framealpha=0.2,loc='upper left')
    axs[2].set_ylim(0,1); axs[2].set_yticks([])
    axs[0].set_title(f'{station_name} — {fault_type} sev={severity:.0%} — Fault Injection Example',
                      color='#ddd',fontsize=9)
    plt.tight_layout()
    safe=re.sub(r'[^\w]','_',f'{station_name}_{fault_type}_{severity}')[:40]
    plt.savefig(out_dir/f'fig_example_{safe}.png',dpi=110,bbox_inches='tight',facecolor=DARK_BG)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='.',
                    help='Cartella radice HKUST (contiene "Time series dataset")')
    ap.add_argument('--out',default='output_fi')
    ap.add_argument('--n-stations',type=int,default=20)
    ap.add_argument('--quick',action='store_true',
                    help='Run veloce: solo F1, 2 severità, 5 stazioni')
    args=ap.parse_args()

    data_dir=Path(args.data); out_dir=Path(args.out)
    out_dir.mkdir(parents=True,exist_ok=True)

    print('═'*72)
    print('  SYMPHONON FV — FAULT INJECTION + GOLD STANDARD COMPARISON')
    print('═'*72)

    # ── Carica dati ───────────────────────────────────────────────────────────
    print('\n[1/4] Caricamento dati HKUST...')
    pv_csvs=find_pv_csvs(data_dir)
    stations={}
    for f in pv_csvs:
        s=load_pv_csv(f)
        if s is not None and len(s)>WIN*20:
            stations[f.stem]=s
    print(f'  Stazioni caricate: {len(stations)}')

    irrad_path=find_irradiance(data_dir)
    all_times=set()
    for s in stations.values(): all_times.update(s.index.tolist())
    t_index=pd.DatetimeIndex(sorted(all_times)).round('1h').unique().sort_values()
    G_series=load_irradiance(irrad_path,t_index)
    print(f'  Irradianza: {"sì" if G_series is not None else "no"}')

    # Seleziona stazioni per gli esperimenti
    rng=np.random.default_rng(RANDOM_SEED)
    station_names=sorted(stations.keys())
    n_exp_stations=min(args.n_stations,len(station_names))
    if args.quick: n_exp_stations=5
    selected=list(rng.choice(station_names,n_exp_stations,replace=False))
    print(f'  Stazioni per esperimento: {n_exp_stations}')

    # ── Costruisci esperimenti ────────────────────────────────────────────────
    print('\n[2/4] Fault injection experiments...')
    if args.quick:
        exp_plan=[('F1_soiling',[0.10,0.20])]
    else:
        exp_plan=[(ft,FAULT_TYPES[ft]['severities']) for ft in FAULT_TYPES]

    all_results=[]
    example_plots_done=0

    for ft,severities in exp_plan:
        durations=FAULT_TYPES[ft]['duration_days']
        for si,(sev,dur) in enumerate(zip(severities,durations)):
            print(f'  {ft}  sev={sev:.0%}  dur={dur}d ...',end=' ')
            det_fv=0; det_gs={a:0 for a in GS_ALPHAS}
            adv_fv=[]; adv_gs={a:[] for a in GS_ALPHAS}
            fa_fv=[]; fa_gs={a:[] for a in GS_ALPHAS}

            for seed_i,name in enumerate(selected):
                ps=stations[name]
                gs_local=G_series.reindex(ps.index,method='nearest',
                                          tolerance=pd.Timedelta('90min')) \
                          if G_series is not None else None

                r=run_experiment(ps,gs_local,ft,sev,dur,name,
                                 exp_id=f'{ft}_{sev}_{name}',seed=RANDOM_SEED+seed_i)
                all_results.append(r)

                det_fv+=r['fv']['detected']
                if r['fv']['advance_days']: adv_fv.append(r['fv']['advance_days'])
                fa_fv.append(r['fv']['fa_rate_mo'])

                for a in GS_ALPHAS:
                    key=f'alpha_{a:.2f}'
                    det_gs[a]+=r['gs'][key]['detected']
                    if r['gs'][key]['advance_days']: adv_gs[a].append(r['gs'][key]['advance_days'])
                    fa_gs[a].append(r['gs'][key]['fa_rate_mo'])

                # Plot esempio: primo esperimento per tipo
                if example_plots_done<3 and seed_i==0:
                    P_inj,fs,fe=inject_fault(ps,gs_local,ft,sev,dur,seed=RANDOM_SEED)
                    alarm_fv,_=run_symphonon_fv(P_inj,gs_local)
                    alarm_gs,pr_gs,pr_ema_gs=gold_standard_pr(P_inj,gs_local,alpha=0.85)
                    plot_example_fault(ps,P_inj,alarm_fv,alarm_gs,pr_gs,pr_ema_gs,
                                       fs,fe,name,ft,sev,out_dir)
                    example_plots_done+=1

            n=len(selected)
            print(f'FV={det_fv/n:.0%}  GS_0.85={det_gs[0.85]/n:.0%}  '
                  f'FV_adv={np.mean(adv_fv):.1f}d  GS_adv={np.mean(adv_gs[0.85]):.1f}d')

    # ── Statistiche aggregate ─────────────────────────────────────────────────
    print('\n[3/4] Statistiche...')
    summary={}
    for ft in FAULT_TYPES:
        exp_f=[r for r in all_results if r['fault_type']==ft]
        if not exp_f: continue
        fv_dr=np.mean([r['fv']['detected'] for r in exp_f])
        fv_adv=np.nanmean([r['fv']['advance_days'] or np.nan for r in exp_f])
        fv_fa=np.mean([r['fv']['fa_rate_mo'] for r in exp_f])
        summary[ft]={'fv':{'dr':fv_dr,'adv_days':fv_adv,'fa':fv_fa},'gs':{}}
        print(f'\n  {ft}:')
        print(f'    Symphonon FV:  DR={fv_dr:.0%}  adv={fv_adv:.1f}d  FA={fv_fa:.3f}/mo')
        for alpha in GS_ALPHAS:
            key=f'alpha_{alpha:.2f}'
            gs_dr=np.mean([r['gs'][key]['detected'] for r in exp_f])
            gs_adv=np.nanmean([r['gs'][key]['advance_days'] or np.nan for r in exp_f])
            gs_fa=np.mean([r['gs'][key]['fa_rate_mo'] for r in exp_f])
            summary[ft]['gs'][key]={'dr':gs_dr,'adv_days':gs_adv,'fa':gs_fa}
            print(f'    GS α={alpha}: DR={gs_dr:.0%}  adv={gs_adv:.1f}d  FA={gs_fa:.3f}/mo')

    # ── Plot ─────────────────────────────────────────────────────────────────
    print('\n[4/4] Plot...')
    plot_roc_curve(all_results,out_dir)
    plot_advance_warning(all_results,out_dir)

    # ── Output JSON ───────────────────────────────────────────────────────────
    output={'protocol':'Fault injection + Gold Standard comparison',
            'n_experiments':len(all_results),
            'n_stations':n_exp_stations,
            'fault_types':list(FAULT_TYPES.keys()),
            'gs_alphas':GS_ALPHAS,
            'adv_window_days':ADV_WINDOW,
            'summary':summary,
            'all_results':all_results}
    with open(out_dir/'results_fi.json','w') as f:
        json.dump(output,f,indent=2,default=str)

    print('\n  ✓ results_fi.json  fig_roc_comparison.png  fig_advance_warning.png')
    print('═'*72)


if __name__=='__main__':
    main()
