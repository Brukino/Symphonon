"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYMPHONON FV v2 — Predictive Maintenance Fotovoltaico                     ║
║  Parser adattato alla struttura reale dataset HKUST Dryad                  ║
║                                                                              ║
║  Struttura confermata:                                                      ║
║    Dataset\Time series dataset\PV generation dataset\                       ║
║      PV stations without panel level optimizer\Site level dataset\          ║
║        CYT (Station 1).csv  ...  colonne: Time|generation(kWh)|power(W)    ║
║      PV stations with panel level optimizer\Inverter level dataset\         ║
║        ...  (44 stazioni)                                                   ║
║    Meteo: Time | Irradiance (W/m2)  risoluzione 1min                       ║
║                                                                              ║
║  ZERO TUNING — pesi e soglie identici a Symphonon P v5w2                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

ESECUZIONE:
  python symphonon_fv_runner.py --data "D:\\...\\Dataset\\Dataset" --out output_fv
  python symphonon_fv_runner.py --data "..." --inspect-only
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, sys, warnings, json, argparse, re
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Parametri (scalati da 10min a 1h, struttura identica) ────────────────────
WIN  = 24;  STEP = 6       # 24h window, 6h stride
RW   = 7;   RW_PR = 14
THR_P = 0.72
L_PERS_A  = 8;  THR_PERS_A = 0.12; THR_VEL_AMP = 0.08
L_VEL_SMOOTH = 4; VEL_SHORT = 2; VEL_LONG = 14
L_PERS_B  = 20; THR_PERS_BASE = 0.62; PERS_FLOOR = 0.10
ALPHA     = 0.80; L_VEL = 30
WARMUP_WIN = 60; SEASON_EMA = 90; SEASON_INIT_WIN = 60
K_CONFIRM = 4; MIN_OFF_WIN = 2
FLEET_MIN = 3; MIN_CLEAN_MONTHS = 2.0

# ── FV-specific ───────────────────────────────────────────────────────────────
G_NIGHT = 50.0; G_LOW = 200.0; G_HIGH = 600.0
CLOUD_ALPHA = 0.20; CLOUD_EMA = 90; CLOUD_SMOOTH = 12; CLOUD_CLIP = 2.0
THR_B_CEIL  = 0.85
DARK_BG = '#0d0d0d'

# ── Sottocartelle HKUST ───────────────────────────────────────────────────────
PV_GEN   = "Time series dataset/PV generation dataset"
NO_OPT   = "PV stations without panel level optimizer/Site level dataset"
WITH_OPT = "PV stations with panel level optimizer/Inverter level dataset"


# ══════════════════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════════════════

def find_pv_csvs(data_dir):
    base = Path(data_dir)
    found = []

    # Cerca ricorsivamente le due cartelle target
    no_opt_folder = None
    with_opt_folder = None

    for d in base.rglob('Site level dataset'):
        if 'without' in str(d).lower():
            no_opt_folder = d; break

    for d in base.rglob('Inverter level dataset'):
        with_opt_folder = d; break

    if no_opt_folder and no_opt_folder.exists():
        csvs = sorted(no_opt_folder.glob('*.csv'))
        found.extend(csvs)
        print(f'  Senza optimizer: {len(csvs)} stazioni')

    if with_opt_folder and with_opt_folder.exists():
        csvs = sorted(with_opt_folder.glob('*.csv'))
        found.extend(csvs)
        print(f'  Con optimizer (inverter): {len(csvs)} stazioni')

    if not found:
        print('  Fallback ricorsivo...')
        found = sorted([f for f in base.rglob('*.csv')
                        if not any(k in f.name.lower()
                                   for k in ['irrad','weather','meteo','readme'])])
        print(f'  Trovati: {len(found)} CSV')
    return found


def load_pv_csv(path):
    """Carica CSV stazione HKUST. Colonne: Time | generation(kWh) | power(W)."""
    try:
        df = pd.read_csv(str(path), low_memory=False)
        if df.empty or len(df.columns) < 2: return None

        # Timestamp (colonna 0)
        tc = df.columns[0]
        df[tc] = pd.to_datetime(df[tc], errors='coerce')
        df = df.dropna(subset=[tc]).set_index(tc).sort_index()

        # Colonna potenza: preferisce power(W) su generation(kWh)
        pc = None
        for c in df.columns:
            if 'power' in c.lower() or c.lower() == 'w':
                pc = c; break
        if pc is None:
            for c in df.columns:
                if 'gen' in c.lower() or 'kwh' in c.lower():
                    pc = c; break
        if pc is None:
            num = df.select_dtypes(include=[np.number]).columns
            if not len(num): return None
            pc = num[0]

        power = pd.to_numeric(df[pc], errors='coerce').clip(lower=0)

        # Se kWh orario → converti in W
        if 'kwh' in pc.lower():
            power = power * 1000.0

        p99 = power.quantile(0.999)
        if p99 > 0: power = power.clip(upper=p99 * 1.5)

        # Resample a 1h
        s = power.resample('1h').mean().dropna()
        return s if len(s) >= WIN * 3 else None
    except:
        return None


def find_irradiance_csv(data_dir):
    """Cerca CSV irradianza (Time | Irradiance W/m2, risoluzione 1min)."""
    for f in Path(data_dir).rglob('*.csv'):
        if any(k in f.name.lower() for k in ['irrad','weather','meteo','solar']):
            return f
    # Controlla header
    for f in sorted(Path(data_dir).rglob('*.csv'))[:15]:
        try:
            h = pd.read_csv(str(f), nrows=0).columns.tolist()
            if any('irrad' in c.lower() for c in h): return f
        except: pass
    return None


def load_irradiance(irrad_path, t_index):
    if irrad_path is None: return None
    try:
        df = pd.read_csv(str(irrad_path), low_memory=False)
        tc = df.columns[0]
        df[tc] = pd.to_datetime(df[tc], errors='coerce')
        df = df.dropna(subset=[tc]).set_index(tc).sort_index()
        gc = next((c for c in df.columns
                   if any(k in c.lower() for k in ['irrad','ghi','solar','rad'])),
                  df.columns[0])
        G = pd.to_numeric(df[gc], errors='coerce').clip(0, 1500)
        G_1h = G.resample('1h').mean()
        G_aln = G_1h.reindex(t_index, method='nearest', tolerance=pd.Timedelta('90min'))
        cov = G_aln.notna().mean()
        print(f'  {irrad_path.name}: coverage={cov:.0%} '
              f'mean={G_aln.mean():.1f} max={G_aln.max():.1f} W/m²')
        return G_aln if cov > 0.20 else None
    except Exception as e:
        print(f'  WARN irradianza: {e}'); return None


# ══════════════════════════════════════════════════════════════════════════════
#  SENSORI E REGIME
# ══════════════════════════════════════════════════════════════════════════════

def build_common_index(stations):
    all_t = set()
    for s in stations.values(): all_t.update(s.index.tolist())
    return pd.DatetimeIndex(sorted(all_t)).round('1h').unique().sort_values()


def build_station_array(power_series, G_series, t_index):
    P = power_series.reindex(t_index, method='nearest',
                              tolerance=pd.Timedelta('90min')).fillna(0.0).clip(0)
    P_arr = P.values

    if G_series is not None:
        G_arr = G_series.reindex(t_index, method='nearest',
                                  tolerance=pd.Timedelta('90min')
                                  ).fillna(0.0).clip(0, 1500).values
    else:
        # Stima da ora del giorno
        h = pd.DatetimeIndex(t_index).hour.values
        G_arr = np.where((h >= 6) & (h <= 19),
                         np.clip(P_arr / (P_arr.max() + 1e-4) * 800, 50, 1000), 0.0)

    day_mask = G_arr > G_NIGHT

    # Performance Ratio
    P_peak = np.nanpercentile(P_arr[day_mask], 98) if day_mask.any() else 1.0
    if P_peak < 1e-3: P_peak = 1.0
    G_safe = np.where(G_arr > G_NIGHT, G_arr, np.nan)
    PR = np.where(G_arr > G_NIGHT,
                  P_arr / (G_safe / 1000.0 * P_peak + 1e-6), np.nan)
    PR = np.clip(PR, 0, 1.5)
    P_norm = np.clip(P_arr / (P_peak + 1e-6), 0, 1.2)

    # EMA trend PR
    PR_f = np.where(np.isnan(PR), 0.0, PR)
    alpha = 2 / 25; PR_ema = np.zeros_like(PR_f); PR_ema[0] = PR_f[0]
    for i in range(1, len(PR_f)):
        PR_ema[i] = alpha * PR_f[i] + (1 - alpha) * PR_ema[i - 1]

    return np.column_stack([PR, P_norm, PR_ema]), day_mask, G_arr


def irrad_regime(G_arr):
    reg = np.where(G_arr < G_LOW, 0, np.where(G_arr < G_HIGH, 1, 2))
    reg[G_arr < G_NIGHT] = -1
    return reg


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE INVARIANTI (identica a Symphonon P)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(a, w):
    a = np.asarray(a, float); alpha = 2/(w+1)
    o = np.zeros_like(a); o[0] = a[0]
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

def _pers(pb,thr=THR_P,L=None):
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
            if oc>=MIN_OFF_WIN: ia=False
            st[i]=0
    return al,st


def compute_inv(arr, regime, t_index, day_mask):
    T,N=arr.shape
    di=np.where(day_mask)[0]
    if len(di)<WIN*3: return None
    bl=arr[di[:WIN*3]].mean(0)
    ti=[]; nr=[]; kr=[]; cr=[]; prr=[]; cir=[]; rw=[]
    for s in range(0,T-WIN,STEP):
        W=arr[s:s+WIN]; rs=regime[s:s+WIN]; dm=day_mask[s:s+WIN]
        if dm.mean()<0.25: continue
        Wd=W[dm]; rd=rs[dm]
        if len(Wd)<5 or np.isnan(Wd).mean()>0.4: continue
        ti.append(s+WIN//2)
        ts=np.nanstd(Wd,0)
        kap=float(np.clip(1-np.nanmean(ts)/(np.nanstd(ts)+1e-8),0,1))
        nr.append(_nd(Wd,bl)); kr.append(kap)
        cr.append(_pc1(Wd)); prr.append(_pr(Wd)); cir.append(_ci(Wd,rd))
        rw.append(int(regime[s+WIN//2]) if regime[s+WIN//2]>=0 else 1)
    if len(ti)<10: return None
    t=np.array(ti); tdt=t_index[t.clip(0,len(t_index)-1)]
    return dict(t_dt=tdt,t_idx=t,n_raw=np.array(nr),k_raw=np.array(kr),
                c_raw=np.array(cr),pr_raw=np.array(prr),ci_raw=np.array(cir),
                regs_w=np.array(rw))


def fleet_dev(inv_by_n):
    all_t=sum([list(v['t_dt']) for v in inv_by_n.values()],[])
    if not all_t: return {n:np.zeros(len(v['n_raw'])) for n,v in inv_by_n.items()}
    grid=pd.date_range(min(all_t),max(all_t),freq='6h')
    gv={n:pd.Series(v['n_raw'],index=v['t_dt']).reindex(
            grid,method='nearest',tolerance=pd.Timedelta('4h')).values
        for n,v in inv_by_n.items()}
    mat=np.vstack(list(gv.values())); nv=(~np.isnan(mat)).sum(0)
    fm=np.full(len(grid),np.nan); ok=nv>=FLEET_MIN
    if ok.any(): fm[ok]=np.nanmedian(mat[:,ok],axis=0)
    def gd(n,inv):
        fma=pd.Series(fm,index=grid).reindex(
            inv['t_dt'],method='nearest',tolerance=pd.Timedelta('4h')).values
        return np.where(np.isnan(fma),0.,inv['n_raw']-fma)
    return {n:gd(n,v) for n,v in inv_by_n.items()}


def cloud_z(G_series,t_index,t_idx):
    n=len(t_idx)
    if G_series is None: return np.zeros(n)
    G=G_series.reindex(t_index[t_idx.clip(0,len(t_index)-1)],
                       method='nearest',tolerance=pd.Timedelta('90min')
                       ).values.copy()
    G[np.isnan(G)]=np.nanmean(G) if np.any(~np.isnan(G)) else 300.
    G=np.clip(G,0,1500)
    CV=np.zeros(n)
    for i in range(6,n):
        seg=G[max(0,i-6):i]; mu=seg.mean()
        CV[i]=seg.std()/(mu+1e-3) if mu>G_NIGHT else 0.
    bl=_ema(CV,CLOUD_EMA); std=np.nanstd(CV[:min(60,n)])+1e-4
    return _ema(np.clip((CV-bl)/std,0.,CLOUD_CLIP),CLOUD_SMOOTH)


def signals(inv,fdev,cz=None):
    n=inv['n_raw']; k=inv['k_raw']; c=inv['c_raw']
    pr=inv['pr_raw']; ci=inv['ci_raw']; rw=inv['regs_w']; tdt=inv['t_dt']
    nrs=_rsig(n,RW); prrs=_rsig(_ema(pr,3),RW_PR); cirs=_rsig(ci,RW)
    Pb=0.45*nrs+0.30*prrs+0.25*cirs
    na=_rdet(n,rw); ps=_pers(Pb,L=L_PERS_A); va=_vamp(na)
    gA=((ps>THR_PERS_A)&(va>THR_VEL_AMP)).astype(float)
    pl=_pers(Pb,L=L_PERS_B); vs=_svp(fdev)
    czz=np.zeros(len(vs))
    if cz is not None and len(cz)==len(vs): czz=cz
    thr=np.minimum(np.maximum(THR_PERS_BASE-ALPHA*vs+CLOUD_ALPHA*czz,PERS_FLOOR),THR_B_CEIL)
    gB=(pl>thr).astype(float)
    al,st=_hdet(Pb,np.clip(gA+gB,0,1))
    return dict(t_dt=tdt,P_base=Pb,noise=nrs,pr=prrs,cintra=cirs,
                pers_short=ps,vel_amp=va,gate_A=gA,pers_long=pl,
                vel_sign=vs,gate_B=gB,thr_b_eff=thr,cloud_z=czz,
                fleet_dev=fdev,alarm=al,stage=st)

count_fa    = lambda tdt,al: sum(1 for a,p in zip(al,np.roll(al,1)) if a and not p)
clean_months= lambda tdt: round((tdt[-1]-tdt[0]).total_seconds()/(86400*30.44),2) if len(tdt) else 0.


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════

def _sty(ax):
    ax.set_facecolor('#111'); ax.tick_params(colors='#888',labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#333')

def plot_station(name,sid,sig,out_dir):
    t=sig['t_dt']
    fig,axs=plt.subplots(3,1,figsize=(16,8),gridspec_kw={'height_ratios':[1.2,1.2,.8]})
    fig.patch.set_facecolor(DARK_BG)
    for ax in axs: _sty(ax)
    axs[0].plot(t,sig['noise'],color='#ff8844',lw=.8,label='noise_drift')
    axs[0].plot(t,sig['pr'],color='#44ccff',lw=.9,label='PR_inv')
    axs[0].plot(t,sig['cintra'],color='#aa66ff',lw=.9,label='compress')
    axs[0].set_ylim(-.05,1.1); axs[0].legend(fontsize=7,labelcolor='#ccc',framealpha=.2,ncol=3)
    axs[1].plot(t,sig['P_base'],color='#44ff88',lw=1.2,label='P_base')
    axs[1].plot(t,sig['thr_b_eff'],color='#ff66cc',lw=.9,ls='--',alpha=.7,label='thr_B')
    axs[1].axhline(THR_P,color='#44ff88',lw=.7,ls=':',alpha=.5)
    axs[1].set_ylim(-.1,1.1); axs[1].legend(fontsize=7,labelcolor='#ccc',framealpha=.2,ncol=3)
    sc={0:'#222',1:'#335533',2:'#555500',3:'#883300'}
    tv=t.values
    for i in range(len(tv)-1):
        axs[2].axvspan(tv[i],tv[i+1],color=sc.get(int(sig['stage'][i]),'#222'),alpha=.8)
    axs[2].set_ylim(0,1); axs[2].set_yticks([])
    axs[0].set_title(f'{name} — Symphonon FV (HKUST)',color='#ddd',fontsize=9)
    safe=re.sub(r'[^\w]','_',name)[:25]
    plt.tight_layout()
    plt.savefig(out_dir/f'fig_{sid:02d}_{safe}.png',dpi=110,bbox_inches='tight',facecolor=DARK_BG)
    plt.close()

def plot_summary(fa_all,names,out_dir):
    REF=[('Kelmarsh v4.1',0.905,'#ff8844'),('Penmanshiel v4.1',0.183,'#44cc88'),
         ('Nørrekær v5-full',0.368,'#aa66ff')]
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axs: _sty(ax)
    fig.suptitle(f'Symphonon FV — HKUST ({len(fa_all)} stazioni) — Zero Tuning',
                  color='#ddd',fontsize=11)
    axs[0].bar(range(len(fa_all)),fa_all,color='#4488cc',alpha=.85,edgecolor='#333')
    for lbl,val,col in REF: axs[0].axhline(val,color=col,lw=1.2,ls='--',label=f'{lbl} {val:.2f}')
    axs[0].axhline(.10,color='#00ff88',lw=1,ls=':',alpha=.7,label='Target 0.10')
    axs[0].set_xticks(range(len(fa_all)))
    axs[0].set_xticklabels([n[:12] for n in names],fontsize=5,rotation=90)
    axs[0].set_ylabel('FA/mese',color='#888'); axs[0].legend(fontsize=7,labelcolor='#ccc',framealpha=.2)
    fam=np.mean(fa_all)
    axs[1].hist(fa_all,bins=min(15,len(fa_all)),color='#4488cc',alpha=.85,edgecolor='#333')
    axs[1].axvline(fam,color='#ff4444',lw=2,label=f'HKUST FV {fam:.3f}')
    for lbl,val,col in REF: axs[1].axvline(val,color=col,lw=1.2,ls='--',alpha=.7,label=lbl)
    axs[1].axvline(.10,color='#00ff88',lw=1,ls=':',alpha=.7)
    axs[1].set_xlabel('FA/mese',color='#888'); axs[1].legend(fontsize=7,labelcolor='#ccc',framealpha=.2)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_summary_fv.png',dpi=150,bbox_inches='tight',facecolor=DARK_BG)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='.',
                    help='Cartella radice HKUST (contiene "Time series dataset")')
    ap.add_argument('--out',default='output_fv')
    ap.add_argument('--inspect-only',action='store_true')
    args=ap.parse_args()

    data_dir=Path(args.data); out_dir=Path(args.out)
    out_dir.mkdir(parents=True,exist_ok=True)

    print('═'*72)
    print('  SYMPHONON FV v2 — HKUST Rooftop PV (Hong Kong 2021-2023)')
    print(f'  data: {data_dir}')
    print('═'*72)

    print('\n[1/5] Ricerca file...')
    irrad_path=find_irradiance_csv(data_dir)
    print(f'  Irradianza: {irrad_path.name if irrad_path else "non trovata"}')
    pv_csvs=find_pv_csvs(data_dir)

    if args.inspect_only:
        for f in pv_csvs[:3]:
            try:
                df=pd.read_csv(str(f),nrows=4)
                print(f'\n  {f.name}  colonne: {list(df.columns)}')
                print(df.to_string(index=False))
            except: pass
        sys.exit(0)

    if not pv_csvs: print('  ERRORE: nessun CSV trovato.'); sys.exit(1)

    print('\n[2/5] Caricamento stazioni...')
    stations={}
    for f in pv_csvs:
        s=load_pv_csv(f)
        if s is not None: stations[f.stem]=s
    print(f'  Caricate: {len(stations)}/{len(pv_csvs)}')
    if not stations: print('  ERRORE.'); sys.exit(1)

    t_index=build_common_index(stations)
    print(f'  Periodo: {t_index[0]} → {t_index[-1]}  ({len(t_index)} campioni 1h)')

    print('\n[3/5] Irradianza...')
    G_series=load_irradiance(irrad_path,t_index)
    if G_series is None: print('  Fallback: stima da ora del giorno')

    print('\n[4/5] Invarianti + fleet...')
    inv_by_n={}
    for name,ps in stations.items():
        arr,dm,G_arr=build_station_array(ps,G_series,t_index)
        if arr.shape[0]<WIN*3: continue
        di=np.where(dm)[0]
        if len(di)<WIN*3: continue
        mu=arr[di].mean(0); std=arr[di].std(0)+1e-8
        az=(arr-mu)/std
        reg=irrad_regime(G_arr)
        ac=az.copy()
        for r in [0,1,2]:
            m=reg==r
            if m.sum()>10: ac[m]-=az[m].mean(0)
        inv=compute_inv(ac,reg,t_index,dm)
        if inv: inv_by_n[name]=inv

    print(f'  Valide: {len(inv_by_n)}/{len(stations)}')
    if not inv_by_n: print('  ERRORE.'); sys.exit(1)

    fd=fleet_dev(inv_by_n)
    dm=[fd[n].mean() for n in inv_by_n]
    print(f'  fleet_dev spread={np.std(dm):.4f}  N={len(inv_by_n)} bias~{1/len(inv_by_n)*100:.1f}%')

    t0=list(inv_by_n.values())[0]['t_idx']
    cz_park=cloud_z(G_series,t_index,t0)
    print(f'  cloud_z: mean={cz_park.mean():.3f}  max={cz_park.max():.3f}')

    print('\n[5/5] Segnali e FA...')
    results={}; plots_done=0
    names_s=sorted(inv_by_n.keys())
    for sid,name in enumerate(names_s):
        inv=inv_by_n[name]; fdev=fd[name]
        cz=cz_park if len(cz_park)==len(fdev) else None
        sig=signals(inv,fdev,cz=cz)
        nm=clean_months(sig['t_dt']); fa=count_fa(sig['t_dt'],sig['alarm'])
        far=round(fa/nm,3) if nm>=MIN_CLEAN_MONTHS else None
        results[name]=dict(fa_ep=fa,n_months=nm,fa_rate=far,insufficient=(far is None))
        fs=f'{far:.3f}/mo' if far is not None else f'insuff({nm:.1f}mo)'
        print(f'  {name[:35]:<35}: FA={fs}  fdev={fdev.mean():+.4f}')
        if plots_done<6 or sid==len(names_s)-1:
            plot_station(name,sid,sig,out_dir); plots_done+=1

    valid={k:v for k,v in results.items() if not v['insufficient']}
    fa_all=[v['fa_rate'] for v in valid.values()]
    fam=float(np.mean(fa_all)) if fa_all else None
    REF={'kelmarsh_v41':0.905,'kelmarsh_v5w2':0.851,
         'penmanshiel_v41':0.183,'norrekaer_v5full':0.368}

    print('\n'+'─'*72)
    if fa_all:
        print(f'  FA medio:{fam:.3f}  med:{np.median(fa_all):.3f}  '
              f'min:{min(fa_all):.3f}  max:{max(fa_all):.3f}')
        print(f'  FA<0.10: {sum(1 for v in fa_all if v<0.10)}/{len(fa_all)}   '
              f'FA<0.25: {sum(1 for v in fa_all if v<0.25)}/{len(fa_all)}')
        for lbl,val in REF.items(): print(f'    {lbl}: {val:.3f}')
        print(f'    HKUST FV (N={len(valid)}): {fam:.3f}')
        plot_summary(fa_all,list(valid.keys()),out_dir)

    out={'dataset':'HKUST Rooftop PV','doi':'10.5061/dryad.m37pvmd99',
         'algorithm':'Symphonon FV v2',
         'params':{'WIN':WIN,'CLOUD_ALPHA':CLOUD_ALPHA,'THR_P':THR_P},
         'n_stations':len(stations),'n_valid':len(valid),
         'irradiance':G_series is not None,
         'fa_mean':round(fam,3) if fam else None,
         'fa_median':round(float(np.median(fa_all)),3) if fa_all else None,
         'fa_below_010':sum(1 for v in fa_all if v<0.10) if fa_all else 0,
         'fa_below_025':sum(1 for v in fa_all if v<0.25) if fa_all else 0,
         'reference_wind':REF,'station_details':results}
    with open(out_dir/'results_fv.json','w') as f: json.dump(out,f,indent=2,default=str)
    lines=['═'*72,'  SYMPHONON FV v2 — HKUST','═'*72,'',
           f'  N: {len(stations)}  valide: {len(valid)}',
           f'  Irradianza: {"sì" if G_series is not None else "no"}','']
    if fa_all:
        lines+=[f'  FA medio:  {fam:.3f}',f'  FA<0.10: {sum(1 for v in fa_all if v<0.10)}/{len(fa_all)}','',
                '  Kelmarsh v4.1:    0.905','  Penmanshiel v4.1: 0.183',
                '  Nørrekær v5-full: 0.368',f'  HKUST FV:         {fam:.3f}']
    lines+=['','═'*72]
    with open(out_dir/'summary_fv.txt','w',encoding='utf-8') as f: f.write('\n'.join(lines))
    print('  ✓ results_fv.json  summary_fv.txt  fig_summary_fv.png')
    print('═'*72)

if __name__=='__main__': main()
