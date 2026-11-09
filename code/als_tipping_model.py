from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root, least_squares
from scipy.stats import qmc, rankdata, pearsonr

OUT=Path(__file__).resolve().parents[1]; FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
SEED=20260825
rng=np.random.default_rng(SEED)

# Literature anchors are observable ALS quantities. Structural coefficients below are provisional proof-of-concept quantities.
ANCHORS={
    'alsfrs_max':48.0,
    'reference_alsfrs':40.0,
    'reference_fvc_pct':90.5,
    'proact_alsfrs_slope_pts_month':1.02,
    'proact_vc_slope_pp_month':2.24,
    'svc_slope_pp_month':2.70,
    'svc_slope_age65_pp_month':3.60,
    'fvc_prognostic_anchor_pct':80.0,
    'alsfrs_prognostic_anchor':38.0,
}

BASE=dict(
    kF=0.080, kFR=0.018,
    kR=0.075, kRF=0.015,
    kM0=0.020, kMF=0.080, kMR=0.100, kMM=0.090, kMH=0.100,
    kHF=0.035, kHR=0.045, kHM=0.100, kHd=0.012,
    theta0=0.340, thetaM=0.280, wF=0.34, wR=0.34, wH=0.32,
    h0=0.015, bM=1.8, bF=1.2, bR=1.5, bH=1.0,
)

RANGES={
    'kF':(0.055,0.110),'kFR':(0.010,0.028),'kR':(0.052,0.105),'kRF':(0.008,0.025),
    'kM0':(0.012,0.032),'kMF':(0.050,0.115),'kMR':(0.060,0.145),'kMM':(0.055,0.125),'kMH':(0.065,0.140),
    'kHF':(0.020,0.055),'kHR':(0.025,0.065),'kHM':(0.065,0.140),'kHd':(0.006,0.020),
    'theta0':(0.29,0.39),'thetaM':(0.20,0.36),
}

def clip01(x): return float(np.clip(x,0,1))

def rhs(t,y,p=BASE):
    F,R,M,H=y
    dF = -p['kF']*M*F - p['kFR']*(1-R)*F
    dR = -p['kR']*M*R - p['kRF']*(1-F)*R
    dM = p['kM0']*(1-M) + p['kMF']*(1-F)*(1-M) + p['kMR']*(1-R)*(1-M) + p['kMM']*M*(1-M) - p['kMH']*H*M
    dH = p['kHF']*F*(1-H) + p['kHR']*R*(1-H) - p['kHM']*M*H - p['kHd']*H
    return np.array([dF,dR,dM,dH])

def margin(F,R,M,H,p=BASE):
    return p['wF']*F+p['wR']*R+p['wH']*H-p['theta0']-p['thetaM']*M

def hazard(F,R,M,H,p=BASE):
    return p['h0']*np.exp(p['bM']*M+p['bF']*(1-F)+p['bR']*(1-R)-p['bH']*H)

def simulate(p=BASE,y0=None,horizon=18.0,dt=0.05):
    if y0 is None: y0=[ANCHORS['reference_alsfrs']/48, ANCHORS['reference_fvc_pct']/100, 0.22, 0.78]
    tt=np.arange(0,horizon+1e-9,dt)
    sol=solve_ivp(lambda t,y: rhs(t,y,p),(0,horizon),y0,t_eval=tt,rtol=1e-8,atol=1e-10,method='RK45')
    F,R,M,H=sol.y
    psi=margin(F,R,M,H,p); hz=hazard(F,R,M,H,p)
    return sol.t,F,R,M,H,psi,hz

def first_crossing(t,psi):
    z=np.where(psi<=0)[0]
    return float(t[z[0]]) if len(z) else np.nan

def regime(psi):
    if psi>0.25: return 'Preserved reserve'
    if psi>0.10: return 'Compensated'
    if psi>0.00: return 'Transitional'
    return 'Decompensated'

# Reference trajectory
tr=simulate(); t,F,R,M,H,psi,hz=tr
ref=pd.DataFrame({'month':t,'ALSFRS_R':F*48,'FVC_pct':R*100,'disease_momentum':M,'homeostatic_reserve':H,'homeostatic_margin':psi,'relative_transition_hazard':hz/hz[0]})
ref['regime']=[regime(x) for x in psi]
ref.to_csv(OUT/'Reference_Trajectory.csv',index=False)

# Scenarios
scenarios={
    'Slow progression':(dict(kF=0.055,kR=0.052),[42/48,.95,.16,.84]),
    'PRO-ACT-aligned reference':({},[40/48,.905,.22,.78]),
    'Respiratory vulnerable':(dict(kR=0.105,kMR=0.125),[40/48,.80,.28,.70]),
    'Fast functional progression':(dict(kF=0.110,kMF=0.105),[38/48,.88,.30,.70]),
    'Low combined reserve':(dict(kF=0.095,kR=0.095),[34/48,.70,.32,.62]),
}
rows=[]
for name,(upd,y0) in scenarios.items():
    p=BASE.copy();p.update(upd)
    tt,FF,RR,MM,HH,PP,HZ=simulate(p,y0)
    i6=np.argmin(abs(tt-6));i12=np.argmin(abs(tt-12));i18=np.argmin(abs(tt-18))
    rows.append(dict(scenario=name,tipping_month=first_crossing(tt,PP),ALSFRS_6m=FF[i6]*48,FVC_6m=RR[i6]*100,ALSFRS_12m=FF[i12]*48,FVC_12m=RR[i12]*100,terminal_ALSFRS=FF[i18]*48,terminal_FVC=RR[i18]*100,terminal_momentum=MM[i18],terminal_reserve=HH[i18],terminal_margin=PP[i18],hazard_multiplier_18m=HZ[i18]/HZ[0],ALSFRS_slope_0_6=(FF[i6]-FF[0])*48/6,FVC_slope_0_6=(RR[i6]-RR[0])*100/6))
sc=pd.DataFrame(rows);sc.to_csv(OUT/'Scenario_Results.csv',index=False)

# Baseline ALSFRS-R x FVC tipping surface
frs_grid=np.arange(28,45,1); fvc_grid=np.arange(50,101,2)
surf=[]
for frs0 in frs_grid:
    for fvc0 in fvc_grid:
        tt,FF,RR,MM,HH,PP,HZ=simulate(BASE,[frs0/48,fvc0/100,.22,.78],18,0.1)
        tc=first_crossing(tt,PP)
        surf.append((frs0,fvc0,tc,PP[np.argmin(abs(tt-12))],PP[-1]))
surf=pd.DataFrame(surf,columns=['baseline_ALSFRS_R','baseline_FVC_pct','tipping_month','margin_12m','margin_18m']);surf.to_csv(OUT/'ALSFRS_FVC_Tipping_Surface.csv',index=False)

# Critical FVC continuation for selected ALSFRS-R values: minimum baseline FVC avoiding tipping by 12 months
crit=[]
for frs0 in np.arange(30,45,2):
    vals=[]
    for fvc0 in np.arange(40,101,1):
        tt,FF,RR,MM,HH,PP,HZ=simulate(BASE,[frs0/48,fvc0/100,.22,.78],12,0.1)
        vals.append((fvc0,PP[-1]))
    ok=[x for x in vals if x[1]>=0]
    crit_fvc=min([x[0] for x in ok]) if ok else np.nan
    crit.append((frs0,crit_fvc))
pd.DataFrame(crit,columns=['baseline_ALSFRS_R','model_critical_FVC_pct_for_nonnegative_12m_margin']).to_csv(OUT/'Tipping_Boundary_Continuation.csv',index=False)

# Global sensitivity LHS on structural parameters
keys=list(RANGES); n_lhs=500
u=qmc.LatinHypercube(d=len(keys),seed=SEED).random(n_lhs)
X=np.zeros_like(u)
for j,k in enumerate(keys): X[:,j]=RANGES[k][0]+u[:,j]*(RANGES[k][1]-RANGES[k][0])
lhs=pd.DataFrame(X,columns=keys);lhs.to_csv(OUT/'LHS_Parameter_Samples.csv',index=False)
outputs=[]
for i,row in lhs.iterrows():
    p=BASE.copy();p.update(row.to_dict())
    tt,FF,RR,MM,HH,PP,HZ=simulate(p,horizon=18,dt=.1)
    outputs.append((PP[-1],first_crossing(tt,PP),FF[-1]*48,RR[-1]*100,MM[-1],HH[-1]))
out=pd.DataFrame(outputs,columns=['terminal_margin','tipping_month','terminal_ALSFRS_R','terminal_FVC_pct','terminal_momentum','terminal_reserve'])
out.to_csv(OUT/'LHS_Model_Outputs.csv',index=False)
# PRCC using rank residualization against all other ranked parameters
from sklearn.linear_model import LinearRegression
pr=[]
y=rankdata(out.terminal_margin.values)
XR=np.column_stack([rankdata(lhs[k].values) for k in keys])
for j,k in enumerate(keys):
    others=[q for q in range(len(keys)) if q!=j]
    rx=XR[:,j]-LinearRegression().fit(XR[:,others],XR[:,j]).predict(XR[:,others])
    ry=y-LinearRegression().fit(XR[:,others],y).predict(XR[:,others])
    r,pv=pearsonr(rx,ry);pr.append((k,r,pv))
pr=pd.DataFrame(pr,columns=['parameter','PRCC_terminal_margin','p_value']).sort_values('PRCC_terminal_margin',key=lambda s:s.abs(),ascending=False);pr.to_csv(OUT/'Global_Sensitivity_PRCC.csv',index=False)

# Local elasticity +/-1%
base_terminal=psi[-1]
el=[]
for k in keys:
    if k not in BASE: continue
    p=BASE.copy();p[k]*=1.01
    *_,PP,_=simulate(p)
    e=((PP[-1]-base_terminal)/(abs(base_terminal)+1e-9))/0.01
    el.append((k,e))
pd.DataFrame(el,columns=['parameter','local_elasticity_terminal_margin']).sort_values('local_elasticity_terminal_margin',key=lambda s:s.abs(),ascending=False).to_csv(OUT/'Local_Elasticity.csv',index=False)

# Stochastic perturbation using Euler-Maruyama around structural dynamics. Noise is illustrative, not estimated from patients.
def stochastic_path(y0=None,horizon=18,dt=.05,sig=(.005,.006,.015,.012)):
    if y0 is None:y0=np.array([40/48,.905,.22,.78],float)
    n=int(horizon/dt)+1; Y=np.zeros((n,4));Y[0]=y0; tt=np.linspace(0,horizon,n)
    for i in range(1,n):
        drift=rhs(tt[i-1],Y[i-1],BASE)
        noise=np.array(sig)*math.sqrt(dt)*rng.normal(size=4)
        Y[i]=np.clip(Y[i-1]+drift*dt+noise,0,1)
    pp=margin(Y[:,0],Y[:,1],Y[:,2],Y[:,3],BASE)
    return tt,Y,pp
strows=[]
for i in range(500):
    tt,Y,PP=stochastic_path()
    tc=first_crossing(tt,PP)
    cls='No crossing by 18m' if np.isnan(tc) else ('Early crossing <=12m' if tc<=12 else 'Late crossing 12-18m')
    strows.append((i,tc,PP[-1],Y[-1,0]*48,Y[-1,1]*100,Y[-1,2],Y[-1,3],cls))
st=pd.DataFrame(strows,columns=['simulation','tipping_month','terminal_margin','terminal_ALSFRS_R','terminal_FVC_pct','terminal_momentum','terminal_reserve','class']);st.to_csv(OUT/'Stochastic_Perturbation_Results.csv',index=False)

# Frozen-observable equilibrium of latent M-H subsystem; this is a quasi-equilibrium, not a stationary ALS disease state.
def latent_rhs(z,F0,R0,p=BASE):
    M,H=z
    dM=p['kM0']*(1-M)+p['kMF']*(1-F0)*(1-M)+p['kMR']*(1-R0)*(1-M)+p['kMM']*M*(1-M)-p['kMH']*H*M
    dH=p['kHF']*F0*(1-H)+p['kHR']*R0*(1-H)-p['kHM']*M*H-p['kHd']*H
    return np.array([dM,dH])
def jac2(z,F0,R0,p=BASE,eps=1e-6):
    J=np.zeros((2,2));f0=latent_rhs(z,F0,R0,p)
    for j in range(2):
        zz=z.copy();zz[j]+=eps;J[:,j]=(latent_rhs(zz,F0,R0,p)-f0)/eps
    return J
anchors=[('Preserved',42/48,.95),('Compensated',38/48,.80),('Depleted',30/48,.60)]
eq=[];eig=[]
for name,F0,R0 in anchors:
    rr=root(lambda z:latent_rhs(z,F0,R0),[.3,.7])
    z=np.clip(rr.x,0,1);vals=np.linalg.eigvals(jac2(z,F0,R0))
    eq.append((name,F0*48,R0*100,z[0],z[1],max(vals.real)))
    for v in vals:eig.append((name,v.real,v.imag))
pd.DataFrame(eq,columns=['anchor','ALSFRS_R_fixed','FVC_pct_fixed','momentum_eq','reserve_eq','dominant_real_eigenvalue']).to_csv(OUT/'Latent_Quasi_Equilibria.csv',index=False)
pd.DataFrame(eig,columns=['anchor','real','imag']).to_csv(OUT/'Latent_Quasi_Equilibrium_Eigenvalues.csv',index=False)

# Numerical robustness: RK45 vs DOP853 and step resolutions
rob=[]
for method in ['RK45','DOP853']:
    for max_step in [1.0,.5,.25,.1]:
        y0=[40/48,.905,.22,.78]; tt=np.linspace(0,18,361)
        sol=solve_ivp(lambda t,y:rhs(t,y,BASE),(0,18),y0,t_eval=tt,method=method,max_step=max_step,rtol=1e-9,atol=1e-11)
        FF,RR,MM,HH=sol.y;PP=margin(FF,RR,MM,HH)
        rob.append((method,max_step,FF[-1]*48,RR[-1]*100,PP[-1],first_crossing(tt,PP)))
pd.DataFrame(rob,columns=['solver','max_step_month','terminal_ALSFRS_R','terminal_FVC_pct','terminal_margin','tipping_month']).to_csv(OUT/'Numerical_Robustness.csv',index=False)

# Synthetic AIC benchmark: monthly noisy pseudo-observations generated from the full structural model.
obs_t=np.arange(0,13,1.0)
full=simulate(BASE,[40/48,.905,.22,.78],12,0.05)
_,fF,fR,*_=full
trueF=np.interp(obs_t,full[0],fF)*48; trueR=np.interp(obs_t,full[0],fR)*100
obsF=trueF+rng.normal(0,.45,len(obs_t)); obsR=trueR+rng.normal(0,1.8,len(obs_t))
synthetic=pd.DataFrame({'month':obs_t,'ALSFRS_R_pseudoobs':obsF,'FVC_pct_pseudoobs':obsR,'ALSFRS_R_truth':trueF,'FVC_pct_truth':trueR});synthetic.to_csv(OUT/'Synthetic_Benchmark_Observations.csv',index=False)

def aic_from_residuals(res,k):
    n=len(res);rss=float(np.sum(np.square(res)));return n*np.log(rss/n+1e-12)+2*k,rss
# Candidate 1: full structural, fit kF,kR,kM0,theta0

def pred_full(par):
    p=BASE.copy();p['kF'],p['kR'],p['kM0'],p['theta0']=par
    simx=simulate(p,[40/48,.905,.22,.78],12,.05)
    FF=np.interp(obs_t,simx[0],simx[1])*48; RR=np.interp(obs_t,simx[0],simx[2])*100
    return FF,RR
res=least_squares(lambda x:np.r_[(pred_full(x)[0]-obsF)/.45,(pred_full(x)[1]-obsR)/1.8],[.08,.075,.02,.34],bounds=([.03,.03,.005,.20],[.15,.16,.06,.50]))
FF1,RR1=pred_full(res.x);rr1=np.r_[(FF1-obsF)/.45,(RR1-obsR)/1.8];a1,rss1=aic_from_residuals(rr1,4)
# Candidate 2: coupled exponential decline F=A exp(-a t), R=B exp(-b t-c(1-Fnorm)t)
def pred_exp(par):
    a,b,c=par; fn=np.exp(-a*obs_t); FF=obsF[0]*fn; RR=obsR[0]*np.exp(-(b+c*(1-fn))*obs_t);return FF,RR
res2=least_squares(lambda x:np.r_[(pred_exp(x)[0]-obsF)/.45,(pred_exp(x)[1]-obsR)/1.8],[.025,.025,.01],bounds=([0,0,0],[.2,.2,.2]));FF2,RR2=pred_exp(res2.x);rr2=np.r_[(FF2-obsF)/.45,(RR2-obsR)/1.8];a2,rss2=aic_from_residuals(rr2,3)
# Candidate 3: independent linear trends
coefF=np.polyfit(obs_t,obsF,1);coefR=np.polyfit(obs_t,obsR,1);FF3=np.polyval(coefF,obs_t);RR3=np.polyval(coefR,obs_t);rr3=np.r_[(FF3-obsF)/.45,(RR3-obsR)/1.8];a3,rss3=aic_from_residuals(rr3,4)
# Candidate 4: single latent reserve logistic curve sharing inflection
def pred_log(par):
    mF,sF,mR,sR,t0,g=par
    q=1/(1+np.exp(g*(obs_t-t0)))
    FF=mF+sF*q;RR=mR+sR*q;return FF,RR
res4=least_squares(lambda x:np.r_[(pred_log(x)[0]-obsF)/.45,(pred_log(x)[1]-obsR)/1.8],[20,20,35,55,8,.25],bounds=([0,0,0,0,0,.01],[48,48,100,100,18,2]));FF4,RR4=pred_log(res4.x);rr4=np.r_[(FF4-obsF)/.45,(RR4-obsR)/1.8];a4,rss4=aic_from_residuals(rr4,6)
mc=pd.DataFrame([('Full regime-switching structural model',a1,rss1,4),('Coupled exponential reserve model',a2,rss2,3),('Independent linear trajectory model',a3,rss3,4),('Shared-inflection latent reserve model',a4,rss4,6)],columns=['model','AIC_synthetic','standardized_RSS','fitted_parameters']).sort_values('AIC_synthetic');mc['delta_AIC']=mc.AIC_synthetic-mc.AIC_synthetic.min();mc.to_csv(OUT/'Synthetic_Model_Comparison_AIC.csv',index=False)

# Exploratory sequential monitoring policy (not treatment recommendation)
# Trigger closer respiratory/functional review when expected 3-month decompensation cost exceeds monitoring burden.
policy=[]
for frs0 in np.arange(30,45,1):
    for fvc0 in np.arange(50,101,2):
        tt,FF,RR,MM,HH,PP,HZ=simulate(BASE,[frs0/48,fvc0/100,.22,.78],3,.05)
        end=PP[-1]; risk=1/(1+np.exp(10*end))
        monitor_cost=.20; wait_cost=risk
        action='Escalated review' if wait_cost>monitor_cost else 'Routine review'
        policy.append((frs0,fvc0,end,risk,action))
pd.DataFrame(policy,columns=['baseline_ALSFRS_R','baseline_FVC_pct','margin_3m','theoretical_decompensation_score','exploratory_policy']).to_csv(OUT/'Exploratory_Sequential_Monitoring_Policy.csv',index=False)

# Summary
r6=ref.iloc[(ref.month-6).abs().argmin()];r12=ref.iloc[(ref.month-12).abs().argmin()];r18=ref.iloc[-1]
counts=st['class'].value_counts().to_dict()
summary={
    'seed':SEED,
    'reference':{
        'tipping_month':first_crossing(t,psi),'ALSFRS_6m':float(r6.ALSFRS_R),'FVC_6m':float(r6.FVC_pct),'ALSFRS_12m':float(r12.ALSFRS_R),'FVC_12m':float(r12.FVC_pct),'terminal_ALSFRS':float(r18.ALSFRS_R),'terminal_FVC':float(r18.FVC_pct),'terminal_margin':float(r18.homeostatic_margin),'hazard_multiplier_18m':float(r18.relative_transition_hazard),'ALSFRS_slope_0_6':float((r6.ALSFRS_R-ref.iloc[0].ALSFRS_R)/6),'FVC_slope_0_6':float((r6.FVC_pct-ref.iloc[0].FVC_pct)/6)
    },
    'uncertainty':{
        'terminal_margin_median':float(out.terminal_margin.median()),'terminal_margin_p2_5':float(out.terminal_margin.quantile(.025)),'terminal_margin_p97_5':float(out.terminal_margin.quantile(.975)),
        'tipping_by_18_fraction':float(out.tipping_month.notna().mean())
    },
    'stochastic_class_counts':counts,
    'top_prcc':pr.head(8).to_dict(orient='records'),
    'aic':mc.to_dict(orient='records'),
    'quasi_equilibria':pd.read_csv(OUT/'Latent_Quasi_Equilibria.csv').to_dict(orient='records')
}
(OUT/'Analysis_Summary.json').write_text(json.dumps(summary,indent=2))

# ---------------- Revised figures: all model outputs are unchanged ----------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import textwrap
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.dpi':140,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
PLUM='#805477'; TEAL='#3d827e'; ROSE='#b66c83'; INK='#354450'; LILAC='#9882b5'; SAND='#b58a64'
def save(name):
    plt.savefig(FIG/name,bbox_inches='tight',dpi=220,facecolor='white');plt.close()
def finish(ax):
    ax.grid(alpha=.13);ax.set_axisbelow(True)
def wiggle(x,y,scale=.004,phase=0):
    # Visualization-only deterministic small-scale texture; NOT new measurements.
    y=np.asarray(y,float);x=np.asarray(x,float)
    amp=max(np.ptp(y),.05)*scale
    return y+amp*(np.sin(9.3*x+phase)+.42*np.sin(21.7*x+phase*.8))
# Figure 1: single-hue serpentine flow; arrows routed outside boxes
fig,ax=plt.subplots(figsize=(16,8));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
items=[(.055,.69,.25,.16,'Clinical Anchors','ALSFRS-R | FVC/SVC\nLiterature progression anchors'),(.375,.69,.25,.16,'Continuous Dynamics','Functional F(t) | Respiratory R(t)\nMomentum M(t) | Homeostasis H(t)'),(.695,.69,.25,.16,'Tipping Geometry','Homeostatic margin Ψ(t)\nCompensation boundary'),(.695,.37,.25,.16,'Latent Regimes','Preserved → Compensated\nTransitional → Decompensated'),(.375,.37,.25,.16,'Risk and Decisions','Transition hazard\nExploratory monitoring'),(.055,.37,.25,.16,'Robustness and Data Science','LHS–PRCC | stochastic analysis\nSynthetic AIC | stability checks')]
for x,y,w,h,title,sub in items:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',fc='#805477',alpha=.13,ec='none'))
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',fc='none',ec=PLUM,lw=1.35))
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',fontsize=11,weight='bold',color=INK)
    ax.text(x+w/2,y+h*.30,sub,ha='center',va='center',fontsize=9.2,color=INK,linespacing=1.5)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.6,color=PLUM,connectionstyle='arc3,rad=0'))
arrow((.317,.77),(.363,.77));arrow((.637,.77),(.683,.77));arrow((.82,.678),(.82,.542));arrow((.683,.45),(.637,.45));arrow((.363,.45),(.317,.45))
ax.text(.5,.955,'Hybrid Regime Switching Architecture for ALS Functional Respiratory Reserve',ha='center',fontsize=16,weight='bold',color=INK)
ax.text(.5,.915,'Literature-parameterized proof-of-concept: no patient level clinical calibration',ha='center',fontsize=10.5,style='italic',color=PLUM)
ax.text(.5,.18,'Conceptual architecture • all transitions and decision boundaries are model-defined, not clinically validated',ha='center',fontsize=10,color=INK)
save('Figure_1_Model_Architecture.png')
# Figure 2: subtle deterministic display texture, no alterations to numerical results
fig,axs=plt.subplots(2,2,figsize=(13,9));axs=axs.ravel()
series=[(ref.ALSFRS_R,'ALSFRS-R Functional Trajectory','ALSFRS-R (0–48)',TEAL,38,'Published prognostic anchor 38'),(ref.FVC_pct,'Respiratory Reserve','FVC (% predicted)',PLUM,80,'Published prognostic anchor 80%')]
for i,(yy,ttl,yl,c,threshold,lab) in enumerate(series):
    axs[i].plot(ref.month,yy,color=c,lw=2.1,alpha=.78,label='ODE trajectory');axs[i].plot(ref.month,wiggle(ref.month,yy,.006,i+1),color=c,lw=.9,alpha=.4,label='Illustrative visual texture');axs[i].axhline(threshold,color=ROSE,ls='--',lw=1.2,label=lab);axs[i].set_title(ttl);axs[i].set_ylabel(yl);axs[i].legend(frameon=False,fontsize=8)
for yy,c,lab,ph in [(ref.disease_momentum,ROSE,'Disease momentum',1),(ref.homeostatic_reserve,TEAL,'Homeostatic reserve',2)]:
    axs[2].plot(ref.month,yy,color=c,lw=2.2,alpha=.82,label=lab);axs[2].plot(ref.month,wiggle(ref.month,yy,.012,ph),color=c,lw=.8,alpha=.4)
axs[2].set_title('Latent Dynamic States');axs[2].legend(frameon=False)
axs[3].plot(ref.month,ref.homeostatic_margin,color=PLUM,lw=2.4,label='Homeostatic margin Ψ');axs[3].plot(ref.month,wiggle(ref.month,ref.homeostatic_margin,.012,3),color=PLUM,lw=.8,alpha=.4)
axs[3].axhline(0,color=ROSE,ls='--',label='Model tipping boundary');axs[3].set_title('Homeostatic Margin');axs[3].legend(frameon=False)
for a in axs:a.set_xlabel('Months');finish(a)
fig.suptitle('Reference Literature-Anchored ALS Trajectory',fontsize=16,weight='bold');fig.tight_layout(rect=[0,.025,1,.96]);fig.text(.5,.005,'Thin textured overlays are illustrative only; underlying ODE solutions and CSV values are unchanged.',ha='center',fontsize=8.5,color=INK);save('Figure_2_Reference_Trajectory.png')
# Figure 3: nonlinear zero contour is the model-derived 12-month boundary
piv=surf.pivot(index='baseline_FVC_pct',columns='baseline_ALSFRS_R',values='margin_12m')
fig,ax=plt.subplots(figsize=(10.8,7));cmap=LinearSegmentedColormap.from_list('margin',['#b67d91','#f6f1ef','#6b9a93'])
im=ax.contourf(piv.columns,piv.index,piv.values,levels=24,cmap=cmap,alpha=.86)
cs=ax.contour(piv.columns,piv.index,piv.values,levels=[0],colors=[INK],linewidths=2.4)
if cs.allsegs[0]:ax.clabel(cs,fmt={0:'Ψ(12 months) = 0'},fontsize=9,inline=True)
ax.set_xlabel('Baseline ALSFRS-R');ax.set_ylabel('Baseline FVC (% predicted)');ax.set_title('Functional Respiratory Homeostatic Margin at 12 Months',weight='bold');fig.colorbar(im,ax=ax,label='Modeled margin Ψ at 12 months');finish(ax);save('Figure_3_ALSFRS_FVC_Tipping_Surface.png')
# Figure 4: use step-shaped discrete threshold rather than fabricated jagged data
B=pd.read_csv(OUT/'Tipping_Boundary_Continuation.csv');fig,ax=plt.subplots(figsize=(10,6))
ax.step(B.baseline_ALSFRS_R,B.model_critical_FVC_pct_for_nonnegative_12m_margin,where='mid',color=PLUM,lw=2.3,alpha=.83,label='Discrete grid-derived boundary');ax.scatter(B.baseline_ALSFRS_R,B.model_critical_FVC_pct_for_nonnegative_12m_margin,color=ROSE,s=55,alpha=.8,zorder=3)
ax.axhline(80,ls='--',color=TEAL,lw=1.4,label='80% published prognostic anchor');ax.set_xlabel('Baseline ALSFRS-R');ax.set_ylabel('Minimum baseline FVC (% predicted)');ax.set_title('Continuation of the 12-Month Compensation Boundary',weight='bold');finish(ax);ax.legend(frameon=False);save('Figure_4_Tipping_Boundary_Continuation.png')
# Figure 5
mat=sc.set_index('scenario')[['tipping_month','ALSFRS_12m','FVC_12m','terminal_margin','hazard_multiplier_18m']].rename(index={'Respiratory-vulnerable':'Respiratory vulnerable'})
norm=mat.copy()
for c in norm:norm[c]=(norm[c]-np.nanmin(norm[c]))/(np.nanmax(norm[c])-np.nanmin(norm[c])+1e-9)
fig,ax=plt.subplots(figsize=(13,6.5));cm=LinearSegmentedColormap.from_list('matrix',['#eee5e8','#bd8da1','#795777']);im=ax.imshow(norm.values,aspect='auto',cmap=cm,alpha=.91,vmin=0,vmax=1)
ax.set_xticks(range(5),['Tipping month','ALSFRS-R 12m','FVC 12m','Terminal margin','Hazard multiplier'],rotation=13,ha='right');ax.set_yticks(range(len(mat)),mat.index)
for i in range(len(mat)):
 for j in range(5):
    v=mat.iloc[i,j];ax.text(j,i,'—' if pd.isna(v) else f'{v:.2f}',ha='center',va='center',color='white' if norm.iloc[i,j]>.55 else INK,fontsize=9,weight='medium')
ax.set_title('Scenario Stress Test Across Functional and Respiratory Reserve',weight='bold');fig.colorbar(im,ax=ax,label='Within-column normalized value');save('Figure_5_Scenario_Stress_Test.png')
# Figure 6
P=pr.head(10).sort_values('PRCC_terminal_margin');fig,ax=plt.subplots(figsize=(10,6.5));ax.barh(P.parameter,P.PRCC_terminal_margin,color=[PLUM if v<0 else TEAL for v in P.PRCC_terminal_margin],alpha=.68,height=.68);ax.axvline(0,color=INK,lw=1);ax.set_xlabel('PRCC with terminal homeostatic margin');ax.set_title('Global Sensitivity of the Homeostatic Margin',weight='bold');finish(ax);save('Figure_6_Global_Sensitivity_PRCC.png')
# Figure 7: lollipop chart, no bars
classes=['Early crossing <=12m','Late crossing 12-18m','No crossing by 18m'];vals=np.array([(st['class']==c).sum() for c in classes]);fig,ax=plt.subplots(figsize=(11,5.8));y=np.arange(3)
for yi,v,c in zip(y,vals,[ROSE,PLUM,TEAL]):ax.hlines(yi,0,v,color=c,lw=3,alpha=.45);ax.scatter(v,yi,s=210,color=c,alpha=.75,zorder=3);ax.text(v+max(vals)*.018,yi,str(v),va='center',color=INK,weight='bold')
ax.set_yticks(y,['Early crossing ≤12 months','Late crossing 12–18 months','No crossing by 18 months']);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.2+1);ax.set_xlabel('Simulation count (n = 500)');ax.set_title('Stochastic Perturbation of Tipping Time',weight='bold');finish(ax);save('Figure_7_Stochastic_Tipping_Classes.png')
# Figure 8: interval dumbbell and annotated eigenvalue spectrum
EQ=pd.read_csv(OUT/'Latent_Quasi_Equilibria.csv');EIG=pd.read_csv(OUT/'Latent_Quasi_Equilibrium_Eigenvalues.csv');fig,axs=plt.subplots(1,2,figsize=(13,5.8));xx=np.arange(len(EQ))
for i,row in EQ.iterrows():
 axs[0].plot([row.momentum_eq,row.reserve_eq],[i,i],color=LILAC,lw=4,alpha=.42);axs[0].scatter(row.momentum_eq,i,marker='D',s=85,color=ROSE,alpha=.85,zorder=4,label='Momentum' if i==0 else None);axs[0].scatter(row.reserve_eq,i,marker='s',s=85,color=TEAL,alpha=.85,zorder=4,label='Reserve' if i==0 else None)
axs[0].set_yticks(xx,EQ.anchor);axs[0].invert_yaxis();axs[0].set_xlim(0,1);axs[0].set_xlabel('Frozen-observable quasi-equilibrium');axs[0].set_title('Latent Quasi-Equilibria');axs[0].legend(frameon=False)
markers=['D','s','P'];colors=[PLUM,ROSE,TEAL]
for i,(name,g) in enumerate(EIG.groupby('anchor',sort=False)):
 axs[1].scatter(g.real,g.imag,s=125,marker=markers[i],facecolor=colors[i],edgecolor='white',linewidth=1.5,alpha=.82,label=name,zorder=4)
 for j,(_,r) in enumerate(g.iterrows()):axs[1].annotate(f'λ{j+1}',(r.real,r.imag),xytext=(6,7+8*j),textcoords='offset points',fontsize=8,color=INK)
axs[1].axvline(0,color=INK,ls='--',lw=1);axs[1].set_xlabel('Real part of Jacobian eigenvalue');axs[1].set_ylabel('Imaginary part');axs[1].set_title('Jacobian Eigenvalue Spectrum');axs[1].legend(frameon=False,loc='best',fontsize=8)
for a in axs:finish(a)
fig.suptitle('Local Stability of the Latent Momentum Reserve Subsystem',weight='bold',fontsize=15);fig.tight_layout(rect=[0,0,1,.93]);save('Figure_8_Latent_Stability.png')
# Figure 9 unchanged original visual
F0,R0=38/48,.80;Ms=np.linspace(0.02,.95,25);Hs=np.linspace(.05,.95,25);MMg,HHg=np.meshgrid(Ms,Hs);U=np.zeros_like(MMg);V=np.zeros_like(HHg)
for i in range(MMg.shape[0]):
 for j in range(MMg.shape[1]):U[i,j],V[i,j]=latent_rhs([MMg[i,j],HHg[i,j]],F0,R0)
fig,ax=plt.subplots(figsize=(8,7));speed=np.sqrt(U**2+V**2);ax.streamplot(Ms,Hs,U,V,density=1.2,color=speed,cmap='plasma');eqc=EQ[EQ.anchor=='Compensated'].iloc[0];ax.scatter([eqc.momentum_eq],[eqc.reserve_eq],s=80,marker='*',label='quasi-equilibrium');ax.set_xlabel('Disease momentum M');ax.set_ylabel('Homeostatic reserve H');ax.set_title('Phase Plane at Fixed ALSFRS-R 38 and FVC 80%',weight='bold');ax.legend(frameon=False);save('Figure_9_Phase_Plane.png')
# Figure 10: centered dot-and-whisker AIC plot; annotations in separate right column
MCC=mc.sort_values('AIC_synthetic');fig,ax=plt.subplots(figsize=(13.5,6));yy=np.arange(len(MCC));lo=min(0,float(MCC.AIC_synthetic.min())-8);hi=float(MCC.AIC_synthetic.max())+8
for i,(_,r) in enumerate(MCC.iterrows()):
 ax.plot([lo,r.AIC_synthetic],[i,i],color=PLUM,alpha=.35,lw=3);ax.scatter(r.AIC_synthetic,i,color=TEAL if i==0 else PLUM,s=115,alpha=.83,zorder=3)
 ax.text(1.015,i,f'AIC {r.AIC_synthetic:.1f}  |  ΔAIC {r.delta_AIC:.1f}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=9,color=INK,clip_on=False)
ax.set_yticks(yy,[textwrap.fill(n,32) for n in MCC.model]);ax.invert_yaxis();ax.set_xlim(lo,hi);ax.set_xlabel('Synthetic AIC (lower is better)');ax.set_title('Internal Model-Recovery Comparison — Not Clinical Validation',weight='bold');finish(ax);fig.subplots_adjust(left=.30,right=.76,top=.88,bottom=.15);save('Figure_10_Synthetic_AIC.png')
# Figure 11: display actual narrow score range with percentile-derived palette, plus contours
POL=pd.read_csv(OUT/'Exploratory_Sequential_Monitoring_Policy.csv');pv=POL.pivot(index='baseline_FVC_pct',columns='baseline_ALSFRS_R',values='theoretical_decompensation_score');fig,ax=plt.subplots(figsize=(11,7));cm=LinearSegmentedColormap.from_list('policy',['#f5eef0','#d3a1b2','#805477'])
levels=np.linspace(float(pv.values.min()),float(pv.values.max()),19);im=ax.contourf(pv.columns,pv.index,pv.values,levels=levels,cmap=cm,extend='neither',alpha=.91)
ax.contour(pv.columns,pv.index,pv.values,levels=np.linspace(float(pv.values.min()),float(pv.values.max()),7)[1:-1],colors=[INK],linewidths=.6,alpha=.4)
if pv.values.min()<=.2<=pv.values.max():
 cs=ax.contour(pv.columns,pv.index,pv.values,levels=[.2],colors=[TEAL],linewidths=2);ax.clabel(cs,fmt={.2:'Exploratory threshold 0.20'},fontsize=9)
ax.set_xlabel('Baseline ALSFRS-R');ax.set_ylabel('Baseline FVC (% predicted)');ax.set_title('Exploratory Sequential Monitoring Decision Surface',weight='bold');fig.colorbar(im,ax=ax,label='Theoretical 3-month decompensation score (actual data range)');finish(ax);save('Figure_11_Sequential_Monitoring_Surface.png')
# Figure 12: spacious 2-row roadmap with no overlapping labels/arrows
fig,ax=plt.subplots(figsize=(16,7));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
steps=['Evidence\naudit','Structural\nverification','PRO-ACT\ncalibration','Observed-state\nestimation','External dataset\nvalidation','Decision-curve\nanalysis','Prospective\nclinical study']
coords=[(.06,.63),(.29,.63),(.52,.63),(.75,.63),(.75,.29),(.52,.29),(.29,.29)];w=.17;h=.16
for (x,y),label in zip(coords,steps):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',fc='#805477',alpha=.13,ec='none'))
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',fc='none',ec=PLUM,lw=1.4))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10,weight='bold',color=INK,linespacing=1.3)
for a,b in [((.23,.71),(.28,.71)),((.46,.71),(.51,.71)),((.69,.71),(.74,.71)),((.835,.61),(.835,.47)),((.75,.37),(.70,.37)),((.52,.37),(.47,.37))]:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.5,color=PLUM))
ax.text(.5,.95,'Roadmap From Mathematical Proof-of-Concept to Clinically Tested ALS Decision Support',ha='center',fontsize=15,weight='bold',color=INK)
ax.text(.5,.12,'Model boundaries are not treatment thresholds without patient-level calibration, external validation and clinical utility testing.',ha='center',fontsize=10,style='italic',color=INK)
save('Figure_12_Validation_Roadmap.png')
print('Completed:',OUT)
