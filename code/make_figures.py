"""Generate Figures 1–12 from previously computed ALS model CSV outputs.

This script does NOT rerun the ODE model, LHS, stochastic experiments, or AIC fitting.
Place it in code/ and put the original model CSV outputs in results/.
Run: python code/make_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results'
FIG = ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
required = {
    'ref':'Reference_Trajectory.csv',
    'sc':'Scenario_Results.csv',
    'surf':'ALSFRS_FVC_Tipping_Surface.csv',
    'pr':'Global_Sensitivity_PRCC.csv',
    'st':'Stochastic_Perturbation_Results.csv',
    'mc':'Synthetic_Model_Comparison_AIC.csv',
}
missing = [name for name in required.values() if not (OUT/name).exists()]
if missing:
    raise FileNotFoundError('Missing results CSV files: ' + ', '.join(missing) +
                            '\nRun the original model first and copy its CSV outputs to results/.')
ref = pd.read_csv(OUT/required['ref'])
sc = pd.read_csv(OUT/required['sc'])
surf = pd.read_csv(OUT/required['surf'])
pr = pd.read_csv(OUT/required['pr'])
st = pd.read_csv(OUT/required['st'])
mc = pd.read_csv(OUT/required['mc'])

# Only used to draw the phase portrait in Figure 9; no simulation or fitting.
BASE = dict(kM0=.020, kMF=.080, kMR=.100, kMM=.090, kMH=.100,
            kHF=.035, kHR=.045, kHM=.100, kHd=.012)
def latent_rhs(z, F0, R0, p=BASE):
    M,H = z
    dM = p['kM0']*(1-M)+p['kMF']*(1-F0)*(1-M)+p['kMR']*(1-R0)*(1-M)+p['kMM']*M*(1-M)-p['kMH']*H*M
    dH = p['kHF']*F0*(1-H)+p['kHR']*R0*(1-H)-p['kHM']*M*H-p['kHd']*H
    return np.array([dM,dH])

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
print('Figures saved to:', FIG)
