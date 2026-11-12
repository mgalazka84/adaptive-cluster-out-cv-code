"""Vector schematics for the article; all coordinates are illustrative.

These figures explain the mathematics and do not add empirical results.
Run with Python, NumPy and Matplotlib. No study inputs are changed.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

OUT=Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelsize':9,'axes.titlesize':9.5,
    'figure.facecolor':'white','savefig.facecolor':'white'})
TEAL='#007f8b'; BLUE='#4166a5'; ORANGE='#cb702d'; GRAY='#84939d'; PURPLE='#765491'

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def crossed_targets():
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.85))
    fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.27,wspace=.38)
    for panel,(ax,kind) in enumerate(zip(axes,['participant','word'])):
        nr,nc=4,5
        for row in range(nr):
            for col in range(nc):
                test=(row==3) if kind=='participant' else (col==4)
                ax.add_patch(Rectangle((col,row),1,1,facecolor='#e9eef1' if not test else '#f3d5be',
                    edgecolor='white',linewidth=2,hatch='///' if test else None))
                if test:ax.plot(col+.5,row+.5,'o',color=ORANGE,ms=4)
        ax.set_xlim(0,nc);ax.set_ylim(nr,0);ax.set_aspect('equal')
        ax.set_xticks(np.arange(nc)+.5,[f'W{x+1}' for x in range(nc)])
        ax.set_yticks(np.arange(nr)+.5,[f'P{x+1}' for x in range(nr)])
        ax.tick_params(length=0,labelsize=8.5)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.xaxis.tick_top()
        ax.set_title(f'({chr(97+panel)}) New {kind}',fontweight='bold',pad=23)
        ax.set_xlabel('Words',labelpad=5);ax.set_ylabel('Participants',labelpad=6)
        explanation=('Retain learned word effects' if kind=='participant' else
                     'Retain learned participant effects')
        ax.text(.5,-.24,explanation,transform=ax.transAxes,ha='center',va='center',fontsize=9)
    fig.legend([Rectangle((0,0),1,1,fc='#e9eef1',ec='white'),Rectangle((0,0),1,1,fc='#f3d5be',hatch='///',ec='white')],
               ['Training observations','Held-out observations'],loc='lower center',ncol=2,
               bbox_to_anchor=(.5,-.01),frameon=False,fontsize=8.8)
    save(fig,'Fig2')

def face_geometry():
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.0))
    fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.19,wspace=.32)
    x=np.linspace(0,1.35,200);y=np.linspace(.1,2.1,200);xx,yy=np.meshgrid(x,y)
    for i,(ax,sgn) in enumerate(zip(axes,[1,-1])):
        z=.5*(yy-1.1)**2+sgn*.4*xx+.6*xx**2
        levels=[-.04,.04,.14,.3,.55,.85,1.2] if sgn<0 else [.04,.14,.3,.55,.85,1.2]
        ax.contour(xx,yy,z,levels=levels,colors='#d0d7dc',linewidths=.8)
        ax.axvline(0,color=PURPLE,lw=2)
        ax.plot([0,0],[.87,1.33],color=PURPLE,lw=7,alpha=.27,solid_capstyle='round')
        ax.plot(0,1.1,'o',color=PURPLE,ms=6,clip_on=False)
        ax.annotate('',xy=(.6 if sgn>0 else .29,1.1),xytext=(.1,1.1),arrowprops={'arrowstyle':'->','color':TEAL if sgn>0 else ORANGE,'lw':1.8})
        ax.text(.36,1.29,'F increases' if sgn>0 else 'F decreases',ha='center',fontsize=9,
                color=TEAL if sgn>0 else ORANGE)
        ax.text(.03,.6,'face-stationary\nroot',fontsize=8.1,color=PURPLE,ha='left')
        if sgn<0:
            ax.plot(1/3,1.1,'x',color=ORANGE,ms=7,mew=1.8)
        ax.set_title(('(a) Positive active derivative' if sgn>0 else '(b) Negative active derivative'),fontweight='bold',loc='left',pad=15)
        ax.set_xlim(-.03,1.35);ax.set_ylim(.1,2.1)
        ax.set_xticks([0],[r'$0$']);ax.set_yticks([])
        ax.set_xlabel(r'Active variance $\theta_a$')
        ax.set_ylabel(r'Free variance $\theta_b$')
        ax.text(.97,.93,'feasible region',ha='right',transform=ax.transAxes,color=GRAY,fontsize=8.5)
    fig.text(.29,.02,'Accept only with a positive lower bound',ha='center',fontsize=8.5,color=TEAL)
    fig.text(.77,.02,'Release this face and continue fitting',ha='center',fontsize=8.5,color=ORANGE)
    save(fig,'Fig3')

def error_propagation():
    fig=plt.figure(figsize=(7.0,2.5))
    axes=[fig.add_axes([.025,.22,.25,.60]),fig.add_axes([.36,.22,.25,.60]),fig.add_axes([.70,.22,.275,.60])]
    for ax in axes:ax.set_aspect('equal');ax.axis('off')
    a,b,c=axes
    a.set_xlim(-.8,.8);a.set_ylim(-.75,.75)
    a.add_patch(Circle((0,0),.55,fc='#ddecf0',ec=TEAL,lw=1.3))
    a.plot(0,0,'o',color='black',ms=3);a.plot(.18,.2,'o',color=TEAL,ms=4)
    a.annotate(r'$u^*$',(.18,.2),xytext=(.28,.23),fontsize=10)
    a.text(-.08,-.17,r'$0$',fontsize=9)
    a.annotate('',xy=(.55,0),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':1,'color':TEAL})
    a.text(.3,-.16,r'$\rho$',fontsize=11,color=TEAL)
    a.set_title('(a) Covariance fit',fontweight='bold',pad=12)
    b.set_xlim(-.48,1.2);b.set_ylim(-.75,.75)
    b.add_patch(Circle((0,0),.28,fc='#e9e1f0',ec=PURPLE,lw=1.3))
    b.plot(0,0,'o',color='black',ms=3);b.plot(.16,.1,'o',color=PURPLE,ms=4)
    b.plot(1.05,0,'x',color=ORANGE,ms=6,mew=1.5)
    b.text(-.13,-.45,r'$\mu_0$',fontsize=10);b.text(.14,.28,r'$\mu^*$',fontsize=10)
    b.text(.99,.15,r'$y_s$',fontsize=10,color=ORANGE)
    b.plot([0,1.05],[0,0],ls='--',lw=.8,color=GRAY,zorder=-1)
    b.text(.62,-.2,r'$e$',fontsize=11)
    b.text(-.18,.12,r'$D$',fontsize=10,color=PURPLE)
    b.set_title('(b) Held-out mean',fontweight='bold',pad=12)
    c.set_aspect('auto');c.set_xlim(0,1);c.set_ylim(0,1)
    c.plot([.08,.92],[.5,.5],lw=7,color='#CEDBE2',solid_capstyle='butt')
    c.plot([.27,.61],[.5,.5],lw=4,color=TEAL,solid_capstyle='butt')
    c.plot([.08,.08],[.42,.58],lw=1.2,color=GRAY);c.plot([.92,.92],[.42,.58],lw=1.2,color=GRAY)
    c.plot(.40,.5,'o',color='black',ms=4)
    c.text(.4,.66,r'$\ell(\theta^*)$',ha='center',fontsize=10)
    c.text(.5,.22,'Gray: norm bound',ha='center',fontsize=8,color=GRAY)
    c.text(.5,.06,'Teal: directional intersection',ha='center',fontsize=8,color=TEAL)
    c.set_title('(c) Squared loss',fontweight='bold',pad=12)
    for p,q in [((.28,.53),(.345,.53)),((.615,.53),(.685,.53))]:
        fig.add_artist(FancyArrowPatch(p,q,transform=fig.transFigure,arrowstyle='->',mutation_scale=12,lw=1.2,color=GRAY))
    fig.text(.31,.14,'Prediction\nidentity',ha='center',va='center',fontsize=8,color=GRAY)
    fig.text(.65,.14,'Residual\ndirection',ha='center',va='center',fontsize=8,color=GRAY)
    fig.text(.15,.015,r'$\|u^*\|\leq\rho$',ha='center',fontsize=10)
    fig.text(.48,.015,r'$\|\mu^*-\mu_0\|\leq D$',ha='center',fontsize=10)
    fig.text(.84,.015,'Computational error only',ha='center',fontsize=8.5)
    save(fig,'Fig4')

if __name__=='__main__':
    crossed_targets();face_geometry();error_propagation()
    print('Created the three explanatory vector figures.')
