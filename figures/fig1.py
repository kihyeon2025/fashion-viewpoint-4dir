import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm
fb="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"; fr="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(fb); fm.fontManager.addfont(fr)
B=fm.FontProperties(fname=fb); R=fm.FontProperties(fname=fr)
NAVY="#1F3A5F"; LIGHT="#E8EEF7"; EDGE="#2E6DB4"; ORG="#E39B3A"
W,H=16.0,6.6
fig=plt.figure(figsize=(W,H),dpi=110); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
def box(x,y,w,h,title,sub,fc=NAVY,tc="white",sfc=LIGHT,hh=None):
    hh=hh or h*0.48
    ax.add_patch(FancyBboxPatch((x,y+h-hh),w,hh,boxstyle="round,pad=0,rounding_size=0.18",fc=fc,ec=fc))
    ax.text(x+w/2,y+h-hh/2,title,ha="center",va="center",color=tc,fontproperties=B,fontsize=29,linespacing=1.15)
    if sub:
        sh=h-hh-0.15
        ax.add_patch(FancyBboxPatch((x,y),w,sh,boxstyle="round,pad=0,rounding_size=0.18",fc=sfc,ec=EDGE,lw=2))
        ax.text(x+w/2,y+sh/2,sub,ha="center",va="center",color="#222",fontproperties=R,fontsize=23,linespacing=1.3)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=22,lw=2.5,color=EDGE))
# left
box(0.2,1.3,3.0,4.0,"DeepFashion2\n라벨 구축","몸통 기준\n4방향\n5,963장")
# branches
box(4.3,3.55,4.6,2.85,"외형 모델\nConvNeXt-T","ImageNet 전이학습\n반전 시 좌↔우 교환",hh=1.45)
box(4.3,0.2,4.6,2.85,"자세 모델\nKeypoint R-CNN","관절 17개 좌표\n→ GBDT 분류기",hh=1.45)
# fusion
box(10.0,1.3,2.7,4.0,"결합","확률\n가중 평균\n(w = 0.6)",fc=ORG)
box(13.6,1.3,2.2,4.0,"4방향\n예측","DF2 896장\n쇼핑몰\n200장")
arr(3.25,3.3,4.25,4.8); arr(3.25,3.3,4.25,1.6)
arr(8.95,4.8,9.95,3.4); arr(8.95,1.6,9.95,3.2)
arr(12.75,3.3,13.55,3.3)
plt.savefig("fig1_new.png",dpi=110)
