import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
O="./"
NAVY,BLUE,LBLUE,RED,GRAY,ORG="#1F3A5F","#3E6FB0","#9DB9DD","#C0504D","#9A9A9A","#E39B3A"

def lab(ax,bars,fmt="{:.1f}",dy=0.6,fs=9):
    for b in bars:
        h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h+dy,fmt.format(h),ha="center",va="bottom",fontsize=fs)

# ---------- 그림 1: 방법별 비교
names=["규칙 기반\n(관절)","자세 모델\n(관절+GBDT)","ResNet50","ConvNeXt-T","ConvNeXt-T\n앙상블","외형-자세\n결합"]
df=[66.85,81.70,80.99,83.15,84.26,86.27]; dfe=[0,0,1.15,0.78,0,0]
sh=[72.00,90.00,85.67,88.67,91.00,93.50]; she=[0,0,4.25,1.15,0,0]
x=np.arange(len(names)); w=0.38
fig,ax=plt.subplots(figsize=(10,4.6))
ax.axvspan(4.5,5.5,color=ORG,alpha=.12,zorder=0)
b1=ax.bar(x-w/2,df,w,color=NAVY,label="DF2 평가셋 (896장)",zorder=2)
b2=ax.bar(x+w/2,sh,w,color=LBLUE,label="쇼핑몰 (200장)",zorder=2)
for bars,e in [(b1,dfe),(b2,she)]:
    for b,ee in zip(bars,e):
        h=b.get_height(); xc=b.get_x()+b.get_width()/2
        if ee: ax.errorbar(xc,h,yerr=ee,color="black",capsize=3,zorder=3)
        ax.text(xc,h+ee+0.6,f"{h:.1f}",ha="center",va="bottom",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(50,100); ax.set_ylabel("4분류 정확도 (%)")
ax.legend(loc="upper left"); ax.grid(axis="y",alpha=.3); ax.set_axisbelow(True)
ax.set_title("방법별 4방향 분류 정확도 (단일 모델은 seed 3회 평균±표준편차)",fontsize=12)
plt.tight_layout(); plt.savefig(O+"fig_method_comparison.png",dpi=200); plt.close()

# ---------- 그림 2: 어블레이션 (ResNet50 vs ConvNeXt-T)
fig,ax=plt.subplots(1,3,figsize=(15,4.4),gridspec_kw={"width_ratios":[1.3,1.3,1]})
# (a) 라벨 출처
g=["DF2 평가셋","쇼핑몰"]; xx=np.arange(2); w=0.2
vals={"ResNet50 · DF2 라벨":[76.49,86.00],"ResNet50 · 직접 라벨":[82.51,92.83],
      "ConvNeXt-T · DF2 라벨":[78.50,83.83],"ConvNeXt-T · 직접 라벨":[85.19,93.17]}
cols=[GRAY,NAVY,"#C9C9C9",BLUE]; hat=["","","//","//"]
for i,(k,v) in enumerate(vals.items()):
    b=ax[0].bar(xx+(i-1.5)*w,v,w,color=cols[i],hatch=hat[i],edgecolor="white",label=k); lab(ax[0],b,fs=8,dy=.4)
ax[0].set_xticks(xx); ax[0].set_xticklabels(g); ax[0].set_ylim(70,97); ax[0].set_ylabel("정면/비정면 정확도 (%)")
ax[0].set_title("(a) 학습 라벨 출처",fontsize=12); ax[0].legend(fontsize=8,loc="upper left"); ax[0].grid(axis="y",alpha=.3); ax[0].set_axisbelow(True)
# (b) 반전 처리
g=["반전 + 라벨 교체\n(제안)","반전 없음","반전 + 라벨 유지"]; xx=np.arange(3); w=0.36
r=[80.99,78.79,63.69]; c=[83.15,81.62,67.04]; rl=[1.84,3.03,35.08]; cl=[0.73,1.65,34.53]
b1=ax[1].bar(xx-w/2,r,w,color=NAVY,label="ResNet50"); b2=ax[1].bar(xx+w/2,c,w,color=BLUE,label="ConvNeXt-T")
lab(ax[1],b1,fs=8,dy=.5); lab(ax[1],b2,fs=8,dy=.5)
g=[f"{a}\n좌↔우 혼동 {r_:.1f} / {c_:.1f}%" for a,r_,c_ in zip(g,rl,cl)]
ax[1].set_xticks(xx); ax[1].set_xticklabels(g,fontsize=9); ax[1].set_ylim(50,90); ax[1].set_ylabel("DF2 4분류 정확도 (%)")
ax[1].set_title("(b) 좌우 반전 처리",fontsize=12); ax[1].legend(fontsize=9,loc="upper right"); ax[1].grid(axis="y",alpha=.3); ax[1].set_axisbelow(True)
# (c) 전이학습
g=["ImageNet\n사전학습","처음부터\n학습"]; xx=np.arange(2)
b1=ax[2].bar(xx-w/2,[80.99,38.28],w,color=NAVY,label="ResNet50"); b2=ax[2].bar(xx+w/2,[83.15,38.99],w,color=BLUE,label="ConvNeXt-T")
lab(ax[2],b1,fs=8,dy=.8); lab(ax[2],b2,fs=8,dy=.8)
ax[2].axhline(25,color=GRAY,ls=":",lw=1); ax[2].text(-0.45,26.5,"무작위 추측(25%)",fontsize=8,color=GRAY,ha="left")
ax[2].set_xticks(xx); ax[2].set_xticklabels(g); ax[2].set_ylim(0,95); ax[2].set_ylabel("DF2 4분류 정확도 (%)")
ax[2].set_title("(c) 전이학습",fontsize=12); ax[2].legend(fontsize=9); ax[2].grid(axis="y",alpha=.3); ax[2].set_axisbelow(True)
plt.tight_layout(); plt.savefig(O+"fig_ablation.png",dpi=200); plt.close()

# ---------- 그림 3: 상호보완성
sets=["DF2 평가셋\n(896장)","쇼핑몰\n(200장)"]
d=np.array([[669,86,63,78],[169,13,11,7]],float); pct=d/d.sum(1,keepdims=True)*100
labs=["둘 다 정답","외형 모델만 정답","자세 모델만 정답","둘 다 오답"]; cc=[LBLUE,NAVY,ORG,RED]
fig,ax=plt.subplots(figsize=(10,2.9)); left=np.zeros(2)
for j in range(4):
    ax.barh(sets,pct[:,j],left=left,color=cc[j],label=labs[j],edgecolor="white")
    for i in range(2):
        if pct[i,j]>3: ax.text(left[i]+pct[i,j]/2,i,f"{int(d[i,j])}장\n{pct[i,j]:.1f}%",ha="center",va="center",fontsize=8.5,color="white" if j else NAVY)
    left+=pct[:,j]
ax.set_xlim(0,100); ax.set_xlabel("비율 (%)"); ax.invert_yaxis()
ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(.5,1.32),fontsize=9,frameon=False)
ax.set_title("외형 모델(ConvNeXt-T 앙상블)과 자세 모델의 정답 겹침",fontsize=12,pad=30)
plt.tight_layout(); plt.savefig(O+"fig_complementarity.png",dpi=200); plt.close()

# ---------- 그림 4: 쇼핑몰 방향별 정확도
dirs=["정면","후면","좌측면","우측면"]; xx=np.arange(4); w=0.26
m={"ResNet50 앙상블":[96,94,94,74],"ConvNeXt-T 앙상블":[96,96,92,80],"외형-자세 결합":[100,96,96,82]}
fig,ax=plt.subplots(figsize=(9,4.6))
for i,((k,v),c) in enumerate(zip(m.items(),[GRAY,BLUE,ORG])):
    b=ax.bar(xx+(i-1)*w,v,w,color=c,label=k); lab(ax,b,fmt="{:.0f}",fs=8,dy=.5)
ax.set_xticks(xx); ax.set_xticklabels(dirs); ax.set_ylim(60,105); ax.set_ylabel("정확도 (%)")
ax.set_title("쇼핑몰 이미지 방향별 정확도 (방향별 50장)",fontsize=12); ax.legend(loc="upper center",bbox_to_anchor=(.5,-.1),ncol=3,fontsize=9,frameon=False)
ax.grid(axis="y",alpha=.3); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(O+"fig_shop_per_direction.png",dpi=200); plt.close()
print("ok")
