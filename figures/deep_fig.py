import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
r=[0,10,20,30,40]
rand_df=[(83.15,.78),(77.86,1.03),(75.00,1.86),(71.80,.56),(66.67,1.87)]
adj_df=[(83.15,.78),(78.87,.28),(75.33,.78),(70.91,2.38),(64.73,.33)]
rand_sh=[(88.67,1.15),(81.00,2.60),(78.17,2.02),(72.33,4.01),(69.67,6.25)]
adj_sh=[(88.67,1.15),(82.50,2.00),(77.00,0.0),(67.50,.50),(63.00,4.77)]
rand_lr=[(0.73,.42),(2.11,.42),(3.67,.64),(5.97,1.24),(9.73,2.38)]
adj_lr=[(0.73,.42),(2.57,.16),(2.94,1.15),(3.95,1.99),(4.41,.55)]
n=[417,1042,2085,4170]
fdf=[(72.47,.85),(77.86,1.16),(80.25,1.75),(83.15,.78)]
fsh=[(73.67,2.47),(82.50,5.57),(85.83,1.26),(88.67,1.15)]
C1,C2="#2E5C8A","#C0504D"
fig,ax=plt.subplots(1,4,figsize=(17,4.2))
def eb(a,x,d,c,m,l): a.errorbar(x,[v for v,_ in d],yerr=[s for _,s in d],fmt=m+"-",color=c,capsize=3,label=l,lw=1.8,ms=6)
for a,(d1,d2,t,yl) in zip(ax[:3],[(rand_df,adj_df,"(a) DF2 평가셋 4분류 정확도","정확도 (%)"),
                                 (rand_sh,adj_sh,"(b) 쇼핑몰 200장 4분류 정확도","정확도 (%)"),
                                 (rand_lr,adj_lr,"(c) DF2 좌↔우 혼동률","혼동률 (%)")]):
    eb(a,r,d1,C1,"o","무작위 노이즈"); eb(a,r,d2,C2,"s","이웃 방향 노이즈")
    a.set_title(t,fontsize=12); a.set_xlabel("학습 라벨 노이즈 비율 (%)"); a.set_ylabel(yl); a.set_xticks(r); a.grid(alpha=.3)
ax[0].legend()
eb(ax[3],n,fdf,C1,"o","DF2 평가셋"); eb(ax[3],n,fsh,C2,"s","쇼핑몰 200장")
ax[3].set_xscale("log"); ax[3].set_xticks(n); ax[3].set_xticklabels([f"{x}\n({p}%)" for x,p in zip(n,[10,25,50,100])])
ax[3].minorticks_off()
ax[3].axhline(77.86,color="gray",ls=":",lw=1.2,label="DF2: 노이즈 10% + 전체 라벨 (77.86)")
ax[3].set_title("(d) 학습 라벨 수에 따른 4분류 정확도",fontsize=12); ax[3].set_xlabel("학습 이미지 수 (사용 비율)"); ax[3].set_ylabel("정확도 (%)"); ax[3].grid(alpha=.3); ax[3].legend(loc="lower right")
plt.tight_layout(); plt.savefig("fig_deep_curves.png",dpi=200)
