import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fr="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fb="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
fm.fontManager.addfont(fr); fm.fontManager.addfont(fb)
plt.rcParams["font.family"]=fm.FontProperties(fname=fr).get_name()
B=fm.FontProperties(fname=fb)
rows=[("DF2 '정면' 라벨",[("정면",73.3,"#9DB9DD","#1F3A5F"),("좌측면",15.1,"#C0504D","w"),("우측면",11.2,"#D98A87","w"),("",0.4,"#888","w")]),
      ("DF2 '측면·후면' 라벨",[("후면",35.2,"#1F3A5F","w"),("좌측면",31.4,"#3E6FB0","w"),("우측면",24.9,"#7FA3D3","w"),("정면",8.5,"#C0504D","w")])]
fig,ax=plt.subplots(figsize=(9,2.6))
for i,(lab,segs) in enumerate(rows):
    left=0
    for name,v,c,tc in segs:
        ax.barh(i,v,left=left,color=c,edgecolor="white",height=0.62)
        if v>5: ax.text(left+v/2,i,(f"{name}\n{v}" if v<16 else f"{name} {v}"),ha="center",va="center",color="white" if tc=="w" else tc,fontproperties=B,fontsize=11 if v<16 else 12,linespacing=1.0)
        left+=v
ax.set_yticks([0,1]); ax.set_yticklabels([r[0] for r in rows],fontsize=12); ax.invert_yaxis()
ax.set_xlim(0,100); ax.set_xlabel("직접 확인한 실제 방향 비율 (%)",fontsize=11)
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_problem.png",dpi=200)
