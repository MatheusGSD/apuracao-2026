r"""Teste do artigo "O swing simples basta para projetar um 2º turno?" (30/10/2026).

O MESMO swing simples (município apurado = share de referência + swing do município nas seções
apuradas; sem seção = + swing da UF), com duas referências: 1º turno de 2026 e 2º turno de 2022
(pareamento exato por uf/mun/zona/seção: 92% das seções de 2026). Ordem real (tot_hora). Não é o
modelo de Mahlmeister: sem pareamento geográfico, razão log, regressão hierárquica nem prior.
Variante A: peso = votos A+B finais de 2026 por município (só o share muda entre as bases).
Variante B: voto contado entra como contado; o que falta pesa pela referência.
Uso: python relatorio\estudos\swing_referencia.py  (pandas; saída em swing_referencia.txt)
"""
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
D = r"C:\dev\apuracao\data"; K=["uf","mun","zona","secao"]; FINAL=51.7852
t2 = pd.read_parquet(f"{D}/2026/t2_secoes.parquet").sort_values("tot_hora",kind="stable").reset_index(drop=True)
t1 = pd.read_parquet(f"{D}/2026/t1_secoes.parquet"); r22 = pd.read_parquet(f"{D}/2022/t2_secoes.parquet")
t2["v"]=t2.a+t2.b; n=len(t2)
tm=t2.groupby(["uf","mun"]).agg(A=("a","sum"),V=("v","sum"))
PCTS=[2,3,5,8,10,15,20,30,40,50,70,90,100]
def prepare(ref, escala):
    x=t2[K+["a","v","aptos"]].merge(ref[K+["a","b","aptos"]].rename(columns={"a":"ra","b":"rb","aptos":"rap"}),on=K,how="left")
    x["par"]=x.ra.notna(); x["rv"]=(x.ra+x.rb)
    # municipio inteiro na referencia
    rm=ref.assign(rv=ref.a+ref.b).groupby(["uf","mun"]).agg(ra=("a","sum"),rv=("rv","sum"),ap=("aptos","sum"))
    rm["rsh"]=rm.rv.where(rm.rv>0); rm["rsh"]=rm.ra/rm.rv
    x=x.join(rm[["rsh","rv","ap"]].rename(columns={"rsh":"msh","rv":"mrv","ap":"map"}),on=["uf","mun"])
    # share ref por secao (fallback muni; fallback UF-nacional -> 0.5 raro)
    x["rs"]=np.where(x.par&(x.rv>0),x.ra/x.rv.replace(0,np.nan),x.msh)
    ufsh=ref.groupby("uf").apply(lambda d:d.a.sum()/(d.a.sum()+d.b.sum()),include_groups=False)
    x["rs"]=x.rs.fillna(x.uf.map(ufsh))
    sc = x.aptos/x.rap if escala else 1.0
    mrate=(x.mrv/x["map"]*x.aptos) if escala else x.mrv/ (x.mrv.where(x.mrv>0))*0  # placeholder
    if escala:
        x["refv"]=np.where(x.par,x.rv*x.aptos/x.rap,mrate)
    else:
        x["refv"]=np.where(x.par,x.rv,x.v)   # t1 sempre pareado
    x["refv"]=np.nan_to_num(x.refv)
    x["rsa"]=x.refv*x.rs
    return x
def run(x, oraculo, diag=False):
    out={}; sw100=None
    for pct in PCTS:
        k=int(round(n*pct/100)); ap=np.arange(n)<k
        s=x[ap&x.par.values&(x.v>0).values&(x.rv>0).values]
        g=s.groupby(["uf","mun"]).agg(a=("a","sum"),v=("v","sum"),ra=("ra","sum"),rv=("rv","sum"))
        g["sw"]=g.a/g.v-g.ra/g.rv
        gu=g.groupby("uf").apply(lambda d:np.average(d.sw,weights=d.rv),include_groups=False)
        nat=np.average(g.sw,weights=g.rv)
        c=x[ap].groupby(["uf","mun"]).agg(ca=("a","sum"),cv=("v","sum"))
        rem=x[~ap].groupby(["uf","mun"]).agg(rsa=("rsa","sum"),rrv=("refv","sum"))
        mm=tm.join(c).join(rem).join(g[["sw"]]).fillna({"ca":0,"cv":0,"rsa":0,"rrv":0})
        mm["sw"]=mm.sw.fillna(pd.Series(mm.index.get_level_values(0).map(gu),index=mm.index)).fillna(nat)
        rsh=np.where(mm.rrv>0,mm.rsa/mm.rrv.replace(0,np.nan),0)
        psh=np.clip(rsh+mm.sw,0,1)
        remV=(mm.V-mm.cv).clip(lower=0) if oraculo else mm.rrv
        PA=mm.ca+remV*psh; PV=mm.cv+remV
        proj=PA.sum()/PV.sum()*100
        out[pct]=proj-FINAL
        if diag and pct in (8,10,15,20,30):
            e=(PA-mm.A)/mm.V.sum()*100; ap_m=mm.cv>0
            out[("d",pct)]=(e[ap_m].sum(),e[~ap_m].sum(),ap_m.mean()*100)
        if pct==100: sw100=np.average(g.sw,weights=g.rv)*100
    return out,sw100
bases={"t1-2026":prepare(t1,False),"t2-2022":prepare(r22,True)}
for var,orc in [("A (oraculo)",True),("B (realista)",False)]:
    print(f"\n=== Variante {var} ===\n%apur  t1-2026  t2-2022")
    R={b:run(x,orc,diag=orc) for b,x in bases.items()}
    for p in PCTS: print(f"{p:5d}  {R['t1-2026'][0][p]:+.4f}  {R['t2-2022'][0][p]:+.4f}")
    for lo,hi in [(2,5),(8,30),(50,90)]:
        s=[]
        for b in R:
            v=[R[b][0][p] for p in PCTS if lo<=p<=hi]; s.append(f"{b}: |e|={np.mean(np.abs(v)):.3f} vies={np.mean(v):+.3f}")
        print(f"{lo}-{hi}%  "+" | ".join(s))
    if orc:
        print("Decomposicao base 2022 (pt de erro): pct  apurados  sem-apurar  (% munic. apurados)")
        for p in (8,10,15,20,30):
            d=R['t2-2022'][0][("d",p)]; print(f"  {p}: {d[0]:+.3f}  {d[1]:+.3f}  ({d[2]:.0f}%)")
        print("  (base 2026:)")
        for p in (8,10,15,20,30):
            d=R['t1-2026'][0][("d",p)]; print(f"  {p}: {d[0]:+.3f}  {d[1]:+.3f}  ({d[2]:.0f}%)")
