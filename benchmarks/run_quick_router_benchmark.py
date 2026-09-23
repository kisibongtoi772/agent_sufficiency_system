from __future__ import annotations
import json, re, time, urllib.request
from pathlib import Path

ENDPOINT="https://blockrun.ai/api/v1/chat/completions"
MODELS={
 "nano":"nvidia/nemotron-3-nano-30b",
 "worker":"nvidia/gpt-oss-20b",
 "planner":"nvidia/nemotron-3.5-lightning",
}
TASKS=[
 ("easy_arithmetic","easy","Compute exactly: (37 * 24) - (18 * 7) + 45.","807"),
 ("medium_shortest_path","medium","Directed weighted edges: A->B 4, A->C 2, B->C 1, B->D 5, C->B 1, C->D 8, C->E 10, D->E 2. What is the shortest-path distance from A to E?","10"),
 ("hard_grid_paths","hard","Count monotone paths in a 5x5 grid from (0,0) to (4,4), moving only right/down. Blocked: (1,1),(1,3),(2,1),(3,3). How many valid paths?","6"),
 ("hard_matrix_chain","hard","A1=10x30, A2=30x5, A3=5x60. Minimum scalar multiplications for A1*A2*A3?","4500"),
]

def call(model,messages):
    payload={"model":model,"messages":messages,"temperature":0,"max_tokens":250}
    req=urllib.request.Request(ENDPOINT,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    t=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=45) as r:
            data=json.loads(r.read().decode())
        content=data["choices"][0]["message"].get("content") or ""
        usage=data.get("usage") or {}
        return {"ok":True,"content":content,"served":data.get("model","unknown"),"latency":time.perf_counter()-t,
                "tokens":int(usage.get("total_tokens") or max(1,(len(json.dumps(messages))+len(content)+3)//4))}
    except Exception as e:
        return {"ok":False,"content":"","served":"error","latency":time.perf_counter()-t,"tokens":0,"error":repr(e)}

def answer(text):
    try:
        o=json.loads(text)
        if isinstance(o,dict) and "answer" in o:return str(o["answer"])
    except: pass
    nums=re.findall(r"-?\d+(?:\.\d+)?",text.replace(",",""))
    return nums[-1] if nums else text.strip()

def worker(prompt,key,plan=None):
    if plan: prompt += "\nPlanner notes:\n"+plan
    return call(MODELS[key],[{"role":"system","content":'Solve carefully. Return only JSON: {"answer":"VALUE"}.'},{"role":"user","content":prompt}])

def planner(prompt):
    return call(MODELS["planner"],[{"role":"system","content":"Give a concise plan for a worker. Under 100 words."},{"role":"user","content":prompt}])

def run(policy,diff,prompt,expected):
    calls=[]
    if policy=="nano_only":
        final=worker(prompt,"nano"); calls=[final]
    elif policy=="gpt_oss_only":
        final=worker(prompt,"worker"); calls=[final]
    elif policy=="planner_worker":
        p=planner(prompt); final=worker(prompt,"worker",p["content"]); calls=[p,final]
    elif policy=="complexity_gated":
        if diff=="easy": final=worker(prompt,"nano"); calls=[final]
        elif diff=="medium": final=worker(prompt,"worker"); calls=[final]
        else:
            p=planner(prompt); final=worker(prompt,"worker",p["content"]); calls=[p,final]
    actual=answer(final["content"]).replace(",","").strip()
    return {"success":actual==expected,"actual":actual,"calls":len(calls),
            "tokens":sum(c["tokens"] for c in calls),"latency":sum(c["latency"] for c in calls),
            "served":[c["served"] for c in calls],"errors":[c.get("error") for c in calls if not c["ok"]]}

def main():
    out=Path("results/model_routing_quick"); out.mkdir(parents=True,exist_ok=True)
    policies=["nano_only","gpt_oss_only","planner_worker","complexity_gated"]
    rows=[]
    for pol in policies:
        for tid,diff,prompt,expected in TASKS:
            r=run(pol,diff,prompt,expected)
            row={"policy":pol,"task":tid,"difficulty":diff,"expected":expected,**r}
            rows.append(row)
            print(pol,tid,"PASS" if r["success"] else "FAIL","served=",r["served"],flush=True)
    summary={}
    for pol in policies:
        x=[r for r in rows if r["policy"]==pol]
        wins=sum(r["success"] for r in x)
        summary[pol]={"pass_rate":wins/len(x),"successes":wins,"tasks":len(x),
                      "total_calls":sum(r["calls"] for r in x),"total_tokens":sum(r["tokens"] for r in x),
                      "tokens_per_solved":sum(r["tokens"] for r in x)/wins if wins else None,
                      "mean_latency_s":sum(r["latency"] for r in x)/len(x),
                      "served_models":sorted({m for r in x for m in r["served"]})}
    result={"models":MODELS,"tasks":len(TASKS),"summary":summary,"rows":rows}
    (out/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
