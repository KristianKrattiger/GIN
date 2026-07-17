"""Local-only FastAPI labeling app: one page + two JSON endpoints.

signals_fn is injected (real: pair_signals bound to a CombinedRelationProposer;
tests: a fake) so the endpoints are model-free under test. The store is the
source of truth; the candidate source only drives ORDERING, so a label for a
pair the source doesn't enumerate is still accepted and stored.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from gin.cartographer.models import Relation
from gin.cartographer.scan import sentence_anchor

from .candidates import CandidateSource, order_backlog
from .models import LabelRecord, pair_key
from .readiness import ReadinessTarget, readiness
from .store import Store

_VALID_RELATIONS = {r.value for r in Relation}


class LabelRequest(BaseModel):
    src_chunk_id: str
    dst_chunk_id: str
    relation: str
    relation_class: Optional[str] = None
    rationale: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_curator_app(
    *,
    store: Store,
    source: CandidateSource,
    signals_fn: Callable[[str, str], dict],
    curator: str = "curator",
    scan_limit: int = 500,
    readiness_target: ReadinessTarget = ReadinessTarget(),
) -> FastAPI:
    app = FastAPI(title="GIN Curator")
    text_by_id = {c.chunk_id: c.text for c in source.chunks()}

    @app.get("/curator/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE_HTML

    @app.get("/curator/next")
    def next_pairs(n: int = 20) -> dict:
        labeled = set(store.fold_current().keys())
        scored = []
        for a, b in source.pairs():
            if pair_key(a.chunk_id, b.chunk_id) in labeled:
                continue
            scored.append(((a, b), signals_fn(a.text, b.text)))
            if len(scored) >= scan_limit:
                break
        ordered = order_backlog(scored, already_labeled=set())
        pairs = [
            {
                "src": a.chunk_id, "dst": b.chunk_id,
                "src_text": a.text, "dst_text": b.text, "signals": sig,
            }
            for (a, b), sig in ordered[:n]
        ]
        return {"pairs": pairs, "labeled": len(labeled), "remaining": len(ordered)}

    @app.get("/curator/readiness")
    def readiness_report() -> dict:
        rep = readiness(store, readiness_target)
        return {
            "new_issue_frame": rep.new_issue_frame,
            "new_agree": rep.new_agree,
            "new_unrelated": rep.new_unrelated,
            "target": {
                "issue_frame": rep.target.issue_frame,
                "agree": rep.target.agree,
                "unrelated": rep.target.unrelated,
            },
            "ready": rep.ready,
        }

    @app.post("/curator/label")
    def label(req: LabelRequest) -> dict:
        if req.relation not in _VALID_RELATIONS:
            raise HTTPException(status_code=422, detail=f"unknown relation {req.relation!r}")
        if req.relation == Relation.CONTRADICTS.value and not req.relation_class:
            raise HTTPException(
                status_code=422, detail="relation_class (story|issue_frame) required for contradicts"
            )
        if req.relation_class is not None and req.relation_class not in {"story", "issue_frame"}:
            raise HTTPException(status_code=422, detail=f"bad relation_class {req.relation_class!r}")

        prior = store.fold_current().get(pair_key(req.src_chunk_id, req.dst_chunk_id))
        src_text = text_by_id.get(req.src_chunk_id, "")
        dst_text = text_by_id.get(req.dst_chunk_id, "")
        rec = LabelRecord(
            id=str(uuid.uuid4()),
            src_chunk_id=req.src_chunk_id,
            dst_chunk_id=req.dst_chunk_id,
            relation=Relation(req.relation),
            relation_class=req.relation_class if req.relation == Relation.CONTRADICTS.value else None,
            rationale=req.rationale,
            curator=curator,
            ts=_now_iso(),
            supersedes=prior.id if prior is not None else None,
            src_anchor=sentence_anchor(src_text) if src_text else None,
            dst_anchor=sentence_anchor(dst_text) if dst_text else None,
        )
        store.append(rec)
        return {"ok": True, "id": rec.id}

    return app


PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>GIN Curator</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
 #progress{color:#666;margin-bottom:1rem}
 .panels{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
 .panel{border:1px solid #ccc;border-radius:6px;padding:1rem;background:#fafafa}
 .panel h3{margin:0 0 .5rem;font-size:.8rem;text-transform:uppercase;color:#888}
 #signals{font-family:ui-monospace,monospace;font-size:.85rem;color:#444;margin:1rem 0;padding:.5rem;background:#f0f0f0;border-radius:4px}
 .rel{margin:.2rem;padding:.5rem .8rem;border:1px solid #999;border-radius:4px;background:#fff;cursor:pointer}
 .rel.sel{background:#2a6;color:#fff;border-color:#2a6}
 #class-row{margin:.6rem 0;display:none}
 textarea{width:100%;min-height:3rem;margin:.5rem 0}
 #save{padding:.5rem 1.2rem;font-size:1rem}
 kbd{background:#eee;border:1px solid #bbb;border-radius:3px;padding:0 .3rem;font-size:.8rem}
</style></head><body>
<h2>GIN Curator</h2>
<div id="progress"></div>
<div class="panels">
  <div class="panel"><h3>A</h3><p id="a"></p></div>
  <div class="panel"><h3>B</h3><p id="b"></p></div>
</div>
<div id="signals"></div>
<div id="rels">
  <button class="rel" data-rel="contradicts">1 contradicts</button>
  <button class="rel" data-rel="corroborates">2 corroborates</button>
  <button class="rel" data-rel="supersedes">3 supersedes</button>
  <button class="rel" data-rel="related_untyped">4 related_untyped</button>
  <button class="rel" data-rel="unrelated">5 unrelated</button>
</div>
<div id="class-row">
  class: <label><input type="radio" name="cls" value="story"> story</label>
  <label><input type="radio" name="cls" value="issue_frame"> issue_frame</label>
</div>
<textarea id="rationale" placeholder="rationale (optional)"></textarea>
<div><button id="save">Save &amp; next <kbd>Enter</kbd></button></div>
<script>
const RELS=["contradicts","corroborates","supersedes","related_untyped","unrelated"];
let queue=[],cur=null,pending=null;
function fmt(x){return x==null?"\\u2013":Number(x).toFixed(3);}
async function loadNext(){
  const d=await (await fetch("/curator/next?n=20")).json();
  queue=d.pairs;
  let rtxt="";
  try{
    const r=await (await fetch("/curator/readiness")).json();
    const t=r.target;
    rtxt=`  |  issue_frame ${r.new_issue_frame}/${t.issue_frame} \\u00b7 agree ${r.new_agree}/${t.agree}`
      +` \\u00b7 unrelated ${r.new_unrelated}/${t.unrelated} \\u00b7 ${r.ready?"READY":"not ready"}`;
  }catch(e){}
  document.getElementById("progress").textContent=`labeled ${d.labeled} \\u00b7 remaining ${d.remaining}${rtxt}`;
  show();
}
function show(){
  pending=null;
  document.querySelectorAll(".rel").forEach(b=>b.classList.remove("sel"));
  document.getElementById("class-row").style.display="none";
  document.querySelectorAll("input[name=cls]").forEach(i=>i.checked=false);
  document.getElementById("rationale").value="";
  if(queue.length===0){loadNext();return;}
  cur=queue.shift();
  document.getElementById("a").textContent=cur.src_text;
  document.getElementById("b").textContent=cur.dst_text;
  const s=cur.signals;
  document.getElementById("signals").textContent=
    `cheap=${s.cheap_verdict}  cos=${fmt(s.cosine)}  p_contra=${fmt(s.nli_p_contra)}  same_story=${s.same_story}`;
}
function pick(rel){
  pending=rel;
  document.querySelectorAll(".rel").forEach(b=>b.classList.toggle("sel",b.dataset.rel===rel));
  document.getElementById("class-row").style.display=(rel==="contradicts")?"block":"none";
}
async function save(){
  if(!pending||!cur)return;
  let cls=null;
  if(pending==="contradicts"){
    const c=document.querySelector("input[name=cls]:checked");
    if(!c){alert("pick story or issue_frame");return;}
    cls=c.value;
  }
  await fetch("/curator/label",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({src_chunk_id:cur.src,dst_chunk_id:cur.dst,relation:pending,
      relation_class:cls,rationale:document.getElementById("rationale").value})});
  show();
}
document.querySelectorAll(".rel").forEach(b=>b.addEventListener("click",()=>pick(b.dataset.rel)));
document.getElementById("save").addEventListener("click",save);
document.addEventListener("keydown",e=>{
  const n=parseInt(e.key);
  if(n>=1&&n<=RELS.length){pick(RELS[n-1]);}
  else if(e.key==="Enter"){e.preventDefault();save();}
});
loadNext();
</script></body></html>
"""
