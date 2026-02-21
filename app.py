#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# MoviePy 2.x
from moviepy import ImageClip, AudioFileClip, VideoFileClip, concatenate_videoclips, CompositeAudioClip

APP = FastAPI(title="Mini Node Editor (n8n-like, lightweight)")

# ====== CONFIG SEGURA + LIGERA ======
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = (BASE_DIR / "WORKSPACE").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Evita que el usuario use rutas fuera del workspace
def safe_path(user_rel: str) -> Path:
    user_rel = str(user_rel).strip().lstrip("\\/")  # evita "C:\..."
    p = (WORKSPACE / user_rel).resolve()
    if not str(p).startswith(str(WORKSPACE)):
        raise ValueError("Ruta fuera de WORKSPACE (bloqueado).")
    return p

def rel_from_workspace(p: Path) -> str:
    return str(p.relative_to(WORKSPACE)).replace("\\", "/")

# ====== MODELOS ======
class PortLink(BaseModel):
    from_node: str
    from_port: str
    to_node: str
    to_port: str

class Node(BaseModel):
    id: str
    type: str
    position: Dict[str, float] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)

class Workflow(BaseModel):
    nodes: List[Node]
    links: List[PortLink]

# ====== MOTOR ======
class ExecutionContext:
    def __init__(self):
        self.outputs: Dict[str, Dict[str, Any]] = {}  # node_id -> {port:value}
        self.logs: List[Dict[str, Any]] = []
        self.step = 0

def build_input_map(workflow: Workflow) -> Dict[str, Dict[str, Dict[str, str]]]:
    m: Dict[str, Dict[str, Dict[str, str]]] = {}
    for link in workflow.links:
        m.setdefault(link.to_node, {})
        m[link.to_node][link.to_port] = {"from_node": link.from_node, "from_port": link.from_port}
    return m

def topo_sort(workflow: Workflow) -> List[Node]:
    nodes_by_id = {n.id: n for n in workflow.nodes}
    indeg = {n.id: 0 for n in workflow.nodes}
    adj = {n.id: [] for n in workflow.nodes}

    for link in workflow.links:
        if link.from_node in nodes_by_id and link.to_node in nodes_by_id:
            adj[link.from_node].append(link.to_node)
            indeg[link.to_node] += 1

    q = [nid for nid, d in indeg.items() if d == 0]
    ordered = []
    while q:
        nid = q.pop(0)
        ordered.append(nid)
        for nxt in adj[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(ordered) != len(workflow.nodes):
        raise ValueError("Grafo con ciclos o links inválidos. No se puede ejecutar.")
    return [nodes_by_id[nid] for nid in ordered]

def resolve_input(node_id: str, port: str, input_map: Dict[str, Dict[str, Dict[str, str]]], ctx: ExecutionContext):
    src = input_map.get(node_id, {}).get(port)
    if not src:
        return None
    return ctx.outputs.get(src["from_node"], {}).get(src["from_port"])

def node_log(ctx: ExecutionContext, node: Node, status: str, extra: Optional[Dict[str, Any]] = None):
    ctx.step += 1
    entry = {"step": ctx.step, "node": node.id, "type": node.type, "status": status}
    if extra:
        entry.update(extra)
    ctx.logs.append(entry)

# ====== NODOS ======

def run_node_text(node: Node, ctx: ExecutionContext) -> Dict[str, Any]:
    """
    Text node:
      data.text -> output: text
    """
    text = str(node.data.get("text", ""))
    return {"text": text}

def run_node_readfile(node: Node, ctx: ExecutionContext) -> Dict[str, Any]:
    """
    ReadFile:
      data.path (rel) -> output: text
    """
    relp = node.data.get("path")
    if not relp:
        raise ValueError("ReadFile requiere data.path")
    p = safe_path(relp)
    if not p.exists():
        raise ValueError(f"Archivo no existe: {relp}")
    text = p.read_text(encoding="utf-8", errors="replace")
    # Recorte para no tragar RAM
    max_chars = int(node.data.get("max_chars", 20000))
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (recortado)"
    return {"text": text}

def run_node_writefile(node: Node, input_map, ctx: ExecutionContext) -> Dict[str, Any]:
    """
    WriteFile:
      input: text (opcional, si no usa data.text)
      data.path (rel)
      data.overwrite (bool)
    output: path
    """
    relp = node.data.get("path")
    if not relp:
        raise ValueError("WriteFile requiere data.path")

    text_in = resolve_input(node.id, "text", input_map, ctx)
    content = text_in if text_in is not None else str(node.data.get("text", ""))

    overwrite = bool(node.data.get("overwrite", True))
    p = safe_path(relp)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise ValueError("WriteFile: el archivo existe y overwrite=false")

    p.write_text(str(content), encoding="utf-8")
    return {"path": rel_from_workspace(p)}

def run_node_mergeaudio(node: Node, input_map, ctx: ExecutionContext) -> Dict[str, Any]:
    """
    MergeAudio:
      inputs: a (ruta), b (ruta)
      data.vol_a (0..1), data.vol_b (0..1)
      data.output (rel) -> mp3
    output: audio
    """
    a_rel = resolve_input(node.id, "a", input_map, ctx) or node.data.get("a")
    b_rel = resolve_input(node.id, "b", input_map, ctx) or node.data.get("b")
    if not a_rel or not b_rel:
        raise ValueError("MergeAudio requiere inputs a y b")

    out_rel = node.data.get("output", "out/merged.mp3")
    vol_a = float(node.data.get("vol_a", 1.0))
    vol_b = float(node.data.get("vol_b", 1.0))

    a = safe_path(a_rel); b = safe_path(b_rel)
    outp = safe_path(out_rel); outp.parent.mkdir(parents=True, exist_ok=True)
    if not a.exists(): raise ValueError(f"Audio A no existe: {a_rel}")
    if not b.exists(): raise ValueError(f"Audio B no existe: {b_rel}")

    # MoviePy AudioFileClip
    clip_a = AudioFileClip(str(a)).with_volume_scaled(vol_a)
    clip_b = AudioFileClip(str(b)).with_volume_scaled(vol_b)

    # Duración: usa la mayor, recorta la otra
    dur = max(clip_a.duration, clip_b.duration)
    clip_a = clip_a.with_duration(dur)
    clip_b = clip_b.with_duration(dur)

    mix = CompositeAudioClip([clip_a, clip_b])
    mix.write_audiofile(str(outp))

    # Cerrar
    mix.close()
    clip_a.close()
    clip_b.close()

    return {"audio": rel_from_workspace(outp)}

def run_node_videoclip(node: Node, input_map, ctx: ExecutionContext) -> Dict[str, Any]:
    """
    VideoClip:
      inputs: image (ruta), audio (ruta)
      config: duration (optional), fps, output (rel), codec, audio_codec, preset, threads
      output: video
    """
    image_rel = resolve_input(node.id, "image", input_map, ctx) or node.data.get("image")
    audio_rel = resolve_input(node.id, "audio", input_map, ctx) or node.data.get("audio")
    if not image_rel or not audio_rel:
        raise ValueError("VideoClip necesita image y audio")

    duration = node.data.get("duration", "")
    fps = int(node.data.get("fps", 24))
    out_rel = node.data.get("output", "out/videoclip.mp4")
    codec = node.data.get("codec", "libx264")
    audio_codec = node.data.get("audio_codec", "aac")

    # Optimización para tu PC: preset ultrafast + threads bajos
    preset = node.data.get("preset", "ultrafast")
    threads = int(node.data.get("threads", 2))

    img = safe_path(image_rel); aud = safe_path(audio_rel)
    outp = safe_path(out_rel); outp.parent.mkdir(parents=True, exist_ok=True)
    if not img.exists(): raise ValueError(f"Imagen no existe: {image_rel}")
    if not aud.exists(): raise ValueError(f"Audio no existe: {audio_rel}")

    audio_clip = AudioFileClip(str(aud))
    use_duration = float(duration) if str(duration).strip() else float(audio_clip.duration)

    img_clip = ImageClip(str(img)).with_duration(use_duration)
    video_clip = img_clip.with_audio(audio_clip)

    video_clip.write_videofile(
        str(outp),
        codec=codec,
        audio_codec=audio_codec,
        fps=fps,
        preset=preset,
        threads=threads
    )

    video_clip.close()
    img_clip.close()
    audio_clip.close()

    return {"video": rel_from_workspace(outp)}

def run_node_concatvideo(node: Node, input_map, ctx: ExecutionContext) -> Dict[str, Any]:
    """
    ConcatVideo:
      input: videos (lista JSON o string con ; separados) o recibir por link a "videos"
      data.output
      data.fps
    output: video
    """
    vids_in = resolve_input(node.id, "videos", input_map, ctx) or node.data.get("videos")
    if not vids_in:
        raise ValueError("ConcatVideo requiere videos")

    # Parse lista
    videos_list = []
    if isinstance(vids_in, list):
        videos_list = vids_in
    else:
        s = str(vids_in).strip()
        if s.startswith("["):
            videos_list = json.loads(s)
        else:
            videos_list = [x.strip() for x in s.split(";") if x.strip()]

    out_rel = node.data.get("output", "out/concat.mp4")
    fps = int(node.data.get("fps", 24))
    preset = node.data.get("preset", "ultrafast")
    threads = int(node.data.get("threads", 2))

    outp = safe_path(out_rel); outp.parent.mkdir(parents=True, exist_ok=True)

    clips = []
    try:
        for relv in videos_list:
            vp = safe_path(relv)
            if not vp.exists():
                raise ValueError(f"Video no existe: {relv}")
            clips.append(VideoFileClip(str(vp)))

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            str(outp),
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset=preset,
            threads=threads
        )
        final.close()
    finally:
        for c in clips:
            try: c.close()
            except: pass

    return {"video": rel_from_workspace(outp)}

NODE_RUNNERS = {
    "Text": run_node_text,
    "ReadFile": run_node_readfile,
    "WriteFile": run_node_writefile,
    "MergeAudio": run_node_mergeaudio,
    "VideoClip": run_node_videoclip,
    "ConcatVideo": run_node_concatvideo,
}

def execute_workflow(workflow: Workflow) -> Dict[str, Any]:
    ctx = ExecutionContext()
    input_map = build_input_map(workflow)
    ordered = topo_sort(workflow)

    for node in ordered:
        node_log(ctx, node, "running")
        runner = NODE_RUNNERS.get(node.type)
        if not runner:
            raise ValueError(f"Nodo no soportado: {node.type}")

        try:
            if node.type in ("WriteFile", "MergeAudio", "VideoClip", "ConcatVideo"):
                out = runner(node, input_map, ctx)
            elif node.type in ("Text", "ReadFile"):
                out = runner(node, ctx)
            else:
                out = runner(node, input_map, ctx)

            ctx.outputs[node.id] = out
            node_log(ctx, node, "done", {"outputs": out})
        except Exception as e:
            node_log(ctx, node, "error", {"error": str(e)})
            raise

    return {
        "ok": True,
        "workspace": str(WORKSPACE),
        "outputs": ctx.outputs,
        "logs": ctx.logs
    }

# ====== API ======
@APP.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(UI_HTML)

@APP.post("/run")
def run_workflow_api(payload: Workflow):
    try:
        result = execute_workflow(payload)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ====== UI (HTML + JS, sin libs) ======
UI_HTML = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Mini Node Editor (Light)</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
body{margin:0;font-family:system-ui,Arial;background:#0b0f14;color:#e6edf3}
header{display:flex;gap:10px;align-items:center;padding:10px 12px;background:#0f1621;border-bottom:1px solid #1f2a3a;flex-wrap:wrap}
button{background:#1f6feb;border:0;color:#fff;padding:8px 10px;border-radius:8px;cursor:pointer}
button.secondary{background:#263445}
.wrap{display:grid;grid-template-columns:1fr 380px;height:calc(100vh - 58px)}
#canvas{position:relative;overflow:hidden;background:radial-gradient(#0f1621 1px,transparent 1px);background-size:24px 24px}
#side{border-left:1px solid #1f2a3a;padding:10px;overflow:auto;background:#0f1621}
.node{position:absolute;width:300px;background:#111a27;border:1px solid #22314a;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.35)}
.title{padding:10px;font-weight:700;background:#0f1621;border-bottom:1px solid #22314a;border-radius:12px 12px 0 0;cursor:move}
.body{padding:10px;display:grid;gap:8px}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px}
.row label{font-size:12px;opacity:.85}
.row input,.row textarea{width:190px;background:#0b0f14;border:1px solid #22314a;color:#e6edf3;border-radius:8px;padding:6px 8px}
.row textarea{height:64px;resize:vertical}
.ports{display:flex;gap:10px}
.col{flex:1}
.port{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:12px}
.dot{width:12px;height:12px;border-radius:50%;border:2px solid #1f6feb;background:#0b0f14;cursor:pointer}
.dot.out{border-color:#2ea043}
pre{white-space:pre-wrap;background:#0b0f14;border:1px solid #22314a;border-radius:10px;padding:10px;font-size:12px}
.hr{border-top:1px solid #22314a;margin:10px 0}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border:1px solid #22314a;border-radius:999px;opacity:.9}
svg{position:absolute;inset:0;pointer-events:none}
.small{font-size:12px;opacity:.9}
</style>
</head>
<body>
<header>
  <button onclick="addNode('Text')">+ Text</button>
  <button onclick="addNode('ReadFile')">+ ReadFile</button>
  <button onclick="addNode('WriteFile')">+ WriteFile</button>
  <button onclick="addNode('MergeAudio')">+ MergeAudio</button>
  <button onclick="addNode('VideoClip')">+ VideoClip</button>
  <button onclick="addNode('ConcatVideo')">+ ConcatVideo</button>
  <button class="secondary" onclick="exportJSON()">Export</button>
  <button class="secondary" onclick="importJSON()">Import</button>
  <button onclick="runWorkflow()">▶ Ejecutar</button>
  <span class="badge">WORKSPACE: ./WORKSPACE</span>
</header>

<div class="wrap">
  <div id="canvas"><svg id="wires"></svg></div>
  <div id="side">
    <div class="small"><b>Guía rápida</b></div>
    <pre>
- Todas las rutas son relativas a WORKSPACE.
- Optimizado para PCs lentos: ejecución secuencial, preset ultrafast, threads=2.
- VideoClip: image + audio -> output mp4
- MergeAudio: a + b -> output mp3
- ConcatVideo: lista de videos (separados por ;) -> output mp4
    </pre>
    <div class="hr"></div>
    <div class="small"><b>Workflow JSON</b></div>
    <pre id="jsonView"></pre>
    <div class="hr"></div>
    <div class="small"><b>Logs</b></div>
    <pre id="logView"></pre>
  </div>
</div>

<script>
const canvas = document.getElementById("canvas");
const wires = document.getElementById("wires");
const jsonView = document.getElementById("jsonView");
const logView = document.getElementById("logView");

let state = { nodes: [], links: [] };
let drag = null;
let linkDraft = null; // {from_node, from_port, x,y}

function uid(prefix="n"){ return prefix + Math.random().toString(16).slice(2,10); }

function defaults(type){
  if(type==="Text") return { text: "Hola mundo" };
  if(type==="ReadFile") return { path: "notas/entrada.txt", max_chars: 20000 };
  if(type==="WriteFile") return { path: "notas/salida.txt", overwrite: true, text: "" };
  if(type==="MergeAudio") return { a:"assets/a.mp3", b:"assets/b.mp3", vol_a:1.0, vol_b:0.6, output:"out/merged.mp3" };
  if(type==="VideoClip") return { image:"assets/image.png", audio:"assets/audio.mp3", duration:"", fps:24, output:"out/videoclip.mp4", preset:"ultrafast", threads:2 };
  if(type==="ConcatVideo") return { videos:"out/v1.mp4;out/v2.mp4", fps:24, output:"out/concat.mp4", preset:"ultrafast", threads:2 };
  return {};
}

function addNode(type){
  const id = uid(type.slice(0,2).toLowerCase()+"_");
  state.nodes.push({ id, type, position:{x:60+state.nodes.length*30,y:60+state.nodes.length*30}, data: defaults(type) });
  render();
}

function esc(s){ return (""+s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;"); }

function portRow(node, dir, port){
  const cls = dir==="out" ? "dot out" : "dot";
  return `<div class="port">${dir==="in" ? `<div class="${cls}" data-dir="in" data-port="${port}"></div><span>${port}</span>`
                                        : `<span>${port}</span><div class="${cls}" data-dir="out" data-port="${port}"></div>`}</div>`;
}

function nodePorts(type){
  // entradas/salidas por nodo
  if(type==="Text") return { ins:[], outs:["text"] };
  if(type==="ReadFile") return { ins:[], outs:["text"] };
  if(type==="WriteFile") return { ins:["text"], outs:["path"] };
  if(type==="MergeAudio") return { ins:["a","b"], outs:["audio"] };
  if(type==="VideoClip") return { ins:["image","audio"], outs:["video"] };
  if(type==="ConcatVideo") return { ins:["videos"], outs:["video"] };
  return { ins:[], outs:[] };
}

function nodeFields(node){
  const t = node.type;
  const d = node.data;

  function inputRow(label, key, type="text"){
    return `<div class="row"><label>${label}</label><input data-k="${key}" type="${type}" value="${esc(d[key] ?? "")}"/></div>`;
  }
  function textareaRow(label, key){
    return `<div class="row"><label>${label}</label><textarea data-k="${key}">${esc(d[key] ?? "")}</textarea></div>`;
  }

  if(t==="Text"){
    return textareaRow("text","text");
  }
  if(t==="ReadFile"){
    return inputRow("path","path") + inputRow("max_chars","max_chars","number");
  }
  if(t==="WriteFile"){
    return inputRow("path","path") + inputRow("overwrite","overwrite") + textareaRow("text(fallback)","text");
  }
  if(t==="MergeAudio"){
    return inputRow("a","a") + inputRow("b","b") + inputRow("vol_a","vol_a","number") + inputRow("vol_b","vol_b","number") + inputRow("output","output");
  }
  if(t==="VideoClip"){
    return inputRow("image","image") + inputRow("audio","audio") + inputRow("duration","duration") + inputRow("fps","fps","number") + inputRow("output","output") + inputRow("preset","preset") + inputRow("threads","threads","number");
  }
  if(t==="ConcatVideo"){
    return inputRow("videos","videos") + inputRow("fps","fps","number") + inputRow("output","output") + inputRow("preset","preset") + inputRow("threads","threads","number");
  }
  return "";
}

function nodeEl(node){
  const el = document.createElement("div");
  el.className="node";
  el.style.left=node.position.x+"px";
  el.style.top=node.position.y+"px";
  el.dataset.id=node.id;

  const ports = nodePorts(node.type);
  const ins = ports.ins.map(p=>portRow(node,"in",p)).join("");
  const outs = ports.outs.map(p=>portRow(node,"out",p)).join("");

  el.innerHTML = `
    <div class="title">${node.type} <span style="opacity:.7;font-weight:500">(${node.id})</span></div>
    <div class="body">
      <div class="ports">
        <div class="col"><div class="small" style="opacity:.85;margin-bottom:4px;">Entradas</div>${ins || "<div class='small' style='opacity:.6'>(none)</div>"}</div>
        <div class="col"><div class="small" style="opacity:.85;margin-bottom:4px;">Salidas</div>${outs || "<div class='small' style='opacity:.6'>(none)</div>"}</div>
      </div>
      <div class="hr"></div>
      ${nodeFields(node)}
      <div class="row"><button class="secondary" data-act="delete">Eliminar</button></div>
    </div>
  `;

  const title = el.querySelector(".title");
  title.addEventListener("mousedown", (e)=>{
    drag={id:node.id, dx:e.clientX-node.position.x, dy:e.clientY-node.position.y};
  });

  el.querySelectorAll("input,textarea").forEach(inp=>{
    inp.addEventListener("input", ()=>{
      const k = inp.dataset.k;
      let v = inp.value;
      if(["fps","threads","max_chars"].includes(k)) v = parseInt(v||"0",10);
      if(["vol_a","vol_b"].includes(k)) v = parseFloat(v||"1");
      node.data[k]=v;
      updateSide();
    });
  });

  el.querySelector('[data-act="delete"]').addEventListener("click", ()=>{
    state.nodes = state.nodes.filter(n=>n.id!==node.id);
    state.links = state.links.filter(l=>l.from_node!==node.id && l.to_node!==node.id);
    render();
  });

  el.querySelectorAll(".dot").forEach(dot=>{
    dot.addEventListener("mousedown", (e)=>{
      e.stopPropagation();
      const port = dot.dataset.port;
      const dir = dot.dataset.dir;
      const nid = node.id;

      const rect = dot.getBoundingClientRect();
      const cx = rect.left + rect.width/2 + canvas.scrollLeft - canvas.getBoundingClientRect().left;
      const cy = rect.top + rect.height/2 + canvas.scrollTop - canvas.getBoundingClientRect().top;

      if(dir==="out"){
        linkDraft = {from_node:nid, from_port:port, x:cx, y:cy};
      }
    });

    dot.addEventListener("mouseup", (e)=>{
      e.stopPropagation();
      if(!linkDraft) return;
      const port = dot.dataset.port;
      const dir = dot.dataset.dir;
      const nid = node.id;
      if(dir==="in"){
        state.links.push({from_node:linkDraft.from_node, from_port:linkDraft.from_port, to_node:nid, to_port:port});
        linkDraft=null;
        render();
      }
    });
  });

  return el;
}

function portCenter(nodeId, dir, port){
  const el = canvas.querySelector(`.node[data-id="${nodeId}"] .dot[data-dir="${dir}"][data-port="${port}"]`);
  if(!el) return null;
  const cRect = canvas.getBoundingClientRect();
  const r = el.getBoundingClientRect();
  const x = (r.left + r.width/2) - cRect.left + canvas.scrollLeft;
  const y = (r.top + r.height/2) - cRect.top + canvas.scrollTop;
  return {x,y};
}

function svgCurve(x1,y1,x2,y2,draft=false){
  const path = document.createElementNS("http://www.w3.org/2000/svg","path");
  const dx = Math.max(60, Math.abs(x2-x1)*0.5);
  const c1x = x1 + dx, c2x = x2 - dx;
  path.setAttribute("d", `M ${x1} ${y1} C ${c1x} ${y1}, ${c2x} ${y2}, ${x2} ${y2}`);
  path.setAttribute("fill","none");
  path.setAttribute("stroke", draft ? "#8b949e" : "#58a6ff");
  path.setAttribute("stroke-width","3");
  path.setAttribute("opacity", draft ? "0.6" : "0.9");
  return path;
}

function drawWires(mx=null,my=null){
  wires.innerHTML="";
  state.links.forEach(l=>{
    const a = portCenter(l.from_node,"out",l.from_port);
    const b = portCenter(l.to_node,"in",l.to_port);
    if(!a||!b) return;
    wires.appendChild(svgCurve(a.x,a.y,b.x,b.y,false));
  });
  if(linkDraft && mx!==null){
    wires.appendChild(svgCurve(linkDraft.x, linkDraft.y, mx, my, true));
  }
}

function updateSide(){
  jsonView.textContent = JSON.stringify(state,null,2);
}

function render(){
  canvas.querySelectorAll(".node").forEach(n=>n.remove());
  state.nodes.forEach(n=>canvas.appendChild(nodeEl(n)));
  drawWires();
  updateSide();
}

window.addEventListener("mousemove",(e)=>{
  if(drag){
    const node = state.nodes.find(n=>n.id===drag.id);
    if(node){
      node.position.x = e.clientX - drag.dx;
      node.position.y = e.clientY - drag.dy;
      const el = canvas.querySelector(`.node[data-id="${node.id}"]`);
      if(el){ el.style.left=node.position.x+"px"; el.style.top=node.position.y+"px"; }
      drawWires();
      updateSide();
    }
  }
  if(linkDraft){
    const cRect = canvas.getBoundingClientRect();
    const mx = e.clientX - cRect.left + canvas.scrollLeft;
    const my = e.clientY - cRect.top + canvas.scrollTop;
    drawWires(mx,my);
  }
});

window.addEventListener("mouseup", ()=>{
  drag=null;
  if(linkDraft){ linkDraft=null; drawWires(); }
});

function exportJSON(){
  const blob = new Blob([JSON.stringify(state,null,2)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="workflow.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

function importJSON(){
  const inp=document.createElement("input");
  inp.type="file"; inp.accept="application/json";
  inp.onchange=()=>{
    const f=inp.files[0];
    const r=new FileReader();
    r.onload=()=>{
      try{ state=JSON.parse(r.result); render(); }
      catch(e){ alert("JSON inválido"); }
    };
    r.readAsText(f);
  };
  inp.click();
}

async function runWorkflow(){
  logView.textContent = "Ejecutando...\n";
  try{
    const res = await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(state)});
    const data = await res.json();
    if(!res.ok){
      logView.textContent += "ERROR:\n" + (data.detail || JSON.stringify(data)) + "\n";
      return;
    }
    logView.textContent += JSON.stringify(data,null,2);
  }catch(e){
    logView.textContent += "ERROR: " + e.toString();
  }
}

render();
</script>
</body>
</html>
"""

# ====== RUN ======
# uvicorn app:APP --reload