"""Create a self-contained viewer of recorded final episodes; no new rollout.

The viewer never loads neural weights, contacts services, or learns. Source
frames are the exact RGB observations recorded during the finite evaluation.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from arena.shared.integrity import read_verified_bytes, verify_checksum_manifest
from arena.universal.analysis import validate_episode_matrix
from arena.universal.protocol import GAMES, evaluation_matrix


TITLES = {"snake": "Snake", "pong": "Pong", "agar": "Agar-like",
          "flappy": "Flappy Bird", "racing": "Corrida"}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid recorded numeric field: {name}")
    return float(value)


def _flag(value, name):
    if type(value) is not bool:
        raise ValueError(f"invalid recorded boolean field: {name}")
    return value


def _q_vector(value, mask, name):
    if not isinstance(value, str):
        raise ValueError(f"recorded {name} must be the producer's JSON vector")
    values = json.loads(value)
    if not isinstance(values, list) or len(values) != 10:
        raise ValueError(f"invalid recorded {name} vector")
    for index, item in enumerate(values):
        if mask[index]:
            _number(item, name)
        elif item is not None:
            raise ValueError("unavailable control must have null recorded Q value")
    return values


def _recorded_arrays(raw):
    """Bound NPY headers before allocating arrays; no object or ZIP bomb load."""
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or {item.filename for item in entries} != {"rgb.npy", "next_rgb.npy"}:
            raise ValueError("recorded frame archive must contain exactly rgb and next_rgb")
        for item in entries:
            if item.file_size > 400*32*32*3+4096:
                raise ValueError("recorded frame member exceeds the finite episode size")
            with archive.open(item) as handle:
                version = np.lib.format.read_magic(handle)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("unsupported recorded frame header")
                shape, fortran, dtype = reader(handle, max_header_size=4096)
                if (dtype != np.dtype(np.uint8) or fortran or len(shape) != 4 or
                        shape[1:] != (32, 32, 3) or not 1 <= shape[0] <= 400 or
                        item.file_size-handle.tell() != math.prod(shape)):
                    raise ValueError("invalid recorded RGB header/layout")
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        frames, after = archive["rgb"], archive["next_rgb"]
    if frames.shape != after.shape:
        raise ValueError("recorded current/next frame dimensions differ")
    return frames, after


def _validate_trace(metadata, frames, after, cell, episode):
    steps = metadata.get("steps")
    if not isinstance(steps, list) or len(steps) != len(frames):
        raise ValueError("recorded trace/frame length mismatch")
    if int(episode["steps"]) != len(steps):
        raise ValueError("demonstration does not match its final matrix episode length")
    if not np.array_equal(after[:-1], frames[1:]):
        raise ValueError("recorded frame sequence is discontinuous")
    previous_mask, reward_total, clipped_total = None, 0., 0.
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or any(step.get(name) != value for name, value in cell.items()):
            raise ValueError("recorded trace identity differs from its fixed episode")
        if type(step.get("step")) is not int or step["step"] != index+1:
            raise ValueError("recorded steps must be contiguous from one")
        mask = step.get("action_mask")
        if (not isinstance(mask, list) or len(mask) != 10 or
                any(type(item) is not bool for item in mask) or not any(mask)):
            raise ValueError("invalid recorded mechanical action mask")
        if previous_mask is not None and mask != previous_mask:
            raise ValueError("recorded mechanical action mask changed with state")
        previous_mask = mask
        action = step.get("action")
        if type(action) is not int or not 0 <= action < 10 or not mask[action]:
            raise ValueError("recorded action is outside the available controls")
        reward = _number(step.get("reward"), "reward")
        clipped = _number(step.get("reward_clipped"), "reward_clipped")
        if not math.isclose(clipped, min(1., max(-1., reward)), abs_tol=1e-6):
            raise ValueError("recorded clipped reward is inconsistent")
        reward_total += reward
        clipped_total += clipped
        terminal, truncated = _flag(step.get("terminated"), "terminated"), _flag(step.get("truncated"), "truncated")
        if terminal and truncated or (index < len(steps)-1 and (terminal or truncated)):
            raise ValueError("recorded termination occurs before the final observation")
        if index == len(steps)-1 and not (terminal or truncated):
            raise ValueError("recorded clip is a prefix, not the completed final episode")
        if (step.get("rgb_sha256") != hashlib.sha256(frames[index].tobytes()).hexdigest() or
                step.get("next_rgb_sha256") != hashlib.sha256(after[index].tobytes()).hexdigest()):
            raise ValueError("recorded RGB does not match its per-step trace hash")
        direct = _q_vector(step.get("q_direct"), mask, "q_direct")
        model = _q_vector(step.get("model_values"), mask, "model_values")
        mixed = _q_vector(step.get("q_mixed"), mask, "q_mixed")
        legal = [i for i, available in enumerate(mask) if available]
        if any(not math.isclose(mixed[i], .8*direct[i]+.2*model[i], rel_tol=1e-6, abs_tol=1e-5) for i in legal):
            raise ValueError("recorded policy mixture is inconsistent")
        if action != max(legal, key=lambda i: mixed[i]):
            raise ValueError("recorded action differs from the recorded greedy shared policy")
    if (not math.isclose(reward_total, float(episode["return"]), rel_tol=1e-6, abs_tol=1e-6) or
            not math.isclose(clipped_total, float(episode["return_clipped"]), rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError("recorded rewards do not match the final matrix return")
    if (int(episode["terminated"]) != int(steps[-1]["terminated"]) or
            int(episode["truncated"]) != int(steps[-1]["truncated"])):
        raise ValueError("recorded termination differs from its final matrix episode")


def build_viewer(evaluation_directory, output, *, expected_manifest_sha256):
    root, destination = Path(evaluation_directory), Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("viewer output must be outside the authenticated evaluation tree")
    read_verified_bytes(root/"SHA256SUMS", expected_sha256=expected_manifest_sha256,
                        max_bytes=1024*1024)
    hashes = verify_checksum_manifest(root)
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if files != set(hashes)|{"SHA256SUMS"}:
        raise ValueError("unlisted evaluation file outside the authenticated manifest")
    required_demo_files = {f"demonstrations/{game}/{name}" for game in GAMES for name in ("frames.npz", "trajectory.json")}
    if {name for name in hashes if name.startswith("demonstrations/")} != required_demo_files:
        raise ValueError("exactly the five fixed demonstration pairs are required")
    status = json.loads(read_verified_bytes(root/"EVALUATION_STATUS.json",
                                            expected_sha256=hashes["EVALUATION_STATUS.json"]))
    if (status.get("status") != "SUCCESS" or status.get("test_only") is not False or
            status.get("completed_episodes") != len(evaluation_matrix())):
        raise ValueError("viewer requires authentic completed evaluation, not a test fixture")
    episodes_raw = read_verified_bytes(root/"episodes.csv", expected_sha256=hashes["episodes.csv"], max_bytes=16*1024*1024)
    indexed, matrix = validate_episode_matrix(list(csv.DictReader(StringIO(episodes_raw.decode("utf-8")))))
    if matrix["status"] != "PASS":
        raise ValueError("viewer requires the complete unique preregistered final matrix")
    episodes = {tuple(vars(key).values()): row for key, row in indexed.items()}
    clips = []
    network_hashes = set()
    for game in GAMES:
        prefix = f"demonstrations/{game}/"
        raw = read_verified_bytes(root/(prefix+"frames.npz"),
                                   expected_sha256=hashes[prefix+"frames.npz"], max_bytes=8*1024*1024)
        metadata = json.loads(read_verified_bytes(root/(prefix+"trajectory.json"),
                              expected_sha256=hashes[prefix+"trajectory.json"], max_bytes=4*1024*1024))
        cell = metadata["cell"]
        if (set(cell) != {"game", "stage", "policy", "model_seed", "environment_seed"} or
                cell["game"] != game or cell["stage"] != "after_adaptation" or
                cell["policy"] != "shared" or cell["model_seed"] != 2207 or
                cell["environment_seed"] != 920001):
            raise ValueError("demonstration is not the preregistered fixed episode")
        digest = metadata.get("network_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid recorded network SHA-256")
        if not isinstance(metadata.get("environment_id"), str) or not re.fullmatch(rf"visual-{game}-v[1-9][0-9]*", metadata["environment_id"]):
            raise ValueError("invalid recorded environment identity")
        episode = episodes[(cell["stage"], cell["policy"], game, cell["model_seed"], cell["environment_seed"])]
        if episode.get("network_sha256") != digest or episode.get("environment_id") != metadata["environment_id"]:
            raise ValueError("demonstration network/environment differs from the final matrix")
        network_hashes.add(digest)
        frames, after = _recorded_arrays(raw)
        _validate_trace(metadata, frames, after, cell, episode)
        # Include the final observation as well; no simulator action is repeated.
        rendered = np.concatenate((frames, after[-1:]), axis=0)
        clips.append({"game": game, "title": TITLES[game], "cell": cell,
                      "network_sha256": metadata["network_sha256"],
                      "environment_id": metadata["environment_id"],
                      "frames": [base64.b64encode(frame.tobytes()).decode("ascii") for frame in rendered],
                      "steps": metadata["steps"]})
    if len(network_hashes) != 1:
        raise ValueError("all five demonstrations must use the SAME shared network weights")
    payload = json.dumps(clips, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = _HTML.replace("__RECORDED_DATA__", payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(html)
    return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "recorded_games": len(clips), "new_episodes": 0, "model_seed": 2207,
            "environment_seed": 920001, "training_started_by_viewer": False}


_HTML = r'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NOVA Neural Arena — reprodução da avaliação</title>
<style>
:root{color-scheme:dark;font:16px system-ui;background:#10141e;color:#e6edf7}
body{margin:0}main{max-width:1000px;margin:auto;padding:24px}h1{font-size:1.65rem;margin:8px 0}
p{color:#b3bfd3;line-height:1.5}.layout{display:grid;grid-template-columns:minmax(280px,1fr) minmax(220px,320px);gap:24px}
.card{background:#1b2231;border:1px solid #34415b;border-radius:16px;padding:20px}
canvas{display:block;width:min(100%,512px);height:auto;image-rendering:pixelated;margin:auto;background:#12161e}
label{display:block;margin:12px 0 6px}button,select,input{font:inherit}button,select{background:#2b3952;color:#f1f6ff;border:1px solid #61789d;border-radius:8px;padding:10px;min-height:44px}
button{cursor:pointer}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #64c6ff;outline-offset:3px}
input[type=range]{width:100%}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}dl{display:grid;grid-template-columns:1fr 1fr;gap:12px}dt{color:#b3bfd3}dd{margin:0;text-align:right;font-weight:600}
details{margin-top:20px;overflow-wrap:anywhere}small{color:#a6b4cb}summary{cursor:pointer}.badge{color:#8dd8ff;font-weight:600;font-size:.85rem}
@media(max-width:680px){main{padding:16px}.layout{grid-template-columns:1fr}.card{padding:16px}}
</style><main><span class="badge">NOVA NEURAL ARENA</span><h1>Reprodução da avaliação</h1>
<p>Veja as decisões registradas da mesma política neural nos cinco jogos. Esta página reproduz os dados e não inicia treinamento. O resultado científico e suas limitações estão no relatório da missão.</p>
<div class="layout"><section class="card" aria-label="Jogo gravado"><canvas id="frame" width="32" height="32" role="img" aria-label="Quadro da partida gravada"></canvas>
<div class="controls"><button id="play" type="button">Reproduzir</button><button id="back" type="button">Anterior</button><button id="next" type="button">Próximo</button><button id="start" type="button">Início</button></div>
<label for="timeline">Quadro da gravação</label><input id="timeline" type="range" min="0" max="0" value="0"><div id="position" aria-live="polite"></div>
</section><aside class="card"><label for="game">Jogo</label><select id="game"></select><label for="speed">Velocidade da reprodução</label><select id="speed"><option value=".5">0,5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
<dl><dt>Ação</dt><dd id="action"></dd><dt>Recompensa do passo</dt><dd id="reward"></dd><dt>Fim do episódio</dt><dd id="ending"></dd></dl>
<details><summary>Origem da gravação</summary><p id="provenance"></p><small>Réplica e episódio foram fixados antes da avaliação. A reprodução não escolhe a melhor partida e não comprova, sozinha, aprendizagem ou transferência.</small></details>
</aside></div></main><script type="application/json" id="data">__RECORDED_DATA__</script>
<script>
'use strict';const clips=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),ctx=$('frame').getContext('2d');
const names=['Parado','Cima','Baixo','Esquerda','Direita','Cima/esquerda','Cima/direita','Baixo/esquerda','Baixo/direita','Botão'];
let clip=0,index=0,playing=false,last=0;for(const [i,c] of clips.entries()){const o=document.createElement('option');o.value=String(i);o.textContent=c.title;$('game').append(o)}
function draw(){const c=clips[clip],raw=atob(c.frames[index]),rgba=new Uint8ClampedArray(32*32*4);for(let p=0;p<1024;p++){rgba[4*p]=raw.charCodeAt(3*p);rgba[4*p+1]=raw.charCodeAt(3*p+1);rgba[4*p+2]=raw.charCodeAt(3*p+2);rgba[4*p+3]=255}ctx.putImageData(new ImageData(rgba,32,32),0,0);$('timeline').max=c.frames.length-1;$('timeline').value=index;$('position').textContent=`${c.title} · quadro ${index+1} de ${c.frames.length}`;const s=c.steps[Math.min(index,c.steps.length-1)];$('action').textContent=index<c.steps.length?(names[s.action]||String(s.action)):'Encerrada';$('reward').textContent=Number(s.reward).toLocaleString('pt-BR',{maximumFractionDigits:4});$('ending').textContent=s.terminated?'Término':s.truncated?'Limite de passos':'Em andamento';$('provenance').textContent=`Modelo ${c.cell.model_seed}; seed do ambiente ${c.cell.environment_seed}; ${c.environment_id}; SHA-256 da rede ${c.network_sha256}`;}
function stop(){playing=false;$('play').textContent='Reproduzir'}function seek(n){index=Math.max(0,Math.min(clips[clip].frames.length-1,n));draw()}
$('game').onchange=()=>{stop();clip=Number($('game').value);seek(0)};$('timeline').oninput=()=>{stop();seek(Number($('timeline').value))};$('back').onclick=()=>{stop();seek(index-1)};$('next').onclick=()=>{stop();seek(index+1)};$('start').onclick=()=>{stop();seek(0)};$('play').onclick=()=>{if(playing){stop();return}if(index===clips[clip].frames.length-1)seek(0);playing=true;last=0;$('play').textContent='Pausar';requestAnimationFrame(tick)};
function tick(now){if(!playing)return;if(!last)last=now;if(now-last>=1000/(20*Number($('speed').value))){last=now;seek(index+1);if(index===clips[clip].frames.length-1){stop();return}}requestAnimationFrame(tick)}draw();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(build_viewer(args.evaluation, args.output,
                      expected_manifest_sha256=args.expected_manifest_sha256), indent=2))


if __name__ == "__main__":
    main()
