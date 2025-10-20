# arenna_viewer.py
# Visualizador web (Flask) de DVR Intelbras por RTSP com configuração via página.
# - Página inicial: IP/usuário/senha + seleção de canais (1..16)
# - Conectar -> inicia threads RTSP e abre tela com ARENNA no topo e layouts (row, 2x2 das 4 primeiras, 4x4 todas)
# - Streams servidos como MJPEG (compatível com desktop e celular na mesma LAN)

import os, time, threading, urllib.parse, atexit, socket, math
from datetime import datetime
from typing import Tuple, Dict, List
import cv2, numpy as np
from flask import Flask, Response, render_template_string, request, redirect, url_for

# ==========================
# Utilidades
# ==========================
def get_lan_ip(fallback="127.0.0.1", probe_host="192.168.0.1"):
    """Descobre o IP LAN desta máquina para você abrir no celular."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((probe_host, 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return fallback

def _coerce_channels(chs) -> Tuple[int, ...]:
    if isinstance(chs, int): return (chs,)
    if isinstance(chs, (list, tuple, set)): return tuple(int(x) for x in chs)
    raise TypeError("channels deve ser int, list, tuple ou set")

def make_rtsp_url(ip, user, password, channel, subtype):
    user_enc = urllib.parse.quote(user, safe="")
    pwd_enc  = urllib.parse.quote(password, safe="")
    # Intelbras/Dahua: /cam/realmonitor?channel=N&subtype=0|1
    return f"rtsp://{user_enc}:{pwd_enc}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

# ==========================
# Captura por canal (thread)
# ==========================
class CaptureThread(threading.Thread):
    def __init__(self, name, url, label, target_h, tcp=True, ffmpeg_timeout_ms=5_000_000, reconnect_delay_s=2.0):
        super().__init__(name=name, daemon=True)
        self.url = url
        self.label = label
        self.target_h = target_h
        self.ffmpeg_timeout_ms = ffmpeg_timeout_ms
        self.reconnect_delay_s = reconnect_delay_s
        self.lock = threading.Lock()
        self.frame = None
        self.stop_flag = False

        # Opções do FFmpeg para RTSP
        opts = []
        if tcp: opts.append("rtsp_transport;tcp")
        if ffmpeg_timeout_ms and ffmpeg_timeout_ms > 0:
            opts += [f"stimeout;{ffmpeg_timeout_ms}", f"max_delay;{ffmpeg_timeout_ms}"]
        if opts:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

        self.cap = None

    def _open(self):
        try:
            if self.cap is not None:
                try: self.cap.release()
                except: pass
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.cap = None

    def _resize_h(self, img, h):
        r = h / max(1, img.shape[0])
        return cv2.resize(img, (int(img.shape[1]*r), h), interpolation=cv2.INTER_AREA)

    def run(self):
        self._open()
        last_try = 0
        while not self.stop_flag:
            ok, frame = (self.cap.read() if self.cap is not None else (False, None))
            if not ok or frame is None:
                if time.time() - last_try >= self.reconnect_delay_s:
                    self._open()
                    last_try = time.time()
                # placeholder para não travar
                ph = np.zeros((self.target_h, self.target_h, 3), dtype=np.uint8)
                cv2.putText(ph, f"{self.label} (reconectando...)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                with self.lock: self.frame = ph
                time.sleep(0.05)
                continue

            fr = self._resize_h(frame, self.target_h)
            cv2.putText(fr, self.label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2, cv2.LINE_AA)
            with self.lock:
                self.frame = fr

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stop_flag = True
        try:
            if self.cap is not None: self.cap.release()
        except: pass

# ==========================
# Estado global dos streams
# ==========================
class StreamState:
    def __init__(self):
        self.lock = threading.RLock()
        self.threads: Dict[int, CaptureThread] = {}
        self.getters: Dict[int, callable] = {}
        self.config = {
            "ip": "192.168.0.18",
            "user": "admin",
            "password": "arenna@123",
            "channels": (1, 2),
            "subtype": 0,
            "target_height": 360,   # um pouco mais leve para MJPEG multi-cliente
        }

    def start_streams(self, ip, user, password, channels, subtype=0, target_height=360):
        """Para tudo e reinicia com a nova configuração."""
        channels = _coerce_channels(channels)
        with self.lock:
            # para antigos
            self.stop_all_locked()
            self.threads.clear()
            self.getters.clear()

            # atualiza config
            self.config.update({
                "ip": ip, "user": user, "password": password,
                "channels": channels, "subtype": subtype,
                "target_height": target_height
            })

            # inicia novos
            for ch in channels:
                url = make_rtsp_url(ip, user, password, ch, subtype)
                t = CaptureThread(name=f"CH{ch}", url=url, label=f"CH{ch}", target_h=target_height)
                t.start()
                self.threads[ch] = t
                self.getters[ch] = t.get_frame

    def stop_all(self):
        with self.lock:
            self.stop_all_locked()

    def stop_all_locked(self):
        for t in list(self.threads.values()):
            try: t.stop()
            except: pass

STATE = StreamState()
atexit.register(STATE.stop_all)

# ==========================
# MJPEG helpers (mosaicos)
# ==========================
def mjpeg_generator(get_bgr_frame_fn, fps=12, show_stamp=True, show_fps=True, jpeg_quality=80):
    boundary = b"--frame"
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    last = time.time()
    ema = 0.0
    alpha = 0.12
    while True:
        frame = get_bgr_frame_fn()
        if frame is None:
            time.sleep(0.03)
            continue
        # overlay timestamp e fps
        if show_stamp:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, ts, (10, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        now = time.time()
        inst = 1.0 / max(1e-6, now - last); last = now
        ema = (1-alpha)*ema + alpha*inst if ema > 0 else inst
        if show_fps:
            txt = f"FPS ~ {ema:0.1f}"
            cv2.putText(frame, txt, (frame.shape[1]-180, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        ok, jpeg = cv2.imencode(".jpg", frame, enc_param)
        if not ok:
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(max(0, 1.0/fps))

def mosaic_getter_row(getters: List[callable]):
    def _getter():
        frames = [g() for g in getters]
        frames = [f for f in frames if f is not None]
        if not frames: return None
        h = max(f.shape[0] for f in frames)
        w = sum(f.shape[1] for f in frames)
        canvas = np.zeros((h, w, 3), np.uint8)
        x = 0
        for f in frames:
            hh, ww = f.shape[:2]
            canvas[0:hh, x:x+ww] = f
            x += ww
        return canvas
    return _getter

def mosaic_getter_grid(getters: List[callable], cols: int = 2, cell_h: int = 360):
    """Grade (2x2, 4x4 etc). Redimensiona cada frame para (cell_w, cell_h) usando 16:9 aproximado."""
    cell_w = int(cell_h * 16/9)  # aproximação comum
    def _getter():
        frames = [g() for g in getters]
        frames = [f if f is not None else np.zeros((cell_h, cell_w, 3), np.uint8) for f in frames]
        if not frames: return None
        rows = math.ceil(len(frames) / cols)
        canvas = np.zeros((rows*cell_h, cols*cell_w, 3), np.uint8)
        for i, f in enumerate(frames):
            r = i // cols
            c = i % cols
            fr = cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            y1, y2 = r*cell_h, (r+1)*cell_h
            x1, x2 = c*cell_w, (c+1)*cell_w
            canvas[y1:y2, x1:x2] = fr
        return canvas
    return _getter

# ==========================
# Flask app
# ==========================
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ARENNA • Conexão DVR</title>
  <style>
    :root{color-scheme:dark}
    body{background:#0f1115;color:#eaeaea;font-family:system-ui,Segoe UI,Roboto,Arial;margin:0}
    header{padding:24px 16px;text-align:center;border-bottom:1px solid #222;background:#0b0d12}
    h1{font-size:42px;letter-spacing:4px;margin:0}
    main{max-width:980px;margin:32px auto;padding:0 16px}
    .card{background:#13151b;border:1px solid #1e2230;border-radius:12px;padding:20px}
    label{display:block;margin:12px 0 6px;font-weight:600}
    input, select{width:100%;padding:12px;border-radius:8px;border:1px solid #2a2f40;background:#0f1320;color:#eaeaea;outline:none}
    .row{display:flex;gap:16px;flex-wrap:wrap}
    .col{flex:1 1 300px}
    button{background:#2563eb;border:none;color:white;padding:12px 18px;border-radius:10px;font-weight:700;cursor:pointer}
    button:hover{filter:brightness(1.08)}
    .muted{color:#a0a5b0;font-size:14px;margin-top:4px}
    .actions{margin-top:18px;display:flex;gap:12px;align-items:center}
    .hint{font-size:13px;color:#9aa3b2}
  </style>
</head>
<body>
  <header><h1>ARENNA</h1></header>
  <main>
    <div class="card">
      <h2 style="margin-top:0">Conectar ao DVR</h2>
      <form method="post" action="{{ url_for('connect') }}">
        <div class="row">
          <div class="col">
            <label>IP do DVR</label>
            <input name="ip" value="{{ ip }}" placeholder="192.168.0.18" required>
          </div>
          <div class="col">
            <label>Usuário</label>
            <input name="user" value="{{ user }}" placeholder="admin" required>
          </div>
          <div class="col">
            <label>Senha</label>
            <input name="password" value="{{ password }}" placeholder="••••••••" required>
          </div>
        </div>

        <div class="row">
          <div class="col">
            <label>Canais (Seleção múltipla: Ctrl/Shift para vários)</label>
            <select name="channels" multiple size="10">
              {% for n in range(1,17) %}
                <option value="{{n}}" {% if n in channels %}selected{% endif %}>Canal {{n}}</option>
              {% endfor %}
            </select>
            <div class="muted">Dica: Segure <b>Ctrl</b> (ou <b>Cmd</b>) para selecionar vários.</div>
          </div>
          <div class="col">
            <label>Qualidade (stream)</label>
            <select name="subtype">
              <option value="0" {% if subtype==0 %}selected{% endif %}>Main (HD / qualidade máxima)</option>
              <option value="1" {% if subtype==1 %}selected{% endif %}>Sub (leve / menor latência)</option>
            </select>
            <label style="margin-top:14px">Altura dos quadros (px)</label>
            <input name="target_height" type="number" min="180" max="1080" step="10" value="{{ target_height }}">
            <div class="hint">Quanto maior, mais definição (e mais CPU).</div>
          </div>
        </div>

        <div class="actions">
          <button type="submit">Conectar</button>
          {% if connected %}
          <a href="{{ url_for('view') }}" style="text-decoration:none">
            <button type="button" style="background:#16a34a">Ir para Visualização</button>
          </a>
          {% endif %}
        </div>
      </form>
    </div>
  </main>
</body>
</html>
"""

VIEW_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ARENNA • Visualização</title>
  <style>
    :root{color-scheme:dark}
    body{background:#0f1115;color:#eaeaea;font-family:system-ui,Segoe UI,Roboto,Arial;margin:0}
    header{padding:18px 12px;text-align:center;border-bottom:1px solid #222;background:#0b0d12}
    h1{font-size:36px;letter-spacing:4px;margin:0}
    main{max-width:1280px;margin:24px auto;padding:0 16px}
    .toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}
    .btn{background:#1f2937;color:#eaeaea;border:1px solid #2a3344;border-radius:10px;padding:8px 12px;text-decoration:none;font-weight:600}
    .btn:hover{filter:brightness(1.05)}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
    .card{background:#13151b;border:1px solid #1e2230;border-radius:12px;padding:12px}
    img{max-width:100%;height:auto;border-radius:8px;border:1px solid #2a2f40}
    .muted{color:#a0a5b0;font-size:14px}
  </style>
</head>
<body>
  <header><h1>ARENNA</h1></header>
  <main>
    <div class="toolbar">
      <a class="btn" href="{{ url_for('view', mode='row') }}">Mosaico Horizontal</a>
      <a class="btn" href="{{ url_for('view', mode='grid2') }}">Grid 2×2 (4 primeiras)</a>
      <a class="btn" href="{{ url_for('view', mode='grid4') }}">Grid 4×4 (todas)</a>
      <a class="btn" href="{{ url_for('index') }}">Reconfigurar</a>
    </div>

    {% if mode=='row' %}
      <div class="card">
        <h3>Mosaico Horizontal</h3>
        <img src="{{ url_for('mosaic_mjpg', mode='row') }}">
      </div>
    {% elif mode=='grid2' %}
      <div class="card">
        <h3>Grid 2×2 (4 primeiras)</h3>
        <img src="{{ url_for('mosaic_mjpg', mode='grid', cols=2, subset='first4') }}">
      </div>
    {% elif mode=='grid4' %}
      <div class="card">
        <h3>Grid 4×4 (todas)</h3>
        <img src="{{ url_for('mosaic_mjpg', mode='grid', cols=4, subset='all') }}">
      </div>
    {% endif %}

    <h3 style="margin-top:22px">Câmeras individuais</h3>
    <div class="grid">
      {% for ch in channels %}
        <div class="card">
          <div class="muted" style="margin:4px 0 6px">Canal {{ch}}</div>
          <img src="{{ url_for('channel_mjpg', ch=ch) }}">
        </div>
      {% endfor %}
    </div>
  </main>
</body>
</html>
"""

# ==========================
# Rotas
# ==========================
@app.route("/", methods=["GET"])
def index():
    cfg = STATE.config.copy()
    return render_template_string(INDEX_HTML,
        ip=cfg["ip"], user=cfg["user"], password=cfg["password"],
        channels=set(cfg["channels"]), subtype=cfg["subtype"],
        target_height=cfg["target_height"], connected=(len(STATE.getters)>0)
    )

@app.route("/connect", methods=["POST"])
def connect():
    ip = request.form.get("ip", "").strip()
    user = request.form.get("user", "").strip()
    password = request.form.get("password", "")
    subtype = int(request.form.get("subtype", "0"))
    target_height = int(request.form.get("target_height", "360"))

    # múltiplos 'channels' (select multiple)
    vals = request.form.getlist("channels")
    channels = tuple(sorted({int(v) for v in vals})) if vals else (1,)

    # inicia
    STATE.start_streams(ip, user, password, channels, subtype=subtype, target_height=target_height)
    return redirect(url_for("view"))

@app.route("/view")
def view():
    mode = request.args.get("mode", "row")  # row | grid2 | grid4
    cfg = STATE.config.copy()
    return render_template_string(VIEW_HTML, mode=mode, channels=cfg["channels"])

@app.route("/ch<int:ch>.mjpg")
def channel_mjpg(ch: int):
    g = STATE.getters.get(ch)
    if not g:
        return "Canal não está ativo. Vá em Reconfigurar e selecione este canal.", 404
    return Response(mjpeg_generator(g), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/mosaic.mjpg")
def mosaic_mjpg():
    mode = request.args.get("mode", "row")
    cols = int(request.args.get("cols", "2"))
    subset = request.args.get("subset", "all")  # all | first4

    chans = list(STATE.config["channels"])
    if subset == "first4":
        chans = chans[:4]

    getters = [STATE.getters[c] for c in chans if c in STATE.getters]
    if not getters:
        return "Nenhum canal ativo.", 404

    if mode == "grid":
        getter = mosaic_getter_grid(getters, cols=cols, cell_h=STATE.config["target_height"])
    else:
        getter = mosaic_getter_row(getters)

    return Response(mjpeg_generator(getter), mimetype="multipart/x-mixed-replace; boundary=frame")

# ==========================
# Main
# ==========================
def main(
    default_ip="192.168.0.18",
    default_user="admin",
    default_password="arenna@123",
    default_channels=(1, 2),
    default_subtype=0,
    default_target_height=360,
    host="0.0.0.0",
    port=8000,
):
    # Pré-configuração inicial (opcional): já deixa algo rodando
    STATE.start_streams(
        ip=default_ip,
        user=default_user,
        password=default_password,
        channels=default_channels,
        subtype=default_subtype,
        target_height=default_target_height,
    )
    lan_ip = get_lan_ip(probe_host=default_ip)
    print(f"-> Acesse na mesma rede:  http://{lan_ip}:{port}")
    print(f"-> Local:                http://127.0.0.1:{port}")
    app.run(host=host, port=port, threaded=True)

if __name__ == "__main__":
    # Ajuste os defaults se quiser; o resto é configurado pela página inicial.
    main(
        default_ip="192.168.0.18",
        default_user="admin",
        default_password="arenna@123",
        default_channels=(1, 2),  # ou (1,) ... até (1,2,...,16)
        default_subtype=0,        # 0=main (HD), 1=sub (leve)
        default_target_height=360,
        host="0.0.0.0",
        port=8000,
    )
