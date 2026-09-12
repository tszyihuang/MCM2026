"""模拟器界面: 测试控制台.

通过本地网页提供模拟器操作界面 (等价于桌面版模拟器的界面):
  * 四个测试模块入口 (问题3/4 演练、问题3/4 正式), 正式测试两次确认;
  * 5 秒倒计时、25 分钟窗口、程序运行时间、虚拟时间、案例编码显示;
  * 「指令与反馈」区域展示机器狗请求与模拟器反馈 (默认最新 1000 条);
  * 日志列表与导出 (不修改文件名)、演练测试结束给出干扰源真值统计;
  * 「中止测试」需两次确认, 正式测试中止同样占用一次机会。

仅使用标准库, 不引入第三方依赖。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .session import SessionManager

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>无线电干扰源环境模拟器</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Microsoft YaHei", system-ui, sans-serif; background:#0f1420; color:#dfe7f5; font-size:13px; }
  header { padding:10px 16px; background:#16203a; border-bottom:1px solid #26314d; display:flex; flex-wrap:wrap; gap:16px; align-items:center; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.5px; }
  .tag { padding:2px 8px; border-radius:10px; background:#26314d; font-size:12px; }
  .tag.ok { background:#14432a; color:#7ee2a8; }
  .tag.warn { background:#4a3410; color:#ffcf70; }
  .tag.bad { background:#4a1a1a; color:#ff9c9c; }
  main { display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px; }
  @media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
  section { background:#131a2b; border:1px solid #26314d; border-radius:8px; padding:12px; }
  section h2 { margin:0 0 10px; font-size:13px; color:#8fb2ff; font-weight:600; letter-spacing:.5px; }
  .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; }
  button { font-family:inherit; font-size:12.5px; padding:9px 10px; border-radius:6px; border:1px solid #2f3d5e;
           background:#1b2540; color:#dfe7f5; cursor:pointer; transition:.15s; }
  button:hover:not(:disabled) { background:#25325a; border-color:#4a63a8; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.primary { background:#1d4ed8; border-color:#3b6ff0; }
  button.danger { background:#7f1d1d; border-color:#b91c1c; }
  button.wide { grid-column: span 2; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { text-align:left; padding:4px 6px; border-bottom:1px solid #1f2a44; }
  th { color:#8fb2ff; font-weight:500; }
  .mono { font-family: Consolas, "Courier New", monospace; }
  .kv { display:grid; grid-template-columns: auto 1fr; gap:2px 10px; font-size:12.5px; }
  .kv b { color:#8fb2ff; font-weight:500; }
  #log { height:320px; overflow:auto; background:#0b101c; border:1px solid #1f2a44; border-radius:6px; padding:6px;
         font-family: Consolas, monospace; font-size:11.5px; line-height:1.5; }
  .req { color:#9ad4ff; } .resp { color:#a8e6a3; } .rej { color:#ffb26e; } .err { color:#ff8f8f; }
  #banner { padding:8px 12px; margin:0 12px; border-radius:6px; display:none; }
  #banner.show { display:block; }
  #banner.info { background:#152a4a; color:#a8c8ff; border:1px solid #2b4a80; }
  #banner.good { background:#12351f; color:#8ee6a8; border:1px solid #1f6b3a; }
  #banner.bad  { background:#3d1616; color:#ffa8a8; border:1px solid #7f2020; }
  .cd { font-size:34px; font-weight:700; text-align:center; padding:8px; color:#ffd479; }
  small { color:#7d8bab; }
</style>
</head>
<body>
<header>
  <h1>无线电干扰源环境模拟器</h1>
  <span id="loginTag" class="tag">登录状态</span>
  <span id="ifaceTag" class="tag">接口未开放</span>
  <span id="baseUrl" class="tag mono"></span>
  <span id="deadlineTag" class="tag"></span>
</header>
<div id="banner"></div>
<main>
  <section>
    <h2>测试模块</h2>
    <div class="grid">
      <button id="btn_q3_practice">问题3 演练测试</button>
      <button id="btn_q4_practice">问题4 演练测试</button>
      <button id="btn_q3_formal" class="primary">问题3 正式测试 (3 次机会)</button>
      <button id="btn_q4_formal" class="primary">问题4 正式测试 (3 次机会)</button>
      <button id="btn_abort" class="danger wide" disabled>中止测试 (需两次确认)</button>
      <button id="btn_export" class="wide" disabled>导出本局行为日志</button>
    </div>
    <h2 style="margin-top:14px">运行状态</h2>
    <div id="countdown" class="cd"></div>
    <div class="kv" id="statusKv"></div>
    <div style="margin-top:10px"><small id="moduleInfo"></small></div>
  </section>

  <section>
    <h2>指令与反馈 (最新 <span id="exchCount">0</span> 条)</h2>
    <div id="log"></div>
    <div style="margin-top:8px"><small>显示上限可在设置中调整; 完整行为写入行为日志。</small></div>
  </section>

  <section>
    <h2>本局结果</h2>
    <div id="report">暂无测试结果。</div>
  </section>

  <section>
    <h2>日志列表 (行为日志 / 测试案例编码)</h2>
    <div id="logList">暂无日志。</div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
let pendingModule = null;
let confirmArmed = false;
let abortArmed = false;

async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}

function fmt(v, digits=2) {
  if (v === null || v === undefined) return '-';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(digits);
  return String(v);
}

function banner(text, kind) {
  const el = $('banner');
  if (!text) { el.className = ''; el.textContent = ''; return; }
  el.className = 'show ' + (kind || 'info');
  el.textContent = text;
}

async function startTest(key) {
  const meta = window.__modules[key] || {};
  const label = meta.title || key;
  if (meta.official) {
    if (!confirmArmed) {
      confirmArmed = true;
      pendingModule = key;
      banner('再次点击「' + label + '」确认开始。正式测试将占用一次机会, 启动后不可取消。', 'bad');
      return;
    }
    confirmArmed = false;
    pendingModule = null;
  }
  const r = await api('/api/start', {method:'POST', headers:{'Content-Type':'application/json'},
                                     body: JSON.stringify({module:key})});
  if (r.ok) { banner('已启动: ' + label + ' | 案例编码 ' + r.case_code, 'good'); }
  else { banner('启动失败: ' + r.error, 'bad'); }
  refresh();
}

async function doAbort() {
  if (!abortArmed) {
    abortArmed = true;
    banner('再次点击「中止测试」确认中止。正式测试中止同样占用一次测试机会。', 'bad');
    setTimeout(() => { abortArmed = false; }, 8000);
    return;
  }
  abortArmed = false;
  const r = await api('/api/abort', {method:'POST'});
  banner(r.ok ? '已手工中止测试。' : ('中止失败: ' + r.error), r.ok ? 'info' : 'bad');
  refresh();
}

async function doExport() {
  const r = await api('/api/export', {method:'POST'});
  if (r.ok) banner('已导出: ' + r.paths.readable + (r.paths.encrypted ? ' | ' + r.paths.encrypted : ''), 'good');
  else banner('导出失败: ' + r.error, 'bad');
}

function renderStatus(s) {
  const st = s.status || {};
  const login = s.login || {};
  $('loginTag').textContent = (login.logged_in ? '已登录 ' : '未登录 ') + (login.team_id || '');
  $('loginTag').className = 'tag ' + (login.logged_in ? 'ok' : 'bad');
  $('ifaceTag').textContent = st.interface_open ? '接口已就绪 (可调用 /enter)' : '接口未开放';
  $('ifaceTag').className = 'tag ' + (st.interface_open ? 'ok' : 'warn');
  $('baseUrl').textContent = s.base_url || '';
  $('deadlineTag').textContent = s.deadline_blocked ? ('已过启动截止: ' + s.deadline_blocked) : '启动截止检查通过';
  $('deadlineTag').className = 'tag ' + (s.deadline_blocked ? 'bad' : 'ok');

  const cd = (st.phase === 'countdown' && st.countdown_left_s > 0) ? Math.ceil(st.countdown_left_s) : 0;
  $('countdown').textContent = cd > 0 ? ('倒计时 ' + cd) : (st.interface_open ? '接口已开放' : '');

  const rows = [
    ['阶段', ({idle:'空闲', preparing:'案例数据准备中', countdown:'5 秒倒计时', armed:'窗口开放/等待 /enter', running:'测试进行中', ended:'测试已结束'})[st.phase] || st.phase],
    ['测试案例编码', st.case_code || '-'],
    ['虚拟时间 (s)', fmt(st.virtual_time_s, 3)],
    ['已清除干扰源', st.cleared === undefined ? '-' : st.cleared],
    ['干扰源总数', st.official ? '正式测试不显示' : (st.source_total === null || st.source_total === undefined ? '-' : st.source_total)],
    ['窗口剩余 (s)', fmt(st.window_left_s, 0)],
    ['程序剩余 (s)', fmt(st.program_left_s, 0)],
    ['虚拟剩余 (s)', fmt(st.virtual_left_s, 0)],
    ['指令条数', st.action_count === undefined ? '-' : st.action_count],
    ['结束原因', st.end_reason ? (st.end_reason + ' · ' + (st.end_reason_text||'')) : '-'],
  ];
  $('statusKv').innerHTML = rows.map(([k,v]) => '<b>'+k+'</b><span class="mono">'+v+'</span>').join('');

  const mods = s.modules || {};
  $('moduleInfo').textContent = Object.values(mods).map(m =>
      m.title + ': 已用 ' + m.attempts_used + (m.max_attempts ? ('/' + m.max_attempts) : ' 次 (不限次数)')).join('  |  ');

  const running = st.phase && st.phase !== 'ended' && st.phase !== 'idle';
  $('btn_abort').disabled = !running;
  $('btn_export').disabled = !s.last_report;
  ['q3_practice','q4_practice'].forEach(k => { $('btn_'+k).disabled = running; });
  ['q3_formal','q4_formal'].forEach(k => {
    const m = mods[k] || {};
    $('btn_'+k).disabled = running || (m.max_attempts !== null && m.attempts_used >= m.max_attempts);
  });

  const ex = (s.exchanges || []).slice(-1000);
  $('exchCount').textContent = ex.length;
  const box = $('log');
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  box.innerHTML = ex.map(e => {
    if (e.path === undefined) return '';
    let cls = 'req', tag = e.path;
    if (e.status !== 200) { cls = (e.status >= 500 ? 'err' : 'rej'); }
    const req = JSON.stringify(e.body);
    let out;
    if (e.response) out = JSON.stringify(e.response);
    else out = JSON.stringify({http_status: e.status, error: e.error, message: e.message});
    if (e.response && e.response.accepted === false) cls = 'rej';
    return '<div><span class="'+cls+'">→ '+tag+'</span> '+escapeHtml(req)+'</div>' +
           '<div class="'+cls+'">← '+escapeHtml(out)+'</div>';
  }).join('');
  if (atBottom) box.scrollTop = box.scrollHeight;

  const logs = s.logs || [];
  $('logList').innerHTML = logs.length ? ('<table><tr><th>时间/模块</th><th>案例编码</th><th>文件</th><th>大小</th><th>上传</th></tr>' +
    logs.map(l => '<tr><td>'+(l.official?'正式':'演练')+'</td><td class="mono">'+l.case_code+'</td><td class="mono">'+
      l.path.split(/[\\\\/]/).pop()+'</td><td>'+(l.size/1024).toFixed(1)+' KB</td><td>'+(l.uploaded?'已上传':'排队中')+'</td></tr>').join('') +
    '</table>') : '暂无日志。';

  const rep = s.last_report;
  if (!rep) { $('report').textContent = '暂无测试结果。'; return; }
  const stt = rep.stats || {};
  const truth = rep.truth || {};
  let html = '<div class="kv">' + [
    ['案例编码', rep.case_code],
    ['结束原因', rep.end_reason + ' · ' + (rep.end_reason_text||'')],
    ['干扰源总数', truth.hidden ? '正式测试不显示' : fmt(truth.source_total, 0)],
    ['已清除个数', fmt(stt.cleared_count, 0)],
    ['被清除比例', (stt.cleared_ratio === undefined || stt.cleared_ratio === null) ? '正式测试不显示' : (stt.cleared_ratio*100).toFixed(1) + '%'],
    ['平均定位清除时间 (s)', fmt(stt.avg_clear_duration_s, 3)],
    ['定位清除总时间 (s)', fmt(stt.total_duration_s, 3)],
    ['程序运行时间 (s)', fmt(stt.program_runtime_s, 2)],
    ['移动距离 (m)', fmt(stt.moved_distance_m, 1)],
    ['检测 / 清除尝试', fmt(stt.measure_count,0) + ' / ' + fmt(stt.clear_attempts,0)],
  ].map(([k,v]) => '<b>'+k+'</b><span class="mono">'+v+'</span>').join('') + '</div>';
  if (!truth.hidden && truth.sources) {
    html += '<h2 style="margin-top:12px">干扰源真值 (演练测试)</h2><table><tr><th>频道</th><th>x</th><th>y</th><th>半径</th><th>类型</th><th>已清除</th></tr>' +
      truth.sources.map(t => '<tr><td>'+t.channel+'</td><td>'+t.position.x.toFixed(1)+'</td><td>'+t.position.y.toFixed(1)+
      '</td><td>'+t.valid_radius_m.toFixed(0)+'</td><td>'+t.kind+'</td><td>'+(t.cleared?'是':'否')+'</td></tr>').join('') + '</table>';
  }
  $('report').innerHTML = html;
}

function escapeHtml(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function refresh() {
  try {
    const s = await api('/api/state');
    window.__modules = s.modules || {};
    renderStatus(s);
  } catch (e) { /* 模拟器可能正在重启 */ }
}

$('btn_q3_practice').onclick = () => startTest('q3_practice');
$('btn_q4_practice').onclick = () => startTest('q4_practice');
$('btn_q3_formal').onclick = () => startTest('q3_formal');
$('btn_q4_formal').onclick = () => startTest('q4_formal');
$('btn_abort').onclick = doAbort;
$('btn_export').onclick = doExport;
refresh();
setInterval(refresh, 500);
</script>
</body>
</html>
"""

# 页面是模块级常量, 预先编码一次即可 (原先每次 GET / 都重新 encode 一遍)
INDEX_HTML_BYTES = INDEX_HTML.encode("utf-8")


class _GuiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RadioInterferenceSimulatorGUI/1.0"
    sys_version = ""
    # 与机器狗接口一致: 界面每 500 ms 一次 /api/state, 关掉 Nagle 以免
    # "发响应头 + 发响应体"两次写与对端延迟确认叠加。
    disable_nagle_algorithm = True

    @property
    def manager(self) -> SessionManager:
        return self.server.manager  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        return

    # ---- helpers ----
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            # 浏览器刷新/关闭页面导致连接提前断开: 不应打印堆栈
            pass

    def _send_html(self, body: bytes = INDEX_HTML_BYTES) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)
        if path in ("/", "/index.html"):
            self._send_html()
        elif path == "/api/state":
            try:
                limit = int(query.get("limit", ["200"])[0])
            except (TypeError, ValueError):
                limit = 200
            self._send_json(200, self.manager.ui_state(log_limit=max(100, min(5000, limit))))
        elif path == "/api/ping":
            self._send_json(200, {"ok": True, "time": time.time()})
        else:
            self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        payload = self._read_json()
        try:
            if path == "/api/start":
                result = self.manager.start_test(str(payload.get("module", "")), seed=payload.get("seed"))
                self._send_json(200, dict({"ok": True}, **result))
            elif path == "/api/abort":
                report = self.manager.abort_test()
                self._send_json(200, {"ok": report is not None, "case_code": (report or {}).get("case_code")})
            elif path == "/api/export":
                paths = self.manager.export_current()
                if paths is None:
                    self._send_json(200, {"ok": False, "error": "当前没有可导出的行为日志"})
                else:
                    self._send_json(200, {"ok": True, "paths": paths})
            elif path == "/api/login":
                state = self.manager.login.login(payload.get("password"))
                self._send_json(200, {"ok": True, "login": state.to_dict()})
            else:
                self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send_json(200, {"ok": False, "error": str(exc)})


class GuiServer:
    """模拟器界面 (网页控制台)。"""

    def __init__(self, manager: SessionManager, host: str = "127.0.0.1", port: int = 2027) -> None:
        self.manager = manager
        self.host = host
        self.port = port
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> int:
        httpd = ThreadingHTTPServer((self.host, self.port), _GuiHandler)
        httpd.daemon_threads = True
        httpd.manager = self.manager  # type: ignore[attr-defined]
        self._httpd = httpd
        self.port = httpd.server_address[1]
        # 与 SimulatorServer 同理: 默认 0.5 s 的 poll_interval 会让每次 stop()
        # 空等半个轮询周期。
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="simulator-gui",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

    @property
    def url(self) -> str:
        return "http://%s:%d/" % (self.host, self.port)
