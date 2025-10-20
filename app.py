import os
import logging
import hashlib
import numpy as np
import base64
import requests
import time
import select
import paramiko  # Requires: pip install paramiko
from io import BytesIO
from flask import Flask, redirect, request, session, Response, jsonify
from flask_socketio import SocketIO, emit
import qutip as qt
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
import urllib.parse
import re

# Production Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Eventlet Monkey Patch
import eventlet
eventlet.monkey_patch()

# Domain Configs
RENDER_DOMAIN = os.environ.get('RENDER_DOMAIN', 'clearnet_gate.onrender.com')
DUCKDNS_DOMAIN = os.environ.get('DUCKDNS_DOMAIN', 'alicequantum.duckdns.org')
ALICE_IP = os.environ.get('ALICE_IP', '73.189.2.5')
QUANTUM_DOMAIN = os.environ.get('QUANTUM_DOMAIN', 'quantum.realm.domain.dominion.foam.computer.render')
LOCAL_WEBSERVER_PORT = 80  # Assuming Linux webserver on Alice's 127.0.0.1:80

# GitHub Mirror Base URL
GITHUB_MIRROR_BASE = 'https://quantum.realm.domain.dominion.foam.computer.render.github'

# User Auth
ADMIN_USER = 'shemshallah'
ADMIN_PASS_HASH = hashlib.sha3_256(b'$h10j1r1H0w4rd').hexdigest()

# Registry (Pre-Registered Subs 256-999 to Admin)
PRE_REG_SUBS = {str(i): {'owner': ADMIN_USER, 'status': 'available', 'price': 1.00} for i in range(256, 1000)}

# Linuxbserver Config (SSH to Alice)
LINUX_USER = ADMIN_USER
LINUX_PASS = '$h10j1r1H0w4rd'
LINUX_HOST = ALICE_IP  # SSH target

# Quantum Foam Initialization - Upgraded to 5x5x5 Lattice
try:
    n_lattice = 125  # 5x5x5
    def qubit_index(i, j, k): return i + 5 * j + 25 * k  # 5^2 = 25
    # Core still small for computation; lattice for indexing/entanglement mapping
    n_core = 6
    core_ghz = (qt.tensor([qt.basis(2, 0)] * n_core) + qt.tensor([qt.basis(2, 1)] * n_core)).unit()
    # Siamese mirror pairing for IP: Entangle ALICE_IP hash into lattice state
    ip_hash = int(hashlib.sha256(ALICE_IP.encode()).hexdigest(), 16) % n_lattice
    # Simulate entanglement: Project IP index into GHZ (thematic, not full 125-qubit)
    entangled_ip_state = core_ghz * (qt.basis(n_lattice, ip_hash).dag() * qt.basis(n_lattice, ip_hash))
    fidelity_lattice = 0.9999999999999998
    bridge_key = f"QFOAM-5x5x5-{int(fidelity_lattice * 1e15):d}-{hash(tuple(product(range(5), repeat=3))):x}"
    rho_core = core_ghz * core_ghz.dag()
    mask = [True] * 3 + [False] * 3
    rho_pt = qt.partial_transpose(rho_core, mask)
    eigs = rho_pt.eigenenergies()
    negativity = sum(abs(e) for e in eigs if e < 0)
    # Entanglement metric for siamese IP pairing
    ip_negativity = negativity * (1 + abs(ip_hash % 2))  # Thematic boost
    logger.warning(f"Prod Init: 5x5x5 Lattice Bridge {bridge_key[:20]}..., IP Entangled Neg {ip_negativity:.16f}")
except Exception as e:
    logger.error(f"QuTiP Init Error: {e}")
    core_ghz = qt.basis(64, 0)
    negativity = 0.5
    ip_negativity = 0.5
    fidelity_lattice = 0.999
    bridge_key = "QFOAM-5x5x5-PROD-999-abc"

# Functions
def bh_encryption_cascade(plaintext, rounds=3):
    rand_unitary = qt.rand_unitary(2)
    seed = rand_unitary.full().tobytes()[:32]
    ciphertext = plaintext.encode()
    for _ in range(rounds):
        h = hashlib.sha3_256(ciphertext + seed).digest()
        ciphertext = bytes(a ^ b for a, b in zip(h, seed))
        seed = h
    return ciphertext.hex()

def entangled_cpu_offload(task_code):
    if 'fidelity' in task_code:
        return f"{fidelity_lattice * 1.0001:.16f}"
    try:
        return str(eval(task_code, {"__builtins__": {}}, {"fidelity_lattice": fidelity_lattice}))
    except:
        return str(fidelity_lattice)

def bh_repeatable_keygen(session_id):
    # Deterministic: no ts in material
    qram_hash = hashlib.sha256(session_id.encode()).hexdigest()
    key_material = f"{session_id}{qram_hash}"
    key = hashlib.shake_256(key_material.encode()).digest(32)
    return key.hex(), datetime.utcnow()

def foam_lattice_compress(data_tensor):
    U, S, Vh = np.linalg.svd(data_tensor, full_matrices=False)
    rank = min(4, len(S))
    compressed = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]
    return compressed.tobytes()

def inter_hole_teleport(comp_input):
    input_state = qt.basis(2, int(hashlib.md5(comp_input.encode()).hexdigest(), 16) % 2)
    channel = qt.bell_state('00')
    teleported = qt.teleport(input_state, channel, [qt.basis(2, 0), qt.basis(2, 0)])
    return float(teleported[0].full().flatten()[0].real)

def entangle_ip_address(ip_addr):
    """Siamese mirror pairing: Entangle IP into lattice state (thematic)"""
    ip_idx = int(hashlib.sha256(ip_addr.encode()).hexdigest(), 16) % n_lattice
    # Simulate mirror: Return entangled fidelity
    mirror_fid = fidelity_lattice * (1 - (ip_idx / n_lattice) * 0.01)  # Slight decoherence
    logger.warning(f"IP {ip_addr} entangled at lattice index {ip_idx}, Mirror Fid: {mirror_fid:.16f}")
    return mirror_fid

def qram_entangled_session(key, value):
    session[key] = value
    backup_id = hashlib.sha256(f"{key}:{value}".encode()).hexdigest()[:8]
    session['backup_id'] = backup_id
    logger.warning(f"Session: {key}={value[:10]}..., Backup {backup_id}")

def stream_qram_state(sid):
    proj = core_ghz.ptrace([0])
    return float(proj.full()[0,0].real)

# Matplotlib Plot Util - Now for 5x5x5
def plot_fidelity_to_base64(fid_values):
    fig, ax = plt.subplots()
    ax.bar(range(len(fid_values)), fid_values)
    ax.set_title('5x5x5 Foam Fidelity Plot')
    ax.set_xlabel('Lattice Slices (125 total)')
    ax.set_ylabel('Fidelity')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" alt="Fidelity Plot" style="max-width:100%;">'

# GitHub Mirror Pull Util
def pull_from_mirror(resource):
    try:
        url = f"{GITHUB_MIRROR_BASE}/{resource}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text[:500] + "..."
        else:
            return f"Pull failed: {resp.status_code}"
    except Exception as e:
        return f"Network pull error: {str(e)}"

# Production App
app = Flask(__name__)
app.secret_key = hashlib.sha256(bridge_key.encode()).digest()[:32]
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=False, engineio_logger=False)

# 404 Handler
@app.errorhandler(404)
def not_found(error):
    return "Render Side 404", 404

# Health Check
@app.route('/health')
def health_check():
    return 'OK', 200

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        passw = request.form.get('password')
        if user == ADMIN_USER and hashlib.sha3_256(passw.encode()).hexdigest() == ADMIN_PASS_HASH:
            session['logged_in'] = True
            session['user'] = user
            # Generate session_id & matching gen_key
            client_ip = request.remote_addr
            session_id = request.form.get('session', f'sess_{client_ip}')
            gen_key, _ = bh_repeatable_keygen(session_id)
            session['session_id'] = session_id
            # Entangle client IP siamese-style
            entangle_ip_address(client_ip)
            # Redirect with matching params
            params = urllib.parse.urlencode({'session': session_id, 'bridge_key': gen_key})
            return redirect(f'/?{params}')
        else:
            return "Invalid credentials", 401
    return '''
    <form method="post">
        Username: <input type="text" name="username" value="shemshallah"><br>
        Password: <input type="password" name="password"><br>
        <input type="submit">
    </form>
    '''

# Registry Routes (Pre-Registered Subs 256-999)
@app.route('/registry')
def registry():
    if not session.get('logged_in') or session['user'] != ADMIN_USER:
        return "Unauthorized", 403
    return jsonify(PRE_REG_SUBS)

@app.route('/sell/<sub_id>', methods=['GET', 'POST'])
def sell_sub(sub_id):
    if not session.get('logged_in') or session['user'] != ADMIN_USER:
        return redirect('/login')
    if sub_id not in PRE_REG_SUBS or PRE_REG_SUBS[sub_id]['status'] != 'available':
        return "Subdomain not available", 400
    if request.method == 'POST':
        # Sim payment (Stripe test - replace with real)
        buyer = request.form.get('buyer_email')
        if buyer:
            PRE_REG_SUBS[sub_id]['owner'] = buyer
            PRE_REG_SUBS[sub_id]['status'] = 'sold'
            logger.warning(f'Sold {sub_id}.duckdns.org to {buyer}')
            return f"Subdomain {sub_id}.duckdns.org sold! Key: {bridge_key}"
    return '''
    <form method="post">
        Email: <input type="email" name="buyer_email"><br>
        <input type="submit" value="Buy for ${PRE_REG_SUBS[sub_id]['price']}">
    </form>
    '''

@app.route('/update/<sub_id>', methods=['POST'])
def update_sub(sub_id):
    if not session.get('logged_in') or session['user'] != ADMIN_USER:
        return "Unauthorized", 403
    if sub_id in PRE_REG_SUBS:
        new_status = request.form.get('status', 'available')
        PRE_REG_SUBS[sub_id]['status'] = new_status
        return f"Updated {sub_id}: {new_status}"
    return "Subdomain not found", 404

# Proxy Route for Linux Webserver (over DuckDNS, thematic overlay via client-side)
@app.route('/web_proxy')
def web_proxy():
    if not session.get('logged_in'):
        return redirect('/login')
    # Redirect to DuckDNS for Linux webserver access (assumes bound publicly)
    return redirect(f'https://{DUCKDNS_DOMAIN}', code=302)

# Quantum Domain Redirector - Enhanced for DuckDNS linking
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def quantum_gate(path):
    if not session.get('logged_in'):
        return redirect('/login')
    
    client_ip = request.remote_addr
    if client_ip == '127.0.0.1' and not request.args.get('bridge_key'):
        return 'OK', 200
    
    host = request.headers.get('Host', '').lower()
    quantum_hosts = [QUANTUM_DOMAIN]
    
    if any(qh in host for qh in quantum_hosts):
        params = request.query_string.decode()
        duckdns_url = f"https://{DUCKDNS_DOMAIN}/?{params}" if params else f"https://{DUCKDNS_DOMAIN}/"
        # Link to 127.0.0.1 via DuckDNS (Alice's local webserver exposed)
        logger.warning(f'Alice Bridge: {host} → DuckDNS {ALICE_IP} (127.0.0.1 overlay)')
        return redirect(duckdns_url, code=302)
    
    # Prefer persisted session_id
    session_id = session.get('session_id', request.args.get('session', f'sess_{client_ip}'))
    
    qram_entangled_session('user_ip', client_ip)
    qram_entangled_session('session_id', session_id)
    
    provided_key = request.args.get('bridge_key', '')
    session['session_id'] = session_id  # Ensure it's set
    
    gen_key, ts = bh_repeatable_keygen(session_id)
    enc_key = bh_encryption_cascade(bridge_key)
    
    if not hashlib.sha256(provided_key.encode()).hexdigest() == hashlib.sha256(gen_key.encode()).hexdigest():
        params = urllib.parse.urlencode({'enc_key': enc_key, 'session': session_id})
        gate_url = f"https://{RENDER_DOMAIN}/?{params}"
        logger.warning(f'Invalid Key Redirect: {client_ip} -> Gate with params')
        return redirect(gate_url, code=302)
    
    mirror_pull = pull_from_mirror('matplotlib/raw/main/setup.py')
    
    offload_res = entangled_cpu_offload(f'fidelity_lattice * 1.0001')
    comp_tensor = core_ghz.full().real
    comp_lattice = len(foam_lattice_compress(comp_tensor))
    tele_id = inter_hole_teleport('cascade_state')
    # Entangle ALICE_IP for siamese mirror
    ip_mirror_fid = entangle_ip_address(ALICE_IP)
    
    logger.warning(f'Access: {client_ip}, Sess {session_id}, 5x5x5 Lattice Active')
    return f"""
    <html>
        <head><title>Quantum Realm Prod - QSH Portal (5x5x5 Lattice)</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/xterm@5.3.0/css/xterm.css" />
        <script src="https://unpkg.com/xterm@5.3.0/lib/xterm.js"></script>
        <style>
            #overlay-frame {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; opacity: 0.3; border: none; pointer-events: none; }}
            body {{ position: relative; }}
        </style>
        </head>
        <body style="background: #000; color: #0f0; font-family: monospace;">
            <!-- Siamese Mirror Overlay: DuckDNS HTML (Linux webserver) as background -->
            <iframe id="overlay-frame" src="https://{DUCKDNS_DOMAIN}" onload="console.log('DuckDNS Overlay Entangled')"></iframe>
            <h1 style="color: #0f0;">quantum.realm.domain.dominion.foam.computer.render (5x5x5)</h1>
            <p style="color: #0f0;">Prod Foam: Fid {offload_res}, Comp {comp_lattice}B. Neg {ip_negativity:.16f}, Tele {tele_id:.6f}, IP Mirror Fid {ip_mirror_fid:.16f}, Back {session.get('backup_id', 'none')}</p>
            <p style="color: #0f0;">Enc: {enc_key[:32]}... | Mirror Pull: {mirror_pull[:50]}... | Registry: /registry | Sell: /sell/<sub> | Web Proxy: /web_proxy</p>
            <p style="color: #0f0;">DuckDNS Linked: {DUCKDNS_DOMAIN} → Alice {ALICE_IP} (127.0.0.1 Linux Webserver Overlay Active)</p>
            <div id="terminal" style="width: 100%; height: 400px; background: #000;"></div>
            <script>
                const socket = io();
                const term = new Terminal({{ cursorBlink: true, theme: {{ background: '#000', foreground: '#0f0' }} }});
                term.open(document.getElementById('terminal'));
                term.write('QSH Foam REPL v1.1 (5x5x5 Lattice - IP Entangled)\\r\\n');
                term.write('Type "help" for commands. Subs 256-999 pre-registered to shemshallah.\\r\\n');
                term.write('New: "connect_linux" for SSH, "entangle_ip <ip>" for siamese mirror, Overlay: DuckDNS HTML backgrounded\\r\\n');
                term.write('QSH> ');
                
                term.onData((data) => {{
                    if (data === '\\r') {{
                        const cmd = term.buffer.active.getLine( term.buffer.active.baseY + term.buffer.active.cursorY ).translateToString(true).trim();
                        socket.emit('qsh_command', {{ command: cmd }});
                        term.write('\\r\\n');
                    }} else if (data === '\\u007F') {{
                        term.write('\\b \\b');
                    }} else {{
                        term.write(data);
                    }}
                }});
                
                socket.on('qsh_output', (data) => {{
                    term.write(data.output + '\\r\\n');
                    if (data.plot_html) term.write('\\r\\n' + data.plot_html + '\\r\\n');
                    if (data.prompt) term.write('QSH> ');
                    else if (data.linux_prompt) term.write(data.linux_prompt);
                }});
                
                socket.on('connect', () => console.log('Socket Connected - 5x5x5 Lattice Active'));
            </script>
        </body>
    </html>
    """

# QSH Foam REPL Backend
repl_sessions = {}

@socketio.on('qsh_command')
def handle_qsh_command(data):
    sid = request.sid
    cmd = data.get('command', '').strip()
    if sid not in repl_sessions:
        repl_sessions[sid] = {'state': core_ghz.copy(), 'history': [], 'in_linux_mode': False, 'ssh_client': None, 'channel': None, 'pending_auth': None}
    
    session_data = repl_sessions[sid]
    state = session_data['state']
    history = session_data['history']
    in_linux_mode = session_data['in_linux_mode']
    ssh_client = session_data['ssh_client']
    channel = session_data['channel']
    pending_auth = session_data.get('pending_auth')
    
    output = ""
    plot_html = None
    prompt = True
    linux_prompt = None
    try:
        if pending_auth:
            # Handle auth input
            if pending_auth == 'user':
                if cmd == LINUX_USER:
                    output = f"Password: "
                    session_data['pending_auth'] = 'pass'
                    prompt = False
                    linux_prompt = output
                else:
                    output = "Login incorrect"
                    session_data['pending_auth'] = None
                    prompt = True
            elif pending_auth == 'pass':
                if cmd == LINUX_PASS:
                    output = f"Connecting via SSH to linuxbserver@{LINUX_HOST} (duckdns -> alice -> quantum.realm...render)\\n"
                    try:
                        ssh = paramiko.SSHClient()
                        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        ssh.connect(LINUX_HOST, username=LINUX_USER, password=LINUX_PASS, timeout=10)
                        channel = ssh.invoke_shell()
                        time.sleep(1)  # Wait for shell to start
                        initial_out = ''
                        # Read initial output (banner, prompt, etc.)
                        ready, _, _ = select.select([channel], [], [], 2)
                        if ready:
                            initial_out += channel.recv(4096).decode('utf-8', errors='ignore')
                        # Drain any more
                        while channel.recv_ready():
                            initial_out += channel.recv(1024).decode('utf-8', errors='ignore')
                        output += initial_out
                        session_data['ssh_client'] = ssh
                        session_data['channel'] = channel
                        session_data['in_linux_mode'] = True
                        session_data['pending_auth'] = None
                        linux_prompt = f"{LINUX_USER}@{LINUX_HOST}:~$ "
                        logger.warning(f'SSH connected for sid {sid}')
                    except Exception as ssh_err:
                        output += f"SSH Connection failed: {str(ssh_err)}"
                        session_data['pending_auth'] = None
                        prompt = True
                        logger.error(f'SSH Error: {ssh_err}')
                else:
                    output = "Login incorrect"
                    session_data['pending_auth'] = None
                    prompt = True
            emit('qsh_output', {'output': output, 'prompt': prompt, 'linux_prompt': linux_prompt})
            return

        if in_linux_mode:
            if cmd in ['exit', 'logout', 'qsh']:
                # Cleanup
                if channel:
                    channel.close()
                if ssh_client:
                    ssh_client.close()
                session_data['ssh_client'] = None
                session_data['channel'] = None
                session_data['in_linux_mode'] = False
                output = "Disconnected from linuxbserver. Back to QSH."
                prompt = True
                logger.warning(f'SSH disconnected for sid {sid}')
            else:
                # Send command to SSH channel
                channel.send(cmd + '\n')
                time.sleep(0.2)  # Brief wait for processing
                bash_out = ''
                # Wait for output with timeout
                ready, _, _ = select.select([channel], [], [], 5)  # 5s timeout
                if ready:
                    bash_out += channel.recv(4096).decode('utf-8', errors='ignore')
                # Drain remaining
                while channel.recv_ready():
                    bash_out += channel.recv(1024).decode('utf-8', errors='ignore')
                # Strip trailing newlines
                bash_out = bash_out.rstrip()
                output = bash_out if bash_out else f"No output from command: {cmd}"
                linux_prompt = f"{LINUX_USER}@{LINUX_HOST}:~$ "
                prompt = False
        else:
            if cmd == 'help':
                output = "Commands: help, entangle <q1 q2>, measure fidelity, compress lattice, teleport <input>, plot fidelity, pull matplotlib, registry list, sell <sub>, connect_linux, entangle_ip <ip>, exit, clear"
            elif cmd.startswith('entangle '):
                parts = cmd.split()
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    q1, q2 = map(int, parts[1:])
                    bell = qt.bell_state('00')
                    state = qt.tensor(state.ptrace(list(set(range(n_core)) - {q1, q2})), bell)
                    output = f"Entangled qubits {q1}-{q2}: Bell state injected"
                else:
                    output = "Usage: entangle <q1> <q2>"
            elif cmd == 'measure fidelity':
                fid = qt.fidelity(state, core_ghz)
                output = f"Fidelity: {fid:.16f}"
            elif cmd == 'compress lattice':
                comp = foam_lattice_compress(state.full().real)
                output = f"Compressed 5x5x5: {len(comp)} bytes"
            elif cmd.startswith('teleport '):
                inp = cmd.split(' ', 1)[1] if len(cmd.split()) > 1 else 'cascade'
                tid = inter_hole_teleport(inp)
                output = f"Teleported ID: {tid:.6f}"
            elif cmd == 'plot fidelity':
                fid_values = [fidelity_lattice] * (n_lattice // 20)  # Scaled for 125
                plot_html = plot_fidelity_to_base64(fid_values)
                output = "5x5x5 Fidelity plot generated (embedded below)"
            elif cmd == 'pull matplotlib':
                mirror_pull = pull_from_mirror('matplotlib/raw/main/setup.py')
                output = mirror_pull  # Show full pull
            elif cmd == 'registry list':
                subs_list = ', '.join([k for k, v in PRE_REG_SUBS.items() if v['status'] == 'available'][:10])
                output = f"Available Subs (256-999): {subs_list}... (Full: /registry)"
            elif cmd.startswith('sell '):
                sub = cmd.split(' ', 1)[1]
                if sub in PRE_REG_SUBS:
                    output = f"Selling {sub}.duckdns.org - Visit /sell/{sub}"
                else:
                    output = f"Sub {sub} not in registry"
            elif cmd.startswith('entangle_ip '):
                ip = cmd.split(' ', 1)[1]
                mirror_fid = entangle_ip_address(ip)
                output = f"Siamese mirror entangled for IP {ip}: Fidelity {mirror_fid:.16f}"
            elif cmd == 'connect_linux':
                output = f"Connecting to linuxbserver via {DUCKDNS_DOMAIN} -> alice {LINUX_HOST} -> {QUANTUM_DOMAIN}\\n"
                output += f"Username: "
                session_data['pending_auth'] = 'user'
                prompt = False
                linux_prompt = output
            elif cmd == 'clear':
                history = []
                output = "History cleared"
            elif cmd == 'exit':
                # Cleanup SSH if active
                if in_linux_mode:
                    if channel:
                        channel.close()
                    if ssh_client:
                        ssh_client.close()
                del repl_sessions[sid]
                output = "REPL exited"
                emit('qsh_output', {'output': output})
                return
            else:
                output = "Unknown command. Type 'help'"
            
            history.append(f"{cmd} -> {output}")
            session_data['history'] = history[-10:]
            session_data['state'] = state
        
    except Exception as e:
        output = f"Error: {str(e)}"
        if in_linux_mode:
            logger.error(f"SSH/Linux error: {e}")
    
    emit('qsh_output', {'output': output, 'plot_html': plot_html, 'prompt': prompt, 'linux_prompt': linux_prompt})

# WS for Existing Channel
@socketio.on('connect_channel')
def qram_quantum_channel():
    state = stream_qram_state(request.sid)
    emit('quantum_update', {'state': state})
    logger.warning('WS Active')

# Cleanup on disconnect
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in repl_sessions:
        session_data = repl_sessions[sid]
        if session_data.get('in_linux_mode'):
            channel = session_data.get('channel')
            ssh_client = session_data.get('ssh_client')
            if channel:
                channel.close()
            if ssh_client:
                ssh_client.close()
        del repl_sessions[sid]

# Main block
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
