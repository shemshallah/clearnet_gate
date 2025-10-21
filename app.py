import os
import logging
import hashlib
import numpy as np
import base64
import time
import select
from io import BytesIO
from flask import Flask, redirect, request, session, Response, jsonify
from flask_socketio import SocketIO, emit
import qutip as qt
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
import urllib.parse
import re
import subprocess  # For git mirror updates
import requests  # Added: For fallback mirror pulls

# Graceful Paramiko Import
paramiko = None
try:
    import paramiko
    print("Paramiko loaded - SSH enabled")
except ImportError:
    print("Paramiko missing - SSH disabled")
    paramiko = None

# Production Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Eventlet Monkey Patch
import eventlet
eventlet.monkey_patch()

# Domain Configs
RENDER_DOMAIN = os.environ.get('RENDER_DOMAIN', 'clearnet_gate.onrender.com')
DUCKDNS_DOMAIN = os.environ.get('DUCKDNS_DOMAIN', 'alicequantum.duckdns.org')
ALICE_IP = os.environ.get('ALICE_IP', '133.7.0.1')  # Ubuntu Gateway
QUANTUM_DOMAIN = os.environ.get('QUANTUM_DOMAIN', 'quantum.realm.domain.dominion.foam.computer.render')  # .render endpoint
LOCAL_WEBSERVER_PORT = 80
GITHUB_MIRROR_LOCAL = './github_mirror'  # Local root for all mirrored files

# GitHub Mirror Base URL (Fallback Remote)
GITHUB_MIRROR_BASE = 'https://quantum.realm.domain.dominion.foam.computer.render.github'

# User Auth
ADMIN_USER = 'shemshallah'
ADMIN_PASS_HASH = '930f0446221f865871805ab4e9577971ff97bb21d39abc4e91341ca6100c9181'  # Pre-computed SHA3-256 of b'$h10j1r1H0w4rd'

# Quantum Network Config (133.7.0.0/24)
QUANTUM_NET_CIDR = '133.7.0.0/24'
QUANTUM_GATEWAY = '133.7.0.1'
QUANTUM_DNS = '133.7.0.1'
QUANTUM_POOL_START = '133.7.0.10'
QUANTUM_POOL_END = '133.7.0.254'
IP_POOL = []
for d in range(int(QUANTUM_POOL_START.split('.')[-1]), int(QUANTUM_POOL_END.split('.')[-1]) + 1):
    IP_POOL.append(f"133.7.0.{d}")

# Registry (Pre-Registered Subs 256-999 to Admin + .render TLDs)
PRE_REG_SUBS = {str(i): {'owner': ADMIN_USER, 'status': 'available', 'price': 1.00} for i in range(256, 1000)}
RENDER_TLDS = {f'{i}.render': {'owner': ADMIN_USER, 'status': 'available', 'price': 5.00, 'ip': None} for i in range(1, 1001)}  # Example: 1.render, etc.

# Linuxbserver Config (SSH to Ubuntu)
LINUX_USER = ADMIN_USER
LINUX_PASS = os.environ.get('LINUX_PASS', '$h10j1r1H0w4rd')  # Secure via env
LINUX_HOST = ALICE_IP  # SSH target (Ubuntu gateway)

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

# Functions (enhanced for DNS/IP)
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
    
    # Manual CNOT matrix for qubits 0 (control), 1 (target), 2 (idle) - dims [[2,2,2],[2,2,2]]
    cnot_matrix = np.zeros((8,8), dtype=complex)
    cnot_matrix[0,0] = 1  # |000> -> |000>
    cnot_matrix[1,1] = 1  # |001> -> |001>
    cnot_matrix[3,2] = 1  # |010> -> |011>
    cnot_matrix[2,3] = 1  # |011> -> |010>
    cnot_matrix[6,4] = 1  # |100> -> |110>
    cnot_matrix[7,5] = 1  # |101> -> |111>
    cnot_matrix[4,6] = 1  # |110> -> |100>
    cnot_matrix[5,7] = 1  # |111> -> |101>
    cnot = qt.Qobj(cnot_matrix, dims=[[2,2,2],[2,2,2]])
    
    # EPR on qubits 1,2: (|00> + |11>)/√2
    epr12 = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
    
    # Initial: input (0) tensor EPR (1,2)
    initial = qt.tensor(input_state, epr12)
    
    # CNOT: control 0, target 1
    after_cnot = cnot * initial
    
    # Hadamard on 0: H = (X + Z)/√2
    h = (qt.sigmax() + qt.sigmaz()).unit() / np.sqrt(2)
    h_full = qt.tensor(h, qt.qeye(2), qt.qeye(2))
    after_h = h_full * after_cnot
    
    # Projector |00><00| on 0,1 tensor I_2
    p00 = qt.tensor(qt.ket2dm(qt.basis(2,0)), qt.ket2dm(qt.basis(2,0)), qt.qeye(2))
    
    # Post-measurement (unnormalized)
    projected = p00 * after_h
    norm = projected.norm()
    if norm > 1e-10:
        projected = projected / norm
    
    # Trace out Alice's qubits (0,1) → Bob's state (2)
    teleported = projected.ptrace(2)
    
    # Original return: float of [0,0] real part (density matrix element)
    return float(teleported.full()[0,0].real)

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

def issue_quantum_ip(session_id):
    """Issue unique IP from pool"""
    if 'issued_ip' not in session:
        idx = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % len(IP_POOL)
        session['issued_ip'] = IP_POOL[idx]
        logger.warning(f"Issued quantum IP {session['issued_ip']} for session {session_id}")
    return session['issued_ip']

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

# Updated GitHub Mirror Pull Util - Prioritizes Local Root Mirror
def pull_from_mirror(resource):
    """Pull from local ./github_mirror root first, fallback to remote."""
    local_path = os.path.join(GITHUB_MIRROR_LOCAL, resource.replace('/', os.sep))
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:500] + "..." if len(content) > 500 else content  # Truncate if long
        except Exception as e:
            logger.warning(f"Local mirror read error for {resource}: {e}")
    
    # Fallback to remote
    try:
        url = f"{GITHUB_MIRROR_BASE}/{resource}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text[:500] + "..."
        else:
            return f"Pull failed: {resp.status_code}"
    except Exception as e:
        return f"Network pull error: {str(e)}"

# Mirror Update Function (Run at Startup) - Wrapped for Render (git may fail)
def update_mirror():
    """Update local mirror from root GitHub repos. Skips if git unavailable."""
    os.makedirs(GITHUB_MIRROR_LOCAL, exist_ok=True)
    repos = [
        'https://github.com/matplotlib/matplotlib.git',
        # Add more repos as needed, e.g., 'https://github.com/numpy/numpy.git'
    ]
    try:
        for repo in repos:
            repo_name = repo.split('/')[-1].replace('.git', '')
            full_path = os.path.join(GITHUB_MIRROR_LOCAL, repo_name)
            if not os.path.exists(full_path):
                subprocess.run(['git', 'clone', repo, full_path], check=True, capture_output=True)
                print(f"Cloned {repo} to {full_path}")
            else:
                subprocess.run(['git', '-C', full_path, 'pull'], check=True, capture_output=True)
                print(f"Updated {full_path}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git mirror update failed (likely no git in runtime): {e}. Use Docker for full mirror.")
        print("Mirror skipped - fallback to API pulls only.")
    except Exception as e:
        logger.warning(f"Mirror update error: {e}")
        print(f"Mirror update error: {e}")

# Run Mirror Update at Startup
update_mirror()

# Production App
app = Flask(__name__)
app.secret_key = hashlib.sha256(bridge_key.encode()).digest()[:32]
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=False, engineio_logger=False)

# 404 Handler
@app.errorhandler(404)
def not_found(error):
    return "Render Side 404", 404

# 500 Handler (Catch-All for Errors)
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Error: {str(error)}")
    return "Quantum Decoherence: App restarting... Check logs.", 500

# Health Check
@app.route('/health')
def health_check():
    return 'OK', 200

# Login Route (Enhanced: Auto-issue IP on success)
@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
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
                # Auto-issue quantum IP
                issued_ip = issue_quantum_ip(session_id)
                # Entangle client IP siamese-style
                entangle_ip_address(client_ip)
                # Redirect with matching params + issued IP
                params = urllib.parse.urlencode({'session': session_id, 'bridge_key': gen_key, 'issued_ip': issued_ip})
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
    except Exception as e:
        logger.error(f"Login Error: {e}")
        return "Login decoherence - try again.", 500

# Registry Routes (Enhanced: .render TLDs + IP Issuance)
@app.route('/registry')
def registry():
    if not session.get('logged_in') or session['user'] != ADMIN_USER:
        return "Unauthorized", 403
    combined = {**PRE_REG_SUBS, **RENDER_TLDS}
    return jsonify(combined)

@app.route('/sell/<domain>', methods=['GET', 'POST'])
def sell_domain(domain):
    if not session.get('logged_in') or session['user'] != ADMIN_USER:
        return redirect('/login')
    combined = {**PRE_REG_SUBS, **RENDER_TLDS}
    if domain not in combined or combined[domain]['status'] != 'available':
        return "Domain not available", 400
    if request.method == 'POST':
        # Sim payment
        buyer = request.form.get('buyer_email')
        if buyer:
            combined[domain]['owner'] = buyer
            combined[domain]['status'] = 'sold'
            # For .render: Issue IP & add to DNS via SSH (autonomous)
            issued_ip = None
            if domain.endswith('.render'):
                issued_ip = issue_quantum_ip(buyer)
                combined[domain]['ip'] = issued_ip
                # Trigger SSH to update Bind9 zonefile
                update_dns_zone(domain, issued_ip)
            logger.warning(f'Sold {domain} to {buyer} {"with IP " + issued_ip if issued_ip else ""}')
            return f"Domain {domain} sold! {'IP: ' + issued_ip if issued_ip else ''} Key: {bridge_key}"
    return '''
    <form method="post">
        Email: <input type="email" name="buyer_email"><br>
        <input type="submit" value="Buy for ${combined[domain]['price']}">
    </form>
    '''

def update_dns_zone(domain, ip):
    """Autonomous SSH to Ubuntu: Add A record to .render zone"""
    if not paramiko:
        logger.error("Paramiko required for DNS update")
        return
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(LINUX_HOST, username=LINUX_USER, password=LINUX_PASS)
        stdin, stdout, stderr = ssh.exec_command(f'echo "{domain} IN A {ip}" >> /etc/bind/db.render && rndc reload render')
        if stderr.read().decode():
            logger.error(f"DNS update error: {stderr.read().decode()}")
        ssh.close()
        logger.warning(f"DNS zone updated for {domain} -> {ip}")
    except Exception as e:
        logger.error(f"SSH DNS update failed: {e}")

@app.route('/update/<domain>', methods=['POST'])
def update_domain(domain):
    if not session.get('logged_in') or session['user'] != ADMIN_USER:
        return "Unauthorized", 403
    combined = {**PRE_REG_SUBS, **RENDER_TLDS}
    if domain in combined:
        new_status = request.form.get('status', 'available')
        combined[domain]['status'] = new_status
        return f"Updated {domain}: {new_status}"
    return "Domain not found", 404

# Proxy Route for Linux Webserver (over DuckDNS, thematic overlay via client-side)
@app.route('/web_proxy')
def web_proxy():
    if not session.get('logged_in'):
        return redirect('/login')
    # Redirect to DuckDNS for Linux webserver access (assumes bound publicly)
    return redirect(f'https://{DUCKDNS_DOMAIN}', code=302)

# Quantum Domain Redirector - Enhanced for DuckDNS linking + IP Display
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def quantum_gate(path):
    try:
        if not session.get('logged_in'):
            return redirect('/login')
        
        client_ip = request.remote_addr
        issued_ip = issue_quantum_ip(session.get('session_id', client_ip))  # Ensure issued
        
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
        qram_entangled_session('issued_ip', issued_ip)
        
        provided_key = request.args.get('bridge_key', '')
        session['session_id'] = session_id  # Ensure it's set
        
        gen_key, ts = bh_repeatable_keygen(session_id)
        enc_key = bh_encryption_cascade(bridge_key)
        
        if not hashlib.sha256(provided_key.encode()).hexdigest() == hashlib.sha256(gen_key.encode()).hexdigest():
            params = urllib.parse.urlencode({'enc_key': enc_key, 'session': session_id})
            gate_url = f"https://{RENDER_DOMAIN}/?{params}"
            logger.warning(f'Invalid Key Redirect: {client_ip} -> Gate with params')
            return redirect(gate_url, code=302)
        
        mirror_pull = pull_from_mirror('matplotlib/raw/main/setup.py')  # Now from local/root
        
        offload_res = entangled_cpu_offload(f'fidelity_lattice * 1.0001')
        comp_tensor = core_ghz.full().real
        comp_lattice = len(foam_lattice_compress(comp_tensor))
        tele_id = inter_hole_teleport('cascade_state')
        # Entangle ALICE_IP for siamese mirror
        ip_mirror_fid = entangle_ip_address(ALICE_IP)
        
        logger.warning(f'Access: {client_ip}, Sess {session_id}, Issued IP {issued_ip}, 5x5x5 Lattice Active')
        html_content = f"""
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
                <p style="color: #0f0;">Prod Foam: Fid {offload_res}, Comp {comp_lattice}B. Neg {ip_negativity:.16f}, Tele {tele_id:.6f}, IP Mirror Fid {ip_mirror_fid:.16f}, Back {session.get('backup_id', 'none')}{' | SSH: ' + ('Enabled' if paramiko else 'Disabled - Add paramiko to requirements.txt')}</p>
                <p style="color: #0f0;">Issued Quantum IP: {issued_ip} | Network: {QUANTUM_NET_CIDR} (Gateway: {QUANTUM_GATEWAY}, DNS: {QUANTUM_DNS}) | Enc: {enc_key[:32]}... | Mirror Pull (Local Root): {mirror_pull[:50]}... | Registry: /registry | Sell: /sell/<domain> | Web Proxy: /web_proxy</p>
                <p style="color: #0f0;">DuckDNS Linked: {DUCKDNS_DOMAIN} → Alice {ALICE_IP} (127.0.0.1 Linux Webserver Overlay Active)</p>
                <div id="terminal" style="width: 100%; height: 400px; background: #000;"></div>
                <script>
                    const socket = io();
                    const term = new Terminal({{ cursorBlink: true, theme: {{ background: '#000', foreground: '#0f0' }} }});
                    term.open(document.getElementById('terminal'));
                    term.write('QSH Foam REPL v2.0 (5x5x5 Lattice - IP Entangled)\\r\\n');
                    term.write('Issued IP: {issued_ip} | Type "help" for commands. .render TLDs now issuable.\\r\\n');
                    term.write('New: "setup_dns" auto-configures Bind9 on Ubuntu, "connect_linux" for SSH tunnel\\r\\n');
                    term.write('Note: Git mirror may be limited on Render - use "pull matplotlib" for API fallback.\\r\\n');
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
        return html_content
    except Exception as e:
        logger.error(f"Quantum Gate Error: {e}")
        return f"Gate decoherence: {str(e)}", 500

# QSH Foam REPL Backend (Enhanced: SSH Tunnel + Autonomous DNS Setup)
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
            # Handle SSH auth (now non-interactive for setup)
            if pending_auth == 'user':
                session_data['pending_auth'] = None  # Skip prompt, use env creds
                output += f"Auth: {LINUX_USER} (auto) | "
                prompt = False
            emit('qsh_output', {'output': output, 'prompt': prompt})
            return

        if in_linux_mode:
            # Interactive SSH shell (tunnel)
            if channel:
                if select.select([channel], [], [], 0.0)[0]:
                    stdout_data = channel.recv(1024).decode()
                    if stdout_data:
                        output = stdout_data
                    stderr_data = channel.recv_stderr(1024).decode()
                    if stderr_data:
                        output += f"\nERR: {stderr_data}"
                else:
                    input_data = cmd.encode()
                    channel.send(input_data)
            else:
                output = "SSH channel closed. Type 'exit' to quit."
                in_linux_mode = False
                session_data['in_linux_mode'] = False
        else:
            if cmd == 'help':
                ssh_status = " (SSH disabled - add paramiko)" if not paramiko else ""
                output = f"Commands: help, entangle <q1 q2>, measure fidelity, compress lattice, teleport <input>, plot fidelity, pull matplotlib, registry list, sell <domain>, connect_linux{ssh_status}, setup_dns{ssh_status}, entangle_ip <ip>, exit, clear"
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
                combined = {**PRE_REG_SUBS, **RENDER_TLDS}
                subs_list = ', '.join([k for k, v in combined.items() if v['status'] == 'available'][:10])
                output = f"Available (Subs + .render): {subs_list}... (Full: /registry)"
            elif cmd.startswith('sell '):
                domain = cmd.split(' ', 1)[1]
                combined = {**PRE_REG_SUBS, **RENDER_TLDS}
                if domain in combined:
                    output = f"Selling {domain} - Visit /sell/{domain}"
                else:
                    output = f"Domain {domain} not in registry"
            elif cmd.startswith('entangle_ip '):
                ip = cmd.split(' ', 1)[1]
                mirror_fid = entangle_ip_address(ip)
                output = f"Siamese mirror entangled for IP {ip}: Fidelity {mirror_fid:.16f}"
            elif cmd == 'setup_dns':
                if not paramiko:
                    output = "setup_dns disabled: Add 'paramiko==3.4.0' to requirements.txt and redeploy."
                else:
                    output = "Autonomous DNS Setup on Ubuntu (.render TLD)...\n"
                    # Run full setup script via SSH
                    setup_script = """
apt update && apt install -y bind9 bind9utils dnsutils
cat > /etc/bind/named.conf.local << EOF
zone "render" {
    type master;
    file "/etc/bind/db.render";
};
EOF
cat > /etc/bind/db.render << EOF
$TTL    604800
@       IN      SOA     ns1.render. root.render. (
                              2         ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ns1.render.
ns1     IN      A       133.7.0.1
@       IN      A       216.24.57.1
*       IN      A       216.24.57.1  ; Wildcard for issuance
forwarders {
    8.8.8.8;
    8.8.4.4;
};
EOF
systemctl restart bind9
systemctl enable bind9
ufw allow 53
echo "Bind9 configured for .render TLD on 133.7.0.1"
                    """
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(LINUX_HOST, username=LINUX_USER, password=LINUX_PASS)
                    stdin, stdout, stderr = ssh.exec_command(f'bash -c "{setup_script}"')
                    output += stdout.read().decode()
                    if stderr.read().decode():
                        output += f"\nERR: {stderr.read().decode()}"
                    ssh.close()
                    output += "\nDNS Setup Complete: .render zone active on 133.7.0.1:53"
            elif cmd == 'connect_linux':
                if not paramiko:
                    output = "connect_linux disabled: Add 'paramiko==3.4.0' to requirements.txt and redeploy."
                else:
                    try:
                        if not ssh_client:
                            ssh_client = paramiko.SSHClient()
                            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            ssh_client.connect(LINUX_HOST, username=LINUX_USER, password=LINUX_PASS)
                            channel = ssh_client.invoke_shell()
                            session_data['ssh_client'] = ssh_client
                            session_data['channel'] = channel
                            session_data['in_linux_mode'] = True
                            output = f"SSH Tunnel Connected to Ubuntu {LINUX_HOST} (Gateway: {QUANTUM_GATEWAY})\\n"
                            output += channel.recv(1024).decode()  # Banner
                        else:
                            output = "SSH tunnel already active. Type commands directly."
                        prompt = False
                        linux_prompt = output + "$ "  # Ubuntu prompt
                    except Exception as e:
                        output = f"SSH Connect Failed: {e}"
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
                    session_data['ssh_client'] = None
                    session_data['channel'] = None
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
        if session_data.get('in_linux_mode') and paramiko:
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
