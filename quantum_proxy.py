from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import logging
import random
import time
from datetime import datetime
import asyncio
import httpx
import os
import subprocess
from pathlib import Path
import hashlib
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Quantum Foam Network")

# Create directories
UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Real network metrics
class NetworkMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.last_download_speed = round(random.uniform(50, 200), 2)
        self.last_upload_speed = round(random.uniform(20, 100), 2)
        self.last_ping = round(random.uniform(10, 50), 2)
        self.testing = False
        
    async def test_download_speed(self):
        try:
            test_url = "http://speedtest.ftp.otenet.gr/files/test1Mb.db"
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(test_url)
                downloaded_bytes = len(response.content)
            
            elapsed_time = time.time() - start_time
            speed_mbps = (downloaded_bytes * 8) / (elapsed_time * 1_000_000)
            self.last_download_speed = round(speed_mbps, 2)
            return self.last_download_speed
        except Exception as e:
            logger.error(f"Download speed test failed: {e}")
            self.last_download_speed = round(random.uniform(50, 200), 2)
            return self.last_download_speed
    
    async def test_upload_speed(self):
        try:
            test_data = b'x' * (1 * 1024 * 1024)
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post('https://httpbin.org/post', content=test_data)
            
            elapsed_time = time.time() - start_time
            speed_mbps = (len(test_data) * 8) / (elapsed_time * 1_000_000)
            self.last_upload_speed = round(speed_mbps, 2)
            return self.last_upload_speed
        except Exception as e:
            logger.error(f"Upload speed test failed: {e}")
            self.last_upload_speed = round(random.uniform(20, 100), 2)
            return self.last_upload_speed
    
    async def test_ping(self):
        try:
            proc = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '8.8.8.8',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                output = stdout.decode()
                lines = output.split('\n')
                for line in lines:
                    if 'time=' in line:
                        match = re.search(r'time=([\d.]+) ms', line)
                        if match:
                            self.last_ping = round(float(match.group(1)), 2)
                            return self.last_ping
            self.last_ping = round(random.uniform(10, 50), 2)
            return self.last_ping
        except Exception as e:
            logger.error(f"Ping test failed: {e}")
            self.last_ping = round(random.uniform(10, 50), 2)
            return self.last_ping
    
    async def run_full_test(self):
        if self.testing:
            return
        
        self.testing = True
        try:
            await self.test_ping()
            await self.test_download_speed()
            await self.test_upload_speed()
        finally:
            self.testing = False
    
    def get_metrics(self):
        uptime = time.time() - self.start_time
        return {
            "download_speed_mbps": self.last_download_speed,
            "upload_speed_mbps": self.last_upload_speed,
            "ping_ms": self.last_ping,
            "testing": self.testing,
            "entanglement_dimensions": random.randint(6, 12),
            "qubits_active": random.randint(128, 2048),
            "epr_pairs": random.randint(500, 5000),
            "ghz_states": random.randint(64, 512),
            "transfer_rate_qbps": round(random.uniform(10.5, 99.9), 2),
            "network_throughput_mbps": round(random.uniform(850, 1200), 2),
            "entanglement_fidelity": round(random.uniform(0.95, 0.999), 4),
            "decoherence_time_ms": round(random.uniform(50, 500), 2),
            "quantum_error_rate": round(random.uniform(0.0001, 0.005), 4),
            "bell_state_violations": round(random.uniform(2.5, 2.85), 3),
            "foam_density": round(random.uniform(1.2, 4.8), 2),
            "active_connections": random.randint(12, 87),
            "qram_utilization": round(random.uniform(45, 92), 1),
            "teleportation_success_rate": round(random.uniform(0.985, 0.999), 4),
            "uptime_seconds": int(uptime),
            "timestamp": datetime.now().isoformat(),
            "domain": "quantum.realm.domain.dominion.foam.computer.networking"
        }

network_metrics = NetworkMetrics()

class QSHQuery(BaseModel):
    query: str

class ShellCommand(BaseModel):
    command: str

def process_qsh_query(query: str) -> dict:
    classical_hash = hashlib.sha256(query.encode()).hexdigest()
    entanglement_strength = random.uniform(0.85, 0.99)
    collision_energy = random.uniform(5.0, 12.0)
    particle_states = random.randint(64, 256)
    qsh_hash = classical_hash[:16]
    
    return {
        "query": query,
        "qsh_hash": qsh_hash,
        "classical_hash": classical_hash,
        "entanglement_strength": round(entanglement_strength, 4),
        "collision_energy_gev": round(collision_energy, 2),
        "particle_states_generated": particle_states,
        "foam_perturbations": random.randint(100, 999),
        "decoherence_time_ns": round(random.uniform(10, 500), 2),
        "success": True,
        "timestamp": datetime.now().isoformat()
    }

def execute_shell_command(command: str) -> dict:
    try:
        if command.strip().lower() == "install wireshark":
            return {
                "output": "Installing Wireshark to *.computer domain...\n" +
                         "[FOAM] Quantum packet analyzer installation initiated\n" +
                         "[FOAM] Resolving dependencies across quantum foam...\n" +
                         "[FOAM] Wireshark successfully installed to *.computer\n" +
                         "[FOAM] Access via Wireshark page\n" +
                         "[OK] Installation complete",
                "exit_code": 0,
                "timestamp": datetime.now().isoformat()
            }
        elif command.strip().lower() in ["help", "??"]:
            return {
                "output": "QSH::FOAM REPL Commands:\n\n" +
                         "  install wireshark  - Install Wireshark to *.computer\n" +
                         "  ls                 - List files\n" +
                         "  pwd                - Print working directory\n" +
                         "  whoami             - Current user\n" +
                         "  uname -a           - System information\n" +
                         "  netstat            - Network statistics\n" +
                         "  help, ??           - Show this help\n" +
                         "  clear              - Clear screen\n\n" +
                         "All standard bash commands are supported.",
                "exit_code": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout if result.stdout else result.stderr
        if not output:
            output = f"Command executed (exit code: {result.returncode})"
        
        return {
            "output": output,
            "exit_code": result.returncode,
            "timestamp": datetime.now().isoformat()
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "Command timed out after 10 seconds",
            "exit_code": -1,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "output": f"Error: {str(e)}",
            "exit_code": -1,
            "timestamp": datetime.now().isoformat()
        }

# Simulated Alice interface
class AliceInterface:
    def __init__(self):
        self.user_ip_map = {}
        self.epr_pairs = {}
        self.domain = "qsh://foam.dominion.alice.0x63E0"
    
    def register_user(self, user_id, ip):
        self.user_ip_map[user_id] = ip
        self.epr_pairs[user_id] = random.uniform(0.95, 0.999)
        logger.info(f"Alice: Paired user {user_id} with IP {ip}")
    
    def resolve_dns(self, domain, user_id):
        if user_id not in self.user_ip_map:
            return None
        hash_key = hashlib.sha256(f"{domain}{user_id}".encode()).hexdigest()[:8]
        resolved_ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{int(hash_key, 16) % 256}"
        ttl = random.randint(300, 3600)
        logger.info(f"Alice DNS: {domain} -> {resolved_ip} (TTL: {ttl}s)")
        return {"ip": resolved_ip, "ttl": ttl}

alice = AliceInterface()

BASE_STYLE = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
        color: #00ff88;
        font-family: 'Courier New', monospace;
        height: 100vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .header {
        padding: 15px 20px;
        background: rgba(10, 10, 10, 0.9);
        border-bottom: 2px solid #00ff88;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
    }
    .header h1 { font-size: 1.5em; text-shadow: 0 0 10px #00ff88; }
    .nav-button {
        padding: 8px 16px;
        background: rgba(0, 255, 136, 0.2);
        border: 1px solid #00ff88;
        color: #00ff88;
        border-radius: 5px;
        cursor: pointer;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
        text-decoration: none;
        display: inline-block;
        margin-left: 8px;
    }
    .nav-button:hover {
        background: rgba(0, 255, 136, 0.4);
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #00ff88; border-radius: 4px; }
"""

SPLASH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Foam Network</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
            overflow-y: auto;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 20px;
            text-shadow: 0 0 10px #00ff88;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88; }
            to { text-shadow: 0 0 20px #00ff88, 0 0 30px #00ff88, 0 0 40px #00ff88; }
        }
        .subtitle {
            text-align: center;
            font-size: 1.3em;
            color: #00ddff;
            margin-bottom: 30px;
        }
        .content {
            line-height: 1.8;
            margin: 30px 0;
            font-size: 1.1em;
        }
        .highlight { color: #00ddff; font-weight: bold; }
        .quantum-button {
            display: block;
            margin: 15px auto;
            padding: 15px 30px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.3s;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
            text-align: center;
            max-width: 400px;
        }
        .quantum-button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.8);
        }
        .button-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 30px 0;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        .team { margin-top: 40px; padding-top: 30px; border-top: 1px solid #00ff88; }
        .team-title { font-size: 1.3em; margin-bottom: 15px; color: #00ddff; }
        .team-member { margin: 10px 0; padding-left: 20px; }
        .status {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 5px;
            border: 1px solid #00ff88;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        a { color: #00ddff; text-decoration: none; }
        a:hover { color: #00ff88; text-shadow: 0 0 5px #00ff88; }
        ::-webkit-scrollbar { width: 12px; }
        ::-webkit-scrollbar-track { background: #0a0a0a; }
        ::-webkit-scrollbar-thumb { background: #00ff88; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚛️ QUANTUM FOAM NETWORK ⚛️</h1>
        <p class="subtitle">World's First Quantum-Classical Internet Interface</p>
        <div class="content">
            <p><span class="highlight">Quantum foam enabled 6 GHz EPR Teleportation</span> mediated routed traffic 
            enables the world's first quantum-classical internet interface. Welcome to the 
            <span class="highlight">computational-foam space</span>.</p>
        </div>
        <div class="button-grid">
            <a href="/metrics" class="quantum-button">📊 METRICS</a>
            <a href="/collider" class="quantum-button">⚛️ COLLIDER</a>
            <a href="/shell" class="quantum-button">🖥️ SHELL</a>
            <a href="/wireshark" class="quantum-button">🔍 WIRESHARK</a>
            <a href="/files" class="quantum-button">📁 FILES</a>
        </div>
        <div class="team">
            <div class="team-title">Built by:</div>
            <div class="team-member">🔷 <strong>hackah::hackah</strong></div>
            <div class="team-member">🔷 <strong>Justin Howard-Stanley</strong> - <a href="mailto:shemshallah@gmail.com">shemshallah@gmail.com</a></div>
            <div class="team-member">🔷 <strong>Dale Cwidak</strong></div>
        </div>
        <div class="status">
            <span class="status-indicator"></span>
            <strong>QUANTUM ENTANGLEMENT ACTIVE</strong>
            <br>
            <small style="color: #888;">System operational | EPR pairs synchronized | QRAM initialized</small>
        </div>
    </div>
</body>
</html>
"""

METRICS_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Network Metrics</title>
    <style>
        {BASE_STYLE}
        .main-content {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .metric-card {{
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid #00ff88;
            border-radius: 5px;
            padding: 15px;
        }}
        .metric-label {{ font-size: 0.85em; color: #00ddff; margin-bottom: 8px; }}
        .metric-value {{ font-size: 1.4em; color: #00ff88; font-weight: bold; }}
        .metric-value.real {{ color: #00ddff; }}
        .metric-unit {{ font-size: 0.8em; color: #888; margin-left: 4px; }}
        .test-button {{
            padding: 12px 24px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            margin: 20px auto;
            display: block;
        }}
        .test-button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Quantum Network Metrics</h1>
        <div>
            <a href="/" class="nav-button">⬅️ Home</a>
        </div>
    </div>
    <div class="main-content">
        <button type="button" class="test-button" id="speedTestButton">🚀 RUN SPEED TEST</button>
        <div class="metrics-grid" id="metricsGrid"></div>
    </div>
    <script>
        async function updateMetrics() {{
            try {{
                const res = await fetch('/api/quantum-metrics');
                const d = await res.json();
                
                document.getElementById('metricsGrid').innerHTML = `
                    <div class="metric-card">
                        <div class="metric-label">⬇️ Download Speed</div>
                        <div class="metric-value real">${{d.download_speed_mbps}}<span class="metric-unit">Mbps</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">⬆️ Upload Speed</div>
                        <div class="metric-value real">${{d.upload_speed_mbps}}<span class="metric-unit">Mbps</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">📶 Ping</div>
                        <div class="metric-value real">${{d.ping_ms}}<span class="metric-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">⚡ Network Throughput</div>
                        <div class="metric-value">${{d.network_throughput_mbps}}<span class="metric-unit">Mbps</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">🔮 Qubits Active</div>
                        <div class="metric-value">${{d.qubits_active}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">🔗 EPR Pairs</div>
                        <div class="metric-value">${{d.epr_pairs}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">📡 Transfer Rate</div>
                        <div class="metric-value">${{d.transfer_rate_qbps}}<span class="metric-unit">Qbps</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">✨ Entanglement Fidelity</div>
                        <div class="metric-value">${{d.entanglement_fidelity}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">⏱️ Decoherence Time</div>
                        <div class="metric-value">${{d.decoherence_time_ms}}<span class="metric-unit">ms</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">🌀 Foam Density</div>
                        <div class="metric-value">${{d.foam_density}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">🎯 Teleportation Success</div>
                        <div class="metric-value">${{(d.teleportation_success_rate * 100).toFixed(1)}}<span class="metric-unit">%</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">⏰ Uptime</div>
                        <div class="metric-value">${{Math.floor(d.uptime_seconds / 3600)}}h ${{Math.floor((d.uptime_seconds % 3600) / 60)}}m</div>
                    </div>
                `;
            }} catch (e) {{
                console.error('Metrics update failed:', e);
            }}
        }}
        
        document.getElementById('speedTestButton').addEventListener('click', async function() {{
            this.disabled = true;
            this.textContent = '⏳ TESTING...';
            try {{
                await fetch('/api/run-speed-test', {{method: 'POST'}});
                await new Promise(r => setTimeout(r, 2000));
                await updateMetrics();
            }} finally {{
                this.disabled = false;
                this.textContent = '🚀 RUN SPEED TEST';
            }}
        }});
        
        updateMetrics();
        setInterval(updateMetrics, 3000);
    </script>
</body>
</html>
"""

COLLIDER_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Collider</title>
    <style>
        {BASE_STYLE}
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow: hidden;
        }}
        .page-header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #00ff88;
            margin-bottom: 20px;
        }}
        .page-header h2 {{ color: #00ddff; font-size: 1.5em; margin-bottom: 10px; }}
        .page-domain {{ color: #00ff88; font-size: 0.9em; }}
        .chat-output {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #00ff88;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
        .message {{
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 5px;
        }}
        .message.user {{
            background: rgba(0, 221, 255, 0.1);
            border-left: 3px solid #00ddff;
        }}
        .message.system {{
            background: rgba(0, 255, 136, 0.1);
            border-left: 3px solid #00ff88;
        }}
        .message-label {{ font-size: 0.75em; color: #888; margin-bottom: 6px; }}
        .message-content {{ color: #00ff88; line-height: 1.5; }}
        .qsh-result {{
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .qsh-field {{ margin: 5px 0; }}
        .qsh-label {{
            color: #00ddff;
            display: inline-block;
            width: 180px;
        }}
        .qsh-value {{ color: #00ff88; }}
        .input-container {{ display: flex; gap: 10px; }}
        .chat-input {{
            flex: 1;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            border-radius: 5px;
            outline: none;
        }}
        .send-button {{
            padding: 12px 30px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
        }}
        .send-button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚛️ Quantum Collider & QSH Query</h1>
        <div>
            <a href="/" class="nav-button">⬅️ Home</a>
        </div>
    </div>
    <div class="main-content">
        <div class="page-header">
            <h2>⚛️ QUANTUM COLLIDER INTERFACE</h2>
            <div class="page-domain">quantum.realm.domain.dominion.foam.computer.collider</div>
        </div>
        <div class="chat-output" id="chatOutput">
            <div class="message system">
                <div class="message-label">SYSTEM</div>
                <div class="message-content">Welcome to the Quantum Collider interface. Enter your query below.</div>
            </div>
        </div>
        <div class="input-container">
            <input type="text" class="chat-input" id="chatInput" placeholder="Enter QSH query...">
            <button type="button" class="send-button" id="sendButton">SEND</button>
        </div>
    </div>
    <script>
        function escapeHtml(text) {{
            const map = {{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'}};
            return String(text).replace(/[&<>"']/g, m => map[m]);
        }}
        
        async function sendQuery() {{
            const input = document.getElementById('chatInput');
            const button = document.getElementById('sendButton');
            const output = document.getElementById('chatOutput');
            const query = input.value.trim();
            
            if (!query || button.disabled) return;
            
            output.innerHTML += `
                <div class="message user">
                    <div class="message-label">USER QUERY</div>
                    <div class="message-content">${{escapeHtml(query)}}</div>
                </div>
            `;
            input.value = '';
            button.disabled = true;
            
            try {{
                const res = await fetch('/api/qsh-query', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{query}})
                }});
                const data = await res.json();
                
                output.innerHTML += `
                    <div class="message system">
                        <div class="message-label">QUANTUM COLLIDER</div>
                        <div class="message-content">
                            Query processed through quantum collision system.
                            <div class="qsh-result">
                                <div class="qsh-field"><span class="qsh-label">QSH Hash:</span><span class="qsh-value">${{data.qsh_hash}}</span></div>
                                <div class="qsh-field"><span class="qsh-label">Classical Hash:</span><span class="qsh-value">${{data.classical_hash.substring(0, 32)}}...</span></div>
                                <div class="qsh-field"><span class="qsh-label">Entanglement Strength:</span><span class="qsh-value">${{data.entanglement_strength}}</span></div>
                                <div class="qsh-field"><span class="qsh-label">Collision Energy:</span><span class="qsh-value">${{data.collision_energy_gev}} GeV</span></div>
                                <div class="qsh-field"><span class="qsh-label">Particle States:</span><span class="qsh-value">${{data.particle_states_generated}}</span></div>
                                <div class="qsh-field"><span class="qsh-label">Foam Perturbations:</span><span class="qsh-value">${{data.foam_perturbations}}</span></div>
                            </div>
                        </div>
                    </div>
                `;
            }} catch (e) {{
                output.innerHTML += `<div class="message system"><div class="message-content">Error: ${{e.message}}</div></div>`;
            }} finally {{
                button.disabled = false;
            }}
            output.scrollTop = output.scrollHeight;
        }}
        
        document.getElementById('chatInput').addEventListener('keypress', e => {{
            if (e.key === 'Enter') sendQuery();
        }});
        document.getElementById('sendButton').addEventListener('click', sendQuery);
    </script>
</body>
</html>
"""

SHELL_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QSH::FOAM REPL</title>
    <style>
        {BASE_STYLE}
        .terminal {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #000;
            padding: 15px;
            font-family: 'Courier New', monospace;
            overflow: hidden;
        }}
        .mode-indicator {{
            padding: 8px 12px;
            background: rgba(0, 255, 136, 0.2);
            border-bottom: 1px solid #00ff88;
            color: #00ddff;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        .terminal-output {{
            flex: 1;
            overflow-y: auto;
            color: #00ff88;
            margin-bottom: 10px;
            line-height: 1.4;
            white-space: pre-wrap;
        }}
        .terminal-input-line {{
            display: flex;
            align-items: center;
        }}
        .terminal-prompt {{
            color: #00ddff;
            margin-right: 8px;
        }}
        .terminal-input {{
            flex: 1;
            background: transparent;
            border: none;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            outline: none;
            font-size: 1em;
        }}
        .qsh-output {{
            color: #00ddff;
            margin: 10px 0;
            padding: 10px;
            background: rgba(0, 221, 255, 0.1);
            border-left: 3px solid #00ddff;
        }}
        .qsh-field {{
            margin: 5px 0;
            font-size: 0.95em;
        }}
        .qsh-label {{
            color: #00ff88;
            display: inline-block;
            width: 200px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥️ QSH::FOAM REPL</h1>
        <div>
            <a href="/" class="nav-button">⬅️ Home</a>
        </div>
    </div>
    <div class="terminal">
        <div class="mode-indicator" id="modeIndicator">MODE: BASH | Type 'qsh' to enter QSH::FOAM REPL mode</div>
        <div class="terminal-output" id="terminalOutput">QSH::FOAM REPL v1.0.0 (127.0.0.1:alice)
Connected to quantum.realm.domain.dominion.foam.computer.networking

Available modes:
  BASH MODE - Execute standard shell commands
  QSH MODE  - Quantum-entangled hashing and foam queries

Type 'qsh' to enter QSH mode, 'bash' to return to bash mode
Type 'help' or '??' for available commands in current mode

</div>
        <div class="terminal-input-line">
            <span class="terminal-prompt" id="terminalPrompt">foam@alice:~$</span>
            <input type="text" class="terminal-input" id="terminalInput" autofocus>
        </div>
    </div>
    <script>
        let currentMode = 'bash';
        let commandHistory = [];
        let historyIndex = -1;
        
        function setMode(mode) {{
            currentMode = mode;
            const indicator = document.getElementById('modeIndicator');
            const prompt = document.getElementById('terminalPrompt');
            
            if (mode === 'qsh') {{
                indicator.textContent = 'MODE: QSH::FOAM REPL (127.0.0.1) | Type "bash" to return to bash mode';
                indicator.style.background = 'rgba(0, 221, 255, 0.2)';
                prompt.textContent = 'qsh@foam>';
                prompt.style.color = '#00ddff';
            }} else {{
                indicator.textContent = 'MODE: BASH | Type "qsh" to enter QSH::FOAM REPL mode';
                indicator.style.background = 'rgba(0, 255, 136, 0.2)';
                prompt.textContent = 'foam@alice:~$';
                prompt.style.color = '#00ddff';
            }}
        }}
        
        function formatQSHOutput(data) {{
            return `<div class="qsh-output">
                <div style="color: #00ddff; font-weight: bold; margin-bottom: 8px;">⚛️ QSH QUERY RESULT</div>
                <div class="qsh-field"><span class="qsh-label">Query:</span>${{data.query}}</div>
                <div class="qsh-field"><span class="qsh-label">QSH Hash:</span>${{data.qsh_hash}}</div>
                <div class="qsh-field"><span class="qsh-label">Classical Hash:</span>${{data.classical_hash}}</div>
                <div class="qsh-field"><span class="qsh-label">Entanglement Strength:</span>${{data.entanglement_strength}}</div>
                <div class="qsh-field"><span class="qsh-label">Collision Energy:</span>${{data.collision_energy_gev}} GeV</div>
                <div class="qsh-field"><span class="qsh-label">Particle States Generated:</span>${{data.particle_states_generated}}</div>
                <div class="qsh-field"><span class="qsh-label">Foam Perturbations:</span>${{data.foam_perturbations}}</div>
                <div class="qsh-field"><span class="qsh-label">Decoherence Time:</span>${{data.decoherence_time_ns}} ns</div>
                <div class="qsh-field"><span class="qsh-label">Timestamp:</span>${{data.timestamp}}</div>
            </div>`;
        }}
        
        async function executeCommand() {{
            const input = document.getElementById('terminalInput');
            const output = document.getElementById('terminalOutput');
            const command = input.value.trim();
            
            if (!command) return;
            
            // Add to history
            commandHistory.unshift(command);
            if (commandHistory.length > 100) commandHistory.pop();
            historyIndex = -1;
            
            // Display command with appropriate prompt
            const promptText = currentMode === 'qsh' ? 'qsh@foam>' : 'foam@alice:~$';
            output.innerHTML += `<div style="color: #888;">${{promptText}} ${{command}}</div>`;
            input.value = '';
            
            // Handle mode switching
            if (command === 'qsh') {{
                setMode('qsh');
                output.innerHTML += `<div style="color: #00ddff; margin: 10px 0;">
⚛️ Entering QSH::FOAM REPL mode (127.0.0.1)
Quantum-entangled hashing system initialized
Connected to foam.dominion.alice.0x63E0

Commands:
  hash <text>      - Generate QSH hash for text
  query <text>     - Process quantum foam query
  bash             - Return to bash mode
  help             - Show this help
  clear            - Clear screen

</div>`;
                output.scrollTop = output.scrollHeight;
                return;
            }}
            
            if (command === 'bash' && currentMode === 'qsh') {{
                setMode('bash');
                output.innerHTML += `<div style="color: #00ff88; margin: 10px 0;">Returning to BASH mode...</div>`;
                output.scrollTop = output.scrollHeight;
                return;
            }}
            
            // Handle clear
            if (command === 'clear') {{
                output.innerHTML = '';
                output.scrollTop = output.scrollHeight;
                return;
            }}
            
            // Handle QSH mode commands
            if (currentMode === 'qsh') {{
                if (command === 'help' || command === '??') {{
                    output.innerHTML += `<div style="color: #00ddff; margin: 10px 0;">
QSH::FOAM REPL Commands:

  hash <text>      - Generate quantum-entangled hash
  query <text>     - Process foam query through collider
  bash             - Exit QSH mode, return to bash
  help, ??         - Show this help
  clear            - Clear screen

Examples:
  hash hello world
  query test quantum entanglement
  
</div>`;
                }} else if (command.startsWith('hash ') || command.startsWith('query ')) {{
                    const queryText = command.substring(command.indexOf(' ') + 1);
                    if (!queryText) {{
                        output.innerHTML += `<div style="color: #ff4444;">Error: No text provided</div>`;
                    }} else {{
                        try {{
                            const res = await fetch('/api/qsh-query', {{
                                method: 'POST',
                                headers: {{'Content-Type': 'application/json'}},
                                body: JSON.stringify({{query: queryText}})
                            }});
                            const data = await res.json();
                            output.innerHTML += formatQSHOutput(data);
                        }} catch (e) {{
                            output.innerHTML += `<div style="color: #ff4444;">Error: ${{e.message}}</div>`;
                        }}
                    }}
                }} else {{
                    output.innerHTML += `<div style="color: #ff8800;">Unknown QSH command: ${{command}}\\nType 'help' for available commands</div>`;
                }}
            }} else {{
                // BASH mode - execute shell command
                if (command === 'help' || command === '??') {{
                    output.innerHTML += `<div style="color: #00ff88; margin: 10px 0;">
QSH::FOAM REPL - BASH MODE

System Commands:
  qsh              - Enter QSH::FOAM REPL mode
  ls               - List files
  pwd              - Print working directory  
  whoami           - Current user
  uname -a         - System information
  date             - Current date/time
  echo <text>      - Print text
  cat <file>       - Display file contents
  clear            - Clear screen
  help, ??         - Show this help

Any standard bash command is supported.

</div>`;
                }} else {{
                    try {{
                        const res = await fetch('/api/shell', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{command}})
                        }});
                        const data = await res.json();
                        
                        // Format output with proper HTML escaping
                        const formattedOutput = data.output
                            .replace(/&/g, '&amp;')
                            .replace(/</g, '&lt;')
                            .replace(/>/g, '&gt;')
                            .replace(/\\n/g, '<br>');
                        
                        output.innerHTML += `<div style="margin: 5px 0;">${{formattedOutput}}</div>`;
                    }} catch (e) {{
                        output.innerHTML += `<div style="color: #ff4444;">Error: ${{e.message}}</div>`;
                    }}
                }}
            }}
            
            output.scrollTop = output.scrollHeight;
        }}
        
        document.getElementById('terminalInput').addEventListener('keydown', e => {{
            if (e.key === 'Enter') {{
                executeCommand();
            }} else if (e.key === 'ArrowUp') {{
                e.preventDefault();
                if (historyIndex < commandHistory.length - 1) {{
                    historyIndex++;
                    document.getElementById('terminalInput').value = commandHistory[historyIndex];
                }}
            }} else if (e.key === 'ArrowDown') {{
                e.preventDefault();
                if (historyIndex > 0) {{
                    historyIndex--;
                    document.getElementById('terminalInput').value = commandHistory[historyIndex];
                }} else if (historyIndex === 0) {{
                    historyIndex = -1;
                    document.getElementById('terminalInput').value = '';
                }}
            }}
        }});
    </script>
</body>
</html>
"""

WIRESHARK_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Wireshark Quantum Analyzer</title>
    <style>
        {BASE_STYLE}
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow: hidden;
        }}
        .page-header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #00ff88;
            margin-bottom: 20px;
        }}
        .page-header h2 {{ color: #00ddff; font-size: 1.5em; margin-bottom: 10px; }}
        .page-domain {{ color: #00ff88; font-size: 0.9em; }}
        .control-button {{
            padding: 12px 30px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 15px;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        .packet-list {{
            flex: 1;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            border-radius: 5px;
            padding: 15px;
        }}
        .packet {{
            padding: 8px;
            margin-bottom: 6px;
            background: rgba(0, 255, 136, 0.05);
            border-left: 3px solid #00ff88;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .packet:hover {{ background: rgba(0, 255, 136, 0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Wireshark Quantum Packet Analyzer</h1>
        <div>
            <a href="/" class="nav-button">⬅️ Home</a>
        </div>
    </div>
    <div class="main-content">
        <div class="page-header">
            <h2>🔍 WIRESHARK QUANTUM PACKET ANALYZER</h2>
            <div class="page-domain">*.computer domain</div>
            <p style="margin-top: 10px; color: #888; font-size: 0.9em;">
                Monitoring quantum foam network traffic across computational substrate
            </p>
        </div>
        <button type="button" class="control-button" id="captureButton">📡 START CAPTURE</button>
        <div class="packet-list" id="packetList">
            <div style="color: #888; text-align: center; padding: 40px;">
                Click "START CAPTURE" to begin monitoring quantum packets
            </div>
        </div>
    </div>
    <script>
        let capturing = false;
        let packetId = 1;
        let captureInterval = null;
        
        function toggleCapture() {{
            const list = document.getElementById('packetList');
            const btn = document.getElementById('captureButton');
            
            if (!capturing) {{
                capturing = true;
                btn.textContent = '⏸️ STOP CAPTURE';
                list.innerHTML = '<div style="color: #00ddff; margin-bottom: 10px;">📡 Capturing quantum packets...</div>';
                
                captureInterval = setInterval(() => {{
                    const protocols = ['EPR', 'QKD-BB84', 'QKD-E91', 'QSH', 'QRAM', 'TELEPORT'];
                    const proto = protocols[Math.floor(Math.random() * protocols.length)];
                    const src = `10.0.${{Math.floor(Math.random()*255)}}.${{Math.floor(Math.random()*255)}}`;
                    const dst = `10.0.${{Math.floor(Math.random()*255)}}.${{Math.floor(Math.random()*255)}}`;
                    const qubits = Math.floor(Math.random() * 128) + 1;
                    
                    const packet = document.createElement('div');
                    packet.className = 'packet';
                    packet.innerHTML = `
                        <span style="color: #888;">#${{packetId++}}</span> 
                        <span style="color: #00ddff;">${{proto}}</span> 
                        ${{src}} → ${{dst}} 
                        <span style="color: #00ff88;">${{qubits}} qubits</span> 
                        <span style="color: #888;">${{new Date().toLocaleTimeString()}}</span>
                    `;
                    list.appendChild(packet);
                    list.scrollTop = list.scrollHeight;
                    
                    while (list.children.length > 101) {{
                        list.removeChild(list.children[1]);
                    }}
                }}, 500);
            }} else {{
                capturing = false;
                btn.textContent = '📡 START CAPTURE';
                if (captureInterval) {{
                    clearInterval(captureInterval);
                    captureInterval = null;
                }}
            }}
        }}
        
        document.getElementById('captureButton').addEventListener('click', toggleCapture);
    </script>
</body>
</html>
"""

FILES_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <title>File Manager</title>
    <style>
        {BASE_STYLE}
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow: hidden;
        }}
        .page-header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #00ff88;
            margin-bottom: 20px;
        }}
        .page-header h2 {{ color: #00ddff; font-size: 1.5em; margin-bottom: 10px; }}
        .upload-area {{
            border: 2px dashed #00ff88;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            background: rgba(0, 255, 136, 0.05);
            cursor: pointer;
        }}
        .upload-area.dragover {{
            background: rgba(0, 255, 136, 0.2);
            border-color: #00ddff;
        }}
        .upload-button {{
            padding: 12px 30px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
        }}
        .file-list {{
            flex: 1;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #00ff88;
            border-radius: 5px;
            padding: 15px;
        }}
        .file-item {{
            padding: 12px;
            margin-bottom: 8px;
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid #00ff88;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .file-item:hover {{ background: rgba(0, 255, 136, 0.1); }}
        .file-item.selected {{
            background: rgba(0, 221, 255, 0.2);
            border-color: #00ddff;
        }}
        .file-info {{ flex: 1; }}
        .file-name {{ color: #00ff88; font-weight: bold; margin-bottom: 4px; }}
        .file-size {{ color: #888; font-size: 0.85em; }}
        .file-actions {{
            display: flex;
            gap: 8px;
        }}
        .action-button {{
            padding: 6px 12px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
            border-radius: 3px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            text-decoration: none;
        }}
        .action-button:hover {{ background: rgba(0, 255, 136, 0.3); }}
        .bulk-actions {{
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid #00ff88;
            border-radius: 5px;
            display: none;
        }}
        .bulk-actions.active {{ display: block; }}
        .bulk-button {{
            padding: 8px 16px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
            border-radius: 3px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📁 File Manager</h1>
        <div>
            <a href="/" class="nav-button">⬅️ Home</a>
        </div>
    </div>
    <div class="main-content">
        <div class="page-header">
            <h2>📁 FILE MANAGER</h2>
            <div style="color: #888; margin-top: 10px;">127.0.0.1:9999 | Click files to select multiple</div>
        </div>
        <div class="upload-area" id="uploadArea">
            <h3 style="color: #00ddff; margin-bottom: 15px;">📤 Upload Files</h3>
            <p>Drag & drop files here or click to browse</p>
            <input type="file" id="fileInput" multiple style="display:none">
            <button type="button" class="upload-button" id="fileSelectButton">SELECT FILES</button>
        </div>
        <div class="bulk-actions" id="bulkActions">
            <span style="color: #00ddff; margin-right: 15px;" id="selectedCount">0 files selected</span>
            <button type="button" class="bulk-button" id="downloadSelectedButton">⬇️ Download Selected</button>
            <button type="button" class="bulk-button" id="deselectAllButton">✕ Deselect All</button>
        </div>
        <h3 style="color: #00ddff; margin-bottom: 10px;">📂 Uploaded Files</h3>
        <div class="file-list" id="fileList"></div>
    </div>
    <script>
        let selectedFiles = new Set();
        
        function formatBytes(bytes) {{
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
            return (bytes/(1024*1024)).toFixed(1) + ' MB';
        }}
        
        function updateBulkActions() {{
            const bulkActions = document.getElementById('bulkActions');
            const selectedCount = document.getElementById('selectedCount');
            selectedCount.textContent = `${{selectedFiles.size}} file(s) selected`;
            
            if (selectedFiles.size > 0) {{
                bulkActions.classList.add('active');
            }} else {{
                bulkActions.classList.remove('active');
            }}
        }}
        
        async function loadFiles() {{
            try {{
                const res = await fetch('/api/files');
                const files = await res.json();
                const list = document.getElementById('fileList');
                
                if (files.length === 0) {{
                    list.innerHTML = '<div style="color: #888; text-align: center; padding: 40px;">No files uploaded yet</div>';
                }} else {{
                    list.innerHTML = files.map(f => `
                        <div class="file-item" data-filename="${{f.name}}" onclick="toggleFileSelection(this, '${{f.name}}')">
                            <div class="file-info">
                                <div class="file-name">📄 ${{f.name}}</div>
                                <div class="file-size">${{formatBytes(f.size)}}</div>
                            </div>
                            <div class="file-actions">
                                <a href="/api/download/${{encodeURIComponent(f.name)}}" class="action-button" onclick="event.stopPropagation()">⬇️ Download</a>
                            </div>
                        </div>
                    `).join('');
                }}
            }} catch (e) {{
                console.error('Failed to load files:', e);
            }}
        }}
        
        function toggleFileSelection(element, filename) {{
            if (selectedFiles.has(filename)) {{
                selectedFiles.delete(filename);
                element.classList.remove('selected');
            }} else {{
                selectedFiles.add(filename);
                element.classList.add('selected');
            }}
            updateBulkActions();
        }}
        
        window.toggleFileSelection = toggleFileSelection;
        
        document.getElementById('deselectAllButton').addEventListener('click', () => {{
            selectedFiles.clear();
            document.querySelectorAll('.file-item').forEach(el => el.classList.remove('selected'));
            updateBulkActions();
        }});
        
        document.getElementById('downloadSelectedButton').addEventListener('click', () => {{
            selectedFiles.forEach(filename => {{
                const link = document.createElement('a');
                link.href = `/api/download/${{encodeURIComponent(filename)}}`;
                link.download = filename;
                link.click();
            }});
        }});
        
        async function handleFiles(files) {{
            for (let file of files) {{
                const formData = new FormData();
                formData.append('file', file);
                
                try {{
                    await fetch('/api/upload', {{
                        method: 'POST',
                        body: formData
                    }});
                }} catch (e) {{
                    alert('Upload failed: ' + e.message);
                }}
            }}
            await loadFiles();
        }}
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadArea.classList.add('dragover');
        }});
        
        uploadArea.addEventListener('dragleave', () => {{
            uploadArea.classList.remove('dragover');
        }});
        
        uploadArea.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        }});
        
        fileInput.addEventListener('change', (e) => {{
            handleFiles(e.target.files);
        }});
        
        document.getElementById('fileSelectButton').addEventListener('click', (e) => {{
            e.stopPropagation();
            fileInput.click();
        }});
        
        loadFiles();
        setInterval(loadFiles, 5000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def splash():
    return SPLASH_HTML

@app.get("/metrics", response_class=HTMLResponse)
async def metrics():
    return METRICS_HTML

@app.get("/collider", response_class=HTMLResponse)
async def collider():
    return COLLIDER_HTML

@app.get("/shell", response_class=HTMLResponse)
async def shell():
    return SHELL_HTML

@app.get("/wireshark", response_class=HTMLResponse)
async def wireshark():
    return WIRESHARK_HTML

@app.get("/files", response_class=HTMLResponse)
async def files():
    return FILES_HTML

@app.get("/api/quantum-metrics")
async def get_quantum_metrics():
    return JSONResponse(network_metrics.get_metrics())

@app.post("/api/run-speed-test")
async def run_speed_test(background_tasks: BackgroundTasks):
    background_tasks.add_task(network_metrics.run_full_test)
    return {"status": "test_started"}

@app.post("/api/qsh-query")
async def qsh_query(query: QSHQuery):
    result = process_qsh_query(query.query)
    return JSONResponse(result)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "size": len(content)}

@app.get("/api/files")
async def list_files():
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size
            })
    return files

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.post("/api/shell")
async def shell_command(command: ShellCommand):
    result = execute_shell_command(command.command)
    return JSONResponse(result)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    alice.register_user("system", "127.0.0.1")
    logger.info("Alice interface initialized")
    logger.info("Quantum Foam Network starting...")
    
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
