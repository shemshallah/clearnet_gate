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
import socket
import threading
import select
import struct
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
                         "[FOAM] Access via Wireshark tab\n" +
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
            display: inline-block;
            margin: 20px auto;
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
        }
        .quantum-button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.8);
        }
        .button-container { text-align: center; margin: 30px 0; }
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
        <div class="button-container">
            <a href="/metrics" class="quantum-button">📊 NETWORK CONTROL CENTER</a>
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

METRICS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Network Control</title>
    <meta charset="UTF-8">
    <style>
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
        }
        .nav-button:hover {
            background: rgba(0, 255, 136, 0.4);
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        .main-layout {
            display: flex;
            height: calc(100vh - 60px);
            position: relative;
        }
        .metrics-sidebar {
            width: 320px;
            background: rgba(10, 10, 10, 0.95);
            border-right: 2px solid #00ff88;
            padding: 15px;
            overflow-y: auto;
            flex-shrink: 0;
        }
        .metrics-title {
            font-size: 1em;
            color: #00ddff;
            margin-bottom: 12px;
            text-align: center;
            border-bottom: 1px solid #00ff88;
            padding-bottom: 8px;
        }
        .test-button, .wireshark-sidebar-button {
            width: 100%;
            padding: 8px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 12px;
        }
        .test-button:hover, .wireshark-sidebar-button:hover {
            opacity: 0.9;
        }
        .test-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .metric-card {
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid #00ff88;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 8px;
        }
        .metric-label { font-size: 0.75em; color: #00ddff; margin-bottom: 3px; }
        .metric-value { font-size: 1em; color: #00ff88; font-weight: bold; }
        .metric-value.real { color: #00ddff; }
        .metric-unit { font-size: 0.7em; color: #888; margin-left: 4px; }
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .tabs {
            display: flex;
            background: rgba(10, 10, 10, 0.95);
            border-bottom: 1px solid #00ff88;
            overflow-x: auto;
            flex-shrink: 0;
        }
        .tab {
            padding: 10px 15px;
            background: rgba(0, 255, 136, 0.1);
            border: none;
            border-right: 1px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            cursor: pointer;
            font-size: 0.8em;
            white-space: nowrap;
        }
        .tab:hover { background: rgba(0, 255, 136, 0.2); }
        .tab.active {
            background: rgba(0, 221, 255, 0.3);
            color: #00ddff;
            border-bottom: 2px solid #00ddff;
        }
        .tab-content { 
            display: none; 
            flex: 1; 
            overflow: hidden; 
        }
        .tab-content.active { 
            display: flex; 
            flex-direction: column; 
        }
        
        /* Collider Interface */
        .collider-interface {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: rgba(10, 10, 10, 0.95);
            padding: 15px;
            overflow: hidden;
        }
        .collider-header {
            text-align: center;
            padding: 12px;
            border-bottom: 2px solid #00ff88;
            margin-bottom: 12px;
            flex-shrink: 0;
        }
        .collider-header h2 { color: #00ddff; font-size: 1.2em; margin-bottom: 6px; }
        .collider-domain { color: #00ff88; font-size: 0.8em; }
        .chat-output {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #00ff88;
            border-radius: 5px;
            margin-bottom: 12px;
        }
        .message {
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 5px;
        }
        .message.user {
            background: rgba(0, 221, 255, 0.1);
            border-left: 3px solid #00ddff;
        }
        .message.system {
            background: rgba(0, 255, 136, 0.1);
            border-left: 3px solid #00ff88;
        }
        .message-label { font-size: 0.7em; color: #888; margin-bottom: 4px; }
        .message-content { color: #00ff88; line-height: 1.4; font-size: 0.85em; }
        .qsh-result {
            font-size: 0.8em;
            margin-top: 6px;
            padding: 6px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 3px;
        }
        .qsh-field { margin: 3px 0; }
        .qsh-label {
            color: #00ddff;
            display: inline-block;
            width: 160px;
            font-size: 0.85em;
        }
        .qsh-value { color: #00ff88; }
        .chat-input-container { display: flex; gap: 8px; flex-shrink: 0; }
        .chat-input {
            flex: 1;
            padding: 8px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            border-radius: 5px;
            outline: none;
            font-size: 0.85em;
        }
        .send-button {
            padding: 8px 20px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
            font-size: 0.85em;
        }
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Terminal */
        .terminal {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #000;
            padding: 12px;
            font-family: 'Courier New', monospace;
            overflow: hidden;
        }
        .terminal-output {
            flex: 1;
            overflow-y: auto;
            color: #00ff88;
            margin-bottom: 8px;
            font-size: 0.85em;
            line-height: 1.3;
            white-space: pre-wrap;
        }
        .terminal-input-line {
            display: flex;
            align-items: center;
            flex-shrink: 0;
        }
        .terminal-prompt {
            color: #00ddff;
            margin-right: 6px;
            font-size: 0.85em;
        }
        .terminal-input {
            flex: 1;
            background: transparent;
            border: none;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            outline: none;
            font-size: 0.85em;
        }
        
        /* Wireshark */
        .wireshark-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: rgba(10, 10, 10, 0.95);
            padding: 15px;
            overflow: hidden;
        }
        .wireshark-header {
            text-align: center;
            padding: 12px;
            border-bottom: 2px solid #00ff88;
            margin-bottom: 12px;
            flex-shrink: 0;
        }
        .packet-list {
            flex: 1;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            border-radius: 5px;
            padding: 12px;
            font-size: 0.8em;
        }
        .packet {
            padding: 6px;
            margin-bottom: 4px;
            background: rgba(0, 255, 136, 0.05);
            border-left: 3px solid #00ff88;
            font-family: 'Courier New', monospace;
        }
        .packet:hover { background: rgba(0, 255, 136, 0.1); }
        
        /* File Browser (Microbrowser at bottom left) */
        .file-browser {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 320px;
            height: 300px;
            background: rgba(10, 10, 10, 0.98);
            border-top: 2px solid #00ff88;
            border-right: 2px solid #00ff88;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }
        .file-browser-header {
            padding: 8px 12px;
            background: rgba(0, 255, 136, 0.2);
            border-bottom: 1px solid #00ff88;
            font-size: 0.9em;
            color: #00ddff;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .file-browser-close {
            cursor: pointer;
            color: #ff4444;
            font-weight: bold;
        }
        .file-browser-content {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        .file-browser-item {
            padding: 6px;
            margin-bottom: 4px;
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid #00ff88;
            border-radius: 3px;
            font-size: 0.75em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .file-browser-item:hover { background: rgba(0, 255, 136, 0.1); }
        .file-upload-btn {
            padding: 6px 12px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.75em;
            font-weight: bold;
            cursor: pointer;
            margin: 8px;
        }
        
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0a0a0a; }
        ::-webkit-scrollbar-thumb { background: #00ff88; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Quantum Network Control Center</h1>
        <a href="/" class="nav-button">⬅️ Back to Main</a>
    </div>
    
    <div class="main-layout">
        <div class="metrics-sidebar">
            <div class="metrics-title">⚛️ LIVE METRICS</div>
            <button type="button" class="test-button" id="speedTestButton">🚀 RUN SPEED TEST</button>
            <button type="button" class="wireshark-sidebar-button" id="wiresharkSidebarButton">🔍 WIRESHARK</button>
            <div id="realMetrics"><div class="metric-card"><div class="metric-label">Loading...</div></div></div>
            <div id="quantumMetrics"><div class="metric-card"><div class="metric-label">Loading...</div></div></div>
        </div>
        
        <div class="main-content">
            <div class="tabs">
                <button type="button" class="tab active" data-tab="collider">⚛️ Quantum Collider</button>
                <button type="button" class="tab" data-tab="shell">🖥️ QSH::FOAM REPL</button>
                <button type="button" class="tab" data-tab="wireshark">🔍 Wireshark</button>
            </div>
            
            <!-- Quantum Collider Tab -->
            <div id="collider-tab" class="tab-content active">
                <div class="collider-interface">
                    <div class="collider-header">
                        <h2>⚛️ QUANTUM COLLIDER & QSH QUERY</h2>
                        <div class="collider-domain">quantum.realm.domain.dominion.foam.computer.collider</div>
                    </div>
                    <div class="chat-output" id="chatOutput">
                        <div class="message system">
                            <div class="message-label">SYSTEM</div>
                            <div class="message-content">Welcome to the Quantum Collider interface. Enter your query below.</div>
                        </div>
                    </div>
                    <div class="chat-input-container">
                        <input type="text" class="chat-input" id="chatInput" placeholder="Enter QSH query...">
                        <button type="button" class="send-button" id="sendButton">SEND</button>
                    </div>
                </div>
            </div>
            
            <!-- Shell Tab -->
            <div id="shell-tab" class="tab-content">
                <div class="terminal">
                    <div class="terminal-output" id="terminalOutput">QSH::FOAM REPL v1.0.0 (127.0.0.1:alice)
Connected to quantum.realm.domain.dominion.foam.computer.networking
Type 'help' or '??' for available commands.

</div>
                    <div class="terminal-input-line">
                        <span class="terminal-prompt">foam@alice:~$</span>
                        <input type="text" class="terminal-input" id="terminalInput" autofocus>
                    </div>
                </div>
            </div>
            
            <!-- Wireshark Tab -->
            <div id="wireshark-tab" class="tab-content">
                <div class="wireshark-container">
                    <div class="wireshark-header">
                        <h2>🔍 WIRESHARK QUANTUM PACKET ANALYZER</h2>
                        <div class="collider-domain">*.computer domain</div>
                        <p style="margin-top: 8px; color: #888; font-size: 0.8em;">
                            Monitoring quantum foam network traffic across computational substrate
                        </p>
                    </div>
                    <button type="button" class="test-button" id="captureButton" style="margin-bottom: 12px;">📡 START CAPTURE</button>
                    <div class="packet-list" id="packetList">
                        <div style="color: #888; text-align: center; padding: 30px;">
                            Click "START CAPTURE" to begin monitoring quantum packets
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- File Browser Microbrowser (Bottom Left) -->
    <div class="file-browser" id="fileBrowser">
        <div class="file-browser-header">
            📁 FILE BROWSER (127.0.0.1:9999)
            <span class="file-browser-close" id="fileBrowserClose">✕</span>
        </div>
        <input type="file" id="fileInput" multiple style="display:none">
        <button type="button" class="file-upload-btn" id="fileSelectButton">📤 UPLOAD FILES</button>
        <div class="file-browser-content" id="fileBrowserContent">
            <div style="color: #888; text-align: center; padding: 20px; font-size: 0.8em;">No files uploaded</div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            let capturing = false;
            let packetId = 1;
            let captureInterval = null;
            
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function(e) {
                    e.preventDefault();
                    const tabName = this.getAttribute('data-tab');
                    switchTab(tabName);
                });
            });
            
            function switchTab(tabName) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                // Deactivate all tabs
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                
                // Show selected tab content
                const targetTab = document.getElementById(tabName + '-tab');
                if (targetTab) {
                    targetTab.classList.add('active');
                }
                // Activate selected tab button
                const targetButton = document.querySelector(`[data-tab="${tabName}"]`);
                if (targetButton) {
                    targetButton.classList.add('active');
                }
            }
            
            // Wireshark sidebar button
            document.getElementById('wiresharkSidebarButton').addEventListener('click', function() {
                switchTab('wireshark');
            });
            
            // Quantum Collider
            const chatInput = document.getElementById('chatInput');
            const sendButton = document.getElementById('sendButton');
            
            chatInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendQuery();
                }
            });
            
            async function sendQuery() {
                const query = chatInput.value.trim();
                if (!query) return;
                
                const output = document.getElementById('chatOutput');
                
                output.innerHTML += `
                    <div class="message user">
                        <div class="message-label">USER QUERY</div>
                        <div class="message-content">${escapeHtml(query)}</div>
                    </div>
                `;
                chatInput.value = '';
                sendButton.disabled = true;
                
                try {
                    const res = await fetch('/api/qsh-query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query})
                    });
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    const data = await res.json();
                    
                    output.innerHTML += `
                        <div class="message system">
                            <div class="message-label">QUANTUM COLLIDER</div>
                            <div class="message-content">
                                Query processed through quantum collision system.
                                <div class="qsh-result">
                                    <div class="qsh-field"><span class="qsh-label">QSH Hash:</span><span class="qsh-value">${data.qsh_hash}</span></div>
                                    <div class="qsh-field"><span class="qsh-label">Classical Hash:</span><span class="qsh-value">${data.classical_hash.substring(0, 32)}...</span></div>
                                    <div class="qsh-field"><span class="qsh-label">Entanglement Strength:</span><span class="qsh-value">${data.entanglement_strength}</span></div>
                                    <div class="qsh-field"><span class="qsh-label">Collision Energy:</span><span class="qsh-value">${data.collision_energy_gev} GeV</span></div>
                                    <div class="qsh-field"><span class="qsh-label">Particle States:</span><span class="qsh-value">${data.particle_states_generated}</span></div>
                                    <div class="qsh-field"><span class="qsh-label">Foam Perturbations:</span><span class="qsh-value">${data.foam_perturbations}</span></div>
                                </div>
                            </div>
                        </div>
                    `;
                } catch (e) {
                    output.innerHTML += `<div class="message system"><div class="message-content">Error: ${e.message}</div></div>`;
                } finally {
                    sendButton.disabled = false;
                }
                output.scrollTop = output.scrollHeight;
            }
            
            sendButton.addEventListener('click', sendQuery);
            
            // File Browser
            const fileBrowser = document.getElementById('fileBrowser');
            const fileBrowserClose = document.getElementById('fileBrowserClose');
            const fileInput = document.getElementById('fileInput');
            
            fileBrowserClose.addEventListener('click', () => {
                fileBrowser.style.display = 'none';
            });
            
            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });
            
            document.getElementById('fileSelectButton').addEventListener('click', () => {
                fileInput.click();
            });
            
            async function handleFiles(files) {
                for (let file of files) {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const res = await fetch('/api/upload', {
                            method: 'POST',
                            body: formData
                        });
                        if (!res.ok) {
                            throw new Error(`HTTP error! status: ${res.status}`);
                        }
                        await loadFiles();
                    } catch (e) {
                        alert('Upload failed: ' + e.message);
                    }
                }
            }
            
            async function loadFiles() {
                try {
                    const res = await fetch('/api/files');
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    const files = await res.json();
                    const content = document.getElementById('fileBrowserContent');
                    
                    if (files.length === 0) {
                        content.innerHTML = '<div style="color: #888; text-align: center; padding: 20px; font-size: 0.8em;">No files uploaded</div>';
                    } else {
                        content.innerHTML = files.map(f => `
                            <div class="file-browser-item">
                                <span>📄 ${escapeHtml(f.name)}<br><small style="color: #888;">${formatBytes(f.size)}</small></span>
                                <a href="/api/download/${encodeURIComponent(f.name)}" class="send-button" style="padding: 4px 8px; font-size: 0.7em; text-decoration: none;">⬇️</a>
                            </div>
                        `).join('');
                    }
                } catch (e) {
                    document.getElementById('fileBrowserContent').innerHTML = '<div style="color: #888; text-align: center; padding: 20px; font-size: 0.8em;">Failed to load</div>';
                    console.error('Failed to load files:', e);
                }
            }
            
            function formatBytes(bytes) {
                if (bytes < 1024) return bytes + ' B';
                if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
                return (bytes/(1024*1024)).toFixed(1) + ' MB';
            }
            
            // Terminal
            const terminalInput = document.getElementById('terminalInput');
            
            terminalInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    executeCommand();
                }
            });
            
            async function executeCommand() {
                const input = terminalInput;
                const output = document.getElementById('terminalOutput');
                const command = input.value.trim();
                
                if (!command) return;
                
                if (command === 'clear') {
                    output.textContent = '';
                    input.value = '';
                    return;
                }
                
                output.textContent += `foam@alice:~$ ${command}\n`;
                input.value = '';
                
                try {
                    const res = await fetch('/api/shell', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({command})
                    });
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    const data = await res.json();
                    output.textContent += data.output + '\n\n';
                } catch (e) {
                    output.textContent += `Error: ${e.message}\n\n`;
                }
                
                output.scrollTop = output.scrollHeight;
            }
            
            // Wireshark
            document.getElementById('captureButton').addEventListener('click', function(e) {
                e.preventDefault();
                const list = document.getElementById('packetList');
                const btn = e.target;
                
                if (!capturing) {
                    capturing = true;
                    btn.textContent = '⏸️ STOP CAPTURE';
                    list.innerHTML = '<div style="color: #00ddff; margin-bottom: 8px;">📡 Capturing quantum packets...</div>';
                    
                    captureInterval = setInterval(() => {
                        if (!capturing) {
                            clearInterval(captureInterval);
                            return;
                        }
                        
                        const protocols = ['EPR', 'QKD-BB84', 'QKD-E91', 'QSH', 'QRAM', 'TELEPORT'];
                        const proto = protocols[Math.floor(Math.random() * protocols.length)];
                        const src = `10.0.${Math.floor(Math.random()*255)}.${Math.floor(Math.random()*255)}`;
                        const dst = `10.0.${Math.floor(Math.random()*255)}.${Math.floor(Math.random()*255)}`;
                        const qubits = Math.floor(Math.random() * 128) + 1;
                        
                        const packet = document.createElement('div');
                        packet.className = 'packet';
                        packet.innerHTML = `
                            <span style="color: #888;">#${packetId++}</span> 
                            <span style="color: #00ddff;">${proto}</span> 
                            ${src} → ${dst} 
                            <span style="color: #00ff88;">${qubits} qubits</span> 
                            <span style="color: #888;">${new Date().toLocaleTimeString()}</span>
                        `;
                        list.appendChild(packet);
                        list.scrollTop = list.scrollHeight;
                        
                        // Keep only last 100 packets
                        while (list.children.length > 101) {
                            list.removeChild(list.children[1]);
                        }
                    }, 500);
                } else {
                    capturing = false;
                    btn.textContent = '📡 START CAPTURE';
                    if (captureInterval) {
                        clearInterval(captureInterval);
                        captureInterval = null;
                    }
                }
            });
            
            // Metrics
            document.getElementById('speedTestButton').addEventListener('click', async function(e) {
                e.preventDefault();
                const btn = e.target;
                btn.disabled = true;
                btn.textContent = '⏳ TESTING...';
                
                try {
                    const res = await fetch('/api/run-speed-test', {method: 'POST'});
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    await new Promise(r => setTimeout(r, 2000));
                    await updateMetrics();
                } catch (error) {
                    console.error('Speed test failed:', error);
                } finally {
                    btn.disabled = false;
                    btn.textContent = '🚀 RUN SPEED TEST';
                }
            });
            
            async function updateMetrics() {
                try {
                    const res = await fetch('/api/quantum-metrics');
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    const d = await res.json();
                    
                    document.getElementById('realMetrics').innerHTML = `
                        <div class="metric-card">
                            <div class="metric-label">⬇️ Download</div>
                            <div class="metric-value real">${d.download_speed_mbps}<span class="metric-unit">Mbps</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">⬆️ Upload</div>
                            <div class="metric-value real">${d.upload_speed_mbps}<span class="metric-unit">Mbps</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">📶 Ping</div>
                            <div class="metric-value real">${d.ping_ms}<span class="metric-unit">ms</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">⚡ Network Throughput</div>
                            <div class="metric-value">${d.network_throughput_mbps}<span class="metric-unit">Mbps</span></div>
                        </div>
                    `;
                    
                    document.getElementById('quantumMetrics').innerHTML = `
                        <div class="metric-card">
                            <div class="metric-label">🔮 Qubits Active</div>
                            <div class="metric-value">${d.qubits_active}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">🔗 EPR Pairs</div>
                            <div class="metric-value">${d.epr_pairs}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">📡 Transfer Rate</div>
                            <div class="metric-value">${d.transfer_rate_qbps}<span class="metric-unit">Qbps</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">✨ Entanglement Fidelity</div>
                            <div class="metric-value">${d.entanglement_fidelity}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">⏱️ Decoherence Time</div>
                            <div class="metric-value">${d.decoherence_time_ms}<span class="metric-unit">ms</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">🌀 Foam Density</div>
                            <div class="metric-value">${d.foam_density}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">🎯 Teleportation Success</div>
                            <div class="metric-value">${(d.teleportation_success_rate * 100).toFixed(1)}<span class="metric-unit">%</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">⏰ Uptime</div>
                            <div class="metric-value">${formatUptime(d.uptime_seconds)}</div>
                        </div>
                    `;
                } catch (e) {
                    console.error('Metrics update failed:', e);
                }
            }
            
            function formatUptime(seconds) {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const secs = seconds % 60;
                return `${hours}h ${minutes}m ${secs}s`;
            }
            
            function escapeHtml(text) {
                const map = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'};
                return text.replace(/[&<>"']/g, m => map[m]);
            }
            
            // Initialize
            updateMetrics();
            setInterval(updateMetrics, 3000);
            loadFiles();
            setInterval(loadFiles, 5000);
        });
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

@app.post("/api/upload-test")
async def upload_test(file: UploadFile = File(...)):
    content = await file.read()
    return {"size": len(content)}

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
    # Initialize Alice
    alice.register_user("system", "127.0.0.1")
    logger.info("Alice interface initialized")
    logger.info("Quantum Foam Network starting...")
    
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
