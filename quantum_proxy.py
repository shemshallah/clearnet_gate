from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
import hashlib
import time
import random
import asyncio
from datetime import datetime
from typing import List
import uvicorn

app = FastAPI(title="Quantum File Network (QFN)")

# Directory for uploads
UPLOAD_DIR = "/opt/render/project/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Config file
CONFIG_FILE = "config.json"

# Mount static files if needed (not used here)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (not used, since using raw HTML)
# templates = Jinja2Templates(directory="templates")

# Load/Save config
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "holo_storage_ip": "138.0.0.1",
        "holo_dns_enabled": True,
        "upload_directory": UPLOAD_DIR,
        "default_interface": "wlan0",
        "default_rf_mode": "quantum",
        "epr_rate": 2500,
        "entanglement_threshold": 0.975
    }

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

config = load_config()

# 1. Root Page (/)
@app.get("/", response_class=HTMLResponse)
async def root_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum File Network | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            text-align: center;
            max-width: 800px;
        }
        h1 {
            font-size: 3.5em;
            margin-bottom: 20px;
            text-shadow: 0 0 20px #00ff88;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 20px #00ff88, 0 0 30px #00ff88, 0 0 40px #00ff88; }
            to { text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88, 0 0 30px #00ff88; }
        }
        .subtitle {
            font-size: 1.2em;
            margin-bottom: 40px;
            opacity: 0.8;
        }
        .nav-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            width: 100%;
        }
        .nav-item {
            padding: 20px;
            background: rgba(0, 255, 136, 0.1);
            border: 2px solid #00ff88;
            border-radius: 10px;
            text-decoration: none;
            color: #00ff88;
            font-size: 1.1em;
            font-weight: bold;
            transition: all 0.3s;
            text-align: center;
        }
        .nav-item:hover {
            background: rgba(0, 255, 136, 0.2);
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 255, 136, 0.3);
        }
        .status-bar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background: rgba(0, 10, 10, 0.9);
            border-bottom: 1px solid #00ff88;
            color: #00ff88;
            font-size: 0.9em;
            display: flex;
            justify-content: space-between;
        }
        .quantum-status {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="status-bar">
        <span>🔗 QFN v2.0 | Holo Storage: 138.0.0.1 | DNS Routing: Enabled</span>
        <span class="quantum-status">⚛️ Entanglement Active | EPR Rate: 2500/s</span>
    </div>
    
    <div class="container">
        <h1>🌌 Quantum File Network</h1>
        <p class="subtitle">Secure, entangled file transfer across quantum foam</p>
        
        <div class="nav-grid">
            <a href="/metrics" class="nav-item">📊 Metrics Dashboard</a>
            <a href="/shell" class="nav-item">💻 QSH Terminal</a>
            <a href="/wireshark" class="nav-item">🔍 Wireshark Analyzer</a>
            <a href="/files" class="nav-item">📁 File Storage</a>
            <a href="/collider" class="nav-item">⚛️ Quantum Collider</a>
            <a href="/config" class="nav-item">⚙️ Configuration</a>
        </div>
    </div>
</body>
</html>
    """)

# 2. Metrics Page (/metrics)
@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Metrics | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff88;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 5px;
            color: #00ff88;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .metric-card {
            padding: 20px;
            background: rgba(0, 136, 255, 0.1);
            border: 1px solid #0088ff;
            border-radius: 8px;
        }
        .metric-title {
            font-size: 1.3em;
            color: #00ddff;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 2em;
            color: #00ff88;
            font-weight: bold;
        }
        .metric-label {
            color: #888;
            margin-top: 5px;
        }
        .speed-test-btn {
            padding: 10px 20px;
            background: #00ff88;
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin-top: 10px;
        }
        .speed-test-btn:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .speed-result {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 5px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>📊 REAL-TIME METRICS</h1>
        
        <div class="metrics-grid" id="metricsGrid">
            <!-- Metrics populated by JS -->
        </div>
        
        <div style="text-align: center; margin-top: 30px;">
            <button class="speed-test-btn" onclick="runSpeedTest()">🚀 Run Network Speed Test</button>
            <div class="speed-result" id="speedResult"></div>
        </div>
    </div>
    
    <script>
        async function updateMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const metrics = await response.json();
                
                const grid = document.getElementById('metricsGrid');
                grid.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-title">🌐 Network Throughput</div>
                        <div class="metric-value">${metrics.network_throughput} Mbps</div>
                        <div class="metric-label">Current upload/download speed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">⚛️ EPR Pair Rate</div>
                        <div class="metric-value">${metrics.epr_rate}/s</div>
                        <div class="metric-label">Entangled pairs generated per second</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">🔒 Entanglement Fidelity</div>
                        <div class="metric-value">${metrics.entanglement_fidelity}%</div>
                        <div class="metric-label">Quantum state preservation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">📡 Holo Storage Latency</div>
                        <div class="metric-value">${metrics.holo_latency} ms</div>
                        <div class="metric-label">Round-trip to storage node</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">🛡️ QKD Key Rate</div>
                        <div class="metric-value">${metrics.qkd_key_rate} bps</div>
                        <div class="metric-label">Secure key generation rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">🌊 Foam Density</div>
                        <div class="metric-value">${metrics.foam_density}</div>
                        <div class="metric-label">Quantum foam perturbations</div>
                    </div>
                `;
            } catch (error) {
                console.error('Error updating metrics:', error);
            }
        }
        
        async function runSpeedTest() {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Testing...';
            const resultDiv = document.getElementById('speedResult');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = 'Running speed test...';
            
            try {
                const start = performance.now();
                const response = await fetch('/api/speed-test');
                const end = performance.now();
                const data = await response.json();
                
                resultDiv.innerHTML = `
                    <strong>Speed Test Results:</strong><br>
                    Download: ${data.download} Mbps<br>
                    Upload: ${data.upload} Mbps<br>
                    Latency: ${(end - start).toFixed(2)} ms<br>
                    Jitter: ${data.jitter} ms
                `;
            } catch (error) {
                resultDiv.innerHTML = 'Speed test failed: ' + error.message;
            } finally {
                btn.disabled = false;
                btn.textContent = '🚀 Run Network Speed Test';
            }
        }
        
        // Update every 5 seconds
        setInterval(updateMetrics, 5000);
        updateMetrics(); // Initial load
    </script>
</body>
</html>
    """)

# API for metrics
@app.get("/api/metrics")
async def get_metrics():
    return {
        "network_throughput": round(random.uniform(50, 1000), 2),
        "epr_rate": config["epr_rate"],
        "entanglement_fidelity": round(random.uniform(95, 99.9), 2),
        "holo_latency": round(random.uniform(10, 50), 2),
        "qkd_key_rate": round(random.uniform(1000, 10000), 0),
        "foam_density": round(random.uniform(1.2, 2.5), 3)
    }

# API for speed test
@app.get("/api/speed-test")
async def speed_test():
    await asyncio.sleep(1)  # Simulate test time
    return {
        "download": round(random.uniform(100, 1000), 2),
        "upload": round(random.uniform(50, 500), 2),
        "jitter": round(random.uniform(1, 10), 2)
    }

# 3. Shell Page (/shell)
@app.get("/shell", response_class=HTMLResponse)
async def shell_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>QSH Terminal | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            height: 100vh;
            overflow: hidden;
        }
        .terminal {
            height: 100vh;
            padding: 20px;
            overflow-y: auto;
            background: #000;
            border: 2px solid #00ff88;
        }
        .prompt {
            color: #00ff88;
            display: flex;
            align-items: center;
        }
        .prompt-symbol {
            margin-right: 5px;
        }
        .input-line {
            display: flex;
            width: 100%;
        }
        .input {
            flex: 1;
            background: transparent;
            border: none;
            color: #00ff88;
            font-family: inherit;
            font-size: inherit;
            outline: none;
        }
        .output {
            margin-top: 10px;
            white-space: pre-wrap;
        }
        .output.ai {
            color: #ff00ff;
        }
        .history {
            margin-bottom: 10px;
        }
        .back-btn {
            position: absolute;
            top: 20px;
            left: 20px;
            padding: 10px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
            text-decoration: none;
            border-radius: 5px;
            z-index: 10;
        }
    </style>
</head>
<body>
    <a href="/" class="back-btn">← Back to Home</a>
    <div class="terminal" id="terminal">
        <div class="history" id="history"></div>
        <div class="prompt">
            <span class="prompt-symbol">QFN:~# </span>
            <div class="input-line">
                <input type="text" class="input" id="commandInput" autofocus placeholder="Enter QSH command..." onkeypress="handleKeyPress(event)">
            </div>
        </div>
    </div>
    
    <script>
        let commandHistory = [];
        let historyIndex = -1;
        
        const terminal = document.getElementById('terminal');
        const historyDiv = document.getElementById('history');
        const input = document.getElementById('commandInput');
        
        function addOutput(text, isAI = false) {
            const outputDiv = document.createElement('div');
            outputDiv.className = `output ${isAI ? 'ai' : ''}`;
            outputDiv.textContent = text;
            historyDiv.appendChild(outputDiv);
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        async function executeCommand(cmd) {
            if (!cmd.trim()) return;
            
            commandHistory.unshift(cmd);
            if (commandHistory.length > 50) commandHistory = commandHistory.slice(0, 50);
            historyIndex = -1;
            
            addOutput(`QFN:~# ${cmd}`);
            
            try {
                const response = await fetch('/api/shell', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: cmd })
                });
                const result = await response.json();
                addOutput(result.output);
                
                // Alice AI response for certain commands
                if (result.ai_response) {
                    setTimeout(() => {
                        addOutput(`Alice: ${result.ai_response}`, true);
                    }, 500);
                }
            } catch (error) {
                addOutput(`Error: ${error.message}`);
            }
            
            input.value = '';
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                const cmd = input.value;
                input.value = '';
                executeCommand(cmd);
            } else if (event.key === 'ArrowUp') {
                if (historyIndex < commandHistory.length - 1) {
                    historyIndex++;
                    input.value = commandHistory[historyIndex];
                }
            } else if (event.key === 'ArrowDown') {
                if (historyIndex > -1) {
                    historyIndex--;
                    input.value = historyIndex === -1 ? '' : commandHistory[historyIndex];
                }
            }
        }
        
        // Initial welcome
        addOutput('Welcome to Quantum Shell (QSH) v1.0');
        addOutput('Type "help" for commands. Alice AI is online for quantum queries.');
    </script>
</body>
</html>
    """)

# API for shell
@app.post("/api/shell")
async def shell_command(request: Request):
    data = await request.json()
    cmd = data.get("command", "").lower()
    
    outputs = {
        "help": "Available commands: metrics, status, epr-gen, qkd-init, holo-upload, alice <query>, clear",
        "metrics": "Network: 850 Mbps | EPR: 2500/s | Fidelity: 98.5%",
        "status": "System: ONLINE | Holo: ACTIVE | Quantum Link: SECURE",
        "epr-gen": "Generating 100 EPR pairs... Success! Fidelity: 97.2%",
        "qkd-init": "Initializing QKD... BB84 protocol engaged. Key rate: 5000 bps",
        "holo-upload": "Uploading to holo storage 138.0.0.1... Complete.",
        "clear": "Clearing terminal... (simulated)"
    }
    
    output = outputs.get(cmd, f"Unknown command: {cmd}. Type 'help' for list.")
    ai_response = None
    if cmd.startswith("alice "):
        query = cmd[6:]
        ai_response = f"Analyzing '{query}'... Quantum insight: Entanglement suggests parallel outcomes in your query."
    
    return {"output": output, "ai_response": ai_response}

# 4. Wireshark Page (/wireshark)
@app.get("/wireshark", response_class=HTMLResponse)
async def wireshark_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Wireshark Analyzer | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff88;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 5px;
            color: #00ff88;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .controls {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        select, button {
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            color: #00ff88;
            border-radius: 5px;
            font-family: inherit;
        }
        button {
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover:not(:disabled) {
            background: rgba(0, 255, 136, 0.2);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .protocol-filters {
            display: flex;
            gap: 10px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .protocol-filters label {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .packets-container {
            height: 600px;
            overflow-y: auto;
            border: 1px solid #333;
            background: #000;
            padding: 10px;
            border-radius: 5px;
        }
        .packet {
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(0, 136, 255, 0.05);
            border-left: 3px solid #0088ff;
            border-radius: 3px;
        }
        .packet-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .packet-protocol {
            color: #ff00ff;
            font-weight: bold;
        }
        .packet-info div {
            margin: 2px 0;
            font-size: 0.9em;
        }
        .rf-info {
            margin-top: 10px;
            font-size: 0.8em;
            color: #888;
        }
        .rf-metric {
            margin-right: 10px;
        }
        .wpa-section {
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 0, 255, 0.1);
            border: 1px solid #ff00ff;
            border-radius: 8px;
        }
        .crack-btn {
            padding: 15px 30px;
            background: #ff00ff;
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
        }
        .wpa-crack-result {
            margin-top: 15px;
            padding: 15px;
            background: rgba(255, 0, 255, 0.2);
            border-radius: 5px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>🔍 WIRESHARK ANALYZER</h1>
        
        <div class="controls">
            <select id="interfaceSelect">
                <option value="wlan0">wlan0 (Wireless)</option>
                <option value="eth0">eth0 (Ethernet)</option>
                <option value="qeth0">qeth0 (Quantum)</option>
            </select>
            <select id="modeSelect">
                <option value="quantum">Quantum EPR</option>
                <option value="4g_lte">4G LTE</option>
                <option value="5g_nr">5G NR</option>
            </select>
            <div class="protocol-filters">
                <label><input type="checkbox" id="tcp"> TCP</label>
                <label><input type="checkbox" id="http"> HTTP</label>
                <label><input type="checkbox" id="dns"> DNS</label>
                <label><input type="checkbox" id="epr"> EPR</label>
                <label><input type="checkbox" id="qkd"> QKD</label>
                <label><input type="checkbox" id="wpa1"> WPA1</label>
                <label><input type="checkbox" id="wpa2"> WPA2</label>
                <label><input type="checkbox" id="wpa3"> WPA3</label>
            </div>
            <button id="startBtn" onclick="startCapture()">▶ Start Capture</button>
            <button id="stopBtn" onclick="stopCapture()" disabled>⏹ Stop</button>
        </div>
        
        <div class="packets-container" id="packetsContainer">
            <div style="text-align: center; color: #666; padding: 20px;">No packets captured yet</div>
        </div>
        
        <div class="wpa-section">
            <h3 style="color: #ff00ff; margin-bottom: 15px;">🔓 WPA Cracking Tool</h3>
            <button class="crack-btn" onclick="crackWPA()">Crack WPA on Port 1337</button>
            <div id="crackResult"></div>
        </div>
    </div>
    
    <script>
        let capturing = false;
        let captureInterval;
        let packetCount = 0;
        
        function getProtocols() {
            const protocols = [];
            if (document.getElementById('tcp').checked) protocols.push('TCP');
            if (document.getElementById('http').checked) protocols.push('HTTP');
            if (document.getElementById('dns').checked) protocols.push('DNS');
            if (document.getElementById('epr').checked) protocols.push('EPR');
            if (document.getElementById('qkd').checked) protocols.push('QKD');
            if (document.getElementById('wpa1').checked) protocols.push('WPA1');
            if (document.getElementById('wpa2').checked) protocols.push('WPA2');
            if (document.getElementById('wpa3').checked) protocols.push('WPA3');
            return protocols.length > 0 ? protocols : ['TCP', 'HTTP', 'EPR'];
        }
        
        function generateRandomIP() {
            return `${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`;
        }
        
        function generateRandomPort() {
            return Math.floor(Math.random() * 65535);
        }
        
        async function capturePacket() {
            const interface = document.getElementById('interfaceSelect').value;
            const mode = document.getElementById('modeSelect').value;
            const protocols = getProtocols();
            const protocol = protocols[Math.floor(Math.random() * protocols.length)];
            
            packetCount++;
            
            // Get RF metrics
            const response = await fetch(`/api/rf-metrics?mode=${mode}&interface=${interface}`);
            const rfMetrics = await response.json();
            
            const packet = {
                id: packetCount,
                time: new Date().toLocaleTimeString(),
                protocol: protocol,
                src: generateRandomIP(),
                dst: generateRandomIP(),
                srcPort: generateRandomPort(),
                dstPort: generateRandomPort(),
                length: Math.floor(Math.random() * 1500) + 64,
                info: getPacketInfo(protocol),
                rfMetrics: rfMetrics
            };
            
            addPacket(packet);
        }
        
        function getPacketInfo(protocol) {
            const infos = {
                'TCP': ['SYN', 'ACK', 'SYN-ACK', 'FIN', 'PSH-ACK', 'RST'],
                'HTTP': ['GET /', 'POST /api', '200 OK', '404 Not Found', '301 Redirect'],
                'DNS': ['Query A', 'Response A', 'Query AAAA', 'Query PTR'],
                'EPR': ['Entanglement Request', 'EPR Pair Created', 'Bell State |Φ+⟩', 'Quantum Teleport'],
                'QKD': ['BB84 Key Exchange', 'E91 Protocol', 'Key Reconciliation', 'Privacy Amplification'],
                'WPA1': ['EAPOL Start', '4-Way Handshake [1/4]', '4-Way Handshake [2/4]'],
                'WPA2': ['EAPOL Start', '4-Way Handshake [1/4]', 'PTK Derivation'],
                'WPA3': ['SAE Commit', 'SAE Confirm', 'PMK Establishment']
            };
            
            const options = infos[protocol] || ['Data Transfer'];
            return options[Math.floor(Math.random() * options.length)];
        }
        
        function formatRFMetrics(rfMetrics) {
            let html = '<div class="rf-info"><strong>RF Metrics:</strong> ';
            
            if (rfMetrics.mode === '4g_lte') {
                html += `<span class="rf-metric">Freq: ${rfMetrics.frequency_mhz} MHz</span>`;
                html += `<span class="rf-metric">BW: ${rfMetrics.bandwidth_mhz} MHz</span>`;
                html += `<span class="rf-metric">Mod: ${rfMetrics.modulation}</span>`;
                html += `<span class="rf-metric">RSSI: ${rfMetrics.rssi_dbm} dBm</span>`;
                html += `<span class="rf-metric">RSRP: ${rfMetrics.rsrp_dbm} dBm</span>`;
                html += `<span class="rf-metric">SINR: ${rfMetrics.sinr_db} dB</span>`;
                html += `<span class="rf-metric">CQI: ${rfMetrics.cqi}</span>`;
                html += `<span class="rf-metric">MIMO: ${rfMetrics.mimo_layers}x</span>`;
                html += `<span class="rf-metric">Cell: ${rfMetrics.cell_id}</span>`;
                html += `<span class="rf-metric">PCI: ${rfMetrics.pci}</span>`;
            } else if (rfMetrics.mode === '5g_nr') {
                html += `<span class="rf-metric">Freq: ${rfMetrics.frequency_mhz} MHz</span>`;
                html += `<span class="rf-metric">BW: ${rfMetrics.bandwidth_mhz} MHz</span>`;
                html += `<span class="rf-metric">Mod: ${rfMetrics.modulation}</span>`;
                html += `<span class="rf-metric">RSSI: ${rfMetrics.rssi_dbm} dBm</span>`;
                html += `<span class="rf-metric">SINR: ${rfMetrics.sinr_db} dB</span>`;
                html += `<span class="rf-metric">MIMO: ${rfMetrics.mimo_layers}x</span>`;
                html += `<span class="rf-metric">Beam: ${rfMetrics.beam_index}</span>`;
                html += `<span class="rf-metric">SCS: ${rfMetrics.scs_khz} kHz</span>`;
                html += `<span class="rf-metric">Band: ${rfMetrics.nr_band}</span>`;
            } else {
                html += `<span class="rf-metric">Freq: ${rfMetrics.frequency_ghz} GHz</span>`;
                html += `<span class="rf-metric">Entanglement: ${rfMetrics.entanglement_strength}</span>`;
                html += `<span class="rf-metric">Fidelity: ${rfMetrics.fidelity}</span>`;
                html += `<span class="rf-metric">Bell: ${rfMetrics.bell_violation}</span>`;
                html += `<span class="rf-metric">EPR Pairs: ${rfMetrics.epr_pairs_active}</span>`;
                html += `<span class="rf-metric">Foam: ${rfMetrics.foam_density}</span>`;
            }
            
            html += '</div>';
            return html;
        }
        
        function addPacket(packet) {
            const container = document.getElementById('packetsContainer');
            
            if (packetCount === 1) {
                container.innerHTML = '';
            }
            
            const packetDiv = document.createElement('div');
            packetDiv.className = 'packet';
            packetDiv.innerHTML = `
                <div class="packet-header">
                    <span class="packet-time">#${packet.id} - ${packet.time}</span>
                    <span class="packet-protocol">${packet.protocol}</span>
                </div>
                <div class="packet-info">
                    <div><strong>Source:</strong> ${packet.src}:${packet.srcPort}</div>
                    <div><strong>Destination:</strong> ${packet.dst}:${packet.dstPort}</div>
                    <div><strong>Length:</strong> ${packet.length} bytes</div>
                    <div><strong>Info:</strong> ${packet.info}</div>
                </div>
                ${formatRFMetrics(packet.rfMetrics)}
            `;
            
            container.insertBefore(packetDiv, container.firstChild);
            
            // Keep only last 50 packets
            while (container.children.length > 50) {
                container.removeChild(container.lastChild);
            }
        }
        
        function startCapture() {
            if (capturing) return;
            
            capturing = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('startBtn').classList.add('capturing');
            
            captureInterval = setInterval(capturePacket, Math.random() * 500 + 200);
        }
        
        function stopCapture() {
            if (!capturing) return;
            
            capturing = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('startBtn').classList.remove('capturing');
            
            clearInterval(captureInterval);
        }
        
        async function crackWPA() {
            const resultDiv = document.getElementById('crackResult');
            resultDiv.innerHTML = '<div class="wpa-crack-result">🔓 Initiating WPA crack on port 1337...</div>';
            
            setTimeout(() => {
                const password = Math.random() > 0.5 ? 'Quantum2024!' : 'P@ssw0rd123';
                resultDiv.innerHTML = `
                    <div class="wpa-crack-result">
                        ✅ <strong>WPA CRACKED!</strong><br>
                        <br>
                        Password: <span style="color: #ff00ff; font-size: 1.5em;">${password}</span><br>
                        Port: 1337<br>
                        Method: Quantum-Accelerated Dictionary Attack<br>
                        Time: ${(Math.random() * 5 + 1).toFixed(2)}s
                    </div>
                `;
            }, 2000);
        }
    </script>
</body>
</html>
    """)

# API for RF metrics
@app.get("/api/rf-metrics")
async def rf_metrics(mode: str, interface: str):
    if mode == "4g_lte":
        return {
            "mode": "4g_lte",
            "frequency_mhz": random.randint(700, 2600),
            "bandwidth_mhz": random.choice([5, 10, 20, 40]),
            "modulation": random.choice(["QPSK", "16QAM", "64QAM"]),
            "rssi_dbm": round(random.uniform(-100, -50), 1),
            "rsrp_dbm": round(random.uniform(-110, -70), 1),
            "sinr_db": round(random.uniform(0, 30), 1),
            "cqi": random.randint(1, 15),
            "mimo_layers": random.choice([2, 4]),
            "cell_id": random.randint(1, 1000),
            "pci": random.randint(0, 503)
        }
    elif mode == "5g_nr":
        return {
            "mode": "5g_nr",
            "frequency_mhz": random.randint(2400, 40000),
            "bandwidth_mhz": random.choice([20, 40, 100, 400]),
            "modulation": random.choice(["QPSK", "16QAM", "64QAM", "256QAM"]),
            "rssi_dbm": round(random.uniform(-100, -50), 1),
            "sinr_db": round(random.uniform(5, 35), 1),
            "mimo_layers": random.choice([4, 8, 16]),
            "beam_index": random.randint(0, 63),
            "scs_khz": random.choice([15, 30, 60, 120]),
            "nr_band": random.choice(["n78", "n41", "n1"])
        }
    else:
        return {
            "mode": "quantum",
            "frequency_ghz": round(random.uniform(0.1, 10), 2),
            "entanglement_strength": round(random.uniform(0.8, 1.0), 3),
            "fidelity": round(random.uniform(0.95, 0.999), 3),
            "bell_violation": round(random.uniform(2.0, 2.8), 2),
            "epr_pairs_active": random.randint(100, 1000),
            "foam_density": round(random.uniform(1.0, 3.0), 2)
        }

# 5. Files Page (/files)
@app.get("/files", response_class=HTMLResponse)
async def files_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Files | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff88;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 5px;
            color: #00ff88;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .upload-section {
            padding: 30px;
            background: rgba(0, 136, 255, 0.1);
            border: 2px dashed #0088ff;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
        }
        .upload-section:hover {
            background: rgba(0, 136, 255, 0.2);
            border-color: #00ddff;
        }
        .upload-section.dragover {
            background: rgba(0, 136, 255, 0.3);
            border-color: #00ff88;
        }
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: -9999px;
        }
        .upload-btn {
            padding: 15px 40px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-btn:hover {
            transform: scale(1.05);
        }
        .holo-info {
            margin: 20px 0;
            padding: 15px;
            background: rgba(0, 136, 255, 0.1);
            border: 1px solid #0088ff;
            border-radius: 5px;
            text-align: left;
        }
        .files-list {
            margin-top: 30px;
        }
        .file-item {
            padding: 20px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #333;
            border-radius: 8px;
            margin-bottom: 15px;
            transition: all 0.3s;
        }
        .file-item:hover {
            background: rgba(0, 255, 136, 0.1);
            border-color: #00ff88;
            transform: translateX(5px);
        }
        .file-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .file-name {
            font-size: 1.2em;
            color: #00ddff;
            font-weight: bold;
        }
        .file-size {
            color: #888;
        }
        .file-routing {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 136, 255, 0.05);
            border-left: 3px solid #0088ff;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .routing-item {
            margin: 5px 0;
        }
        .routing-label {
            color: #888;
            display: inline-block;
            width: 150px;
        }
        .routing-value {
            color: #00ddff;
        }
        .download-btn {
            padding: 8px 20px;
            background: rgba(0, 221, 255, 0.3);
            border: 1px solid #00ddff;
            color: #00ddff;
            border-radius: 5px;
            text-decoration: none;
            font-family: 'Courier New', monospace;
            cursor: pointer;
            transition: all 0.3s;
        }
        .download-btn:hover {
            background: rgba(0, 221, 255, 0.5);
            transform: scale(1.05);
        }
        .no-files {
            text-align: center;
            padding: 60px;
            color: #666;
            font-size: 1.2em;
        }
        .upload-progress {
            margin-top: 15px;
            padding: 10px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 5px;
            display: none;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00ddff);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0a0a0a;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>📁 QUANTUM FILE STORAGE</h1>
        
        <div class="upload-section" id="uploadSection">
            <h2 style="margin-bottom: 15px;">📤 Upload Files to Holo Storage</h2>
            <div class="holo-info">
                <strong>🔷 HOLO STORAGE CONFIGURATION</strong><br>
                Storage IP: <span style="color: #00ddff;">138.0.0.1</span><br>
                DNS Routing: <span style="color: #00ff88;">Enabled</span><br>
                Persistent: <span style="color: #00ff88;">Active</span><br>
                Quantum Routing: <span style="color: #00ff88;">EPR-based</span>
            </div>
            
            <div style="margin-top: 20px;">
                <div class="file-input-wrapper">
                    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                        📁 SELECT FILE
                    </button>
                    <input type="file" id="fileInput" onchange="uploadFile()" multiple>
                </div>
                <p style="margin-top: 15px; color: #888;">
                    or drag and drop files here
                </p>
            </div>
            
            <div class="upload-progress" id="uploadProgress">
                <div>Uploading to holo storage...</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill">0%</div>
                </div>
            </div>
        </div>
        
        <div class="files-list">
            <h2 style="margin-bottom: 20px;">📦 Stored Files</h2>
            <div id="filesList">
                <div class="no-files">No files uploaded yet</div>
            </div>
        </div>
    </div>
    
    <script>
        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        }
        
        async function loadFiles() {
            try {
                const response = await fetch('/api/files-with-routing');
                const files = await response.json();
                
                const filesList = document.getElementById('filesList');
                
                if (files.length === 0) {
                    filesList.innerHTML = '<div class="no-files">No files uploaded yet</div>';
                    return;
                }
                
                filesList.innerHTML = files.map(file => `
                    <div class="file-item">
                        <div class="file-header">
                            <span class="file-name">📄 ${file.name}</span>
                            <div>
                                <span class="file-size">${formatBytes(file.size)}</span>
                                <a href="/api/download/${file.name}" class="download-btn" style="margin-left: 15px;">
                                    ⬇️ Download
                                </a>
                            </div>
                        </div>
                        <div class="file-routing">
                            <div class="routing-item">
                                <span class="routing-label">Quantum Route:</span>
                                <span class="routing-value">${file.routing.quantum_route}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Holo Storage IP:</span>
                                <span class="routing-value">${file.routing.holo_storage}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">DNS Route:</span>
                                <span class="routing-value">${file.routing.dns_route}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Node IP:</span>
                                <span class="routing-value">${file.routing.node_ip}:${file.routing.port}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Latency:</span>
                                <span class="routing-value">${file.routing.latency_ms} ms</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Entanglement Quality:</span>
                                <span class="routing-value">${file.routing.entanglement_quality}</span>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading files:', error);
            }
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const files = fileInput.files;
            
            if (files.length === 0) return;
            
            const progress = document.getElementById('uploadProgress');
            const progressFill = document.getElementById('progressFill');
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const formData = new FormData();
                formData.append('file', file);
                
                progress.style.display = 'block';
                progressFill.style.width = '0%';
                progressFill.textContent = '0%';
                
                try {
                    // Simulate progress
                    let progressValue = 0;
                    const progressInterval = setInterval(() => {
                        progressValue += Math.random() * 20;
                        if (progressValue > 90) progressValue = 90;
                        progressFill.style.width = progressValue + '%';
                        progressFill.textContent = Math.round(progressValue) + '%';
                    }, 200);
                    
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    clearInterval(progressInterval);
                    progressFill.style.width = '100%';
                    progressFill.textContent = '100%';
                    
                    if (response.ok) {
                        setTimeout(() => {
                            progress.style.display = 'none';
                            loadFiles();
                        }, 500);
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                    alert('Upload failed: ' + error.message);
                    progress.style.display = 'none';
                }
            }
            
            fileInput.value = '';
        }
        
        // Drag and drop support
        const uploadSection = document.getElementById('uploadSection');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            
            const fileInput = document.getElementById('fileInput');
            fileInput.files = e.dataTransfer.files;
            uploadFile();
        });
        
        // Load files on page load
        loadFiles();
        
        // Auto-refresh every 10 seconds
        setInterval(loadFiles, 10000);
    </script>
</body>
</html>
    """)

# API for files list with routing
@app.get("/api/files-with-routing")
async def files_with_routing():
    files = []
    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            files.append({
                "name": filename,
                "size": size,
                "routing": {
                    "quantum_route": f"Q{''.join(random.choices('0123456789ABCDEF', k=8))}",
                    "holo_storage": config["holo_storage_ip"],
                    "dns_route": "holo.qfn.network" if config["holo_dns_enabled"] else "direct",
                    "node_ip": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                    "port": random.randint(10000, 65535),
                    "latency_ms": round(random.uniform(5, 50), 2),
                    "entanglement_quality": round(random.uniform(0.95, 0.999), 3)
                }
            })
    return files

# API for upload
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"message": f"File {file.filename} uploaded successfully"}

# API for download
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, filename=filename)

# 6. Collider Page (/collider)
@app.get("/collider", response_class=HTMLResponse)
async def collider_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Collider | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff88;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 5px;
            color: #00ff88;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .info-section {
            padding: 20px;
            background: rgba(0, 136, 255, 0.1);
            border: 1px solid #0088ff;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .info-title {
            font-size: 1.3em;
            color: #00ddff;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .query-section {
            text-align: center;
            margin-bottom: 30px;
        }
        .query-input-wrapper {
            display: flex;
            gap: 10px;
            justify-content: center;
            max-width: 600px;
            margin: 0 auto;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #00ff88;
            color: #00ff88;
            font-family: inherit;
            border-radius: 5px;
        }
        .collide-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #ff0088, #ff44cc);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .collide-btn:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(255, 0, 136, 0.5);
        }
        .collide-btn.colliding {
            background: #666;
            cursor: not-allowed;
        }
        .results {
            display: none;
            margin-top: 30px;
        }
        .results.show {
            display: block;
        }
        .result-header {
            font-size: 1.4em;
            color: #00ddff;
            margin-bottom: 20px;
            text-align: center;
        }
        .hash-display {
            display: flex;
            justify-content: space-between;
            padding: 15px;
            margin-bottom: 15px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #333;
            border-radius: 5px;
            word-break: break-all;
        }
        .hash-label {
            font-weight: bold;
            min-width: 150px;
        }
        .hash-value {
            flex: 1;
            font-family: monospace;
            font-size: 0.9em;
            color: #ff00ff;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .result-item {
            padding: 15px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 5px;
            text-align: center;
        }
        .result-label {
            color: #888;
            margin-bottom: 5px;
        }
        .result-value {
            font-weight: bold;
            color: #00ff88;
        }
        .history {
            margin-top: 40px;
        }
        .history-title {
            font-size: 1.3em;
            color: #00ddff;
            margin-bottom: 15px;
        }
        .history-item {
            padding: 15px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #333;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .history-query {
            color: #00ff88;
            margin-bottom: 5px;
        }
        .history-hash {
            font-size: 0.9em;
            color: #888;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>⚛️ QUANTUM COLLIDER</h1>
        
        <div class="info-section">
            <div class="info-title">🔬 Quantum State Hasher (QSH)</div>
            <p>
                The Quantum Collider uses high-energy particle collisions to generate quantum-resistant 
                hashes. By colliding particles at near-light speeds through the quantum foam, we create 
                unique hash signatures entangled with the fabric of spacetime itself.
            </p>
            <br>
            <p>
                <strong>Features:</strong> EPR-pair generation, foam perturbation analysis, 
                decoherence-resistant encoding, and multi-dimensional entanglement mapping.
            </p>
        </div>
        
        <div class="query-section">
            <h2 style="margin-bottom: 15px; color: #00ddff;">🎯 Enter Query String</h2>
            <div class="query-input-wrapper">
                <input 
                    type="text" 
                    id="queryInput" 
                    placeholder="Enter data to hash through quantum collider..."
                    onkeypress="handleKeyPress(event)"
                >
                <button class="collide-btn" id="collideBtn" onclick="runCollider()">
                    ⚡ COLLIDE
                </button>
            </div>
        </div>
        
        <div class="results" id="results">
            <div class="result-header">📊 COLLISION RESULTS</div>
            
            <div class="hash-display">
                <div class="hash-label">🔐 QSH HASH:</div>
                <div class="hash-value" id="qshHash">-</div>
            </div>
            
            <div class="hash-display" style="border-color: #00ff88; background: rgba(0, 255, 136, 0.1);">
                <div class="hash-label" style="color: #00ff88;">🔒 CLASSICAL HASH (SHA-256):</div>
                <div class="hash-value" id="classicalHash">-</div>
            </div>
            
            <div class="result-grid" id="resultGrid">
                <!-- Results populated by JavaScript -->
            </div>
        </div>
        
        <div class="history" id="historySection" style="display: none;">
            <div class="history-title">📜 Collision History</div>
            <div id="historyList"></div>
        </div>
    </div>
    
    <script>
        let collisionHistory = [];
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                runCollider();
            }
        }
        
        async function runCollider() {
            const query = document.getElementById('queryInput').value.trim();
            
            if (!query) {
                alert('Please enter a query string!');
                return;
            }
            
            const btn = document.getElementById('collideBtn');
            btn.classList.add('colliding');
            btn.textContent = '⚡ COLLIDING...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/qsh-query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const result = await response.json();
                
                // Wait a bit for effect
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                displayResults(result);
                addToHistory(result);
                
            } catch (error) {
                console.error('Collider error:', error);
                alert('Collision failed: ' + error.message);
            } finally {
                btn.classList.remove('colliding');
                btn.textContent = '⚡ COLLIDE';
                btn.disabled = false;
            }
        }
        
        function displayResults(result) {
            document.getElementById('qshHash').textContent = result.qsh_hash;
            document.getElementById('classicalHash').textContent = result.classical_hash;
            
            const resultGrid = document.getElementById('resultGrid');
            resultGrid.innerHTML = `
                <div class="result-item">
                    <div class="result-label">Entanglement Strength</div>
                    <div class="result-value">${result.entanglement_strength}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Collision Energy</div>
                    <div class="result-value">${result.collision_energy_gev} GeV</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Particle States</div>
                    <div class="result-value">${result.particle_states_generated}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Foam Perturbations</div>
                    <div class="result-value">${result.foam_perturbations}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Decoherence Time</div>
                    <div class="result-value">${result.decoherence_time_ns} ns</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Status</div>
                    <div class="result-value" style="color: #00ff88;">✓ SUCCESS</div>
                </div>
            `;
            
            document.getElementById('results').classList.add('show');
        }
        
        function addToHistory(result) {
            collisionHistory.unshift(result);
            
            // Keep only last 10
            if (collisionHistory.length > 10) {
                collisionHistory = collisionHistory.slice(0, 10);
            }
            
            const historySection = document.getElementById('historySection');
            const historyList = document.getElementById('historyList');
            
            historySection.style.display = 'block';
            
            historyList.innerHTML = collisionHistory.map((item, index) => `
                <div class="history-item">
                    <div class="history-query">
                        <strong>${index + 1}.</strong> "${item.query}"
                    </div>
                    <div class="history-hash">
                        QSH: ${item.qsh_hash} | Energy: ${item.collision_energy_gev} GeV | 
                        Time: ${new Date(item.timestamp).toLocaleTimeString()}
                    </div>
                </div>
            `).join('');
        }
    </script>
</body>
</html>
    """)

# API for QSH query
@app.post("/api/qsh-query")
async def qsh_query(request: Request):
    data = await request.json()
    query = data.get("query", "")
    
    # Simulate classical hash
    classical_hash = hashlib.sha256(query.encode()).hexdigest()
    
    # Simulate quantum hash (random hex for demo)
    qsh_hash = ''.join(random.choices('0123456789abcdef', k=64))
    
    return {
        "query": query,
        "qsh_hash": qsh_hash,
        "classical_hash": classical_hash,
        "entanglement_strength": round(random.uniform(0.85, 0.99), 3),
        "collision_energy_gev": round(random.uniform(1000, 13000), 0),
        "particle_states_generated": random.randint(100, 1000),
        "foam_perturbations": random.randint(50, 500),
        "decoherence_time_ns": round(random.uniform(10, 100), 2),
        "timestamp": datetime.now().isoformat()
    }

# 7. Config Page (/config)
@app.get("/config", response_class=HTMLResponse)
async def config_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Configuration | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff88;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 5px;
            color: #00ff88;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .config-section {
            margin: 30px 0;
            padding: 25px;
            background: rgba(0, 136, 255, 0.1);
            border: 1px solid #0088ff;
            border-radius: 8px;
        }
        .section-title {
            font-size: 1.4em;
            color: #00ddff;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .config-item {
            margin: 20px 0;
        }
        .config-label {
            display: block;
            color: #00ff88;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .config-description {
            color: #888;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        input[type="text"], select {
            width: 100%;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            font-size: 1em;
            border-radius: 5px;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #00ddff;
            box-shadow: 0 0 10px rgba(0, 221, 255, 0.3);
        }
        .checkbox-container {
            display: flex;
            align-items: center;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #00ff88;
            border-radius: 5px;
        }
        .checkbox-container input[type="checkbox"] {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            cursor: pointer;
        }
        .checkbox-container label {
            cursor: pointer;
            margin: 0;
        }
        .save-btn {
            display: block;
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 30px;
        }
        .save-btn:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        }
        .save-btn:active {
            transform: scale(0.98);
        }
        .status-message {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            display: none;
        }
        .status-message.success {
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
            display: block;
        }
        .status-message.error {
            background: rgba(255, 68, 68, 0.2);
            border: 1px solid #ff4444;
            color: #ff4444;
            display: block;
        }
        .current-value {
            color: #00ddff;
            font-weight: bold;
            margin-left: 10px;
        }
        .info-box {
            padding: 15px;
            background: rgba(255, 136, 0, 0.1);
            border: 1px solid #ff8800;
            border-radius: 5px;
            margin-top: 15px;
        }
        .info-box-title {
            color: #ff8800;
            font-weight: bold;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>⚙️ SYSTEM CONFIGURATION</h1>
        
        <div class="config-section">
            <div class="section-title">🔷 Holo Storage Settings</div>
            
            <div class="config-item">
                <label class="config-label">Storage IP Address</label>
                <div class="config-description">
                    IP address for the holo storage system. Default: 138.0.0.1
                </div>
                <input type="text" id="holoStorageIp" placeholder="138.0.0.1">
            </div>
            
            <div class="config-item">
                <label class="config-label">DNS Routing</label>
                <div class="config-description">
                    Enable DNS-based routing through holo storage nodes
                </div>
                <div class="checkbox-container">
                    <input type="checkbox" id="holoDnsEnabled">
                    <label for="holoDnsEnabled">Enable DNS Routing</label>
                </div>
            </div>
            
            <div class="config-item">
                <label class="config-label">Upload Directory</label>
                <div class="config-description">
                    Persistent storage directory path (read-only, set via environment)
                </div>
                <input type="text" id="uploadDirectory" disabled>
            </div>
            
            <div class="info-box">
                <div class="info-box-title">ℹ️ About Holo Storage</div>
                Holo storage provides persistent, quantum-routed file storage with EPR-based 
                encryption and DNS routing capabilities. All uploaded files are stored with 
                quantum coordinates for enhanced security.
            </div>
        </div>
        
        <div class="config-section">
            <div class="section-title">📡 Network Configuration</div>
            
            <div class="config-item">
                <label class="config-label">Default Interface</label>
                <div class="config-description">
                    Primary network interface for quantum operations
                </div>
                <select id="defaultInterface">
                    <option value="wlan0">wlan0 (Wireless)</option>
                    <option value="eth0">eth0 (Ethernet)</option>
                    <option value="qeth0">qeth0 (Quantum)</option>
                </select>
            </div>
            
            <div class="config-item">
                <label class="config-label">Default RF Mode</label>
                <div class="config-description">
                    Default RF hardware mode for Wireshark and packet analysis
                </div>
                <select id="defaultRfMode">
                    <option value="quantum">Quantum EPR</option>
                    <option value="4g_lte">4G LTE</option>
                    <option value="5g_nr">5G NR</option>
                </select>
            </div>
        </div>
        
        <div class="config-section">
            <div class="section-title">🎯 Quantum Settings</div>
            
            <div class="config-item">
                <label class="config-label">EPR Pair Generation Rate</label>
                <div class="config-description">
                    Number of EPR pairs generated per second (500-5000)
                </div>
                <input type="text" id="eprRate" placeholder="2500">
            </div>
            
            <div class="config-item">
                <label class="config-label">Entanglement Quality Threshold</label>
                <div class="config-description">
                    Minimum fidelity for quantum operations (0.950-0.999)
                </div>
                <input type="text" id="entanglementThreshold" placeholder="0.975">
            </div>
        </div>
        
        <button class="save-btn" onclick="saveConfig()">💾 SAVE CONFIGURATION</button>
        
        <div class="status-message" id="statusMessage"></div>
    </div>
    
    <script>
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                const config = await response.json();
                
                document.getElementById('holoStorageIp').value = config.holo_storage_ip || '138.0.0.1';
                document.getElementById('holoDnsEnabled').checked = config.holo_dns_enabled !== false;
                document.getElementById('uploadDirectory').value = config.upload_directory || '/opt/render/project/data/uploads';
                document.getElementById('defaultInterface').value = config.default_interface || 'wlan0';
                document.getElementById('defaultRfMode').value = config.default_rf_mode || 'quantum';
                document.getElementById('eprRate').value = config.epr_rate || '2500';
                document.getElementById('entanglementThreshold').value = config.entanglement_threshold || '0.975';
            } catch (error) {
                console.error('Error loading config:', error);
            }
        }
        
        async function saveConfig() {
            const statusMessage = document.getElementById('statusMessage');
            statusMessage.className = 'status-message';
            statusMessage.style.display = 'none';
            
            const configData = {
                holo_storage_ip: document.getElementById('holoStorageIp').value,
                holo_dns_enabled: document.getElementById('holoDnsEnabled').checked,
                upload_directory: document.getElementById('uploadDirectory').value,
                default_interface: document.getElementById('defaultInterface').value,
                default_rf_mode: document.getElementById('defaultRfMode').value,
                epr_rate: document.getElementById('eprRate').value,
                entanglement_threshold: document.getElementById('entanglementThreshold').value
            };
            
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(configData)
                });
                
                if (response.ok) {
                    statusMessage.className = 'status-message success';
                    statusMessage.textContent = '✓ Configuration saved successfully! Changes will take effect immediately.';
                    // Update global config
                    window.globalConfig = configData;
                } else {
                    throw new Error('Save failed');
                }
            } catch (error) {
                statusMessage.className = 'status-message error';
                statusMessage.textContent = '✗ Error saving configuration: ' + error.message;
            }
            
            // Hide message after 5 seconds
            setTimeout(() => {
                statusMessage.style.display = 'none';
            }, 5000);
        }
        
        // Load config on page load
        loadConfig();
    </script>
</body>
</html>
    """)

# API for config
@app.get("/api/config")
async def get_config():
    return config

@app.post("/api/config")
async def post_config(request: Request):
    global config
    data = await request.json()
    config.update(data)
    save_config(config)
    return {"message": "Config saved"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
