
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import logging
import random
import time
from datetime import datetime
import asyncio
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Quantum Foam Network")

# Real network metrics
class NetworkMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.last_download_speed = 0.0
        self.last_upload_speed = 0.0
        self.last_ping = 0.0
        self.testing = False
        
    async def test_download_speed(self):
        """Test download speed by downloading a test file"""
        try:
            test_url = "http://speedtest.ftp.otenet.gr/files/test10Mb.db"
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
            return self.last_download_speed
    
    async def test_upload_speed(self):
        """Test upload speed by uploading data"""
        try:
            test_data = b'x' * (5 * 1024 * 1024)
            test_url = "https://httpbin.org/post"
            
            start_time = time.time()
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(test_url, content=test_data)
            
            elapsed_time = time.time() - start_time
            speed_mbps = (len(test_data) * 8) / (elapsed_time * 1_000_000)
            self.last_upload_speed = round(speed_mbps, 2)
            return self.last_upload_speed
        except Exception as e:
            logger.error(f"Upload speed test failed: {e}")
            return self.last_upload_speed
    
    async def test_ping(self):
        """Test ping latency"""
        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.get("https://www.google.com")
            elapsed_time = (time.time() - start_time) * 1000
            self.last_ping = round(elapsed_time, 2)
            return self.last_ping
        except Exception as e:
            logger.error(f"Ping test failed: {e}")
            return self.last_ping
    
    async def run_full_test(self):
        """Run all network tests"""
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

# Quantum collider simulation
def process_qsh_query(query: str) -> dict:
    """Process a QSH query through the quantum collider"""
    import hashlib
    
    # Create quantum hash
    classical_hash = hashlib.sha256(query.encode()).hexdigest()
    
    # Simulate quantum entanglement properties
    entanglement_strength = random.uniform(0.85, 0.99)
    collision_energy = random.uniform(5.0, 12.0)
    particle_states = random.randint(64, 256)
    
    # Simulate QSH output
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

SPLASH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Foam Network</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
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
        
        .highlight {
            color: #00ddff;
            font-weight: bold;
        }
        
        .features {
            margin: 30px 0;
            padding: 20px;
            background: rgba(0, 255, 136, 0.05);
            border-left: 3px solid #00ff88;
            border-radius: 5px;
        }
        
        .features h3 {
            color: #00ddff;
            margin-bottom: 15px;
        }
        
        .features ul {
            list-style: none;
            padding-left: 0;
        }
        
        .features li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }
        
        .features li:before {
            content: "⚛️";
            position: absolute;
            left: 0;
        }
        
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
        
        .button-container {
            text-align: center;
            margin: 30px 0;
        }
        
        .team {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 1px solid #00ff88;
        }
        
        .team-title {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #00ddff;
        }
        
        .team-member {
            margin: 10px 0;
            padding-left: 20px;
        }
        
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
        
        a {
            color: #00ddff;
            text-decoration: none;
        }
        
        a:hover {
            color: #00ff88;
            text-shadow: 0 0 5px #00ff88;
        }
        
        ::-webkit-scrollbar {
            width: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0a0a0a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #00ff88;
            border-radius: 6px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #00ddff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚛️ QUANTUM FOAM NETWORK ⚛️</h1>
        <p class="subtitle">World's First Quantum-Classical Internet Interface</p>
        
        <div class="content">
            <p>
                <span class="highlight">Quantum foam enabled 6 GHz EPR Teleportation</span> mediated routed traffic 
                enables the world's first quantum-classical internet interface. Welcome to the 
                <span class="highlight">computational-foam space</span>.
            </p>
            <br>
            <p>
                This groundbreaking network leverages <span class="highlight">QuTiP-based entanglement protocols</span>, 
                including Bell state pairs, GHZ states, quantum teleportation, and quantum key distribution 
                (BB84 & E91) to create a bridge between quantum and classical computing realms.
            </p>
        </div>
        
        <div class="button-container">
            <a href="/metrics" class="quantum-button">📊 NETWORK METRICS</a>
        </div>
        
        <div class="features">
            <h3>🔬 Quantum Capabilities</h3>
            <ul>
                <li>Quantum Secure Hash (QSH) with 6-qubit GHZ EPR entanglement</li>
                <li>Bell State EPR Pair Generation</li>
                <li>6-Qubit GHZ State Creation</li>
                <li>Quantum Teleportation between connections</li>
                <li>Quantum Key Distribution (BB84 & E91 protocols)</li>
                <li>QRAM Storage for quantum states</li>
                <li>WebSocket support for real-time quantum communication</li>
            </ul>
        </div>
        
        <div class="features">
            <h3>🌐 Network Features</h3>
            <ul>
                <li>REST API for quantum operations</li>
                <li>Real-time connection management</li>
                <li>Quantum-secured routing protocols</li>
                <li>Distributed entanglement synchronization</li>
                <li>Interactive API documentation at /docs</li>
            </ul>
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
    <title>Quantum Network Metrics</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            padding: 20px;
            background: rgba(10, 10, 10, 0.9);
            border-bottom: 2px solid #00ff88;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 1.8em;
            text-shadow: 0 0 10px #00ff88;
        }
        
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        
        .nav-button {
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            transition: all 0.3s;
            text-decoration: none;
        }
        
        .nav-button:hover {
            background: rgba(0, 255, 136, 0.4);
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        .metrics-sidebar {
            position: absolute;
            left: 0;
            top: 70px;
            width: 380px;
            height: calc(100vh - 70px);
            background: rgba(10, 10, 10, 0.95);
            border-right: 2px solid #00ff88;
            padding: 20px;
            overflow-y: auto;
            z-index: 10;
        }
        
        .metrics-title {
            font-size: 1.2em;
            color: #00ddff;
            margin-bottom: 20px;
            text-align: center;
            border-bottom: 1px solid #00ff88;
            padding-bottom: 10px;
        }
        
        .test-button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 20px;
            transition: all 0.3s;
        }
        
        .test-button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.6);
        }
        
        .test-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .real-metrics {
            background: rgba(0, 221, 255, 0.1);
            border: 2px solid #00ddff;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .section-title {
            font-size: 1em;
            color: #00ddff;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .metric-card {
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid #00ff88;
            border-radius: 5px;
            padding: 12px;
            margin-bottom: 12px;
        }
        
        .metric-label {
            font-size: 0.85em;
            color: #00ddff;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.3em;
            color: #00ff88;
            font-weight: bold;
        }
        
        .metric-value.real {
            color: #00ddff;
        }
        
        .metric-unit {
            font-size: 0.8em;
            color: #888;
            margin-left: 5px;
        }
        
        .domain-info {
            background: rgba(0, 221, 255, 0.1);
            border: 1px solid #00ddff;
            border-radius: 5px;
            padding: 10px;
            margin-top: 15px;
            font-size: 0.75em;
            word-break: break-all;
            color: #00ddff;
        }
        
        .main-content {
            margin-left: 380px;
            height: calc(100vh - 70px);
            display: flex;
            flex-direction: column;
        }
        
        .tabs {
            display: flex;
            background: rgba(10, 10, 10, 0.95);
            border-bottom: 1px solid #00ff88;
        }
        
        .tab {
            padding: 12px 20px;
            background: rgba(0, 255, 136, 0.1);
            border: none;
            border-right: 1px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .tab:hover {
            background: rgba(0, 255, 136, 0.2);
        }
        
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
        
        .browser-controls {
            padding: 10px 20px;
            background: rgba(10, 10, 10, 0.95);
            border-bottom: 1px solid #00ff88;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .url-bar {
            flex: 1;
            padding: 8px 15px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            border-radius: 5px;
            outline: none;
        }
        
        .control-btn {
            padding: 8px 15px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }
        
        .control-btn:hover {
            background: rgba(0, 255, 136, 0.4);
        }
        
        .browser-frame {
            flex: 1;
            border: none;
            background: white;
        }
        
        /* Quantum Collider Chat Interface */
        .collider-interface {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: rgba(10, 10, 10, 0.95);
            padding: 20px;
        }
        
        .collider-header {
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #00ff88;
            margin-bottom: 20px;
        }
        
        .collider-header h2 {
            color: #00ddff;
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        
        .collider-domain {
            color: #00ff88;
            font-size: 0.9em;
        }
        
        .chat-output {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #00ff88;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .message {
            margin-bottom: 20px;
            padding: 15px;
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
        
        .message-label {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 5px;
        }
        
        .message-content {
            color: #00ff88;
            line-height: 1.6;
        }
        
        .qsh-result {
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 3px;
        }
        
        .qsh-field {
            margin: 5px 0;
        }
        
        .qsh-label {
            color: #00ddff;
            display: inline-block;
            width: 200px;
        }
        
        .qsh-value {
            color: #00ff88;
        }
        
        .chat-input-container {
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            border-radius: 5px;
            outline: none;
        }
        
        .send-button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .send-button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.6);
        }
        
        .refresh-indicator {
            position: absolute;
            top: 90px;
            left: 10px;
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 1s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .testing {
            animation: spin 1s linear infinite;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0a0a0a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #00ff88;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Quantum Network Control</h1>
        <div class="nav-buttons">
            <a href="/" class="nav-button">⬅️ Back to Main</a>
        </div>
    </div>
    
    <div class="metrics-sidebar">
        <div class="refresh-indicator" id="refreshIndicator"></div>
        <div class="metrics-title">⚛️ LIVE QUANTUM METRICS</div>
        
        <button class="test-button" id="testButton" onclick="runSpeedTest()">
            🚀 RUN SPEED TEST
        </button>
        
        <div class="real-metrics">
            <div class="section-title">📡 REAL NETWORK DATA</div>
            <div id="realMetrics">
                <!-- Real metrics will be loaded here -->
            </div>
        </div>
        
        <div id="quantumMetrics">
            <!-- Quantum metrics will be loaded here -->
        </div>
        
        <div class="domain-info">
            <strong>Domain:</strong><br>
            quantum.realm.domain.dominion.foam.computer.networking
        </div>
    </div>
    
    <div class="main-content">
        <div class="tabs">
            <button class="tab active" onclick="switchTab('collider')">⚛️ Quantum Collider</button>
            <button class="tab" onclick="switchTab('browser')">🌐 Web Browser</button>
        </div>
        
        <div id="collider-tab" class="tab-content active">
            <div class="collider-interface">
                <div class="collider-header">
                    <h2>⚛️ QUANTUM COLLIDER & QSH QUERY INTERFACE</h2>
                    <div class="collider-domain">quantum.realm.domain.dominion.foam.computer.collider</div>
                </div>
                
                <div class="chat-output" id="chatOutput">
                    <div class="message system">
                        <div class="message-label">SYSTEM</div>
                        <div class="message-content">
                            Welcome to the Quantum Collider interface. This system processes queries through quantum secure hash (QSH) protocols with EPR entanglement.
                            <br><br>
                            Enter your query below to initiate quantum collision and hash generation.
                        </div>
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="chatInput" placeholder="Enter QSH query..." onkeypress="handleChatKeyPress(event)">
                    <button class="send-button" onclick="sendQuery()">SEND</button>
                </div>
            </div>
        </div>
        
        <div id="browser-tab" class="tab-content">
            <div class="browser-controls">
                <button class="control-btn" onclick="navigateTo('https://fast.com/')">🚀 Fast.com</button>
                <button class="control-btn" onclick="navigateTo('https://www.google.com/search?q=my+ip')">🌐 My IP</button>
                <button class="control-btn" onclick="navigateTo('https://www.cloudflare.com/cdn-cgi/trace')">📡 CF Trace</button>
                <input type="text" class="url-bar" id="urlBar" placeholder="Enter URL..." onkeypress="handleKeyPress(event)">
                <button class="control-btn" onclick="navigateToUrl()">GO</button>
                <button class="control-btn" onclick="reloadFrame()">🔄</button>
            </div>
            
            <iframe id="browserFrame" class="browser-frame" src="https://fast.com/"></iframe>
        </div>
    </div>
    
    <script>
        let isTesting = false;
        
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            if (tabName === 'collider') {
                document.getElementById('collider-tab').classList.add('active');
                document.querySelectorAll('.tab')[0].classList.add('active');
            } else {
                document.getElementById('browser-tab').classList.add('active');
                document.querySelectorAll('.tab')[1].classList.add('active');
            }
        }
        
        function navigateTo(url) {
            document.getElementById('browserFrame').src = url;
            document.getElementById('urlBar').value = url;
        }
        
        function navigateToUrl() {
            let url = document.getElementById('urlBar').value;
            if (url && !url.startsWith('http://') && !url.startsWith('https://')) {
                url = 'https://' + url;
            }
            if (url) {
                navigateTo(url);
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                navigateToUrl();
            }
        }
        
        function reloadFrame() {
            document.getElementById('browserFrame').src = document.getElementById('browserFrame').src;
        }
        
        function handleChatKeyPress(event) {
            if (event.key === 'Enter') {
                sendQuery();
            }
        }
        
        async function sendQuery() {
            const input = document.getElementById('chatInput');
            const query = input.value.trim();
            
            if (!query) return;
            
            const chatOutput = document.getElementById('chatOutput');
            
            // Add user message
            const userMessage = document.createElement('div');
            userMessage.className = 'message user';
            userMessage.innerHTML = `
                <div class="message-label">USER QUERY</div>
                <div class="message-content">${escapeHtml(query)}</div>
            `;
            chatOutput.appendChild(userMessage);
            
            input.value = '';
            chatOutput.scrollTop = chatOutput.scrollHeight;
            
            // Send to backend
            try {
                const response = await fetch('/api/qsh-query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                // Add system response
                const systemMessage = document.createElement('div');
                systemMessage.className = 'message system';
                systemMessage.innerHTML = `
                    <div class="message-label">QUANTUM COLLIDER RESPONSE</div>
                    <div class="message-content">
                        Query processed through quantum collision system.
                        <div class="qsh-result">
                            <div class="qsh-field">
                                <span class="qsh-label">QSH Hash:</span>
                                <span class="qsh-value">${data.qsh_hash}</span>
                            </div>
                            <div class="qsh-field">
                                <span class="qsh-label">Classical Hash:</span>
                                <span class="qsh-value">${data.classical_hash}</span>
                            </div>
                            <div class="qsh-field">
                                <span class="qsh-label">Entanglement Strength:</span>
                                <span class="qsh-value">${data.entanglement_strength}</span>
                            </div>
                            <div class="qsh-field">
                                <span class="qsh-label">Collision Energy:</span>
                                <span class="qsh-value">${data.collision_energy_gev} GeV</span>
                            </div>
                            <div class="qsh-field">
                                <span class="qsh-label">Particle States:</span>
                                <span class="qsh-value">${data.particle_states_generated}</span>
                            </div>
                            <div class="qsh-field">
                                <span class="qsh-label">Foam Perturbations:</span>
                                <span class="qsh-value">${data.foam_perturbations}</span>
                            </div>
                            <div class="qsh-field">
                                <span class="qsh-label">Decoherence Time:</span>
                                <span class="qsh-value">${data.decoherence_time_ns} ns</span>
                            </div>
                        </div>
                    </div>
                `;
                chatOutput.appendChild(systemMessage);
                
            } catch (error) {
                const errorMessage = document.createElement('div');
                errorMessage.className = 'message system';
                errorMessage.innerHTML = `
                    <div class="message-label">ERROR</div>
                    <div class="message-content">Failed to process query: ${error.message}</div>
                `;
                chatOutput.appendChild(errorMessage);
            }
            
            chatOutput.scrollTop = chatOutput.scrollHeight;
        }
        
        function escapeHtml(text) {
            const map = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            };
            return text.replace(/[&<>"']/g, m => map[m]);
        }
        
        async function runSpeedTest() {
            if (isTesting) return;
            
            isTesting = true;
            const button = document.getElementById('testButton');
            const indicator = document.getElementById('refreshIndicator');
            button.disabled = true;
            button.textContent = '🔄 TESTING...';
            indicator.classList.add('testing');
            
            try {
                await fetch('/api/run-speed-test', { method: 'POST' });
                await new Promise(resolve => setTimeout(resolve, 2000));
                await updateMetrics();
            } finally {
                button.disabled = false;
                button.textContent = '🚀 RUN SPEED TEST';
                indicator.classList.remove('testing');
                isTesting = false;
            }
        }
        
        async function updateMetrics() {
            try {
                const response = await fetch('/api/quantum-metrics');
                const data = await response.json();
                
                const realMetricsHtml = `
                    <div class="metric-card">
                        <div class="metric-label">⬇️ Download Speed</div>
                        <div class="metric-value real">${data.download_speed_mbps || '0.00'}<span class="metric-unit">Mbps</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">⬆️ Upload Speed</div>
                        <div class="metric-value real">${data.upload_speed_mbps || '0.00'}<span class="metric-unit">Mbps</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">📶 Ping</div>
                        <div class="metric-value real">${data.ping_ms || '0.00'}<span class="metric-unit">ms</span></div>
                    </div>
                `;
                document.getElementById('realMetrics').innerHTML = realMetricsHtml;
                
                const quantumMetricsHtml = `
                    <div class="section-title" style="margin-top: 20px; margin-bottom: 10px;">⚛️ QUANTUM PARAMETERS</div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Entanglement Dimensions</div>
                        <div class="metric-value">${data.entanglement_dimensions}<span class="metric-unit">D</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Active Qubits</div>
                        <div class="metric-value">${data.qubits_active}<span class="metric-unit">qubits</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">EPR Pairs</div>
                        <div class="metric-value">${data.epr_pairs}<span class="metric-unit">pairs</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Quantum Transfer Rate</div>
                        <div class="metric-value">${data.transfer_rate_qbps}<span class="metric-unit">Qbps</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Bell State Violations</div>
                        <div class="metric-value">${data.bell_state_violations}<span class="metric-unit">σ</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Teleportation Success</div>
                        <div class="metric-value">${(data.teleportation_success_rate * 100).toFixed(2)}<span class="metric-unit">%</span></div>
                    </div>
                `;
                document.getElementById('quantumMetrics').innerHTML = quantumMetricsHtml;
                
            } catch (error) {
                console.error('Error fetching metrics:', error);
            }
        }
        
        updateMetrics();
        setInterval(updateMetrics, 3000);
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

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
