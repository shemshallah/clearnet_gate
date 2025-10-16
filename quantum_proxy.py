
```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Quantum Foam Network")

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
            <a href="/metrics" class="quantum-button">📊 NETWORK METRICS & SPEEDTEST</a>
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
        
        .nav-button.active {
            background: rgba(0, 221, 255, 0.3);
            border-color: #00ddff;
            color: #00ddff;
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
        }
        
        .control-btn:hover {
            background: rgba(0, 255, 136, 0.4);
        }
        
        .browser-frame {
            flex: 1;
            border: none;
            background: white;
        }
        
        .back-link {
            display: inline-block;
            padding: 10px 20px;
            margin: 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
            border-radius: 5px;
            text-decoration: none;
            transition: all 0.3s;
        }
        
        .back-link:hover {
            background: rgba(0, 255, 136, 0.4);
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Quantum Network Metrics</h1>
        <div class="nav-buttons">
            <a href="/" class="nav-button">⬅️ Back to Main</a>
        </div>
    </div>
    
    <div class="browser-controls">
        <button class="control-btn" onclick="navigateTo('https://www.speedtest.net/')">⚡ Speedtest</button>
        <button class="control-btn" onclick="navigateTo('https://fast.com/')">🚀 Fast.com</button>
        <button class="control-btn" onclick="navigateTo('https://www.google.com/search?q=my+ip')">🌐 My IP</button>
        <button class="control-btn" onclick="navigateTo('https://www.cloudflare.com/cdn-cgi/trace')">📡 CF Trace</button>
        <input type="text" class="url-bar" id="urlBar" placeholder="Enter URL..." onkeypress="handleKeyPress(event)">
        <button class="control-btn" onclick="navigateToUrl()">GO</button>
        <button class="control-btn" onclick="reloadFrame()">🔄</button>
    </div>
    
    <iframe id="browserFrame" class="browser-frame" src="https://www.speedtest.net/"></iframe>
    
    <script>
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
        
        // Update URL bar when iframe navigates (if same-origin)
        setInterval(() => {
            try {
                const iframe = document.getElementById('browserFrame');
                if (iframe.contentWindow.location.href !== 'about:blank') {
                    document.getElementById('urlBar').value = iframe.contentWindow.location.href;
                }
            } catch(e) {
                // Cross-origin restriction - ignore
            }
        }, 1000);
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

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
