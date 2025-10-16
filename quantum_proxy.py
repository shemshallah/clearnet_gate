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
        body {
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
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
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #00ddff;
            margin-bottom: 30px;
        }
        .content {
            line-height: 1.8;
            margin: 20px 0;
        }
        .team {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #00ff88;
        }
        .status {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚛️ QUANTUM FOAM NETWORK ⚛️</h1>
        <p class="subtitle">World's First Quantum-Classical Internet Interface</p>
        
        <div class="content">
            <p><strong>Quantum foam enabled 6 GHz EPR Teleportation</strong> mediated routed traffic 
            enables the world's first quantum-classical internet interface.</p>
            <p>Welcome to the <strong>computational-foam space</strong>.</p>
        </div>
        
        <div class="team">
            <strong>Built by:</strong><br>
            🔷 hackah::hackah<br>
            🔷 Justin Howard-Stanley - shemshallah@gmail.com<br>
            🔷 Dale Cwidak
        </div>
        
        <div class="status">
            <strong>⚡ QUANTUM ENTANGLEMENT ACTIVE</strong><br>
            <small>System operational</small>
        </div>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def splash():
    return SPLASH_HTML

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
