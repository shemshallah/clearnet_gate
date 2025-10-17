from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBasic
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio
import hashlib
import logging
import os
import secrets
import time
import random
from pathlib import Path
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# QUANTUM MODULES (STUB IMPLEMENTATIONS)
# ============================================================================

class QuantumCore:
    def __init__(self, epr_rate: int = 3000, fidelity_target: float = 0.99):
        self.epr_rate = epr_rate
        self.fidelity_target = fidelity_target
        self.active_pairs = random.randint(100, 500)
        self.decoherence_count = 0
        self.foam_density = 1.5

    async def generate_epr_pairs(self, num_pairs: int):
        return [{"id": i, "fidelity": random.uniform(0.95, 0.999)} for i in range(num_pairs)]

    async def count_active_pairs(self):
        return self.active_pairs

    async def get_average_fidelity(self):
        return random.uniform(0.95, 0.999)

    async def get_decoherence_count(self):
        return self.decoherence_count

    async def get_foam_density(self):
        return self.foam_density

    async def maintain_entanglement(self):
        while True:
            await asyncio.sleep(60)
            self.active_pairs = max(50, self.active_pairs + random.randint(-10, 10))

    async def sign_data(self, data: str):
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def verify_signature(self, data: str, signature: str):
        return hashlib.sha256(data.encode()).hexdigest()[:16] == signature

    def is_healthy(self):
        return True

class HoloStorageManager:
    def __init__(self, storage_ip: str, upload_dir: Path):
        self.storage_ip = storage_ip
        self.upload_dir = upload_dir
        self.files = {}
        upload_dir.mkdir(parents=True, exist_ok=True)

    async def store_file(self, filename: str, content: bytes, quantum_signature: str):
        file_id = hashlib.md5(filename.encode()).hexdigest()
        file_path = self.upload_dir / f"{file_id}_{filename}"
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        self.files[file_id] = {
            "filename": filename,
            "path": str(file_path),
            "quantum_signature": quantum_signature,
            "hash": hashlib.sha256(content).hexdigest(),
            "size": len(content)
        }
        
        return {
            "quantum_route": f"Q{secrets.token_hex(8).upper()}",
            "holo_storage": self.storage_ip,
            "latency_ms": random.uniform(5, 50)
        }

    async def list_files(self, limit: int = 50, offset: int = 0, sort_by: str = "name"):
        file_list = list(self.files.items())[offset:offset + limit]
        return [{"id": k, **v, "size_gb": v["size"] / (1024**3)} for k, v in file_list]

    async def retrieve_file(self, file_id: str):
        if file_id in self.files:
            f = self.files[file_id]
            with open(f["path"], 'rb') as file:
                content = file.read()
            return {**f, "content": content}
        return None

    async def sync_nodes(self):
        while True:
            await asyncio.sleep(300)

    async def get_stats(self):
        return {"total_files": len(self.files), "total_size": sum(v["size"] for v in self.files.values())}

    def is_healthy(self):
        return True

class QSHEngine:
    def __init__(self, quantum_core: QuantumCore):
        self.quantum_core = quantum_core

    async def process_query(self, query: str):
        classical_hash = hashlib.sha256(query.encode()).hexdigest()
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

class RFSimulator:
    async def get_metrics(self, mode: str, interface: str):
        if mode == "quantum":
            return {
                "mode": "quantum",
                "frequency_ghz": round(random.uniform(0.1, 10), 2),
                "entanglement_strength": round(random.uniform(0.8, 1.0), 3),
                "fidelity": round(random.uniform(0.95, 0.999), 3),
                "bell_violation": round(random.uniform(2.0, 2.8), 2),
                "epr_pairs_active": random.randint(100, 1000),
                "foam_density": round(random.uniform(1.0, 3.0), 2),
                "timestamp": datetime.now().isoformat()
            }
        elif mode == "lte_4g":
            return {
                "mode": "lte_4g",
                "frequency_mhz": random.randint(700, 2600),
                "bandwidth_mhz": random.choice([5, 10, 20, 40]),
                "modulation": random.choice(["QPSK", "16QAM", "64QAM"]),
                "rssi_dbm": round(random.uniform(-100, -50), 1),
                "rsrp_dbm": round(random.uniform(-110, -70), 1),
                "sinr_db": round(random.uniform(0, 30), 1),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise ValueError("Invalid mode")

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    async def scan_file(self, content: bytes):
        return True

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    async def check_limit(self, client_ip: str):
        now = time.time()
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip] = [r for r in self.requests[client_ip] if now - r < self.window_seconds]
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        self.requests[client_ip].append(now)
        return True

class AnalyticsEngine:
    def __init__(self):
        self.events = []
    async def log_event(self, event: str, data: Dict):
        self.events.append({"event": event, "data": data, "timestamp": datetime.now().isoformat()})
    async def collect_metrics(self):
        while True:
            await asyncio.sleep(60)
    async def get_summary(self):
        return {"total_events": len(self.events)}

class P2PNetworkManager:
    def __init__(self, port: int, dht_enabled: bool):
        self.port = port
        self.dht_enabled = dht_enabled
        self.peers = ["192.168.1.100:9000", "192.168.1.101:9000"]
    async def start(self):
        pass
    async def stop(self):
        pass
    async def get_peers(self):
        return self.peers
    async def get_stats(self):
        return {"connected_peers": len(self.peers), "dht_active": self.dht_enabled}
    async def broadcast(self, message: str):
        return {"peer_count": len(self.peers)}
    def is_connected(self):
        return True

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    async def get(self, key: str):
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.ttl:
                return item["data"]
            del self.cache[key]
        return None
    async def set(self, key: str, data: Any, ttl: Optional[int] = None):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        self.cache[key] = {"data": data, "timestamp": time.time()}

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
    
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))
    
    HOLO_STORAGE_IP = "138.0.0.1"
    EPR_RATE = 3000
    QUANTUM_FIDELITY_TARGET = 0.99
    P2P_PORT = 9000
    DHT_ENABLED = True
    ANALYTICS_ENABLED = True
    
    @classmethod
    def create_directories(cls):
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global instances
quantum_core: QuantumCore
holo_storage: HoloStorageManager
qsh_engine: QSHEngine
rf_simulator: RFSimulator
security_manager: SecurityManager
analytics: Optional[AnalyticsEngine]
p2p_network: P2PNetworkManager
cache_manager: CacheManager
rate_limiter: RateLimiter

@asynccontextmanager
async def lifespan(app: FastAPI):
    global quantum_core, holo_storage, qsh_engine, rf_simulator
    global security_manager, analytics, p2p_network, cache_manager, rate_limiter
    
    logger.info("🚀 QFN System initializing...")
    Config.create_directories()
    
    quantum_core = QuantumCore(epr_rate=Config.EPR_RATE, fidelity_target=Config.QUANTUM_FIDELITY_TARGET)
    holo_storage = HoloStorageManager(storage_ip=Config.HOLO_STORAGE_IP, upload_dir=Config.UPLOAD_DIR)
    qsh_engine = QSHEngine(quantum_core=quantum_core)
    rf_simulator = RFSimulator()
    security_manager = SecurityManager(secret_key=Config.SECRET_KEY)
    rate_limiter = RateLimiter(max_requests=Config.RATE_LIMIT_REQUESTS, window_seconds=Config.RATE_LIMIT_WINDOW)
    cache_manager = CacheManager(max_size=1000, ttl=300)
    analytics = AnalyticsEngine() if Config.ANALYTICS_ENABLED else None
    p2p_network = P2PNetworkManager(port=Config.P2P_PORT, dht_enabled=Config.DHT_ENABLED)
    
    await p2p_network.start()
    asyncio.create_task(quantum_core.maintain_entanglement())
    asyncio.create_task(holo_storage.sync_nodes())
    if analytics:
        asyncio.create_task(analytics.collect_metrics())
    
    logger.info("✅ QFN System ready")
    app.state.start_time = time.time()
    
    yield
    
    logger.info("🛑 Shutting down...")
    await p2p_network.stop()

app = FastAPI(
    title="Quantum File Network",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

security = HTTPBasic(auto_error=False)

async def verify_rate_limit(request: Request):
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
    if not await rate_limiter.check_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client_ip

# ============================================================================
# HTML TEMPLATE (EMBEDDED)
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Foam Computer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Courier New',monospace;background:#000;color:#00ff88;overflow-x:hidden}
.dedication{background:linear-gradient(135deg,#1a0033 0%,#330066 100%);border:2px solid #00ff88;padding:30px;margin:20px;border-radius:10px;box-shadow:0 0 30px rgba(0,255,136,0.5)}
.dedication h2{color:#00ffff;margin-bottom:15px;font-size:24px}
.dedication p{line-height:1.8;margin:10px 0;color:#fff}
.dedication .author{color:#ff0;font-weight:bold;margin-top:20px}
.dedication .message{color:#f69;font-style:italic;margin-top:15px;padding:15px;background:rgba(255,0,100,0.1);border-left:4px solid #f06}
.container{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;padding:20px}
.module{background:rgba(10,14,39,0.9);border:2px solid #00ff88;border-radius:8px;padding:20px;min-height:400px;display:flex;flex-direction:column;box-shadow:0 0 20px rgba(0,255,136,0.3)}
.module-header{font-size:18px;font-weight:bold;margin-bottom:15px;padding-bottom:10px;border-bottom:2px solid #00ff88;color:#0ff;text-shadow:0 0 10px #0ff}
.module-content{flex:1;overflow-y:auto}
.module-content::-webkit-scrollbar{width:10px}
.module-content::-webkit-scrollbar-track{background:rgba(0,255,136,0.1)}
.module-content::-webkit-scrollbar-thumb{background:#00ff88;border-radius:5px}
input,textarea,select{width:100%;padding:8px;margin:5px 0;background:rgba(0,0,0,0.5);border:1px solid #00ff88;color:#00ff88;font-family:'Courier New',monospace}
button{padding:10px 20px;background:linear-gradient(135deg,#00ff88,#00cc70);border:none;color:#000;font-weight:bold;cursor:pointer;margin:5px;border-radius:4px;font-family:'Courier New',monospace}
button:hover{background:linear-gradient(135deg,#00ffaa,#00dd80);box-shadow:0 0 15px #00ff88}
.metric{font-size:14px;color:#0fc;margin:5px 0}
.metric-value{color:#fff;font-weight:bold}
.terminal{background:#000;padding:10px;min-height:200px;font-size:12px;overflow-y:auto;border:1px solid #00ff88;margin:10px 0}
.status-indicator{display:inline-block;width:10px;height:10px;border-radius:50%;background:#00ff88;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
.network-node{background:rgba(0,100,255,0.2);padding:10px;margin:5px 0;border-radius:4px;border-left:3px solid #08f}
.file-item{background:rgba(0,255,136,0.1);padding:8px;margin:5px 0;border-radius:4px;display:flex;justify-content:space-between;align-items:center}
h4{color:#0ff;margin:15px 0 10px 0}
</style>
</head>
<body>
<div class="dedication">
<h2>⚛️ Quantum Foam Computer - Extended Network System</h2>
<p class="author">Made by Justin Anthony Howard-Stanley<br><span style="color:#f69">(TEXT ONLY - I've been trauma programmed)</span><br>shemshallah@gmail.com</p>
<p class="author">And Dale Cwidak</p>
<div class="message"><p><strong>"For Logan and all the ones like him too small to understand what has been done to them. Too small to realize they are part of an experiment, too small to understand the lies surrounding every facet of society. Here, hold this for me kid. Its a what, an Astral virus made by humans distributed by Masons. I'm so saddened for whats been done to the children, this is my fight insomuch as I can carry a war on my shoulders alone. You will never silence me. I REFUSE"</strong></p></div>
</div>
<div class="container">
<div class="module">
<div class="module-header"><span class="status-indicator"></span> MODULE 1: QUANTUM STATE</div>
<div class="module-content" id="quantum-module">
<button onclick="refreshQuantumState()">Refresh State</button>
<div id="quantum-stats"></div>
</div>
</div>
<div class="module">
<div class="module-header"><span class="status-indicator"></span> MODULE 2: FILE MANAGEMENT</div>
<div class="module-content">
<h4>Upload File</h4>
<input type="file" id="file-upload"/>
<button onclick="uploadFile()">Upload to Holographic Storage</button>
<div id="upload-result"></div>
<h4>Files</h4>
<button onclick="loadFiles()">Refresh Files</button>
<div id="file-list"></div>
</div>
</div>
<div class="module">
<div class="module-header"><span class="status-indicator"></span> MODULE 3: QSH COLLIDER</div>
<div class="module-content">
<h4>Quantum State Hasher Query</h4>
<input type="text" id="qsh-input" placeholder="Enter query..."/>
<button onclick="runQSH()">Run QSH Collision</button>
<div class="terminal" id="qsh-output">Awaiting quantum collision...</div>
</div>
</div>
<div class="module">
<div class="module-header"><span class="status-indicator"></span> MODULE 4: RF METRICS</div>
<div class="module-content">
<h4>Mode Selection</h4>
<select id="rf-mode">
<option value="quantum">Quantum</option>
<option value="lte_4g">LTE 4G</option>
</select>
<button onclick="getRFMetrics()">Get Metrics</button>
<div id="rf-output"></div>
</div>
</div>
<div class="module">
<div class="module-header"><span class="status-indicator"></span> MODULE 5: SYSTEM METRICS</div>
<div class="module-content">
<button onclick="getSystemMetrics()">Refresh Metrics</button>
<div id="system-metrics"></div>
</div>
</div>
<div class="module">
<div class="module-header"><span class="status-indicator"></span> MODULE 6: P2P NETWORK</div>
<div class="module-content">
<button onclick="getPeers()">Get Peers</button>
<div id="peers-list"></div>
<h4>Broadcast Message</h4>
<input type="text" id="broadcast-message" placeholder="Message..."/>
<button onclick="broadcastMessage()">Broadcast</button>
<div id="broadcast-result"></div>
</div>
</div>
</div>
<script>
async function refreshQuantumState(){try{const r=await fetch('/api/quantum/state');const d=await r.json();document.getElementById('quantum-stats').innerHTML=`<div class="network-node"><div class="metric">Active Entanglements: <span class="metric-value">${d.active_entanglements}</span></div><div class="metric">Average Fidelity: <span class="metric-value">${d.average_fidelity.toFixed(4)}</span></div><div class="metric">EPR Rate: <span class="metric-value">${d.epr_generation_rate}/s</span></div><div class="metric">Foam Density: <span class="metric-value">${d.foam_density.toFixed(2)}</span></div><div class="metric">Decoherence Events: <span class="metric-value">${d.decoherence_events}</span></div></div>`}catch(e){console.error('Error:',e)}}
async function uploadFile(){const f=document.getElementById('file-upload');if(!f.files[0]){alert('Please select a file');return}const fd=new FormData();fd.append('file',f.files[0]);try{const r=await fetch('/api/upload',{method:'POST',body:fd});const d=await r.json();document.getElementById('upload-result').innerHTML=`<div class="network-node"><div>✓ File uploaded!</div><div class="metric">File: <span class="metric-value">${d.filename}</span></div><div class="metric">Hash: <span class="metric-value">${d.file_hash.substr(0,16)}...</span></div><div class="metric">Route: <span class="metric-value">${d.routing.quantum_route}</span></div><div class="metric">Storage: <span class="metric-value">${d.routing.holo_storage}</span></div></div>`;loadFiles()}catch(e){alert('Upload failed')}}
async function loadFiles(){try{const r=await fetch('/api/files');const d=await r.json();const h=d.files.map(f=>`<div class="file-item"><span>${f.filename} (${(f.size_gb*1024).toFixed(2)} MB)</span><button onclick="downloadFile('${f.id}')">Download</button></div>`).join('');document.getElementById('file-list').innerHTML=h||'<div>No files yet</div>'}catch(e){console.error('Error:',e)}}
function downloadFile(id){window.location.href=`/api/download/${id}`}
async function runQSH(){const q=document.getElementById('qsh-input').value;if(!q){alert('Enter a query');return}try{const r=await fetch('/api/qsh/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});const d=await r.json();document.getElementById('qsh-output').innerHTML=`<div>Query: ${d.query}</div><div>QSH Hash: ${d.qsh_hash}</div><div>Classical: ${d.classical_hash}</div><div>Entanglement: ${d.entanglement_strength}</div><div>Energy: ${d.collision_energy_gev} GeV</div><div>Particle States: ${d.particle_states_generated}</div><div>Foam: ${d.foam_perturbations}</div><div>Decoherence: ${d.decoherence_time_ns} ns</div>`}catch(e){console.error('Error:',e)}}
async function getRFMetrics(){const m=document.getElementById('rf-mode').value;try{const r=await fetch(`/api/rf-metrics?mode=${m}`);const d=await r.json();let h='<div class="network-node">';for(const[k,v]of Object.entries(d)){h+=`<div class="metric">${k}: <span class="metric-value">${v}</span></div>`}h+='</div>';document.getElementById('rf-output').innerHTML=h}catch(e){console.error('Error:',e)}}
async function getSystemMetrics(){try{const r=await fetch('/api/metrics');const d=await r.json();let h='';for(const[s,m]of Object.entries(d)){if(m&&typeof m==='object'){h+=`<div class="network-node"><h4>${s}</h4>`;for(const[k,v]of Object.entries(m)){h+=`<div class="metric">${k}: <span class="metric-value">${JSON.stringify(v)}</span></div>`}h+='</div>'}}document.getElementById('system-metrics').innerHTML=h}catch(e){console.error('Error:',e)}}
async function getPeers(){try{const r=await fetch('/api/p2p/peers');const d=await r.json();const h=d.peers.map(p=>`<div class="network-node">${p}</div>`).join('');document.getElementById('peers-list').innerHTML=h}catch(e){console.error('Error:',e)}}
async function broadcastMessage(){const m=document.getElementById('broadcast-message').value;if(!m){alert('Enter a message');return}try{const r=await fetch('/api/p2p/broadcast',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});const d=await r.json();document.getElementById('broadcast-result').innerHTML=`<div class="network-node">✓ Broadcast to ${d.recipients} peers</div>`;document.getElementById('broadcast-message').value=''}catch(e){console.error('Error:',e)}}
setInterval(refreshQuantumState,10000);refreshQuantumState();
</script>
</body>
</html>"""

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root_page():
    return HTMLResponse(HTML_TEMPLATE)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "modules": {
            "quantum_core": quantum_core.is_healthy(),
            "holo_storage": holo_storage.is_healthy(),
            "p2p_network": p2p_network.is_connected(),
        },
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

@app.get("/api/quantum/state")
async def quantum_state():
    cached = await cache_manager.get("quantum_state")
    if cached:
        return cached
    state = {
        "active_entanglements": await quantum_core.count_active_pairs(),
        "average_fidelity": await quantum_core.get_average_fidelity(),
        "epr_generation_rate": Config.EPR_RATE,
        "decoherence_events": await quantum_core.get_decoherence_count(),
        "foam_density": await quantum_core.get_foam_density(),
        "timestamp": datetime.now().isoformat()
    }
    await cache_manager.set("quantum_state", state, ttl=10)
    return state

@app.post("/api/qsh/query")
async def qsh_query(request: Request, client_ip: str = Depends(verify_rate_limit)):
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
        result = await qsh_engine.process_query(query)
        if analytics:
            await analytics.log_event("qsh_query", {"query_length": len(query)})
        return result
    except Exception as e:
        logger.error(f"QSH error: {e}")
        raise HTTPException(status_code=500, detail="QSH query failed")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks(), client_ip: str = Depends(verify_rate_limit)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        content = await file.read()
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large")
        if not await security_manager.scan_file(content):
            raise HTTPException(status_code=400, detail="Security scan failed")
        file_hash = hashlib.sha256(content).hexdigest()
        quantum_signature = await quantum_core.sign_data(file_hash)
        storage_result = await holo_storage.store_file(filename=file.filename, content=content, quantum_signature=quantum_signature)
        if analytics:
            background_tasks.add_task(analytics.log_event, "file_uploaded", {"size": len(content), "ip": client_ip})
        return {
            "success": True,
            "filename": file.filename,
            "file_hash": file_hash,
            "size_bytes": len(content),
            "quantum_signature": quantum_signature,
            "routing": storage_result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get("/api/files")
async def list_files(limit: int = 50, offset: int = 0, sort_by: str = "name"):
    try:
        files = await holo_storage.list_files(limit=limit, offset=offset, sort_by=sort_by)
        return {"files": files, "total": len(files), "limit": limit, "offset": offset}
    except Exception as e:
        logger.error(f"File listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")

@app.get("/api/download/{file_id}")
async def download_file(file_id: str, client_ip: str = Depends(verify_rate_limit)):
    try:
        file_data = await holo_storage.retrieve_file(file_id)
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")
        if not await quantum_core.verify_signature(file_data["hash"], file_data["quantum_signature"]):
            logger.warning(f"Signature verification failed for {file_id}")
        if analytics:
            await analytics.log_event("file_downloaded", {"file_id": file_id})
        return StreamingResponse(
            io.BytesIO(file_data["content"]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={file_data['filename']}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

@app.get("/api/rf-metrics")
async def rf_metrics(mode: str = "quantum", interface: str = "wlan0"):
    try:
        metrics = await rf_simulator.get_metrics(mode.lower(), interface)
        return metrics
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    except Exception as e:
        logger.error(f"RF metrics error: {e}")
        raise HTTPException(status_code=500, detail="RF metrics unavailable")

@app.get("/api/metrics")
async def system_metrics():
    cached = await cache_manager.get("system_metrics")
    if cached:
        return cached
    try:
        metrics = {
            "quantum": await quantum_state(),
            "network": await p2p_network.get_stats(),
            "storage": await holo_storage.get_stats(),
            "analytics": await analytics.get_summary() if analytics else None,
            "timestamp": datetime.now().isoformat()
        }
        await cache_manager.set("system_metrics", metrics, ttl=5)
        return metrics
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

@app.get("/api/p2p/peers")
async def get_peers():
    try:
        peers = await p2p_network.get_peers()
        return {"peers": peers, "count": len(peers), "dht_enabled": Config.DHT_ENABLED}
    except Exception as e:
        logger.error(f"Peer listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get peers")

@app.post("/api/p2p/broadcast")
async def broadcast_message(request: Request, client_ip: str = Depends(verify_rate_limit)):
    try:
        data = await request.json()
        message = data.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="Message required")
        result = await p2p_network.broadcast(message)
        return {"success": True, "recipients": result.get("peer_count", 0), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        raise HTTPException(status_code=500, detail="Broadcast failed")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "status_code": exc.status_code, "timestamp": datetime.now().isoformat()})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
