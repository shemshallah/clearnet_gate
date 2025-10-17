

from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
import random
from pathlib import Path

# --- STUB IMPLEMENTATIONS FOR MISSING MODULES ---
# These are minimal stubs to make the app runnable. In production, replace with real implementations.

class QuantumCore:
    def __init__(self, epr_rate: int = 3000, fidelity_target: float = 0.99):
        self.epr_rate = epr_rate
        self.fidelity_target = fidelity_target
        self.active_pairs = 0
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
            self.active_pairs = max(0, self.active_pairs - 1)

    async def sign_data(self, data: str):
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def verify_signature(self, data: str, signature: str):
        return hashlib.sha256(data.encode()).hexdigest()[:16] == signature

    def is_healthy(self):
        return True

class EntanglementManager:
    pass  # Not used directly

class HoloStorageManager:
    def __init__(self, storage_ip: str, upload_dir: Path):
        self.storage_ip = storage_ip
        self.upload_dir = upload_dir
        self.files = {}  # In-memory stub

    async def store_file(self, filename: str, content: bytes, quantum_signature: str):
        file_id = hashlib.md5(filename.encode()).hexdigest()
        self.files[file_id] = {
            "filename": filename,
            "content": content,
            "quantum_signature": quantum_signature,
            "hash": hashlib.sha256(content).hexdigest()
        }
        return {
            "quantum_route": f"Q{secrets.token_hex(8).upper()}",
            "holo_storage": self.storage_ip,
            "latency_ms": random.uniform(5, 50)
        }

    async def list_files(self, limit: int = 50, offset: int = 0, sort_by: str = "name"):
        file_list = list(self.files.values())[offset:offset + limit]
        return [{"id": k, **v} for k, v in list(self.files.items())[offset:offset + limit]]

    async def retrieve_file(self, file_id: str):
        if file_id in self.files:
            f = self.files[file_id]
            return f
        return None

    async def sync_nodes(self):
        while True:
            await asyncio.sleep(300)  # Every 5 min

    async def get_stats(self):
        return {"total_files": len(self.files), "total_size": sum(len(v["content"]) for v in self.files.values())}

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
    def __init__(self):
        pass

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
                "cqi": random.randint(1, 15),
                "mimo_layers": random.choice([2, 4]),
                "cell_id": random.randint(1, 1000),
                "pci": random.randint(0, 503),
                "timestamp": datetime.now().isoformat()
            }
        elif mode == "nr_5g":
            return {
                "mode": "nr_5g",
                "frequency_mhz": random.randint(2400, 40000),
                "bandwidth_mhz": random.choice([20, 40, 100, 400]),
                "modulation": random.choice(["QPSK", "16QAM", "64QAM", "256QAM"]),
                "rssi_dbm": round(random.uniform(-100, -50), 1),
                "sinr_db": round(random.uniform(5, 35), 1),
                "mimo_layers": random.choice([4, 8, 16]),
                "beam_index": random.randint(0, 63),
                "scs_khz": random.choice([15, 30, 60, 120]),
                "nr_band": random.choice(["n78", "n41", "n1"]),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise ValueError("Invalid mode")

class RFMode:
    def __init__(self, mode: str):
        if mode not in ["quantum", "lte_4g", "nr_5g", "wifi_6e", "satellite"]:
            raise ValueError("Invalid mode")
        self.value = mode

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    async def scan_file(self, content: bytes):
        return True  # Stub: always pass

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # In-memory

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

    async def generate_report(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        return {"summary": "Stub report", "events_count": len(self.events)}

class BlockchainLedger:
    def __init__(self):
        self.chain = [{"index": 0, "data": "genesis", "hash": "0"}]

    async def add_file_record(self, file_hash: str, route: str):
        pass  # Stub

    async def get_chain(self):
        return self.chain

    async def get_stats(self):
        return {"length": len(self.chain), "difficulty": 1}

    async def validate_chain(self):
        return True

    async def is_synced(self):
        return True

class AIOptimizer:
    def __init__(self, quantum_core: QuantumCore):
        self.quantum_core = quantum_core

    async def optimize_parameters(self, target_metric: str):
        return {
            "epr_rate": self.quantum_core.epr_rate + random.randint(-100, 100),
            "fidelity_target": self.quantum_core.fidelity_target + random.uniform(-0.01, 0.01),
            "improvement_percentage": random.uniform(0, 5)
        }

class P2PNetworkManager:
    def __init__(self, port: int, dht_enabled: bool):
        self.port = port
        self.dht_enabled = dht_enabled
        self.peers = ["192.168.1.100:9000", "192.168.1.101:9000"]  # Stub peers

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
            # Simple eviction: remove oldest
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        self.cache[key] = {"data": data, "timestamp": time.time()}

# --- CONFIGURATION ---
class Config:
    """Centralized configuration with environment variable support"""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"
    
    # Security (Pulled from docker-compose.yml environment variables)
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))  # 100MB
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "quantum")

    # Quantum settings
    HOLO_STORAGE_IP = os.getenv("HOLO_STORAGE_IP", "138.0.0.1")
    EPR_RATE = int(os.getenv("EPR_RATE", "3000"))
    ENTANGLEMENT_THRESHOLD = float(os.getenv("ENTANGLEMENT_THRESHOLD", "0.98"))
    QUANTUM_FIDELITY_TARGET = float(os.getenv("QUANTUM_FIDELITY_TARGET", "0.99"))
    
    # Network
    P2P_PORT = int(os.getenv("P2P_PORT", "9000"))
    DHT_ENABLED = os.getenv("DHT_ENABLED", "true").lower() == "true"
    
    # Features
    BLOCKCHAIN_ENABLED = os.getenv("BLOCKCHAIN_ENABLED", "true").lower() == "true"
    AI_OPTIMIZATION = os.getenv("AI_OPTIMIZATION", "true").lower() == "true"
    ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").lower() == "true"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.UPLOAD_DIR, cls.STATIC_DIR, cls.TEMPLATES_DIR]:
            directory.resolve().mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory.name}")

# --- GLOBAL INSTANCES & LIFESPAN MANAGEMENT ---
quantum_core: QuantumCore
holo_storage: HoloStorageManager
qsh_engine: QSHEngine
rf_simulator: RFSimulator
security_manager: SecurityManager
analytics: Optional[AnalyticsEngine]
blockchain: Optional[BlockchainLedger]
ai_optimizer: Optional[AIOptimizer]
p2p_network: P2PNetworkManager
cache_manager: CacheManager
rate_limiter: RateLimiter

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global quantum_core, holo_storage, qsh_engine, rf_simulator
    global security_manager, analytics, blockchain, ai_optimizer
    global p2p_network, cache_manager, rate_limiter
    
    logger.info("🚀 QFN System initializing...")
    
    try:
        # Create directories
        Config.create_directories()
        
        # Initialize core modules
        quantum_core = QuantumCore(
            epr_rate=Config.EPR_RATE,
            fidelity_target=Config.QUANTUM_FIDELITY_TARGET
        )
        holo_storage = HoloStorageManager(
            storage_ip=Config.HOLO_STORAGE_IP,
            upload_dir=Config.UPLOAD_DIR
        )
        qsh_engine = QSHEngine(quantum_core=quantum_core)
        rf_simulator = RFSimulator()
        
        # Security and rate limiting
        security_manager = SecurityManager(secret_key=Config.SECRET_KEY)
        rate_limiter = RateLimiter(
            max_requests=Config.RATE_LIMIT_REQUESTS,
            window_seconds=Config.RATE_LIMIT_WINDOW
        )
        
        # Cache manager
        cache_manager = CacheManager(max_size=1000, ttl=300)
        
        # Optional features initialization
        analytics = AnalyticsEngine() if Config.ANALYTICS_ENABLED else None
        blockchain = BlockchainLedger() if Config.BLOCKCHAIN_ENABLED else None
        ai_optimizer = AIOptimizer(quantum_core=quantum_core) if Config.AI_OPTIMIZATION else None
        
        if analytics: logger.info("✓ Analytics engine enabled")
        if blockchain: logger.info("✓ Blockchain ledger enabled")
        if ai_optimizer: logger.info("✓ AI optimizer enabled")
        
        # P2P network
        p2p_network = P2PNetworkManager(
            port=Config.P2P_PORT,
            dht_enabled=Config.DHT_ENABLED
        )
        await p2p_network.start()
        logger.info(f"✓ P2P network started on port {Config.P2P_PORT}")
        
        # Start background tasks
        asyncio.create_task(quantum_core.maintain_entanglement())
        asyncio.create_task(holo_storage.sync_nodes())
        if analytics:
            asyncio.create_task(analytics.collect_metrics())
        
        logger.info("✅ QFN System ready - All modules initialized")
        
        yield
        
    finally:
        # Cleanup
        logger.info("🛑 Shutting down QFN System...")
        await p2p_network.stop()
        if blockchain:
            # Placeholder for blockchain finalization
            pass
        logger.info("✅ Cleanup complete")

# Create FastAPI app
app = FastAPI(
    title="Quantum File Network (QFN)",
    description="Advanced quantum-inspired file sharing and networking platform",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=Config.STATIC_DIR), name="static")

# Security dependency
security = HTTPBasic(auto_error=False)

async def verify_rate_limit(request: Request):
    """Rate limiting middleware dependency"""
    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    if not await rate_limiter.check_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client_ip

# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root_page():
    """Enhanced root dashboard"""
    html_path = Config.TEMPLATES_DIR / "dashboard.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    
    # Fallback to a basic template if dashboard.html is missing
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>QFN Dashboard</title></head>
    <body>
        <h1>🌌 Quantum File Network</h1>
        <p>Dashboard template missing. See API docs.</p>
        <nav>
            <a href="/api/docs">API Documentation</a> |
            <a href="/health">Health Check</a>
        </nav>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "modules": {
            "quantum_core": quantum_core.is_healthy(),
            "holo_storage": holo_storage.is_healthy(),
            "p2p_network": p2p_network.is_connected(),
            "blockchain": blockchain.is_synced() if blockchain else "disabled",
        },
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

# --- QUANTUM OPERATIONS ---

@app.post("/api/quantum/entangle")
async def create_entanglement(
    num_pairs: int = 10,
    client_ip: str = Depends(verify_rate_limit)
):
    """Generate EPR pairs for quantum communication"""
    try:
        pairs = await quantum_core.generate_epr_pairs(num_pairs)
        
        if analytics:
            await analytics.log_event("entanglement_created", {"pairs": num_pairs, "ip": client_ip})
        
        return {
            "success": True,
            "pairs_generated": len(pairs),
            "average_fidelity": sum(p["fidelity"] for p in pairs) / len(pairs) if pairs else 0.0,
            "pairs": pairs[:5],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Entanglement creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quantum/state")
async def quantum_state():
    """Get current quantum network state"""
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
async def qsh_query(
    request: Request,
    client_ip: str = Depends(verify_rate_limit)
):
    """Quantum Secure Hash query - enhanced collision simulation"""
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
        
        result = await qsh_engine.process_query(query)
        
        if blockchain:
            # Placeholder for blockchain transaction
            pass
        
        if analytics:
            await analytics.log_event("qsh_query", {"query_length": len(query)})
        
        return result
        
    except Exception as e:
        logger.error(f"QSH query error: {e}")
        raise HTTPException(status_code=500, detail="QSH query failed")

# --- FILE OPERATIONS ---

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    client_ip: str = Depends(verify_rate_limit)
):
    """Enhanced file upload with quantum routing"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        content = await file.read()
        
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large (max {Config.MAX_FILE_SIZE/1024/1024}MB)")
        
        if not await security_manager.scan_file(content):
            raise HTTPException(status_code=400, detail="File failed security scan")
        
        file_hash = hashlib.sha256(content).hexdigest()
        quantum_signature = await quantum_core.sign_data(file_hash)
        
        storage_result = await holo_storage.store_file(
            filename=file.filename,
            content=content,
            quantum_signature=quantum_signature
        )
        
        if blockchain:
            background_tasks.add_task(
                blockchain.add_file_record,
                file_hash,
                storage_result["quantum_route"]
            )
        
        if analytics:
            background_tasks.add_task(
                analytics.log_event,
                "file_uploaded",
                {"size": len(content), "ip": client_ip}
            )
        
        return {
            "success": True,
            "filename": file.filename,
            "file_hash": file_hash,
            "size_bytes": len(content),
            "quantum_signature": quantum_signature,
            "routing": storage_result,
            "blockchain_pending": blockchain is not None,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get("/api/files")
async def list_files(
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "name"
):
    """List files with quantum routing information"""
    try:
        files = await holo_storage.list_files(limit=limit, offset=offset, sort_by=sort_by)
        return {
            "files": files,
            "total": len(files),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"File listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")

@app.get("/api/download/{file_id}")
async def download_file(
    file_id: str,
    client_ip: str = Depends(verify_rate_limit)
):
    """Download file with quantum verification"""
    try:
        file_data = await holo_storage.retrieve_file(file_id)
        
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")
        
        if not await quantum_core.verify_signature(
            file_data["hash"],
            file_data["quantum_signature"]
        ):
            logger.warning(f"Quantum signature verification failed for {file_id}")
        
        if analytics:
            await analytics.log_event("file_downloaded", {"file_id": file_id})
        
        return StreamingResponse(
            iter([file_data["content"]]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={file_data['filename']}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

# --- RF SIMULATION & METRICS ---

@app.get("/api/rf-metrics")
async def rf_metrics(mode: str = "quantum", interface: str = "wlan0"):
    """Advanced RF metrics simulation"""
    try:
        rf_mode = RFMode(mode.lower()) 
        metrics = await rf_simulator.get_metrics(rf_mode.value, interface)
        return metrics
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Must be one of: quantum, lte_4g, nr_5g, wifi_6e, satellite")
    except Exception as e:
        logger.error(f"RF metrics error: {e}")
        raise HTTPException(status_code=500, detail="RF metrics unavailable")

@app.get("/api/metrics")
async def system_metrics():
    """Comprehensive system metrics"""
    cached = await cache_manager.get("system_metrics")
    if cached:
        return cached
    
    try:
        metrics = {
            "quantum": await quantum_state(),
            "network": await p2p_network.get_stats(),
            "storage": await holo_storage.get_stats(),
            "analytics": await analytics.get_summary() if analytics else None,
            "blockchain": await blockchain.get_stats() if blockchain else None,
            "timestamp": datetime.now().isoformat()
        }
        
        await cache_manager.set("system_metrics", metrics, ttl=5)
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

# --- AI OPTIMIZATION (Optional Module) ---

@app.post("/api/ai/optimize")
async def ai_optimize(
    request: Request,
    client_ip: str = Depends(verify_rate_limit)
):
    """AI-powered quantum parameter optimization"""
    if not ai_optimizer:
        raise HTTPException(status_code=503, detail="AI optimization not enabled")
    
    try:
        data = await request.json()
        target_metric = data.get("target", "fidelity")
        
        recommendations = await ai_optimizer.optimize_parameters(target_metric)
        
        return {
            "success": True,
            "target_metric": target_metric,
            "recommendations": recommendations,
            "estimated_improvement": recommendations.get("improvement_percentage", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI optimization error: {e}")
        raise HTTPException(status_code=500, detail="Optimization failed")

# --- P2P NETWORK ---

@app.get("/api/p2p/peers")
async def get_peers():
    """Get connected peers"""
    try:
        peers = await p2p_network.get_peers()
        return {
            "peers": peers,
            "count": len(peers),
            "dht_enabled": Config.DHT_ENABLED
        }
    except Exception as e:
        logger.error(f"Peer listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get peers")

@app.post("/api/p2p/broadcast")
async def broadcast_message(
    request: Request,
    client_ip: str = Depends(verify_rate_limit)
):
    """Broadcast message to P2P network"""
    try:
        data = await request.json()
        message = data.get("message")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message required")
        
        result = await p2p_network.broadcast(message)
        
        return {
            "success": True,
            "recipients": result.get("peer_count", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        raise HTTPException(status_code=500, detail="Broadcast failed")

# --- BLOCKCHAIN LEDGER (Optional Module) ---

@app.get("/api/blockchain/chain")
async def get_blockchain():
    """Get blockchain ledger"""
    if not blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not enabled")
    
    try:
        chain = await blockchain.get_chain()
        return {
            "chain": chain,
            "length": len(chain),
            "is_valid": await blockchain.validate_chain()
        }
    except Exception as e:
        logger.error(f"Blockchain retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chain")

# --- ANALYTICS & REPORTING ---

@app.get("/api/analytics/report")
async def analytics_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Generate analytics report"""
    if not analytics:
        raise HTTPException(status_code=503, detail="Analytics not enabled")
    
    try:
        report = await analytics.generate_report(start_date, end_date)
        return report
    except Exception as e:
        logger.error(f"Analytics report error: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed")

# --- ADMIN ENDPOINTS ---

@app.post("/api/admin/config")
async def update_config(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(security)
):
    """Update system configuration (admin only)"""
    
    # Secure authentication check
    if not credentials or \
       not secrets.compare_digest(credentials.username, "admin") or \
       not secrets.compare_digest(credentials.password, Config.ADMIN_PASSWORD):
        response = JSONResponse(
            status_code=401, 
            content={"detail": "Invalid credentials"}
        )
        response.headers["WWW-Authenticate"] = "Basic"
        return response

    try:
        data = await request.json()
        logger.info(f"Admin received config update request: {data}")
        
        # NOTE: Actual config hot-reloading is complex; this is a placeholder.
        # In a real system, this would trigger a controlled restart or module reconfiguration.
        
        return {
            "success": True,
            "message": "Configuration request received. System may require restart for full changes.",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail="Config update failed")

# ============================================================================
# ERROR HANDLERS & STARTUP
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} from {request.client.host}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.on_event("startup")
async def startup_event():
    # Set app state for health check and uptime calculation
    app.state.start_time = time.time()
    logger.info("⚡ QFN FastAPI application started")

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qfn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    # This is for local development only. Production relies on Gunicorn/CMD in Dockerfile.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=True
    )
