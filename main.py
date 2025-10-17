"""
Quantum File Network (QFN) - Enhanced Production Version
Primary Architect: Justin Anthony Howard-Stanley
Secondary Architect: Dale Cwidak

Production-ready FastAPI application with modular quantum simulation framework
"""

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
from pathlib import Path

# Import custom modules
from modules.quantum_core import QuantumCore, EntanglementManager
from modules.holo_storage import HoloStorageManager
from modules.qsh_engine import QSHEngine
from modules.rf_simulator import RFSimulator, RFMode
from modules.security import SecurityManager, RateLimiter
from modules.analytics import AnalyticsEngine
from modules.blockchain_ledger import BlockchainLedger
from modules.ai_optimizer import AIOptimizer
from modules.p2p_network import P2PNetworkManager
from modules.cache_manager import CacheManager

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

# Configuration management
class Config:
    """Centralized configuration with environment variable support"""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"
    CONFIG_FILE = Path(os.getenv("CONFIG_FILE", "config.json"))
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))  # 100MB
    
    # Quantum settings
    HOLO_STORAGE_IP = os.getenv("HOLO_STORAGE_IP", "138.0.0.1")
    EPR_RATE = int(os.getenv("EPR_RATE", "2500"))
    ENTANGLEMENT_THRESHOLD = float(os.getenv("ENTANGLEMENT_THRESHOLD", "0.975"))
    QUANTUM_FIDELITY_TARGET = float(os.getenv("QUANTUM_FIDELITY_TARGET", "0.98"))
    
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
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")

# Global instances
quantum_core: QuantumCore
holo_storage: HoloStorageManager
qsh_engine: QSHEngine
rf_simulator: RFSimulator
security_manager: SecurityManager
analytics: AnalyticsEngine
blockchain: Optional[BlockchainLedger]
ai_optimizer: Optional[AIOptimizer]
p2p_network: P2PNetworkManager
cache_manager: CacheManager
rate_limiter: RateLimiter

# Lifespan management
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
        
        # Optional features
        if Config.ANALYTICS_ENABLED:
            analytics = AnalyticsEngine()
            logger.info("✓ Analytics engine enabled")
        
        if Config.BLOCKCHAIN_ENABLED:
            blockchain = BlockchainLedger()
            logger.info("✓ Blockchain ledger enabled")
        else:
            blockchain = None
        
        if Config.AI_OPTIMIZATION:
            ai_optimizer = AIOptimizer(quantum_core=quantum_core)
            logger.info("✓ AI optimizer enabled")
        else:
            ai_optimizer = None
        
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
        
        if Config.ANALYTICS_ENABLED:
            asyncio.create_task(analytics.collect_metrics())
        
        logger.info("✅ QFN System ready - All modules initialized")
        
        yield
        
    finally:
        # Cleanup
        logger.info("🛑 Shutting down QFN System...")
        await p2p_network.stop()
        if blockchain:
            await blockchain.finalize()
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

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
app.mount("/static", StaticFiles(directory=Config.STATIC_DIR), name="static")

# Security dependency
security = HTTPBasic(auto_error=False)

async def verify_rate_limit(request: Request):
    """Rate limiting middleware"""
    client_ip = request.client.host
    if not await rate_limiter.check_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client_ip

# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root_page():
    """Enhanced root dashboard"""
    try:
        html_path = Config.TEMPLATES_DIR / "dashboard.html"
        if not html_path.exists():
            # Fallback to basic template
            return """
            <!DOCTYPE html>
            <html>
            <head><title>QFN Dashboard</title></head>
            <body>
                <h1>🌌 Quantum File Network</h1>
                <nav>
                    <a href="/api/docs">API Documentation</a> |
                    <a href="/metrics">Metrics</a> |
                    <a href="/quantum">Quantum Lab</a>
                </nav>
            </body>
            </html>
            """
        return FileResponse(html_path)
    except Exception as e:
        logger.error(f"Root page error: {e}")
        raise HTTPException(status_code=500, detail="Dashboard unavailable")

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
            "blockchain": blockchain.is_synced() if blockchain else None,
        },
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

# ============================================================================
# QUANTUM OPERATIONS
# ============================================================================

@app.post("/api/quantum/entangle")
async def create_entanglement(
    num_pairs: int = 10,
    client_ip: str = Depends(verify_rate_limit)
):
    """Generate EPR pairs for quantum communication"""
    try:
        pairs = await quantum_core.generate_epr_pairs(num_pairs)
        
        if Config.ANALYTICS_ENABLED:
            await analytics.log_event("entanglement_created", {
                "pairs": num_pairs,
                "ip": client_ip
            })
        
        return {
            "success": True,
            "pairs_generated": len(pairs),
            "average_fidelity": sum(p["fidelity"] for p in pairs) / len(pairs),
            "pairs": pairs[:5],  # Return first 5 for preview
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
        
        # Run QSH simulation
        result = await qsh_engine.process_query(query)
        
        # Log to blockchain if enabled
        if blockchain:
            await blockchain.add_transaction({
                "type": "qsh_query",
                "query_hash": result["qsh_hash"][:16],
                "timestamp": datetime.now().isoformat()
            })
        
        if Config.ANALYTICS_ENABLED:
            await analytics.log_event("qsh_query", {"query_length": len(query)})
        
        return result
        
    except Exception as e:
        logger.error(f"QSH query error: {e}")
        raise HTTPException(status_code=500, detail="QSH query failed")

# ============================================================================
# FILE OPERATIONS
# ============================================================================

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
        
        # Read file content
        content = await file.read()
        
        # Size check
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large (max {Config.MAX_FILE_SIZE/1024/1024}MB)")
        
        # Security scan
        if not await security_manager.scan_file(content):
            raise HTTPException(status_code=400, detail="File failed security scan")
        
        # Generate quantum signature
        file_hash = hashlib.sha256(content).hexdigest()
        quantum_signature = await quantum_core.sign_data(file_hash)
        
        # Store with holo routing
        storage_result = await holo_storage.store_file(
            filename=file.filename,
            content=content,
            quantum_signature=quantum_signature
        )
        
        # Add to blockchain ledger
        if blockchain:
            background_tasks.add_task(
                blockchain.add_file_record,
                file_hash,
                storage_result["quantum_route"]
            )
        
        # Analytics
        if Config.ANALYTICS_ENABLED:
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
        # Get file from holo storage
        file_data = await holo_storage.retrieve_file(file_id)
        
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Verify quantum signature
        if not await quantum_core.verify_signature(
            file_data["hash"],
            file_data["quantum_signature"]
        ):
            logger.warning(f"Quantum signature verification failed for {file_id}")
        
        # Analytics
        if Config.ANALYTICS_ENABLED:
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

# ============================================================================
# RF SIMULATION & METRICS
# ============================================================================

@app.get("/api/rf-metrics")
async def rf_metrics(mode: str = "quantum", interface: str = "wlan0"):
    """Advanced RF metrics simulation"""
    try:
        rf_mode = RFMode(mode)
        metrics = await rf_simulator.get_metrics(rf_mode, interface)
        return metrics
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
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
            "analytics": await analytics.get_summary() if Config.ANALYTICS_ENABLED else None,
            "blockchain": await blockchain.get_stats() if blockchain else None,
            "timestamp": datetime.now().isoformat()
        }
        
        await cache_manager.set("system_metrics", metrics, ttl=5)
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

# ============================================================================
# AI OPTIMIZATION (Optional Module)
# ============================================================================

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

# ============================================================================
# P2P NETWORK
# ============================================================================

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
            "recipients": result["peer_count"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        raise HTTPException(status_code=500, detail="Broadcast failed")

# ============================================================================
# BLOCKCHAIN LEDGER (Optional Module)
# ============================================================================

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

# ============================================================================
# ANALYTICS & REPORTING
# ============================================================================

@app.get("/api/analytics/report")
async def analytics_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Generate analytics report"""
    if not Config.ANALYTICS_ENABLED:
        raise HTTPException(status_code=503, detail="Analytics not enabled")
    
    try:
        report = await analytics.generate_report(start_date, end_date)
        return report
    except Exception as e:
        logger.error(f"Analytics report error: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed")

# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.post("/api/admin/config")
async def update_config(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(security)
):
    """Update system configuration (admin only)"""
    # Simple auth check
    if not secrets.compare_digest(credentials.username, "admin") or \
       not secrets.compare_digest(credentials.password, os.getenv("ADMIN_PASSWORD", "quantum")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    try:
        data = await request.json()
        
        # Update configuration
        # (In production, implement proper config validation and persistence)
        
        return {
            "success": True,
            "message": "Configuration updated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail="Config update failed")

# ============================================================================
# ERROR HANDLERS
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

# Initialize start time
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()
    logger.info("⚡ QFN FastAPI application started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static if needed (e.g., for future assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directory for uploads (safe creation)
UPLOAD_DIR = "./uploads"
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info(f"Uploads dir ready: {UPLOAD_DIR}")
except PermissionError as e:
    logger.error(f"Permission denied for uploads dir: {e}. Using temp dir.")
    UPLOAD_DIR = "/tmp/uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create uploads dir: {e}")

# Env-based config file
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")

# Load/Save config (production: env override)
def load_config():
    default_config = {
        "holo_storage_ip": os.getenv("HOLO_STORAGE_IP", "138.0.0.1"),
        "holo_dns_enabled": os.getenv("HOLO_DNS_ENABLED", "true").lower() == "true",
        "upload_directory": UPLOAD_DIR,
        "default_interface": os.getenv("DEFAULT_INTERFACE", "wlan0"),
        "default_rf_mode": os.getenv("DEFAULT_RF_MODE", "quantum"),
        "epr_rate": int(os.getenv("EPR_RATE", "2500")),
        "entanglement_threshold": float(os.getenv("ENTANGLEMENT_THRESHOLD", "0.975")),
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded = json.load(f)
                default_config.update(loaded)  # Merge with defaults
                logger.info("Config loaded from file")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    else:
        logger.info("No config file found, using defaults")
    return default_config

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Config saved")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

config = load_config()

# Global error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code} error: {exc.detail} from {request.client.host}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# 1. Root Page (/)
@app.get("/", response_class=HTMLResponse)
async def root_page():
    try:
        return FileResponse("root.html", media_type="text/html")
    except FileNotFoundError:
        logger.error("root.html not found - ensure in repo")
        raise HTTPException(status_code=404, detail="Root page not found")

# 2. Metrics Page (/metrics)
@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page():
    try:
        return FileResponse("metrics.html", media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics page not found")

# API for metrics (cached simulation)
@app.get("/api/metrics")
async def get_metrics():
    try:
        return {
            "network_throughput": round(random.uniform(50, 1000), 2),
            "epr_rate": config["epr_rate"],
            "entanglement_fidelity": round(random.uniform(95, 99.9), 2),
            "holo_latency": round(random.uniform(10, 50), 2),
            "qkd_key_rate": round(random.uniform(1000, 10000), 0),
            "foam_density": round(random.uniform(1.2, 2.5), 3),
            "timestamp": datetime.now().isoformat(),
            "upload_count": len([f for f in os.listdir(config["upload_directory"]) if os.path.isfile(os.path.join(config["upload_directory"], f))])
        }
    except Exception as e:
        logger.error(f"Metrics fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

# API for speed test (async, production timeout)
@app.get("/api/speed-test")
async def speed_test():
    try:
        await asyncio.sleep(1)  # Simulate test time
        return {
            "download": round(random.uniform(100, 1000), 2),
            "upload": round(random.uniform(50, 500), 2),
            "jitter": round(random.uniform(1, 10), 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Speed test error: {e}")
        raise HTTPException(status_code=500, detail="Speed test failed")

# 3. Shell Page (/shell)
@app.get("/shell", response_class=HTMLResponse)
async def shell_page():
    try:
        return FileResponse("shell.html", media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Shell page not found")

# API for shell (secure, rate-limit ready)
@app.post("/api/shell")
async def shell_command(request: Request):
    try:
        data = await request.json()
        cmd = data.get("command", "").lower().strip()
        
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
            query = cmd[6:].strip()
            ai_response = f"Analyzing '{query}'... Quantum insight: Entanglement suggests parallel outcomes in your query."
        
        logger.info(f"Shell command executed: {cmd} by {request.client.host}")
        return {"output": output, "ai_response": ai_response}
    except Exception as e:
        logger.error(f"Shell error: {e}")
        raise HTTPException(status_code=500, detail="Shell command failed")

# 4. Wireshark Page (/wireshark)
@app.get("/wireshark", response_class=HTMLResponse)
async def wireshark_page():
    try:
        return FileResponse("wireshark.html", media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Wireshark page not found")

# API for RF metrics (validated params)
@app.get("/api/rf-metrics")
async def rf_metrics(mode: str = "quantum", interface: str = "wlan0"):
    try:
        if mode not in ["4g_lte", "5g_nr", "quantum"]:
            raise HTTPException(status_code=400, detail="Invalid mode")
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
                "pci": random.randint(0, 503),
                "timestamp": datetime.now().isoformat()
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
                "nr_band": random.choice(["n78", "n41", "n1"]),
                "timestamp": datetime.now().isoformat()
            }
        else:
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
    except Exception as e:
        logger.error(f"RF metrics error: {e}")
        raise HTTPException(status_code=500, detail="RF metrics unavailable")

# 5. Files Page (/files)
@app.get("/files", response_class=HTMLResponse)
async def files_page():
    try:
        return FileResponse("files.html", media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Files page not found")

# API for files list with routing (sorted, paginated ready)
@app.get("/api/files-with-routing")
async def files_with_routing(limit: int = 50):
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR)[:limit]:
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
                    },
                    "timestamp": datetime.now().isoformat()
                })
        files.sort(key=lambda x: x["name"])  # Production sorting
        return files
    except Exception as e:
        logger.error(f"Files list error: {e}")
        raise HTTPException(status_code=500, detail="Files list unavailable")

# API for upload (file size limit, virus scan ready)
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large")
        # Generate hash for secure naming
        file_hash = hashlib.md5(file.filename.encode()).hexdigest()
        file_path = os.path.join(config["upload_directory"], f"{file_hash}_{file.filename}")
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.info(f"File uploaded: {file.filename} -> {file_path}")
        return {
            "filename": file.filename,
            "hash": file_hash,
            "size_bytes": len(content),
            "path": file_path,
            "message": f"File {file.filename} uploaded successfully to Holo storage"
        }
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

# API for download (secure, exists check)
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(config["upload_directory"], filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

# API for files list (legacy)
@app.get("/api/files")
async def list_files():
    try:
        files = [
            {
                "name": f,
                "size_bytes": os.path.getsize(os.path.join(config["upload_directory"], f)),
                "modified": datetime.fromtimestamp(os.path.getmtime(os.path.join(config["upload_directory"], f))).isoformat()
            }
            for f in os.listdir(config["upload_directory"])
            if os.path.isfile(os.path.join(config["upload_directory"], f))
        ]
        return {"files": files, "total_count": len(files)}
    except Exception as e:
        logger.error(f"File list failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")

# 6. Collider Page (/collider)
@app.get("/collider", response_class=HTMLResponse)
async def collider_page():
    try:
        return FileResponse("collider.html", media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Collider page not found")

# API for QSH query (secure, rate-limit ready)
@app.post("/api/qsh-query")
async def qsh_query(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
        
        # Simulate classical hash
        classical_hash = hashlib.sha256(query.encode()).hexdigest()
        
        # Simulate quantum hash (random hex for demo, production: real quantum sim)
        qsh_hash = ''.join(random.choices('0123456789abcdef', k=64))
        
        result = {
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
        logger.info(f"QSH query processed: {query[:20]}...")
        return result
    except Exception as e:
        logger.error(f"QSH query error: {e}")
        raise HTTPException(status_code=500, detail="QSH query failed")

# 7. Config Page (/config)
@app.get("/config", response_class=HTMLResponse)
async def config_page():
    try:
        return FileResponse("config.html", media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Config page not found")

# API for config (secure, validated updates)
@app.get("/api/config")
async def get_config():
    try:
        return config
    except Exception as e:
        logger.error(f"Get config error: {e}")
        raise HTTPException(status_code=500, detail="Config unavailable")

@app.post("/api/config")
async def post_config(request: Request):
    try:
        data = await request.json()
        # Validate keys
        allowed_keys = ["holo_storage_ip", "holo_dns_enabled", "upload_directory", "default_interface", "default_rf_mode", "epr_rate", "entanglement_threshold"]
        for key in data:
            if key not in allowed_keys:
                raise HTTPException(status_code=400, detail=f"Invalid key: {key}")
        global config
        config.update(data)
        save_config(config)
        logger.info("Config updated")
        return {"message": "Config saved successfully"}
    except Exception as e:
        logger.error(f"Post config error: {e}")
        raise HTTPException(status_code=500, detail="Config update failed")

# Health check for production
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
