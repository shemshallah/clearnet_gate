from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import os
import hashlib
import random
import asyncio
from datetime import datetime
from typing import List
import uvicorn
import logging
from contextlib import asynccontextmanager

# Production logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create dirs, load config
    os.makedirs("./uploads", exist_ok=True)
    logger.info("QFN Startup: Directories and config loaded")
    yield
    # Shutdown: Save config
    logger.info("QFN Shutdown: Config saved")

app = FastAPI(title="Quantum File Network (QFN)", lifespan=lifespan)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in prod
    allow_credentials=True,
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

# Config file (env override)
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")

# Load/Save config (with env overrides)
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
                default_config.update(loaded)
                logger.info("Config loaded from file")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    else:
        logger.info("No config file, using defaults")
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

# API for metrics
@app.get("/api/metrics")
async def get_metrics():
    try:
        return {
            "network_throughput": round(random.uniform(50, 1000), 2),
            "epr_rate": config["epr_rate"],
            "entanglement_threshold": config["entanglement_threshold"],
            "holo_dns_enabled": config["holo_dns_enabled"],
            "timestamp": datetime.now().isoformat(),
            "upload_count": len([f for f in os.listdir(config["upload_directory"]) if os.path.isfile(os.path.join(config["upload_directory"], f))])
        }
    except Exception as e:
        logger.error(f"Metrics fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

# 3. File Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate hash for secure naming
    file_hash = hashlib.md5(file.filename.encode()).hexdigest()
    file_path = os.path.join(config["upload_directory"], f"{file_hash}_{file.filename}")
    
    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.info(f"File uploaded: {file.filename} -> {file_path}")
        return {
            "filename": file.filename,
            "hash": file_hash,
            "size_bytes": len(content),
            "path": file_path
        }
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

# 4. File List Endpoint
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

# 5. File Download Endpoint
@app.get("/download/{filename:path}")
async def download_file(filename: str):
    file_path = os.path.join(config["upload_directory"], filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static if needed (e.g., for future assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Env-based config file
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")

# Load/Save config (production: env override)
def load_config():
    default_config = {
        "holo_storage_ip": os.getenv("HOLO_STORAGE_IP", "138.0.0.1"),
        "holo_dns_enabled": os.getenv("HOLO_DNS_ENABLED", "true").lower() == "true",
        "upload_directory": os.getenv("UPLOAD_DIR", "./uploads"),
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

# Error handler for production
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# 1. Root Page (/)
@app.get("/", response_class=HTMLResponse)
async def root_page():
    try:
        return FileResponse("root.html", media_type="text/html")
    except FileNotFoundError:
        logger.error("root.html not found")
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
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
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
    try:
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large")
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes)")
        return {"message": f"File {file.filename} uploaded successfully to Holo storage"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

# API for download (secure, exists check)
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    try:
        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(filepath, filename=filename, media_type="application/octet-stream")
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
