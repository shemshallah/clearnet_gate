from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import hashlib
import logging
import os
import secrets
import time
import random
from pathlib import Path
import socket
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path("uploads")
TEMPLATES_DIR = Path("templates")
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Quantum Foam Network",
    description="Advanced quantum-inspired networking platform",
    version="2.0.0"
)

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
files_db = {}
quantum_state = {
    "active_pairs": 1500,
    "fidelity": 0.985,
    "epr_rate": 3000,
    "decoherence_count": 42,
    "foam_density": 1.523
}

# EPR DNS Routing Configuration (from Alice node)
EPR_DNS = {
    "alice": {
        "node_id": "ALICE-QFN-01",
        "entangled_domain": "clearnet_chat.onrender.com",
        "routing_protocol": "EPR-DNS-v1",
        "fidelity": 0.998,
        "latency_ns": round(random.uniform(1.2, 5.0), 2)
    }
}

# Helper functions
def generate_quantum_hash(data: str) -> dict:
    """Generate quantum-inspired hash"""
    classical_hash = hashlib.sha256(data.encode()).hexdigest()
    qsh_hash = ''.join(random.choices('0123456789abcdef', k=64))
    return {
        "query": data,
        "qsh_hash": qsh_hash,
        "classical_hash": classical_hash,
        "entanglement_strength": round(random.uniform(0.85, 0.99), 3),
        "collision_energy_gev": round(random.uniform(1000, 13000), 0),
        "particle_states_generated": random.randint(100, 1000),
        "foam_perturbations": random.randint(50, 500),
        "decoherence_time_ns": round(random.uniform(10, 100), 2),
        "timestamp": datetime.now().isoformat()
    }

def get_network_stats() -> dict:
    """Get network statistics"""
    try:
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent_gb": round(net_io.bytes_sent / (1024**3), 3),
            "bytes_recv_gb": round(net_io.bytes_recv / (1024**3), 3),
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
    except:
        return {
            "bytes_sent_gb": 0.0,
            "bytes_recv_gb": 0.0,
            "packets_sent": 0,
            "packets_recv": 0
        }

def resolve_epr_dns(node: str = "alice") -> dict:
    """Resolve EPR DNS routing from specified node"""
    if node in EPR_DNS:
        routing = EPR_DNS[node].copy()
        routing["resolved_domain"] = EPR_DNS[node]["entangled_domain"]
        routing["timestamp"] = datetime.now().isoformat()
        return routing
    else:
        raise HTTPException(status_code=404, detail=f"EPR DNS node '{node}' not found")

# Routes
@app.get("/")
async def root():
    """Redirect root to clearnet chat service via EPR routing"""
    return RedirectResponse(url="https://clearnet_chat.onrender.com", status_code=302)

@app.get("/collider")
async def collider_page(request: Request):
    """Serve collider page with dynamic data"""
    collider_data = {
        "black_hole": {
            "address": "138.0.0.100",
            "mass_solar": 4.31e6,
            "event_horizon_km": 1.27e7,
            "status": "ACTIVE"
        },
        "white_hole": {
            "address": "138.0.0.200",
            "outflow_rate": 3.84e33,
            "status": "EMITTING"
        },
        "interface": {
            "qsh_link": "ESTABLISHED",
            "data_rate_gbps": 1.23e15,
            "entanglement_pairs": 47283
        },
        "energies_ev": [random.uniform(100, 13000) for _ in range(10)],  # Sample for chart
        "timestamp": datetime.now().isoformat()
    }
    return templates.TemplateResponse("collider.html", {"request": request, "collider": collider_data})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

@app.get("/api/quantum/state")
async def get_quantum_state():
    """Get quantum system state"""
    return {
        "active_entanglements": quantum_state["active_pairs"],
        "average_fidelity": quantum_state["fidelity"],
        "epr_generation_rate": quantum_state["epr_rate"],
        "decoherence_events": quantum_state["decoherence_count"],
        "foam_density": quantum_state["foam_density"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/quantum/entangle")
async def create_entanglement(request: Request):
    """Generate EPR pairs"""
    data = await request.json()
    num_pairs = data.get("num_pairs", 10)
    
    pairs = [
        {"id": i, "fidelity": round(random.uniform(0.95, 0.999), 3)}
        for i in range(num_pairs)
    ]
    
    return {
        "success": True,
        "pairs_generated": len(pairs),
        "average_fidelity": sum(p["fidelity"] for p in pairs) / len(pairs),
        "pairs": pairs[:5],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/qsh-query")
async def qsh_query(request: Request):
    """Quantum State Hasher query"""
    data = await request.json()
    query = data.get("query", "").strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    
    result = generate_quantum_hash(query)
    return result

@app.get("/api/network/scan")
async def scan_network():
    """Scan network topology with EPR DNS integration"""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        hostname = "unknown"
        local_ip = "127.0.0.1"
    
    epr_routing = resolve_epr_dns("alice")
    
    return {
        "hostname": hostname,
        "local_ip": local_ip,
        "quantum_ip": f"138.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "epr_dns_routing": epr_routing,
        "stats": get_network_stats(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/network/tcp_proxy")
async def tcp_proxy_status(request: Request):
    """Get TCP proxy status with EPR routing"""
    client_ip = request.client.host if request.client else "unknown"
    epr_routing = resolve_epr_dns("alice")
    return {
        "user_ip": client_ip,
        "quantum_ip": f"138.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "proxy_ip": "138.0.0.1",
        "epr_routing": epr_routing["resolved_domain"],
        "entangled": True,
        "latency_ms": round(random.uniform(5, 50), 2)
    }

@app.post("/api/network/request_node")
async def request_node(request: Request):
    """Request network node assignment with EPR DNS"""
    data = await request.json()
    route = data.get("route", "")
    email = data.get("email", "")
    epr_routing = resolve_epr_dns("alice")
    
    return {
        "success": True,
        "assigned_qip": f"Q{secrets.token_hex(8).upper()}",
        "route": route,
        "email": email,
        "epr_dns_target": epr_routing["resolved_domain"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/epr/dns/resolve/{node}")
async def epr_dns_resolve(node: str):
    """Explicit EPR DNS resolution endpoint"""
    return resolve_epr_dns(node)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file to quantum storage"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    file_id = secrets.token_hex(16)
    
    # Store file info
    files_db[file_id] = {
        "id": file_id,
        "filename": file.filename,
        "size": len(content),
        "hash": file_hash,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to disk
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    file_path.write_bytes(content)
    
    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "file_hash": file_hash,
        "size_bytes": len(content),
        "quantum_route": f"Q{secrets.token_hex(8).upper()}",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/files/list")
async def list_files(page: int = 1, limit: int = 50):
    """List uploaded files"""
    files = list(files_db.values())
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "files": [
            {
                "id": f["id"],
                "filename": f["filename"],
                "size_gb": f["size"] / (1024**3),
                "timestamp": f["timestamp"]
            }
            for f in files[start:end]
        ],
        "total": len(files),
        "page": page,
        "total_pages": (len(files) + limit - 1) // limit if files else 1
    }

@app.get("/api/files/download/{file_id}")
async def download_file(file_id: str):
    """Download file"""
    if file_id not in files_db:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_data = files_db[file_id]
    
    return StreamingResponse(
        iter([file_data["content"]]),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{file_data["filename"]}"'
        }
    )

@app.get("/api/btc/status")
async def btc_status():
    """Bitcoin mirror status"""
    return {
        "blockchain_height": random.randint(800000, 850000),
        "mempool_size": round(random.uniform(50, 300), 1),
        "price_usd": round(random.uniform(40000, 70000), 2),
        "connected_peers": random.randint(8, 125),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/btc/wallet/generate")
async def generate_btc_wallet():
    """Generate Bitcoin wallet address"""
    address = "1" + "".join(random.choices("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz", k=33))
    return {
        "address": address,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/email/send")
async def send_email(request: Request):
    """Send quantum email"""
    data = await request.json()
    return {
        "success": True,
        "to": data.get("to"),
        "subject": data.get("subject"),
        "quantum_signature": secrets.token_hex(16),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/qsh/execute")
async def execute_qsh_command(request: Request):
    """Execute QSH shell command"""
    data = await request.json()
    command = data.get("command", "")
    
    if command == "help":
        output = "Available commands: help, status, entangle, scan"
    elif command == "status":
        output = f"Quantum Core: Active\nFidelity: {quantum_state['fidelity']}\nEPR Rate: {quantum_state['epr_rate']}"
    else:
        output = f"Executing: {command}\nQuantum shell v1.0 ready."
    
    return {"output": output}

@app.get("/api/collider/status")
async def collider_status():
    """Get collider system status"""
    return {
        "black_hole": {
            "address": "138.0.0.100",
            "mass_solar": 4.31e6,
            "event_horizon_km": 1.27e7,
            "status": "ACTIVE"
        },
        "white_hole": {
            "address": "138.0.0.200",
            "outflow_rate": 3.84e33,
            "status": "EMITTING"
        },
        "interface": {
            "qsh_link": "ESTABLISHED",
            "data_rate_gbps": 1.23e15,
            "entanglement_pairs": 47283
        }
    }

@app.get("/api/collider/spectrum")
async def collider_spectrum():
    """Get collision spectrum data"""
    energies = [random.uniform(100, 13000) for _ in range(100)]
    return {
        "format": "JSON",
        "energies_ev": energies,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/metrics")
async def system_metrics():
    """Get comprehensive system metrics"""
    return {
        "quantum": get_quantum_state(),
        "network": get_network_stats(),
        "storage": {
            "total_files": len(files_db),
            "total_size_gb": sum(f["size"] for f in files_db.values()) / (1024**3) if files_db else 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Initialize app state on startup"""
    app.state.start_time = time.time()
    logger.info("⚡ Quantum Foam Network started successfully")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions"""
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
