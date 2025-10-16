from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import json
import os
import hashlib
import random
import asyncio
from datetime import datetime
from typing import List
import uvicorn

app = FastAPI(title="Quantum File Network (QFN)")

# Directory for uploads
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Config file
CONFIG_FILE = "config.json"

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
    return FileResponse("root.html", media_type="text/html")

# 2. Metrics Page (/metrics)
@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page():
    return FileResponse("metrics.html", media_type="text/html")

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
    return FileResponse("shell.html", media_type="text/html")

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
    return FileResponse("wireshark.html", media_type="text/html")

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
    return FileResponse("files.html", media_type="text/html")

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
    return FileResponse("collider.html", media_type="text/html")

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
    return FileResponse("config.html", media_type="text/html")

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
