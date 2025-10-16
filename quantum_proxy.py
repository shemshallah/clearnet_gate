from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
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
import requests
from io import BytesIO

app = FastAPI(title="Quantum File Network (QFN)")

# Directory for uploads (now unused, kept for config compatibility)
UPLOAD_DIR = "/opt/render/project/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Config file
CONFIG_FILE = "config.json"

# Mount static files if needed (not used here)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

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
@app.get("/")
async def root_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. Metrics Page (/metrics)
@app.get("/metrics")
async def metrics_page(request: Request):
    return templates.TemplateResponse("metrics.html", {"request": request})

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
@app.get("/shell")
async def shell_page(request: Request):
    return templates.TemplateResponse("shell.html", {"request": request})

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
@app.get("/wireshark")
async def wireshark_page(request: Request):
    return templates.TemplateResponse("wireshark.html", {"request": request})

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
@app.get("/files")
async def files_page(request: Request):
    return templates.TemplateResponse("files.html", {"request": request})

# API for files list with routing (now proxies to holo storage)
@app.get("/api/files-with-routing")
async def files_with_routing():
    holo_url = f"http://{config['holo_storage_ip']}/api/files-with-routing"
    try:
        response = requests.get(holo_url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except requests.exceptions.RequestException:
        return []

# API for upload (now proxies to holo storage)
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    holo_url = f"http://{config['holo_storage_ip']}/api/upload"
    files = {"file": (file.filename, content, file.content_type)}
    try:
        response = requests.post(holo_url, files=files, timeout=30)
        if response.status_code == 200:
            return {"message": f"File {file.filename} uploaded to holo storage successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Upload to holo storage failed: {response.text}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Connection to holo storage failed: {str(e)}")

# API for download (now proxies from holo storage)
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    holo_url = f"http://{config['holo_storage_ip']}/api/download/{filename}"
    try:
        response = requests.get(holo_url, timeout=10)
        if response.status_code == 200:
            return FileResponse(BytesIO(response.content), filename=filename, media_type="application/octet-stream")
        else:
            raise HTTPException(status_code=404, detail="File not found on holo storage")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Connection to holo storage failed: {str(e)}")

# 6. Collider Page (/collider)
@app.get("/collider")
async def collider_page(request: Request):
    return templates.TemplateResponse("collider.html", {"request": request})

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
@app.get("/config")
async def config_page(request: Request):
    return templates.TemplateResponse("config.html", {"request": request})

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
