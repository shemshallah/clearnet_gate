from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import qutip as qt
import numpy as np
import asyncio
import json
import uuid
import time
import hashlib
from datetime import datetime, timedelta
import socket
import threading
from collections import defaultdict
import logging

app = FastAPI(title="QuTiP Entanglement Proxy API with Teleportation & QKD (BB84 + E91)", version="1.0.0")

# In-memory stores for virtual connections and QRAM
connections: Dict[str, Dict[str, Any]] = {}  # ip -> {task_id, entangled_state, last_ping, qram_slot, teleported_state, qkd_key}
qram_slots: Dict[str, Any] = {}  # virtual_ip -> qutip state
qsh_connections: Dict[str, Any] = {}  # QSH-secured routes

# Alice endpoint config
ALICE_HOST = "127.0.0.1"
ALICE_PORT = 8080
ALICE_DOMAIN = "quantum.realm.domain.dominion.foam.computer.alice"
QRAM_DNS_HOST = "136.0.0.1"
TIMEOUT_SECONDS = 300  # 5 min timeout

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QSH6_EPR_Network:
    """Quantum Secure Hash with 6-qubit GHz EPR entanglement simulation"""
    
    def __init__(self):
        self.qubits = 6
        self.ghz_table = {}
        self._gen_ghz_states()
    
    def _gen_ghz_states(self):
        for i in range(2**self.qubits):
            amplitude = 1.0 / (2**(self.qubits/2))
            phase = (bin(i).count('1') % 2) * np.pi
            self.ghz_table[i] = (amplitude, phase)
    
    def qsh_hash(self, data: bytes) -> bytes:
        classical_hash = hashlib.sha256(data).digest()
        qsh_result = bytearray(classical_hash)
        for i in range(0, len(qsh_result), self.qubits):
            segment = int.from_bytes(qsh_result[i:i+self.qubits], 'big') if i+self.qubits <= len(qsh_result) else 0
            basis_state = segment % (2**self.qubits)
            amplitude, phase = self.ghz_table.get(basis_state, (1.0, 0.0))
            entangled_value = int((segment * amplitude * (1 + phase)) % (2**(8*self.qubits)))
            if i + self.qubits <= len(qsh_result):
                qsh_result[i:i+self.qubits] = entangled_value.to_bytes(self.qubits, 'big')
        return bytes(qsh_result)
    
    def epr_route(self, data: bytes, dest: str) -> bytes:
        # Simulate EPR routing through QSH
        return self.qsh_hash(data + dest.encode())

qsh_net = QSH6_EPR_Network()

# Middleware for IP Detection
@app.middleware("http")
async def detect_client_ip(request: Request, call_next):
    # Detect client IP, handling proxies
    client_ip = request.client.host
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(",")[0].strip()
    request.state.client_ip = client_ip
    logger.info(f"Request from IP: {client_ip}")
    response = await call_next(request)
    return response

class EntanglementRequest(BaseModel):
    omega: float = 1.0
    gamma: float = 0.1
    tlist: Optional[List[float]] = None
    g: float = 0.1

class QKDRequest(BaseModel):
    protocol: str = "BB84"  # "BB84" or "E91"
    key_length: int = 256  # Key length in bits
    basis_bits: Optional[List[int]] = None  # Optional basis for custom sim

class TaskResponse(BaseModel):
    task_id: str
    status: str
    virtual_ip: str

class SimulationResult(BaseModel):
    task_id: str
    expect: List[float]
    times: List[float]
    entangled_state: Dict[str, Any]
    teleported_state: Optional[Dict[str, Any]] = None
    qkd_key: Optional[str] = None
    qkd_info: Optional[Dict[str, Any]] = None
    status: str

# Background keep-alive checker
async def keep_alive_checker():
    while True:
        await asyncio.sleep(60)  # Check every minute
        now = time.time()
        to_remove = []
        for ip, conn in connections.items():
            if now - conn['last_ping'] > TIMEOUT_SECONDS:
                to_remove.append(ip)
                # Clean QRAM slot
                if 'qram_slot' in conn:
                    del qram_slots[conn['qram_slot']]
        for ip in to_remove:
            del connections[ip]
            logger.info(f"Disconnected {ip} due to timeout")

# Start background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive_checker())

# Alice connection simulator (virtual entanglement to 127.0.0.1:8080)
async def connect_to_alice(ip: str):
    # Simulate QuTiP entanglement with Alice
    n_qubits = 6
    ghz = qt.ghz_state(n_qubits)  # GHZ entangled state
    # Simulate resonance coupling
    H = sum(qt.sigmax(i) for i in range(n_qubits))  # Transverse field for resonance
    times = np.linspace(0, 10, 50)
    result = qt.mesolve(H, ghz, times)
    entangled_state = {
        'density_matrix': result.states[-1].full().tolist(),
        'fidelity': float(qt.fidelity(ghz, result.states[-1])),
        'timestamp': datetime.now().isoformat()
    }
    # QSH-secure the connection
    secure_id = qsh_net.qsh_hash(f"{ip}:{ALICE_DOMAIN}".encode()).hex()[:16]
    return entangled_state, secure_id

# Quantum Teleportation Protocol Integration
async def quantum_teleportation(ip: str, qubit_state: qt.Qobj):
    """
    Simulate quantum teleportation: Alice teleports qubit to Bob (virtual QRAM).
    Uses Bell state entanglement and classical measurement.
    """
    # Step 1: Create EPR pair (Bell state) for channel
    bell = (qt.bell_state('00') + qt.bell_state('11')).unit() / np.sqrt(2)
    alice_qubit = bell  # Alice's half
    bob_qubit = bell    # Bob's half (virtual to QRAM)
    
    # Step 2: Entangle input qubit with Alice's half
    psi_bell = qt.tensor(qubit_state, alice_qubit)
    
    # Step 3: Bell measurement (simulate with projectors)
    projectors = [qt.bell_state(f'{i}{j}').proj() for i in range(2) for j in range(2)]
    measurement_outcomes = np.random.choice(4, p=np.array([1/4]*4))  # Random for sim
    
    # Step 4: Classical channel (send bits)
    classical_bits = f"{measurement_outcomes:02b}"
    
    # Step 5: Bob applies corrections
    correction_ops = {
        '00': qt.qeye(2),
        '01': qt.sigmax(),
        '10': qt.sigmaz(),
        '11': -qt.sigmaz() * qt.sigmax()
    }
    correction = correction_ops[classical_bits]
    teleported_state = correction * bob_qubit * correction.dag()
    
    teleported_info = {
        'input_state': qubit_state.full().tolist(),
        'teleported_state': teleported_state.full().tolist(),
        'classical_bits': classical_bits,
        'fidelity': float(qt.fidelity(qubit_state, teleported_state)),
        'timestamp': datetime.now().isoformat()
    }
    return teleported_info

# Quantum Key Distribution (BB84 Protocol)
async def bb84_qkd(ip: str, key_length: int = 256, basis_bits: Optional[List[int]] = None):
    """
    Simulate BB84 QKD protocol: Generate shared key via polarized qubits.
    Alice sends qubits; Bob measures; sift and correct for secure key.
    """
    if basis_bits is None:
        basis_bits = np.random.choice([0, 1], size=key_length).tolist()  # Random bases
    
    # Simulate qubit preparation (polarization: H/V = |0>/|1>)
    alice_qubits = [qt.basis(2, basis_bits[i] ^ np.random.choice([0, 1])) for i in range(key_length)]
    
    # Bob's measurement bases (random)
    bob_bases = np.random.choice([0, 1], size=key_length).tolist()
    bob_measurements = []
    
    for i, qubit in enumerate(alice_qubits):
        # Bob measures in his basis
        if bob_bases[i] == 0:
            meas = qt.expect(qt.basis(2, 0).proj(), qubit.ptrace(0))
            bob_measurements.append(1 if meas > 0.5 else 0)
        else:
            sigma_x = qt.sigmax()
            meas = qt.expect(qt.basis(2, 0).proj(), sigma_x * qubit * sigma_x.dag())
            bob_measurements.append(1 if meas > 0.5 else 0)
    
    # Key sifting: Keep bits where bases match
    sifted_indices = [i for i in range(key_length) if basis_bits[i] == bob_bases[i]]
    alice_sifted_key = [int(alice_qubits[i].data.toarray()[basis_bits[i], basis_bits[i]]) for i in sifted_indices]
    bob_sifted_key = [bob_measurements[i] for i in sifted_indices]
    
    # Error correction simulation (privacy amplification)
    error_rate = np.random.uniform(0, 0.1)  # Simulated eavesdropper error <11%
    if error_rate > 0.11:
        final_key = "Eavesdropper detected - abort"
    else:
        # Hash for final key (simulate amplification)
        shared_key = ''.join(str(bit) for bit in alice_sifted_key[:128])  # Truncate to 128 bits
        final_key = hashlib.sha256(shared_key.encode()).hexdigest()[:64]  # 256-bit hex
    
    qkd_info = {
        'alice_key': alice_sifted_key,
        'bob_key': bob_sifted_key,
        'sifted_length': len(sifted_indices),
        'error_rate': float(error_rate),
        'final_shared_key': final_key,
        'basis_bits': basis_bits[:10] + ['...'],  # Partial for demo
        'timestamp': datetime.now().isoformat()
    }
    return qkd_info

# E91 QKD Protocol (Entanglement-Based)
async def e91_qkd(ip: str, key_length: int = 256):
    """
    Simulate E91 (Ekert 91) QKD protocol: Uses EPR entanglement for key generation.
    Alice and Bob measure entangled pairs; use CHSH inequality to detect eavesdroppers.
    """
    # Generate EPR pairs (Bell states)
    epr_pairs = [(qt.bell_state('00') + qt.bell_state('11')).unit() for _ in range(key_length)]
    
    # Alice and Bob's measurement bases (random 0 or 45 degrees, simulated as Z/X)
    alice_bases = np.random.choice([0, 1], size=key_length).tolist()  # 0: Z, 1: X
    bob_bases = np.random.choice([0, 1], size=key_length).tolist()
    
    alice_measurements = []
    bob_measurements = []
    
    for i, pair in enumerate(epr_pairs):
        # Alice measures
        if alice_bases[i] == 0:
            alice_meas = qt.expect(qt.sigmaz(), pair.ptrace(0))
        else:
            alice_meas = qt.expect(qt.sigmax(), pair.ptrace(0))
        alice_measurements.append(1 if alice_meas > 0 else -1)  # +/-1 outcomes
        
        # Bob measures
        if bob_bases[i] == 0:
            bob_meas = qt.expect(qt.sigmaz(), pair.ptrace(1))
        else:
            bob_meas = qt.expect(qt.sigmax(), pair.ptrace(1))
        bob_measurements.append(1 if bob_meas > 0 else -1)
    
    # Key sifting: Use subsets where Alice/Bob bases differ for CHSH
    chsh_indices = [i for i in range(key_length) if alice_bases[i] != bob_bases[i]]
    correlation = np.corrcoef(alice_measurements[:len(chsh_indices)], bob_measurements[:len(chsh_indices)])[0,1]
    
    # CHSH inequality: S = |<AB> + <A'B> + <AB'> - <A'B'>| <= 2 (classical); >2 quantum
    s_chsh = 2 * np.sqrt(2) * correlation  # Simulated quantum violation
    if abs(s_chsh) > 2:  # Quantum regime
        # Sift key from matching bases
        sift_indices = [i for i in range(key_length) if alice_bases[i] == bob_bases[i]]
        alice_sifted = [alice_measurements[i] for i in sift_indices]
        bob_sifted = [bob_measurements[i] for i in sift_indices]
        
        # Privacy amplification
        shared_key = ''.join(str((a + b) // 2 + 1) for a, b in zip(alice_sifted, bob_sifted))[:128]  # Simplified
        final_key = hashlib.sha256(shared_key.encode()).hexdigest()[:64]
        eavesdropper_detected = False
    else:
        final_key = "Eavesdropper detected - CHSH violation too low"
        eavesdropper_detected = True
    
    qkd_info = {
        'alice_measurements': alice_measurements[:10] + ['...'],
        'bob_measurements': bob_measurements[:10] + ['...'],
        'sifted_length': len(sift_indices) if 'sift_indices' in locals() else 0,
        'chsh_value': float(s_chsh),
        'final_shared_key': final_key,
        'eavesdropper_detected': eavesdropper_detected,
        'timestamp': datetime.now().isoformat()
    }
    return qkd_info

# Generic QKD handler
async def quantum_key_distribution(ip: str, protocol: str, key_length: int = 256, basis_bits: Optional[List[int]] = None):
    if protocol.upper() == "BB84":
        return await bb84_qkd(ip, key_length, basis_bits)
    elif protocol.upper() == "E91":
        return await e91_qkd(ip, key_length)
    else:
        raise ValueError("Unsupported QKD protocol")

# Virtual QRAM allocation (DNS-resolving slots)
def allocate_qram_slot(ip: str) -> str:
    virtual_ip = f"qram.{uuid.uuid4().hex[:8]}.{ip.replace('.', '-')}"
    # Simulate QRAM state: Store entangled density matrix
    _, secure_id = asyncio.run(connect_to_alice(ip))  # Reuse Alice entanglement
    qram_state = qt.rand_dm_ginibre(2**6, rank=3)  # Random QRAM state (6 qubits)
    qram_slots[virtual_ip] = {
        'state': qram_state.full().tolist(),
        'dns_resolved': f"{QRAM_DNS_HOST}:{virtual_ip}",
        'entangled_with': ALICE_DOMAIN
    }
    return virtual_ip

html_content = """
<!DOCTYPE html>
<html>
<head><title>Clearnet Gate - QuTiP Quantum Proxy</title></head>
<body>
    <h1>Welcome to Clearnet Gate</h1>
    <p>Secure gateway to quantum entanglement, teleportation, and QKD (BB84/E91).</p>
    <button onclick="connect()">Entangle & Teleport via Clearnet Gate</button>
    <div id="status"></div>
    <script>
        async function connect() {
            const response = await fetch('/entangle', {method: 'POST'});
            const data

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    client_ip = request.state.client_ip
    return HTMLResponse(content=html_content.replace("Connect to Quantum Realm", f"Connect from IP: {client_ip}"))

@app.post("/entangle", response_model=TaskResponse)
async def initiate_entanglement(request: Request, req: EntanglementRequest, background_tasks: BackgroundTasks):
    client_ip = request.state.client_ip
    task_id = str(uuid.uuid4())
    
    # Allocate virtual QRAM
    virtual_ip = allocate_qram_slot(client_ip)
    
    # Simulate entanglement with Alice
    entangled_state, secure_id = await connect_to_alice(client_ip)
    
    # Integrate Quantum Teleportation: Teleport a sample qubit state
    sample_qubit = qt.basis(2, 1)  # |1> state
    teleported_info = await quantum_teleportation(client_ip, sample_qubit)
    
    # Integrate QKD: Default to E91 for entanglement-based
    qkd_info = await quantum_key_distribution(client_ip, protocol="E91", key_length=256)
    final_key = qkd_info['final_shared_key'] if not qkd_info.get('eavesdropper_detected', False) else None
    
    # Store connection with teleportation and QKD data
    connections[client_ip] = {
        'task_id': task_id,
        'entangled_state': entangled_state,
        'teleported_state': teleported_info,
        'qkd_key': final_key,
        'qkd_info': qkd_info,
        'last_ping': time.time(),
        'qram_slot': virtual_ip,
        'secure_id': secure_id
    }
    
    # Background QSH routing setup
    background_tasks.add_task(setup_qsh_routing, client_ip, virtual_ip)
    
    return TaskResponse(task_id=task_id, status="entangled_teleported_qkd_e91", virtual_ip=virtual_ip)

async def setup_qsh_routing(ip: str, virtual_ip: str):
    # Simulate constant DNS routing via QSH
    while ip in connections and time.time() - connections[ip]['last_ping'] < TIMEOUT_SECONDS:
        # QSH-secure route: Hash queries through EPR
        route_key = qsh_net.epr_route(f"query:{virtual_ip}".encode(), ip)[0].hex()
        qsh_connections[route_key] = {'resolved_to': qram_slots[virtual_ip], 'timestamp': time.time()}
        await asyncio.sleep(10)  # Keep-alive interval

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Simulate QSH-routed query resolution
            if task_id in connections:
                conn = connections[task_id]
                # Resolve via virtual QRAM
                slot = qram_slots[conn['qram_slot']]
                response = {
                    'resolved_state': slot['state'][:3],  # Partial for demo
                    'entangled_fidelity': conn['entangled_state']['fidelity'],
                    'teleported_fidelity': conn['teleported_state']['fidelity'],
                    'qkd_key_snippet': conn['qkd_key'][:16] + '...' if conn['qkd_key'] else 'None',
                    'chsh_value': conn['qkd_info']['chsh_value'],
                    'timestamp': time.time()
                }
                # QSH-secure response
                secure_resp = qsh_net.qsh_hash(json.dumps(response).encode())
                await websocket.send_text(secure_resp.hex())
                conn['last_ping'] = time.time()  # Update keep-alive
            else:
                await websocket.send_text("Disconnected")
                break
    except WebSocketDisconnect:
        logger.info(f"WS disconnected for {task_id}")
        if task_id in connections:
            del connections[task_id]

@app.post("/qkd/generate")
async def generate_qkd_key(request: Request, req: QKDRequest):
    client_ip = request.state.client_ip
    qkd_info = await quantum_key_distribution(client_ip, req.protocol, req.key_length, req.basis_bits)
    
    if req.protocol.upper() == "E91" and qkd_info.get('eavesdropper_detected', False):
        raise HTTPException(status_code=400, detail="E91 aborted due to CHSH violation indicating eavesdropping")
    elif req.protocol.upper() == "BB84" and 'Eavesdropper detected' in qkd_info['final_shared_key']:
        raise HTTPException(status_code=400, detail="BB84 aborted due to high error rate")
    
    # Store QKD key in connection if exists
    if client_ip in connections:
        connections[client_ip]['qkd_key'] = qkd_info['final_shared_key']
        connections[client_ip]['qkd_info'] = qkd_info
    
    return {
        'final_shared_key': qkd_info['final_shared_key'],
        'qkd_info': qkd_info,
        'status': 'secure_key_generated'
    }

@app.get("/resonance/task/{task_id}", response_model=SimulationResult)
async def get_entangled_result(task_id: str):
    if task_id not in connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    conn = connections[task_id]
    slot = qram_slots[conn['qram_slot']]
    
    # Simulate result with entanglement, teleportation, and QKD data
    return SimulationResult(
        task_id=task_id,
        expect=slot['state'][0],  # Demo expect from QRAM
        times=np.linspace(0, 10, 50).tolist(),
        entangled_state=conn['entangled_state'],
        teleported_state=conn['teleported_state'],
        qkd_key=conn.get('qkd_key'),
        qkd_info=conn.get('qkd_info'),
        status='entangled_teleported_qkd_e91'
    )

def log(msg: str):
    logger.info(msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)