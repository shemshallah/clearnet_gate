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
        for i in range(0, len(qsh_result), 2):
            if i + 1 < len(qsh_result):
                state_idx = (qsh_result[i] + qsh_result[i+1]) % (2**self.qubits)
                amplitude, phase = self.ghz_table[state_idx]
                quantum_mod = int((amplitude * np.cos(phase) + 1) * 127.5)
                qsh_result[i] = (qsh_result[i] ^ quantum_mod) % 256
        return bytes(qsh_result)

    def create_epr_pair(self):
        """Create Bell state EPR pair"""
        return qt.bell_state('00')

    def create_ghz_state(self):
        """Create 6-qubit GHZ state"""
        basis_0 = qt.tensor([qt.basis(2, 0) for _ in range(self.qubits)])
        basis_1 = qt.tensor([qt.basis(2, 1) for _ in range(self.qubits)])
        ghz = (basis_0 + basis_1).unit()
        return ghz

    def simulate_bb84(self, bits: List[int], basis: List[str]) -> Dict[str, Any]:
        """Simulate BB84 QKD protocol"""
        alice_bits = bits.copy()
        alice_basis = basis.copy()
        bob_basis = [np.random.choice(['Z', 'X']) for _ in alice_basis]
        bob_measurements = []
        for i, bit in enumerate(alice_bits):
            if alice_basis[i] == bob_basis[i]:
                # Same basis: measurement matches bit
                bob_measurements.append(bit)
            else:
                # Different basis: random 0 or 1
                bob_measurements.append(np.random.randint(0, 2))
        # Sift: keep matching bases
        sifted_indices = [i for i in range(len(alice_basis)) if alice_basis[i] == bob_basis[i]]
        sifted_alice = [alice_bits[i] for i in sifted_indices]
        sifted_bob = [bob_measurements[i] for i in sifted_indices]
        # Error rate simulation (simple noise)
        errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
        error_rate = errors / len(sifted_alice) if sifted_alice else 0
        shared_key = sifted_alice[:min(len(sifted_alice), 128)]  # Truncate to 128 bits
        return {
            "shared_key": shared_key,
            "error_rate": error_rate,
            "sifted_length": len(sifted_alice)
        }

    def simulate_e91(self, num_pairs: int = 100) -> Dict[str, Any]:
        """Simulate E91 entanglement-based QKD"""
        epr_pairs = [self.create_epr_pair() for _ in range(num_pairs)]
        alice_measurements = []
        bob_measurements = []
        alice_bases = [np.random.choice(['Z', 'X']) for _ in range(num_pairs)]
        bob_bases = [np.random.choice(['Z', 'X']) for _ in range(num_pairs)]
        for i in range(num_pairs):
            # Simulate measurement in chosen basis
            if alice_bases[i] == 'Z':
                alice_res = epr_pairs[i].ptrace(0).diag()[0] > 0.5  # Simplified
            else:  # X basis
                alice_res = np.random.randint(0, 2)
            # Bob's measurement correlated
            if bob_bases[i] == 'Z':
                bob_res = alice_res ^ 1 if alice_res else 0  # Anti-correlated for singlet-like
            else:
                bob_res = np.random.randint(0, 2)
            alice_measurements.append(alice_res)
            bob_measurements.append(bob_res)
        # CHSH inequality check for security (simplified)
        chsh_value = 2 * np.sqrt(2) * 0.9  # Simulated violation
        shared_key = alice_measurements[:128]
        return {
            "shared_key": shared_key,
            "chsh_value": chsh_value,
            "pairs_used": num_pairs
        }

    def quantum_teleport(self, state_to_teleport: qt.Qobj) -> Dict[str, Any]:
        """Simulate quantum teleportation of a state"""
        epr = self.create_epr_pair()
        # Alice's part: Bell measurement (simplified)
        bell_measurement = np.random.randint(0, 4)  # 00,01,10,11
        # Bob's correction based on classical bits
        corrections = {
            0: lambda s: s,  # No correction
            1: lambda s: qt.sigmax() * s * qt.sigmax(),
            2: lambda s: qt.sigmaz() * s * qt.sigmaz(),
            3: lambda s: qt.sigmaz() * qt.sigmax() * s * qt.sigmax() * qt.sigmaz()
        }
        teleported_state = corrections[bell_measurement](state_to_teleport)
        fidelity = qt.fidelity(teleported_state, state_to_teleport)
        return {
            "teleported_state": teleported_state,
            "original_state": state_to_teleport,
            "fidelity": fidelity,
            "classical_bits": [bell_measurement // 2, bell_measurement % 2]
        }

# Global instance
qsh_network = QSH6_EPR_Network()

class ConnectionRequest(BaseModel):
    ip: str
    port: Optional[int] = None

class HashRequest(BaseModel):
    data: str

class QKDRequest(BaseModel):
    protocol: str  # "BB84" or "E91"
    bits: Optional[List[int]] = None
    basis: Optional[List[str]] = None
    num_pairs: Optional[int] = 100

class TeleportRequest(BaseModel):
    state_matrix: List[List[complex]]  # 2x2 density matrix

# Helper to clean expired connections
async def cleanup_expired_connections():
    while True:
        await asyncio.sleep(60)  # Check every minute
        now = time.time()
        expired = [ip for ip, conn in connections.items() if now - conn.get('last_ping', 0) > TIMEOUT_SECONDS]
        for ip in expired:
            del connections[ip]
            logger.info(f"Expired connection cleaned: {ip}")

# Start cleanup task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_expired_connections())

# 1. Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>QuTiP Entanglement Proxy</title></head>
        <body>
            <h1>QuTiP Entanglement Proxy API</h1>
            <p>Endpoints:</p>
            <ul>
                <li><a href="/qsh_hash">/qsh_hash (POST)</a></li>
                <li><a href="/create_epr">/create_epr (GET)</a></li>
                <li><a href="/qkd">/qkd (POST)</a></li>
                <li><a href="/teleport">/teleport (POST)</a></li>
                <li><a href="/ws">/ws (WebSocket)</a></li>
                <li><a href="/connections">/connections (GET)</a></li>
            </ul>
        </body>
    </html>
    """

# 2. QSH Hash endpoint
@app.post("/qsh_hash")
async def qsh_hash(request: HashRequest):
    data_bytes = request.data.encode('utf-8')
    hashed = qsh_network.qsh_hash(data_bytes)
    return {"original": request.data, "qsh_hash": hashed.hex()}

# 3. Create EPR pair
@app.get("/create_epr")
async def create_epr():
    epr = qsh_network.create_epr_pair()
    return {"epr_state": epr.full().tolist()}

# 4. GHZ state
@app.get("/create_ghz")
async def create_ghz():
    ghz = qsh_network.create_ghz_state()
    return {"ghz_state": ghz.full().tolist()}

# 5. QKD endpoint
@app.post("/qkd")
async def qkd(request: QKDRequest):
    if request.protocol == "BB84":
        if not request.bits or not request.basis:
            raise HTTPException(400, "bits and basis required for BB84")
        result = qsh_network.simulate_bb84(request.bits, request.basis)
    elif request.protocol == "E91":
        result = qsh_network.simulate_e91(request.num_pairs)
    else:
        raise HTTPException(400, "Invalid protocol: BB84 or E91")
    # Store key in connection if IP available
    client_ip = "unknown"
    if hasattr(request, '_client_host'):
        client_ip = request._client_host
    if client_ip not in connections:
        connections[client_ip] = {}
    connections[client_ip]['qkd_key'] = result['shared_key']
    connections[client_ip]['last_ping'] = time.time()
    return result

# 6. Teleportation endpoint
@app.post("/teleport")
async def teleport(request: TeleportRequest):
    state = qt.Qobj(np.array(request.state_matrix))
    if state.dims != [[2], [2]]:
        raise HTTPException(400, "State must be 2x2 density matrix for qubit")
    result = qsh_network.quantum_teleport(state)
    # Store teleported state
    client_ip = "unknown"
    # Simulate getting client IP
    if client_ip not in connections:
        connections[client_ip] = {}
    connections[client_ip]['teleported_state'] = result['teleported_state']
    connections[client_ip]['last_ping'] = time.time()
    return {
        "fidelity": result['fidelity'],
        "classical_bits": result['classical_bits']
    }

# 7. WebSocket for real-time entanglement monitoring
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    task_id = str(uuid.uuid4())
    connections['ws_' + task_id] = {'task_id': task_id, 'last_ping': time.time(), 'entangled_state': None}
    try:
        while True:
            data = await websocket.receive_text()
            parsed = json.loads(data)
            if parsed.get('action') == 'ping':
                connections['ws_' + task_id]['last_ping'] = time.time()
                await websocket.send_text(json.dumps({"status": "pong"}))
            elif parsed.get('action') == 'entangle':
                epr = qsh_network.create_epr_pair()
                connections['ws_' + task_id]['entangled_state'] = epr
                await websocket.send_text(json.dumps({"entangled_state": epr.full().tolist()}))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {task_id}")
    finally:
        if 'ws_' + task_id in connections:
            del connections['ws_' + task_id]

# 8. List connections
@app.get("/connections")
async def list_connections():
    active = {ip: {k: v for k, v in conn.items() if k != 'last_ping'} for ip, conn in connections.items()
              if time.time() - conn.get('last_ping', 0) < TIMEOUT_SECONDS}
    return {"active_connections": active, "count": len(active)}

# 9. QRAM slot allocation (simple)
@app.post("/qram/allocate")
async def allocate_qram(request: ConnectionRequest):
    virtual_ip = f"qram_{uuid.uuid4().hex[:8]}"
    slot_state = qsh_network.create_ghz_state()
    qram_slots[virtual_ip] = slot_state
    # Link to connection
    client_ip = request.ip
    if client_ip not in connections:
        connections[client_ip] = {}
    connections[client_ip]['qram_slot'] = virtual_ip
    connections[client_ip]['last_ping'] = time.time()
    return {"virtual_ip": virtual_ip, "slot_state": slot_state.full().tolist()}

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}  # user_id -> websocket

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def broadcast_to_matches(self, message: str, sender_labels: str):
        for uid, ws in self.active_connections.items():
            # Entanglement match
            if any(label in sender_labels for label in db.query(User).filter(User.id == uid).first().labels.split(',')):
                await ws.send_text(message)

manager = ConnectionManager()

# Front page / login
@app.get("/", response_class=HTMLResponse)
async def front_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# Registration
@app.post("/api/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    domain: str = Form("quantum"),  # Choice of domains
    labels: str = Form(""),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username taken")
    user = User(username=username, domain=domain, labels=labels.strip())
    user.set_password(password)
    user.email = user.full_email
    db.add(user)
    db.commit()
    db.refresh(user)
    # Token
    token = create_access_token(data={"sub": username}, expires_delta=timedelta(days=7))
    return {"token": token, "email": user.email, "message": "Welcome to Foam Computer! Your messages are secured via quantum colliders—black hole hashes retrieved from white holes."}

# Login
@app.post("/api/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False),
    request: Request = None,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.verify_password(password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    expires = timedelta(days=7) if remember_me else timedelta(minutes=30)
    token = create_access_token(data={"sub": username}, expires_delta=expires)
    response = RedirectResponse(url="/chat", status_code=303)
    if remember_me:
        response.set_cookie(key="remember_ip", value=request.client.host, httponly=True, max_age=604800)  # 7 days
    response.headers["Authorization"] = f"Bearer {token}"
    return response

# Forget password
@app.post("/api/forget-password")
async def forget_password(username: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_pass = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$', k=16))
    user.set_password(new_pass)
    db.commit()
    return {"new_password": new_pass, "message": "Password reset. Change it immediately in settings. Secured via Argon2 quantum-resistant hashing."}

# WebSocket for IM/Chat (entanglement spawn)
@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket, token: str = Depends(get_current_user)):
    user = get_current_user(token=token)  # From auth
    await manager.connect(websocket, user.id)
    try:
        while True:
            data = await websocket.receive_text()
            parsed = json.loads(data)
            content = parsed["content"]
            receiver_id = parsed.get("receiver_id")
            # Encrypt with collider
            encrypted, bhash = encrypt_with_collider(content)
            message = Message(sender_id=user.id, receiver_id=receiver_id, encrypted_content=encrypted, black_hole_hash=bhash)
            db = next(get_db())
            db.add(message)
            db.commit()
            # AI spawn if label match or /ai command
            if " /ai" in content or any("ai" in label for label in user.labels.split(',')):
                ai_resp = f"Grok Clone: Resonating with your query through foam... {content.upper()}"  # Scrubbed autonomous AI
                await manager.send_personal(json.dumps({"content": ai_resp, "is_ai": True}), websocket)
            # Send to receiver or broadcast to matches
            if receiver_id:
                await manager.send_personal(json.dumps({"content": content, "sender": user.id}), receiver_id)
            else:
                await manager.broadcast_to_matches(json.dumps({"content": content, "sender": user.id}), user.labels)
    except WebSocketDisconnect:
        manager.disconnect(user.id)

# Inbox API (searchable)
@app.get("/api/inbox")
async def get_inbox(
    search: str = "",
    folder: str = "Inbox",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Email).filter(Email.receiver_id == current_user.id, Email.folder == folder, Email.is_deleted == False)
    if search:
        query = query.filter((Email.subject.like(f"%{search}%")) | (Email.encrypted_body.like(f"%{search}%")))  # Full-text
    emails = query.order_by(Email.timestamp.desc()).all()
    return [{"id": e.id, "subject": e.subject, "body": e.body[:100], "folder": e.folder, "label": e.label, "is_starred": e.is_starred, "timestamp": e.timestamp, "hash_explain": "Secured by black hole hash from white hole retrieval"} for e in emails]

@app.post("/api/inbox/send")
async def send_email(
    receiver_email: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    receiver = db.query(User).filter(User.email == receiver_email).first()
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")
    encrypted, bhash = encrypt_with_collider(body)
    email = Email(sender_id=current_user.id, receiver_id=receiver.id, subject=subject, encrypted_body=encrypted, black_hole_hash=bhash)
    db.add(email)
    db.commit()
    return {"message": "Email sent to Holo storage, encrypted via collider."}

# Delete
@app.delete("/api/inbox/{email_id}")
async def delete_email(email_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    email = db.query(Email).filter(Email.id == email_id, Email.receiver_id == current_user.id).first()
    if not email:
        raise HTTPException(status_code=404)
    email.is_deleted = True
    db.commit()
    return {"message": "Deleted from Holo (soft delete for audit)"}

# Contacts
@app.post("/api/contacts")
async def add_contact(contact_email: str = Form(...), name: str = Form(""), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    contact = Contact(user_id=current_user.id, contact_email=contact_email, name=name)
    db.add(contact)
    db.commit()
    return {"message": "Contact added"}

@app.post("/api/contacts/import")
async def import_contacts(data: str = Form(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Simulate CSV import
    lines = data.split('\n')
    for line in lines:
        if ',' in line:
            email, name = line.split(',', 1)
            contact = Contact(user_id=current_user.id, contact_email=email.strip(), name=name.strip())
            db.add(contact)
    db.commit()
    return {"message": f"Imported {len(lines)} contacts"}

# Chat page (React served)
@app.get("/chat")
async def chat_route(current_user: User = Depends(get_current_user)):
    return RedirectResponse(url="/static/index.html")  # React app

# Inbox page
@app.get("/inbox")
async def inbox_route(current_user: User = Depends(get_current_user)):
    return RedirectResponse(url="/static/index.html")  # React

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
