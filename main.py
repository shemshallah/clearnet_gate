from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import uuid
import time
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import os
import requests  # For calling gate service

app = FastAPI(title="Clearnet Chat Service", version="1.0.0")

# In-memory stores
chat_rooms: Dict[str, List[Dict[str, Any]]] = {}  # room_id -> list of {username, message, timestamp, encrypted: bool}
users: Dict[str, Dict[str, Any]] = {}  # username -> {email, password_hash, registered_at, active_rooms: list}
active_websockets: Dict[str, List[WebSocket]] = {}  # room_id -> list of active WebSockets
tokens: Dict[str, Dict[str, Any]] = {}  # token -> {username, expiry}

# Email config (custom, env-based)
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT")) if os.getenv("EMAIL_PORT") else None
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM", "no-reply@clearnet.chat")

# Gate service config (for quantum ops)
GATE_URL = os.getenv("GATE_URL", "http://localhost:8001")  # e.g., http://gate:8001

# Token expiry (1 hour)
TOKEN_EXPIRY = 3600

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class ChatMessage(BaseModel):
    room_id: str
    message: str
    encrypt: bool = False
    username: str

class JoinRequest(BaseModel):
    room_id: str
    username: str

# Custom email sending (skips if config incomplete)
def send_email(to_email: str, subject: str, body: str):
    if not all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS]):
        logger.warning(f"[EMAIL SKIPPED] To: {to_email}, Subject: {subject}")
        return
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        text = msg.as_string()
        server.sendmail(EMAIL_FROM, to_email, text)
        server.quit()
        logger.info(f"Email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

# Helper: Generate token
def generate_token(username: str) -> str:
    token = str(uuid.uuid4())
    tokens[token] = {
        "username": username,
        "expiry": time.time() + TOKEN_EXPIRY
    }
    return token

# Helper: Validate token
def validate_token(token: str) -> Optional[str]:
    if token in tokens:
        if time.time() < tokens[token]["expiry"]:
            return tokens[token]["username"]
        else:
            del tokens[token]
    return None

# Helper: Get QSH hash from gate service
def get_qsh_hash(data: str) -> str:
    try:
        resp = requests.post(f"{GATE_URL}/qsh_hash", json={"data": data})
        return resp.json()["qsh_hash"] if resp.status_code == 200 else data  # Fallback to plain
    except:
        logger.warning("Gate service unavailable, using plain text")
        return data

# Helper: Get QKD key from gate service (simple BB84 sim)
def get_qkd_key() -> List[int]:
    try:
        resp = requests.post(f"{GATE_URL}/qkd", json={"protocol": "BB84", "bits": [0,1]*64, "basis": ["Z"]*128})
        return resp.json()["shared_key"] if resp.status_code == 200 else [0]*64  # Fallback key
    except:
        logger.warning("Gate service unavailable, using fallback key")
        return [0]*64

# 1. Root endpoint with chat UI
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Clearnet Chat</title></head>
        <body>
            <h1>Clearnet Chat - Quantum Secure Messaging</h1>
            <div id="register-form" style="display: block;">
                <h2>Register</h2>
                <input id="reg-username" placeholder="Username" /><br/>
                <input id="reg-email" placeholder="Email" type="email" /><br/>
                <input id="reg-password" placeholder="Password" type="password" /><br/>
                <button onclick="registerUser()">Register</button>
            </div>
            <div id="login-form" style="display: none;">
                <h2>Login</h2>
                <input id="login-username" placeholder="Username" /><br/>
                <input id="login-password" placeholder="Password" type="password" /><br/>
                <button onclick="loginUser()">Login</button>
            </div>
            <div id="chat-section" style="display: none;">
                <h2>Chat Rooms</h2>
                <input id="room-id" placeholder="Room ID" />
                <button onclick="joinRoom()">Join/Create Room</button>
                <div id="messages"></div>
                <input id="message-input" placeholder="Type message..." />
                <button onclick="sendMessage()">Send</button>
            </div>
            <script>
                // Deployed API configuration
                const API_URL = 'https://clearnet-chat-xxxx.onrender.com';  // Replace with your Render URL

                let currentUser = null;
                let currentRoom = null;
                let ws = null;
                let token = localStorage.getItem('chat_token');  // Persist token

                // Register
                const registerUser = async () => {
                    const username = document.getElementById('reg-username').value;
                    const email = document.getElementById('reg-email').value;
                    const password = document.getElementById('reg-password').value;
                    const res = await fetch(`${API_URL}/register`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({username, email, password})
                    });
                    const data = await res.json();
                    if (data.success) {
                        alert('Registered! Please check email.');
                    } else {
                        alert(data.error || 'Registration failed');
                    }
                };

                // Login (modified to handle token)
                const login = async (username, password) => {
                    const res = await fetch(`${API_URL}/login`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({username, password})
                    });
                    return await res.json();
                };

                const loginUser = async () => {
                    const username = document.getElementById('login-username').value;
                    const password = document.getElementById('login-password').value;
                    const data = await login(username, password);
                    if (data.success) {
                        currentUser = username;
                        token = data.token;  // Assume backend returns {success: true, token: '...'}
                        localStorage.setItem('chat_token', token);
                        document.getElementById('register-form').style.display = 'none';
                        document.getElementById('login-form').style.display = 'none';
                        document.getElementById('chat-section').style.display = 'block';
                        connectChat(token);  // Connect WS on login
                    } else {
                        alert(data.error || 'Login failed');
                    }
                };

                // WebSocket connection (with token)
                const connectChat = (token) => {
                    ws = new WebSocket(`wss://${API_URL.split('https://')[1]}/ws?token=${token}`);  // Use wss for secure
                    ws.onopen = () => console.log('Connected to chat');
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'message') {
                            document.getElementById('messages').innerHTML += `<p><strong>${data.username}:</strong> ${data.message}</p>`;
                        } else if (data.type === 'rooms') {
                            // Handle room updates if needed
                        }
                    };
                    ws.onerror = (e) => console.error('WS error:', e);
                    ws.onclose = () => console.log('Disconnected');
                };

                // Join Room
                const joinRoom = () => {
                    currentRoom = document.getElementById('room-id').value || 'default';
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({action: 'join', room: currentRoom, username: currentUser}));
                    } else {
                        alert('Connect to chat first');
                    }
                };

                // Send Message
                const sendMessage = () => {
                    const msg = document.getElementById('message-input').value;
                    if (msg && currentRoom && ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({action: 'message', room: currentRoom, username: currentUser, message: msg}));
                        document.getElementById('message-input').value = '';
                    } else {
                        alert('Join a room and connect first');
                    }
                };

                // Auto-connect if token exists
                if (token) {
                    connectChat(token);
                    document.getElementById('register-form').style.display = 'none';
                    document.getElementById('login-form').style.display = 'block';  // Or show chat if valid
                }
            </script>
        </body>
    </html>
    """

# 2. User Registration
@app.post("/register")
async def register_user(user: UserRegister):
    if user.username in users:
        raise HTTPException(400, "Username taken")
    password_hash = hashlib.sha256(user.password.encode()).hexdigest()
    users[user.username] = {
        "email": user.email,
        "password_hash": password_hash,
        "registered_at": datetime.now().isoformat(),
        "active_rooms": []
    }
    send_email(user.email, "Welcome to Clearnet Chat", f"Hi {user.username}, your account is ready. Quantum-secure chats await!")
    return {"success": True}

# 3. User Login (with token generation)
@app.post("/login")
async def login_user(request: Request, user: UserRegister):
    if user.username not in users:
        raise HTTPException(401, "Invalid credentials")
    stored_hash = users[user.username]["password_hash"]
    input_hash = hashlib.sha256(user.password.encode()).hexdigest()
    if stored_hash != input_hash:
        raise HTTPException(401, "Invalid credentials")
    token = generate_token(user.username)
    return {"success": True, "token": token}

# 4. Join Room
@app.post("/rooms/join")
async def join_room(req: JoinRequest):
    username = req.username
    room_id = req.room_id
    if username not in users:
        raise HTTPException(401, "Unauthorized")
    if room_id not in chat_rooms:
        chat_rooms[room_id] = []
    if room_id not in users[username]["active_rooms"]:
        users[username]["active_rooms"].append(room_id)
        send_email(users[username]["email"], f"Joined Room: {room_id}", f"You've joined {room_id}.")
    return {"success": True, "room_id": room_id, "history": chat_rooms[room_id][-50:]}

# 5. Send Message (REST fallback)
@app.post("/rooms/message")
async def send_message(msg: ChatMessage):
    username = msg.username
    if username not in users or msg.room_id not in users[username]["active_rooms"]:
        raise HTTPException(401, "Unauthorized")
    stored_msg = get_qsh_hash(msg.message) if msg.encrypt else msg.message  # Use gate for hash/encrypt
    message_data = {"username": username, "message": stored_msg, "timestamp": datetime.now().isoformat(), "encrypted": msg.encrypt}
    chat_rooms[msg.room_id].append(message_data)
    # Email notifications
    for u, data in users.items():
        if msg.room_id in data["active_rooms"]:
            plain_msg = msg.message[:50] + "..." if not msg.encrypt else "[Encrypted]"
            send_email(data["email"], "New Message", f"{username}: {plain_msg}")
    return {"success": True}

# 6. WebSocket Chat (with token validation)
@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    # Extract token from query params
    query = websocket.query_params.get("token")
    username = validate_token(query)
    if not username:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return
    try:
        while True:
            data = await websocket.receive_text()
            parsed = json.loads(data)
            action = parsed.get('action')
            if action == 'join':
                room_id = parsed['room']
                if username not in users or room_id not in users[username]["active_rooms"]:
                    await websocket.send_text(json.dumps({"error": "Unauthorized"}))
                    continue
                if room_id not in active_websockets:
                    active_websockets[room_id] = []
                active_websockets[room_id].append(websocket)
                history = chat_rooms.get(room_id, [])[-50:]
                await websocket.send_text(json.dumps({"type": "joined", "room": room_id, "history": history}))
                for ws in active_websockets.get(room_id, []):
                    if ws != websocket:
                        await ws.send_text(json.dumps({"type": "user_joined", "username": username}))
            elif action == 'message':
                room_id = parsed['room']
                message = parsed['message']
                encrypt = parsed.get('encrypt', False)
                if room_id not in active_websockets or websocket not in active_websockets[room_id]:
                    await websocket.send_text(json.dumps({"error": "Unauthorized"}))
                    continue
                stored_msg = get_qsh_hash(message) if encrypt else message
                message_data = {"type": "message", "username": username, "message": stored_msg, "timestamp": datetime.now().isoformat(), "encrypted": encrypt}
                chat_rooms[room_id].append({"username": username, "message": stored_msg, "timestamp": message_data["timestamp"], "encrypted": encrypt})
                for ws in active_websockets.get(room_id, []):
                    await ws.send_text(json.dumps(message_data))
            elif action == 'ping':
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup WS
        for room_id, wss in list(active_websockets.items()):
            active_websockets[room_id] = [ws for ws in wss if not ws.client == websocket.client]

# Retained quantum endpoints for integration (e.g., secure keys for chat)
@app.post("/qsh_hash")
async def qsh_hash(data: str):
    hashed = qsh_network.qsh_hash(data.encode('utf-8'))
    return {"qsh_hash": hashed.hex()}

@app.get("/create_epr")
async def create_epr():
    epr = qsh_network.create_epr_pair()
    return {"epr_state": epr.full().tolist()}

@app.post("/qkd")
async def qkd(request: QKDRequest):
    if request.protocol == "BB84":
        if not request.bits or not request.basis:
            raise HTTPException(400, "bits and basis required for BB84")
        result = qsh_network.simulate_bb84(request.bits, request.basis)
    elif request.protocol == "E91":
        result = qsh_network.simulate_e91(request.num_pairs)
    else:
        raise HTTPException(400, "Invalid protocol")
    client_ip = "unknown"  # Enhance with middleware
    if client_ip not in connections:
        connections[client_ip] = {}
    connections[client_ip]['qkd_key'] = result['shared_key']
    connections[client_ip]['last_ping'] = time.time()
    return result

@app.post("/teleport")
async def teleport(request: TeleportRequest):
    state = qt.Qobj(np.array(request.state_matrix))
    if state.dims != [[2], [2]]:
        raise HTTPException(400, "Invalid state")
    result = qsh_network.quantum_teleport(state)
    client_ip = "unknown"
    if client_ip not in connections:
        connections[client_ip] = {}
    connections[client_ip]['teleported_state'] = result['teleported_state']
    connections[client_ip]['last_ping'] = time.time()
    return {"fidelity": result['fidelity'], "classical_bits": result['classical_bits']}

@app.get("/connections")
async def list_connections():
    active = {ip: conn for ip, conn in connections.items() if time.time() - conn.get('last_ping', 0) < TIMEOUT_SECONDS}
    return {"active_connections": active, "count": len(active)}

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
