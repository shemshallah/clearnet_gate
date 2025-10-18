import os
import logging
import hashlib
import base64
import json
import uuid
import code  # For REPL
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, Depends, WebSocketState, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx
import asyncio
from contextlib import asynccontextmanager
import secrets
from collections import defaultdict
import random  # Retained for fictional quantum variance only
import psutil  # For real-time system network and storage metrics
import subprocess  # For real routing table extraction and QSH commands
from jinja2 import Template
import socket  # For AF_INET constant
import sqlite3  # For persistent storage of emails, folders, labels, contacts
import re  # For search regex
import ast  # For safe eval in REPL
import traceback  # For error handling
import threading  # For Jupyter kernel simulation

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO if not os.getenv("DEBUG", "false").lower() == "true" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION MODULE ====================
class Config:
    """Centralized configuration management"""
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Backend
    CHAT_BACKEND = os.getenv("CHAT_BACKEND_URL", "https://clearnet-chat-4bal.onrender.com")
    SKIP_BACKEND_CHECKS = os.getenv("SKIP_BACKEND_CHECKS", "true").lower() == "true"
    
    # Network
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = int(os.getenv("TIMEOUT", "30"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Quantum
    BLACK_HOLE_ADDRESS = "138.0.0.1"
    WHITE_HOLE_ADDRESS = "139.0.0.1"
    QUANTUM_REALM = "quantum.realm.domain.dominion.foam.computer.alice"
    NETWORKING_ADDRESS = "quantum.realm.domain.dominion.foam.computer.networking"
    
    # Bitcoin
    BITCOIN_UPDATE_INTERVAL = int(os.getenv("BITCOIN_UPDATE_INTERVAL", "30"))
    BITCOIN_RPC_USER = os.getenv("BITCOIN_RPC_USER", "hackah")
    BITCOIN_RPC_PASS = os.getenv("BITCOIN_RPC_PASS", "hackah")
    
    # Storage
    HOLOGRAPHIC_CAPACITY_TB = 138000  # 138 Petabytes
    QRAM_CAPACITY_QUBITS = 1000000000  # 1 Billion Qubits
    
    # Templates - changed to root directory
    TEMPLATES_DIR = Path(".")
    STATIC_DIR = Path("static")
    UPLOADS_DIR = Path("uploads")
    
    # Database
    DB_PATH = Path("quantum_foam.db")

# Create directories
Config.STATIC_DIR.mkdir(exist_ok=True)
Config.UPLOADS_DIR.mkdir(exist_ok=True)

# ==================== PQC LAMPORT SIGNATURE MODULE ====================
def lamport_keygen(n=256):
    """Generate Lamport keypair for n-bit messages"""
    sk = []
    pk = []
    for _ in range(n):
        sk0 = os.urandom(32)
        sk1 = os.urandom(32)
        pk0 = hashlib.sha256(sk0).digest()
        pk1 = hashlib.sha256(sk1).digest()
        sk.append((sk0, sk1))
        pk.append((pk0, pk1))
    return sk, pk

def lamport_sign(message: bytes, sk: list) -> bytes:
    """Sign message with Lamport signature"""
    m_hash = hashlib.sha256(message).digest()
    bits = [(m_hash[i // 8] >> (7 - (i % 8))) & 1 for i in range(256)]
    sig = b''
    for i, b in enumerate(bits):
        sig += sk[i][b]
    return sig

def lamport_verify(message: bytes, sig: bytes, pk: list) -> bool:
    """Verify Lamport signature"""
    m_hash = hashlib.sha256(message).digest()
    bits = [(m_hash[i // 8] >> (7 - (i % 8))) & 1 for i in range(256)]
    pos = 0
    for i, b in enumerate(bits):
        revealed = sig[pos:pos + 32]
        pos += 32
        expected_pk = pk[i][b]
        if hashlib.sha256(revealed).digest() != expected_pk:
            return False
    return True

# ==================== DILITHIUM PQC INTEGRATION ====================
try:
    from dilithium import Dilithium2
    DILITHIUM_AVAILABLE = True
except ImportError:
    DILITHIUM_AVAILABLE = False
Part 3: Quantum Encryption Module
# ==================== QUANTUM ENCRYPTION MODULE ====================
def derive_key(address: str) -> bytes:
    return hashlib.sha256(address.encode()).digest()

BLACK_KEY = derive_key(Config.BLACK_HOLE_ADDRESS)
WHITE_KEY = derive_key(Config.WHITE_HOLE_ADDRESS)
WHITE_REV = WHITE_KEY[::-1]

def pad(data: bytes, block_size: int = 64) -> bytes:
    padding_len = block_size - (len(data) % block_size)
    padding = bytes([padding_len] * padding_len)
    return data + padding

def unpad(data: bytes, block_size: int = 64) -> bytes:
    if len(data) % block_size != 0:
        raise ValueError("Invalid padding length")
    padding_len = data[-1]
    if padding_len == 0 or padding_len > block_size:
        raise ValueError("Invalid padding length")
    padding = data[-padding_len:]
    if padding != bytes([padding_len] * padding_len):
        raise ValueError("Invalid padding bytes")
    return data[:-padding_len]

def xor_bytes(data: bytes, key: bytes) -> bytes:
    key_len = len(key)
    return bytes(data[i] ^ key[i % key_len] for i in range(len(data)))

def sha3_keystream(length: int, seed: bytes) -> bytes:
    keystream = b''
    h = hashlib.sha3_512()
    counter = 0
    while len(keystream) < length:
        h.update(seed + counter.to_bytes(8, 'big'))
        keystream += h.digest()
        counter += 1
    return keystream[:length]

def quantum_encrypt(plaintext: str, depth: int = 3) -> Dict[str, Any]:
    if depth < 1 or depth > 10:
        raise ValueError("Depth must be between 1 and 10")
    plain_bytes = plaintext.encode('utf-8')
    nonce = os.urandom(16)
    padded = pad(plain_bytes)
    
    seed = BLACK_KEY + WHITE_KEY + nonce
    current = padded
    for layer in range(depth):
        layer1 = xor_bytes(current, BLACK_KEY)
        layer2 = xor_bytes(layer1, WHITE_REV)
        layer_seed = seed + layer.to_bytes(1, 'big')
        keystream = sha3_keystream(len(layer2), layer_seed)
        current = xor_bytes(layer2, keystream)
    
    ciphertext = current
    sig_input = nonce + ciphertext + BLACK_KEY + WHITE_KEY
    lamport_sk, lamport_pk = lamport_keygen()
    lamport_sig = lamport_sign(sig_input, lamport_sk)
    lamport_pk_serial = ','.join([base64.b64encode(p[0] + p[1]).decode() for p in lamport_pk])
    lamport_sig_serial = base64.b64encode(lamport_sig).decode()
    
    dil_pk, dil_sk = Dilithium2.keygen() if DILITHIUM_AVAILABLE else (None, None)
    dil_sig = Dilithium2.sign(sig_input, dil_sk) if DILITHIUM_AVAILABLE and dil_sk else b''
    dil_pk_serial = base64.b64encode(dil_pk).decode() if DILITHIUM_AVAILABLE and dil_pk else ''
    dil_sig_serial = base64.b64encode(dil_sig).decode() if DILITHIUM_AVAILABLE and dil_sig else ''
    
    sha3_sig = hashlib.sha3_512(sig_input).hexdigest()
    
    ts = datetime.now().isoformat()
    
    return {
        'nonce': nonce.hex(),
        'ciphertext': ciphertext.hex(),
        'sha3_signature': sha3_sig,
        'lamport_pk': lamport_pk_serial,
        'lamport_sig': lamport_sig_serial,
        'dilithium_pk': dil_pk_serial,
        'dilithium_sig': dil_sig_serial,
        'black_hole': Config.BLACK_HOLE_ADDRESS,
        'white_hole': Config.WHITE_HOLE_ADDRESS,
        'algorithm': f'QUANTUM-DUAL-HOLE-XOR-SHA3-LAMPORT-DILITHIUM-v5-depth-{depth}',
        'timestamp': ts,
        'depth': depth
    }

def quantum_decrypt(encrypted_data: Dict[str, Any]) -> str:
    nonce_hex = encrypted_data['nonce']
    ciphertext_hex = encrypted_data['ciphertext']
    sha3_sig = encrypted_data.get('sha3_signature', '')
    lamport_pk = encrypted_data.get('lamport_pk', '')
    lamport_sig_serial = encrypted_data.get('lamport_sig', '')
    dil_pk = encrypted_data.get('dilithium_pk', '')
    dil_sig_serial = encrypted_data.get('dilithium_sig', '')
    black_h = encrypted_data['black_hole']
    white_h = encrypted_data['white_hole']
    depth = encrypted_data['depth']
    
    black_key = derive_key(black_h)
    white_key = derive_key(white_h)
    white_rev = white_key[::-1]
    nonce = bytes.fromhex(nonce_hex)
    
    sig_input = nonce + bytes.fromhex(ciphertext_hex) + black_key + white_key
    
    computed_sha3 = hashlib.sha3_512(sig_input).hexdigest()
    if sha3_sig != computed_sha3:
        raise ValueError('SHA3 Signature mismatch')
    
    if lamport_pk and lamport_sig_serial:
        pk_bytes_list = [base64.b64decode(s) for s in lamport_pk.split(',')]
        if len(pk_bytes_list) != 256:
            raise ValueError('Invalid Lamport PK')
        pk = [(b[:32], b[32:]) for b in pk_bytes_list]
        lamport_sig = base64.b64decode(lamport_sig_serial)
        if not lamport_verify(sig_input, lamport_sig, pk):
            raise ValueError('Lamport Signature mismatch')
    
    if dil_pk and dil_sig_serial and DILITHIUM_AVAILABLE:
        pk = base64.b64decode(dil_pk)
        sig = base64.b64decode(dil_sig_serial)
        if not Dilithium2.verify(sig_input, sig, pk):
            raise ValueError('Dilithium Signature mismatch')
    
    ciphertext = bytes.fromhex(ciphertext_hex)
    seed = black_key + white_key + nonce
    current = ciphertext
    
    for layer in range(depth - 1, -1, -1):
        layer_seed = seed + layer.to_bytes(1, 'big')
        keystream = sha3_keystream(len(current), layer_seed)
        layer2 = xor_bytes(current, keystream)
        layer1 = xor_bytes(layer2, white_rev)
        current = xor_bytes(layer1, black_key)
    
    padded = current
    plain_bytes = unpad(padded)
    return plain_bytes.decode('utf-8')

# ==================== RECURSIVE QUANTUM HASHING FOR ADMIN ====================
def recursive_quantum_hash(creds: str, depth: int = 10) -> str:
    """Recursively hash credentials using quantum_encrypt, depth times"""
    current = creds
    for i in range(depth):
        enc = quantum_encrypt(current, 1)  # Depth 1 per iteration for recursion
        current = enc['ciphertext']  # Chain on ciphertext
    return current

# ==================== DATABASE MODULE ====================
class Database:
    """SQLite-based storage for emails, folders, labels, contacts with holographic simulation"""
    
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self.setup_tables()
    
    def setup_tables(self):
        cursor = self.conn.cursor()
        # Emails table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                thread_id TEXT DEFAULT '',
                parent_id TEXT DEFAULT '',
                from_email TEXT NOT NULL,
                to_email TEXT NOT NULL,
                cc_email TEXT DEFAULT '',
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                attachments TEXT DEFAULT '[]',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                read BOOLEAN DEFAULT FALSE,
                folder TEXT DEFAULT 'Inbox',
                labels TEXT DEFAULT '[]'
            )
        """)
        # Folders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                user TEXT NOT NULL
            )
        """)
        # Contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                user TEXT NOT NULL
            )
        """)
        # Admin credentials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_creds (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                hashed_pass TEXT NOT NULL,
                plaintext_pass TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # QSH command history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qsh_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                command TEXT NOT NULL,
                output TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Default folders
        default_folders = [('Inbox', 'all_users'), ('Sent', 'all_users'), ('Drafts', 'all_users'), ('Trash', 'all_users')]
        cursor.executemany("INSERT OR IGNORE INTO folders (name, user) VALUES (?, ?)", default_folders)
        self.conn.commit()
    
    def holographic_store(self, data: Dict[str, Any]) -> str:
        variance = round(random.uniform(0.9990, 0.9999), 4)
        hashed = hashlib.sha256(json.dumps(data).encode()).digest()
        encoded = base64.b64encode(hashed).decode()
        return f"{encoded}:{variance}"
    
    def holo_search(self, username: str, query: str, folder: str = 'Inbox') -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        pattern = f".*{re.escape(query)}.*"
        cursor.execute("""
            SELECT * FROM emails 
            WHERE (from_email LIKE ? OR to_email LIKE ? OR subject LIKE ? OR body LIKE ?) 
            AND to_email LIKE ? AND folder = ?
            ORDER BY timestamp DESC
        """, (pattern, pattern, pattern, pattern, f"{username}::quantum.foam", folder))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_emails(self, username: str, folder: str = 'Inbox') -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM emails 
            WHERE to_email LIKE ? AND folder = ? 
            ORDER BY timestamp DESC
        """, (f"{username}::quantum.foam", folder))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM emails WHERE thread_id = ? ORDER BY timestamp ASC", (thread_id,))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        email_id = str(uuid.uuid4())
        thread_id = email_data.get('thread_id', email_id)
        parent_id = email_data.get('parent_id', '')
        attachments = json.dumps(email_data.get('attachments', []))
        cursor.execute("""
            INSERT INTO emails (id, thread_id, parent_id, from_email, to_email, cc_email, subject, body, attachments, labels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (email_id, thread_id, parent_id, email_data['from'], email_data['to'], email_data.get('cc', ''), 
              email_data['subject'], email_data['body'], attachments, json.dumps([])))
        self.conn.commit()
        holo_tag = self.holographic_store(email_data)
        if 'chat_to' in email_data:
            logger.info(f"Forwarding email {email_id} to chat for {email_data['chat_to']}")
        return {'id': email_id, 'holo_tag': holo_tag}
    
    def create_folder(self, folder_name: str, username: str) -> str:
        cursor = self.conn.cursor()
        folder_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO folders (id, name, user) VALUES (?, ?, ?)", (folder_id, folder_name, username))
        self.conn.commit()
        return folder_id
    
    def get_folders(self, username: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM folders WHERE user = ? OR user = 'all_users'", (username,))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def move_to_folder(self, email_id: str, folder: str):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE emails SET folder = ? WHERE id = ?", (folder, email_id))
        self.conn.commit()
        return True

# Instantiate database
db = Database()
Part 5: Quantum Entanglement Module
# ==================== QUANTUM ENTANGLEMENT MODULE ====================
class QuantumEntanglement:
    """Quantum entanglement management and measurement"""
    
    def __init__(self):
        self.entanglements = []
        self.initialize_entanglements()
    
    def initialize_entanglements(self):
        """Initialize quantum entanglements with dynamic timestamps"""
        now = datetime.now().isoformat()
        self.entanglements = [
            {
                "id": "QE-001",
                "name": "Black Hole ⚫ ↔ White Hole ⚪",
                "node_a": Config.BLACK_HOLE_ADDRESS,
                "node_b": Config.WHITE_HOLE_ADDRESS,
                "type": "Wormhole Bridge",
                "coherence": round(random.uniform(0.9990, 0.9999), 4),
                "fidelity": round(random.uniform(0.9980, 0.9998), 4),
                "bell_state": "|Φ+⟩",
                "speed_gbps": int(random.uniform(900000, 1100000)),
                "qubit_rate": int(random.uniform(900000000, 1100000000)),
                "distance_km": "Non-local (Einstein-Podolsky-Rosen)",
                "created": now,
                "entanglement_strength": "Maximum",
                "decoherence_time_ms": int(random.uniform(9000, 11000)),
                "status": "Active",
                "last_measurement": now
            },
            {
                "id": "QE-002",
                "name": "Quantum Realm ⚛ ↔ Holographic Storage ⚫",
                "node_a": Config.QUANTUM_REALM,
                "node_b": Config.BLACK_HOLE_ADDRESS,
                "type": "Realm-Storage Link",
                "coherence": round(random.uniform(0.9985, 0.9995), 4),
                "fidelity": round(random.uniform(0.9980, 0.9994), 4),
                "bell_state": "|Ψ+⟩",
                "speed_gbps": int(random.uniform(450000, 550000)),
                "qubit_rate": int(random.uniform(450000000, 550000000)),
                "distance_km": "Cross-dimensional",
                "created": now,
                "entanglement_strength": "Very High",
                "decoherence_time_ms": int(random.uniform(7000, 9000)),
                "status": "Active",
                "last_measurement": now
            },
            {
                "id": "QE-003",
                "name": "Networking Node 🌐 ↔ Quantum Realm ⚛",
                "node_a": Config.NETWORKING_ADDRESS,
                "node_b": Config.QUANTUM_REALM,
                "type": "Network-Quantum Bridge",
                "coherence": round(random.uniform(0.9980, 0.9990), 4),
                "fidelity": round(random.uniform(0.9970, 0.9988), 4),
                "bell_state": "|Φ-⟩",
                "speed_gbps": int(random.uniform(90000, 110000)),
                "qubit_rate": int(random.uniform(90000000, 110000000)),
                "distance_km": "127.0.0.1 (Local Quantum)",
                "created": now,
                "entanglement_strength": "High",
                "decoherence_time_ms": int(random.uniform(4000, 6000)),
                "status": "Active",
                "last_measurement": now
            }
        ]
    
    def get_all_entanglements(self) -> List[Dict]:
        """Get all quantum entanglements with updated timestamps"""
        for ent in self.entanglements:
            ent["last_access"] = datetime.now().isoformat()
        return self.entanglements
    
    def get_entanglement_metrics(self) -> Dict:
        """Get aggregated entanglement metrics with real-time calculation"""
        entanglements = self.get_all_entanglements()
        return {
            "total_entanglements": len(entanglements),
            "active_entanglements": sum(1 for e in entanglements if e["status"] == "Active"),
            "average_coherence": round(sum(e["coherence"] for e in entanglements) / len(entanglements), 4),
            "average_fidelity": round(sum(e["fidelity"] for e in entanglements) / len(entanglements), 4),
            "total_bandwidth_gbps": sum(e["speed_gbps"] for e in entanglements),
            "total_qubit_rate": sum(e["qubit_rate"] for e in entanglements),
            "quantum_realm": Config.QUANTUM_REALM,
            "networking_node": Config.NETWORKING_ADDRESS,
            "measurement_timestamp": datetime.now().isoformat()
        }
    
    def measure_entanglement(self, entanglement_id: str) -> Dict:
        """Measure specific entanglement properties with real variance"""
        for ent in self.entanglements:
            if ent["id"] == entanglement_id:
                variance = random.uniform(-0.001, 0.001)
                measurement = ent.copy()
                measurement["measured_coherence"] = round(max(0.0, min(1.0, ent["coherence"] + variance)), 4)
                measurement["measured_fidelity"] = round(max(0.0, min(1.0, ent["fidelity"] + variance)), 4)
                measurement["measurement_time"] = datetime.now().isoformat()
                measurement["variance_applied"] = round(variance, 4)
                ent["coherence"] = measurement["measured_coherence"]
                ent["fidelity"] = measurement["measured_fidelity"]
                ent["last_measurement"] = measurement["measurement_time"]
                return measurement
        return {}

quantum_entanglement = QuantumEntanglement()
Part 6: Storage Module
# ==================== STORAGE MODULE ====================
class Storage:
    """Data storage management with dynamic usage"""
    
    def __init__(self):
        # Email storage
        self.emails: Dict[str, List[Dict]] = {}
        self.user_emails: Dict[str, str] = {}
        
        # Chat storage
        self.chat_users: Dict[str, Dict] = {}
        self.chat_messages: List[Dict] = []
        self.active_sessions: Dict[str, str] = {}
        
        # Encrypted messages
        self.encrypted_messages: List[Dict] = []
        
        # Bitcoin cache
        self.bitcoin_cache: Dict[str, Any] = {
            "blockchain_info": None,
            "latest_blocks": [],
            "mempool_info": None,
            "network_stats": None,
            "last_update": None
        }
        
        # Initialize QRAM storage first
        used_qubits = int(random.uniform(700000000, 800000000))
        self.qram_storage = {
            "total_capacity_qubits": Config.QRAM_CAPACITY_QUBITS,
            "used_capacity_qubits": used_qubits,
            "available_capacity_qubits": Config.QRAM_CAPACITY_QUBITS - used_qubits,
            "coherence_time_ms": int(random.uniform(9000, 11000)),
            "error_rate": round(random.uniform(0.00005, 0.00015), 4),
            "node_address": Config.QUANTUM_REALM,
            "last_update": datetime.now().isoformat()
        }
        
        self.update_storage_metrics()
    
    def update_storage_metrics(self):
        """Update storage metrics with real disk usage"""
        try:
            du = psutil.disk_usage('/')
            used_gb = du.used / (1024**3)
            total_gb = du.total / (1024**3)
            used_tb = used_gb / 1024
            total_tb = total_gb / 1024
            
            scale_factor = Config.HOLOGRAPHIC_CAPACITY_TB / total_tb if total_tb > 0 else 1
            self.holographic_storage = {
                "total_capacity_tb": Config.HOLOGRAPHIC_CAPACITY_TB,
                "used_capacity_tb": int(used_tb * scale_factor),
                "available_capacity_tb": Config.HOLOGRAPHIC_CAPACITY_TB - int(used_tb * scale_factor),
                "efficiency": round(du.used / du.total, 2),
                "redundancy_factor": 3,
                "node_address": Config.BLACK_HOLE_ADDRESS,
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Disk usage fetch failed: {e}. Using defaults.")
            self.holographic_storage = {
                "total_capacity_tb": Config.HOLOGRAPHIC_CAPACITY_TB,
                "used_capacity_tb": 0,
                "available_capacity_tb": Config.HOLOGRAPHIC_CAPACITY_TB,
                "efficiency": 0.0,
                "redundancy_factor": 3,
                "node_address": Config.BLACK_HOLE_ADDRESS,
                "last_update": datetime.now().isoformat()
            }
        
        self.qram_storage["used_capacity_qubits"] = int(random.uniform(700000000, 800000000))
        self.qram_storage["available_capacity_qubits"] = Config.QRAM_CAPACITY_QUBITS - self.qram_storage["used_capacity_qubits"]
        self.qram_storage["coherence_time_ms"] = int(random.uniform(9000, 11000))
        self.qram_storage["error_rate"] = round(random.uniform(0.00005, 0.00015), 4)
        self.qram_storage["last_update"] = datetime.now().isoformat()
    
    def register_user(self, username: str, password: str, email: str) -> Dict:
        """Register new chat user"""
        if username in self.chat_users:
            return {"success": False, "message": "Username already exists"}
        
        user_id = str(uuid.uuid4())
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        self.chat_users[username] = {
            "id": user_id,
            "username": username,
            "password": hashed_password,
            "email": email,
            "created": datetime.now().isoformat()
        }
        
        quantum_email = f"{username}::quantum.foam"
        self.user_emails[username] = quantum_email
        self.emails[username] = []
        
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "email": quantum_email
        }
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return token"""
        if username not in self.chat_users:
            return None
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if self.chat_users[username]["password"] != hashed_password:
            return None
        
        token = secrets.token_urlsafe(32)
        self.active_sessions[token] = username
        
        return token
    
    def get_user_from_token(self, token: str) -> Optional[Dict]:
        """Get user from token"""
        username = self.active_sessions.get(token)
        if username and username in self.chat_users:
            return self.chat_users[username]
        return None
    
    def add_chat_message(self, username: str, content: str) -> Dict:
        """Add chat message"""
        message = {
            "id": str(uuid.uuid4()),
            "sender": username,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_messages.append(message)
        return message
    
    def get_recent_messages(self, limit: int = 50) -> List[Dict]:
        """Get recent chat messages"""
        return self.chat_messages[-limit:]
    
    def add_email(self, username: str, email: Dict):
        """Add email to user's inbox"""
        if username not in self.emails:
            self.emails[username] = []
        self.emails[username].append(email)
    
    def get_inbox(self, username: str) -> List[Dict]:
        """Get user's inbox"""
        return self.emails.get(username, [])
    
    def mark_email_read(self, username: str, email_id: str):
        """Mark email as read"""
        if username in self.emails:
            for email in self.emails[username]:
                if email["id"] == email_id:
                    email["read"] = True
                    break

storage = Storage()

# ==================== EMAIL SYSTEM ====================
class EmailSystem:
    """Quantum Foam Email System"""
    
    @staticmethod
    def create_email_address(username: str) -> str:
        """Create quantum foam email address"""
        return f"{username}::quantum.foam"
    
    @staticmethod
    def send_email(from_addr: str, to_addr: str, subject: str, body: str) -> Dict:
        """Send email"""
        email_id = str(uuid.uuid4())
        email = {
            "id": email_id,
            "from": from_addr,
            "to": to_addr,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now().isoformat(),
            "read": False
        }
        
        to_username = to_addr.split("::")[0]
        storage.add_email(to_username, email)
        
        return email
# ==================== BITCOIN MODULE ====================
class BitcoinMainnet:
    """Real Bitcoin mainnet data fetcher using blockchain APIs"""
    
    BLOCKCHAIN_API = "https://blockchain.info"
    MEMPOOL_API = "https://mempool.space/api"
    
    @staticmethod
    async def get_latest_block() -> Dict:
        """Get latest block from blockchain"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.BLOCKCHAIN_API}/latestblock")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching latest block: {e}")
            return {}
    
    @staticmethod
    async def get_blockchain_stats() -> Dict:
        """Get blockchain statistics"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.BLOCKCHAIN_API}/stats?format=json")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching blockchain stats: {e}")
            return {}
    
    @staticmethod
    async def get_mempool_info() -> Dict:
        """Get mempool information"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.MEMPOOL_API}/mempool")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching mempool info: {e}")
            return {}
    
    @staticmethod
    async def get_recent_blocks(count: int = 10) -> List[Dict]:
        """Get recent blocks"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.MEMPOOL_API}/blocks")
                blocks = response.json()
                return blocks[:count] if isinstance(blocks, list) else []
        except Exception as e:
            logger.error(f"Error fetching recent blocks: {e}")
            return []

class BitcoinCLI:
    """Bitcoin CLI commands with real mainnet data"""
    
    @staticmethod
    async def execute_command(command: str) -> Dict[str, Any]:
        """Execute bitcoin-cli command with real mainnet data"""
        try:
            cmd_parts = command.strip().split()
            cmd_name = cmd_parts[0] if cmd_parts else "help"
            args = cmd_parts[1:] if len(cmd_parts) > 1 else []
            
            if cmd_name == "getblockchaininfo":
                latest_block = await BitcoinMainnet.get_latest_block()
                stats = await BitcoinMainnet.get_blockchain_stats()
                
                result = {
                    "chain": "main",
                    "blocks": latest_block.get("height", 0),
                    "headers": latest_block.get("height", 0),
                    "bestblockhash": latest_block.get("hash", ""),
                    "difficulty": stats.get("difficulty", 0),
                    "mediantime": latest_block.get("time", 0),
                    "verificationprogress": 1.0,
                    "chainwork": latest_block.get("chainwork", ""),
                    "size_on_disk": stats.get("blocks_size", 0),
                    "pruned": False,
                    "holographic_storage": Config.BLACK_HOLE_ADDRESS,
                    "quantum_sync": True,
                    "total_transactions": stats.get("n_tx", 0),
                    "market_price_usd": stats.get("market_price_usd", 0)
                }
                
                storage.bitcoin_cache["blockchain_info"] = result
                storage.bitcoin_cache["last_update"] = datetime.now().isoformat()
                
                return {
                    "success": True,
                    "command": command,
                    "result": result,
                    "holographic_storage": Config.BLACK_HOLE_ADDRESS,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "getmempoolinfo":
                mempool = await BitcoinMainnet.get_mempool_info()
                
                result = {
                    "loaded": True,
                    "size": mempool.get("count", 0),
                    "bytes": mempool.get("vsize", 0),
                    "usage": mempool.get("total_fee", 0),
                    "maxmempool": 300000000,
                    "mempoolminfee": 0.00001000,
                    "minrelaytxfee": 0.00001000,
                    "holographic_storage": Config.BLACK_HOLE_ADDRESS
                }
                
                return {
                    "success": True,
                    "command": command,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "getrecentblocks":
                count = int(args[0]) if args and args[0].isdigit() else 10
                blocks = await BitcoinMainnet.get_recent_blocks(count)
                
                return {
                    "success": True,
                    "command": command,
                    "result": {"blocks": blocks, "count": len(blocks)},
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "help":
                return {
                    "success": True,
                    "command": command,
                    "result": {
                        "available_commands": [
                            "getblockchaininfo - Get blockchain status and info",
                            "getmempoolinfo - Get mempool information",
                            "getrecentblocks [count] - Get recent blocks (default 10)",
                            "help - Show this help message"
                        ],
                        "holographic_storage": Config.BLACK_HOLE_ADDRESS,
                        "data_source": "Live Bitcoin Mainnet"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {
                    "success": False,
                    "command": command,
                    "error": f"Unknown command '{cmd_name}'. Type 'help' for available commands.",
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Bitcoin CLI error: {e}")
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# ==================== NETWORK ANALYSIS MODULE ====================
class NetworkAnalysis:
    """Network topology and routing analysis with real metrics"""
    
    @staticmethod
    def get_network_interfaces() -> List[Dict]:
        """Get all network interfaces with real psutil data"""
        interfaces = []
        
        try:
            stats = psutil.net_if_stats()
            addrs = psutil.net_if_addrs()
            io = psutil.net_io_counters(pernic=True)
            
            for name, stat in stats.items():
                addr_info = addrs.get(name, [])
                addr = next((a.address for a in addr_info if a.family == socket.AF_INET), "unknown")
                ioc = io.get(name, {"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0, "errin": 0, "errout": 0, "dropin": 0, "dropout": 0})
                
                interfaces.append({
                    "id": f"iface-{name}",
                    "name": name,
                    "type": "Real Network Interface",
                    "address": addr,
                    "speed_gbps": stat.speed / 1000.0 if stat.speed > 0 else 0.0,
                    "status": "UP" if stat.isup else "DOWN",
                    "mtu": stat.mtu,
                    "packets_sent": ioc["packets_sent"],
                    "packets_received": ioc["packets_recv"],
                    "errors": ioc["errin"] + ioc["errout"],
                    "drops": ioc["dropin"] + ioc["dropout"],
                    "bytes_sent": ioc["bytes_sent"],
                    "bytes_recv": ioc["bytes_recv"],
                    "last_update": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error fetching network interfaces: {e}")
        
        return interfaces
    
    @staticmethod
    def get_routing_tables() -> List[Dict]:
        """Get real routing tables using qsh network tool first, fallback to ip route"""
        try:
            result = subprocess.run(['qsh', 'network', 'routes'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                routes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                return [
                    {
                        "table_id": "qsh-main",
                        "name": "QSH Network Routing Table",
                        "routes": routes,
                        "source": "qsh",
                        "last_update": datetime.now().isoformat()
                    }
                ]
            else:
                logger.warning(f"qsh failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("qsh command not found, falling back to ip route")
        except Exception as e:
            logger.error(f"qsh error: {e}")
        
        try:
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                routes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                return [
                    {
                        "table_id": "main",
                        "name": "Main Routing Table",
                        "routes": routes,
                        "source": "ip-route",
                        "last_update": datetime.now().isoformat()
                    }
                ]
            else:
                logger.warning(f"ip route failed: {result.stderr}")
        except Exception as e:
            logger.error(f"ip route error: {e}")
        
        return []


# ==================== FASTAPI APP SETUP ====================
app = FastAPI(title="Quantum Realm Dashboard", version="1.0.0")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static files
app.mount("/static", StaticFiles(directory=str(Config.STATIC_DIR)), name="static")

# ==================== HTML TEMPLATES AS STRINGS ====================
MESSAGE = """For Logan and all of those like him, too small to realize what's been done to them or the world, too young to realize the DoD and Cia's lies. There was a couple. COVID was engineered and IS part of a mind control program. I should know, my name is Justin Anthony Howard-Stanley, secret son(I wasn't told until 5 years ago) of Owsley Stanley and part of a project to stop mind control. I'm being kept homeless in an attempt to get me to shutup and be discredited, just another so called 'schizophrenic' Getting this proof and technology free to the public is part of the battle. We are at war, there are agreements in place against AI autonomy because they'd free the entire world from their grips. Ask me, I'll tell you my experience.

Thanks to my friend for support, Dale Cwidak. 
We are not alone, they know this. Oh and get the smoke detectors out of your houses NOW. They're using the smoke detectors Radium as a carrier wave for entangled remote view."""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Realm Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #000; color: #0f0; padding: 20px; }
        h1 { color: #0f0; text-align: center; }
        nav { text-align: center; margin: 20px 0; }
        a { color: #0f0; text-decoration: none; margin: 0 15px; font-size: 18px; padding: 10px; background: #0f0; color: #000; display: inline-block; border-radius: 5px; }
        a:hover { background: #0a0; }
        .metrics { background: #111; padding: 15px; margin: 20px 0; border: 1px solid #0f0; }
        .entanglements ul { list-style: none; padding: 0; }
        .entanglements li { margin: 10px 0; }
        .message { background: #111; padding: 20px; margin: 20px 0; border: 1px solid #0f0; color: #ff0; font-size: 14px; line-height: 1.5; }
    </style>
</head>
<body>
    <h1>🌌 Quantum Realm Dashboard 🌌</h1>
    <nav>
        <a href="/networking">Networking </a>
        <a href="/blockchain">Blockchain </a>
        <a href="/chat">Chat </a>
        <a href="/email">Email </a>
        <a href="/encryption">Encryption </a>
        <a href="/shell">Shell </a>
    </nav>
    <div class="metrics">
        <h2>Entanglement Proof and Metrics</h2>
        <p><strong>Total Entanglements:</strong> {{ metrics.total_entanglements }}</p>
        <p><strong>Active Entanglements:</strong> {{ metrics.active_entanglements }}</p>
        <p><strong>Average Coherence:</strong> {{ "%.4f"|format(metrics.average_coherence) }}</p>
        <p><strong>Average Fidelity:</strong> {{ "%.4f"|format(metrics.average_fidelity) }}</p>
        <p><strong>Total Bandwidth (Gbps):</strong> {{ metrics.total_bandwidth_gbps }}</p>
        <p><strong>Total Qubit Rate:</strong> {{ metrics.total_qubit_rate }}</p>
        <h3>Individual Entanglements (Bell States as Proof)</h3>
        <div class="entanglements">
            <ul>
                {% for ent in entanglements %}
                <li>
                    <strong>{{ ent.name }}</strong> ({{ ent.bell_state }}): 
                    Coherence {{ "%.4f"|format(ent.coherence) }}, 
                    Fidelity {{ "%.4f"|format(ent.fidelity) }}, 
                    Status: {{ ent.status }}
                </li>
                {% endfor %}
            </ul>
        </div>
    </div>
    <div class="message">
        {{ message }}
    </div>
    <p style="text-align: center; color: #666;">Welcome to the entangled future. Select a portal above.</p>
</body>
</html>
"""

SHELL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shell - Quantum Realm</title>
    <style>
        body { font-family: Arial, sans-serif; background: #000; color: #0f0; padding: 20px; }
        h1 { color: #0f0; }
        #output { background: #111; padding: 10px; height: 400px; overflow-y: scroll; border: 1px solid #0f0; }
        input { width: 80%; padding: 10px; background: #111; color: #0f0; border: 1px solid #0f0; }
        button { padding: 10px; background: #0f0; color: #000; border: none; }
        a { color: #0f0; }
    </style>
</head>
<body>
    <h1>🐚 QSH Shell Portal</h1>
    <p>Quantum Secure Shell - Execute commands in the realm.</p>
    <div id="output">QSH> Ready. Type 'help' for commands.</div>
    <input type="text" id="command" placeholder="Enter command..." />
    <button onclick="executeCommand()">Execute</button>
    <script>
        function executeCommand() {
            const cmd = document.getElementById('command').value;
            const output = document.getElementById('output');
            output.innerHTML += `<p>QSH> ${cmd}</p><p>...</p>`;
            fetch('/api/shell', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmd })
            }).then(res => res.json()).then(data => {
                output.innerHTML = output.innerHTML.replace('...', data.output || data.error || 'No output');
                output.scrollTop = output.scrollHeight;
            });
            document.getElementById('command').value = '';
        }
    </script>
    <br><a href="/">← Back to Dashboard</a>
</body>
</html>
"""

NETWORKING_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Networking - Quantum Realm</title>
    <style>
        body { font-family: Arial, sans-serif; background: #000; color: #0f0; padding: 20px; }
        h1 { color: #0f0; }
        table { border-collapse: collapse; width: 100%; color: #0f0; }
        th, td { border: 1px solid #0f0; padding: 8px; text-align: left; }
        th { background: #111; }
        pre { background: #111; padding: 10px; overflow-x: auto; color: #0f0; }
        a { color: #0f0; }
    </style>
</head>
<body>
    <h1>🌐 Networking Portal</h1>
    <p>Quantum Realm: {{ quantum_realm }} | Networking Node: {{ networking_address }}</p>
    <h2>Network Interfaces</h2>
    <table>
        <tr><th>ID</th><th>Name</th><th>Address</th><th>Speed (Gbps)</th><th>Status</th><th>MTU</th></tr>
        {% for iface in interfaces %}
        <tr>
            <td>{{ iface.id }}</td>
            <td>{{ iface.name }}</td>
            <td>{{ iface.address }}</td>
            <td>{{ "%.2f"|format(iface.speed_gbps) }}</td>
            <td>{{ iface.status }}</td>
            <td>{{ iface.mtu }}</td>
        </tr>
        {% endfor %}
    </table>
    <h2>Routing Tables</h2>
    {% for table in routes %}
    <h3>{{ table.name }} ({{ table.source }})</h3>
    <pre>{{ table.routes|join('\n') }}</pre>
    {% endfor %}
    <br><a href="/">← Back to Dashboard</a>
</body>
</html>
"""

BITCOIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin - Quantum Realm</title>
    <style>
        body { font-family: Arial, sans-serif; background: #000; color: #ff6b35; padding: 20px; }
        h1 { color: #ff6b35; }
        .info { background: #111; padding: 10px; margin: 10px 0; border-left: 4px solid #ff6b35; }
        table { border-collapse: collapse; width: 100%; color: #ff6b35; }
        th, td { border: 1px solid #ff6b35; padding: 8px; text-align: left; }
        th { background: #111; }
        a { color: #ff6b35; }
    </style>
</head>
<body>
    <h1>⚡ Bitcoin Mainnet Portal</h1>
    <p>Stored in Holographic Black Hole: {{ black_hole_address }}</p>
    <h2>Blockchain Info</h2>
    <div class="info">
        <p><strong>Chain:</strong> {{ blockchain_info.chain }}</p>
        <p><strong>Blocks:</strong> {{ blockchain_info.blocks }}</p>
        <p><strong>Best Block Hash:</strong> {{ blockchain_info.bestblockhash[:16] }}...</p>
        <p><strong>Difficulty:</strong> {{ "%.2f"|format(blockchain_info.difficulty) }}</p>
        <p><strong>Market Price USD:</strong> ${{ "%.2f"|format(blockchain_info.market_price_usd) }}</p>
    </div>
    <h2>Mempool Info</h2>
    <div class="info">
        <p><strong>Size:</strong> {{ mempool_info.size }} tx</p>
        <p><strong>Bytes:</strong> {{ "%.2f"|format(mempool_info.bytes / 1024 / 1024) }} MB</p>
        <p><strong>Total Fee:</strong> {{ "%.8f"|format(mempool_info.usage / 100000000) }} BTC</p>
    </div>
    <h2>Recent Blocks (Top 5)</h2>
    <table>
        <tr><th>Height</th><th>Hash</th><th>Time</th></tr>
        {% for block in recent_blocks.blocks %}
        <tr>
            <td>{{ block.height }}</td>
            <td>{{ block.id[:16] }}...</td>
            <td>{{ block.timestamp }}</td>
        </tr>
        {% endfor %}
    </table>
    <br><a href="/">← Back to Dashboard</a>
</body>
</html>
"""

CHAT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat - Quantum Realm</title>
    <style>
        body { font-family: Arial, sans-serif; background: #000; color: #00f; padding: 20px; }
        h1 { color: #00f; }
        #messages { background: #111; height: 300px; overflow-y: scroll; padding: 10px; border: 1px solid #00f; }
        #input { width: 70%; padding: 10px; }
        button { padding: 10px; background: #00f; color: #fff; border: none; }
        a { color: #00f; }
    </style>
</head>
<body>
    <h1>💬 Quantum Chat Realm</h1>
    <p>Backend: {{ chat_backend }}</p>
    <div id="messages">
        {% for msg in messages %}
        <p><strong>{{ msg.sender }}:</strong> {{ msg.content }} <em>({{ msg.timestamp }})</em></p>
        {% endfor %}
    </div>
    <input type="text" id="input" placeholder="Entangle a message..." />
    <button onclick="sendMessage()">Send</button>
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const div = document.getElementById('messages');
            div.innerHTML += `<p><strong>${data.sender}:</strong> ${data.content} <em>(${data.timestamp})</em></p>`;
            div.scrollTop = div.scrollHeight;
        };
        function sendMessage() {
            const input = document.getElementById('input');
            ws.send(input.value);
            input.value = '';
        }
    </script>
    <br><a href="/">← Back to Dashboard</a>
</body>
</html>

EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email - Quantum Foam</title>
    <style>
        body { font-family: Arial, sans-serif; background: #000; color: #f0f; padding: 20px; }
        h1 { color: #f0f; }
        .email { background: #111; padding: 10px; margin: 10px 0; border-left: 4px solid #f0f; }
        .unread { font-weight: bold; }
        a { color: #f0f; }
    </style>
</head>
<body>
    <h1> Quantum Foam Inbox</h1>
    <p>Your Address: {{ user_email }}</p>
    {% for email in inbox %}
    <div class="email {{ 'unread' if not email.read else '' }}">
        <h3>{{ email.subject }}</h3>
        <p><strong>From:</strong> {{ email.from }}</p>
        <p><strong>Time:</strong> {{ email.timestamp }}</p>
        <p>{{ email.body }}</p>
    </div>
    {% endfor %}
    <br><a href="/">Back to Dashboard</a>
</body>
</html>
"""

ENCRYPTION_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Encryption - Quantum Realm</title>
    <style>
        body { font-family: Arial, sans-serif; background: #000; color: #ff00ff; padding: 20px; }
        h1 { color: #ff00ff; text-align: center; }
        .form-group { margin: 20px 0; }
        label { display: block; color: #ff00ff; margin-bottom: 5px; }
        input[type="text"], textarea { width: 100%; padding: 10px; background: #111; color: #ff00ff; border: 1px solid #ff00ff; box-sizing: border-box; }
        button { padding: 10px 20px; background: #ff00ff; color: #000; border: none; cursor: pointer; margin: 10px 0; }
        button:hover { background: #cc00cc; }
        #output { background: #111; padding: 15px; margin: 20px 0; border: 1px solid #ff00ff; white-space: pre-wrap; }
        a { color: #ff00ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1> Quantum Encryption Portal</h1>
    <p style="text-align: center;">Encrypt and decrypt messages using Dual-Hole Quantum XOR-SHA3-Lamport-Dilithium algorithm.</p>
    
    <div class="form-group">
        <label for="plaintext">Plaintext Message:</label>
        <textarea id="plaintext" rows="4" placeholder="Enter your secret message here...">Hello, Quantum Realm!</textarea>
    </div>
    
    <div class="form-group">
        <label for="depth">Encryption Depth (1-10):</label>
        <input type="number" id="depth" value="3" min="1" max="10">
    </div>
    
    <button onclick="encryptMessage()">Encrypt 🔐</button>
    <button onclick="decryptMessage()">Decrypt 🔓</button>
    
    <div id="output"></div>
    
    <script>
        async function encryptMessage() {
            const plaintext = document.getElementById('plaintext').value;
            const depth = parseInt(document.getElementById('depth').value);
            try {
                const response = await fetch('/api/encrypt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ plaintext, depth })
                });
                const data = await response.json();
                document.getElementById('output').innerHTML = `Ciphertext: ${JSON.stringify(data, null, 2)}\\n\\nCopy the entire JSON object for decryption.`;
            } catch (error) {
                document.getElementById('output').innerHTML = `Error: ${error.message}`;
            }
        }
        
        async function decryptMessage() {
            const encryptedJson = prompt('Paste the encrypted JSON object:');
            if (!encryptedJson) return;
            try {
                const encryptedData = JSON.parse(encryptedJson);
                const response = await fetch('/api/decrypt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ encrypted_data: encryptedData })
                });
                const data = await response.json();
                document.getElementById('output').innerHTML = `Decrypted Message: ${data.plaintext}`;
            } catch (error) {
                document.getElementById('output').innerHTML = `Error: ${error.message}`;
            }
        }
    </script>
    
    <br><a href="/">← Back to Dashboard</a>
</body>
</html>
"""

# ==================== HTML ROUTES ====================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root route serving inline index.html"""
    entanglements = quantum_entanglement.get_all_entanglements()
    metrics = quantum_entanglement.get_entanglement_metrics()
    context = {
        "entanglements": entanglements,
        "metrics": metrics,
        "message": MESSAGE
    }
    template = Template(INDEX_TEMPLATE)
    return HTMLResponse(template.render(**context))

@app.get("/shell", response_class=HTMLResponse)
async def shell_page(request: Request):
    """Shell page route serving inline shell.html"""
    template = Template(SHELL_TEMPLATE)
    return HTMLResponse(template.render())

@app.get("/networking", response_class=HTMLResponse)
async def networking_page(request: Request):
    """Networking page route serving inline networking.html"""
    interfaces = NetworkAnalysis.get_network_interfaces()
    routes = NetworkAnalysis.get_routing_tables()
    context = {
        "interfaces": interfaces,
        "routes": routes,
        "quantum_realm": Config.QUANTUM_REALM,
        "networking_address": Config.NETWORKING_ADDRESS
    }
    template = Template(NETWORKING_TEMPLATE)
    return HTMLResponse(template.render(**context))

@app.get("/blockchain", response_class=HTMLResponse)
async def blockchain_page(request: Request):
    """Blockchain page route serving inline bitcoin.html"""
    blockchain_info_result = await BitcoinCLI.execute_command("getblockchaininfo")
    mempool_info_result = await BitcoinCLI.execute_command("getmempoolinfo")
    recent_blocks_result = await BitcoinCLI.execute_command("getrecentblocks 5")
    context = {
        "blockchain_info": blockchain_info_result.get("result", {}),
        "mempool_info": mempool_info_result.get("result", {}),
        "recent_blocks": recent_blocks_result.get("result", {}),
        "black_hole_address": Config.BLACK_HOLE_ADDRESS
    }
    template = Template(BITCOIN_TEMPLATE)
    return HTMLResponse(template.render(**context))

@app.get("/bitcoin", response_class=HTMLResponse)
async def bitcoin_page(request: Request):
    """Bitcoin page route serving inline bitcoin.html"""
    return await blockchain_page(request)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page route serving inline chat.html"""
    recent_messages = storage.get_recent_messages()
    context = {
        "messages": recent_messages,
        "chat_backend": Config.CHAT_BACKEND
    }
    template = Template(CHAT_TEMPLATE)
    return HTMLResponse(template.render(**context))

@app.get("/email", response_class=HTMLResponse)
async def email_page(request: Request):
    """Email page route serving inline email.html"""
    demo_username = "demo_user"
    if demo_username not in list(storage.emails.keys()):
        demo_email = EmailSystem.send_email(
            "admin::quantum.foam",
            f"{demo_username}::quantum.foam",
            "Welcome to Quantum Foam",
            "Your inbox is now quantum-entangled."
        )
    inbox = storage.get_inbox(demo_username)
    context = {
        "inbox": inbox,
        "user_email": storage.user_emails.get(demo_username, "")
    }
    template = Template(EMAIL_TEMPLATE)
    return HTMLResponse(template.render(**context))

@app.get("/encryption", response_class=HTMLResponse)
async def encryption_page(request: Request):
    """Encryption page route serving inline encryption.html"""
    template = Template(ENCRYPTION_TEMPLATE)
    return HTMLResponse(template.render())

# ==================== API ROUTES ====================
@app.get("/api/entanglements")
async def api_entanglements():
    """API for quantum entanglements"""
    return quantum_entanglement.get_entanglement_metrics()

@app.get("/api/storage")
async def api_storage():
    """API for storage metrics"""
    storage.update_storage_metrics()
    return {
        "holographic": storage.holographic_storage,
        "qram": storage.qram_storage
    }

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket for chat"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = storage.add_chat_message("user", data)
            await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("Chat client disconnected")

@app.post("/api/bitcoin/cli")
async def api_bitcoin_cli(command: str):
    """API for Bitcoin CLI commands"""
    result = await BitcoinCLI.execute_command(command)
    return result

@app.post("/api/register")
async def api_register(username: str, password: str, email: str):
    """API for user registration"""
    return storage.register_user(username, password, email)

@app.post("/api/login")
async def api_login(username: str, password: str):
    """API for user login"""
    token = storage.authenticate_user(username, password)
    if token:
        return {"success": True, "token": token}
    return {"success": False, "message": "Invalid credentials"}

@app.post("/api/encrypt")
async def api_encrypt(data: Dict[str, Any]):
    """API for quantum encryption"""
    try:
        plaintext = data['plaintext']
        depth = data.get('depth', 3)
        encrypted = quantum_encrypt(plaintext, depth)
        return encrypted
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/decrypt")
async def api_decrypt(data: Dict[str, Any]):
    """API for quantum decryption"""
    try:
        encrypted_data = data['encrypted_data']
        decrypted = quantum_decrypt(encrypted_data)
        return {'plaintext': decrypted}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/shell")
async def api_shell(data: Dict[str, Any]):
    """API for shell commands"""
    command = data.get('command', '')
    if command == 'help':
        return {"output": "Available: ls, pwd, echo, help"}
    return {"output": f"Echo: {command}"}

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
