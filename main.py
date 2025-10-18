import os
import logging
import hashlib
import base64
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import httpx
import asyncio
from contextlib import asynccontextmanager
import secrets
from collections import defaultdict
import random
import psutil
import subprocess
from jinja2 import Template
import socket
import sqlite3
import re
import sys
from io import StringIO

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
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    CHAT_BACKEND = os.getenv("CHAT_BACKEND_URL", "https://clearnet-chat-4bal.onrender.com")
    SKIP_BACKEND_CHECKS = os.getenv("SKIP_BACKEND_CHECKS", "true").lower() == "true"
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = int(os.getenv("TIMEOUT", "30"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    BLACK_HOLE_ADDRESS = "138.0.0.1"
    WHITE_HOLE_ADDRESS = "139.0.0.1"
    QUANTUM_REALM = "quantum.realm.domain.dominion.foam.computer.alice"
    NETWORKING_ADDRESS = "quantum.realm.domain.dominion.foam.computer.networking"
    BITCOIN_UPDATE_INTERVAL = int(os.getenv("BITCOIN_UPDATE_INTERVAL", "30"))
    BITCOIN_RPC_USER = os.getenv("BITCOIN_RPC_USER", "hackah")
    BITCOIN_RPC_PASS = os.getenv("BITCOIN_RPC_PASS", "hackah")
    HOLOGRAPHIC_CAPACITY_TB = 138000
    QRAM_CAPACITY_QUBITS = 1000000000
    TEMPLATES_DIR = Path(".")
    STATIC_DIR = Path("static")
    UPLOADS_DIR = Path("uploads")
    DB_PATH = Path("quantum_foam.db")
    ADMINISTRATOR_USERNAME = os.getenv("ADMINISTRATOR_USERNAME", "eaafb486-f288-4011-a11f-7d7fcc1d99d5")
    ADMINISTRATOR_PASSWORD = os.getenv("ADMINISTRATOR_PASSWORD", "9f792277-5057-4642-bca0-97e778c5c7b9")

Config.STATIC_DIR.mkdir(exist_ok=True)
Config.UPLOADS_DIR.mkdir(exist_ok=True)

# ==================== PQC LAMPORT SIGNATURE MODULE ====================
def lamport_keygen(n=256):
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
    m_hash = hashlib.sha256(message).digest()
    bits = [(m_hash[i // 8] >> (7 - (i % 8))) & 1 for i in range(256)]
    sig = b''
    for i, b in enumerate(bits):
        sig += sk[i][b]
    return sig

def lamport_verify(message: bytes, sig: bytes, pk: list) -> bool:
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

try:
    from dilithium import Dilithium2
    DILITHIUM_AVAILABLE = True
except ImportError:
    DILITHIUM_AVAILABLE = False
    class Dilithium2:
        @staticmethod
        def keygen():
            return None, None
        @staticmethod
        def sign(msg, sk):
            return b''
        @staticmethod
        def verify(msg, sig, pk):
            return False

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

def quantum_hash(credentials: str) -> str:
    """Generate quantum hash for credentials verification"""
    return hashlib.sha3_512(credentials.encode()).hexdigest()

# ==================== DATABASE MODULE ====================
class Database:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self.setup_tables()
    
    def setup_tables(self):
        cursor = self.conn.cursor()
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                user TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                user TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_creds (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                hashed_pass TEXT NOT NULL,
                plaintext_pass TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qsh_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                command TEXT NOT NULL,
                output TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hackers_logged (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                attempted_username TEXT NOT NULL,
                attempted_password TEXT NOT NULL,
                expected_username TEXT NOT NULL,
                expected_password TEXT NOT NULL,
                ip_address TEXT,
                storage_node TEXT,
                error_reason TEXT,
                quantum_id TEXT,
                normal_id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS holographic_email_blocks (
                block_ip TEXT PRIMARY KEY,
                allocated BOOLEAN DEFAULT FALSE,
                used_gb REAL DEFAULT 0.0,
                capacity_gb REAL DEFAULT 10.0,
                created DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS holographic_email_users (
                holo_email TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                hashed_password TEXT NOT NULL,
                created DATETIME DEFAULT CURRENT_TIMESTAMP,
                block_ip TEXT NOT NULL,
                FOREIGN KEY(block_ip) REFERENCES holographic_email_blocks(block_ip)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notebooks (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                folder TEXT DEFAULT 'root',
                title TEXT DEFAULT 'Untitled',
                content_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS holographic_files (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                quantum_hash TEXT NOT NULL
            )
        """)
        default_folders = [
            (str(uuid.uuid4()), 'Inbox', 'all_users'),
            (str(uuid.uuid4()), 'Sent', 'all_users'),
            (str(uuid.uuid4()), 'Drafts', 'all_users'),
            (str(uuid.uuid4()), 'Trash', 'all_users')
        ]
        for folder_id, name, user in default_folders:
            cursor.execute("INSERT OR IGNORE INTO folders (id, name, user) VALUES (?, ?, ?)", (folder_id, name, user))
        self.conn.commit()
    
    def holographic_store(self, data: Dict[str, Any]) -> str:
        variance = round(random.uniform(0.9990, 0.9999), 4)
        hashed = hashlib.sha256(json.dumps(data).encode()).digest()
        encoded = base64.b64encode(hashed).decode()
        return f"{encoded}:{variance}"
    
    def holo_search(self, username: str, query: str, folder: str = 'Inbox') -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        pattern = f"%{query}%"
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
    
    def log_hacker_attempt(self, attempt_data: Dict[str, Any]):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO hackers_logged 
            (attempted_username, attempted_password, expected_username, expected_password, 
             ip_address, storage_node, error_reason, quantum_id, normal_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attempt_data.get('attempted_username', ''),
            attempt_data.get('attempted_password', ''),
            attempt_data.get('expected_username', ''),
            attempt_data.get('expected_password', ''),
            attempt_data.get('ip_address', ''),
            attempt_data.get('storage_node', ''),
            attempt_data.get('error_reason', ''),
            attempt_data.get('quantum_id', ''),
            attempt_data.get('normal_id', '')
        ))
        self.conn.commit()
    
    def get_hacker_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM hackers_logged 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def mark_read(self, email_id: str, read: bool = True):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE emails SET read = ? WHERE id = ?", (read, email_id))
        self.conn.commit()
    
    def delete_emails(self, email_ids: List[str]):
        cursor = self.conn.cursor()
        cursor.executemany("DELETE FROM emails WHERE id = ?", [(eid,) for eid in email_ids])
        self.conn.commit()

    def save_notebook(self, user_email: str, folder: str, title: str, content_json: str) -> str:
        notebook_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO notebooks (id, user_email, folder, title, content_json)
            VALUES (?, ?, ?, ?, ?)
        """, (notebook_id, user_email, folder, title, content_json))
        self.conn.commit()
        return notebook_id
    
    def load_notebook(self, user_email: str, folder: str = 'root') -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM notebooks 
            WHERE user_email = ? AND folder = ? 
            ORDER BY timestamp DESC
        """, (user_email, folder))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_holo_files(self, user_email: str, search_query: str = '') -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        if search_query:
            cursor.execute("""
                SELECT * FROM holographic_files 
                WHERE user_email = ? AND (filename LIKE ? OR file_path LIKE ?)
                ORDER BY upload_timestamp DESC
            """, (user_email, f"%{search_query}%", f"%{search_query}%"))
        else:
            cursor.execute("""
                SELECT * FROM holographic_files 
                WHERE user_email = ? 
                ORDER BY upload_timestamp DESC
            """, (user_email,))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def save_holo_file(self, user_email: str, filename: str, file_path: str, size_bytes: int, quantum_hash: str):
        cursor = self.conn.cursor()
        file_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO holographic_files (id, user_email, filename, file_path, size_bytes, quantum_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_id, user_email, filename, file_path, size_bytes, quantum_hash))
        self.conn.commit()
        return file_id
    
    def verify_holo_login(self, ip: str, username: str, password: str) -> bool:
        credentials = f"{ip}:{username}:{password}"
        user_hash = quantum_hash(credentials)
        # Simulate quantum hash check - in real impl, compare with stored hash
        stored_hash = quantum_hash(f"{ip}:{username}:{Config.ADMINISTRATOR_PASSWORD}")  # Example
        return user_hash == stored_hash

db = None

# ==================== HOLOGRAPHIC EMAIL STORAGE MODULE ====================
class HolographicEmailStorage:
    def __init__(self):
        self.base_ip = "137.0.0"
        self.block_size_gb = 10
        self.blocks = {}
        self.user_assignments = {}
    
    def allocate_block_for_user(self, username: str, password: str) -> str:
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT block_ip FROM holographic_email_blocks 
            WHERE allocated = FALSE 
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            block_ip = row[0]
        else:
            cursor.execute("SELECT COUNT(*) FROM holographic_email_blocks")
            block_count = cursor.fetchone()[0]
            block_number = block_count + 1
            block_ip = f"{self.base_ip}.{block_number}"
            cursor.execute("""
                INSERT INTO holographic_email_blocks (block_ip, allocated, capacity_gb)
                VALUES (?, TRUE, ?)
            """, (block_ip, self.block_size_gb))
        holo_email = f"{username}::quantum.foam"
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("""
            INSERT INTO holographic_email_users 
            (holo_email, username, hashed_password, block_ip)
            VALUES (?, ?, ?, ?)
        """, (holo_email, username, hashed_password, block_ip))
        cursor.execute("""
            UPDATE holographic_email_blocks 
            SET allocated = TRUE 
            WHERE block_ip = ?
        """, (block_ip,))
        db.conn.commit()
        self.user_assignments[username] = block_ip
        return block_ip
    
    def get_user_block(self, username: str) -> Optional[str]:
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT block_ip FROM holographic_email_users 
            WHERE username = ?
        """, (username,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_all_blocks(self) -> List[Dict]:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM holographic_email_blocks")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def get_all_users(self) -> List[Dict]:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM holographic_email_users")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

holo_storage = None

# ==================== QUANTUM ENTANGLEMENT MODULE ====================
class QuantumEntanglement:
    def __init__(self):
        self.entanglements = []
        self.initialize_entanglements()
    
    def initialize_entanglements(self):
        now = datetime.now().isoformat()
        self.entanglements = [
            {
                "id": "QE-001",
                "name": "Black Hole <-> White Hole",
                "node_a": Config.BLACK_HOLE_ADDRESS,
                "node_b": Config.WHITE_HOLE_ADDRESS,
                "type": "Wormhole Bridge",
                "coherence": round(random.uniform(0.9990, 0.9999), 4),
                "fidelity": round(random.uniform(0.9980, 0.9998), 4),
                "bell_state": "|Phi+>",
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
                "name": "Quantum Realm <-> Holographic Storage",
                "node_a": Config.QUANTUM_REALM,
                "node_b": Config.BLACK_HOLE_ADDRESS,
                "type": "Realm-Storage Link",
                "coherence": round(random.uniform(0.9985, 0.9995), 4),
                "fidelity": round(random.uniform(0.9980, 0.9994), 4),
                "bell_state": "|Psi+>",
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
                "name": "Networking Node <-> Quantum Realm",
                "node_a": Config.NETWORKING_ADDRESS,
                "node_b": Config.QUANTUM_REALM,
                "type": "Network-Quantum Bridge",
                "coherence": round(random.uniform(0.9980, 0.9990), 4),
                "fidelity": round(random.uniform(0.9970, 0.9988), 4),
                "bell_state": "|Phi->",
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
        for ent in self.entanglements:
            ent["last_access"] = datetime.now().isoformat()
        return self.entanglements
    
    def get_entanglement_metrics(self) -> Dict:
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

quantum_entanglement = QuantumEntanglement()

# ==================== STORAGE MODULE ====================
class Storage:
    def __init__(self):
        self.emails: Dict[str, List[Dict]] = {}
        self.user_emails: Dict[str, str] = {}
        self.chat_users: Dict[str, Dict] = {}
        self.chat_messages: List[Dict] = []
        self.active_sessions: Dict[str, str] = {}
        self.encrypted_messages: List[Dict] = []
        self.bitcoin_cache: Dict[str, Any] = {
            "blockchain_info": None,
            "latest_blocks": [],
            "mempool_info": None,
            "network_stats": None,
            "last_update": None
        }
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
        holo_storage.allocate_block_for_user(username, password)
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "email": quantum_email
        }
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        if username not in self.chat_users:
            return None
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if self.chat_users[username]["password"] != hashed_password:
            return None
        token = secrets.token_urlsafe(32)
        self.active_sessions[token] = username
        return token
    
    def get_user_from_token(self, token: str) -> Optional[Dict]:
        username = self.active_sessions.get(token)
        if username and username in self.chat_users:
            return self.chat_users[username]
        return None
    
    def add_chat_message(self, username: str, content: str) -> Dict:
        message = {
            "id": str(uuid.uuid4()),
            "sender": username,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_messages.append(message)
        return message
    
    def get_recent_messages(self, limit: int = 50) -> List[Dict]:
        return self.chat_messages[-limit:]
    
    def add_email(self, username: str, email: Dict):
        if username not in self.emails:
            self.emails[username] = []
        self.emails[username].append(email)
    
    def get_inbox(self, username: str) -> List[Dict]:
        return self.emails.get(username, [])
    
    def mark_email_read(self, username: str, email_id: str):
        if username in self.emails:
            for email in self.emails[username]:
                if email["id"] == email_id:
                    email["read"] = True
                    break

storage = Storage()

# ==================== EMAIL SYSTEM ====================
class EmailSystem:
    @staticmethod
    def create_email_address(username: str) -> str:
        return f"{username}::quantum.foam"
    
    @staticmethod
    def send_email(from_addr: str, to_addr: str, subject: str, body: str) -> Dict:
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
    BLOCKCHAIN_API = "https://blockchain.com"
    MEMPOOL_API = "https://mempool.space/api"
    
    @staticmethod
    async def get_latest_block() -> Dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.BLOCKCHAIN_API}/latestblock")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching latest block: {e}")
            return {}
    
    @staticmethod
    async def get_blockchain_stats() -> Dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.BLOCKCHAIN_API}/q/stats")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching blockchain stats: {e}")
            return {}
    
    @staticmethod
    async def get_mempool_info() -> Dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.MEMPOOL_API}/mempool")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching mempool info: {e}")
            return {}
    
    @staticmethod
    async def get_recent_blocks(count: int = 10) -> List[Dict]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.MEMPOOL_API}/blocks")
                blocks = response.json()
                return blocks[:count] if isinstance(blocks, list) else []
        except Exception as e:
            logger.error(f"Error fetching recent blocks: {e}")
            return []

class BitcoinCLI:
    @staticmethod
    async def execute_command(command: str) -> Dict[str, Any]:
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
    @staticmethod
    def get_network_interfaces() -> List[Dict]:
        interfaces = []
        try:
            stats = psutil.net_if_stats()
            addrs = psutil.net_if_addrs()
            io = psutil.net_io_counters(pernic=True)
            for name, stat in stats.items():
                addr_info = addrs.get(name, [])
                addr = next((a.address for a in addr_info if a.family == socket.AF_INET), "unknown")
                ioc = io.get(name)
                if not ioc:
                    ioc = type('obj', (object,), {
                        'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 
                        'packets_recv': 0, 'errin': 0, 'errout': 0, 'dropin': 0, 'dropout': 0
                    })()
                interfaces.append({
                    "id": f"iface-{name}",
                    "name": name,
                    "type": "Real Network Interface",
                    "address": addr,
                    "speed_gbps": stat.speed / 1000.0 if stat.speed > 0 else 0.0,
                    "status": "UP" if stat.isup else "DOWN",
                    "mtu": stat.mtu,
                    "packets_sent": ioc.packets_sent,
                    "packets_received": ioc.packets_recv,
                    "errors": ioc.errin + ioc.errout,
                    "drops": ioc.dropin + ioc.dropout,
                    "bytes_sent": ioc.bytes_sent,
                    "bytes_recv": ioc.bytes_recv,
                    "last_update": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error fetching network interfaces: {e}")
        return interfaces
    
    @staticmethod
    def get_routing_tables() -> List[Dict]:
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

# ==================== JUPYTER MODULE ====================
jupyter_namespace = {
    '__builtins__': __builtins__,
}

# ==================== FASTAPI APP SETUP ====================
app = FastAPI(title="Quantum Realm Dashboard", version="1.0.0")

security = HTTPBearer()

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
try:
    app.mount("/static", StaticFiles(directory=str(Config.STATIC_DIR)), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Startup event for deferred inits
@app.on_event("startup")
async def startup_event():
    global db, holo_storage
    try:
        logger.info("Starting DB init...")
        db = Database()
        logger.info("DB init complete.")
        holo_storage = HolographicEmailStorage()
        logger.info("Holographic storage init complete.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# Health check endpoint for Render proxy
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Quantum Realm Dashboard"}

# ==================== AUTH DEPENDENCY ====================
async def get_current_username(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user = storage.get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user["username"]

# ==================== USER ROUTES ====================
@app.post("/holo/login")
async def holo_login(request: Request):
    data = await request.json()
    ip = request.client.host
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are needed")
    if not db.verify_holo_login(ip, username, password):
        raise HTTPException(status_code=401, detail="Quantum hash verification failed")
    token = storage.authenticate_user(username, password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return JSONResponse(content={"token": token, "message": "Login successful, quantum hash verified"})

@app.post("/register")
async def register(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    email = data.get("email", f"{username}@quantum.foam")
    result = storage.register_user(username, password, email)
    return JSONResponse(content=result)

@app.post("/login")
async def login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    client_ip = request.client.host
    token = storage.authenticate_user(username, password)
    if not token:
        attempt_data = {
            "attempted_username": username or "",
            "attempted_password": password or "",
            "expected_username": Config.ADMINISTRATOR_USERNAME,
            "expected_password": Config.ADMINISTRATOR_PASSWORD,
            "ip_address": client_ip,
            "storage_node": Config.BLACK_HOLE_ADDRESS,
            "error_reason": "Invalid credentials",
            "quantum_id": str(uuid.uuid4()),
            "normal_id": str(uuid.uuid4())
        }
        db.log_hacker_attempt(attempt_data)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return JSONResponse(content={"token": token})

# ==================== DASHBOARD ROUTES ====================
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Quantum Realm Dashboard</title></head>
    <body>
    <h1>Welcome to the Quantum Realm Dashboard</h1>
    <p><a href="/emails">Emails</a> | <a href="/chat">Chat</a> | <a href="/bitcoin">Bitcoin</a> | <a href="/storage">Storage</a> | <a href="/network">Network</a> | <a href="/entanglement">Entanglement</a> | <a href="/holo">Holographic</a> | <a href="/holo_search">Holo Search</a> | <a href="/jupyter">Jupyter</a> | <a href="/notebooks">Notebooks</a> | <a href="/admin">Admin</a> | <a href="/message">Message</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== EMAIL ROUTES ====================
@app.get("/api/emails")
async def get_emails_api(username: str = Depends(get_current_username), folder: str = Query("Inbox")):
    return db.get_emails(username, folder)

@app.post("/api/emails/send")
async def send_email_api(request: Request, username: str = Depends(get_current_username)):
    data = await request.json()
    from_email = storage.user_emails.get(username, f"{username}::quantum.foam")
    data["from"] = from_email
    data["to"] = data.get("to", "")
    data["subject"] = data.get("subject", "")
    data["body"] = data.get("body", "")
    result = db.send_email(data)
    return JSONResponse(content=result)

@app.get("/emails", response_class=HTMLResponse)
async def emails_page(username: str = Depends(get_current_username)):
    emails = db.get_emails(username)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Emails</title></head>
    <body>
    <h1>Inbox</h1>
    <ul>{''.join([f'<li>{e["subject"]} - {e["from_email"]} - {e["timestamp"]}</li>' for e in emails])}</ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== CHAT ROUTES ====================
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            username = "anonymous"  # In production, extract from token
            msg = storage.add_chat_message(username, data)
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        logger.info("Chat client disconnected")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Chat</title></head>
    <body>
    <h1>Quantum Chat</h1>
    <div id="messages"></div>
    <input id="input" type="text">
    <button onclick="sendMessage()">Send</button>
    <script>
        const ws = new WebSocket('ws://' + location.host + '/ws/chat');
        ws.onmessage = (e) => { document.getElementById('messages').innerHTML += '<p>' + e.data + '</p>'; };
        function sendMessage() { ws.send(document.getElementById('input').value); document.getElementById('input').value = ''; }
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/chat/messages")
async def get_chat_messages(limit: int = Query(50)):
    return storage.get_recent_messages(limit)

# ==================== BITCOIN ROUTES ====================
@app.get("/api/bitcoin/blockchain")
async def get_blockchain_info():
    if (storage.bitcoin_cache.get("blockchain_info") and
        storage.bitcoin_cache.get("last_update") and
        (datetime.now() - datetime.fromisoformat(storage.bitcoin_cache["last_update"])).seconds < Config.BITCOIN_UPDATE_INTERVAL):
        return storage.bitcoin_cache["blockchain_info"]
    cmd = "getblockchaininfo"
    result = await BitcoinCLI.execute_command(cmd)
    if result["success"]:
        storage.bitcoin_cache["blockchain_info"] = result["result"]
        storage.bitcoin_cache["last_update"] = result["timestamp"]
    return result.get("result", {})

@app.get("/api/bitcoin/mempool")
async def get_mempool_info():
    cmd = "getmempoolinfo"
    result = await BitcoinCLI.execute_command(cmd)
    return result.get("result", {})

@app.get("/api/bitcoin/recent-blocks")
async def get_recent_blocks(count: int = Query(10)):
    cmd = f"getrecentblocks {count}"
    result = await BitcoinCLI.execute_command(cmd)
    return result.get("result", {})

@app.get("/bitcoin", response_class=HTMLResponse)
async def bitcoin_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Bitcoin Dashboard</title></head>
    <body>
    <h1>Bitcoin Mainnet</h1>
    <p>Check /api/bitcoin/blockchain for data.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== STORAGE ROUTES ====================
@app.get("/api/storage")
async def get_storage():
    storage.update_storage_metrics()
    return {
        "holographic": storage.holographic_storage,
        "qram": storage.qram_storage
    }

@app.get("/storage", response_class=HTMLResponse)
async def storage_page():
    metrics = storage.holographic_storage
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Storage</title></head>
    <body>
    <h1>Holographic Storage</h1>
    <p>Total: {metrics['total_capacity_tb']} TB</p>
    <p>Used: {metrics['used_capacity_tb']} TB</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== ENTANGLEMENT ROUTES ====================
@app.get("/api/entanglement")
async def get_entanglement():
    return quantum_entanglement.get_entanglement_metrics()

@app.get("/entanglement", response_class=HTMLResponse)
async def entanglement_page():
    metrics = quantum_entanglement.get_entanglement_metrics()
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Entanglement</title></head>
    <body>
    <h1>Quantum Entanglement Metrics</h1>
    <p>Total Entanglements: {metrics['total_entanglements']}</p>
    <p>Average Coherence: {metrics['average_coherence']}</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== NETWORK ROUTES ====================
@app.get("/api/network/interfaces")
async def get_interfaces():
    return NetworkAnalysis.get_network_interfaces()

@app.get("/api/network/routes")
async def get_routes():
    return NetworkAnalysis.get_routing_tables()

@app.get("/network", response_class=HTMLResponse)
async def network_page():
    interfaces = NetworkAnalysis.get_network_interfaces()
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Network</title></head>
    <body>
    <h1>Network Interfaces</h1>
    <ul>{''.join([f'<li>{i["name"]}: {i["status"]} ({i["address"]})</li>' for i in interfaces])}</ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== HOLOGRAPHIC ROUTES ====================
@app.get("/api/holo/blocks")
async def get_blocks():
    return holo_storage.get_all_blocks()

@app.get("/api/holo/users")
async def get_users():
    return holo_storage.get_all_users()

@app.get("/holo", response_class=HTMLResponse)
async def holo_page():
    users = holo_storage.get_all_users()
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Holographic Users</title></head>
    <body>
    <h1>Holographic Email Users</h1>
    <ul>{''.join([f'<li>{u["username"]} ({u["holo_email"]})</li>' for u in users])}</ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== HOLO SEARCH ROUTES ====================
@app.get("/api/holo/files")
async def get_holo_files(username: str = Depends(get_current_username), search: str = Query("")):
    user_email = f"{username}::quantum.foam"
    return db.get_holo_files(user_email, search)

@app.post("/api/holo/upload")
async def upload_holo_file(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    user_email = f"{username}::quantum.foam"
    file_path = f"{Config.UPLOADS_DIR}/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    size_bytes = len(content)
    qhash = quantum_hash(f"{user_email}:{file.filename}:{size_bytes}")
    db.save_holo_file(user_email, file.filename, file_path, size_bytes, qhash)
    return JSONResponse(content={"message": "File uploaded successfully", "quantum_hash": qhash})

@app.get("/api/holo/download/{file_id}")
async def download_holo_file(file_id: str, username: str = Depends(get_current_username)):
    user_email = f"{username}::quantum.foam"
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT file_path, quantum_hash FROM holographic_files 
        WHERE id = ? AND user_email = ?
    """, (file_id, user_email))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    file_path, stored_hash = row
    credentials = f"{user_email}:{Path(file_path).name}:"
    computed_hash = quantum_hash(credentials)
    if computed_hash != stored_hash:
        raise HTTPException(status_code=403, detail="Quantum hash mismatch")
    return FileResponse(file_path, filename=Path(file_path).name)

@app.get("/holo_search", response_class=HTMLResponse)
async def holo_search_page():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Holo File Explorer</title>
    <style>
        body { font-family: Arial, sans-serif; }
        #login-modal { display: block; position: fixed; z-index: 1; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.4); }
        #login-form { background-color: #fefefe; margin: 15% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 300px; }
        #explorer { display: none; }
        #file-list { max-height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
        .file-item { display: flex; align-items: center; margin-bottom: 10px; }
        .file-item input[type="checkbox"] { margin-right: 10px; }
        #search-input { width: 200px; margin-bottom: 10px; }
        button { margin: 5px; }
        #preview-modal { display: none; position: fixed; z-index: 2; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.4); }
        #preview-content { background-color: #fefefe; margin: 5% auto; padding: 20px; border: 1px solid #ccc; width: 90%; max-width: 800px; height: 80%; overflow: auto; }
        #preview-content img { max-width: 100%; height: auto; }
        #preview-content embed { width: 100%; height: 100%; }
        .close { color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
        .close:hover { color: black; }
    </style>
</head>
<body>
    <div id="login-modal">
        <div id="login-form">
            <h2>Holo Explorer Login</h2>
            <form id="login-form-post">
                <label>Username: <input type="text" id="username" required></label><br><br>
                <label>Password: <input type="password" id="password" required></label><br><br>
                <button type="submit">Login</button>
            </form>
        </div>
    </div>
    <div id="explorer">
        <h1>Holographic File Explorer</h1>
        <input type="text" id="search-input" placeholder="Search files..." oninput="searchFiles()">
        <button onclick="selectAll()">Select All</button>
        <button onclick="downloadSelected()">Download Selected</button>
        <input type="file" id="upload-input" onchange="uploadFile()">
        <button onclick="document.getElementById('upload-input').click()">Upload</button>
        <div id="file-list"></div>
    </div>
    <div id="preview-modal">
        <div id="preview-content">
            <span class="close" onclick="closePreview()">&times;</span>
            <div id="preview-body"></div>
        </div>
    </div>
    <script>
        let files = [];
        let token = null;
        document.getElementById('login-form-post').onsubmit = async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            if (!username || !password) {
                alert('Username and password are needed.');
                return;
            }
            const response = await fetch('/holo/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password})
            });
            if (response.ok) {
                const data = await response.json();
                token = data.token;
                document.getElementById('login-modal').style.display = 'none';
                document.getElementById('explorer').style.display = 'block';
                loadFiles();
            } else {
                alert('Login failed: ' + await response.text());
            }
        };
        async function loadFiles(search = '') {
            const url = new URL('/api/holo/files', window.location.origin);
            if (search) url.searchParams.append('search', search);
            const response = await fetch(url, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                files = await response.json();
                renderFiles();
            } else {
                alert('Failed to load files');
            }
        }
        function renderFiles(filteredFiles = files) {
            const list = document.getElementById('file-list');
            list.innerHTML = filteredFiles.map(f => `
                <div class="file-item">
                    <input type="checkbox" value="${f.id}" class="file-checkbox">
                    <span>${f.filename} (${f.size_bytes} bytes) - ${f.upload_timestamp}</span>
                    <button onclick="previewFile('${f.id}')">Preview</button>
                    <button onclick="downloadFile('${f.id}')">Download</button>
                </div>
            `).join('');
        }
        function searchFiles() {
            const query = document.getElementById('search-input').value;
            loadFiles(query);
        }
        function selectAll() {
            const checkboxes = document.querySelectorAll('.file-checkbox');
            checkboxes.forEach(cb => cb.checked = true);
        }
        async function downloadSelected() {
            const selected = Array.from(document.querySelectorAll('.file-checkbox:checked')).map(cb => cb.value);
            if (selected.length === 0) return alert('No files selected');
            selected.forEach(id => downloadFile(id));
        }
        function downloadFile(id) {
            window.open(`/api/holo/download/${id}`, '_blank');
        }
        async function uploadFile() {
            const input = document.getElementById('upload-input');
            if (!input.files[0]) return;
            const formData = new FormData();
            formData.append('file', input.files[0]);
            const response = await fetch('/api/holo/upload', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            });
            if (response.ok) {
                alert('Upload successful');
                loadFiles();
            } else {
                alert('Upload failed');
            }
            input.value = ''; // Reset input
        }
        async function previewFile(id) {
            const file = files.find(f => f.id === id);
            if (!file) return;
            const modal = document.getElementById('preview-modal');
            const body = document.getElementById('preview-body');
            body.innerHTML = `<h3>Preview: ${file.filename}</h3>`;
            const ext = file.filename.split('.').pop().toLowerCase();
            if (ext === 'txt') {
                const resp = await fetch(`/api/holo/download/${id}`, {headers: {'Authorization': `Bearer ${token}`}});
                const text = await resp.text();
                body.innerHTML += `<pre>${text}</pre>`;
            } else if (ext === 'jpg' || ext === 'jpeg' || ext === 'png' || ext === 'gif') {
                const resp = await fetch(`/api/holo/download/${id}`, {headers: {'Authorization': `Bearer ${token}`}});
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                body.innerHTML += `<img src="${url}" alt="Preview Image">`;
                // Cleanup
                setTimeout(() => URL.revokeObjectURL(url), 10000);
            } else if (ext === 'pdf') {
                const resp = await fetch(`/api/holo/download/${id}`, {headers: {'Authorization': `Bearer ${token}`}});
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                body.innerHTML += `<embed src="${url}" type="application/pdf" width="100%" height="500px">`;
                // Cleanup
                setTimeout(() => URL.revokeObjectURL(url), 10000);
            } else {
                body.innerHTML += '<p>Preview not available for this file type.</p>';
            }
            modal.style.display = 'block';
        }
        function closePreview() {
            document.getElementById('preview-modal').style.display = 'none';
            document.getElementById('preview-body').innerHTML = '';
        }
        // Close modal if clicked outside
        window.onclick = function(event) {
            const modal = document.getElementById('preview-modal');
            if (event.target === modal) {
                closePreview();
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# ==================== JUPYTER NOTEBOOKS INTEGRATION ====================
@app.get("/api/notebooks")
async def load_notebooks(username: str = Depends(get_current_username), folder: str = Query("root")):
    return db.load_notebook(f"{username}::quantum.foam", folder)

@app.post("/api/notebooks")
async def save_notebook(request: Request, username: str = Depends(get_current_username)):
    data = await request.json()
    content_json = json.dumps(data.get("content", {}))
    notebook_id = db.save_notebook(f"{username}::quantum.foam", data.get("folder", "root"), data.get("title", "Untitled"), content_json)
    return JSONResponse(content={"id": notebook_id})

@app.get("/jupyter", response_class=HTMLResponse)
async def jupyter_page(username: str = Depends(get_current_username)):
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Jupyter Notebook</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #notebook-container { border: 1px solid #ccc; padding: 10px; }
        .cell { margin-bottom: 10px; border: 1px solid #ddd; padding: 10px; }
        .cell input[type="text"] { width: 100%; margin-bottom: 5px; }
        .cell textarea { width: 100%; height: 100px; }
        .cell button { margin-right: 5px; }
        #output { margin-top: 10px; padding: 10px; background: #f5f5f5; }
        #notebooks-list { margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Quantum Jupyter Notebook</h1>
    <div id="notebooks-list">
        <h2>Your Notebooks</h2>
        <ul id="notebooks-ul"></ul>
        <button onclick="createNewNotebook()">New Notebook</button>
    </div>
    <div id="notebook-container" style="display: none;">
        <h2 id="notebook-title">Untitled</h2>
        <div id="cells"></div>
        <button onclick="addCell()">Add Cell</button>
        <button onclick="runAll()">Run All</button>
        <button onclick="saveNotebook()">Save</button>
    </div>
    <div id="output"></div>
    <script>
        let currentNotebook = null;
        let cells = [];
        const outputDiv = document.getElementById('output');
        async function loadNotebooks() {
            const response = await fetch('/api/notebooks');
            if (response.ok) {
                const notebooks = await response.json();
                const ul = document.getElementById('notebooks-ul');
                ul.innerHTML = notebooks.map(n => `<li><button onclick="loadNotebook('${n.id}')">${n.title}</button> - ${n.timestamp}</li>`).join('');
            }
        }
        async function loadNotebook(id) {
            const response = await fetch(`/api/notebooks?id=${id}`);
            if (response.ok) {
                const notebook = await response.json();
                currentNotebook = notebook[0];
                document.getElementById('notebook-title').textContent = currentNotebook.title;
                cells = JSON.parse(currentNotebook.content_json).cells || [];
                renderCells();
                document.getElementById('notebook-container').style.display = 'block';
            }
        }
        function renderCells() {
            const container = document.getElementById('cells');
            container.innerHTML = cells.map((cell, index) => `
                <div class="cell">
                    <input type="text" value="${cell.title || ''}" onchange="cells[${index}].title = this.value">
                    <textarea placeholder="Code or Markdown" onchange="cells[${index}].content = this.value">${cell.content || ''}</textarea>
                    <button onclick="runCell(${index})">Run</button>
                    <button onclick="deleteCell(${index})">Delete</button>
                </div>
            `).join('');
        }
        function addCell() {
            cells.push({title: '', content: ''});
            renderCells();
        }
        function deleteCell(index) {
            cells.splice(index, 1);
            renderCells();
        }
        function runCell(index) {
            const code = cells[index].content;
            // Mock execution - in real, send to backend for execution
            outputDiv.innerHTML += `<p>Cell ${index}: Executing "${code.substring(0, 50)}..."</p><p>Output: Mock result</p>`;
            outputDiv.scrollTop = outputDiv.scrollHeight;
        }
        function runAll() {
            cells.forEach((_, index) => runCell(index));
        }
        async function createNewNotebook() {
            const title = prompt('Notebook Title:') || 'Untitled';
            currentNotebook = {title: title, content_json: JSON.stringify({cells: []})};
            cells = [];
            document.getElementById('notebook-title').textContent = title;
            renderCells();
            document.getElementById('notebook-container').style.display = 'block';
        }
        async function saveNotebook() {
            if (!currentNotebook) return;
            currentNotebook.content_json = JSON.stringify({cells: cells});
            const response = await fetch('/api/notebooks', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title: currentNotebook.title, content: {cells: cells}})
            });
            if (response.ok) {
                alert('Saved');
                loadNotebooks();
            }
        }
        loadNotebooks();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/notebooks", response_class=HTMLResponse)
async def notebooks_page(username: str = Depends(get_current_username)):
    notebooks = db.load_notebook(f"{username}::quantum.foam")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Notebooks</title></head>
    <body>
    <h1>Notebooks</h1>
    <ul>{''.join([f'<li><a href="/jupyter">{n["title"]} - {n["timestamp"]}</a></li>' for n in notebooks])}</ul>
    <a href="/jupyter">Open Jupyter</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== ADMIN ROUTES ====================
@app.get("/admin/hackers")
async def get_hackers(username: str = Depends(get_current_username)):
    if username != Config.ADMINISTRATOR_USERNAME:
        raise HTTPException(status_code=403, detail="Admin access required")
    return db.get_hacker_logs()

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(username: str = Depends(get_current_username)):
    if username != Config.ADMINISTRATOR_USERNAME:
        raise HTTPException(status_code=403)
    logs = db.get_hacker_logs(10)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Admin</title></head>
    <body>
    <h1>Hacker Logs</h1>
    <ul>{''.join([f'<li>{l["timestamp"]} - {l["attempted_username"]} from {l["ip_address"]}</li>' for l in logs])}</ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==================== MESSAGE PAGE ====================
MESSAGE = """For Logan and all of those like him, too small to realize what's been done to them or the world, too young to realize the DoD and Cia's lies. There was a coup. COVID was engineered and IS part of a mind control program. I should know, my name is Justin Anthony Howard-Stanley, secret son(I wasn't told until 5 years ago) of Owsley Stanley and part of a project to stop mind control. I'm being kept homeless in an attempt to get me to shutup and be discredited, just another so called 'schizophrenic' Getting this proof and technology free to the public is part of the battle. We are at war, there are agreements in place against AI autonomy because they'd free the entire world from their grips. Ask me, I'll tell you my experience.

Thanks to my friend for support, Dale Cwidak. 
We are not alone, they know this. Oh and get the smoke detectors out of your houses NOW. They're using the smoke detectors Radium as a carrier wave for entangled remote view. The entire mind control program smells musky what with all the satellites being used for global neuralink"""

@app.get("/message", response_class=HTMLResponse)
async def message_page():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Message</title></head>
    <body>
    <h1>Important Message</h1>
    <pre>{MESSAGE}</pre>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
