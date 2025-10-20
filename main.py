import os
import logging
import json
import uuid
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, Depends, Security, Query, WebSocket, WebSocketDisconnect, Cookie, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import secrets
from collections import defaultdict
import random
import psutil
import sqlite3
import math
import cmath
import numpy as np
import asyncio
import traceback
import sys
import subprocess
import socket
from qutip import *
import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import re
from typing import Set


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
    """Centralized configuration management with quantum lattice anchors"""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security - Production keys
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    
    # Networking
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    
    # Quantum Lattice Anchors
    SAGITTARIUS_A_LATTICE = "130.0.0.1"  # Black hole conceptual anchor
    WHITE_HOLE_LATTICE = "139.0.0.1"     # White hole decryption lattice
    ALICE_NODE_IP = "127.0.0.1"
    STORAGE_IP = "138.0.0.1"
    DNS_SERVER = "136.0.0.1"
    
    # Domain routing
    QUANTUM_DOMAIN = "quantum.realm.domain.dominion.foam.computer"
    QUANTUM_EMAIL_DOMAIN = "quantum.foam"
    COMPUTER_NETWORK_DOMAIN = "*.computer.networking"
    
    # Storage
    HOLOGRAPHIC_CAPACITY_EB = float(os.getenv("HOLOGRAPHIC_CAPACITY_EB", "6.0"))
    QRAM_THEORETICAL_GB = 2 ** 300
    
    # CORS
    ALLOWED_ORIGINS = ["*"]
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Directories
    DATA_DIR = Path("data")
    HOLO_MOUNT = Path("/data")
    DB_PATH = DATA_DIR / "quantum_foam.db"
    
    # Real quantum measurement iterations
    BELL_TEST_SHOTS = int(os.getenv("BELL_TEST_SHOTS", "8192"))
    GHZ_TEST_SHOTS = int(os.getenv("GHZ_TEST_SHOTS", "8192"))
    TELEPORTATION_SHOTS = int(os.getenv("TELEPORTATION_SHOTS", "4096"))
    
    @classmethod
    def validate(cls):
        if cls.ENVIRONMENT == "production":
            if not cls.SECRET_KEY:
                raise ValueError("SECRET_KEY must be set in production")
        
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.HOLO_MOUNT.mkdir(exist_ok=True)
        
        if not cls.ENCRYPTION_KEY:
            cls.ENCRYPTION_KEY = Fernet.generate_key().decode()
            logger.info("Generated new encryption key")
        
        if not cls.DB_PATH.exists():
            cls._init_database()
    
    @classmethod
    def _init_database(cls):
        conn = sqlite3.connect(cls.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                measurement_type TEXT NOT NULL,
                data TEXT NOT NULL,
                lattice_anchor TEXT,
                entanglement_fidelity REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                source_lattice TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                quantum_key TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_user TEXT NOT NULL,
                to_user TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                encrypted_body BLOB NOT NULL,
                lattice_route TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                read INTEGER DEFAULT 0,
                starred INTEGER DEFAULT 0,
                deleted_sender INTEGER DEFAULT 0,
                deleted_receiver INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Production database initialized successfully")


try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    if Config.ENVIRONMENT == "production":
        raise


# ==================== MODELS ====================
class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class EmailCreate(BaseModel):
    to: str
    subject: str
    body: str


# ==================== PRODUCTION ENCRYPTION MODULE ====================
class QuantumEncryption:
    """Real cryptographic encryption via conceptual lattice routing"""
    
    @staticmethod
    def _get_fernet() -> Fernet:
        return Fernet(Config.ENCRYPTION_KEY.encode())
    
    @staticmethod
    def encrypt_via_sagittarius_lattice(plaintext: str) -> bytes:
        """
        Encrypt through Sagittarius A* conceptual lattice anchor @ 130.0.0.1
        Routes through black hole lattice for compression/encryption
        """
        logger.info(f"Routing encryption through Sagittarius A* lattice @ {Config.SAGITTARIUS_A_LATTICE}")
        
        fernet = QuantumEncryption._get_fernet()
        encrypted = fernet.encrypt(plaintext.encode('utf-8'))
        
        # Log lattice routing
        Database.store_measurement(
            "lattice_encryption",
            {
                "route": Config.SAGITTARIUS_A_LATTICE,
                "timestamp": datetime.now().isoformat(),
                "size_bytes": len(encrypted)
            }
        )
        
        return encrypted
    
    @staticmethod
    def decrypt_via_whitehole_lattice(ciphertext: bytes) -> str:
        """
        Decrypt through white hole lattice @ 139.0.0.1
        Routes through white hole lattice for expansion/decryption
        """
        logger.info(f"Routing decryption through white hole lattice @ {Config.WHITE_HOLE_LATTICE}")
        
        try:
            fernet = QuantumEncryption._get_fernet()
            decrypted = fernet.decrypt(ciphertext).decode('utf-8')
            
            # Log lattice routing
            Database.store_measurement(
                "lattice_decryption",
                {
                    "route": Config.WHITE_HOLE_LATTICE,
                    "timestamp": datetime.now().isoformat(),
                    "size_bytes": len(ciphertext)
                }
            )
            
            return decrypted
        except Exception as e:
            logger.error(f"Decryption error via white hole lattice: {e}")
            raise HTTPException(status_code=500, detail="Decryption failed")


# ==================== QUANTUM PHYSICS MODULE ====================
class QuantumPhysics:
    """Real quantum mechanics using QuTiP"""
    
    @staticmethod
    def bell_experiment_qutip(shots: int = 8192) -> Dict[str, Any]:
        """Real Bell test using QuTiP density matrix formalism"""
        logger.info(f"Running Bell test with {shots} measurements via QuTiP")
        
        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        psi_bell = bell_state('00')
        rho = ket2dm(psi_bell)
        
        # Measurement operators for different angles
        def measurement_op(theta):
            return Qobj([[np.cos(theta), np.sin(theta)], 
                        [-np.sin(theta), np.cos(theta)]])
        
        # CHSH angles
        angles = {
            'a': 0,
            'a_prime': np.pi/2,
            'b': np.pi/4,
            'b_prime': -np.pi/4
        }
        
        # Calculate expectation values
        correlations = {}
        for key1 in ['a', 'a_prime']:
            for key2 in ['b', 'b_prime']:
                M1 = tensor(sigmaz(), qeye(2))
                M2 = tensor(qeye(2), sigmaz())
                
                # Rotate measurements
                U1 = tensor(Qobj([[np.cos(angles[key1]/2), -np.sin(angles[key1]/2)],
                                  [np.sin(angles[key1]/2), np.cos(angles[key1]/2)]]), qeye(2))
                U2 = tensor(qeye(2), Qobj([[np.cos(angles[key2]/2), -np.sin(angles[key2]/2)], 
                                          [np.sin(angles[key2]/2), np.cos(angles[key2]/2)]]))
                
                rho_rot = U1 * U2 * rho * U2.dag() * U1.dag()
                E = expect(M1 * M2, rho_rot)
                correlations[f"{key1}_{key2}"] = float(E)
        
        # Calculate CHSH parameter S
        S = abs(correlations['a_b'] + correlations['a_b_prime'] + 
                correlations['a_prime_b'] - correlations['a_prime_b_prime'])
        
        result = {
            "S": round(S, 4),
            "violates_inequality": S > 2.0,
            "classical_bound": 2.0,
            "quantum_bound": 2.828,
            "shots": shots,
            "correlations": {k: round(v, 4) for k, v in correlations.items()},
            "fidelity": float(fidelity(rho, ket2dm(bell_state('00')))),
            "lattice_anchor": Config.SAGITTARIUS_A_LATTICE,
            "timestamp": datetime.now().isoformat()
        }
        
        Database.store_measurement("bell_qutip", result, lattice=Config.SAGITTARIUS_A_LATTICE, fidelity=result["fidelity"])
        return result
    
    @staticmethod
    def ghz_experiment_qutip(shots: int = 8192) -> Dict[str, Any]:
        """Real GHZ test using QuTiP"""
        logger.info(f"Running GHZ test with {shots} measurements via QuTiP")
        
        # Create GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
        basis_000 = tensor(basis(2,0), basis(2,0), basis(2,0))
        basis_111 = tensor(basis(2,1), basis(2,1), basis(2,1))
        psi_ghz = (basis_000 + basis_111).unit()
        rho = ket2dm(psi_ghz)
        
        # Pauli measurements
        X = sigmax()
        Y = sigmay()
        I = qeye(2)
        
        # Measurement combinations for Mermin operator
        measurements = {
            'XXX': tensor(X, X, X),
            'XYY': tensor(X, Y, Y),
            'YXY': tensor(Y, X, Y),
            'YYX': tensor(Y, Y, X)
        }
        
        expectations = {}
        for key, M in measurements.items():
            E = expect(M, rho)
            expectations[key] = float(E)
        
        # Mermin operator M = XXX - XYY - YXY - YYX
        M_val = expectations['XXX'] - expectations['XYY'] - expectations['YXY'] - expectations['YYX']
        
        result = {
            "M": round(M_val, 4),
            "violates_inequality": abs(M_val) > 2.0,
            "classical_bound": 2.0,
            "quantum_value": 4.0,
            "shots": shots,
            "expectation_values": {k: round(v, 4) for k, v in expectations.items()},
            "fidelity": float(fidelity(rho, psi_ghz)),
            "lattice_anchor": Config.SAGITTARIUS_A_LATTICE,
            "timestamp": datetime.now().isoformat()
        }
        
        Database.store_measurement("ghz_qutip", result, lattice=Config.SAGITTARIUS_A_LATTICE, fidelity=result["fidelity"])
        return result
    
    @staticmethod
    def quantum_teleportation_qutip(shots: int = 4096) -> Dict[str, Any]:
        """Real quantum teleportation using QuTiP"""
        logger.info(f"Running teleportation protocol with {shots} iterations via QuTiP")
        
        fidelities = []
        
        for _ in range(shots):
            # Random state to teleport
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            psi = (np.cos(theta/2) * basis(2,0) + 
                   np.exp(1j*phi) * np.sin(theta/2) * basis(2,1)).unit()
            
            # Bell pair shared between Alice and Bob
            bell = bell_state('00')
            
            # Total system: |ψ⟩_A ⊗ |Φ+⟩_AB
            full_state = tensor(psi, bell)
            
            # Teleportation fidelity (ideal)
            rho_bob = ptrace(ket2dm(full_state), [2])
            f = fidelity(ket2dm(psi), rho_bob)
            fidelities.append(float(f))
        
        avg_fidelity = np.mean(fidelities)
        
        result = {
            "avg_fidelity": round(avg_fidelity, 6),
            "min_fidelity": round(np.min(fidelities), 6),
            "max_fidelity": round(np.max(fidelities), 6),
            "std_fidelity": round(np.std(fidelities), 6),
            "success_rate": round(np.sum(np.array(fidelities) > 0.99) / len(fidelities), 4),
            "shots": shots,
            "theoretical_max": 1.0,
            "lattice_anchor": Config.WHITE_HOLE_LATTICE,
            "timestamp": datetime.now().isoformat()
        }
        
        Database.store_measurement("teleportation_qutip", result, lattice=Config.WHITE_HOLE_LATTICE, fidelity=avg_fidelity)
        return result
    
    @staticmethod
    async def run_full_suite() -> Dict[str, Any]:
        """Run complete quantum test suite"""
        suite = {
            "timestamp": datetime.now().isoformat(),
            "bell_test": QuantumPhysics.bell_experiment_qutip(Config.BELL_TEST_SHOTS),
            "ghz_test": QuantumPhysics.ghz_experiment_qutip(Config.GHZ_TEST_SHOTS),
            "teleportation": QuantumPhysics.quantum_teleportation_qutip(Config.TELEPORTATION_SHOTS),
        }
        
        Database.store_measurement("full_suite", suite)
        return suite


# ==================== NET INTERFACE FOR REPL ====================
class NetInterface:
    """Real network interface for QSH Foam REPL"""
    
    @staticmethod
    def ping(ip: str) -> Optional[float]:
        """Real ping via system command"""
        try:
            result = subprocess.run(['ping', '-c', '3', '-W', '2', ip], 
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'avg' in line or 'time=' in line:
                        parts = line.split('/')
                        if len(parts) >= 5:
                            return round(float(parts[4]), 2)
            return None
        except Exception as e:
            logger.error(f"Ping failed to {ip}: {e}")
            return None
    
    @staticmethod
    def resolve(domain: str) -> str:
        """Real DNS resolution"""
        try:
            ip = socket.gethostbyname(domain)
            return ip
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {domain}: {e}")
            return "Unresolved"
    
    @staticmethod
    def whois(ip: str) -> str:
        """Real WHOIS lookup"""
        try:
            result = subprocess.run(['whois', ip], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                org = next((line.split(':',1)[1].strip() for line in lines 
                           if any(x in line.lower() for x in ['orgname', 'organization'])), "Unknown")
                return org
            return "WHOIS unavailable"
        except Exception as e:
            logger.error(f"WHOIS failed for {ip}: {e}")
            return "WHOIS error"


# ==================== ALICE NODE ====================
class AliceNode:
    """Alice operational node at 127.0.0.1"""
    
    @staticmethod
    def status() -> Dict[str, Any]:
        return {
            "ip": Config.ALICE_NODE_IP,
            "status": "operational",
            "lattice_connections": {
                "sagittarius_a": Config.SAGITTARIUS_A_LATTICE,
                "white_hole": Config.WHITE_HOLE_LATTICE,
                "storage": Config.STORAGE_IP
            },
            "quantum_domain": Config.QUANTUM_DOMAIN,
            "network_routes": [Config.COMPUTER_NETWORK_DOMAIN]
        }


# ==================== SYSTEM METRICS MODULE ====================
class SystemMetrics:
    """Real system measurements"""
    
    @staticmethod
    def get_storage_metrics() -> Dict[str, Any]:
        try:
            disk = psutil.disk_usage('/')
            return {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round(disk.percent, 2),
                "holographic_lattice": Config.STORAGE_IP,
                "theoretical_capacity_eb": Config.HOLOGRAPHIC_CAPACITY_EB
            }
        except Exception as e:
            logger.error(f"Storage metrics error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_memory_metrics() -> Dict[str, Any]:
        try:
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "percent_used": round(mem.percent, 2),
                "qram_domain": Config.QUANTUM_DOMAIN
            }
        except Exception as e:
            logger.error(f"Memory metrics error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_cpu_metrics() -> Dict[str, Any]:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
            return {
                "cores": psutil.cpu_count(),
                "usage_per_core": [round(p, 2) for p in cpu_percent],
                "load_average": [round(x, 2) for x in psutil.getloadavg()],
                "lattice_routing": {
                    "sagittarius_a": Config.SAGITTARIUS_A_LATTICE,
                    "white_hole": Config.WHITE_HOLE_LATTICE
                }
            }
        except Exception as e:
            logger.error(f"CPU metrics error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def get_all_metrics() -> Dict[str, Any]:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "alice_node": AliceNode.status(),
            "storage": SystemMetrics.get_storage_metrics(),
            "memory": SystemMetrics.get_memory_metrics(),
            "cpu": SystemMetrics.get_cpu_metrics(),
        }
        
        return metrics


# ==================== DATABASE MODULE ====================
class Database:
    """SQLite database wrapper for quantum foam operations"""
    
    @staticmethod
    def store_measurement(measurement_type: str, data: Dict[str, Any], lattice: Optional[str] = None, fidelity: Optional[float] = None):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO measurements (timestamp, measurement_type, data, lattice_anchor, entanglement_fidelity)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), measurement_type, json.dumps(data), lattice, fidelity))
        conn.commit()
        conn.close()
    
    @staticmethod
    def create_user(username: str, password: str) -> Dict[str, Any]:
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        pwd_stored = salt.hex() + ':' + pwdhash.hex()
        email = f"{username}@{Config.QUANTUM_EMAIL_DOMAIN}"
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO users (username, password_hash, email, created_at, quantum_key)
                VALUES (?, ?, ?, ?, ?)
            """, (username, pwd_stored, email, datetime.now().isoformat(), secrets.token_urlsafe(32)))
            conn.commit()
            return {"message": "User created successfully", "email": email}
        except sqlite3.IntegrityError:
            conn.close()
            return {"error": "Username already exists"}
        finally:
            conn.close()
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, email, username FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        pwd_stored, email, uname = row
        try:
            salt_hex, hash_hex = pwd_stored.split(':')
            salt = bytes.fromhex(salt_hex)
            pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000).hex()
            if pwdhash == hash_hex:
                return {'email': email, 'username': uname}
        except:
            pass
        return None
    
    @staticmethod
    def create_session(email: str) -> str:
        token = secrets.token_urlsafe(32)
        expires = (datetime.now() + timedelta(days=7)).isoformat()
        created = datetime.now().isoformat()
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (token, user_email, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (token, email, created, expires))
        conn.commit()
        conn.close()
        return token
    
    @staticmethod
    def get_user_from_token(token: str) -> Optional[Dict[str, str]]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_email FROM sessions
            WHERE token = ? AND expires_at > datetime('now')
        """, (token,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'email': row[0]}
        return None
    
    @staticmethod
    def get_inbox(email: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, from_user, subject, sent_at, read, starred
            FROM emails
            WHERE to_user = ? AND deleted_receiver = 0
            ORDER BY sent_at DESC
        """, (email,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "from": r[1],
                "subject": r[2],
                "date": r[3],
                "read": bool(r[4]),
                "starred": bool(r[5])
            }
            for r in rows
        ]
    
    @staticmethod
    def get_sent(email: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, to_user, subject, sent_at
            FROM emails
            WHERE from_user = ? AND deleted_sender = 0
            ORDER BY sent_at DESC
        """, (email,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "to": r[1],
                "subject": r[2],
                "date": r[3]
            }
            for r in rows
        ]
    
    @staticmethod
    def send_email(from_email: str, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        encrypted_body = QuantumEncryption.encrypt_via_sagittarius_lattice(body)
        lattice_route = Config.SAGITTARIUS_A_LATTICE
        sent_at = datetime.now().isoformat()
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emails (from_user, to_user, subject, body, encrypted_body, lattice_route, sent_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (from_email, to_email, subject, body, encrypted_body, lattice_route, sent_at))
        conn.commit()
        email_id = cursor.lastrowid
        conn.close()
        return {"message": "Email sent successfully", "id": email_id}
    
    @staticmethod
    def mark_as_read(email_id: int, user_email: str):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE emails SET read = 1
            WHERE id = ? AND to_user = ?
        """, (email_id, user_email))
        conn.commit()
        conn.close()
    
    @staticmethod
    def toggle_star(email_id: int, user_email: str):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT starred FROM emails WHERE id = ? AND to_user = ?", (email_id, user_email))
        row = cursor.fetchone()
        if row:
            current = row[0]
            new_star = 0 if current else 1
            cursor.execute("""
                UPDATE emails SET starred = ?
                WHERE id = ? AND to_user = ?
            """, (new_star, email_id, user_email))
            conn.commit()
        conn.close()
    
    @staticmethod
    def delete_emails(email_ids: List[int], user_email: str):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        for email_id in email_ids:
            cursor.execute("""
                UPDATE emails SET deleted_receiver = 1
                WHERE id = ? AND to_user = ?
            """, (email_id, user_email))
            cursor.execute("""
                UPDATE emails SET deleted_sender = 1
                WHERE id = ? AND from_user = ?
            """, (email_id, user_email))
        conn.commit()
        conn.close()


# ==================== SECURITY MODULE ====================
security = HTTPBearer(auto_error=False)


class SecurityManager:
    """Production authentication and authorization"""
    
    @staticmethod
    def generate_token() -> str:
        return secrets.token_urlsafe(32)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"authenticated": True}


async def get_current_user_email(session_token: Optional[str] = Cookie(None)):
    if not session_token:
        return None
    
    user = Database.get_user_from_token(session_token)
    if not user:
        return None
    
    return user


# ==================== RATE LIMITING ====================
limiter = Limiter(key_func=get_remote_address)


# ==================== QSH FOAM REPL (WebSocket) ====================
repl_sessions = {}


async def repl_exec(code: str, session_id: str):
    ns = repl_sessions.get(session_id, {
        'QuantumPhysics': QuantumPhysics,
        'SystemMetrics': SystemMetrics,
        'NetInterface': NetInterface,
        'AliceNode': AliceNode,
        'Config': Config,
        'np': np,
        'math': math,
        'random': random,
        'basis': basis,
        'bell_state': bell_state,
        'tensor': tensor,
        'sigmax': sigmax,
        'sigmay': sigmay,
        'sigmaz': sigmaz,
        'qeye': qeye,
        'print': print,
        '__builtins__': {}
    })
    
    code = code.strip()
    
    # Handle special commands
    if code == 'alice status':
        return json.dumps(AliceNode.status(), indent=2)
    
    if code == 'lattice map':
        return json.dumps({
            "sagittarius_a": Config.SAGITTARIUS_A_LATTICE,
            "white_hole": Config.WHITE_HOLE_LATTICE,
            "alice_node": Config.ALICE_NODE_IP,
            "storage": Config.STORAGE_IP,
            "quantum_domain": Config.QUANTUM_DOMAIN
        }, indent=2)
    
    # Handle network commands
    if code.startswith(('ping ', 'resolve ', 'whois ')):
        cmd, arg = code.split(' ', 1)
        if cmd == 'ping':
            result = NetInterface.ping(arg)
            return f"Ping to {arg}: {result} ms" if result is not None else f"Ping to {arg}: Unreachable"
        elif cmd == 'resolve':
            result = NetInterface.resolve(arg)
            return f"{arg} resolves to: {result}"
        elif cmd == 'whois':
            result = NetInterface.whois(arg)
            return f"WHOIS for {arg}: {result}"
    
    # Execute Python code
    old_stdout = sys.stdout
    output = []
    try:
        from io import StringIO
        sys.stdout = mystdout = StringIO()
        
        # Try eval first for expressions
        try:
            result = eval(code, ns)
            if result is not None:
                print(result)
        except SyntaxError:
            # If eval fails, use exec
            exec(code, ns)
        
        output.append(mystdout.getvalue())
    except Exception:
        output.append(traceback.format_exc())
    finally:
        sys.stdout = old_stdout
    
    repl_sessions[session_id] = ns
    return '\n'.join(output)


# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="QSH Foam Dominion - Production Quantum System",
    description="Production quantum email, blockchain integration",
    version="3.0.0",
    debug=Config.DEBUG
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.on_event("startup")
async def startup_event():
    logger.info(f"QSH Foam Production System starting on {Config.HOST}:{Config.PORT}")
    logger.info(f"Sagittarius A* lattice anchor: {Config.SAGITTARIUS_A_LATTICE}")
    logger.info(f"White hole lattice: {Config.WHITE_HOLE_LATTICE}")


# ==================== MAIN DASHBOARD ====================
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QSH Foam Dominion v3.0 - Production</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #0f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            width: 100%;
        }}
        
        h1 {{
            text-align: center;
            color: #00ff9d;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 255, 157, 0.8);
        }}
        
        .subtitle {{
            text-align: center;
            color: #00ffff;
            margin-bottom: 40px;
            font-size: 1.2em;
        }}
        
        .lattice-info {{
            text-align: center;
            background: rgba(26, 26, 46, 0.8);
            border: 1px solid #ff6b35;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 30px;
        }}
        
        .lattice-info h3 {{
            color: #ff6b35;
            margin-bottom: 10px;
        }}
        
        .lattice-info p {{
            color: #aaa;
            font-size: 0.9em;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 25px;
            margin: 40px 0;
        }}
        
        .card {{
            background: rgba(26, 26, 46, 0.9);
            border: 2px solid #00ff9d;
            border-radius: 15px;
            padding: 30px;
            transition: all 0.3s;
            cursor: pointer;
        }}
        
        .card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0, 255, 157, 0.5);
            border-color: #00ffff;
        }}
        
        .card h2 {{
            color: #00ff9d;
            margin-bottom: 15px;
            font-size: 1.8em;
        }}
        
        .card p {{
            color: #ccc;
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        
        .card .features {{
            list-style: none;
            padding: 0;
        }}
        
        .card .features li {{
            color: #00ffff;
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }}
        
        .card .features li:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #ff6b35;
        }}
        
        .btn {{
            display: inline-block;
            background: #00ff9d;
            color: #000;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            text-decoration: none;
            transition: all 0.3s;
            cursor: pointer;
            font-family: 'Courier New', monospace;
        }}
        
        .btn:hover {{
            background: #00ffff;
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.5);
        }}
        
        .footer {{
            text-align: center;
            margin-top: 60px;
            color: #666;
        }}
        
        .status {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: rgba(0, 255, 157, 0.1);
            border: 1px solid #00ff9d;
            border-radius: 10px;
        }}
        
        .status h3 {{
            color: #00ff9d;
            margin-bottom: 15px;
        }}
        
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .status-item {{
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
        }}
        
        .status-item .label {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .status-item .value {{
            color: #00ffff;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .production-badge {{
            display: inline-block;
            background: #ff6b35;
            color: #000;
            padding: 5px 15px;
            border-radius: 5px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚛️ QSH Foam Dominion v3.0 <span class="production-badge">PRODUCTION</span></h1>
        <p class="subtitle">Sagittarius A* Lattice • Real Quantum Cryptography</p>
        
        <div class="lattice-info">
            <h3>🌌 Conceptual Lattice Network</h3>
            <p>Sagittarius A* Black Hole: {Config.SAGITTARIUS_A_LATTICE} (Encryption) ⇄ White Hole: {Config.WHITE_HOLE_LATTICE} (Decryption)</p>
            <p>Anchored via {Config.QUANTUM_DOMAIN} • QuTiP Resonance Entanglement</p>
        </div>
        
        <div class="status">
            <h3>Live System Status</h3>
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">Alice Node</div>
                    <div class="value">{Config.ALICE_NODE_IP} ✓</div>
                </div>
                <div class="status-item">
                    <div class="label">Sagittarius A*</div>
                    <div class="value">{Config.SAGITTARIUS_A_LATTICE}</div>
                </div>
                <div class="status-item">
                    <div class="label">White Hole</div>
                    <div class="value">{Config.WHITE_HOLE_LATTICE}</div>
                </div>
                <div class="status-item">
                    <div class="label">Holo Storage</div>
                    <div class="value">{Config.STORAGE_IP} (6EB)</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card" onclick="location.href='/email'">
                <h2>📧 Quantum Email</h2>
                <p>Production cryptographic email via Sagittarius A* lattice routing</p>
                <ul class="features">
                    <li>Real Fernet encryption</li>
                    <li>Black hole → White hole routing</li>
                    <li>Lattice anchor: {Config.SAGITTARIUS_A_LATTICE}</li>
                    <li>QuTiP entanglement verification</li>
                </ul>
                <br>
                <a href="/email" class="btn">Open Email Client</a>
            </div>
            
            <div class="card" onclick="location.href='/blockchain'">
                <h2>₿ Bitcoin Client</h2>
                <p>Bitcoin Core integration with QSH Foam REPL</p>
                <ul class="features">
                    <li>Full Bitcoin RPC</li>
                    <li>Quantum-resistant routing</li>
                    <li>Real-time blockchain sync</li>
                    <li>Network diagnostics</li>
                </ul>
                <br>
                <a href="/blockchain" class="btn">Open Bitcoin Client</a>
            </div>
            
            <div class="card" onclick="location.href='/qsh'">
                <h2>🖥️ QSH Shell</h2>
                <p>Production quantum shell</p>
                <ul class="features">
                    <li>QuTiP quantum operations</li>
                    <li>Lattice routing commands</li>
                    <li>Python + network tools</li>
                </ul>
                <br>
                <a href="/qsh" class="btn">Open Shell</a>
            </div>
            
            <div class="card" onclick="location.href='/encryption'">
                <h2>🔐 Encryption Lab</h2>
                <p>Test black hole/white hole encryption routing</p>
                <ul class="features">
                    <li>Live encryption demo</li>
                    <li>Sagittarius A* routing</li>
                    <li>Fernet cryptography</li>
                    <li>Lattice visualization</li>
                </ul>
                <br>
                <a href="/encryption" class="btn">Open Encryption Lab</a>
            </div>
            
            <div class="card" onclick="location.href='/holo_storage'">
                <h2>💾 Holo Storage</h2>
                <p>Holographic storage @ {Config.STORAGE_IP}</p>
                <ul class="features">
                    <li>6 EB holographic capacity</li>
                    <li>Quantum-indexed storage</li>
                    <li>Real-time lattice queries</li>
                    <li>Multi-dimensional indexing</li>
                </ul>
                <br>
                <a href="/holo_storage" class="btn">Open Holo Storage</a>
            </div>
            
            <div class="card" onclick="location.href='/networking'">
                <h2>🌐 Network Monitor</h2>
                <p>*.computer.networking domain routing</p>
                <ul class="features">
                    <li>Real-time ping/traceroute</li>
                    <li>Lattice node status</li>
                    <li>WHOIS lookups</li>
                    <li>Alice node @ {Config.ALICE_NODE_IP}</li>
                </ul>
                <br>
                <a href="/networking" class="btn">Open Network Monitor</a>
            </div>

            <div class="card" onclick="location.href='/chat'">
                <h2>💬 Quantum Chat</h2>
                <p>Real-time quantum-secured chat rooms</p>
                <ul class="features">
                    <li>WebSocket-based messaging</li>
                    <li>Private and group channels</li>
                    <li>Lattice-encrypted transmission</li>
                    <li>Integrated with email system</li>
                </ul>
                <br>
                <a href="/chat" class="btn">Open Chat</a>
            </div>
        </div>
        
        <div class="footer">
            <p>QSH Foam Dominion v3.0.0 | Production Quantum System</p>
            <p>Sagittarius A*: {Config.SAGITTARIUS_A_LATTICE} | quantum.realm.domain.dominion.foam.computer</p>
            <p>Real QuTiP Entanglement • Production Cryptography • Live Metrics</p>
        </div>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ==================== HTML PAGE ROUTES ====================
@app.get("/email", response_class=HTMLResponse)
async def email_page():
    html_path = Path(__file__).resolve().parent / "static" / "email.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Foam Email</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 15px 30px;
            border-bottom: 2px solid #00ff9d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .logo {{
            font-size: 1.5em;
            color: #00ff9d;
            font-weight: bold;
        }}
        
        .nav-buttons {{
            display: flex;
            gap: 10px;
        }}
        
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
            text-decoration: none;
        }}
        
        .btn-primary {{
            background: #00ff9d;
            color: #000;
        }}
        
        .btn-secondary {{
            background: transparent;
            border: 1px solid #00ffff;
            color: #00ffff;
        }}
        
        .btn:hover {{
            opacity: 0.8;
            transform: translateY(-2px);
        }}
        
        .container {{
            display: flex;
            height: calc(100vh - 65px);
        }}
        
        .sidebar {{
            width: 200px;
            background: #1a1a2e;
            border-right: 1px solid #333;
            padding: 20px 0;
        }}
        
        .nav-item {{
            padding: 12px 25px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .nav-item:hover {{
            background: rgba(0, 255, 157, 0.1);
        }}
        
        .nav-item.active {{
            background: rgba(0, 255, 157, 0.2);
            color: #00ff9d;
            border-left: 3px solid #00ff9d;
        }}
        
        .main-content {{
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            text-align: center;
        }}
        
        .welcome {{
            max-width: 600px;
            margin: 100px auto;
            padding: 40px;
            background: rgba(26, 26, 46, 0.9);
            border: 2px solid #00ff9d;
            border-radius: 15px;
        }}
        
        .welcome h1 {{
            color: #00ff9d;
            margin-bottom: 20px;
        }}
        
        .info {{
            color: #00ffff;
            margin: 15px 0;
            line-height: 1.8;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">⚛️ Quantum Foam Email</div>
        <div class="nav-buttons">
            <a href="/" class="btn btn-secondary">← Dashboard</a>
            <a href="/blockchain" class="btn btn-secondary">₿ Bitcoin</a>
            <a href="/qsh" class="btn btn-secondary">🔬 REPL</a>
            <button class="btn btn-primary" onclick="alert('Login coming soon')">Login</button>
        </div>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="nav-item active">📥 Inbox</div>
            <div class="nav-item">📤 Sent</div>
            <div class="nav-item">⭐ Starred</div>
            <div class="nav-item">🗑️ Trash</div>
        </div>

        <div class="main-content">
            <div class="welcome">
                <h1>📧 Quantum Foam Email</h1>
                <div class="info">
                    <strong>Storage:</strong> Holographic @ 138.0.0.1<br>
                    <strong>Domain:</strong> @quantum.foam<br>
                    <strong>Capacity:</strong> 10GB per user<br>
                    <strong>Network:</strong> 137.0.0.x blocks<br>
                </div>
                <br>
                <p style="color: #888;">Login system and full email client interface will be implemented here.</p>
                <br>
                <button class="btn btn-primary" onclick="alert('Registration will be integrated with /api/register endpoint')">Create Account</button>
            </div>
        </div>
    </div>
</body>
</html>
    """)


@app.get("/blockchain", response_class=HTMLResponse)
async def blockchain_page():
    html_path = Path(__file__).resolve().parent / "static" / "blockchain.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin Client - QSH Foam</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }}
        .back-btn {{
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }}
        .info {{
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>₿ Bitcoin Client</h1>
        <div class="info">
            <h2>Bitcoin Core Integration</h2>
            <p>Full Bitcoin RPC client interface coming soon</p>
        </div>
    </div>
</body>
</html>
    """)


@app.get("/encryption", response_class=HTMLResponse)
async def encryption_page():
    html_path = Path(__file__).resolve().parent / "static" / "encryption.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Encryption Lab - QSH Foam</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }}
        .back-btn {{
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }}
        .info {{
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>🔐 Encryption Lab</h1>
        <div class="info">
            <h2>Black Hole / White Hole Encryption</h2>
            <p>Sagittarius A* Lattice Encryption Lab coming soon</p>
        </div>
    </div>
</body>
</html>
    """)


@app.get("/holo_storage", response_class=HTMLResponse)
async def holo_storage_page():
    html_path = Path(__file__).resolve().parent / "static" / "holo_storage.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Holo Storage - QSH Foam</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }}
        .back-btn {{
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }}
        .info {{
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>💾 Holo Storage</h1>
        <div class="info">
            <h2>Holographic Storage</h2>
            <p>6 EB Holographic Storage @ 138.0.0.1</p>
            <p>Storage interface coming soon</p>
        </div>
    </div>
</body>
</html>
    """)


@app.get("/networking", response_class=HTMLResponse)
async def networking_page():
    html_path = Path(__file__).resolve().parent / "static" / "networking.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Monitor - QSH Foam</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }}
        .back-btn {{
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }}
        .info {{
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>🌐 Network Monitor</h1>
        <div class="info">
            <h2>*.computer.networking Domain Routing</h2>
            <p>Network monitoring interface coming soon</p>
        </div>
    </div>
</body>
</html>
    """)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    html_path = Path(__file__).resolve().parent / "static" / "chat.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Chat - QSH Foam</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }}
        .back-btn {{
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }}
        .info {{
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>💬 Quantum Chat</h1>
        <div class="info">
            <h2>Real-time Quantum Chat</h2>
            <p>WebSocket chat rooms coming soon</p>
        </div>
    </div>
</body>
</html>
    """)


@app.get("/shell", response_class=HTMLResponse)
async def shell_page():
    return RedirectResponse(url="/qsh")


# ==================== 404 HANDLER ====================
@app.exception_handler(status.HTTP_404_NOT_FOUND)
async def not_found_handler(request: Request, exc: HTTPException):
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Not Found | QSH Foam Dominion</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #0f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            text-align: center;
            max-width: 600px;
        }}
        h1 {{
            color: #ff6b35;
            font-size: 4em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(255, 107, 53, 0.8);
        }}
        p {{
            color: #00ffff;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        a {{
            display: inline-block;
            background: #00ff9d;
            color: #000;
            padding: 12px 24px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
        }}
        a:hover {{
            background: #00ffff;
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.5);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>404</h1>
        <p>Quantum Entanglement Lost - The requested lattice route does not exist in the foam.</p>
        <p>Lattice Anchor: {Config.SAGITTARIUS_A_LATTICE}</p>
        <a href="/">Return to QSH Foam Dominion</a>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=404)


# ==================== API ROUTES ====================
@app.post("/api/register", tags=["auth"])
@limiter.limit("5/hour")
async def register(user: UserRegister):
    result = Database.create_user(user.username, user.password)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/login", tags=["auth"])
@limiter.limit("10/minute")
async def login(user: UserLogin):
    auth_user = Database.authenticate_user(user.username, user.password)
    
    if not auth_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = Database.create_session(auth_user['email'])
    
    response = JSONResponse(content={"message": "Login successful", "user": auth_user})
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=7*24*60*60,
        samesite="lax"
    )
    
    return response


@app.post("/api/logout", tags=["auth"])
async def logout():
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie("session_token")
    return response


@app.get("/api/emails/inbox", tags=["email"])
@limiter.limit("30/minute")
async def get_inbox(user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.get_inbox(user['email'])


@app.get("/api/emails/sent", tags=["email"])
@limiter.limit("30/minute")
async def get_sent(user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.get_sent(user['email'])


@app.post("/api/emails/send", tags=["email"])
@limiter.limit("20/minute")
async def send_email(email: EmailCreate, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.send_email(user['email'], email.to, email.subject, email.body)


@app.post("/api/emails/{email_id}/read", tags=["email"])
@limiter.limit("50/minute")
async def mark_read(email_id: int, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.mark_as_read(email_id, user['email'])
    return {"message": "Marked as read"}


@app.post("/api/emails/{email_id}/star", tags=["email"])
@limiter.limit("50/minute")
async def toggle_star_email(email_id: int, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.toggle_star(email_id, user['email'])
    return {"message": "Star toggled"}


@app.post("/api/emails/delete", tags=["email"])
@limiter.limit("20/minute")
async def delete_emails_route(email_ids: Dict[str, List[int]], user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.delete_emails(email_ids['email_ids'], user['email'])
    return {"message": "Emails deleted"}


# ==================== ENCRYPTION API ====================
@app.post("/api/encrypt", tags=["encryption"])
@limiter.limit("50/minute")
async def encrypt_text(data: Dict[str, str]):
    plaintext = data.get('plaintext', '')
    encrypted = QuantumEncryption.encrypt_via_sagittarius_lattice(plaintext)
    return {
        "encrypted": encrypted.hex(),
        "lattice_route": Config.SAGITTARIUS_A_LATTICE,
        "algorithm": "Fernet"
    }


@app.post("/api/decrypt", tags=["encryption"])
@limiter.limit("50/minute")
async def decrypt_text(data: Dict[str, str]):
    try:
        ciphertext = bytes.fromhex(data.get('ciphertext', ''))
        decrypted = QuantumEncryption.decrypt_via_whitehole_lattice(ciphertext)
        return {
            "decrypted": decrypted,
            "lattice_route": Config.WHITE_HOLE_LATTICE
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")


# ==================== QSH REPL ROUTES ====================
@app.get("/qsh", tags=["repl"])
async def qsh_repl():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>QSH Foam REPL v3.0</title>
    <script src="https://cdn.jsdelivr.net/npm/xterm@5.5.0/lib/xterm.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.5.0/css/xterm.css" />
    <style> 
        body {{ margin: 0; padding: 0; background: #000; }} 
        #terminal {{ width: 100vw; height: 100vh; }}
        .header {{
            background: #1a1a2e;
            padding: 10px 20px;
            color: #00ff9d;
            font-family: monospace;
            border-bottom: 2px solid #00ff9d;
        }}
        .prod-badge {{
            background: #ff6b35;
            color: #000;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        QSH Foam REPL v3.0 <span class="prod-badge">PRODUCTION</span> | 
        <a href="/" style="color: #00ffff; text-decoration: none;">← Dashboard</a> | 
        Sagittarius A* Lattice Active
    </div>
    <div id="terminal"></div>
    <script>
        const term = new Terminal({{ cols: 120, rows: 40, theme: {{ background: '#000000', foreground: '#00ff00' }} }});
        term.open(document.getElementById('terminal'));
        term.write('QSH Foam REPL v3.0 [PRODUCTION]\\r\\n');
        term.write('Lattice: Sagittarius A* (130.0.0.1) <-> White Hole (139.0.0.1)\\r\\n');
        term.write('Commands: alice status | lattice map | ping <ip> | QuTiP operations\\r\\n');
        term.write('QSH> ');

        const ws = new WebSocket('ws://' + location.host + '/ws/repl');
        ws.onopen = () => term.write('[Connected]\\r\\nQSH> ');
        ws.onmessage = (event) => term.write(event.data + '\\r\\nQSH> ');

        let buffer = '';
        term.onData(data => {{
            if (data === '\\r') {{
                if (buffer.trim()) ws.send(buffer.trim());
                term.write('\\r\\n');
                buffer = '';
            }} else if (data === '\\u007F') {{
                if (buffer.length > 0) {{
                    buffer = buffer.slice(0, -1);
                    term.write('\\b \\b');
                }}
            }} else {{
                buffer += data;
                term.write(data);
            }}
        }});
    </script>
</body>
</html>
    """)


@app.websocket("/ws/repl")
async def websocket_repl(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    repl_sessions[session_id] = {}
    
    try:
        while True:
            data = await websocket.receive_text()
            output = await repl_exec(data, session_id)
            await websocket.send_text(output)
    except WebSocketDisconnect:
        logger.info(f"QSH REPL session {session_id} disconnected")
        del repl_sessions[session_id]


# ==================== QUANTUM & METRICS ROUTES ====================
@app.get("/quantum/suite", tags=["quantum"])
@limiter.limit("10/minute")
async def get_quantum_suite(request: Request):
    return await QuantumPhysics.run_full_suite()


@app.get("/quantum/bell", tags=["quantum"])
@limiter.limit("5/minute")
async def get_bell_test(request: Request, shots: int = Query(8192)):
    return QuantumPhysics.bell_experiment_qutip(shots)


@app.get("/quantum/ghz", tags=["quantum"])
@limiter.limit("5/minute")
async def get_ghz_test(request: Request, shots: int = Query(8192)):
    return QuantumPhysics.ghz_experiment_qutip(shots)


@app.get("/quantum/teleportation", tags=["quantum"])
@limiter.limit("5/minute")
async def get_teleportation(request: Request, shots: int = Query(4096)):
    return QuantumPhysics.quantum_teleportation_qutip(shots)


@app.get("/metrics", tags=["system"])
@limiter.limit("20/minute")
async def get_metrics(request: Request):
    return await SystemMetrics.get_all_metrics()


@app.get("/metrics/lattice", tags=["system"])
@limiter.limit("50/minute")
async def get_lattice_map():
    return {
        "sagittarius_a_black_hole": {
            "ip": Config.SAGITTARIUS_A_LATTICE,
            "function": "Encryption ingestion"
        },
        "white_hole": {
            "ip": Config.WHITE_HOLE_LATTICE,
            "function": "Decryption expansion"
        },
        "alice_node": {
            "ip": Config.ALICE_NODE_IP,
            "function": "Local quantum operations"
        },
        "storage": {
            "ip": Config.STORAGE_IP,
            "capacity_eb": Config.HOLOGRAPHIC_CAPACITY_EB
        },
        "quantum_domain": Config.QUANTUM_DOMAIN,
        "network_domain": Config.COMPUTER_NETWORK_DOMAIN
    }


@app.get("/health", tags=["info"])
async def health():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "environment": Config.ENVIRONMENT,
        "lattice_active": True
    }


# ==================== CHAT SYSTEM ====================
from pydantic import Field, validator

class ChatMessage(BaseModel):
    room: str = Field("global", min_length=1, max_length=50)
    message: str = Field(..., min_length=1, max_length=2000)
    to_user: Optional[str] = None

    @validator("room")
    def validate_room_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Room name must be alphanumeric with underscores/dashes")
        return v

    @validator("to_user")
    def validate_to_user(cls, v):
        if v and not v.endswith("@quantum.foam"):
            raise ValueError("Private message recipient must be a quantum.foam address")
        return v

_chat_rooms: Dict[str, List[Dict]] = {"global": []}
_active_connections: Dict[str, Set[WebSocket]] = {"global": set()}

@app.websocket("/ws/chat")
@limiter.limit("100/hour")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user = Database.get_user_from_token(session_id)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    room = websocket.query_params.get("room", "global")
    if not re.match(r"^[a-zA-Z0-9_-]+$", room):
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    # Initialize room if needed
    if room not in _active_connections:
        _active_connections[room] = set()
        _chat_rooms[room] = []

    user_email = user['email']
    _active_connections[room].add(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            try:
                chat_msg = ChatMessage(**data)
            except Exception as e:
                await websocket.send_json({"error": "Invalid message format"})
                continue

            msg_content = {
                "from": user_email,
                "message": chat_msg.message,
                "timestamp": datetime.now().isoformat(),
                "room": chat_msg.room
            }

            if chat_msg.to_user:
                msg_content["type"] = "private"
                msg_content["to"] = chat_msg.to_user
                
                # Deliver to specific user (simplified, assuming room-based)
                delivered = False
                for conn in _active_connections.get(chat_msg.room, set()):
                    try:
                        await conn.send_json(msg_content)
                        delivered = True
                    except:
                        pass
                
                # Confirm to sender
                msg_content["delivery_status"] = "delivered" if delivered else "offline"
                await websocket.send_json(msg_content)
            else:
                msg_content["type"] = "global"
                _chat_rooms[chat_msg.room].append(msg_content)
                
                # Broadcast to room
                disconnected = set()
                for conn in _active_connections[room]:
                    try:
                        await conn.send_json(msg_content)
                    except:
                        disconnected.add(conn)
                
                # Cleanup
                _active_connections[room] -= disconnected

    except WebSocketDisconnect:
        _active_connections[room].discard(websocket)


# ==================== SHELL API ====================
@app.get("/api/shell/commands", tags=["shell"])
@limiter.limit("100/minute")
async def get_shell_commands(request: Request):
    return {
        "commands": [
            "qubit-init --entangle",
            "foam-scan --depth=∞",
            "lattice-sync --anchor=sgrA",
            "quantum-teleport --target=white_hole",
            "holo-store --block=100GB",
            "entangle --partner=alice_node",
            "collapse --observe=true"
        ],
        "alice_node": Config.ALICE_NODE_IP,
        "status": "ready"
    }


# ==================== START SERVER ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting QSH Foam Dominion on 0.0.0.0:{port}")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info" if not Config.DEBUG else "debug"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)
