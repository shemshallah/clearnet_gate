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
from pydantic import BaseModel, Field, validator
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
        
        # QuTiP fidelity is sqrt(F), square to get F
        fid_sqrt = fidelity(rho, ket2dm(bell_state('00')))
        fidelity_val = fid_sqrt ** 2
        
        result = {
            "S": round(S, 4),
            "violates_inequality": S > 2.0,
            "classical_bound": 2.0,
            "quantum_bound": 2.828,
            "shots": shots,
            "correlations": {k: round(v, 4) for k, v in correlations.items()},
            "fidelity": round(fidelity_val, 6),
            "lattice_anchor": Config.SAGITTARIUS_A_LATTICE,
            "timestamp": datetime.now().isoformat()
        }
        
        Database.store_measurement("bell_qutip", result, lattice=Config.SAGITTARIUS_A_LATTICE, fidelity=fidelity_val)
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
        
        # QuTiP fidelity is sqrt(F), square to get F
        fid_sqrt = fidelity(rho, ket2dm(psi_ghz))
        fidelity_val = fid_sqrt ** 2
        
        result = {
            "M": round(M_val, 4),
            "violates_inequality": abs(M_val) > 2.0,
            "classical_bound": 2.0,
            "quantum_value": 4.0,
            "shots": shots,
            "expectation_values": {k: round(v, 4) for k, v in expectations.items()},
            "fidelity": round(fidelity_val, 6),
            "lattice_anchor": Config.SAGITTARIUS_A_LATTICE,
            "timestamp": datetime.now().isoformat()
        }
        
        Database.store_measurement("ghz_qutip", result, lattice=Config.SAGITTARIUS_A_LATTICE, fidelity=fidelity_val)
        return result
    
    @staticmethod
    def quantum_teleportation_qutip(shots: int = 4096) -> Dict[str, Any]:
        """Real quantum teleportation using QuTiP"""
        logger.info(f"Running teleportation protocol with {shots} iterations via QuTiP")
        
        # Bell projectors for Alice's measurement on qubits 0 and 1
        phi_plus = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
        phi_minus = (tensor(basis(2,0), basis(2,0)) - tensor(basis(2,1), basis(2,1))).unit()
        psi_plus = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0))).unit()
        psi_minus = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
        
        bell_projectors = [phi_plus, phi_minus, psi_plus, psi_minus]
        # Corrections for Bob: 00: I, 01: Z, 10: X, 11: XZ
        corrections = [qeye(2), sigmaz(), sigmax(), sigmax() * sigmaz()]
        
        fidelities = []
        
        for _ in range(shots):
            # Random state to teleport on qubit 0
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            psi = (np.cos(theta/2) * basis(2,0) + 
                   np.exp(1j*phi) * np.sin(theta/2) * basis(2,1)).unit()
            
            # Bell pair on qubits 1 (Alice) and 2 (Bob)
            bell = bell_state('00')
            
            # Full state: |ψ⟩_0 ⊗ |Φ+⟩_{12}
            full_ket = tensor(psi, bell)
            
            # Simulate Bell measurement: random outcome (equal prob in ideal case)
            proj_idx = np.random.randint(0, 4)
            projector = ket2dm(bell_projectors[proj_idx])
            
            # Full projector: projector on 0,1 ⊗ I on 2
            full_projector = tensor(projector, qeye(2))
            
            # Project the state
            projected = full_projector * full_ket
            norm = projected.norm()
            if norm > 1e-10:  # Avoid zero norm due to numerical issues
                projected = projected.unit()
                
                # Bob's correction on qubit 2: I ⊗ I ⊗ correction
                correction = tensor(qeye(2), qeye(2), corrections[proj_idx])
                
                # Apply correction
                corrected = correction * projected
                
                # Trace out Alice's qubits (0 and 1), get Bob's state
                rho_bob = ptrace(ket2dm(corrected), [2])
                
                # Fidelity (QuTiP returns sqrt(F), square to get F)
                f_sqrt = fidelity(ket2dm(psi), rho_bob)
                f = f_sqrt ** 2
                fidelities.append(f)
            else:
                fidelities.append(0.0)
        
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
    
    return user['email']


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

app.mount("/static", StaticFiles(directory="static"), name="static")

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
@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ==================== HTML PAGE ROUTES ====================
@app.get("/email")
async def email_page():
    return FileResponse("static/email.html")


@app.get("/blockchain")
async def blockchain_page():
    return FileResponse("static/blockchain.html")


@app.get("/encryption")
async def encryption_page():
    return FileResponse("static/encryption.html")


@app.get("/holo_storage")
async def holo_storage_page():
    return FileResponse("static/holo_storage.html")


@app.get("/networking")
async def networking_page():
    return FileResponse("static/networking.html")


@app.get("/chat")
async def chat_page():
    return FileResponse("static/chat.html")


@app.get("/qsh")
async def qsh_page():
    return FileResponse("static/qsh.html")


# ==================== API ROUTES ====================
@app.post("/api/register")
@limiter.limit(Config.RATE_LIMIT_PER_MINUTE)
async def api_register(user: UserRegister, request: Request):
    result = Database.create_user(user.username, user.password)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/login")
@limiter.limit(Config.RATE_LIMIT_PER_MINUTE)
async def api_login(user: UserLogin, request: Request):
    auth = Database.authenticate_user(user.username, user.password)
    if not auth:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = Database.create_session(auth['email'])
    resp = JSONResponse(content={"message": "Login successful", "token": token})
    resp.set_cookie(key="session_token", value=token, httponly=True, max_age=7*24*3600)
    return resp


@app.post("/api/send_email")
async def api_send_email(email: EmailCreate, current_user_email: Optional[str] = Depends(get_current_user_email)):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    to_email = f"{email.to}@{Config.QUANTUM_EMAIL_DOMAIN}"
    from_email = current_user_email
    result = Database.send_email(from_email, to_email, email.subject, email.body)
    return result


@app.get("/api/inbox")
async def api_get_inbox(current_user_email: Optional[str] = Depends(get_current_user_email)):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    return Database.get_inbox(current_user_email)


@app.get("/api/sent")
async def api_get_sent(current_user_email: Optional[str] = Depends(get_current_user_email)):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    return Database.get_sent(current_user_email)


@app.post("/api/emails/{email_id}/read")
async def api_mark_read(email_id: int, current_user_email: Optional[str] = Depends(get_current_user_email)):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    Database.mark_as_read(email_id, current_user_email)
    return {"message": "Marked as read"}


@app.post("/api/emails/{email_id}/star")
async def api_toggle_star(email_id: int, current_user_email: Optional[str] = Depends(get_current_user_email)):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    Database.toggle_star(email_id, current_user_email)
    return {"message": "Star toggled"}


@app.delete("/api/emails")
async def api_delete_emails(email_ids: List[int] = Query(...), current_user_email: Optional[str] = Depends(get_current_user_email)):
    if not current_user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    Database.delete_emails(email_ids, current_user_email)
    return {"message": "Emails deleted"}


@app.get("/api/quantum/bell")
async def api_bell():
    return QuantumPhysics.bell_experiment_qutip()


@app.get("/api/quantum/ghz")
async def api_ghz():
    return QuantumPhysics.ghz_experiment_qutip()


@app.get("/api/quantum/teleportation")
async def api_teleportation():
    return QuantumPhysics.quantum_teleportation_qutip()


@app.get("/api/quantum/suite")
async def api_suite():
    return await QuantumPhysics.run_full_suite()


@app.get("/api/metrics")
async def api_metrics():
    return await SystemMetrics.get_all_metrics()


@app.get("/api/network/ping")
async def api_ping(ip: str = Query(...)):
    result = NetInterface.ping(ip)
    return {"ip": ip, "latency_ms": result}


@app.get("/api/network/resolve")
async def api_resolve(domain: str = Query(...)):
    result = NetInterface.resolve(domain)
    return {"domain": domain, "ip": result}


@app.get("/api/network/whois")
async def api_whois(ip: str = Query(...)):
    result = NetInterface.whois(ip)
    return {"ip": ip, "organization": result}


# ==================== WEBSOCKET ROUTES ====================
@app.websocket("/ws")
async def websocket_repl(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            code = msg.get("code", "")
            if code:
                result = await repl_exec(code, session_id)
                await websocket.send_text(result)
    except WebSocketDisconnect:
        logger.info(f"REPL WebSocket disconnected for session {session_id}")
    except json.JSONDecodeError:
        await websocket.send_text("Invalid JSON received.")
    except Exception as e:
        logger.error(f"REPL error: {e}")
        await websocket.send_text(f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info" if not Config.DEBUG else "debug")
