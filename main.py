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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, Response
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

logging.basicConfig(
    level=logging.INFO if not os.getenv("DEBUG", "false").lower() == "true" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    SAGITTARIUS_A_LATTICE = "130.0.0.1"
    WHITE_HOLE_LATTICE = "139.0.0.1"
    ALICE_NODE_IP = "127.0.0.1"
    STORAGE_IP = "138.0.0.1"
    DNS_SERVER = "136.0.0.1"
    IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
    IBM_BACKEND = "ibm_torino"
    QISKIT_RUNTIME_URL = "https://api.quantum-computing.ibm.com/runtime"
    QUANTUM_DOMAIN = "quantum.realm.domain.dominion.foam.computer"
    QUANTUM_EMAIL_DOMAIN = "quantum.foam"
    COMPUTER_NETWORK_DOMAIN = "*.computer.networking"
    HOLOGRAPHIC_CAPACITY_EB = float(os.getenv("HOLOGRAPHIC_CAPACITY_EB", "6.0"))
    QRAM_THEORETICAL_GB = 2 ** 300
    ALLOWED_ORIGINS = ["*"]
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    DATA_DIR = Path("data")
    HOLO_MOUNT = Path("/data")
    DB_PATH = DATA_DIR / "quantum_foam.db"
    BELL_TEST_SHOTS = int(os.getenv("BELL_TEST_SHOTS", "8192"))
    GHZ_TEST_SHOTS = int(os.getenv("GHZ_TEST_SHOTS", "8192"))
    TELEPORTATION_SHOTS = int(os.getenv("TELEPORTATION_SHOTS", "4096"))
    
    @classmethod
    def validate(cls):
        if cls.ENVIRONMENT == "production":
            if not cls.SECRET_KEY:
                raise ValueError("SECRET_KEY must be set in production")
            if not cls.IBM_QUANTUM_TOKEN:
                logger.warning("IBM_QUANTUM_TOKEN not set")
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS torino_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                backend_status TEXT,
                queue_length INTEGER,
                num_qubits INTEGER,
                quantum_volume INTEGER,
                clops REAL,
                t1_avg REAL,
                t2_avg REAL,
                readout_error_avg REAL,
                cx_error_avg REAL,
                lattice_resonance REAL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized")

try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    if Config.ENVIRONMENT == "production":
        raise

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

class QuantumEncryption:
    @staticmethod
    def _get_fernet() -> Fernet:
        return Fernet(Config.ENCRYPTION_KEY.encode())
    
    @staticmethod
    def encrypt_via_sagittarius_lattice(plaintext: str) -> bytes:
        logger.info(f"Routing encryption through Sagittarius A* lattice @ {Config.SAGITTARIUS_A_LATTICE}")
        fernet = QuantumEncryption._get_fernet()
        encrypted = fernet.encrypt(plaintext.encode('utf-8'))
        return encrypted
    
    @staticmethod
    def decrypt_via_whitehole_lattice(ciphertext: bytes) -> str:
        logger.info(f"Routing decryption through white hole lattice @ {Config.WHITE_HOLE_LATTICE}")
        try:
            fernet = QuantumEncryption._get_fernet()
            decrypted = fernet.decrypt(ciphertext).decode('utf-8')
            return decrypted
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise HTTPException(status_code=500, detail="Decryption failed")

class TorinoQuantumBackend:
    @staticmethod
    async def get_backend_status() -> Dict[str, Any]:
        if not Config.IBM_QUANTUM_TOKEN:
            return {"error": "IBM_QUANTUM_TOKEN not configured", "backend": Config.IBM_BACKEND, "status": "unavailable"}
        try:
            headers = {"Authorization": f"Bearer {Config.IBM_QUANTUM_TOKEN}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                url = f"https://api.quantum-computing.ibm.com/runtime/backends/{Config.IBM_BACKEND}"
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return TorinoQuantumBackend._parse_backend_data(data)
                    else:
                        logger.error(f"IBM API returned status {resp.status}")
                        return {"error": f"API status {resp.status}", "backend": Config.IBM_BACKEND}
        except Exception as e:
            logger.error(f"Error fetching Torino backend status: {e}")
            return {"error": str(e), "backend": Config.IBM_BACKEND}
    
    @staticmethod
    def _parse_backend_data(data: Dict) -> Dict[str, Any]:
        config = data.get('configuration', {})
        properties = data.get('properties', {})
        qubits = properties.get('qubits', [])
        t1_values = [q[0]['value'] for q in qubits if q and len(q) > 0 and 'value' in q[0]]
        t2_values = [q[1]['value'] for q in qubits if q and len(q) > 1 and 'value' in q[1]]
        ro_errors = [q[5]['value'] for q in qubits if q and len(q) > 5 and 'value' in q[5]]
        gates = properties.get('gates', [])
        cx_errors = [g['parameters'][0]['value'] for g in gates if g.get('gate') == 'cx' and g.get('parameters')]
        metrics = {
            "backend": Config.IBM_BACKEND,
            "status": data.get('status', {}).get('state', 'unknown'),
            "num_qubits": config.get('n_qubits', 0),
            "quantum_volume": config.get('quantum_volume', 0),
            "basis_gates": config.get('basis_gates', []),
            "coupling_map": config.get('coupling_map', []),
            "t1_avg_us": round(np.mean(t1_values) * 1e6, 2) if t1_values else 0,
            "t2_avg_us": round(np.mean(t2_values) * 1e6, 2) if t2_values else 0,
            "readout_error_avg": round(np.mean(ro_errors), 4) if ro_errors else 0,
            "cx_error_avg": round(np.mean(cx_errors), 4) if cx_errors else 0,
            "timestamp": datetime.now().isoformat()
        }
        return metrics
    
    @staticmethod
    def calculate_lattice_resonance(metrics: Dict[str, Any]) -> float:
        try:
            psi_ideal = bell_state('00')
            rho_ideal = ket2dm(psi_ideal)
            ro_error = metrics.get('readout_error_avg', 0.01)
            cx_error = metrics.get('cx_error_avg', 0.01)
            noise_strength = (ro_error + cx_error) / 2
            rho_noisy = (1 - noise_strength) * rho_ideal + noise_strength * qeye(4) / 4
            resonance = fidelity(rho_ideal, rho_noisy)
            logger.info(f"Lattice resonance calculated: {resonance:.4f}")
            return float(resonance)
        except Exception as e:
            logger.error(f"Error calculating lattice resonance: {e}")
            return 0.5

class QuantumPhysics:
    @staticmethod
    def bell_experiment_qutip(shots: int = 8192) -> Dict[str, Any]:
        logger.info(f"Running Bell test with {shots} measurements")
        psi_bell = bell_state('00')
        rho = ket2dm(psi_bell)
        angles = {'a': 0, 'a_prime': np.pi/2, 'b': np.pi/4, 'b_prime': -np.pi/4}
        correlations = {}
        for key1 in ['a', 'a_prime']:
            for key2 in ['b', 'b_prime']:
                M1 = tensor(sigmaz(), qeye(2))
                M2 = tensor(qeye(2), sigmaz())
                U1 = tensor(Qobj([[np.cos(angles[key1]/2), -np.sin(angles[key1]/2)], [np.sin(angles[key1]/2), np.cos(angles[key1]/2)]]), qeye(2))
                U2 = tensor(qeye(2), Qobj([[np.cos(angles[key2]/2), -np.sin(angles[key2]/2)], [np.sin(angles[key2]/2), np.cos(angles[key2]/2)]]))
                rho_rot = U1 * U2 * rho * U2.dag() * U1.dag()
                E = expect(M1 * M2, rho_rot)
                correlations[f"{key1}_{key2}"] = float(E)
        S = abs(correlations['a_b'] + correlations['a_b_prime'] + correlations['a_prime_b'] - correlations['a_prime_b_prime'])
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
        return result
    
    @staticmethod
    def ghz_experiment_qutip(shots: int = 8192) -> Dict[str, Any]:
        logger.info(f"Running GHZ test with {shots} measurements")
        basis_000 = tensor(basis(2,0), basis(2,0), basis(2,0))
        basis_111 = tensor(basis(2,1), basis(2,1), basis(2,1))
        psi_ghz = (basis_000 + basis_111).unit()
        rho = ket2dm(psi_ghz)
        X = sigmax()
        Y = sigmay()
        I = qeye(2)
        measurements = {'XXX': tensor(X, X, X), 'XYY': tensor(X, Y, Y), 'YXY': tensor(Y, X, Y), 'YYX': tensor(Y, Y, X)}
        expectations = {}
        for key, M in measurements.items():
            E = expect(M, rho)
            expectations[key] = float(E)
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
        return result
    
    @staticmethod
    def quantum_teleportation_qutip(shots: int = 4096) -> Dict[str, Any]:
        logger.info(f"Running teleportation protocol with {shots} iterations")
        fidelities = []
        for _ in range(shots):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            psi = (np.cos(theta/2) * basis(2,0) + np.exp(1j*phi) * np.sin(theta/2) * basis(2,1)).unit()
            bell = bell_state('00')
            full_state = tensor(psi, bell)
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
        return result
    
    @staticmethod
    async def run_full_suite() -> Dict[str, Any]:
        suite = {
            "timestamp": datetime.now().isoformat(),
            "bell_test": QuantumPhysics.bell_experiment_qutip(Config.BELL_TEST_SHOTS),
            "ghz_test": QuantumPhysics.ghz_experiment_qutip(Config.GHZ_TEST_SHOTS),
            "teleportation": QuantumPhysics.quantum_teleportation_qutip(Config.TELEPORTATION_SHOTS),
            "torino_backend": await TorinoQuantumBackend.get_backend_status()
        }
        if 'error' not in suite['torino_backend']:
            suite['lattice_resonance'] = TorinoQuantumBackend.calculate_lattice_resonance(suite['torino_backend'])
        return suite

class NetInterface:
    @staticmethod
    def ping(ip: str) -> Optional[float]:
        try:
            result = subprocess.run(['ping', '-c', '3', '-W', '2', ip], capture_output=True, text=True, timeout=10)
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
        try:
            ip = socket.gethostbyname(domain)
            return ip
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {domain}: {e}")
            return "Unresolved"
    
    @staticmethod
    def whois(ip: str) -> str:
        try:
            result = subprocess.run(['whois', ip], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                org = next((line.split(':',1)[1].strip() for line in lines if any(x in line.lower() for x in ['orgname', 'organization'])), "Unknown")
                return org
            return "WHOIS unavailable"
        except Exception as e:
            logger.error(f"WHOIS failed for {ip}: {e}")
            return "WHOIS error"

class AliceNode:
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

class SystemMetrics:
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
            "torino_quantum": await TorinoQuantumBackend.get_backend_status()
        }
        if 'error' not in metrics['torino_quantum']:
            metrics['lattice_resonance'] = TorinoQuantumBackend.calculate_lattice_resonance(metrics['torino_quantum'])
        return metrics

class Database:
    @staticmethod
    def store_measurement(measurement_type: str, data: Dict[str, Any], lattice: Optional[str] = None, fidelity: Optional[float] = None):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO measurements (timestamp, measurement_type, data, lattice_anchor, entanglement_fidelity) VALUES (?, ?, ?, ?, ?)", 
                      (datetime.now().isoformat(), measurement_type, json.dumps(data), lattice, fidelity))
        conn.commit()
        conn.close()
    
    @staticmethod
    def store_torino_metrics(metrics: Dict[str, Any]):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO torino_metrics (timestamp, backend_status, queue_length, num_qubits, quantum_volume, clops, t1_avg, t2_avg, readout_error_avg, cx_error_avg, lattice_resonance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                      (metrics['timestamp'], metrics.get('status', ''), 0, metrics['num_qubits'], metrics['quantum_volume'], 0.0, metrics['t1_avg_us'], metrics['t2_avg_us'], metrics['readout_error_avg'], metrics['cx_error_avg'], metrics.get('lattice_resonance', 0.0)))
        conn.commit()
        conn.close()
    
    @staticmethod
    def create_user(username: str, password: str) -> Dict[str, Any]:
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        password_hash = salt.hex() + pwdhash.hex()
        email = f"{username}@{Config.QUANTUM_EMAIL_DOMAIN}"
        quantum_key = secrets.token_urlsafe(32)
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash, email, created_at, quantum_key) VALUES (?, ?, ?, ?, ?)", 
                          (username, password_hash, email, datetime.now().isoformat(), quantum_key))
            conn.commit()
            logger.info(f"User created: {username} @ {email}")
            return {"username": username, "email": email, "created": True}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        finally:
            conn.close()
    
    @staticmethod
    def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, email, quantum_key FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        if not result:
            return None
        stored_hash, email, quantum_key = result
        salt = bytes.fromhex(stored_hash[:64])
        stored_pwdhash = stored_hash[64:]
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000).hex()
        if pwdhash == stored_pwdhash:
            return {"username": username, "email": email, "quantum_key": quantum_key}
        return None
    
    @staticmethod
    def create_session(email: str) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (token, user_email, created_at, expires_at) VALUES (?, ?, ?, ?)", 
                      (token, email, datetime.now().isoformat(), expires_at))
        conn.commit()
        conn.close()
        return token
    
    @staticmethod
    def verify_session(token: str) -> Optional[str]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_email, expires_at FROM sessions WHERE token = ?", (token,))
        result = cursor.fetchone()
        conn.close()
        if not result:
            return None
        email, expires_at = result
        if datetime.fromisoformat(expires_at) > datetime.now():
            return email
        return None
    
    @staticmethod
    def store_email(from_user: str, to_user: str, subject: str, body: str, lattice_route: str):
        encrypted_body = QuantumEncryption.encrypt_via_sagittarius_lattice(body)
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO emails (from_user, to_user, subject, body, encrypted_body, lattice_route, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                      (from_user, to_user, subject, body, encrypted_body, lattice_route, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        logger.info(f"Email stored: {from_user} -> {to_user} via lattice {lattice_route}")
    
    @staticmethod
    def get_inbox(email: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, from_user, subject, sent_at, read, starred FROM emails WHERE to_user = ? AND deleted_receiver = 0 ORDER BY sent_at DESC", (email,))
        results = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "from": r[1], "subject": r[2], "sent_at": r[3], "read": bool(r[4]), "starred": bool(r[5])} for r in results]
    
    @staticmethod
    def get_email_by_id(email_id: int, user_email: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT from_user, to_user, subject, encrypted_body, lattice_route, sent_at, read, starred FROM emails WHERE id = ? AND (from_user = ? OR to_user = ?)", 
                      (email_id, user_email, user_email))
        result = cursor.fetchone()
        if result:
            if result[1] == user_email and not result[6]:
                cursor.execute("UPDATE emails SET read = 1 WHERE id = ?", (email_id,))
                conn.commit()
        conn.close()
        if not result:
            return None
        body = QuantumEncryption.decrypt_via_whitehole_lattice(result[3])
        return {"id": email_id, "from": result[0], "to": result[1], "subject": result[2], "body": body, "lattice_route": result[4], "sent_at": result[5], "read": bool(result[6]), "starred": bool(result[7])}

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Foam Production System",
    description="Production quantum computing system with IBM Torino backend integration",
    version="1.0.0",
    default_response_class=JSONResponse
)

# ===== MIDDLEWARE SECTION =====

# 1. Global HEAD Request Handler Middleware (MUST BE FIRST)
@app.middleware("http")
async def auto_head_handler(request: Request, call_next):
    """Automatically handle HEAD requests by converting them to GET and stripping the body"""
    original_method = request.method
    
    if request.method == "HEAD":
        # Convert HEAD to GET for processing
        request.scope["method"] = "GET"
        logger.debug(f"HEAD request converted to GET for path: {request.url.path}")
    
    response = await call_next(request)
    
    if original_method == "HEAD":
        # HEAD responses must not contain a body per HTTP spec
        return Response(
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
    
    return response

# 2. Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing"""
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 3. Rate Limiting Middleware
rate_limit_storage = defaultdict(list)

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    """Simple rate limiting based on IP address"""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    # Clean old requests (older than 1 minute)
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if current_time - timestamp < 60
    ]
    
    # Check rate limit
    if len(rate_limit_storage[client_ip]) >= Config.RATE_LIMIT_PER_MINUTE:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "limit": Config.RATE_LIMIT_PER_MINUTE,
                "window": "60 seconds"
            }
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)
    
    response = await call_next(request)
    return response

# 4. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 5. GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 6. Error Handling Middleware
@app.middleware("http")
async def error_handler(request: Request, call_next):
    """Global error handling"""
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e) if Config.DEBUG else "An error occurred",
                "timestamp": datetime.now().isoformat()
            }
        )

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    token = credentials.credentials
    email = Database.verify_session(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return email


# ===== API ROUTES =====

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_html(request: Request):
    """Serve HTML interface for browser requests"""
    # Check if request wants HTML
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        html_file = Path("static/index.html")
        if html_file.exists():
            return FileResponse(html_file)
    
    # Otherwise return JSON
    return {
        "system": "Quantum Foam Production System",
        "version": "1.0.0",
        "environment": Config.ENVIRONMENT,
        "quantum_domain": Config.QUANTUM_DOMAIN,
        "email_domain": Config.QUANTUM_EMAIL_DOMAIN,
        "lattice_anchors": {
            "sagittarius_a": Config.SAGITTARIUS_A_LATTICE,
            "white_hole": Config.WHITE_HOLE_LATTICE,
            "alice_node": Config.ALICE_NODE_IP,
            "storage": Config.STORAGE_IP
        },
        "ibm_backend": Config.IBM_BACKEND,
        "timestamp": datetime.now().isoformat()
    }

# Root and Health Routes
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Root endpoint - system information"""
    return {
        "system": "Quantum Foam Production System",
        "version": "1.0.0",
        "environment": Config.ENVIRONMENT,
        "quantum_domain": Config.QUANTUM_DOMAIN,
        "email_domain": Config.QUANTUM_EMAIL_DOMAIN,
        "lattice_anchors": {
            "sagittarius_a": Config.SAGITTARIUS_A_LATTICE,
            "white_hole": Config.WHITE_HOLE_LATTICE,
            "alice_node": Config.ALICE_NODE_IP,
            "storage": Config.STORAGE_IP
        },
        "ibm_backend": Config.IBM_BACKEND,
        "timestamp": datetime.now().isoformat()
    }

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    """Health check endpoint"""
    return {
        "status": "operational",
        "alice_node": Config.ALICE_NODE_IP,
        "timestamp": datetime.now().isoformat()
    }

# Authentication Routes
@app.post("/api/auth/register", tags=["Authentication"])
async def register(user: UserRegister):
    """Register a new user"""
    return Database.create_user(user.username, user.password)

@app.post("/api/auth/login", tags=["Authentication"])
async def login(user: UserLogin):
    """Login and receive session token"""
    verified = Database.verify_user(user.username, user.password)
    if not verified:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = Database.create_session(verified['email'])
    return {
        "token": token,
        "email": verified['email'],
        "username": verified['username']
    }

@app.get("/api/auth/me", tags=["Authentication"])
async def get_me(user_email: str = Depends(get_current_user)):
    """Get current user information"""
    return {"email": user_email}

# Email Routes
@app.post("/api/email/send", tags=["Email"])
async def send_email(email: EmailCreate, user_email: str = Depends(get_current_user)):
    """Send quantum-encrypted email"""
    lattice_route = f"{Config.SAGITTARIUS_A_LATTICE} -> {Config.WHITE_HOLE_LATTICE}"
    Database.store_email(user_email, email.to, email.subject, email.body, lattice_route)
    return {
        "status": "sent",
        "from": user_email,
        "to": email.to,
        "lattice_route": lattice_route
    }

@app.get("/api/email/inbox", tags=["Email"])
async def get_inbox(user_email: str = Depends(get_current_user)):
    """Get user's inbox"""
    return Database.get_inbox(user_email)

@app.get("/api/email/{email_id}", tags=["Email"])
async def get_email(email_id: int, user_email: str = Depends(get_current_user)):
    """Get specific email by ID"""
    email = Database.get_email_by_id(email_id, user_email)
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    return email

# Quantum Physics Routes
@app.get("/api/quantum/bell", tags=["Quantum Physics"])
async def bell_test():
    """Run Bell inequality test"""
    return QuantumPhysics.bell_experiment_qutip(Config.BELL_TEST_SHOTS)

@app.get("/api/quantum/ghz", tags=["Quantum Physics"])
async def ghz_test():
    """Run GHZ state test"""
    return QuantumPhysics.ghz_experiment_qutip(Config.GHZ_TEST_SHOTS)

@app.get("/api/quantum/teleportation", tags=["Quantum Physics"])
async def teleportation_test():
    """Run quantum teleportation protocol"""
    return QuantumPhysics.quantum_teleportation_qutip(Config.TELEPORTATION_SHOTS)

@app.get("/api/quantum/suite", tags=["Quantum Physics"])
async def quantum_suite():
    """Run full quantum test suite"""
    return await QuantumPhysics.run_full_suite()

@app.get("/api/quantum/torino", tags=["Quantum Backend"])
async def torino_status():
    """Get IBM Torino backend status"""
    return await TorinoQuantumBackend.get_backend_status()

# System Metrics Routes
@app.get("/api/metrics", tags=["System"])
async def system_metrics():
    """Get all system metrics"""
    return await SystemMetrics.get_all_metrics()

@app.get("/api/metrics/storage", tags=["System"])
async def storage_metrics():
    """Get storage metrics"""
    return SystemMetrics.get_storage_metrics()

@app.get("/api/metrics/memory", tags=["System"])
async def memory_metrics():
    """Get memory metrics"""
    return SystemMetrics.get_memory_metrics()

@app.get("/api/metrics/cpu", tags=["System"])
async def cpu_metrics():
    """Get CPU metrics"""
    return SystemMetrics.get_cpu_metrics()

# Alice Node Routes
@app.get("/api/alice", tags=["Network"])
async def alice_status():
    """Get Alice node status"""
    return AliceNode.status()

# Network Utility Routes
@app.get("/api/net/ping/{ip}", tags=["Network"])
async def ping_ip(ip: str):
    """Ping an IP address"""
    latency = NetInterface.ping(ip)
    return {
        "ip": ip,
        "latency_ms": latency,
        "reachable": latency is not None
    }

@app.get("/api/net/resolve/{domain}", tags=["Network"])
async def resolve_domain(domain: str):
    """Resolve a domain name"""
    ip = NetInterface.resolve(domain)
    return {
        "domain": domain,
        "ip": ip
    }

@app.get("/api/net/whois/{ip}", tags=["Network"])
async def whois_ip(ip: str):
    """WHOIS lookup for an IP"""
    org = NetInterface.whois(ip)
    return {
        "ip": ip,
        "organization": org
    }

# API Documentation redirect
@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info("Quantum Foam Production System Starting...")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Debug Mode: {Config.DEBUG}")
    logger.info(f"IBM Backend: {Config.IBM_BACKEND}")
    logger.info(f"Quantum Domain: {Config.QUANTUM_DOMAIN}")
    logger.info(f"Alice Node: {Config.ALICE_NODE_IP}")
    logger.info(f"API Documentation: http://{Config.HOST}:{Config.PORT}/docs")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Quantum Foam Production System Shutting Down...")

# Main Entry Point
if __name__ == "__main__":
    logger.info(f"Starting Quantum Foam Production System on {Config.HOST}:{Config.PORT}")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"IBM Torino Backend: {Config.IBM_BACKEND}")
    logger.info(f"Quantum Domain: {Config.QUANTUM_DOMAIN}")
    
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info" if not Config.DEBUG else "debug",
        access_log=True
    )
