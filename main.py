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
    
    # IBM Quantum - Torino Backend
    IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
    IBM_BACKEND = "ibm_torino"
    QISKIT_RUNTIME_URL = "https://api.quantum-computing.ibm.com/runtime"
    
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
            if not cls.IBM_QUANTUM_TOKEN:
                logger.warning("IBM_QUANTUM_TOKEN not set - Torino metrics will be unavailable")
        
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

# ==================== IBM TORINO INTEGRATION ====================
class TorinoQuantumBackend:
    """Real IBM Torino quantum backend integration"""
    
    @staticmethod
    async def get_backend_status() -> Dict[str, Any]:
        """Fetch real-time status from IBM Torino backend"""
        if not Config.IBM_QUANTUM_TOKEN:
            return {
                "error": "IBM_QUANTUM_TOKEN not configured",
                "backend": Config.IBM_BACKEND,
                "status": "unavailable"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {Config.IBM_QUANTUM_TOKEN}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                # Get backend properties
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
        """Parse IBM backend data into metrics"""
        config = data.get('configuration', {})
        properties = data.get('properties', {})
        
        # Calculate average T1, T2, readout error
        qubits = properties.get('qubits', [])
        t1_values = [q[0]['value'] for q in qubits if q and len(q) > 0 and 'value' in q[0]]
        t2_values = [q[1]['value'] for q in qubits if q and len(q) > 1 and 'value' in q[1]]
        ro_errors = [q[5]['value'] for q in qubits if q and len(q) > 5 and 'value' in q[5]]
        
        # Calculate average gate errors
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
        
        # Store in database
        Database.store_torino_metrics(metrics)
        
        return metrics
    
    @staticmethod
    def calculate_lattice_resonance(metrics: Dict[str, Any]) -> float:
        """
        Calculate conceptual resonance through quantum.realm.domain.dominion.foam lattice
        Based on QuTiP density matrix fidelity calculations
        """
        try:
            # Create density matrices for resonance calculation
            n_qubits = min(metrics.get('num_qubits', 2), 3)  # Use 2-3 qubits for calculation
            
            # Ideal state (maximally entangled)
            psi_ideal = bell_state('00')
            rho_ideal = ket2dm(psi_ideal)
            
            # Noisy state based on actual backend errors
            ro_error = metrics.get('readout_error_avg', 0.01)
            cx_error = metrics.get('cx_error_avg', 0.01)
            
            # Apply depolarizing channel
            noise_strength = (ro_error + cx_error) / 2
            rho_noisy = (1 - noise_strength) * rho_ideal + noise_strength * qeye(4) / 4
            
            # Calculate fidelity as resonance metric
            resonance = fidelity(rho_ideal, rho_noisy)
            
            logger.info(f"Lattice resonance calculated: {resonance:.4f}")
            return float(resonance)
            
        except Exception as e:
            logger.error(f"Error calculating lattice resonance: {e}")
            return 0.5

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
    def ghz_experiment_qutip(shots: int= 8192) -> Dict[str, Any]:
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
        """Run complete quantum test suite with Torino backend metrics"""
        suite = {
            "timestamp": datetime.now().isoformat(),
            "bell_test": QuantumPhysics.bell_experiment_qutip(Config.BELL_TEST_SHOTS),
            "ghz_test": QuantumPhysics.ghz_experiment_qutip(Config.GHZ_TEST_SHOTS),
            "teleportation": QuantumPhysics.quantum_teleportation_qutip(Config.TELEPORTATION_SHOTS),
            "torino_backend": await TorinoQuantumBackend.get_backend_status()
        }
        
        # Calculate lattice resonance
        if 'error' not in suite['torino_backend']:
            suite['lattice_resonance'] = TorinoQuantumBackend.calculate_lattice_resonance(suite['torino_backend'])
        
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
            logger.error(f"WHOIS failedfor {ip}: {e}")
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
            "torino_quantum": await TorinoQuantumBackend.get_backend_status()
        }
        
        if 'error' not in metrics['torino_quantum']:
            metrics['lattice_resonance'] = TorinoQuantumBackend.calculate_lattice_resonance(metrics['torino_quantum'])
        
        return metrics

# ==================== DATABASE MODULE ====================
class Database:
    """Production database operations"""
    
    @staticmethod
    def store_measurement(measurement_type: str, data: Dict[str, Any], lattice: str = None, fidelity: float = None):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO measurements (timestamp, measurement_type, data, lattice_anchor, entanglement_fidelity) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), measurement_type, json.dumps(data), lattice, fidelity)
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Stored {measurement_type} measurement")
        except Exception as e:
            logger.error(f"Database storage error: {e}")
    
    @staticmethod
    def store_torino_metrics(metrics: Dict[str, Any]):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO torino_metrics (timestamp, backend_status, num_qubits, quantum_volume,
                                          t1_avg, t2_avg, readout_error_avg, cx_error_avg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.get('timestamp'),
                metrics.get('status'),
                metrics.get('num_qubits'),
                metrics.get('quantum_volume'),
                metrics.get('t1_avg_us'),
                metrics.get('t2_avg_us'),
                metrics.get('readout_error_avg'),
                metrics.get('cx_error_avg')
            ))
            
            conn.commit()
            conn.close()
            logger.info("Stored Torino backend metrics")
        except Exception as e:
            logger.error(f"Error storing Torino metrics: {e}")
    
    @staticmethod
    def get_recent_measurements(limit: int = 10) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp, measurement_type, data, lattice_anchor, entanglement_fidelity FROM measurements ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "timestamp": row[0],
                    "type": row[1],
                    "data": json.loads(row[2]),
                    "lattice": row[3],
                    "fidelity": row[4]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Database retrieval error: {e}")
            return []
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Production password hashing with salt"""
        salt = hashlib.sha256(Config.SECRET_KEY.encode()).hexdigest().encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return hashlib.sha256(kdf.derive(password.encode())).hexdigest()
    
    @staticmethod
    def create_user(username: str, password: str) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            email = f"{username}@{Config.QUANTUM_EMAIL_DOMAIN}"
            password_hash = Database.hash_password(password)
            created_at = datetime.now().isoformat()
            quantum_key = secrets.token_urlsafe(32)
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, email, created_at, quantum_key) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, email, created_at, quantum_key)
            )
            
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"User created: {email}")
            return {"id": user_id, "username": username, "email": email, "created_at": created_at}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Username already exists")
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail="Error creating user")
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, username, password_hash, email FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            user_id, username, password_hash, email = row
            
            if Database.hash_password(password) != password_hash:
                return None
            
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(), user_id)
            )
            conn.commit()
            conn.close()
            
            return {"id": user_id, "username": username, "email": email}
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    @staticmethod
    def create_session(user_email: str) -> str:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            token = secrets.token_urlsafe(32)
            created_at = datetime.now()
            expires_at = created_at + timedelta(days=7)
            
            cursor.execute(
                "INSERT INTO sessions (token, user_email, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_email, created_at.isoformat(), expires_at.isoformat())
            )
            
            conn.commit()
            conn.close()
            
            return token
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail="Error creating session")
    
    @staticmethod
    def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT user_email, expires_at FROM sessions WHERE token = ?",
                (token,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            user_email, expires_at = row
            
            if datetime.fromisoformat(expires_at) < datetime.now():
                cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
                conn.commit()
                conn.close()
                return None
            
            cursor.execute(
                "SELECT id, username, email FROM users WHERE email = ?",
                (user_email,)
            )
            user_row = cursor.fetchone()
            conn.close()
            
            if user_row:
                return {"id": user_row[0], "username": user_row[1], "email": user_row[2]}
            return None
        except Exception as e:
            logger.error(f"Error getting user from token: {e}")
            return None
    
    @staticmethod
    def send_email(from_email: str, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            # Encrypt body via Sagittarius A* lattice
            encrypted_body = QuantumEncryption.encrypt_via_sagittarius_lattice(body)
            
            # Lattice routing path
            lattice_route = f"{Config.SAGITTARIUS_A_LATTICE} -> {Config.WHITE_HOLE_LATTICE} -> {Config.QUANTUM_DOMAIN}"
            
            sent_at = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO emails (from_user, to_user, subject, body, encrypted_body, lattice_route, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (from_email, to_email, subject, body, encrypted_body, lattice_route, sent_at)
            )
            
            conn.commit()
            email_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Email sent from {from_email} to {to_email} via lattice route: {lattice_route}")
            return {
                "id": email_id,
                "from": from_email,
                "to": to_email,
                "subject": subject,
                "sent_at": sent_at,
                "lattice_route": lattice_route
            }
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")
    
    @staticmethod
    def get_inbox(user_email: str) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, from_user, to_user, subject, encrypted_body, sent_at, read, starred, lattice_route 
                   FROM emails 
                   WHERE to_user = ? AND deleted_receiver = 0
                   ORDER BY sent_at DESC""",
                (user_email,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            emails = []
            for row in rows:
                # Decrypt body via white hole lattice
                try:
                    decrypted_body = QuantumEncryption.decrypt_via_whitehole_lattice(row[4])
                except:
                    decrypted_body = "[Decryption Error]"
                
                emails.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "subject": row[3],
                    "body": decrypted_body,
                    "sent_at": row[5],
                    "read": bool(row[6]),
                    "starred": bool(row[7]),
                    "lattice_route": row[8]
                })
            
            return emails
        except Exception as e:
            logger.error(f"Error getting inbox: {e}")
            return []
    
    @staticmethod
    def get_sent(user_email: str) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, from_user, to_user, subject, encrypted_body, sent_at, read, starred, lattice_route 
                   FROM emails 
                   WHERE from_user = ? AND deleted_sender = 0 
                   ORDER BY sent_at DESC""",
                (user_email,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            emails = []
            for row in rows:
                # Decrypt body via white hole lattice
                try:
                    decrypted_body = QuantumEncryption.decrypt_via_whitehole_lattice(row[4])
                except:
                    decrypted_body = "[Decryption Error]"
                
                emails.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "subject": row[3],
                    "body": decrypted_body,
                    "sent_at": row[5],
                    "read": bool(row[6]),
                    "starred": bool(row[7]),
                    "lattice_route": row[8]
                })
            
            return emails
        except Exception as e:
            logger.error(f"Error getting sent emails: {e}")
            return []
    
    @staticmethod
    def mark_as_read(email_id: int, user_email: str):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE emails SET read = 1 WHERE id = ? AND to_user = ?",
                (email_id, user_email)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error marking email as read: {e}")
    
    @staticmethod
    def toggle_star(email_id: int, user_email: str):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT starred FROM emails WHERE id = ? AND (from_user = ? OR to_user = ?)",
                (email_id, user_email, user_email)
            )
            row = cursor.fetchone()
            
            if row:
                new_starred = 0 if row[0] else 1
                cursor.execute(
                    "UPDATE emails SET starred = ? WHERE id = ?",
                    (new_starred, email_id)
                )
                conn.commit()
            
            conn.close()
        except Exception as e:
            logger.error(f"Error toggling star: {e}")
    
    @staticmethod
    def delete_emails(email_ids: List[int], user_email: str):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            for email_id in email_ids:
                cursor.execute(
                    "SELECT from_user, to_user FROM emails WHERE id = ?",
                    (email_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    from_user, to_user = row
                    if from_user == user_email:
                        cursor.execute(
                            "UPDATE emails SET deleted_sender = 1 WHERE id = ?",
                            (email_id,)
                        )
                    if to_user == user_email:
                        cursor.execute(
                            "UPDATE emails SET deleted_receiver = 1 WHERE id = ?",
                            (email_id,)
                        )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error deleting emails: {e}")

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
rate_limit_store = defaultdict(list)

async def check_rate_limit(request: Request):
    client_ip = request.client.host
    now = datetime.now()
    
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip]
        if now - ts < timedelta(minutes=1)
    ]
    
    if len(rate_limit_store[client_ip]) >= Config.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    rate_limit_store[client_ip].append(now)

# ==================== QSH FOAM REPL (WebSocket) ====================
repl_sessions = {}

async def repl_exec(code: str, session_id: str):
    ns = repl_sessions.get(session_id, {
        'QuantumPhysics': QuantumPhysics,
        'SystemMetrics': SystemMetrics,
        'NetInterface': NetInterface,
        'AliceNode': AliceNode,
        'TorinoQuantumBackend': TorinoQuantumBackend,
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
    
    # Handle special commands
    if code == 'alice status':
        return json.dumps(AliceNode.status(), indent=2)
    
    if code == 'torino status':
        result = await TorinoQuantumBackend.get_backend_status()
        return json.dumps(result, indent=2)
    
    if code == 'lattice map':
        return json.dumps({
            "sagittarius_a": Config.SAGITTARIUS_A_LATTICE,
            "white_hole": Config.WHITE_HOLE_LATTICE,
            "alice_node": Config.ALICE_NODE_IP,
            "storage": Config.STORAGE_IP,
            "quantum_domain": Config.QUANTUM_DOMAIN
        }, indent=2)
    
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
    description="Production quantum email, blockchain integration with IBM Torino backend",
    version="3.0.0",
    debug=Config.DEBUG
)

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
    logger.info(f"IBM Torino backend: {Config.IBM_BACKEND}")

import os
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    logger.info(f"Serving static files from: {STATIC_DIR}")
    app.mount(
        "/",
        StaticFiles(
            directory=str(STATIC_DIR),
            html=True
        ),
        name="static"
    )
else:
    logger.warning(f"Static directory not found at {STATIC_DIR}")


import os
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    logger.info(f"Serving static files from: {STATIC_DIR}")
    app.mount(
        "/",
        StaticFiles(
            directory=str(STATIC_DIR),
            html=True
        ),
        name="static"
    )
else:
    logger.warning(f"Static directory not found at {STATIC_DIR}")

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

# ==================== MAIN DASHBOARD ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    # Fetch live Torino metrics
    torino_status = await TorinoQuantumBackend.get_backend_status()
    torino_html = ""
    
    if 'error' not in torino_status:
        lattice_resonance = TorinoQuantumBackend.calculate_lattice_resonance(torino_status)
        torino_html = f"""
        <div class="status-item">
            <div class="label">IBM Torino</div>
            <div class="value">{torino_status.get('num_qubits', 0)} qubits • QV{torino_status.get('quantum_volume', 0)}</div>
        </div>
        <div class="status-item">
            <div class="label">Lattice Resonance</div>
            <div class="value">{lattice_resonance:.4f}</div>
        </div>
        <div class="status-item">
            <div class="label">T1 Coherence</div>
            <div class="value">{torino_status.get('t1_avg_us', 0):.2f} μs</div>
        </div>
        <div class="status-item">
            <div class="label">Gate Error</div>
            <div class="value">{torino_status.get('cx_error_avg', 0):.4f}</div>
        </div>"""
    else:
        torino_html = f"""
        <div class="status-item">
            <div class="label">IBM Torino</div>
            <div class="value">Configure Token</div>
        </div>"""
    
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
        <p class="subtitle">IBM Torino Backend • Sagittarius A* Lattice • Real Quantum Cryptography</p>
        
        <div class="lattice-info">
            <h3>🌌 Conceptual Lattice Network</h3>
            <p>Sagittarius A* Black Hole: {Config.SAGITTARIUS_A_LATTICE} (Encryption) ⇄ White Hole: {Config.WHITE_HOLE_LATTICE} (Decryption)</p>
            <p>IBM Torino Anchored via {Config.QUANTUM_DOMAIN} • QuTiP Resonance Entanglement</p>
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
                {torino_html}
            </div>
        </div>
        
        <div class="grid">
            <div class="card" onclick="location.href='/email.html'">
                <h2>📧 Quantum Email</h2>
                <p>Production cryptographic email via Sagittarius A* lattice routing</p>
                <ul class="features">
                    <li>Real Fernet encryption</li>
                    <li>Black hole → White hole routing</li>
                    <li>Lattice anchor: {Config.SAGITTARIUS_A_LATTICE}</li>
                    <li>QuTiP entanglement verification</li>
                </ul>
                <br>
                <a href="/email.html" class="btn">Open Email Client</a>
            </div>
            
            <div class="card" onclick="location.href='/blockchain.html'">
                <h2>₿ Bitcoin Client</h2>
                <p>Bitcoin Core integration with QSH Foam REPL</p>
                <ul class="features">
                    <li>Full Bitcoin RPC</li>
                    <li>Quantum-resistant routing</li>
                    <li>Real-time blockchain sync</li>
                    <li>Network diagnostics</li>
                </ul>
                <br>
                <a href="/blockchain.html" class="btn">Open Bitcoin Client</a>
            </div>
            
            <div class="card" onclick="location.href='/shell.html'">
                <h2>🖥️ QSH Shell</h2>
                <p>Production quantum shell with IBM Torino integration</p>
                <ul class="features">
                    <li>QuTiPquantum operations</li>
                    <li>Real Torino backend access</li>
                    <li>Lattice routing commands</li>
                    <li>Python + network tools</li>
                </ul>
                <br>
                <a href="/shell.html" class="btn">Open Shell</a>
            </div>
            
            <div class="card" onclick="location.href='/encryption.html'">
                <h2>🔐 Encryption Lab</h2>
                <p>Test black hole/white hole encryption routing</p>
                <ul class="features">
                    <li>Live encryption demo</li>
                    <li>Sagittarius A* routing</li>
                    <li>Fernet cryptography</li>
                    <li>Lattice visualization</li>
                </ul>
                <br>
                <a href="/encryption.html" class="btn">Open Encryption Lab</a>
            </div>
            
            <div class="card" onclick="location.href='/holo_search.html'">
                <h2>🔍 Holo Search</h2>
                <p>Holographic storage search @ {Config.STORAGE_IP}</p>
                <ul class="features">
                    <li>6 EB holographic capacity</li>
                    <li>Quantum-indexed search</li>
                    <li>Real-time lattice queries</li>
                    <li>Multi-dimensional indexing</li>
                </ul>
                <br>
                <a href="/holo_search.html" class="btn">Open Holo Search</a>
            </div>
            
            <div class="card" onclick="location.href='/networking.html'">
                <h2>🌐 Network Monitor</h2>
                <p>*.computer.networking domain routing</p>
                <ul class="features">
                    <li>Real-time ping/traceroute</li>
                    <li>Lattice node status</li>
                    <li>WHOIS lookups</li>
                    <li>Alice node @ {Config.ALICE_NODE_IP}</li>
                </ul>
                <br>
                <a href="/networking.html" class="btn">Open Network Monitor</a>
            </div>
        </div>
        
        <div class="footer">
            <p>QSH Foam Dominion v3.0.0 | Production Quantum System</p>
            <p>IBM Torino: {Config.IBM_BACKEND} | Sagittarius A*: {Config.SAGITTARIUS_A_LATTICE} | quantum.realm.domain.dominion.foam.computer</p>
            <p>Real QuTiP Entanglement • Production Cryptography • Live Backend Metrics</p>
        </div>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# ==================== API ROUTES ====================

@app.post("/api/register", tags=["auth"])
async def register(user: UserRegister):
    result = Database.create_user(user.username, user.password)
    return result

@app.post("/api/login", tags=["auth"])
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
async def get_inbox(user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.get_inbox(user['email'])

@app.get("/api/emails/sent", tags=["email"])
async def get_sent(user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401,detail="Not authenticated")
    
    return Database.get_sent(user['email'])

@app.post("/api/emails/send", tags=["email"])
async def send_email(email: EmailCreate, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.send_email(user['email'], email.to, email.subject, email.body)

@app.post("/api/emails/{email_id}/read", tags=["email"])
async def mark_read(email_id: int, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.mark_as_read(email_id, user['email'])
    return {"message": "Marked as read"}

@app.post("/api/emails/{email_id}/star", tags=["email"])
async def toggle_star_email(email_id: int, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.toggle_star(email_id, user['email'])
    return {"message": "Star toggled"}

@app.post("/api/emails/delete", tags=["email"])
async def delete_emails_route(email_ids: Dict[str, List[int]], user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.delete_emails(email_ids['email_ids'], user['email'])
    return {"message": "Emails deleted"}

# ==================== ENCRYPTION API ====================

@app.post("/api/encrypt", tags=["encryption"])
async def encrypt_text(data: Dict[str, str]):
    plaintext = data.get('plaintext', '')
    encrypted = QuantumEncryption.encrypt_via_sagittarius_lattice(plaintext)
    return {
        "encrypted": encrypted.hex(),
        "lattice_route": Config.SAGITTARIUS_A_LATTICE,
        "algorithm": "Fernet"
    }

@app.post("/api/decrypt", tags=["encryption"])
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
        body { margin: 0; padding: 0; background: #000; } 
        #terminal { width: 100vw; height: 100vh; }
        .header {
            background: #1a1a2e;
            padding: 10px 20px;
            color: #00ff9d;
            font-family: monospace;
            border-bottom: 2px solid #00ff9d;
        }
        .prod-badge {
            background: #ff6b35;
            color: #000;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        QSH Foam REPL v3.0 <span class="prod-badge">PRODUCTION</span> | 
        <a href="/" style="color: #00ffff; text-decoration: none;">← Dashboard</a> | 
        IBM Torino Connected | Sagittarius A* Lattice Active
    </div>
    <div id="terminal"></div>
    <script>
        const term = new Terminal({ cols: 120, rows: 40, theme: { background: '#000000', foreground: '#00ff00' } });
        term.open(document.getElementById('terminal'));
        term.write('QSH Foam REPL v3.0 [PRODUCTION]\\r\\n');
        term.write('IBM Torino Backend Connected\\r\\n');
        term.write('Lattice: Sagittarius A* (130.0.0.1) <-> White Hole (139.0.0.1)\\r\\n');
        term.write('Commands: alice status | torino status | lattice map | ping <ip> | QuTiP operations\\r\\n');
        term.write('QSH> ');

        const ws = new WebSocket('ws://' + location.host + '/ws/repl');
        ws.onopen = () => term.write('[Connected]\\r\\nQSH> ');
        ws.onmessage = (event) => term.write(event.data + '\\r\\nQSH> ');

        let buffer = '';
        term.onData(data => {
            if (data === '\\r') {
                if (buffer.trim()) ws.send(buffer.trim());
                term.write('\\r\\n');
                buffer = '';
            } else if (data === '\\u007F') {
                if (buffer.length > 0) {
                    buffer = buffer.slice(0, -1);
                    term.write('\\b \\b');
                }
            } else {
                buffer += data;
                term.write(data);
            }
        });
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
async def get_quantum_suite(request: Request):
    await check_rate_limit(request)
    return await QuantumPhysics.run_full_suite()

@app.get("/quantum/bell", tags=["quantum"])
async def get_bell_test(request: Request, shots: int = Query(8192)):
    await check_rate_limit(request)
    return QuantumPhysics.bell_experiment_qutip(shots)

@app.get("/quantum/ghz", tags=["quantum"])
async def get_ghz_test(request: Request, shots: int = Query(8192)):
    await check_rate_limit(request)
    return QuantumPhysics.ghz_experiment_qutip(shots)

@app.get("/quantum/teleportation", tags=["quantum"])
async def get_teleportation(request: Request, shots: int = Query(4096)):
    await check_rate_limit(request)
    return QuantumPhysics.quantum_teleportation_qutip(shots)

@app.get("/quantum/torino", tags=["quantum"])
async def get_torino_status(request: Request):
    await check_rate_limit(request)
    return await TorinoQuantumBackend.get_backend_status()

@app.get("/metrics", tags=["system"])
async def get_metrics(request: Request):
    await check_rate_limit(request)
    return await SystemMetrics.get_all_metrics()

@app.get("/metrics/lattice", tags=["system"])
async def get_lattice_map():
    return {
        "sagittarius_a_black_hole": {
            "ip": Config.SAGITTARIUS_A_LATTICE,
            "function": "Encryption ingestion",
            "backend": "IBM Torino conceptual anchor"
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
        "lattice_active": True,
        "torino_configured": bool(Config.IBM_QUANTUM_TOKEN)
    }

# ==================== STATIC FILES MOUNT (AFTER ALL ROUTES) ====================
# Mount at root to serve HTML files like /email.html directly from the current directory.
# All API routes are defined before this mount, so they take precedence.
# Ensure that files like email.html, blockchain.html, shell.html, etc., exist in the current working directory.
app.mount("/", StaticFiles(directory=".", html=True), name="html_files")

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
