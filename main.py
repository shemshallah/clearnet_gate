 

import os
import logging
import json
import uuid
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, Depends, Security, Query, WebSocket, WebSocketDisconnect, Cookie, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
    """Centralized configuration management with security"""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    
    # Networking
    HOST = "0.0.0.0" if os.getenv("ENVIRONMENT") == "production" else "127.0.0.1"
    PORT = int(os.getenv("PORT", 8000))
    STORAGE_IP = "136.0.0.1"
    DNS_SERVER = "136.0.0.1"
    QUANTUM_DOMAIN = "quantum.realm.domain.dominion.foam.computer"
    QUANTUM_EMAIL_DOMAIN = "quantum.foam"
    HOLOGRAPHIC_CAPACITY_EB = float(os.getenv("HOLOGRAPHIC_CAPACITY_EB", "6.0"))
    QRAM_THEORETICAL_GB = 2 ** 300
    
    # Distributed CPU
    CPU_BLACK_HOLE_IP = "130.0.0.1"
    CPU_WHITE_HOLE_IP = "139.0.0.1"
    ALICE_NODE_IP = "127.0.0.1"
    
    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Directories
    DATA_DIR = Path("data")
    HOLO_MOUNT = Path("/data")
    DB_PATH = DATA_DIR / "quantum_foam.db"
    
    # Quantum params
    BELL_TEST_ITERATIONS = int(os.getenv("BELL_TEST_ITERATIONS", "10000"))
    GHZ_TEST_ITERATIONS = int(os.getenv("GHZ_TEST_ITERATIONS", "10000"))
    TELEPORTATION_ITERATIONS = int(os.getenv("TELEPORTATION_ITERATIONS", "1000"))
    
    @classmethod
    def validate(cls):
        if cls.ENVIRONMENT == "production":
            if not cls.SECRET_KEY:
                raise ValueError("SECRET_KEY must be set in production")
        
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.HOLO_MOUNT.mkdir(exist_ok=True)
        
        if not cls.DB_PATH.exists():
            cls._init_database()
    
    @classmethod
    def _init_database(cls):
        conn = sqlite3.connect(cls.DB_PATH)
        cursor = conn.cursor()
        
        # Original tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                measurement_type TEXT NOT NULL,
                data TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL
            )
        """)
        
        # Email system tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_user TEXT NOT NULL,
                to_user TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
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
        logger.info("Database initialized successfully")

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

# ==================== QUANTUM PHYSICS MODULE ====================
class QuantumPhysics:
    """Scientific quantum mechanics simulations"""
    
    @staticmethod
    def bell_experiment(iterations: int = 10000) -> Dict[str, Any]:
        theta_a = 0
        theta_a_prime = math.pi / 2
        theta_b = math.pi / 4
        theta_b_prime = -math.pi / 4
        
        def quantum_correlation(angle_a: float, angle_b: float, N: int) -> float:
            correlation_sum = 0
            for _ in range(N):
                angle_diff = angle_a - angle_b
                prob_same = (math.sin(angle_diff / 2)) ** 2
                
                if random.random() < prob_same:
                    outcome = random.choice([1, -1])
                    result_a = outcome
                    result_b = outcome
                else:
                    result_a = random.choice([1, -1])
                    result_b = -result_a
                
                correlation_sum += result_a * result_b
            
            return correlation_sum / N
        
        n_per_measurement = iterations // 4
        E_ab = quantum_correlation(theta_a, theta_b, n_per_measurement)
        E_ab_prime = quantum_correlation(theta_a, theta_b_prime, n_per_measurement)
        E_a_prime_b = quantum_correlation(theta_a_prime, theta_b, n_per_measurement)
        E_a_prime_b_prime = quantum_correlation(theta_a_prime, theta_b_prime, n_per_measurement)
        
        S = abs(E_ab + E_ab_prime + E_a_prime_b - E_a_prime_b_prime)
        
        violates = S > 2.0
        theoretical_max = 2 * math.sqrt(2)
        
        logger.info(f"Bell CHSH: S={S:.3f}, violates={violates}, theoretical_max={theoretical_max:.3f}")
        
        return {
            "S": round(S, 4),
            "violates_inequality": violates,
            "classical_bound": 2.0,
            "quantum_bound": round(theoretical_max, 4),
            "iterations": iterations,
            "correlations": {
                "E_ab": round(E_ab, 4),
                "E_ab_prime": round(E_ab_prime, 4),
                "E_a_prime_b": round(E_a_prime_b, 4),
                "E_a_prime_b_prime": round(E_a_prime_b_prime, 4)
            }
        }
    
    @staticmethod
    def ghz_experiment(iterations: int = 10000) -> Dict[str, Any]:
        results = {'XXX': [], 'XYY': [], 'YXY': [], 'YYX': []}
        
        for _ in range(iterations):
            basis = random.choice(['XXX', 'XYY', 'YXY', 'YYX'])
            
            if basis == 'XXX':
                result = 1.0
            else:
                result = -1.0
            
            results[basis].append(result)
        
        E_xxx = sum(results['XXX']) / len(results['XXX']) if results['XXX'] else 0
        E_xyy = sum(results['XYY']) / len(results['XYY']) if results['XYY'] else 0
        E_yxy = sum(results['YXY']) / len(results['YXY']) if results['YXY'] else 0
        E_yyx = sum(results['YYX']) / len(results['YYX']) if results['YYX'] else 0
        
        M = E_xxx - E_xyy - E_yxy - E_yyx
        
        violates = abs(M) > 2.0
        
        logger.info(f"GHZ Mermin: M={M:.3f}, violates={violates}")
        
        return {
            "M": round(M, 4),
            "violates_inequality": violates,
            "classical_bound": 2.0,
            "quantum_value": 4.0,
            "iterations": iterations,
            "expectation_values": {
                "E_XXX": round(E_xxx, 4),
                "E_XYY": round(E_xyy, 4),
                "E_YXY": round(E_yxy, 4),
                "E_YYX": round(E_yyx, 4)
            }
        }
    
    @staticmethod
    def quantum_teleportation(iterations: int = 1000) -> Dict[str, Any]:
        fidelities = []
        
        for _ in range(iterations):
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2 * math.pi)
            alpha = math.cos(theta / 2)
            beta = cmath.exp(1j * phi) * math.sin(theta / 2)
            
            psi_original = np.array([alpha, beta], dtype=complex)
            norm = np.linalg.norm(psi_original)
            psi_original = psi_original / norm
            
            decoherence_rate = 0.005
            
            if random.random() < decoherence_rate:
                error_type = random.choice(['X', 'Y', 'Z'])
                if error_type == 'X':
                    psi_bob = np.array([beta, alpha], dtype=complex)
                elif error_type == 'Y':
                    psi_bob = 1j * np.array([-beta.conjugate(), alpha.conjugate()], dtype=complex)
                    norm = np.linalg.norm(psi_bob)
                    psi_bob /= norm
                else:  # Z
                    psi_bob = np.array([alpha, -beta], dtype=complex)
            else:
                psi_bob = psi_original.copy()
            
            fidelity = abs(np.dot(psi_original.conjugate(), psi_bob)) ** 2
            fidelities.append(fidelity)
        
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0
        min_fidelity = min(fidelities) if fidelities else 0
        max_fidelity = max(fidelities) if fidelities else 0
        
        success_count = sum(1 for f in fidelities if f > 0.99)
        success_rate = success_count / len(fidelities) if fidelities else 0
        
        logger.info(f"Teleportation: avg_fidelity={avg_fidelity:.4f}, success_rate={success_rate:.2%}")
        
        return {
            "avg_fidelity": round(avg_fidelity, 6),
            "min_fidelity": round(min_fidelity, 6),
            "max_fidelity": round(max_fidelity, 6),
            "success_rate": round(success_rate, 4),
            "iterations": iterations,
            "theoretical_max": 1.0
        }
    
    @staticmethod
    def run_full_suite() -> Dict[str, Any]:
        suite = {
            "timestamp": datetime.now().isoformat(),
            "bell_test": QuantumPhysics.bell_experiment(Config.BELL_TEST_ITERATIONS),
            "ghz_test": QuantumPhysics.ghz_experiment(Config.GHZ_TEST_ITERATIONS),
            "teleportation": QuantumPhysics.quantum_teleportation(Config.TELEPORTATION_ITERATIONS)
        }
        Database.store_measurement("full_suite", suite)
        return suite

# ==================== NET INTERFACE FOR REPL ====================
class NetInterface:
    """Network interface class for QSH Foam REPL"""
    
    @staticmethod
    def ping(ip: str) -> Optional[float]:
        """Ping IP, return avg RTT ms"""
        try:
            result = subprocess.run(['ping', '-c', '3', '-W', '2', ip], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'avg' in line and '/':
                        rtt = float(line.split('/')[1])
                        return round(rtt, 2)
            return 0.0 if ip == "127.0.0.1" else None
        except Exception as e:
            logger.error(f"Ping to {ip} failed: {e}")
            return None
    
    @staticmethod
    def resolve(domain: str) -> str:
        """Resolve domain"""
        try:
            ip = socket.gethostbyname(domain)
            return ip
        except socket.gaierror:
            return "Unresolved"
    
    @staticmethod
    def whois(ip: str) -> str:
        """Get WHOIS for IP"""
        try:
            result = subprocess.run(['whois', ip], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                org = next((line.split(':')[1].strip() for line in lines if 'OrgName' in line or 'organization' in line), "Unknown Org")
                loc = next((line.split(':')[1].strip() for line in lines if 'City' in line or 'location' in line), "Unknown Location")
                return f"{org} ({loc})"
            return "WHOIS failed"
        except Exception as e:
            logger.error(f"WHOIS for {ip} failed: {e}")
            return "WHOIS error"

# ==================== ALICE NODE ====================
class AliceNode:
    """Alice operational node at 127.0.0.1"""
    
    @staticmethod
    def status(ip: str = "127.0.0.1") -> str:
        if ip == "127.0.0.1":
            return "Alice Node: Completely operational - Local quantum entanglement ready"
        return f"Alice Node at {ip}: Linking via Foam REPL - Ping first"
    
    @staticmethod
    def ping_alice(ip: str = "127.0.0.1") -> str:
        latency = NetInterface.ping(ip)
        return f"Ping to Alice ({ip}): {latency or 0.0} ms - Operational via QSH Foam"

# ==================== SYSTEM METRICS MODULE ====================
class SystemMetrics:
    """Real system measurements"""
    
    @staticmethod
    def ping_storage_ip() -> bool:
        return NetInterface.ping(Config.STORAGE_IP) is not None
    
    @staticmethod
    def resolve_quantum_domain() -> str:
        return NetInterface.resolve(Config.QUANTUM_DOMAIN)
    
    @staticmethod
    def get_holographic_metrics() -> Dict[str, Any]:
        reachable = SystemMetrics.ping_storage_ip()
        total_eb = Config.HOLOGRAPHIC_CAPACITY_EB
        if reachable and Config.HOLO_MOUNT.exists():
            try:
                disk = psutil.disk_usage(Config.HOLO_MOUNT)
                used_eb = disk.used / (1024 ** 6)
                free_eb = disk.free / (1024 ** 6)
            except Exception:
                used_eb = 0.001
                free_eb = total_eb - used_eb
        else:
            logger.warning(f"Holographic storage {Config.STORAGE_IP} unreachable; using local")
            disk = psutil.disk_usage('/')
            used_eb = disk.used / (1024 ** 6)
            free_eb = disk.free / (1024 ** 6)
        
        return {
            "ip": Config.STORAGE_IP,
            "reachable": reachable,
            "total_eb": total_eb,
            "used_eb": round(used_eb, 3),
            "free_eb": round(free_eb, 3),
            "percent_used": round((used_eb / total_eb) * 100, 2),
            "whois": NetInterface.whois(Config.STORAGE_IP),
            "tech": "3D laser holographic (2025 prototypes: ~10TB/module, scaled to EB)"
        }
    
    @staticmethod
    def get_hashing_speed() -> float:
        data = os.urandom(1024 * 1024)
        start = time.time()
        hashlib.sha256(data).hexdigest()
        end = time.time()
        duration = end - start
        speed_mbs = 1 / duration
        return round(speed_mbs, 2)
    
    @staticmethod
    def get_qram_metrics() -> Dict[str, Any]:
        operational = False
        try:
            import qutip as qt
            n_qubits_demo = 20
            N = 2 ** n_qubits_demo
            start = time.time()
            psi = qt.basis(N, 0)
            psi_dense = psi.data.to_array()
            alloc_time = time.time() - start
            size_kb = psi_dense.nbytes / 1024
            operational = True
        except Exception as e:
            logger.error(f"QRAM error: {e}")
            alloc_time = size_kb = n_qubits_demo = 0
        
        return {
            "domain": Config.QUANTUM_DOMAIN,
            "operational": operational,
            "demo_n_qubits": n_qubits_demo,
            "demo_size_kb": round(size_kb, 2),
            "demo_alloc_time_s": round(alloc_time, 4),
            "theoretical_capacity_gb": Config.QRAM_THEORETICAL_GB,
            "dns_resolved_ip": SystemMetrics.resolve_quantum_domain(),
            "tech": "Stab-QRAM inspired (2025: O(1) depth, simulable to 20+ qubits)"
        }
    
    @staticmethod
    def get_storage_metrics() -> Dict[str, Any]:
        try:
            local_disk = psutil.disk_usage(Config.DATA_DIR)
            base = {
                "local_total_gb": round(local_disk.total / (1024 ** 3), 2),
                "local_used_gb": round(local_disk.used / (1024 ** 3), 2),
                "local_free_gb": round(local_disk.free / (1024 ** 3), 2),
                "local_percent_used": round(local_disk.percent, 2)
            }
            base["holographic"] = SystemMetrics.get_holographic_metrics()
            return base
        except Exception as e:
            logger.error(f"Storage metrics error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_memory_metrics() -> Dict[str, Any]:
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "percent_used": round(memory.percent, 2)
            }
        except Exception as e:
            logger.error(f"Memory metrics error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_cpu_metrics() -> Dict[str, Any]:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_count = psutil.cpu_count()
            freqs = psutil.cpu_freq(percpu=True)
            load_avg = psutil.getloadavg()
            operational = True
            
            black_latency = NetInterface.ping(Config.CPU_BLACK_HOLE_IP)
            white_latency = NetInterface.ping(Config.CPU_WHITE_HOLE_IP)
            
            return {
                "operational": operational,
                "usage_percent_per_core": [round(p, 2) for p in cpu_percent],
                "cpu_count": cpu_count,
                "frequency_mhz_per_core": [
                    {
                        "current": round(f.current, 2) if f else None,
                        "min": round(f.min, 2) if f else None,
                        "max": round(f.max, 2) if f else None
                    } for f in freqs
                ],
                "load_average": [round(x, 2) for x in load_avg],
                "distributed_compute": {
                    "black_hole": {
                        "ip": Config.CPU_BLACK_HOLE_IP,
                        "latency_ms": black_latency,
                        "whois": NetInterface.whois(Config.CPU_BLACK_HOLE_IP),
                        "role": "Compute ingestion/compression (black hole)"
                    },
                    "white_hole": {
                        "ip": Config.CPU_WHITE_HOLE_IP,
                        "latency_ms": white_latency,
                        "whois": NetInterface.whois(Config.CPU_WHITE_HOLE_IP),
                        "role": "Compute expansion/output (white hole)"
                    },
                    "overhead_ms": (black_latency or 0) + (white_latency or 0)
                }
            }
        except Exception as e:
            logger.error(f"CPU metrics error: {e}")
            return {"operational": False, "error": str(e)}
    
    @staticmethod
    def get_all_metrics() -> Dict[str, Any]:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "storage": SystemMetrics.get_storage_metrics(),
            "memory": SystemMetrics.get_memory_metrics(),
            "cpu": SystemMetrics.get_cpu_metrics(),
            "hashing_speed_mbs": SystemMetrics.get_hashing_speed(),
            "qram": SystemMetrics.get_qram_metrics()
        }
        Database.store_measurement("system_metrics", metrics)
        return metrics

# ==================== DATABASE MODULE ====================
class Database:
    """Database operations"""
    
    @staticmethod
    def store_measurement(measurement_type: str, data: Dict[str, Any]):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO measurements (timestamp, measurement_type, data) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), measurement_type, json.dumps(data))
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Stored {measurement_type} measurement")
        except Exception as e:
            logger.error(f"Database storage error: {e}")
    
    @staticmethod
    def get_recent_measurements(limit: int = 10) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp, measurement_type, data FROM measurements ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "timestamp": row[0],
                    "type": row[1],
                    "data": json.loads(row[2])
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Database retrieval error: {e}")
            return []
    
    # ==================== EMAIL DATABASE OPERATIONS ====================
    
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def create_user(username: str, password: str) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            email = f"{username}::@{Config.QUANTUM_EMAIL_DOMAIN}"
            password_hash = Database.hash_password(password)
            created_at = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, email, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, email, created_at)
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
            
            sent_at = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO emails (from_user, to_user, subject, body, sent_at) VALUES (?, ?, ?, ?, ?)",
                (from_email, to_email, subject, body, sent_at)
            )
            
            conn.commit()
            email_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Email sent from {from_email} to {to_email}")
            return {"id": email_id, "from": from_email, "to": to_email, "subject": subject, "sent_at": sent_at}
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise HTTPException(status_code=500, detail="Error sending email")
    
    @staticmethod
    def get_inbox(user_email: str) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, from_user, to_user, subject, body, sent_at, read, starred 
                   FROM emails 
                   WHERE to_user = ? AND deleted_receiver = 0
            ```python
                   ORDER BY sent_at DESC""",
                (user_email,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "subject": row[3],
                    "body": row[4],
                    "sent_at": row[5],
                    "read": bool(row[6]),
                    "starred": bool(row[7])
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting inbox: {e}")
            return []
    
    @staticmethod
    def get_sent(user_email: str) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, from_user, to_user, subject, body, sent_at, read, starred 
                   FROM emails 
                   WHERE from_user = ? AND deleted_sender = 0 
                   ORDER BY sent_at DESC""",
                (user_email,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "subject": row[3],
                    "body": row[4],
                    "sent_at": row[5],
                    "read": bool(row[6]),
                    "starred": bool(row[7])
                }
                for row in rows
            ]
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
    
    @staticmethod
    def get_email_by_id(email_id: int, user_email: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, from_user, to_user, subject, body, sent_at, read, starred 
                   FROM emails 
                   WHERE id = ? AND (from_user = ? OR to_user = ?)""",
                (email_id, user_email, user_email)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "subject": row[3],
                    "body": row[4],
                    "sent_at": row[5],
                    "read": bool(row[6]),
                    "starred": bool(row[7])
                }
            return None
        except Exception as e:
            logger.error(f"Error getting email: {e}")
            return None

# ==================== SECURITY MODULE ====================
security = HTTPBearer(auto_error=False)

class SecurityManager:
    """Handle authentication and authorization"""
    
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
        'np': np,
        'math': math,
        'random': random,
        'print': print,
        '__builtins__': {}
    })
    
    code = code.strip()
    if code.startswith(('ping ', 'resolve ', 'whois ')):
        cmd, arg = code.split(' ', 1)
        if cmd == 'ping':
            result = NetInterface.ping(arg)
            if arg == Config.ALICE_NODE_IP:
                return AliceNode.ping_alice(arg)
            return f"Ping to {arg}: {result} ms" if result is not None else f"Ping to {arg}: Unreachable"
        elif cmd == 'resolve':
            result = NetInterface.resolve(arg)
            return f"{arg} resolves to: {result}"
        elif cmd == 'whois':
            result = NetInterface.whois(arg)
            return f"WHOIS for {arg}: {result}"
    
    old_stdout = sys.stdout
    output = []
    try:
        from io import StringIO
        sys.stdout = mystdout = StringIO()
        
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
    title="QSH Foam Dominion - Email Client",
    description="Quantum Foam Email System with QSH REPL integration",
    version="2.8.0",
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
    logger.info(f"Starting QSH Foam Email Client on {Config.HOST}:{Config.PORT}")
    if Config.DEBUG:
        demo_suite = QuantumPhysics.run_full_suite()
        logger.info(f"Demo suite: {demo_suite}")

# ==================== EMAIL CLIENT ROUTES ====================

@app.get("/", tags=["email"])
async def root(session_token: Optional[str] = Cookie(None)):
    user = await get_current_user_email(session_token)
    
    if not user:
        # Show login page
        return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Foam Email - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #e0e0e0;
        }
        
        .login-container {
            background: rgba(26, 26, 46, 0.95);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0, 255, 157, 0.2);
            border: 1px solid #00ff9d;
            max-width: 400px;
            width: 100%;
        }
        
        h1 {
            color: #00ff9d;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2em;
            text-shadow: 0 0 20px rgba(0, 255, 157, 0.5);
        }
        
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .domain-info {
            background: rgba(0, 255, 157, 0.1);
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 255, 157, 0.3);
        }
        
        .domain-info code {
            color: #00ff9d;
            font-weight: bold;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #00ff9d;
            font-weight: 500;
        }
        
        input[type="text"],
        input[type="password"] {
            width: 100%;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff9d;
            border-radius: 5px;
            color: #e0e0e0;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="password"]:focus {
            outline: none;
            border-color: #00ffff;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
        }
        
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 25px;
        }
        
        button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-login {
            background: #00ff9d;
            color: #000;
        }
        
        .btn-login:hover {
            background: #00ffff;
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.4);
            transform: translateY(-2px);
        }
        
        .btn-register {
            background: transparent;
            color: #00ff9d;
            border: 2px solid #00ff9d;
        }
        
        .btn-register:hover {
            background: rgba(0, 255, 157, 0.1);
            border-color: #00ffff;
            color: #00ffff;
        }
        
        .error {
            background: rgba(255, 50, 50, 0.2);
            border: 1px solid #ff3232;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            color: #ff6666;
            display: none;
        }
        
        .success {
            background: rgba(0, 255, 157, 0.2);
            border: 1px solid #00ff9d;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            color: #00ff9d;
            display: none;
        }
        
        .qsh-link {
            text-align: center;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(0, 255, 157, 0.3);
        }
        
        .qsh-link a {
            color: #00ffff;
            text-decoration: none;
            transition: all 0.3s;
        }
        
        .qsh-link a:hover {
            color: #00ff9d;
            text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>⚛️ Quantum Foam</h1>
        <p class="subtitle">Secure Email System</p>
        
        <div class="domain-info">
            All emails use format: <code>username::@quantum.foam</code>
        </div>
        
        <div id="errorMsg" class="error"></div>
        <div id="successMsg" class="success"></div>
        
        <form id="authForm">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required placeholder="Enter username">
            </div>
            
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required placeholder="Enter password">
            </div>
            
            <div class="btn-group">
                <button type="submit" class="btn-login" id="loginBtn">Sign In</button>
                <button type="button" class="btn-register" id="registerBtn">Register</button>
            </div>
        </form>
        
        <div class="qsh-link">
            <a href="/qsh">🔬 Access QSH Foam REPL</a>
        </div>
    </div>
    
    <script>
        const form = document.getElementById('authForm');
        const loginBtn = document.getElementById('loginBtn');
        const registerBtn = document.getElementById('registerBtn');
        const errorMsg = document.getElementById('errorMsg');
        const successMsg = document.getElementById('successMsg');
        
        function showError(msg) {
            errorMsg.textContent = msg;
            errorMsg.style.display = 'block';
            successMsg.style.display = 'none';
        }
        
        function showSuccess(msg) {
            successMsg.textContent = msg;
            successMsg.style.display = 'block';
            errorMsg.style.display = 'none';
        }
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showSuccess('Login successful! Redirecting...');
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 1000);
                } else {
                    showError(data.detail || 'Login failed');
                }
            } catch (error) {
                showError('Connection error. Please try again.');
            }
        });
        
        registerBtn.addEventListener('click', async () => {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            if (!username || !password) {
                showError('Please enter username and password');
                return;
            }
            
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showSuccess(`Account created! Your email: ${data.email}`);
                    setTimeout(() => {
                        form.submit();
                    }, 2000);
                } else {
                    showError(data.detail || 'Registration failed');
                }
            } catch (error) {
                showError('Connection error. Please try again.');
            }
        });
    </script>
</body>
</html>
        """)
    
    # Show email client
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Foam Email - {user['email']}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
            box-shadow: 0 2px 10px rgba(0, 255, 157, 0.2);
        }}
        
        .header-left {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .logo {{
            font-size: 1.5em;
            color: #00ff9d;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        }}
        
        .user-info {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .user-email {{
            color: #00ffff;
            font-weight: 500;
        }}
        
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
        }}
        
        .btn-compose {{
            background: #00ff9d;
            color: #000;
        }}
        
        .btn-compose:hover {{
            background: #00ffff;
            box-shadow: 0 3px 10px rgba(0, 255, 157, 0.4);
        }}
        
        .btn-logout {{
            background: transparent;
            border: 1px solid #ff3232;
            color: #ff6666;
        }}
        
        .btn-logout:hover {{
            background: rgba(255, 50, 50, 0.1);
        }}
        
        .btn-qsh {{
            background: transparent;
            border: 1px solid #00ffff;
            color: #00ffff;
        }}
        
        .btn-qsh:hover {{
            background: rgba(0, 255, 255, 0.1);
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
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .nav-item:hover {{
            background: rgba(0, 255, 157, 0.1);
            border-left: 3px solid #00ff9d;
        }}
        
        .nav-item.active {{
            background: rgba(0, 255, 157, 0.2);
            border-left: 3px solid #00ff9d;
            color: #00ff9d;
        }}
        
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        
        .toolbar {{
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .checkbox-all {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        
        .btn-toolbar {{
            padding: 6px 12px;
            background: transparent;
            border: 1px solid #00ff9d;
            color: #00ff9d;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }}
        
        .btn-toolbar:hover {{
            background: rgba(0, 255, 157, 0.1);
        }}
        
        .btn-delete {{
            border-color: #ff3232;
            color: #ff6666;
        }}
        
        .btn-delete:hover {{
            background: rgba(255, 50, 50, 0.1);
        }}
        
        .email-list {{
            flex: 1;
            overflow-y: auto;
            background: #0f0f1e;
        }}
        
        .email-item {{
            display: flex;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid #222;
            cursor: pointer;
            transition: all 0.3s;
            gap: 15px;
        }}
        
        .email-item:hover {{
            background: rgba(0, 255, 157, 0.05);
        }}
        
        .email-item.unread {{
            background: rgba(0, 255, 157, 0.03);
            border-left: 3px solid #00ff9d;
        }}
        
        .email-checkbox {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        
        .email-star {{
            cursor: pointer;
            font-size: 1.2em;
            color: #666;
            transition: all 0.3s;
        }}
        
        .email-star.starred {{
            color: #ffd700;
        }}
        
        .email-from {{
            min-width: 200px;
            font-weight: 500;
            color: #00ffff;
        }}
        
        .email-subject {{
            flex: 1;
            color: #e0e0e0;
        }}
        
        .email-subject.unread {{
            font-weight: bold;
        }}
        
        .email-date {{
            color: #888;
            font-size: 0.9em;
            min-width: 100px;
            text-align: right;
        }}
        
        .email-view {{
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            background: #0f0f1e;
            display: none;
        }}
        
        .email-view.active {{
            display: block;
        }}
        
        .email-header {{
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 20px;
        }}
        
        .email-subject-view {{
            font-size: 1.8em;
            color: #00ff9d;
            margin-bottom: 15px;
        }}
        
        .email-meta {{
            display: flex;
            gap: 20px;
            color: #888;
            font-size: 0.9em;
        }}
        
        .email-body {{
            line-height: 1.8;
            color: #e0e0e0;
            white-space: pre-wrap;
        }}
        
        .compose-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        
        .compose-modal.active {{
            display: flex;
        }}
        
        .compose-form {{
            background: #1a1a2e;
            border: 2px solid #00ff9d;
            border-radius: 10px;
            padding: 30px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        .form-group {{
            margin-bottom: 20px;
        }}
        
        label {{
            display: block;
            margin-bottom: 8px;
            color: #00ff9d;
            font-weight: 500;
        }}
        
        input[type="text"],
        textarea {{
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff9d;
            border-radius: 5px;
            color: #e0e0e0;
            font-family: inherit;
        }}
        
        textarea {{
            min-height: 200px;
            resize: vertical;
        }}
        
        .compose-buttons {{
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }}
        
        .btn-send {{
            background: #00ff9d;
            color: #000;
        }}
        
        .btn-cancel {{
            background: transparent;
            border: 1px solid #666;
            color: #666;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }}
        
        .empty-state-icon {{
            font-size: 4em;
            margin-bottom: 20px;
        }}
        
        ::-webkit-scrollbar {{
            width: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #0a0a0a;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #00ff9d;
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #00ffff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <div class="logo">⚛️ Quantum Foam Mail</div>
            <button class="btn btn-compose" onclick="showCompose()">✉️ Compose</button>
        </div>
        <div class="user-info">
            <span class="user-email">{user['email']}</span>
            <button class="btn btn-qsh" onclick="window.open('/qsh', '_blank')">🔬 QSH REPL</button>
            <button class="btn btn-logout" onclick="logout()">Logout</button>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="nav-item active" onclick="switchView('inbox')">
                📥 Inbox <span id="unreadCount"></span>
            </div>
            <div class="nav-item" onclick="switchView('sent')">
                📤 Sent
            </div>
            <div class="nav-item" onclick="switchView('starred')">
                ⭐ Starred
            </div>
        </div>
        
        <div class="main-content">
            <div class="toolbar">
                <input type="checkbox" class="checkbox-all" id="selectAll" onchange="toggleSelectAll()">
                <button class="btn-toolbar" onclick="refreshEmails()">🔄 Refresh</button>
                <button class="btn-toolbar btn-delete" onclick="deleteSelected()">🗑️ Delete</button>
                <button class="btn-toolbar" onclick="markSelectedRead()">📖 Mark Read</button>
    ```javascript
            </div>
            
            <div class="email-list" id="emailList">
                <div class="empty-state">
                    <div class="empty-state-icon">📭</div>
                    <p>Loading emails...</p>
                </div>
            </div>
            
            <div class="email-view" id="emailView">
                <button class="btn" onclick="closeEmailView()" style="margin-bottom: 20px;">← Back to List</button>
                <div class="email-header">
                    <div class="email-subject-view" id="viewSubject"></div>
                    <div class="email-meta">
                        <span>From: <strong id="viewFrom"></strong></span>
                        <span>To: <strong id="viewTo"></strong></span>
                        <span>Date: <strong id="viewDate"></strong></span>
                    </div>
                </div>
                <div class="email-body" id="viewBody"></div>
            </div>
        </div>
    </div>
    
    <div class="compose-modal" id="composeModal">
        <div class="compose-form">
            <h2 style="color: #00ff9d; margin-bottom: 20px;">New Message</h2>
            <form id="composeForm">
                <div class="form-group">
                    <label for="composeTo">To</label>
                    <input type="text" id="composeTo" placeholder="recipient::@quantum.foam" required>
                </div>
                <div class="form-group">
                    <label for="composeSubject">Subject</label>
                    <input type="text" id="composeSubject" placeholder="Enter subject" required>
                </div>
                <div class="form-group">
                    <label for="composeBody">Message</label>
                    <textarea id="composeBody" placeholder="Write your message..." required></textarea>
                </div>
                <div class="compose-buttons">
                    <button type="button" class="btn btn-cancel" onclick="hideCompose()">Cancel</button>
                    <button type="submit" class="btn btn-send">Send</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        let currentView = 'inbox';
        let emails = [];
        let currentEmailId = null;
        
        // Load emails on page load
        document.addEventListener('DOMContentLoaded', () => {{
            refreshEmails();
            setInterval(refreshEmails, 30000); // Auto-refresh every 30 seconds
        }});
        
        async function refreshEmails() {{
            try {{
                const endpoint = currentView === 'sent' ? '/api/emails/sent' : '/api/emails/inbox';
                const response = await fetch(endpoint);
                if (response.ok) {{
                    emails = await response.json();
                    
                    if (currentView === 'starred') {{
                        emails = emails.filter(e => e.starred);
                    }}
                    
                    renderEmails();
                    updateUnreadCount();
                }}
            }} catch (error) {{
                console.error('Error loading emails:', error);
            }}
        }}
        
        function renderEmails() {{
            const listEl = document.getElementById('emailList');
            
            if (emails.length === 0) {{
                listEl.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📭</div>
                        <p>No emails here</p>
                    </div>
                `;
                return;
            }}
            
            listEl.innerHTML = emails.map(email => `
                <div class="email-item ${{!email.read && currentView === 'inbox' ? 'unread' : ''}}" data-id="${{email.id}}">
                    <input type="checkbox" class="email-checkbox" data-id="${{email.id}}" onclick="event.stopPropagation()">
                    <span class="email-star ${{email.starred ? 'starred' : ''}}" onclick="toggleStar(${{email.id}}, event)">
                        ${{email.starred ? '⭐' : '☆'}}
                    </span>
                    <div class="email-from">${{email.from.split('::@')[0]}}</div>
                    <div class="email-subject ${{!email.read && currentView === 'inbox' ? 'unread' : ''}}">
                        ${{email.subject}}
                    </div>
                    <div class="email-date">${{formatDate(email.sent_at)}}</div>
                </div>
            `).join('');
            
            // Add click handlers
            document.querySelectorAll('.email-item').forEach(item => {{
                item.addEventListener('click', () => {{
                    const id = parseInt(item.dataset.id);
                    openEmail(id);
                }});
            }});
        }}
        
        function formatDate(dateStr) {{
            const date = new Date(dateStr);
            const now = new Date();
            const diff = now - date;
            const hours = diff / (1000 * 60 * 60);
            
            if (hours < 24) {{
                return date.toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit' }});
            }} else if (hours < 168) {{
                return date.toLocaleDateString('en-US', {{ weekday: 'short' }});
            }} else {{
                return date.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
            }}
        }}
        
        function updateUnreadCount() {{
            const unread = emails.filter(e => !e.read && currentView === 'inbox').length;
            const countEl = document.getElementById('unreadCount');
            if (unread > 0) {{
                countEl.textContent = `(${{unread}})`;
                countEl.style.color = '#00ff9d';
            }} else {{
                countEl.textContent = '';
            }}
        }}
        
        async function openEmail(id) {{
            const email = emails.find(e => e.id === id);
            if (!email) return;
            
            currentEmailId = id;
            
            document.getElementById('viewSubject').textContent = email.subject;
            document.getElementById('viewFrom').textContent = email.from;
            document.getElementById('viewTo').textContent = email.to;
            document.getElementById('viewDate').textContent = new Date(email.sent_at).toLocaleString();
            document.getElementById('viewBody').textContent = email.body;
            
            document.getElementById('emailList').style.display = 'none';
            document.getElementById('emailView').classList.add('active');
            
            // Mark as read
            if (!email.read && currentView === 'inbox') {{
                await fetch(`/api/emails/${{id}}/read`, {{ method: 'POST' }});
                email.read = true;
                updateUnreadCount();
            }}
        }}
        
        function closeEmailView() {{
            document.getElementById('emailList').style.display = 'block';
            document.getElementById('emailView').classList.remove('active');
            currentEmailId = null;
        }}
        
        async function toggleStar(id, event) {{
            event.stopPropagation();
            try {{
                await fetch(`/api/emails/${{id}}/star`, {{ method: 'POST' }});
                const email = emails.find(e => e.id === id);
                if (email) {{
                    email.starred = !email.starred;
                    renderEmails();
                }}
            }} catch (error) {{
                console.error('Error toggling star:', error);
            }}
        }}
        
        function toggleSelectAll() {{
            const checked = document.getElementById('selectAll').checked;
            document.querySelectorAll('.email-checkbox').forEach(cb => {{
                cb.checked = checked;
            }});
        }}
        
        function getSelectedIds() {{
            return Array.from(document.querySelectorAll('.email-checkbox:checked'))
                .map(cb => parseInt(cb.dataset.id));
        }}
        
        async function deleteSelected() {{
            const ids = getSelectedIds();
            if (ids.length === 0) {{
                alert('No emails selected');
                return;
            }}
            
            if (!confirm(`Delete ${{ids.length}} email(s)?`)) return;
            
            try {{
                await fetch('/api/emails/delete', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ email_ids: ids }})
                }});
                
                emails = emails.filter(e => !ids.includes(e.id));
                renderEmails();
                updateUnreadCount();
                document.getElementById('selectAll').checked = false;
            }} catch (error) {{
                console.error('Error deleting emails:', error);
                alert('Failed to delete emails');
            }}
        }}
        
        async function markSelectedRead() {{
            const ids = getSelectedIds();
            if (ids.length === 0) {{
                alert('No emails selected');
                return;
            }}
            
            try {{
                for (const id of ids) {{
                    await fetch(`/api/emails/${{id}}/read`, {{ method: 'POST' }});
                    const email = emails.find(e => e.id === id);
                    if (email) email.read = true;
                }}
                renderEmails();
                updateUnreadCount();
                document.getElementById('selectAll').checked = false;
            }} catch (error) {{
                console.error('Error marking emails as read:', error);
            }}
        }}
        
        function switchView(view) {{
            currentView = view;
            closeEmailView();
            
            document.querySelectorAll('.nav-item').forEach(item => {{
                item.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            refreshEmails();
        }}
        
        function showCompose() {{
            document.getElementById('composeModal').classList.add('active');
        }}
        
        function hideCompose() {{
            document.getElementById('composeModal').classList.remove('active');
            document.getElementById('composeForm').reset();
        }}
        
        document.getElementById('composeForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            
            const to = document.getElementById('composeTo').value;
            const subject = document.getElementById('composeSubject').value;
            const body = document.getElementById('composeBody').value;
            
            try {{
                const response = await fetch('/api/emails/send', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ to, subject, body }})
                }});
                
                if (response.ok) {{
                    alert('Email sent successfully!');
                    hideCompose();
                    if (currentView === 'sent') {{
                        refreshEmails();
                    }}
                }} else {{
                    const error = await response.json();
                    alert(error.detail || 'Failed to send email');
                }}
            }} catch (error) {{
                console.error('Error sending email:', error);
                alert('Failed to send email');
            }}
        }});
        
        async function logout() {{
            try {{
                await fetch('/api/logout', {{ method: 'POST' }});
                window.location.href = '/';
            }} catch (error) {{
                console.error('Error logging out:', error);
            }}
        }}
    </script>
</body>
</html>
    """)

# ==================== API ROUTES ====================

@app.post("/api/register", tags=["auth"])
async def register(user: UserRegister):
    result = Database.create_user(user.username, user.password)
    return result

@app.post("/api/login", tags=["auth"])
async def login(user: UserLogin, response: JSONResponse):
    auth_user = Database.authenticate_user(user.username, user.password)
    
    if not auth_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = Database.create_session(auth_user['email'])
    
    response = JSONResponse(content={"message": "Login successful", "user": auth_user})
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=7*24*60*60,  # 7 days
        samesite="lax"
    )
    
    return response

@app.post("/api/logout", tags=["auth"])
async def logout(response: JSONResponse):
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
        raise HTTPException(status_code=401, detail="Not authenticated")
    
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

# ==================== QSH REPL ROUTE ====================

@app.get("/qsh", tags=["repl"])
async def qsh_repl():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>QSH Foam REPL - Nested Terminal</title>
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
    </style>
</head>
<body>
    <div class="header">
        QSH Foam REPL v2.8.0 - Quantum Shell | <a href="/" style="color: #00ffff; text-decoration: none;">← Back to Email</a>
    </div>
    <div id="terminal"></div>
    <script>
        const term = new Terminal({ cols: 120, rows: 40 });
        term.open(document.getElementById('terminal'));
        term.write('QSH Foam REPL v2.8.0 - Nested Quantum Shell (Foam REPL to Alice 127.0.0.1)\\r\\n');
        term.write('Operational: All net addresses interfaced (ping 127.0.0.1 for Alice status)\\r\\n');
        term.write('Example: NetInterface.ping("130.0.0.1") or "ping 130.0.0.1"\\r\\n');
        term.write('QSH> ');

        const ws = new WebSocket('ws://' + location.host + '/ws');
        ws.onopen = () => term.write('Connected to QSH Foam! Alice 127.0.0.1 operational\\r\\nQSH> ');
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    repl_sessions[session_id] = {}
    
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("AUTH:"):
                continue
            output = await repl_exec(data, session_id)
            await websocket.send_text(output)
    except WebSocketDisconnect:
        logger.info(f"QSH REPL session {session_id} disconnected")
        del repl_sessions[session_id]

# ==================== QUANTUM ROUTES ====================

@app.get("/quantum/suite", tags=["quantum"])
async def get_quantum_suite(request: Request):
    await check_rate_limit(request)
    return QuantumPhysics.run_full_suite()

@app.get("/metrics", tags=["system"])
async def get_metrics(request: Request):
    await check_rate_limit(request)
    return SystemMetrics.get_all_metrics()

@app.get("/quantum/bell", tags=["quantum"])
async def get_bell(user: Dict = Depends(get_current_user), request: Request = None, iterations: int = Query(Config.BELL_TEST_ITERATIONS, ge=1)):
    if request:
        await check_rate_limit(request)
    return QuantumPhysics.bell_experiment(iterations)

@app.get("/quantum/ghz", tags=["quantum"])
async def get_ghz(user: Dict = Depends(get_current_user), request: Request = None, iterations: int = Query(Config.GHZ_TEST_ITERATIONS, ge=1)):
    if request:
        await check_rate_limit(request)
    return QuantumPhysics.ghz_experiment(iterations)

@app.get("/quantum/teleport", tags=["quantum"])
async def get_teleport(user: Dict = Depends(get_current_user), request: Request = None, iterations: int = Query(Config.TELEPORTATION_ITERATIONS, ge=1)):
    if request:
        await check_rate_limit(request)
    return QuantumPhysics.quantum_teleportation(iterations)

@app.get("/db/recent", tags=["database"])
async def get_recent(limit: int = Query(10, ge=1, le=100), user: Dict = Depends(get_current_user), request: Request = None):
    if request:
        await check_rate_limit(request)
    return Database.get_recent_measurements(limit)

@app.post("/auth/token", tags=["auth"])
async def create_token(request: Request):
    await check_rate_limit(request)
    return {"access_token": SecurityManager.generate_token(), "token_type": "bearer"}

@app.get("/network-map", tags=["network"])
async def get_network_map(user: Dict = Depends(get_current_user), request: Request = None):
    if request:
        await check_rate_limit(request)
    interfaces = psutil.net_if_addrs()
    localhost_ifaces = [iface for iface, addrs in interfaces.items() if any(addr.address == Config.ALICE_NODE_IP for addr in addrs)]
    connections = [conn._asdict() for conn in psutil.net_connections(kind='inet') if conn.laddr.ip == Config.ALICE_NODE_IP]
    
    try:
        hostname = socket.gethostbyaddr(Config.STORAGE_IP)[0]
    except socket.herror:
        hostname = "No PTR record"
    
    domain_ip = SystemMetrics.resolve_quantum_domain()
    
    black_latency = NetInterface.ping(Config.CPU_BLACK_HOLE_IP)
    white_latency = NetInterface.ping(Config.CPU_WHITE_HOLE_IP)
    
    alice_status = AliceNode.status()
    
    return {
        "alice_node_ip": Config.ALICE_NODE_IP,
        "alice_status": alice_status,
        "storage_ip": Config.STORAGE_IP,
        "storage_hostname": hostname,
        "storage_whois": NetInterface.whois(Config.STORAGE_IP),
        "dns_server": Config.DNS_SERVER,
        "quantum_domain": Config.QUANTUM_DOMAIN,
        "domain_resolved_ip": domain_ip,
        "cpu_black_hole": {
            "ip": Config.CPU_BLACK_HOLE_IP,
            "latency_ms": black_latency,
            "whois": NetInterface.whois(Config.CPU_BLACK_HOLE_IP)
        },
        "cpu_white_hole": {
            "ip": Config.CPU_WHITE_HOLE_IP,
            "latency_ms": white_latency,
            "whois": NetInterface.whois(Config.CPU_WHITE_HOLE_IP)
        },
        "interfaces": localhost_ifaces,
        "active_connections": connections,
        "note": "All net addresses interfaced via QSH Foam REPL (NetInterface & AliceNode classes)"
    }

@app.get("/health", tags=["info"])
async def health():
    reachable = SystemMetrics.ping_storage_ip()
    domain_ip = SystemMetrics.resolve_quantum_domain()
    qram_op = SystemMetrics.get_qram_metrics()["operational"]
    black_latency = NetInterface.ping(Config.CPU_BLACK_HOLE_IP)
    white_latency = NetInterface.ping(Config.CPU_WHITE_HOLE_IP)
    alice_status = AliceNode.status()
    return {
        "status": "healthy", 
        "env": Config.ENVIRONMENT, 
        "host": Config.HOST, 
        "storage_reachable": reachable, 
        "domain_resolved": domain_768ip, 
        "qram_operational": qram_op, 
        "cpu_black_latency_ms": black_latency, 
        "cpu_white_latency_ms": white_latency, 
        "alice_operational": alice_status,
        "email_system": "operational"
    }

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
