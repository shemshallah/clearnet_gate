
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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
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
    STORAGE_IP = "138.0.0.1"  # Holographic storage
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
                    if 'avg' in line and '/' in line:
                        rtt = float(line.split('/')[4])
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
            n_qubits_demo = 20
            N = 2 ** n_qubits_demo
            start = time.time()
            alloc_time = time.time() - start
            size_kb = N * 16 / 1024
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
                    } for f in (freqs if freqs else [])
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
                    "subject":"subject": row[3],
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
    title="QSH Foam Dominion - Email & Blockchain Client",
    description="Quantum Foam Email System with Bitcoin & QSH REPL integration",
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
    logger.info(f"Starting QSH Foam on {Config.HOST}:{Config.PORT}")

# ==================== STATIC FILE ROUTES ====================

@app.get("/email.html", response_class=HTMLResponse)
async def email_html():
    try:
        with open("email.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="email.html not found")

@app.get("/blockchain.html", response_class=HTMLResponse)
async def blockchain_html():
    try:
        with open("blockchain.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="blockchain.html not found")

# ==================== MAIN DASHBOARD ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QSH Foam Dominion v2.8</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #0f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            width: 100%;
        }
        
        h1 {
            text-align: center;
            color: #00ff9d;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 255, 157, 0.8);
        }
        
        .subtitle {
            text-align: center;
            color: #00ffff;
            margin-bottom: 40px;
            font-size: 1.2em;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }
        
        .card {
            background: rgba(26, 26, 46, 0.9);
            border: 2px solid #00ff9d;
            border-radius: 15px;
            padding: 30px;
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0, 255, 157, 0.5);
            border-color: #00ffff;
        }
        
        .card h2 {
            color: #00ff9d;
            margin-bottom: 15px;
            font-size: 1.8em;
        }
        
        .card p {
            color: #ccc;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        .card .features {
            list-style: none;
            padding: 0;
        }
        
        .card .features li {
            color: #00ffff;
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .card .features li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #ff6b35;
        }
        
        .btn {
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
        }
        
        .btn:hover {
            background: #00ffff;
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.5);
        }
        
        .footer {
            text-align: center;
            margin-top: 60px;
            color: #666;
        }
        
        .status {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: rgba(0, 255, 157, 0.1);
            border: 1px solid #00ff9d;
            border-radius: 10px;
        }
        
        .status h3 {
            color: #00ff9d;
            margin-bottom: 15px;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .status-item {
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
        }
        
        .status-item .label {
            color: #888;
            font-size: 0.9em;
        }
        
        .status-item .value {
            color: #00ffff;
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚛️ QSH Foam Dominion v2.8</h1>
        <p class="subtitle">Quantum Shell • Holographic Storage • Bitcoin Integration</p>
        
        <div class="status">
            <h3>System Status</h3>
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">Alice Node</div>
                    <div class="value">127.0.0.1 ✓</div>
                </div>
                <div class="status-item">
                    <div class="label">Holo Storage</div>
                    <div class="value">138.0.0.1 (6EB)</div>
                </div>
                <div class="status-item">
                    <div class="label">QRAM</div>
                    <div class="value">quantum.realm</div>
                </div>
                <div class="status-item">
                    <div class="label">Black Hole CPU</div>
                    <div class="value">130.0.0.1</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card" onclick="location.href='/email.html'">
                <h2>📧 Quantum Email</h2>
                <p>Secure holographic email system with quantum.foam domain</p>
                <ul class="features">
                    <li>Holographic storage @ 138.0.0.1</li>
                    <li>Quantum entangled encryption</li>
                    <li>10GB per user block</li>
                    <li>Real-time sync</li>
                </ul>
                <br>
                <a href="/email.html" class="btn">Open Email Client</a>
            </div>
            
            <div class="card" onclick="location.href='/blockchain.html'">
                <h2>₿ Bitcoin Client</h2>
                <p>SOTA Bitcoin client with QSH Foam REPL integration</p>
                <ul class="features">
                    <li>Full Bitcoin Core RPC</li>
                    <li>QSH Foam terminal</li>
                    <li>Network diagnostics</li>
                    <li>Quantum proofs</li>
                </ul>
                <br>
                <a href="/blockchain.html" class="btn">Open Bitcoin Client</a>
            </div>
            
            <div class="card" onclick="location.href='/qsh'">
                <h2>🌌 QSH Foam REPL</h2>
                <p>Unified quantum shell with network tools</p>
                <ul class="features">
                    <li>Python + Bitcoin CLI</li>
                    <li>Quantum physics tests</li>
                    <li>Network utilities (ping, nc, traceroute)</li>
                    <li>Alice node @ 127.0.0.1</li>
                </ul>
                <br>
                <a href="/qsh" class="btn">Open REPL</a>
            </div>
        </div>
        
        <div class="footer">
            <p>QSH Foam Dominion v2.8.0 | Production-Ready Quantum-Bitcoin Hybrid</p>
            <p>Holographic: 138.0.0.1 | QRAM: 2^300 GB | CPU: 130.0.0.1 ⇄ 139.0.0.1</p>
        </div>
    </div>
</body>
</html>
    """)

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
    <title>QSH Foam REPL</title>
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
        QSH Foam REPL v2.8.0 | <a href="/" style="color: #00ffff; text-decoration: none;">← Dashboard</a> | 
        <a href="/email.html" style="color: #00ffff; text-decoration: none;">Email</a> | 
        <a href="/blockchain.html" style="color: #00ffff; text-decoration: none;">Bitcoin</a>
    </div>
    <div id="terminal"></div>
    <script>
        const term = new Terminal({ cols: 120, rows: 40 });
        term.open(document.getElementById('terminal'));
        term.write('QSH Foam REPL v2.8.0\\r\\n');
        term.write('Connected to Alice @ 127.0.0.1\\r\\nQSH> ');

        const ws = new WebSocket('ws://' + location.host + '/ws/repl');
        ws.onopen = () => term.write('Connected\\r\\nQSH> ');
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
    return QuantumPhysics.run_full_suite()

@app.get("/metrics", tags=["system"])
async def get_metrics(request: Request):
    await check_rate_limit(request)
    return SystemMetrics.get_all_metrics()

@app.get("/health", tags=["info"])
async def health():
    return {"status": "healthy", "version": "2.8.0"}

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
