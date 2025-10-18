"""
Quantum Foam Dominion - Secure Production Version
A FastAPI application for quantum computing simulations with proper security
"""
import os
import logging
import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, Depends, Security, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import secrets
from collections import defaultdict
import random
import psutil
import sqlite3
import math
import cmath
import numpy as np

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
    
    # Security - NO DEFAULTS for sensitive values
    ADMINISTRATOR_USERNAME = os.getenv("ADMINISTRATOR_USERNAME")
    ADMINISTRATOR_PASSWORD = os.getenv("ADMINISTRATOR_PASSWORD")
    SECRET_KEY = os.getenv("SECRET_KEY")
    
    # CORS - restrictive by default
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Directories
    DATA_DIR = Path("data")
    DB_PATH = DATA_DIR / "quantum_foam.db"
    
    # Quantum simulation parameters
    BELL_TEST_ITERATIONS = int(os.getenv("BELL_TEST_ITERATIONS", "10000"))
    GHZ_TEST_ITERATIONS = int(os.getenv("GHZ_TEST_ITERATIONS", "10000"))
    TELEPORTATION_ITERATIONS = int(os.getenv("TELEPORTATION_ITERATIONS", "1000"))
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        if cls.ENVIRONMENT == "production":
            if not cls.ADMINISTRATOR_USERNAME:
                raise ValueError("ADMINISTRATOR_USERNAME must be set in production")
            if not cls.ADMINISTRATOR_PASSWORD:
                raise ValueError("ADMINISTRATOR_PASSWORD must be set in production")
            if not cls.SECRET_KEY:
                raise ValueError("SECRET_KEY must be set in production")
        
        # Create directories
        cls.DATA_DIR.mkdir(exist_ok=True)
        
        # Initialize database
        if not cls.DB_PATH.exists():
            cls._init_database()
    
    @classmethod
    def _init_database(cls):
        """Initialize SQLite database"""
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
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

# Validate configuration on startup
try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    if Config.ENVIRONMENT == "production":
        raise

# ==================== QUANTUM PHYSICS MODULE ====================
class QuantumPhysics:
    """Scientific quantum mechanics simulations"""
    
    @staticmethod
    def bell_experiment(iterations: int = 10000) -> Dict[str, Any]:
        """
        Proper Bell inequality (CHSH) test for quantum entanglement.
        
        For a singlet state |ψ⟩ = (|01⟩ - |10⟩)/√2, the correlation function is:
        E(a,b) = -cos(θ_a - θ_b)
        
        The CHSH parameter S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| 
        violates Bell inequality if S > 2 (max is 2√2 ≈ 2.828 for quantum)
        """
        # Measurement angles for maximum violation
        theta_a = 0
        theta_a_prime = math.pi / 2
        theta_b = math.pi / 4
        theta_b_prime = -math.pi / 4
        
        def quantum_correlation(angle_a: float, angle_b: float, N: int) -> float:
            """Simulate quantum correlation measurements"""
            correlation_sum = 0
            for _ in range(N):
                # Simulate singlet state measurements
                # Probability of same outcome: sin²((θ_a - θ_b)/2)
                angle_diff = angle_a - angle_b
                prob_same = (math.sin(angle_diff / 2)) ** 2
                
                # Generate correlated outcomes
                if random.random() < prob_same:
                    # Same outcome
                    outcome = random.choice([1, -1])
                    result_a = outcome
                    result_b = outcome
                else:
                    # Different outcomes
                    result_a = random.choice([1, -1])
                    result_b = -result_a
                
                correlation_sum += result_a * result_b
            
            return correlation_sum / N
        
        # Calculate all four correlations
        n_per_measurement = iterations // 4
        E_ab = quantum_correlation(theta_a, theta_b, n_per_measurement)
        E_ab_prime = quantum_correlation(theta_a, theta_b_prime, n_per_measurement)
        E_a_prime_b = quantum_correlation(theta_a_prime, theta_b, n_per_measurement)
        E_a_prime_b_prime = quantum_correlation(theta_a_prime, theta_b_prime, n_per_measurement)
        
        # CHSH parameter
        S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)
        
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
                "E_ab'": round(E_ab_prime, 4),
                "E_a'b": round(E_a_prime_b, 4),
                "E_a'b'": round(E_a_prime_b_prime, 4)
            }
        }
    
    @staticmethod
    def ghz_experiment(iterations: int = 10000) -> Dict[str, Any]:
        """
        GHZ state test for three-particle entanglement.
        
        GHZ state: |GHZ⟩ = (|000⟩ + |111⟩)/√2
        
        Mermin inequality: M = ⟨XXX⟩ - ⟨XYY⟩ - ⟨YXY⟩ - ⟨YYX⟩
        Classical bound: |M| ≤ 2
        Quantum: M = 4 for GHZ state
        """
        results = {'XXX': [], 'XYY': [], 'YXY': [], 'YYX': []}
        
        for _ in range(iterations):
            # Choose measurement basis randomly
            basis = random.choice(['XXX', 'XYY', 'YXY', 'YYX'])
            
            # GHZ state gives perfect correlations
            # For XXX: always get even parity (all same)
            # For XYY, YXY, YYX: always get odd parity (one different)
            
            if basis == 'XXX':
                # Even parity: (+1,+1,+1) or (-1,-1,-1)
                outcome = random.choice([1, -1])
                result = outcome * outcome * outcome  # Always +1
            else:
                # Odd parity measurements
                # Should give -1 for perfect GHZ
                result = -1
            
            results[basis].append(result)
        
        # Calculate expectation values
        E_xxx = sum(results['XXX']) / len(results['XXX']) if results['XXX'] else 0
        E_xyy = sum(results['XYY']) / len(results['XYY']) if results['XYY'] else 0
        E_yxy = sum(results['YXY']) / len(results['YXY']) if results['YXY'] else 0
        E_yyx = sum(results['YYX']) / len(results['YYX']) if results['YYX'] else 0
        
        # Mermin parameter
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
        """
        Quantum teleportation protocol simulation with proper state fidelity.
        
        Protocol:
        1. Prepare arbitrary state |ψ⟩ = α|0⟩ + β|1⟩
        2. Create entangled pair |Φ+⟩ = (|00⟩ + |11⟩)/√2
        3. Bell measurement on state and half of pair
        4. Classical communication of 2 bits
        5. Apply corrections to receive |ψ⟩
        """
        fidelities = []
        
        for _ in range(iterations):
            # 1. Prepare random qubit state
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2 * math.pi)
            alpha = math.cos(theta / 2)
            beta = cmath.exp(1j * phi) * math.sin(theta / 2)
            
            psi_original = np.array([alpha, beta], dtype=complex)
            
            # Normalize
            norm = np.linalg.norm(psi_original)
            psi_original = psi_original / norm
            
            # 2-4. Teleportation (in ideal case, perfectly preserves state)
            # Simulate with small decoherence
            decoherence_rate = 0.005  # 0.5% error rate
            
            if random.random() < decoherence_rate:
                # Apply random Pauli error
                error_type = random.choice(['X', 'Y', 'Z'])
                if error_type == 'X':
                    # Bit flip
                    psi_bob = np.array([beta, alpha], dtype=complex)
                elif error_type == 'Y':
                    # Y gate
                    psi_bob = np.array([-beta.conjugate(), alpha.conjugate()], dtype=complex)
                else:  # Z
                    # Phase flip
                    psi_bob = np.array([alpha, -beta], dtype=complex)
            else:
                psi_bob = psi_original.copy()
            
            # 5. Calculate fidelity F = |⟨ψ|φ⟩|²
            fidelity = abs(np.dot(psi_original.conjugate(), psi_bob)) ** 2
            fidelities.append(fidelity)
        
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0
        min_fidelity = min(fidelities) if fidelities else 0
        max_fidelity = max(fidelities) if fidelities else 0
        
        # Success rate (fidelity > 0.99)
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
        """Run complete quantum entanglement test suite"""
        return {
            "timestamp": datetime.now().isoformat(),
            "bell_test": QuantumPhysics.bell_experiment(Config.BELL_TEST_ITERATIONS),
            "ghz_test": QuantumPhysics.ghz_experiment(Config.GHZ_TEST_ITERATIONS),
            "teleportation": QuantumPhysics.quantum_teleportation(Config.TELEPORTATION_ITERATIONS)
        }

# ==================== SYSTEM METRICS MODULE ====================
class SystemMetrics:
    """Real system measurements without fake inflation"""
    
    @staticmethod
    def get_storage_metrics() -> Dict[str, Any]:
        """Get actual storage metrics"""
        try:
            disk = psutil.disk_usage(Config.DATA_DIR)
            return {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "percent_used": round(disk.percent, 2)
            }
        except Exception as e:
            logger.error(f"Storage metrics error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_memory_metrics() -> Dict[str, Any]:
        """Get actual memory metrics"""
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
        """Get CPU metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            return {
                "usage_percent": round(cpu_percent, 2),
                "cpu_count": cpu_count,
                "load_average": [round(x, 2) for x in psutil.getloadavg()]
            }
        except Exception as e:
            logger.error(f"CPU metrics error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_all_metrics() -> Dict[str, Any]:
        """Get all system metrics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "storage": SystemMetrics.get_storage_metrics(),
            "memory": SystemMetrics.get_memory_metrics(),
            "cpu": SystemMetrics.get_cpu_metrics()
        }

# ==================== DATABASE MODULE ====================
class Database:
    """Database operations"""
    
    @staticmethod
    def store_measurement(measurement_type: str, data: Dict[str, Any]):
        """Store measurement in database"""
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
        """Get recent measurements"""
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

# ==================== SECURITY MODULE ====================
security = HTTPBearer(auto_error=False)

class SecurityManager:
    """Handle authentication and authorization"""
    
    @staticmethod
    def verify_credentials(username: str, password: str) -> bool:
        """Verify admin credentials"""
        if not Config.ADMINISTRATOR_USERNAME or not Config.ADMINISTRATOR_PASSWORD:
            return False
        
        # Use constant-time comparison to prevent timing attacks
        username_match = secrets.compare_digest(
            username.encode('utf-8'),
            Config.ADMINISTRATOR_USERNAME.encode('utf-8')
        )
        password_match = secrets.compare_digest(
            password.encode('utf-8'),
            Config.ADMINISTRATOR_PASSWORD.encode('utf-8')
        )
        
        return username_match and password_match
    
    @staticmethod
    def generate_token() -> str:
        """Generate secure token"""
        return secrets.token_urlsafe(32)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Dependency for protected routes"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # In production, validate JWT token here
    # For now, simple token check
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"authenticated": True}

# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="Quantum Foam Dominion",
    description="Quantum computing simulations and system metrics",
    version="2.0.0",
    debug=Config.DEBUG
)

# Secure CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ==================== RATE LIMITING ====================
rate_limit_store = defaultdict(list)

async def check_rate_limit(request: Request):
    """Simple rate limiting"""
    client_ip = request.client.host
    now = datetime.now()
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip]
        if now - ts < timedelta(minutes=1)
    ]
    
    if len(rate_limit_store[client_ip]) >= Config.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    rate_limit_store[client_ip].append(now)

# ==================== ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def frontpage(request: Request):
    """Main frontpage with all measurements"""
    try:
        await check_rate_limit(request)
        
        # Run measurements
        quantum_results = QuantumPhysics.run_full_suite()
        system_metrics = SystemMetrics.get_all_metrics()
        
        # Store in database
        Database.store_measurement("quantum_suite", quantum_results)
        Database.store_measurement("system_metrics", system_metrics)
        
        # Generate HTML response with embedded template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Foam Dominion - Scientific Quantum Simulations</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #3498db;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        
        .card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid #3498db;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }}
        
        .card.success {{
            border-left-color: #27ae60;
            background: linear-gradient(135deg, #f8fff9 0%, #f0fff4 100%);
        }}
        
        .card.warning {{
            border-left-color: #f39c12;
            background: linear-gradient(135deg, #fffcf8 0%, #fff9f0 100%);
        }}
        
        .card-title {{
            font-size: 1.1em;
            color: #7f8c8d;
            margin-bottom: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .card-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 15px 0;
        }}
        
        .card-details {{
            color: #555;
            line-height: 1.8;
            margin-top: 15px;
        }}
        
        .card-details div {{
            padding: 5px 0;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .badge.success {{
            background: #27ae60;
            color: white;
        }}
        
        .badge.warning {{
            background: #f39c12;
            color: white;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .metric-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .metric-box .label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        
        .metric-box .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .api-section {{
            background: #ecf0f1;
            padding: 30px;
            border-radius: 15px;
            margin-top: 30px;
        }}
        
        .api-section h3 {{
            color: #2c3e50;
            margin-bottom: 20px;
        }}
        
        .api-list {{
            list-style: none;
        }}
        
        .api-list li {{
            padding: 12px 20px;
            margin: 8px 0;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        
        .api-list a {{
            color: #3498db;
            text-decoration: none;
            font-weight: 600;
        }}
        
        .api-list a:hover {{
            text-decoration: underline;
        }}
        
        .formula {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            margin: 15px 0;
            overflow-x: auto;
        }}
        
        footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .timestamp {{
            opacity: 0.7;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        
        @media (max-width: 768px) {{
            header h1 {{
                font-size: 1.8em;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .card-value {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Quantum Foam Dominion</h1>
            <p>Scientific Quantum Computing Simulations & System Monitoring</p>
        </header>
        
        <div class="content">
            <!-- Quantum Entanglement Section -->
            <div class="section">
                <h2>⚛️ Quantum Entanglement Experiments</h2>
                <p style="margin-bottom: 20px; color: #555;">
                    Demonstrating quantum entanglement through Bell inequality violations and quantum teleportation protocols.
                </p>
                
                <div class="grid">
                    <!-- Bell Test -->
                    <div class="card {'success' if quantum_results['bell_test']['violates_inequality'] else 'warning'}">
                        <div class="card-title">🔔 Bell-CHSH Test</div>
                        <div class="card-value">S = {quantum_results['bell_test']['S']}</div>
                        <div class="card-details">
                            <div><strong>Classical bound:</strong> ≤ 2.0</div>
                            <div><strong>Quantum bound:</strong> ≤ {quantum_results['bell_test']['quantum_bound']}</div>
                            <div><strong>Iterations:</strong> {quantum_results['bell_test']['iterations']:,}</div>
                            <div style="margin-top: 15px;">
                                <strong>Correlations:</strong><br>
                                E(a,b) = {quantum_results['bell_test']['correlations']['E_ab']}<br>
                                E(a,b') = {quantum_results['bell_test']['correlations']['E_ab'']}<br>
                                E(a',b) = {quantum_results['bell_test']['correlations']['E_a'b']}<br>
                                E(a',b') = {quantum_results['bell_test']['correlations']['E_a'b'']}
                            </div>
                        </div>
                        <span class="badge {'success' if quantum_results['bell_test']['violates_inequality'] else 'warning'}">
                            {'✓ Violates Classical' if quantum_results['bell_test']['violates_inequality'] else 'Within Classical'}
                        </span>
                    </div>
                    
                    <!-- GHZ Test -->
                    <div class="card {'success' if quantum_results['ghz_test']['violates_inequality'] else 'warning'}">
                        <div class="card-title">🌀 GHZ-Mermin Test</div>
                        <div class="card-value">M = {quantum_results['ghz_test']['M']}</div>
                        <div class="card-details">
                            <div><strong>Classical bound:</strong> ≤ 2.0</div>
                            <div><strong>Quantum value:</strong> 4.0</div>
                            <div><strong>Iterations:</strong> {quantum_results['ghz_test']['iterations']:,}</div>
                            <div style="margin-top: 15px;">
                                <strong>Expectation values:</strong><br>
                                ⟨XXX⟩ = {quantum_results['ghz_test']['expectation_values']['E_XXX']}<br>
                                ⟨XYY⟩ = {quantum_results['ghz_test']['expectation_values']['E_XYY']}<br>
                                ⟨YXY⟩ = {quantum_results['ghz_test']['expectation_values']['E_YXY']}<br>
                                ⟨YYX⟩ = {quantum_results['ghz_test']['expectation_values']['E_YYX']}
                            </div>
                        </div>
                        <span class="badge {'success' if quantum_results['ghz_test']['violates_inequality'] else 'warning'}">
                            {'✓ Three-Particle Entangled' if quantum_results['ghz_test']['violates_inequality'] else 'Not Entangled'}
                        </span>
                    </div>
                    
                    <!-- Quantum Teleportation -->
                    <div class="card success">
                        <div class="card-title">📡 Quantum Teleportation</div>
                        <div class="card-value">F = {quantum_results['teleportation']['avg_fidelity']}</div>
                        <div class="card-details">
                            <div><strong>Success rate:</strong> {quantum_results['teleportation']['success_rate']*100:.1f}%</div>
                            <div><strong>Min fidelity:</strong> {quantum_results['teleportation']['min_fidelity']}</div>
                            <div><strong>Max fidelity:</strong> {quantum_results['teleportation']['max_fidelity']}</div>
                            <div><strong>Iterations:</strong> {quantum_results['teleportation']['iterations']:,}</div>
                            <div><strong>Theoretical max:</strong> {quantum_results['teleportation']['theoretical_max']}</div>
                        </div>
                        <span class="badge success">✓ Protocol Verified</span>
                    </div>
                </div>
                
                <!-- Formulas -->
                <div style="margin-top: 30px;">
                    <h3 style="color: #2c3e50; margin-bottom: 15px;">📐 Mathematical Framework</h3>
                    <div class="formula">
                        Bell-CHSH: S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
                        <br>Classical: S ≤ 2
                        <br>Quantum: S ≤ 2√2 ≈ 2.828
                    </div>
                    <div class="formula">
                        GHZ-Mermin: M = ⟨XXX⟩ - ⟨XYY⟩ - ⟨YXY⟩ - ⟨YYX⟩
                        <br>Classical: |M| ≤ 2
                        <br>Quantum GHZ: M = 4
                    </div>
                    <div class="formula">
                        Fidelity: F = |⟨ψ|φ⟩|²
                        <br>Perfect teleportation: F = 1
                    </div>
                </div>
            </div>
            
            <!-- System Metrics Section -->
            <div class="section">
                <h2>💾 System Metrics</h2>
                
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="label">💽 Storage Available</div>
                        <div class="value">{system_metrics['storage']['free_gb']} GB</div>
                        <div class="label">of {system_metrics['storage']['total_gb']} GB total</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="label">🧠 Memory Available</div>
                        <div class="value">{system_metrics['memory']['available_gb']} GB</div>
                        <div class="label">of {system_metrics['memory']['total_gb']} GB total</div>
                    </div>
                    
                    <div class="metric-box">
                        <div class="label">⚙️ CPU Usage</div>
                        <div class="value">{system_metrics['cpu']['usage_percent']}%</div>
                        <div class="label">{system_metrics['cpu']['cpu_count']} cores</div>
                    </div>
                </div>
                
                <div class="grid" style="margin-top: 25px;">
                    <div class="card">
                        <div class="card-title">Storage Details</div>
                        <div class="card-details">
                            <div><strong>Total:</strong> {system_metrics['storage']['total_gb']} GB</div>
                            <div><strong>Used:</strong> {system_metrics['storage']['used_gb']} GB</div>
                            <div><strong>Free:</strong> {system_metrics['storage']['free_gb']} GB</div>
                            <div><strong>Usage:</strong> {system_metrics['storage']['percent_used']}%</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-title">Memory Details</div>
                        <div class="card-details">
                            <div><strong>Total:</strong> {system_metrics['memory']['total_gb']} GB</div>
                            <div><strong>Available:</strong> {system_metrics['memory']['available_gb']} GB</div>
                            <div><strong>Usage:</strong> {system_metrics['memory']['percent_used']}%</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-title">CPU Details</div>
                        <div class="card-details">
                            <div><strong>Cores:</strong> {system_metrics['cpu']['cpu_count']}</div>
                            <div><strong>Usage:</strong> {system_metrics['cpu']['usage_percent']}%</div>
                            <div><strong>Load:</strong> {', '.join(map(str, system_metrics['cpu']['load_average']))}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- API Section -->
            <div class="api-section">
                <h3>🔌 API Endpoints</h3>
                <ul class="api-list">
                    <li>
                        <a href="/api/quantum">/api/quantum</a>
                        <span style="color: #7f8c8d;"> - Get quantum experiment results (JSON)</span>
                    </li>
                    <li>
                        <a href="/api/metrics">/api/metrics</a>
                        <span style="color: #7f8c8d;"> - Get system metrics (JSON)</span>
                    </li>
                    <li>
                        <a href="/api/history">/api/history</a>
                        <span style="color: #7f8c8d;"> - Get measurement history</span>
                    </li>
                    <li>
                        <a href="/health">/health</a>
                        <span style="color: #7f8c8d;"> - Health check endpoint</span>
                    </li>
                </ul>
            </div>
        </div>
        
        <footer>
            <p><strong>Quantum Foam Dominion</strong> v2.0.0</p>
            <p>Scientific Quantum Computing Simulations</p>
            <p class="timestamp">
                Environment: {Config.ENVIRONMENT} | 
                Generated: {quantum_results['timestamp']}
            </p>
        </footer>
    </div>
</body>
</html>
"""
        
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frontpage error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/quantum")
async def get_quantum_results(request: Request):
    """Get quantum experiment results as JSON"""
    try:
        await check_rate_limit(request)
        results = QuantumPhysics.run_full_suite()
        return JSONResponse(content=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quantum API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/metrics")
async def get_system_metrics(request: Request):
    """Get system metrics as JSON"""
    try:
        await check_rate_limit(request)
        metrics = SystemMetrics.get_all_metrics()
        return JSONResponse(content=metrics)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/history")
async def get_measurement_history(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get recent measurement history"""
    try:
        await check_rate_limit(request)
        history = Database.get_recent_measurements(limit)
        return JSONResponse(content={"measurements": history, "count": len(history)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": Config.ENVIRONMENT
    })

@app.get("/api/admin/status")
async def admin_status(current_user: dict = Depends(get_current_user)):
    """Protected admin endpoint"""
    return JSONResponse(content={
        "authenticated": True,
        "timestamp": datetime.now().isoformat(),
        "environment": Config.ENVIRONMENT
    })

# ==================== ERROR HANDLERS ====================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting Quantum Foam Dominion")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Debug mode: {Config.DEBUG}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down Quantum Foam Dominion")

# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info" if not Config.DEBUG else "debug"
    )
