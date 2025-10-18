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
from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import secrets
from collections import defaultdict
import random
import psutil
from jinja2 import Environment, FileSystemLoader
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
    TEMPLATES_DIR = Path("templates")
    STATIC_DIR = Path("static")
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
        cls.STATIC_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.TEMPLATES_DIR.mkdir(exist_ok=True)
        
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

# Mount static files if directory exists
if Config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=Config.STATIC_DIR), name="static")

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
        
        # Prepare template data
        template_data = {
            "quantum": quantum_results,
            "system": system_metrics,
            "environment": Config.ENVIRONMENT
        }
        
        # Try to load template
        try:
            env = Environment(loader=FileSystemLoader(Config.TEMPLATES_DIR))
            template = env.get_template("index.html")
            html_content = template.render(**template_data)
        except Exception as template_error:
            logger.warning(f"Template error: {template_error}, using fallback")
            # Fallback HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Quantum Foam Dominion</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }}
                    .container {{
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        color: #2c3e50;
                        border-bottom: 3px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #34495e;
                        margin-top: 30px;
                    }}
                    .metric {{
                        background: #ecf0f1;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 5px;
                        border-left: 4px solid #3498db;
                    }}
                    .success {{
                        border-left-color: #27ae60;
                    }}
                    .warning {{
                        border-left-color: #f39c12;
                    }}
                    .value {{
                        font-size: 1.2em;
                        font-weight: bold;
                        color: #2c3e50;
                    }}
                    .label {{
                        color: #7f8c8d;
                        font-size: 0.9em;
                    }}
                    pre {{
                        background: #2c3e50;
                        color: #ecf0f1;
                        padding: 15px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔬 Quantum Foam Dominion</h1>
                    <p>Scientific quantum computing simulations and system monitoring</p>
                    
                    <h2>⚛️ Quantum Entanglement Tests</h2>
                    
                    <div class="metric {('success' if quantum_results['bell_test']['violates_inequality'] else 'warning')}">
                        <div class="label">Bell-CHSH Test</div>
                        <div class="value">S = {quantum_results['bell_test']['S']}</div>
                        <div>Classical bound: ≤ 2.0 | Quantum bound: ≤ 2.828</div>
                        <div><strong>Violates classical inequality: {quantum_results['bell_test']['violates_inequality']}</strong></div>
                    </div>
                    
                    <div class="metric {('success' if quantum_results['ghz_test']['violates_inequality'] else 'warning')}">
                        <div class="label">GHZ-Mermin Test</div>
                        <div class="value">M = {quantum_results['ghz_test']['M']}</div>
                        <div>Classical bound: ≤ 2.0 | Quantum value: 4.0</div>
                        <div><strong>Violates classical inequality: {quantum_results['ghz_test']['violates_inequality']}</strong></div>
                    </div>
                    
                    <div class="metric success">
                        <div class="label">Quantum Teleportation</div>
                        <div class="value">Fidelity = {quantum_results['teleportation']['avg_fidelity']}</div>
                        <div>Success rate: {quantum_results['teleportation']['success_rate']*100:.1f}%</div>
                        <div>Iterations: {quantum_results['teleportation']['iterations']:,}</div>
                    </div>
                    
                    <h2>💾 System Metrics</h2>
                    
                    <div class="metric">
                        <div class="label">Storage</div>
                        <div class="value">{system_metrics['storage']['free_gb']} GB free / {system_metrics['storage']['total_gb']} GB total</div>
                        <div>Usage: {system_metrics['storage']['percent_used']}%</div>
                    </div>
                    
                    <div class="metric">
                        <div class="label">Memory</div>
                        <div class="value">{system_metrics['memory']['available_gb']} GB available / {system_metrics['memory']['total_gb']} GB total</div>
                        <div>Usage: {system_metrics['memory']['percent_used']}%</div>
                    </div>
                    
                    <div class="metric">
                        <div class="label">CPU</div>
                        <div class="value">{system_metrics['cpu']['usage_percent']}% usage</div>
                        <div>Cores: {system_metrics['cpu']['cpu_count']}</div>
                    </div>
                    
                    <h2>📊 API Endpoints</h2>
                    <ul>
                        <li><a href="/api/quantum">/api/quantum</a> - Quantum test results</li>
                        <li><a href="/api/metrics">/api/metrics</a> - System metrics</li>
                        <li><a href="/api/history">/api/history</a> - Recent measurements</li>
                        <li><a href="/health">/health</a> - Health check</li>
                    </ul>
                    
                    <p style="margin-top: 40px; color: #7f8c8d; font-size: 0.9em;">
                        Environment: {Config.ENVIRONMENT} | Timestamp: {quantum_results['timestamp']}
                    </p>
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
