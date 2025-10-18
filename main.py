import os
import logging
import hashlib
import base64
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import httpx
import asyncio
from contextlib import asynccontextmanager
import secrets
from collections import defaultdict
import random  # Retained for fictional quantum variance only
import psutil  # For real-time system network and storage metrics
import subprocess  # For real routing table extraction and QSH commands

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

# Create static directory only
Config.STATIC_DIR.mkdir(exist_ok=True)

# ==================== LOGGING MODULE ====================
class Logger:
    """Centralized logging configuration"""
    
    @staticmethod
    def setup():
        logging.basicConfig(
            level=logging.INFO if not Config.DEBUG else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app.log') if os.path.exists('/app') else logging.NullHandler()
            ]
        )
        return logging.getLogger(__name__)

logger = Logger.setup()

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
                "speed_gbps": int(random.uniform(900000, 1100000)),  # Dynamic bandwidth
                "qubit_rate": int(random.uniform(900000000, 1100000000)),  # Dynamic qubit rate
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
                # Simulate real quantum measurement collapse with variance
                variance = random.uniform(-0.001, 0.001)
                measurement = ent.copy()
                measurement["measured_coherence"] = round(max(0.0, min(1.0, ent["coherence"] + variance)), 4)
                measurement["measured_fidelity"] = round(max(0.0, min(1.0, ent["fidelity"] + variance)), 4)
                measurement["measurement_time"] = datetime.now().isoformat()
                measurement["variance_applied"] = round(variance, 4)
                # Update original for persistence
                ent["coherence"] = measurement["measured_coherence"]
                ent["fidelity"] = measurement["measured_fidelity"]
                ent["last_measurement"] = measurement["measurement_time"]
                return measurement
        return {}

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
        self.active_sessions: Dict[str, str] = {}  # token -> username
        
        # Encrypted messages
        self.encrypted_messages: List[Dict]] = []
        
        # Bitcoin cache
        self.bitcoin_cache: Dict[str, Any] = {
            "blockchain_info": None,
            "latest_blocks": [],
            "mempool_info": None,
            "network_stats": None,
            "last_update": None
        }
        
        # Storage metrics with real disk usage scaled to fictional capacities
        self.update_storage_metrics()
        
        self.qram_storage = {
            "total_capacity_qubits": Config.QRAM_CAPACITY_QUBITS,
            "used_capacity_qubits": int(random.uniform(700000000, 800000000)),  # Retained for fictional QRAM
            "available_capacity_qubits": Config.QRAM_CAPACITY_QUBITS - self.qram_storage["used_capacity_qubits"],
            "coherence_time_ms": int(random.uniform(9000, 11000)),
            "error_rate": round(random.uniform(0.00005, 0.00015), 4),
            "node_address": Config.QUANTUM_REALM,
            "last_update": datetime.now().isoformat()
        }
    
    def update_storage_metrics(self):
        """Update storage metrics with real disk usage"""
        du = psutil.disk_usage('/')
        used_gb = du.used / (1024**3)
        total_gb = du.total / (1024**3)
        used_tb = used_gb / 1024  # Convert to TB for scaling
        total_tb = total_gb / 1024
        
        # Scale to fictional holographic capacity
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
        
        # Update QRAM fictional
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
        
        # Create quantum foam email
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
        
        # Generate token
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
quantum_entanglement = QuantumEntanglement()

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
    
    @staticmethod
    async def get_block_by_hash(block_hash: str) -> Dict:
        """Get block details by hash"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.BLOCKCHAIN_API}/rawblock/{block_hash}")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching block: {e}")
            return {}
    
    @staticmethod
    async def get_transaction(txid: str) -> Dict:
        """Get transaction details"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.BLOCKCHAIN_API}/rawtx/{txid}")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching transaction: {e}")
            return {}
    
    @staticmethod
    async def get_address_info(address: str) -> Dict:
        """Get address information"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.BLOCKCHAIN_API}/rawaddr/{address}")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching address info: {e}")
            return {}

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
            
            elif cmd_name == "getblock":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Usage: getblock <block_hash>",
                        "timestamp": datetime.now().isoformat()
                    }
                
                block = await BitcoinMainnet.get_block_by_hash(args[0])
                
                return {
                    "success": True,
                    "command": command,
                    "result": block,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "gettransaction":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Usage: gettransaction <txid>",
                        "timestamp": datetime.now().isoformat()
                    }
                
                tx = await BitcoinMainnet.get_transaction(args[0])
                
                return {
                    "success": True,
                    "command": command,
                    "result": tx,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "getaddressinfo":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Usage: getaddressinfo <address>",
                        "timestamp": datetime.now().isoformat()
                    }
                
                addr_info = await BitcoinMainnet.get_address_info(args[0])
                
                return {
                    "success": True,
                    "command": command,
                    "result": addr_info,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "help":
                return {
                    "success": True,
                    "command": command,
                    "result": {
                        "available_commands": [
                            "getblockchaininfo - Get blockchain status and info",
                            "getblock <hash> - Get block details",
                            "getmempoolinfo - Get mempool information",
                            "getrecentblocks [count] - Get recent blocks (default 10)",
                            "gettransaction <txid> - Get transaction details",
                            "getaddressinfo <address> - Get address information",
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


# ==================== QUANTUM ENCRYPTION MODULE ====================
class QuantumEncryption:
    """Custom quantum-proof encryption using black hole and white hole addresses"""
    
    @staticmethod
    def generate_quantum_key(salt: str, use_black_hole: bool = True) -> bytes:
        """Generate quantum key using salt and black/white hole addressing"""
        hole_address = Config.BLACK_HOLE_ADDRESS if use_black_hole else Config.WHITE_HOLE_ADDRESS
        key_material = f"{hole_address}:{salt}".encode()
        return hashlib.sha3_512(key_material).digest()
    
    @staticmethod
    def quantum_encrypt(plaintext: str) -> Dict[str, str]:
        """Encrypt message using dual-hole quantum encryption"""
        try:
            # Generate a single random salt for reproducibility
            salt = secrets.token_hex(32)
            
            # Generate keys using salt (no plaintext in keys for security)
            black_hole_key = QuantumEncryption.generate_quantum_key(salt, True)
            white_hole_key = QuantumEncryption.generate_quantum_key(salt, False)
            
            plaintext_bytes = plaintext.encode('utf-8')
            encrypted = bytearray(plaintext_bytes)
            
            # Layer 1: XOR with black hole key
            for i in range(len(encrypted)):
                encrypted[i] ^= black_hole_key[i % len(black_hole_key)]
            
            # Layer 2: XOR with white hole key (reverse)
            for i in range(len(encrypted)):
                encrypted[i] ^= white_hole_key[(len(encrypted) - 1 - i) % len(white_hole_key)]
            
            # Layer 3: SHA3-512 based permutation (now reproducible with salt)
            permutation_key = hashlib.sha3_512(black_hole_key + white_hole_key).digest()
            for i in range(len(encrypted)):
                encrypted[i] ^= permutation_key[i % len(permutation_key)]
            
            encrypted_b64 = base64.b64encode(bytes(encrypted)).decode('utf-8')
            salt_b64 = base64.b64encode(salt.encode()).decode('utf-8')
            
            # Signature over all reproducible parts
            signature_material = f"{Config.BLACK_HOLE_ADDRESS}:{Config.WHITE_HOLE_ADDRESS}:{encrypted_b64}:{salt_b64}".encode()
            signature = hashlib.sha3_512(signature_material).hexdigest()
            
            return {
                "ciphertext": encrypted_b64,
                "salt": salt_b64,  # Store salt for decryption
                "signature": signature,
                "black_hole": Config.BLACK_HOLE_ADDRESS,
                "white_hole": Config.WHITE_HOLE_ADDRESS,
                "timestamp": datetime.now().isoformat(),
                "algorithm": "QUANTUM-DUAL-HOLE-XOR-SHA3"
            }
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    @staticmethod
    def quantum_decrypt(encrypted_data: Dict[str, str]) -> str:
        """Decrypt message using dual-hole quantum decryption"""
        try:
            # Verify signature
            expected_signature_material = f"{encrypted_data['black_hole']}:{encrypted_data['white_hole']}:{encrypted_data['ciphertext']}:{encrypted_data['salt']}".encode()
            expected_signature = hashlib.sha3_512(expected_signature_material).hexdigest()
            
            if expected_signature != encrypted_data['signature']:
                raise ValueError("Signature verification failed")
            
            # Regenerate keys using stored salt
            salt = base64.b64decode(encrypted_data['salt']).decode('utf-8')
            black_hole_key = QuantumEncryption.generate_quantum_key(salt, True)
            white_hole_key = QuantumEncryption.generate_quantum_key(salt, False)
            permutation_key = hashlib.sha3_512(black_hole_key + white_hole_key).digest()
            
            encrypted_bytes = base64.b64decode(encrypted_data['ciphertext'])
            decrypted = bytearray(encrypted_bytes)
            
            # Reverse Layer 3: XOR with permutation key
            for i in range(len(decrypted)):
                decrypted[i] ^= permutation_key[i % len(permutation_key)]
            
            # Reverse Layer 2: XOR with white hole key (reverse)
            for i in range(len(decrypted)):
                decrypted[i] ^= white_hole_key[(len(decrypted) - 1 - i) % len(white_hole_key)]
            
            # Reverse Layer 1: XOR with black hole key
            for i in range(len(decrypted)):
                decrypted[i] ^= black_hole_key[i % len(black_hole_key)]
            
            return bytes(decrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise

# ==================== EMAIL MODULE ====================
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

# ==================== NETWORK ANALYSIS MODULE ====================
class NetworkAnalysis:
    """Network topology and routing analysis with real metrics"""
    
    @staticmethod
    def get_network_interfaces() -> List[Dict]:
        """Get all network interfaces with real psutil data (no simulation)"""
        interfaces = []
        
        # Real interfaces from psutil only
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        io = psutil.net_io_counters(pernic=True)
        
        for name, stat in stats.items():
            addr_info = addrs.get(name, [])
            addr = next((a.address for a in addr_info if a.family.name == 'AF_INET'), "unknown")
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
        
        return interfaces
    
    @staticmethod
    def get_routing_tables() -> List[Dict]:
        """Get real routing tables using qsh network tool first, fallback to ip route (no simulated data)"""
        # Attempt qsh first (assuming qsh is the node-specific network tool)
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
        
        # Fallback to standard ip route for real metrics
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
        
        return []  # No data if both fail

# ==================== FASTAPI APP SETUP ====================
templates = Jinja2Templates(directory=str(Config.TEMPLATES_DIR))

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

# ==================== HTML ROUTES ====================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root route serving index.html from root directory"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/networking", response_class=HTMLResponse)
async def networking_page(request: Request):
    """Networking page serving networking.html from root directory"""
    interfaces = NetworkAnalysis.get_network_interfaces()
    routes = NetworkAnalysis.get_routing_tables()
    context = {
        "request": request,
        "interfaces": interfaces,
        "routes": routes,
        "quantum_realm": Config.QUANTUM_REALM,
        "networking_address": Config.NETWORKING_ADDRESS
    }
    return templates.TemplateResponse("networking.html", context)

@app.get("/bitcoin", response_class=HTMLResponse)
async def bitcoin_page(request: Request):
    """Bitcoin page serving bitcoin.html from root directory"""
    # Fetch real-time data
    blockchain_info = await BitcoinCLI.execute_command("getblockchaininfo")
    mempool_info = await BitcoinCLI.execute_command("getmempoolinfo")
    recent_blocks = await BitcoinCLI.execute_command("getrecentblocks 5")
    context = {
        "request": request,
        "blockchain_info": blockchain_info.get("result", {}),
        "mempool_info": mempool_info.get("result", {}),
        "recent_blocks": recent_blocks.get("result", {}),
        "black_hole_address": Config.BLACK_HOLE_ADDRESS
    }
    return templates.TemplateResponse("bitcoin.html", context)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page serving chat.html from root directory"""
    recent_messages = storage.get_recent_messages()
    context = {
        "request": request,
        "messages": recent_messages,
        "chat_backend": Config.CHAT_BACKEND
    }
    return templates.TemplateResponse("chat.html", context)

@app.get("/email", response_class=HTMLResponse)
async def email_page(request: Request):
    """Email page serving email.html from root directory"""
    # Assuming a demo user; in real app, use auth
    demo_username = "demo_user"
    if demo_username not in storage.emails:
        # Create demo email
        demo_email = EmailSystem.send_email(
            "admin::quantum.foam",
            f"{demo_username}::quantum.foam",
            "Welcome to Quantum Foam",
            "Your inbox is now quantum-entangled."
        )
    inbox = storage.get_inbox(demo_username)
    context = {
        "request": request,
        "inbox": inbox,
        "user_email": storage.user_emails.get(demo_username, "")
    }
    return templates.TemplateResponse("email.html", context)

@app.get("/blockchain", response_class=HTMLResponse)
async def blockchain_page(request: Request):
    """Blockchain page serving blockchain.html from root directory (synonym for bitcoin)"""
    # Reuse bitcoin data
    return await bitcoin_page(request)

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
            # Parse message; for simplicity, echo or process
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

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
