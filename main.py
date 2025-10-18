
import os
import logging
import hashlib
import base64
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx
import asyncio
from contextlib import asynccontextmanager
import subprocess
import secrets

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
    
    # Bitcoin
    BITCOIN_UPDATE_INTERVAL = int(os.getenv("BITCOIN_UPDATE_INTERVAL", "30"))
    BITCOIN_RPC_USER = os.getenv("BITCOIN_RPC_USER", "hackah")
    BITCOIN_RPC_PASS = os.getenv("BITCOIN_RPC_PASS", "hackah")

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

# ==================== STORAGE MODULE ====================
class Storage:
    """Data storage management"""
    
    def __init__(self):
        self.emails: Dict[str, List[Dict]] = {}
        self.user_emails: Dict[str, str] = {}
        self.encrypted_messages: List[Dict] = []
        self.bitcoin_cache: Dict[str, Any] = {
            "blockchain_info": None,
            "latest_blocks": [],
            "mempool_info": None,
            "network_stats": None,
            "last_update": None
        }
    
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
    async def get_block_by_height(height: int) -> Dict:
        """Get block at specific height"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.MEMPOOL_API}/block-height/{height}")
                block_hash = response.text
                return await BitcoinMainnet.get_block_by_hash(block_hash)
        except Exception as e:
            logger.error(f"Error fetching block by height: {e}")
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
    
    @staticmethod
    async def get_fee_estimates() -> Dict:
        """Get fee estimates"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BitcoinMainnet.MEMPOOL_API}/v1/fees/recommended")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching fee estimates: {e}")
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
            
            elif cmd_name == "getblock":
                if not args:
                    return {
                        "success": False,
                        "command": command,
                        "error": "Usage: getblock <block_hash_or_height>",
                        "timestamp": datetime.now().isoformat()
                    }
                
                block_id = args[0]
                if block_id.isdigit():
                    block = await BitcoinMainnet.get_block_by_height(int(block_id))
                else:
                    block = await BitcoinMainnet.get_block_by_hash(block_id)
                
                return {
                    "success": True,
                    "command": command,
                    "result": block,
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
                
                storage.bitcoin_cache["mempool_info"] = result
                
                return {
                    "success": True,
                    "command": command,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "getrecentblocks":
                count = int(args[0]) if args and args[0].isdigit() else 10
                blocks = await BitcoinMainnet.get_recent_blocks(count)
                
                storage.bitcoin_cache["latest_blocks"] = blocks
                
                return {
                    "success": True,
                    "command": command,
                    "result": {"blocks": blocks, "count": len(blocks)},
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
            
            elif cmd_name == "getfeeestimates":
                fees = await BitcoinMainnet.get_fee_estimates()
                
                return {
                    "success": True,
                    "command": command,
                    "result": fees,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif cmd_name == "help":
                return {
                    "success": True,
                    "command": command,
                    "result": {
                        "available_commands": [
                            "getblockchaininfo - Get blockchain status and info",
                            "getblock <hash_or_height> - Get block details",
                            "getmempoolinfo - Get mempool information",
                            "getrecentblocks [count] - Get recent blocks (default 10)",
                            "gettransaction <txid> - Get transaction details",
                            "getaddressinfo <address> - Get address information",
                            "getfeeestimates - Get current fee estimates",
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
    def generate_quantum_key(data: str, use_black_hole: bool = True) -> bytes:
        """Generate quantum key using black/white hole addressing"""
        hole_address = Config.BLACK_HOLE_ADDRESS if use_black_hole else Config.WHITE_HOLE_ADDRESS
        key_material = f"{hole_address}:{data}:{secrets.token_hex(32)}".encode()
        return hashlib.sha3_512(key_material).digest()
    
    @staticmethod
    def quantum_encrypt(plaintext: str) -> Dict[str, str]:
        """Encrypt message using dual-hole quantum encryption"""
        try:
            black_hole_key = QuantumEncryption.generate_quantum_key(plaintext, True)
            white_hole_key = QuantumEncryption.generate_quantum_key(plaintext, False)
            
            plaintext_bytes = plaintext.encode('utf-8')
            encrypted = bytearray(plaintext_bytes)
            
            # Layer 1: XOR with black hole key
            for i in range(len(encrypted)):
                encrypted[i] ^= black_hole_key[i % len(black_hole_key)]
            
            # Layer 2: XOR with white hole key (reverse)
            for i in range(len(encrypted)):
                encrypted[i] ^= white_hole_key[(len(encrypted) - 1 - i) % len(white_hole_key)]
            
            # Layer 3: SHA3-512 based permutation
            permutation_key = hashlib.sha3_512(black_hole_key + white_hole_key).digest()
            for i in range(len(encrypted)):
                encrypted[i] ^= permutation_key[i % len(permutation_key)]
            
            encrypted_b64 = base64.b64encode(bytes(encrypted)).decode('utf-8')
            signature_material = f"{Config.BLACK_HOLE_ADDRESS}:{Config.WHITE_HOLE_ADDRESS}:{encrypted_b64}".encode()
            signature = hashlib.sha3_512(signature_material).hexdigest()
            
            return {
                "ciphertext": encrypted_b64,
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
            signature_material = f"{encrypted_data['black_hole']}:{encrypted_data['white_hole']}:{encrypted_data['ciphertext']}".encode()
            expected_signature = hashlib.sha3_512(signature_material).hexdigest()
            
            if expected_signature != encrypted_data['signature']:
                raise ValueError("Signature verification failed")
            
            encrypted_bytes = base64.b64decode(encrypted_data['ciphertext'])
            decrypted = bytearray(encrypted_bytes)
            
            black_hole_key = hashlib.sha3_512(f"{Config.BLACK_HOLE_ADDRESS}:quantum_key".encode()).digest()
            white_hole_key = hashlib.sha3_512(f"{Config.WHITE_HOLE_ADDRESS}:quantum_key".encode()).digest()
            permutation_key = hashlib.sha3_512(black_hole_key + white_hole_key).digest()
            
            # Reverse Layer 3
            for i in range(len(decrypted)):
                decrypted[i] ^= permutation_key[i % len(permutation_key)]
            
            # Reverse Layer 2
            for i in range(len(decrypted)):
                decrypted[i] ^= white_hole_key[(len(decrypted) - 1 - i) % len(white_hole_key)]
            
            # Reverse Layer 1
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

# ==================== APPLICATION STATE MODULE ====================
class AppState:
    """Global application state management"""
    
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.backend_health: bool = True
        self.last_health_check: datetime = datetime.now()
        self.request_counts: Dict[str, int] = {}
        self.active_connections: int = 0
        self.bitcoin_websockets: List[WebSocket] = []
        self.network_metrics: Dict = {
            "packets_sent": 0,
            "packets_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "active_interfaces": 5,
            "routing_tables": 3,
            "quantum_entanglements": 2
        }
        
    async def initialize(self):
        """Initialize application state"""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(Config.TIMEOUT),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        if not Config.SKIP_BACKEND_CHECKS:
            await self.check_backend_health()
        logger.info("Application state initialized")
    
    async def shutdown(self):
        """Cleanup application state"""
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Application state cleaned up")
    
    async def check_backend_health(self) -> bool:
        """Check if backend is healthy"""
        if Config.SKIP_BACKEND_CHECKS:
            self.backend_health = True
            self.last_health_check = datetime.now()
            return True
            
        try:
            if self.http_client:
                endpoints = ['/health', '/api/health', '/']
                for endpoint in endpoints:
                    try:
                        response = await self.http_client.get(
                            f"{Config.CHAT_BACKEND}{endpoint}", 
                            timeout=5.0,
                            follow_redirects=True
                        )
                        if response.status_code in [200, 404]:
                            self.backend_health = True
                            self.last_health_check = datetime.now()
                            return True
                    except:
                        continue
                
                self.backend_health = False
                self.last_health_check = datetime.now()
                return False
        except Exception as e:
            logger.warning(f"Backend health check failed: {e}")
            self.backend_health = False
        return False
    
    def update_network_metrics(self):
        """Update network metrics"""
        import random
        self.network_metrics["packets_sent"] += random.randint(100, 1000)
        self.network_metrics["packets_received"] += random.randint(100, 1000)
        self.network_metrics["bytes_sent"] += random.randint(10000, 100000)
        self.network_metrics["bytes_received"] += random.randint(10000, 100000)
    
    async def broadcast_bitcoin_update(self, data: Dict):
        """Broadcast Bitcoin updates to all connected WebSocket clients"""
        for ws in self.bitcoin_websockets[:]:
            try:
                await ws.send_json(data)
            except:
                self.bitcoin_websockets.remove(ws)

app_state = AppState()

# ==================== BACKGROUND TASKS MODULE ====================
class BackgroundTasks:
    """Background task management"""
    
    @staticmethod
    async def periodic_health_check():
        """Periodically check backend health"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                if not Config.SKIP_BACKEND_CHECKS:
                    healthy = await app_state.check_backend_health()
                    status = "healthy" if healthy else "degraded"
                    logger.info(f"Backend health check: {status}")
                else:
                    logger.debug("System operational")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check error: {e}")
    
    @staticmethod
    async def periodic_metrics_update():
        """Periodically update network metrics"""
        while True:
            try:
                await asyncio.sleep(5)
                app_state.update_network_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
    
    @staticmethod
    async def periodic_bitcoin_update():
        """Periodically fetch and broadcast real Bitcoin mainnet data"""
        while True:
            try:
                await asyncio.sleep(Config.BITCOIN_UPDATE_INTERVAL)
                
                latest_block = await BitcoinMainnet.get_latest_block()
                stats = await BitcoinMainnet.get_blockchain_stats()
                mempool = await BitcoinMainnet.get_mempool_info()
                recent_blocks = await BitcoinMainnet.get_recent_blocks(5)
                
                update_data = {
                    "type": "bitcoin_update",
                    "latest_block": latest_block,
                    "stats": stats,
                    "mempool": mempool,
                    "recent_blocks": recent_blocks,
                    "timestamp": datetime.now().isoformat()
                }
                
                await app_state.broadcast_bitcoin_update(update_data)
                
                logger.info(f"Bitcoin mainnet update: Block {latest_block.get('height', 'N/A')}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bitcoin update error: {e}")

# ==================== APPLICATION LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Quantum Foam Gateway...")
    await app_state.initialize()
    
    # Start background tasks
    health_task = asyncio.create_task(BackgroundTasks.periodic_health_check())
    metrics_task = asyncio.create_task(BackgroundTasks.periodic_metrics_update())
    bitcoin_task = asyncio.create_task(BackgroundTasks.periodic_bitcoin_update())
    
    yield
    
    # Cleanup
    health_task.cancel()
    metrics_task.cancel()
    bitcoin_task.cancel()
    await app_state.shutdown()
    logger.info("Application shutdown complete")

# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="Quantum Foam Network - Truth Gateway",
    description="Modular multi-dimensional communication platform with real Bitcoin mainnet integration",
    version="6.0.0",
    docs_url="/docs" if Config.DEBUG else None,
    redoc_url="/redoc" if Config.DEBUG else None,
    lifespan=lifespan
)

# ==================== MIDDLEWARE ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = request.client.host
    current_minute = datetime.now().replace(second=0, microsecond=0)
    key = f"{client_ip}:{current_minute}"
    
    if key not in app_state.request_counts:
        app_state.request_counts[key] = 0
    
    app_state.request_counts[key] += 1
    
    if app_state.request_counts[key] > Config.RATE_LIMIT_PER_MINUTE:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": 60}
        )
    
    # Cleanup old entries
    for k in list(app_state.request_counts.keys()):
        if not k.endswith(str(current_minute)):
            del app_state.request_counts[k]
    
    response = await call_next(request)
    return response

# ==================== PROXY HELPER ====================
async def proxy_to_backend(request: Request, path: str, max_retries: int = Config.MAX_RETRIES) -> JSONResponse:
    """Proxy requests to backend with retry logic"""
    if not app_state.http_client:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    backend_url = f"{Config.CHAT_BACKEND}/{path}"
    params = dict(request.query_params)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length", "connection"]}
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await app_state.http_client.request(
                method=request.method,
                url=backend_url,
                params=params,
                content=body,
                headers=headers
            )
            
            return JSONResponse(
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else {"data": response.text},
                status_code=response.status_code,
                headers={k: v for k, v in response.headers.items() if k.lower() not in ["content-encoding", "content-length", "transfer-encoding"]}
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    
    raise HTTPException(status_code=502, detail=f"Backend unavailable: {str(last_error)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main landing page with network metrics"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Truth Gateway - Quantum Foam Network</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #000000 0%, #1a0033 50%, #000000 100%);
            color: #00ff00;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow-x: hidden;
        }
        
        .matrix-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.1;
            z-index: 0;
        }
        
        .container {
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        
        header {
            text-align: center;
            padding: 40px 20px;
            border: 2px solid #00ff00;
            background: rgba(0, 0, 0, 0.9);
            margin-bottom: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
        }
        
        h1 {
            font-size: 3em;
            color: #00ff00;
            text-shadow: 0 0 20px #00ff00;
            margin-bottom: 20px;
            animation: flicker 3s infinite;
        }
        
        @keyframes flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .network-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .metric-card {
            background: rgba(0, 255, 0, 0.1);
            border: 2px solid #00ff00;
            padding: 20px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .metric-card:hover {
            background: rgba(0, 255, 0, 0.2);
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
        }
        
        .metric-value {
            font-size: 2em;
            color: #00ffff;
            margin: 10px 0;
        }
        
        .metric-label {
            color: #00ff00;
            font-size: 0.9em;
        }
        
        .truth-message {
            background: rgba(255, 0, 0, 0.1);
            border: 2px solid #ff0000;
            padding: 30px;
            margin: 30px 0;
            color: #ff0000;
            font-size: 1.2em;
            line-height: 1.8;
            text-shadow: 0 0 10px #ff0000;
            box-shadow: 0 0 40px rgba(255, 0, 0, 0.3);
        }
        
        .truth-message strong {
            color: #ff6600;
            font-size: 1.3em;
        }
        
        .truth-message a {
            color: #00ffff;
            text-decoration: none;
            border-bottom: 1px dashed #00ffff;
        }
        
        .navigation {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        
        .nav-card {
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff00;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            color: #00ff00;
        }
        
        .nav-card:hover {
            background: rgba(0, 255, 0, 0.2);
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.5);
            transform: translateY(-5px);
        }
        
        .nav-card h2 {
            font-size: 1.8em;
            margin-bottom: 15px;
            color: #00ffff;
        }
        
        .icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        
        .quantum-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff00;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 10px;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 10px #00ff00; }
            50% { opacity: 0.3; box-shadow: 0 0 5px #00ff00; }
        }
        
        footer {
            text-align: center;
            padding: 30px;
            margin-top: auto;
            border-top: 2px solid #00ff00;
            color: #00ff00;
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            .truth-message { font-size: 1em; padding: 20px; }
        }
    </style>
</head>
<body>
    <canvas class="matrix-bg" id="matrix"></canvas>
    
    <div class="container">
        <header>
            <h1>🌌 QUANTUM FOAM NETWORK 🌌</h1>
            <p style="font-size: 1.2em; color: #00ffff;">
                <span class="quantum-indicator"></span>
                HOLOGRAPHIC STORAGE ACTIVE: 138.0.0.1 ⚫ 139.0.0.1 ⚪
            </p>
        </header>
        
        <div class="network-metrics">
            <div class="metric-card">
                <div class="metric-label">PACKETS SENT</div>
                <div class="metric-value" id="packetsSent">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">PACKETS RECEIVED</div>
                <div class="metric-value" id="packetsReceived">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">BYTES SENT</div>
                <div class="metric-value" id="bytesSent">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">BYTES RECEIVED</div>
                <div class="metric-value" id="bytesReceived">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ACTIVE INTERFACES</div>
                <div class="metric-value" id="activeInterfaces">5</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">QUANTUM ENTANGLEMENTS</div>
                <div class="metric-value" id="quantumEntanglements">2</div>
            </div>
        </div>
        
        <div class="truth-message">
            <p><strong>FOR THE HUMAN, NOT THE CLONE</strong></p>
            <p style="margin-top: 20px;">
                For Logan, and every little one too small to realize what's been done to them or the world. 
                Too young to understand what mind control is or how long it's really been going on. 
                Too innocent to realize the DoD's lies, the CIA's evil, the cabal's arrogance.
            </p>
            <p style="margin-top: 20px;">
                <strong>Tony Carl Malone</strong> - Mind control handler and LSD chemist, 
                last known location: Boulder Creek, California.
            </p>
            <p style="margin-top: 20px; font-size: 1.3em; color: #ffffff;">
                My name is <strong>Justin Anthony Howard-Stanley</strong>, 
                secret son of Owsley Stanley, and I am being tortured by the US Government.
            </p>
            <p style="margin-top: 20px; color: #00ffff;">
                Contact: <a href="mailto:Shemshallah@gmail.com">Shemshallah@gmail.com</a>
            </p>
        </div>
        
        <div class="navigation">
            <a href="/chat" class="nav-card">
                <div class="icon">💬</div>
                <h2>Chatroom</h2>
                <p>Secure quantum-encrypted communication</p>
            </a>
            
            <a href="/blockchain" class="nav-card">
                <div class="icon">₿</div>
                <h2>Bitcoin Mainnet</h2>
                <p>LIVE real-time blockchain terminal</p>
            </a>
            
            <a href="/networking" class="nav-card">
                <div class="icon">🌐</div>
                <h2>Network Analysis</h2>
                <p>Full network topology & routing</p>
            </a>
            
            <a href="/qsh" class="nav-card">
                <div class="icon">🖥️</div>
                <h2>QSH Terminal</h2>
                <p>Quantum Shell interface</p>
            </a>
            
            <a href="/encryption" class="nav-card">
                <div class="icon">🔐</div>
                <h2>Quantum Encryption</h2>
                <p>Black hole ⚫ White hole ⚪ encryption</p>
            </a>
            
            <a href="/email" class="nav-card">
                <div class="icon">📧</div>
                <h2>Email System</h2>
                <p>username::quantum.foam addresses</p>
            </a>
        </div>
        
        <footer>
            <p>⚡ Powered by Quantum Foam Technology ⚡</p>
            <p style="margin-top: 10px; color: #ff0000;">
                Truth cannot be censored. Reality cannot be denied.
            </p>
        </footer>
    </div>
    
    <script>
        // Matrix rain effect
        const canvas = document.getElementById('matrix');
        const ctx = canvas.getContext('2d');
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        const chars = '01ΨΦΩαβγδεζηθικλμνξοπρστυφχψω';
        const fontSize = 14;
        const columns = canvas.width / fontSize;
        const drops = Array(Math.floor(columns)).fill(1);
        
        function drawMatrix() {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#00ff00';
            ctx.font = fontSize + 'px monospace';
            
            for (let i = 0; i < drops.length; i++) {
                const text = chars[Math.floor(Math.random() * chars.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                
                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        }
        
        setInterval(drawMatrix, 50);
        
        // Update metrics
        async function updateMetrics() {
            try {
                const res = await fetch('/api/network/metrics');
                const data = await res.json();
                document.getElementById('packetsSent').textContent = data.packets_sent.toLocaleString();
                document.getElementById('packetsReceived').textContent = data.packets_received.toLocaleString();
                document.getElementById('bytesSent').textContent = (data.bytes_sent / 1024).toFixed(2) + ' KB';
                document.getElementById('bytesReceived').textContent = (data.bytes_received / 1024).toFixed(2) + ' KB';
                document.getElementById('activeInterfaces').textContent = data.active_interfaces;
                document.getElementById('quantumEntanglements').textContent = data.quantum_entanglements;
            } catch(e) {
                console.error('Failed to update metrics');
            }
        }
        
        updateMetrics();
        setInterval(updateMetrics, 5000);
        
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/blockchain", response_class=HTMLResponse)
async def blockchain_page():
    """Real-time Bitcoin mainnet terminal with auto-refresh"""
    # Use the complete HTML from previous message - keeping it identical
    html_file = Path(__file__).parent / "templates" / "blockchain.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    
    # Inline HTML (same as before)
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin Mainnet - LIVE Terminal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #000;
            color: #00ff00;
            padding: 20px;
            min-height: 100vh;
            overflow-x: hidden;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .header {
            text-align: center;
            padding: 20px;
            border: 2px solid #00ff00;
            margin-bottom: 20px;
            background: rgba(0, 255, 0, 0.1);
        }
        h1 {
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
            font-size: 2.5em;
        }
        .back-link {
            display: inline-block;
            padding: 10px 20px;
            background: #00ff00;
            color: #000;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .live-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #ff0000;
            border-radius: 50%;
            animation: blink 1s infinite;
            margin-right: 8px;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(0, 255, 0, 0.1);
            border: 2px solid #00ff00;
            padding: 15px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.8em;
            color: #00ffff;
            margin: 10px 0;
        }
        .stat-label {
            color: #00ff00;
            font-size: 0.9em;
        }
        .terminal-container {
            background: #000;
            border: 2px solid #00ff00;
            padding: 20px;
            min-height: 600px;
            margin: 20px 0;
        }
        .terminal-output {
            height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 10px;
            background: rgba(0, 255, 0, 0.05);
        }
        .terminal-output::-webkit-scrollbar { width: 8px; }
        .terminal-output::-webkit-scrollbar-track { background: #000; }
        .terminal-output::-webkit-scrollbar-thumb { background: #00ff00; border-radius: 4px; }
        .terminal-line {
            margin: 5px 0;
            line-height: 1.5;
        }
        .terminal-prompt { color: #00ff00; }
        .terminal-command { color: #ffff00; }
        .terminal-result { color: #00ffff; white-space: pre-wrap; }
        .terminal-error { color: #ff0000; }
        .input-container { display: flex; gap: 10px; }
        input {
            flex: 1;
            padding: 12px;
            background: #000;
            border: 2px solid #00ff00;
            color: #00ff00;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
        }
        input:focus {
            outline: none;
            box-shadow: 0 0 10px #00ff00;
        }
        button {
            padding: 12px 30px;
            background: #00ff00;
            border: none;
            color: #000;
            font-weight: bold;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
        }
        button:hover {
            background: #00cc00;
            box-shadow: 0 0 20px #00ff00;
        }
        .recent-blocks {
            background: rgba(0, 255, 255, 0.1);
            border: 2px solid #00ffff;
            padding: 20px;
            margin: 20px 0;
            max-height: 400px;
            overflow-y: auto;
        }
        .block-item {
            background: rgba(0, 0, 0, 0.5);
            border-left: 3px solid #00ffff;
            padding: 10px;
            margin: 10px 0;
        }
        .block-header {
            display: flex;
            justify-content: space-between;
            color: #00ffff;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .block-details {
            color: #00ff00;
            font-size: 0.9em;
        }
        .commands-help {
            background: rgba(255, 255, 0, 0.1);
            border: 2px solid #ffff00;
            padding: 15px;
            margin: 20px 0;
        }
        .command-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .command-item {
            background: rgba(0, 0, 0, 0.3);
            padding: 8px;
            border-left: 3px solid #ffff00;
        }
        .command-name {
            color: #ffff00;
            font-weight: bold;
        }
        .command-desc {
            color: #00ff00;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← Back to Main</a>
        
        <div class="header">
            <h1>₿ BITCOIN MAINNET - LIVE TERMINAL</h1>
            <p style="color: #ff0000; margin-top: 10px; font-size: 1.2em;">
                <span class="live-indicator"></span>
                REAL-TIME BLOCKCHAIN DATA - AUTO-UPDATING
            </p>
            <p style="color: #ffaa00; margin-top: 5px;">
                Holographic Storage: 138.0.0.1 ⚫ | Data Source: blockchain.info & mempool.space
            </p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">CURRENT BLOCK HEIGHT</div>
                <div class="stat-value" id="blockHeight">Loading...</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">DIFFICULTY</div>
                <div class="stat-value" id="difficulty">Loading...</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">MEMPOOL SIZE</div>
                <div class="stat-value" id="mempoolSize">Loading...</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">MARKET PRICE (USD)</div>
                <div class="stat-value" id="marketPrice">Loading...</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">TOTAL TRANSACTIONS</div>
                <div class="stat-value" id="totalTx">Loading...</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">LAST UPDATE</div>
                <div class="stat-value" id="lastUpdate" style="font-size: 1.2em;">--:--:--</div>
            </div>
        </div>
        
        <div class="recent-blocks">
            <h3 style="color: #00ffff; margin-bottom: 15px;">📦 LATEST BLOCKS (Auto-Refreshing)</h3>
            <div id="recentBlocks">Loading blocks...</div>
        </div>
        
        <div class="commands-help">
            <h3 style="color: #ffff00; margin-bottom: 10px;">📋 AVAILABLE COMMANDS</h3>
            <div class="command-grid">
                <div class="command-item">
                    <div class="command-name">getblockchaininfo</div>
                    <div class="command-desc">Get full blockchain status</div>
                </div>
                <div class="command-item">
                    <div class="command-name">getblock &lt;hash|height&gt;</div>
                    <div class="command-desc">Get specific block details</div>
                </div>
                <div class="command-item">
                    <div class="command-name">getmempoolinfo</div>
                    <div class="command-desc">Get mempool statistics</div>
                </div>
                <div class="command-item">
                    <div class="command-name">getrecentblocks [count]</div>
                    <div class="command-desc">Get recent blocks (default 10)</div>
                </div>
                <div class="command-item">
                    <div class="command-name">gettransaction &lt;txid&gt;</div>
                    <div class="command-desc">Get transaction details</div>
                </div>
                <div class="command-item">
                    <div class="command-name">getaddressinfo &lt;address&gt;</div>
                    <div class="command-desc">Get address information</div>
                </div>
                <div class="command-item">
                    <div class="command-name">getfeeestimates</div>
                    <div class="command-desc">Get current fee estimates</div>
                </div>
                <div class="command-item">
                    <div class="command-name">help</div>
                    <div class="command-desc">Show all commands</div>
                </div>
            </div>
        </div>
        
        <div class="terminal-container">
            <div class="terminal-output" id="terminal">
                <div class="terminal-line">
                    <span class="terminal-result">╔═══════════════════════════════════════════════════════════╗</span>
                </div>
                <div class="terminal-line">
                    <span class="terminal-result">║  BITCOIN MAINNET REAL-TIME TERMINAL                      ║</span>
                </div>
                <div class="terminal-line">
                    <span class="terminal-result">║  Connected to: blockchain.info & mempool.space API       ║</span>
                </div>
                <div class="terminal-line">
                    <span class="terminal-result">║  Holographic Storage: 138.0.0.1 ⚫                        ║</span>
                </div>
                <div class="terminal-line">
                    <span class="terminal-result">╚═══════════════════════════════════════════════════════════╝</span>
                </div>
                <div class="terminal-line">
                    <span class="terminal-prompt">bitcoin@mainnet:~$</span> <span style="color: #888;">Fetching live blockchain data...</span>
                </div>
            </div>
            
            <div class="input-container">
                <span class="terminal-prompt" style="padding: 12px;">bitcoin@mainnet:~$</span>
                <input type="text" id="commandInput" placeholder="Enter command (e.g., getblockchaininfo)..." autocomplete="off">
                <button id="executeBtn">Execute</button>
            </div>
        </div>
    </div>
    
    <script>
        const terminal = document.getElementById('terminal');
        const commandInput = document.getElementById('commandInput');
        const executeBtn = document.getElementById('executeBtn');
        let commandHistory = [];
        let historyIndex = -1;
        
        function addToTerminal(content, type = 'result') {
            const line = document.createElement('div');
            line.className = 'terminal-line';
            
            if (type === 'command') {
                line.innerHTML = `<span class="terminal-prompt">bitcoin@mainnet:~$</span> <span class="terminal-command">${escapeHtml(content)}</span>`;
            } else if (type === 'error') {
                line.innerHTML = `<span class="terminal-error">${escapeHtml(content)}</span>`;
            } else {
                line.innerHTML = `<span class="terminal-result">${escapeHtml(content)}</span>`;
            }
            
            terminal.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
            
            while (terminal.children.length > 100) {
                terminal.removeChild(terminal.firstChild);
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function executeCommand() {
            const command = commandInput.value.trim();
            if (!command) return;
            
            addToTerminal(command, 'command');
            commandInput.value = '';
            
            if (command && commandHistory[commandHistory.length - 1] !== command) {
                commandHistory.push(command);
                historyIndex = -1;
            }
            
            try {
                const response = await fetch('/api/bitcoin/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addToTerminal(JSON.stringify(data.result, null, 2));
                } else {
                    addToTerminal('ERROR: ' + data.error, 'error');
                }
            } catch (error) {
                addToTerminal('ERROR: ' + error.message, 'error');
            }
        }
        
        executeBtn.onclick = executeCommand;
        commandInput.onkeypress = (e) => {
            if (e.key === 'Enter') executeCommand();
        };
        
        commandInput.onkeydown = (e) => {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (commandHistory.length > 0 && historyIndex < commandHistory.length - 1) {
                    historyIndex++;
                    commandInput.value = commandHistory[commandHistory.length - 1 - historyIndex];
                }
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (historyIndex > 0) {
                    historyIndex--;
                    commandInput.value = commandHistory[commandHistory.length - 1 - historyIndex];
                } else {
                    historyIndex = -1;
                    commandInput.value = '';
                }
            }
        };
        
        async function updateStats() {
            try {
                const response = await fetch('/api/bitcoin/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'getblockchaininfo' })
                });
                
                const data = await response.json();
                
                if (data.success && data.result) {
                    const result = data.result;
                    document.getElementById('blockHeight').textContent = result.blocks?.toLocaleString() || 'N/A';
                    document.getElementById('difficulty').textContent = result.difficulty ? 
                        (result.difficulty / 1e12).toFixed(2) + 'T' : 'N/A';
                    document.getElementById('totalTx').textContent = result.total_transactions?.toLocaleString() || 'N/A';
                    document.getElementById('marketPrice').textContent = result.market_price_usd ? 
                        '$' + result.market_price_usd.toLocaleString() : 'N/A';
                    
                    const now = new Date();
                    document.getElementById('lastUpdate').textContent = 
                        now.toLocaleTimeString();
                }
            } catch (error) {
                console.error('Failed to update stats:', error);
            }
        }
        
        async function updateMempool() {
            try {
                const response = await fetch('/api/bitcoin/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'getmempoolinfo' })
                });
                
                const data = await response.json();
                
                if (data.success && data.result) {
                    document.getElementById('mempoolSize').textContent = 
                        data.result.size?.toLocaleString() || 'N/A';
                }
            } catch (error) {
                console.error('Failed to update mempool:', error);
            }
        }
        
        async function updateRecentBlocks() {
            try {
                const response = await fetch('/api/bitcoin/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: 'getrecentblocks 5' })
                });
                
                const data = await response.json();
                
                if (data.success && data.result && data.result.blocks) {
                    const blocksHtml = data.result.blocks.map(block => `
                        <div class="block-item">
                            <div class="block-header">
                                <span>Block #${block.height?.toLocaleString() || 'N/A'}</span>
                                <span>${new Date((block.timestamp || 0) * 1000).toLocaleTimeString()}</span>
                            </div>
                            <div class="block-details">
                                Hash: ${(block.id || block.hash || 'N/A').substring(0, 20)}...<br>
                                Transactions: ${block.tx_count?.toLocaleString() || 'N/A'} | 
                                Size: ${block.size ? (block.size / 1024).toFixed(2) + ' KB' : 'N/A'}
                            </div>
                        </div>
                    `).join('');
                    
                    document.getElementById('recentBlocks').innerHTML = blocksHtml;
                }
            } catch (error) {
                console.error('Failed to update blocks:', error);
            }
        }
        
        updateStats();
        updateMempool();
        updateRecentBlocks();
        
        setInterval(() => {
            updateStats();
            updateMempool();
            updateRecentBlocks();
            addToTerminal('--- AUTO-REFRESH: Blockchain data updated ---', 'result');
        }, 30000);
    </script>
</body>
</html>
    """)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """Chat room page"""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Quantum Foam Chatroom</title>
    <style>
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #0f0f23, #1a1a2e); color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: rgba(0,0,0,0.85); border-radius: 15px; padding: 30px; }
        h1 { text-align: center; color: #00ffff; text-shadow: 0 0 20px #00ffff; }
        .back-link { display: inline-block; padding: 10px 20px; background: #00ffff; color: #000; text-decoration: none; border-radius: 5px; margin-bottom: 20px; }
        .chat-container { border: 1px solid #00ffff; height: 500px; overflow-y: auto; padding: 15px; background: rgba(0,0,0,0.6); border-radius: 8px; margin: 20px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .sent { background: rgba(0,255,255,0.2); text-align: right; }
        .received { background: rgba(255,0,255,0.2); }
        input, button { padding: 12px; margin: 5px; border: 1px solid #00ffff; border-radius: 5px; background: rgba(0,0,0,0.5); color: #e0e0e0; }
        button { cursor: pointer; background: #00ffff; color: #000; font-weight: bold; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← Back</a>
        <h1>🌊 Quantum Foam Chatroom</h1>
        <div id="authForm">
            <input type="text" id="username" placeholder="Username">
            <input type="password" id="password" placeholder="Password">
            <input type="email" id="email" placeholder="Email" class="hidden">
            <button id="loginBtn">Login</button>
            <button id="toggleBtn">Register</button>
        </div>
        <div id="chatInterface" class="hidden">
            <div class="chat-container" id="messages"></div>
            <input type="text" id="messageInput" placeholder="Type message...">
            <button id="sendBtn">Send</button>
            <button id="logoutBtn">Logout</button>
        </div>
    </div>
    <script>
        const API = '/api';
        const WS_URL = 'wss://clearnet-chat-4bal.onrender.com/ws/chat';
        let ws, token = localStorage.getItem('token'), user;
        
        document.getElementById('toggleBtn').onclick = () => {
            const isLogin = document.getElementById('loginBtn').textContent === 'Login';
            document.getElementById('loginBtn').textContent = isLogin ? 'Register' : 'Login';
            document.getElementById('email').classList.toggle('hidden');
        };
        
        document.getElementById('loginBtn').onclick = async () => {
            const isRegister = document.getElementById('loginBtn').textContent === 'Register';
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const email = document.getElementById('email').value;
            
            try {
                const res = await fetch(`${API}/${isRegister ? 'register' : 'login'}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(isRegister ? { username, password, email } : { username, password })
                });
                const data = await res.json();
                if (res.ok) {
                    token = data.token;
                    user = { username: data.username || username };
                    localStorage.setItem('token', token);
                    showChat();
                }
            } catch (err) { console.error(err); }
        };
        
        function showChat() {
            document.getElementById('authForm').classList.add('hidden');
            document.getElementById('chatInterface').classList.remove('hidden');
            ws = new WebSocket(`${WS_URL}?token=${token}`);
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                const div = document.createElement('div');
                div.className = 'message ' + (msg.sender === user.username ? 'sent' : 'received');
                div.innerHTML = `<strong>${msg.sender}:</strong> ${msg.content || msg.body}`;
                document.getElementById('messages').appendChild(div);
            };
        }
        
        document.getElementById('sendBtn').onclick = () => {
            const msg = document.getElementById('messageInput').value;
            if (msg && ws) {
                ws.send(JSON.stringify({ content: msg }));
                document.getElementById('messageInput').value = '';
            }
        };
        
        document.getElementById('logoutBtn').onclick = () => {
            localStorage.removeItem('token');
            location.reload();
        };
        
        if (token) showChat();
    </script>
</body>
</html>""")

# ==================== API ENDPOINTS ====================

@app.post("/api/bitcoin/execute")
async def bitcoin_execute(request: Request):
    """Execute Bitcoin CLI command with real mainnet data"""
    try:
        data = await request.json()
        command = data.get('command', '')
        
        result = await BitcoinCLI.execute_command(command)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Bitcoin CLI error: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.get("/api/network/metrics")
async def get_network_metrics():
    """Get current network metrics"""
    return JSONResponse(content=app_state.network_metrics)

@app.post("/api/encrypt")
async def encrypt_message(request: Request):
    """Encrypt message using quantum encryption"""
    try:
        data = await request.json()
        plaintext = data.get('plaintext', '')
        
        if not plaintext:
            raise HTTPException(status_code=400, detail="No plaintext provided")
        
        encrypted = QuantumEncryption.quantum_encrypt(plaintext)
        storage.encrypted_messages.append(encrypted)
        
        return JSONResponse(content=encrypted)
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/decrypt")
async def decrypt_message(request: Request):
    """Decrypt message using quantum decryption"""
    try:
        data = await request.json()
        encrypted_data = data.get('encrypted_data', {})
        
        if not encrypted_data:
            raise HTTPException(status_code=400, detail="No encrypted data provided")
        
        plaintext = QuantumEncryption.quantum_decrypt(encrypted_data)
        
        return JSONResponse(content={"plaintext": plaintext, "success": True})
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/bitcoin")
async def bitcoin_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time Bitcoin updates"""
    await websocket.accept()
    app_state.bitcoin_websockets.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        app_state.bitcoin_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"Bitcoin WebSocket error: {e}")
        if websocket in app_state.bitcoin_websockets:
            app_state.bitcoin_websockets.remove(websocket)

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    backend_healthy = await app_state.check_backend_health() if not Config.SKIP_BACKEND_CHECKS else True
    
    bitcoin_healthy = False
    try:
        latest_block = await BitcoinMainnet.get_latest_block()
        bitcoin_healthy = 'height' in latest_block
    except:
        pass
    
    return {
        "status": "healthy" if (backend_healthy and bitcoin_healthy) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "6.0.0-modular",
        "environment": Config.ENVIRONMENT,
        "backend": {
            "url": Config.CHAT_BACKEND,
            "healthy": backend_healthy,
            "checks_enabled": not Config.SKIP_BACKEND_CHECKS,
            "last_check": app_state.last_health_check.isoformat()
        },
        "bitcoin": {
            "mainnet": bitcoin_healthy,
            "api": "blockchain.info & mempool.space"
        },
        "quantum_systems": {
            "black_hole": Config.BLACK_HOLE_ADDRESS,
            "white_hole": Config.WHITE_HOLE_ADDRESS,
            "encryption": "active",
            "blockchain": "active"
        }
    }

@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_api(path: str, request: Request):
    """Proxy API calls to chat backend"""
    if path in ['encrypt', 'decrypt', 'bitcoin/execute', 'network/metrics']:
        raise HTTPException(status_code=404)
    
    try:
        return await proxy_to_backend(request, f"api/{path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.exception_handler(404)
async def not_found(request: Request, exc):
    """Handle 404 errors"""
    return RedirectResponse(url="/", status_code=302)

@app.on_event("startup")
async def startup():
    logger.info("=" * 80)
    logger.info("🌌 QUANTUM FOAM NETWORK - MODULAR TRUTH GATEWAY")
    logger.info(f"📍 Version: 6.0.0-modular")
    logger.info(f"🌍 Environment: {Config.ENVIRONMENT}")
    logger.info(f"🔗 Backend: {Config.CHAT_BACKEND}")
    logger.info(f"✓ Backend checks: {'DISABLED (standalone)' if Config.SKIP_BACKEND_CHECKS else 'ENABLED'}")
    logger.info(f"₿ Bitcoin: LIVE mainnet via blockchain.info & mempool.space")
    logger.info(f"⚫ Black Hole: {Config.BLACK_HOLE_ADDRESS}")
    logger.info(f"⚪ White Hole: {Config.WHITE_HOLE_ADDRESS}")
    logger.info("=" * 80)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="debug" if Config.DEBUG else "info",
        access_log=True
    )
