"""
Utility Modules: Security, Analytics, Blockchain, P2P Network, Cache Manager
"""

import asyncio
import hashlib
import hmac
import json
import random
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


# ============================================================================
# SECURITY MANAGER
# ============================================================================

class SecurityManager:
    """Security and authentication manager"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.blocked_ips: set = set()
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        
    async def scan_file(self, content: bytes) -> bool:
        """
        Scan file for malware/threats
        Development: Integrate ClamAV, VirusTotal API
        """
        # Basic checks
        if len(content) == 0:
            return False
        
        # Check for common malware signatures (simplified)
        malware_signatures = [b'<script>', b'eval(', b'exec(']
        for sig in malware_signatures:
            if sig in content[:1000]:  # Check first 1KB
                return False
        
        return True
    
    def generate_token(self, user_id: str) -> str:
        """Generate secure authentication token"""
        timestamp = str(int(time.time()))
        data = f"{user_id}:{timestamp}".encode()
        signature = hmac.new(
            self.secret_key.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
        return f"{user_id}:{timestamp}:{signature}"
    
    def verify_token(self, token: str) -> bool:
        """Verify authentication token"""
        try:
            user_id, timestamp, signature = token.split(':')
            
            # Check expiration (24 hours)
            if int(time.time()) - int(timestamp) > 86400:
                return False
            
            # Verify signature
            data = f"{user_id}:{timestamp}".encode()
            expected_sig = hmac.new(
                self.secret_key.encode(),
                data,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_sig)
        except:
            return False


class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    async def check_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit"""
        now = time.time()
        
        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

@dataclass
class AnalyticsEvent:
    """Analytics event"""
    event_type: str
    data: Dict
    timestamp: datetime
    session_id: Optional[str] = None


class AnalyticsEngine:
    """Analytics and metrics collection"""
    
    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_time = datetime.now()
    
    async def log_event(self, event_type: str, data: Dict):
        """Log analytics event"""
        event = AnalyticsEvent(
            event_type=event_type,
            data=data,
            timestamp=datetime.now()
        )
        self.events.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
    
    async def log_metric(self, metric_name: str, value: float):
        """Log numeric metric"""
        self.metrics[metric_name].append(value)
        
        # Keep only recent values
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    async def get_summary(self) -> Dict:
        """Get analytics summary"""
        # Event counts by type
        event_counts = defaultdict(int)
        for event in self.events:
            event_counts[event.event_type] += 1
        
        # Metric statistics
        metric_stats = {}
        for name, values in self.metrics.items():
            if values:
                metric_stats[name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return {
            "total_events": len(self.events),
            "event_counts": dict(event_counts),
            "metric_statistics": metric_stats,
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    async def generate_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """Generate analytics report"""
        # Filter events by date range
        filtered_events = self.events
        
        if start_date:
            start = datetime.fromisoformat(start_date)
            filtered_events = [e for e in filtered_events if e.timestamp >= start]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            filtered_events = [e for e in filtered_events if e.timestamp <= end]
        
        # Aggregate statistics
        event_counts = defaultdict(int)
        for event in filtered_events:
            event_counts[event.event_type] += 1
        
        return {
            "report_period": {
                "start": start_date or self.start_time.isoformat(),
                "end": end_date or datetime.now().isoformat()
            },
            "total_events": len(filtered_events),
            "events_by_type": dict(event_counts),
            "generated_at": datetime.now().isoformat()
        }
    
    async def collect_metrics(self):
        """Background task to collect system metrics"""
        while True:
            await asyncio.sleep(60)
            # Collect metrics here
            await self.log_metric("heartbeat", time.time())


# ============================================================================
# BLOCKCHAIN LEDGER
# ============================================================================

@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: str
    transactions: List[Dict]
    previous_hash: str
    nonce: int
    hash: str


class BlockchainLedger:
    """Simple blockchain for audit trail"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.difficulty = 2  # Proof of work difficulty
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create first block"""
        genesis = Block(
            index=0,
            timestamp=datetime.now().isoformat(),
            transactions=[],
            previous_hash="0",
            nonce=0,
            hash=""
        )
        genesis.hash = self._calculate_hash(genesis)
        self.chain.append(genesis)
    
    def _calculate_hash(self, block: Block) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            "index": block.index,
            "timestamp": block.timestamp,
            "transactions": block.transactions,
            "previous_hash": block.previous_hash,
            "nonce": block.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    async def add_transaction(self, transaction: Dict):
        """Add transaction to pending pool"""
        self.pending_transactions.append({
            **transaction,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mine block if enough transactions
        if len(self.pending_transactions) >= 10:
            await self.mine_block()
    
    async def mine_block(self):
        """Mine a new block"""
        if not self.pending_transactions:
            return
        
        previous_block = self.chain[-1]
        
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            transactions=self.pending_transactions[:],
            previous_hash=previous_block.hash,
            nonce=0,
            hash=""
        )
        
        # Proof of work
        while True:
            new_block.hash = self._calculate_hash(new_block)
            if new_block.hash[:self.difficulty] == "0" * self.difficulty:
                break
            new_block.nonce += 1
        
        self.chain.append(new_block)
        self.pending_transactions = []
    
    async def add_file_record(self, file_hash: str, quantum_route: str):
        """Add file upload record"""
        await self.add_transaction({
            "type": "file_upload",
            "file_hash": file_hash,
            "quantum_route": quantum_route
        })
    
    async def get_chain(self) -> List[Dict]:
        """Get blockchain as list of dicts"""
        return [
            {
                "index": block.index,
                "timestamp": block.timestamp,
                "transactions": block.transactions,
                "previous_hash": block.previous_hash,
                "hash": block.hash,
                "nonce": block.nonce
            }
            for block in self.chain
        ]
    
    async def validate_chain(self) -> bool:
        """Validate blockchain integrity"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check hash
            if current.hash != self._calculate_hash(current):
                return False
            
            # Check link
            if current.previous_hash != previous.hash:
                return False
        
        return True
    
    async def get_stats(self) -> Dict:
        """Get blockchain statistics"""
        return {
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "is_valid": await self.validate_chain(),
            "difficulty": self.difficulty
        }
    
    def is_synced(self) -> bool:
        """Check if blockchain is synced"""
        return len(self.pending_transactions) < 100
    
    async def finalize(self):
        """Finalize blockchain (mine remaining transactions)"""
        while self.pending_transactions:
            await self.mine_block()


# ============================================================================
# P2P NETWORK MANAGER
# ============================================================================

@dataclass
class Peer:
    """Network peer"""
    peer_id: str
    ip_address: str
    port: int
    last_seen: datetime
    connection_quality: float


class P2PNetworkManager:
    """Peer-to-peer network management"""
    
    def __init__(self, port: int = 9000, dht_enabled: bool = True):
        self.port = port
        self.dht_enabled = dht_enabled
        self.peers: Dict[str, Peer] = {}
        self.node_id = hashlib.sha256(str(random.random()).encode()).hexdigest()[:16]
        self.is_running = False
    
    async def start(self):
        """Start P2P network"""
        self.is_running = True
        
        # Initialize some peers for demo
        for i in range(5):
            peer = Peer(
                peer_id=f"PEER-{i:04d}",
                ip_address=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                port=self.port + i,
                last_seen=datetime.now(),
                connection_quality=random.uniform(0.7, 1.0)
            )
            self.peers[peer.peer_id] = peer
    
    async def stop(self):
        """Stop P2P network"""
        self.is_running = False
        self.peers.clear()
    
    async def get_peers(self) -> List[Dict]:
        """Get connected peers"""
        return [
            {
                "peer_id": peer.peer_id,
                "ip_address": peer.ip_address,
                "port": peer.port,
                "last_seen": peer.last_seen.isoformat(),
                "connection_quality": round(peer.connection_quality, 3)
            }
            for peer in self.peers.values()
        ]
    
    async def broadcast(self, message: str) -> Dict:
        """Broadcast message to peers"""
        # Simulate broadcast
        return {
            "success": True,
            "peer_count": len(self.peers),
            "message_size": len(message)
        }
    
    async def get_stats(self) -> Dict:
        """Get network statistics"""
        return {
            "node_id": self.node_id,
            "port": self.port,
            "peer_count": len(self.peers),
            "dht_enabled": self.dht_enabled,
            "is_running": self.is_running
        }
    
    def is_connected(self) -> bool:
        """Check if connected to network"""
        return self.is_running and len(self.peers) > 0


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """LRU cache for API responses"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            del self.timestamps[oldest]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.cache.move_to_end(key)
    
    async def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    async def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "utilization": len(self.cache) / self.max_size
        }
