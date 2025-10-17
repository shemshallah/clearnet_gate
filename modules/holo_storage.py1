"""
Module 2: Holographic Storage Manager
Distributed quantum-inspired storage with holographic addressing

Development opportunities:
- Implement IPFS/Filecoin integration
- Add erasure coding (Reed-Solomon)
- Build distributed hash table (DHT)
- Implement content-addressable storage
- Add deduplication and compression
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import random


@dataclass
class HoloNode:
    """Holographic storage node"""
    node_id: str
    ip_address: str
    port: int
    storage_capacity_gb: float
    used_storage_gb: float
    latency_ms: float
    reliability_score: float  # 0.0 to 1.0
    last_seen: datetime
    
    @property
    def available_storage_gb(self) -> float:
        return self.storage_capacity_gb - self.used_storage_gb
    
    @property
    def is_online(self) -> bool:
        elapsed = (datetime.now() - self.last_seen).total_seconds()
        return elapsed < 60


@dataclass
class FileMetadata:
    """Metadata for stored files"""
    file_id: str
    filename: str
    file_hash: str
    size_bytes: int
    quantum_signature: str
    quantum_route: str
    storage_nodes: List[str]  # Node IDs storing replicas
    upload_time: datetime
    access_count: int
    last_accessed: datetime
    encryption: Optional[str] = None


class HoloStorageManager:
    """Manages distributed holographic storage"""
    
    def __init__(self, storage_ip: str = "138.0.0.1", upload_dir: Path = Path("./uploads")):
        self.storage_ip = storage_ip
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.nodes: Dict[str, HoloNode] = {}
        self.file_metadata: Dict[str, FileMetadata] = {}
        self.routing_table: Dict[str, List[str]] = {}
        
        # Initialize some nodes
        self._initialize_nodes()
        
    def _initialize_nodes(self):
        """Initialize storage nodes"""
        for i in range(5):
            node = HoloNode(
                node_id=f"HOLO-{i:04d}",
                ip_address=f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                port=10000 + i,
                storage_capacity_gb=random.uniform(100, 1000),
                used_storage_gb=random.uniform(10, 50),
                latency_ms=random.uniform(5, 50),
                reliability_score=random.uniform(0.85, 0.99),
                last_seen=datetime.now()
            )
            self.nodes[node.node_id] = node
    
    def is_healthy(self) -> bool:
        """Check if storage system is healthy"""
        online_nodes = sum(1 for node in self.nodes.values() if node.is_online)
        return online_nodes >= 3
    
    async def store_file(
        self,
        filename: str,
        content: bytes,
        quantum_signature: str,
        replicas: int = 3
    ) -> Dict:
        """Store file with quantum routing"""
        
        # Generate file hash
        file_hash = hashlib.sha256(content).hexdigest()
        file_id = file_hash[:16]
        
        # Generate quantum route
        quantum_route = self._generate_quantum_route()
        
        # Select optimal nodes for storage
        storage_nodes = self._select_storage_nodes(len(content), replicas)
        
        # Save file locally
        file_path = self.upload_dir / f"{file_id}_{filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Create metadata
        metadata = FileMetadata(
            file_id=file_id,
            filename=filename,
            file_hash=file_hash,
            size_bytes=len(content),
            quantum_signature=quantum_signature,
            quantum_route=quantum_route,
            storage_nodes=[node.node_id for node in storage_nodes],
            upload_time=datetime.now(),
            access_count=0,
            last_accessed=datetime.now()
        )
        
        self.file_metadata[file_id] = metadata
        
        # Update node storage
        bytes_per_node = len(content) / replicas
        for node in storage_nodes:
            node.used_storage_gb += bytes_per_node / (1024**3)
        
        return {
            "file_id": file_id,
            "quantum_route": quantum_route,
            "holo_storage": self.storage_ip,
            "storage_nodes": [
                {
                    "node_id": node.node_id,
                    "ip": node.ip_address,
                    "latency_ms": node.latency_ms
                }
                for node in storage_nodes
            ],
            "replicas": replicas,
            "sharding": "enabled" if len(content) > 10_000_000 else "disabled"
        }
    
    def _generate_quantum_route(self) -> str:
        """Generate quantum routing identifier"""
        route_id = ''.join(random.choices('0123456789ABCDEF', k=16))
        return f"QR:{route_id}"
    
    def _select_storage_nodes(self, file_size: int, replicas: int) -> List[HoloNode]:
        """Select optimal nodes using quantum-inspired selection"""
        
        # Filter online nodes with capacity
        available_nodes = [
            node for node in self.nodes.values()
            if node.is_online and node.available_storage_gb > file_size / (1024**3)
        ]
        
        if len(available_nodes) < replicas:
            # Need more nodes, use all available
            return available_nodes
        
        # Score nodes based on latency, reliability, and capacity
        def node_score(node: HoloNode) -> float:
            latency_score = 1.0 / (1.0 + node.latency_ms / 100)
            capacity_score = node.available_storage_gb / node.storage_capacity_gb
            return (
                node.reliability_score * 0.4 +
                latency_score * 0.3 +
                capacity_score * 0.3
            )
        
        # Sort by score and take top N
        sorted_nodes = sorted(available_nodes, key=node_score, reverse=True)
        return sorted_nodes[:replicas]
    
    async def retrieve_file(self, file_id: str) -> Optional[Dict]:
        """Retrieve file from storage"""
        
        if file_id not in self.file_metadata:
            return None
        
        metadata = self.file_metadata[file_id]
        
        # Find file on disk
        file_path = None
        for path in self.upload_dir.glob(f"{file_id}_*"):
            file_path = path
            break
        
        if not file_path or not file_path.exists():
            return None
        
        # Read content
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Update access stats
        metadata.access_count += 1
        metadata.last_accessed = datetime.now()
        
        return {
            "file_id": file_id,
            "filename": metadata.filename,
            "content": content,
            "hash": metadata.file_hash,
            "quantum_signature": metadata.quantum_signature,
            "size": metadata.size_bytes
        }
    
    async def list_files(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "name"
    ) -> List[Dict]:
        """List stored files"""
        
        files = []
        for metadata in self.file_metadata.values():
            # Get storage node info
            nodes_info = []
            for node_id in metadata.storage_nodes:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    nodes_info.append({
                        "node_id": node.node_id,
                        "ip": node.ip_address,
                        "latency_ms": node.latency_ms
                    })
            
            files.append({
                "file_id": metadata.file_id,
                "name": metadata.filename,
                "size": metadata.size_bytes,
                "hash": metadata.file_hash,
                "quantum_route": metadata.quantum_route,
                "storage_nodes": nodes_info,
                "upload_time": metadata.upload_time.isoformat(),
                "access_count": metadata.access_count,
                "last_accessed": metadata.last_accessed.isoformat()
            })
        
        # Sort
        if sort_by == "name":
            files.sort(key=lambda x: x["name"])
        elif sort_by == "size":
            files.sort(key=lambda x: x["size"], reverse=True)
        elif sort_by == "date":
            files.sort(key=lambda x: x["upload_time"], reverse=True)
        
        return files[offset:offset + limit]
    
    async def get_stats(self) -> Dict:
        """Get storage statistics"""
        online_nodes = [n for n in self.nodes.values() if n.is_online]
        
        total_capacity = sum(n.storage_capacity_gb for n in online_nodes)
        total_used = sum(n.used_storage_gb for n in online_nodes)
        
        return {
            "total_files": len(self.file_metadata),
            "total_nodes": len(self.nodes),
            "online_nodes": len(online_nodes),
            "total_capacity_gb": round(total_capacity, 2),
            "used_storage_gb": round(total_used, 2),
            "available_storage_gb": round(total_capacity - total_used, 2),
            "utilization_percent": round((total_used / total_capacity) * 100, 2) if total_capacity > 0 else 0,
            "average_latency_ms": round(sum(n.latency_ms for n in online_nodes) / len(online_nodes), 2) if online_nodes else 0,
            "average_reliability": round(sum(n.reliability_score for n in online_nodes) / len(online_nodes), 3) if online_nodes else 0
        }
    
    async def sync_nodes(self):
        """Background task to sync nodes"""
        while True:
            await asyncio.sleep(30)
            
            # Simulate node status updates
            for node in self.nodes.values():
                # Random latency fluctuation
                node.latency_ms = max(5, node.latency_ms + random.gauss(0, 2))
                node.last_seen = datetime.now()
                
                # Simulate reliability changes
                node.reliability_score += random.gauss(0, 0.01)
                node.reliability_score = max(0.7, min(0.99, node.reliability_score))
    
    async def rebalance_storage(self):
        """
        Rebalance files across nodes for optimal performance
        
        Development opportunity: Implement intelligent rebalancing:
        - Move hot files to low-latency nodes
        - Replicate popular files
        - Consolidate cold storage
        """
        pass
    
    async def garbage_collect(self):
        """
        Remove orphaned files and clean up storage
        
        Development opportunity: Implement:
        - TTL-based expiration
        - LRU cache eviction
        - Deduplication
        """
        pass
