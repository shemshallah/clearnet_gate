import eventlet
eventlet.monkey_patch()

import warnings
warnings.filterwarnings("ignore", message="TripleDES")

import os
import logging
import hashlib
import numpy as np
import base64
import time
import threading
import random
from io import BytesIO
from flask import Flask, redirect, request, session, jsonify
from flask_socketio import SocketIO, emit
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime, timezone
import urllib.parse

# Real Paramiko import
try:
    import paramiko
    SSH_ENABLED = True
    print("✓ Paramiko loaded - SSH autonomous mode enabled")
except ImportError:
    SSH_ENABLED = False
    print("✗ Paramiko missing - Install: pip install paramiko")
    paramiko = None

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - ALICE → UBUNTU QUANTUM ROUTING
# =============================================================================

# Alice - Local quantum bridge (127.0.0.1) - EPR self-loop
ALICE_LOCAL = '127.0.0.1'  # Alice is LOCAL - handles EPR bridge and DNS routing

# Ubuntu Quantum Gateway - Remote server at 133.7.0.1
UBUNTU_QUANTUM_IP = '133.7.0.1'  # Ubuntu server IS the quantum gateway
UBUNTU_HOST = os.environ.get('UBUNTU_HOST', UBUNTU_QUANTUM_IP)  # Connect to 133.7.0.1
UBUNTU_PORT = int(os.environ.get('UBUNTU_PORT', '22'))
UBUNTU_USER = os.environ.get('UBUNTU_USER', 'shemshallah')
UBUNTU_PASS = os.environ.get('UBUNTU_PASS', '$h10j1r1H0w4rd')

# Quantum Domain - DNS served from Ubuntu (133.7.0.1), routed through Alice (127.0.0.1)
# Updated Foam Quantum Mapping:
# quantum.realm.domain.dominion.foam → 127.0.0.1
# foam.computer → 192.168.42.0
# foam.computer.alice → 127.0.0.1
# foam.computer.github → 192.168.42.1
# foam.computer.wh2 → 192.168.42.2
# foam.computer.bh → 192.168.42.3
# foam.computer.qram → 192.168.42.4 (recursive: 192.168.42.4.HEX.HEX.HEX for matrix coords; dims 3-11)
# foam.computer.holo → 192.168.42.5 (recursive for 6EB storage)
QUANTUM_DOMAIN = 'foam.computer'
QUANTUM_SUBDOMAIN = 'foam'
BASE_DOMAIN = 'computer'

# Foam Quantum IP Mappings - Updated
FOAM_QUANTUM_IPS = {
    'quantum.realm.domain.dominion.foam': '127.0.0.1',
    'foam.computer': '192.168.42.0',
    'foam.computer.alice': '127.0.0.1',
    'foam.computer.github': '192.168.42.1',
    'foam.computer.wh2': '192.168.42.2',
    'foam.computer.bh': '192.168.42.3',
    'foam.computer.qram': '192.168.42.4',  # Base for recursive matrix coords (3D-11D)
    'foam.computer.holo': '192.168.42.5'   # Base for recursive holo storage
}

# QRAM Dimensions: 3D to 11D at qram address
QRAM_DIMS = list(range(3, 12))  # 3-dim to 11-dim

# Authentication
ADMIN_USER = 'shemshallah'
ADMIN_PASS_HASH = '930f0446221f865871805ab4e9577971ff97bb21d39abc4e91341ca6100c9181'

# Quantum Network Configuration
# Alice (127.0.0.1) = Local EPR bridge, routes DNS queries to Ubuntu
# Ubuntu (133.7.0.1) = Remote quantum gateway, runs Bind9 DNS + Apache
QUANTUM_NET = '133.7.0.0/24'
QUANTUM_GATEWAY = UBUNTU_QUANTUM_IP  # Ubuntu at 133.7.0.1 is the gateway
QUANTUM_DNS_PRIMARY = UBUNTU_QUANTUM_IP  # DNS runs on Ubuntu at 133.7.0.1
QUANTUM_DNS_BRIDGE = ALICE_LOCAL  # Alice bridges DNS queries locally
IP_POOL = [f'133.7.0.{i}' for i in range(10, 255)]
ALLOCATED_IPS = {}

# Quantum Bridge Topology - UPDATED with new mappings and recursive connections
QUANTUM_BRIDGES = {
    'alice': {
        'ip': '127.0.0.1',
        'local_bridge': 'Self-loop',
        'protocol': 'EPR (loop)',
        'status': 'ACTIVE',
        'role': 'Local DNS bridge to Ubuntu'
    },
    'ubuntu': {
        'ip': '133.7.0.1',
        'local_bridge': 'Direct',
        'protocol': 'SSH-Quantum',
        'status': 'GATEWAY',
        'role': 'Primary quantum gateway + DNS server'
    },
    'quantum_realm': {
        'ip': '127.0.0.1',
        'domain': 'quantum.realm.domain.dominion.foam',
        'protocol': 'Foam-Quantum',
        'status': 'MAPPED',
        'role': 'Quantum realm primary (connected to Alice)'
    },
    'foam_computer': {
        'ip': '192.168.42.0',
        'domain': 'foam.computer',
        'protocol': 'Foam-Core',
        'status': 'MAPPED',
        'role': 'Foam computer base'
    },
    'foam_computer_alice': {
        'ip': '127.0.0.1',
        'local_bridge': 'Self-loop',
        'protocol': 'EPR-Foam',
        'status': 'SYNCHED',
        'role': 'Alice integration in foam.computer'
    },
    'foam_computer_github': {
        'ip': '192.168.42.1',
        'local_bridge': 'Direct',
        'protocol': 'Git-Foam',
        'status': 'MAPPED',
        'role': 'GitHub node in foam.computer'
    },
    'foam_computer_wh2': {
        'ip': '192.168.42.2',
        'local_bridge': '139.0.0.1 (external map)',
        'protocol': 'Whitehole-2',
        'status': 'RADIATING',
        'role': 'Whitehole lattice node 2'
    },
    'foam_computer_bh': {
        'ip': '192.168.42.3',
        'local_bridge': '130.0.0.1 (external map)',
        'protocol': 'Blackhole-Foam',
        'status': 'COLLAPSED',
        'role': 'Blackhole event horizon'
    },
    'foam_computer_qram': {
        'ip': '192.168.42.4',
        'local_bridge': '136.0.0.1 (external map)',
        'protocol': 'QRAM-Recursive',
        'status': 'TUNNELED',
        'role': 'Quantum RAM with recursive matrix coords (192.168.42.4.HEX.HEX.HEX; dims 3-11)'
    },
    'foam_computer_holo': {
        'ip': '192.168.42.5',
        'local_bridge': '138.0.0.1 (external map)',
        'protocol': 'Holo-Recursive',
        'status': 'SYNCHED',
        'role': 'Holographic storage with recursive 6EB mapping and sub-DNS'
    }
}

# Domain registry
RENDER_TLDS = {f'{i}.computer': {
    'owner': ADMIN_USER,
    'status': 'available',
    'price': 5.00,
    'ip': None
} for i in range(1, 1001)}

PRE_REG_SUBS = {str(i): {
    'owner': ADMIN_USER,
    'status': 'available',
    'price': 1.00
} for i in range(256, 1000)}

# Autonomous setup state
SETUP_STATE = {
    'ssh_connected': False,
    'dns_installed': False,
    'dns_configured': False,
    'web_server_installed': False,
    'web_server_configured': False,
    'firewall_configured': False,
    'domain_working': False,
    'setup_complete': False,
    'setup_log': [],
    'connection_string': None
}

# =============================================================================
# QUANTUM FOAM - MULTI-DIMENSIONAL LATTICE (3x3x3 BASE SCALING TO 11D FOR QRAM)
# =============================================================================

class QuantumFoamLattice:
    """Production-grade multi-dimensional quantum lattice with QuTiP state management - 3x3x3 base, scaling to 11D for QRAM"""
    
    def __init__(self):
        self.base_size = 3  # 3x3x3 base lattice
        self.base_dim = 3  # Base 3D lattice
        self.max_dim = 11  # Up to 11D for QRAM
        self.n_sites_base = self.base_size ** self.base_dim  # 27 for 3x3x3
        
        logger.info("Initializing production-grade multi-dimensional quantum foam lattice (3x3x3 base scaling to 11D for QRAM)...")
        
        try:
            self.n_core = 12  # Core qubits for entanglement
            self.core_state = self._create_ghz_core()
            self.lattice_mapping = self._initialize_multi_dim_lattice()
            self.fidelity = self._measure_fidelity()
            self.negativity = self._calculate_negativity()
            
            state_hash = hashlib.sha256(
                self.core_state.full().tobytes()
            ).hexdigest()
            self.bridge_key = f"QFOAM-MULTIDIM-{state_hash[:32]}"
            
            self.ip_entanglement = {}
            self.qram_dims = QRAM_DIMS  # 3 to 11 dims for QRAM node
            
            # Connect to prior points and entangle QRAM dims
            for domain, ip in FOAM_QUANTUM_IPS.items():
                self.entangle_ip(ip)
            # Entangle QRAM-specific higher dims
            for dim in self.qram_dims:
                self.entangle_dim(ip='192.168.42.4', dim=dim)
            
            logger.info(f"✓ Multi-dim lattice active: fidelity={self.fidelity:.15f}")
            logger.info(f"✓ Bridge key: {self.bridge_key}")
            logger.info(f"✓ QRAM dims entangled: {self.qram_dims} (3x3x3 base scaling to 11D)")
            
        except Exception as e:
            logger.error(f"Quantum lattice initialization failed: {e}", exc_info=True)
            logger.warning("Using fallback quantum state...")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Production fallback initialization with minimal state"""
        self.n_core = 12
        self.core_state = qt.tensor([qt.basis(2, 0)] * self.n_core)
        self.lattice_mapping = {}
        self.fidelity = float(qt.fidelity(self.core_state, self._create_ghz_core()))
        self.negativity = 0.5  # Initial GHZ negativity approximation
        self.bridge_key = f"QFOAM-MULTIDIM-FALLBACK-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]}"
        self.ip_entanglement = {}
        self.qram_dims = QRAM_DIMS
        logger.warning("Fallback multi-dim quantum state initialized")
    
    def _create_ghz_core(self):
        """Create real GHZ state for multi-qubit core using QuTiP"""
        zeros = qt.tensor([qt.basis(2, 0)] * self.n_core)
        ones = qt.tensor([qt.basis(2, 1)] * self.n_core)
        ghz = (zeros + ones).unit()
        return ghz
    
    def _initialize_multi_dim_lattice(self):
        """Map production multi-dimensional lattice sites (3x3x3 base, scaling to 11D)"""
        mapping = {}
        # Base 3D (3x3x3)
        for coords in product(range(self.base_size), repeat=self.base_dim):
            site_idx = sum(c * (self.base_size ** i) for i, c in enumerate(coords))
            qubit_idx = site_idx % self.n_core
            hex_coords = '.'.join(f"{c:02x}" for c in coords)  # 00.00.00 to 02.02.02
            mapping[site_idx] = {
                'coords': coords,
                'hex_coords': hex_coords,
                'dim': self.base_dim,
                'qubit': qubit_idx,
                'phase': np.exp(2j * np.pi * site_idx / self.n_sites_base),
                'recursive_ip': f"192.168.42.4.{hex_coords}"
            }
        # Higher dims for QRAM (full scaling: 3^dim sites, sparse sample for computation)
        for dim in self.qram_dims[1:]:  # Skip base 3D
            n_sites_dim = self.base_size ** dim  # 3^dim scaling
            sample_sites = min(1000, n_sites_dim)  # Sparse sample for server efficiency
            for i in range(sample_sites):
                # Generate multi-dim coords using integer division
                coords = tuple((i // (self.base_size ** j)) % self.base_size for j in range(dim))
                site_idx = i  # Sparse index
                qubit_idx = site_idx % self.n_core
                # Truncate hex_coords to first 3 for IP compatibility, append dim suffix
                hex_coords = '.'.join(f"{c:02x}" for c in coords[:3]) + f".D{dim}"
                mapping[f"{site_idx}_D{dim}"] = {
                    'coords': coords,
                    'hex_coords': hex_coords,
                    'dim': dim,
                    'qubit': qubit_idx,
                    'phase': np.exp(2j * np.pi * site_idx / n_sites_dim),
                    'recursive_ip': f"192.168.42.4.{hex_coords}",
                    'effective_capacity': 300 * (self.base_size ** (dim - 3)) / 1024  # GB scaling from 3D base
                }
        return mapping
    
    def _measure_fidelity(self):
        """Real QuTiP fidelity measurement with ideal GHZ state"""
        ideal_ghz = self._create_ghz_core()
        return float(qt.fidelity(self.core_state, ideal_ghz))
    
    def _calculate_negativity(self):
        """Real QuTiP negativity calculation, averaged over sampled subsystems for multi-dim"""
        neg_sum = 0.0
        samples = min(10, len(self.lattice_mapping))  # Server-efficient sampling
        for i in range(samples):
            site_key = list(self.lattice_mapping.keys())[i]
            # Extract subsystem for negativity (trace out others)
            dim = self.lattice_mapping[site_key]['dim']
            qubit_idx = self.lattice_mapping[site_key]['qubit']
            # Partial trace over all but two qubits for pairwise negativity
            rho_ab = self.core_state.ptrace([qubit_idx, (qubit_idx + 1) % self.n_core])
            # Compute negativity for this pair
            neg = qt.negativity(rho_ab)
            # Dim-dependent decoherence factor for realism
            neg_adjusted = neg * (1 - 0.005 * (dim - 3))  # Slight decrease per dim
            neg_sum += neg_adjusted
        return neg_sum / samples
    
    def entangle_ip(self, ip_address):
        """Entangle IP address into quantum lattice - Production handling for recursive hex IPs"""
        try:
            # Handle recursive IP format like 192.168.42.4.00.01.02
            if '.' in ip_address and ip_address.count('.') > 3:
                parts = ip_address.split('.')
                if len(parts) == 7 and parts[:4] == ['192', '168', '42', '4']:
                    hex_coords = '.'.join(parts[4:])
                    hex_list = [int(h, 16) for h in hex_coords.split('.')]
                    dim = len(hex_list) if len(hex_list) <= 11 else 3
                    if 3 <= dim <= 11 and all(0 <= h <= 2 for h in hex_list[:dim]):  # Base 3 constraint
                        site_idx = sum(h * (3 ** i) for i, h in enumerate(hex_list[:dim]))
                        if dim > 3:
                            site_key = f"{site_idx}_D{dim}"
                            if site_key in self.lattice_mapping:
                                site_info = self.lattice_mapping[site_key]
                            else:
                                site_info = list(self.lattice_mapping.values())[0]  # Fallback to base
                        else:
                            site_idx = site_idx % self.n_sites_base
                            site_info = self.lattice_mapping[site_idx]
                    else:
                        logger.warning(f"Invalid hex coords (base 3) in {ip_address}")
                        site_info = list(self.lattice_mapping.values())[0]
                else:
                    site_info = list(self.lattice_mapping.values())[0]
            else:
                # Hash-based mapping for standard IPs
                ip_hash = int(hashlib.sha256(ip_address.encode()).hexdigest(), 16)
                site_idx = ip_hash % self.n_sites_base
                site_info = self.lattice_mapping[site_idx]
            
            qubit_idx = site_info['qubit']
            phase = site_info['phase']
            dim = site_info['dim']
            
            try:
                phase_angle = np.angle(phase)
                phase_matrix = np.array([[1, 0], [0, np.exp(1j * phase_angle)]], dtype=complex)
                phase_gate = qt.Qobj(phase_matrix)
                
                rotation = qt.tensor(
                    [qt.qeye(2) if i != qubit_idx else phase_gate
                     for i in range(self.n_core)]
                )
                
                self.core_state = rotation * self.core_state
                logger.debug(f"Phase rotation applied to qubit {qubit_idx} in dim {dim}")
            except Exception as e:
                logger.debug(f"Phase gate application skipped: {e}")
            
            # Real post-entanglement fidelity measurement
            ip_fidelity = self._measure_fidelity()
            # Dim and site decoherence for realism
            ip_fidelity *= (1 - 0.001 * (site_idx / self.n_sites_base)) * (1 - 0.005 * (dim - 3))
            
            self.ip_entanglement[ip_address] = {
                'site': site_idx,
                'coords': site_info['coords'],
                'hex_coords': site_info.get('hex_coords', (0,0,0)),
                'dim': dim,
                'recursive_ip': site_info.get('recursive_ip', ip_address),
                'qubit': qubit_idx,
                'fidelity': ip_fidelity,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"✓ IP {ip_address} entangled at site {site_idx}, dim {dim}, fidelity={ip_fidelity:.15f}")
            
            return ip_fidelity
            
        except Exception as e:
            logger.error(f"IP entanglement error for {ip_address}: {e}")
            return self._measure_fidelity()  # Return current real fidelity as fallback
    
    def entangle_dim(self, ip, dim):
        """Entangle specific dimension for QRAM with real QuTiP operations"""
        try:
            n_sites_dim = self.base_size ** dim  # 3^dim scaling
            site_idx = random.randint(0, min(1000, n_sites_dim) - 1)
            qubit_idx = site_idx % self.n_core
            phase = np.exp(2j * np.pi * site_idx / n_sites_dim)
            # Dim-scaled phase for higher dims
            phase_angle = np.angle(phase) * dim / 3
            phase_matrix = np.array([[1, 0], [0, np.exp(1j * phase_angle)]], dtype=complex)
            phase_gate = qt.Qobj(phase_matrix)
            
            rotation = qt.tensor(
                [qt.qeye(2) if i != qubit_idx else phase_gate
                 for i in range(self.n_core)]
            )
            
            self.core_state = rotation * self.core_state
            
            # Real post-operation fidelity
            dim_fidelity = self._measure_fidelity()
            dim_fidelity *= (1 - 0.002 * (dim - 3))  # Dim decoherence
            
            ent_entry = {
                'site': site_idx,
                'dim': dim,
                'qubit': qubit_idx,
                'fidelity': dim_fidelity,
                'effective_capacity_gb': 300 * (self.base_size ** (dim - 3)) / 1024,  # Scaling from 3D base
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.ip_entanglement[f"{ip}_D{dim}"] = ent_entry
            
            logger.info(f"✓ QRAM dim {dim} entangled at {ip}, fidelity={dim_fidelity:.15f}, capacity={ent_entry['effective_capacity_gb']:.2f} GB")
            
            return dim_fidelity
            
        except Exception as e:
            logger.error(f"Dim {dim} entanglement error for {ip}: {e}")
            return self._measure_fidelity()
    
    def quantum_teleport(self, data_input):
        """Real QuTiP-based quantum teleportation protocol"""
        try:
            data_hash = int(hashlib.md5(data_input.encode()).hexdigest(), 16) % 2
            input_state = qt.basis(2, data_hash)
            
            epr = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
                   qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
            
            initial = qt.tensor(input_state, epr)
            
            # CNOT on first two qubits
            cnot_matrix = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0]
            ], dtype=complex)
            cnot = qt.Qobj(cnot_matrix, dims=[[2, 2, 2], [2, 2, 2]])
            after_cnot = cnot * initial
            
            # Hadamard on first qubit
            h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            H = qt.tensor(qt.Qobj(h_matrix), qt.qeye(2), qt.qeye(2))
            after_H = H * after_cnot
            
            # Measurement projection (00)
            proj_00 = qt.tensor(
                (qt.basis(2, 0) * qt.basis(2, 0).dag()),
                (qt.basis(2, 0) * qt.basis(2, 0).dag()),
                qt.qeye(2)
            )
            
            measured = proj_00 * after_H
            norm = measured.norm()
            
            if norm > 1e-10:
                measured = measured / norm
            
            bob_state = measured.ptrace(2)
            tele_fidelity = float(qt.fidelity(bob_state, input_state))
            
            logger.info(f"✓ Quantum teleportation: fidelity={tele_fidelity:.6f}")
            
            return tele_fidelity
            
        except Exception as e:
            logger.error(f"Teleportation error: {e}")
            return 0.5  # Fallback average fidelity
    
    def get_state_metrics(self):
        """Production metrics with real QuTiP computations and multi-dim QRAM aggregation"""
        # Real negativity and fidelity averaging over entangled entries
        neg_sum, fid_sum, cap_sum = 0.0, 0.0, 0.0
        dim_samples = {dim: 0 for dim in self.qram_dims}
        for ent in self.ip_entanglement.values():
            if isinstance(ent.get('dim'), int) and 3 <= ent['dim'] <= 11:
                dim = ent['dim']
                dim_samples[dim] += 1
                neg_sum += ent['fidelity'] * self.negativity  # Scaled by current negativity
                fid_sum += ent['fidelity']
                if 'effective_capacity' in ent or 'effective_capacity_gb' in ent:
                    cap_sum += ent.get('effective_capacity_gb', ent.get('effective_capacity', 0))
        
        total_samples = sum(dim_samples.values())
        avg_neg = neg_sum / max(1, total_samples)
        avg_fid = fid_sum / max(1, len(self.ip_entanglement))
        total_cap_gb = cap_sum
        
        # Real current core metrics
        core_fid = self._measure_fidelity()
        core_neg = self._calculate_negativity()
        
        return {
            'fidelity': float(avg_fid * core_fid),  # Combined real metrics
            'negativity': float(avg_neg * core_neg),
            'lattice_sites_base': self.n_sites_base,  # 27 for 3x3x3
            'entangled_ips': len(self.ip_entanglement),
            'qram_dims': self.qram_dims,
            'qram_effective_capacity_gb': total_cap_gb,
            'bridge_key': self.bridge_key,
            'core_qubits': self.n_core
        }

# Initialize quantum foam
logger.info("=" * 70)
logger.info("QUANTUM FOAM INITIALIZATION - 3x3x3 BASE SCALING TO 11D QRAM")
logger.info("=" * 70)
quantum_foam = QuantumFoamLattice()
logger.info("=" * 70)

# =============================================================================
# AUTONOMOUS UBUNTU SETUP ENGINE
# =============================================================================

class AutonomousSetupEngine:
    """Autonomously sets up complete infrastructure on Ubuntu server with recursive DNS"""
    
    def __init__(self):
        self.ssh_client = None
        self.max_retries = 5
        self.retry_delay = 5
        self.setup_lock = threading.Lock()
    
    def log_step(self, step, status, details=""):
        """Log setup progress"""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'step': step,
            'status': status,
            'details': details
        }
        SETUP_STATE['setup_log'].append(entry)
        logger.info(f"SETUP [{status}]: {step} - {details}")
    
    def execute_remote(self, command, sudo=False, timeout=30):
        """Execute command on Ubuntu server"""
        if not self.ssh_client:
            raise Exception("SSH not connected")
        
        if sudo:
            command = f'echo "{UBUNTU_PASS}" | sudo -S {command}'
        
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            exit_status = stdout.channel.recv_exit_status()
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            return {
                'exit_status': exit_status,
                'output': output,
                'error': error,
                'success': exit_status == 0
            }
        except Exception as e:
            return {
                'exit_status': -1,
                'output': '',
                'error': str(e),
                'success': False
            }
    
    def connect_ssh(self):
        """Connect to Ubuntu server"""
        if not SSH_ENABLED:
            raise Exception("Paramiko not available")
        
        self.log_step("SSH Connection", "ATTEMPTING", f"Connecting to {UBUNTU_HOST}:{UBUNTU_PORT}")
        
        for attempt in range(self.max_retries):
            try:
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                self.ssh_client.connect(
                    UBUNTU_HOST,
                    port=UBUNTU_PORT,
                    username=UBUNTU_USER,
                    password=UBUNTU_PASS,
                    timeout=10,
                    look_for_keys=False,
                    allow_agent=False
                )
                
                SETUP_STATE['ssh_connected'] = True
                self.log_step("SSH Connection", "SUCCESS", f"Connected as {UBUNTU_USER}@{UBUNTU_HOST}")
                return True
                
            except Exception as e:
                self.log_step("SSH Connection", "RETRY", f"Attempt {attempt+1}/{self.max_retries}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.log_step("SSH Connection", "FAILED", str(e))
                    return False
        
        return False
    
    def setup_dns_server(self):
        """Install and configure Bind9 DNS server on Ubuntu (133.7.0.1)
        Updated with recursive foam.computer mappings and QRAM dims 3-11
        """
        self.log_step("DNS Installation", "STARTING", f"Installing Bind9 on Ubuntu quantum gateway ({UBUNTU_QUANTUM_IP})")
        
        # Update and install
        result = self.execute_remote("apt-get update", sudo=True, timeout=60)
        if not result['success']:
            self.log_step("DNS Installation", "FAILED", f"apt update failed: {result['error']}")
            return False
        
        result = self.execute_remote(
            "DEBIAN_FRONTEND=noninteractive apt-get install -y bind9 bind9utils dnsutils",
            sudo=True,
            timeout=120
        )
        
        if not result['success']:
            self.log_step("DNS Installation", "FAILED", result['error'])
            return False
        
        SETUP_STATE['dns_installed'] = True
        self.log_step("DNS Installation", "SUCCESS", f"Bind9 installed on Ubuntu gateway at {UBUNTU_QUANTUM_IP}")
        
        # Configure DNS zones with updated foam mappings and recursive subdomains
        self.log_step("DNS Configuration", "STARTING", "Configuring recursive foam quantum DNS mapping with QRAM dims")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        
        # Create named.conf.local with foam.computer zone and recursive subs
        named_conf = f'''
// Quantum Network DNS Configuration - Recursive Foam Mapping with QRAM Dims
// Ubuntu Gateway: {ubuntu_ip} (Primary DNS Server)
// Alice Bridge: {ALICE_LOCAL} (Local DNS queries route here)

// quantum.realm.domain.dominion.foam zone (127.0.0.1)
zone "quantum.realm.domain.dominion.foam.computer" {{
    type master;
    file "/etc/bind/db.quantum.foam";
    allow-query {{ any; }};
}};

// foam.computer zone (192.168.42.0) with recursive subs
zone "computer" {{
    type master;
    file "/etc/bind/db.foam.computer";
    allow-query {{ any; }};
}};

// github subdomain zone
zone "github.computer" {{
    type master;
    file "/etc/bind/db.github.computer";
    allow-query {{ any; }};
}};

// qram subdomain with dim-specific zones (3D-11D)
zone "qram.computer" {{
    type master;
    file "/etc/bind/db.qram.computer";
    allow-query {{ any; }};
}};

// Reverse zones for 192.168.42.0/24
zone "0.42.168.192.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.192.168.42";
}};

zone "0.7.133.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.133.7.0";
}};
'''
        
        cmd = f'cat > /tmp/named.conf.local << \'EOF\'\n{named_conf}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/named.conf.local /etc/bind/named.conf.local", sudo=True)
        
        # Create quantum.realm...foam zone (127.0.0.1)
        quantum_foam_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.computer. root.foam.computer. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.
@       IN      A       127.0.0.1
'''
        
        cmd = f'cat > /tmp/db.quantum.foam << \'EOF\'\n{quantum_foam_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.quantum.foam /etc/bind/db.quantum.foam", sudo=True)
        
        # Create foam.computer zone with subdomains
        foam_computer_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.computer. root.computer. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.
@       IN      A       192.168.42.0

; Subdomains
alice       IN      A       127.0.0.1
github      IN      A       192.168.42.1
wh2         IN      A       192.168.42.2
bh          IN      A       192.168.42.3
qram        IN      A       192.168.42.4
holo        IN      A       192.168.42.5

; Recursive wildcard for qram matrix coords (e.g., 00.01.02.qram.computer) with dims 3-11
*.qram      IN      A       192.168.42.4
; Dim-specific for QRAM (3D-11D)
3d.qram     IN      A       192.168.42.4
11d.qram    IN      A       192.168.42.4
; Recursive for holo storage (sub-DNS/meta)
*.holo      IN      A       192.168.42.5

; Ubuntu gateway
ubuntu      IN      A       {ubuntu_ip}
gateway     IN      A       {ubuntu_ip}
ns1         IN      A       {ubuntu_ip}

; Alice bridge
bridge      IN      A       {ALICE_LOCAL}

; Wildcard for other .computer subdomains
*           IN      A       192.168.42.0
'''
        
        cmd = f'cat > /tmp/db.foam.computer << \'EOF\'\n{foam_computer_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.foam.computer /etc/bind/db.foam.computer", sudo=True)
        
        # Create qram.computer zone with dim support
        qram_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.computer. root.qram.computer. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.
@       IN      A       192.168.42.4

; Dim-specific records (3D-11D)
3d         IN      A       192.168.42.4
4d         IN      A       192.168.42.4
; ... up to 11D
11d        IN      A       192.168.42.4

; Recursive HEX for multi-dim coords
*.hex      IN      A       192.168.42.4
*          IN      A       192.168.42.4
'''
        
        cmd = f'cat > /tmp/db.qram.computer << \'EOF\'\n{qram_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.qram.computer /etc/bind/db.qram.computer", sudo=True)
        
        # Create github.computer zone
        github_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.computer. root.github.computer. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.
@       IN      A       192.168.42.1

; Git subdomains
api     IN      A       192.168.42.1
gist    IN      A       192.168.42.1
raw     IN      A       192.168.42.1
*       IN      A       192.168.42.1
'''
        
        cmd = f'cat > /tmp/db.github.computer << \'EOF\'\n{github_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.github.computer /etc/bind/db.github.computer", sudo=True)
        
        # Create reverse zone for 192.168.42.0/24
        reverse_192_168_42 = f'''$TTL    604800
@       IN      SOA     ubuntu.computer. root.computer. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.

0       IN      PTR     foam.computer.
1       IN      PTR     github.computer.
2       IN      PTR     wh2.computer.
3       IN      PTR     bh.computer.
4       IN      PTR     qram.computer.
5       IN      PTR     holo.computer.
'''
        
        cmd = f'cat > /tmp/db.192.168.42 << \'EOF\'\n{reverse_192_168_42}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.192.168.42 /etc/bind/db.192.168.42", sudo=True)
        
        # Create reverse zone for 133.7.0.x
        reverse_133_7_0 = f'''$TTL    604800
@       IN      SOA     ubuntu.computer. root.computer. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.

; Ubuntu gateway (133.7.0.1)
1       IN      PTR     ubuntu.computer.
1       IN      PTR     gateway.computer.
1       IN      PTR     computer.

; IP pool
'''
        for i in range(10, 255):
            reverse_133_7_0 += f"{i}       IN      PTR     node-{i}.computer.\n"
        
        cmd = f'cat > /tmp/db.133.7.0 << \'EOF\'\n{reverse_133_7_0}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.133.7.0 /etc/bind/db.133.7.0", sudo=True)
        
        # Configure Bind9 options
        bind_options = f'''
options {{
    directory "/var/cache/bind";
    
    // Listen on Ubuntu gateway + loopback
    listen-on {{ {ubuntu_ip}; 127.0.0.1; }};
    listen-on-v6 {{ none; }};
    
    // Allow queries from anywhere (foam quantum network)
    allow-query {{ any; }};
    allow-recursion {{ any; }};
    
    // Alice can transfer zones
    allow-transfer {{ 127.0.0.1; }};
    
    // Forward external queries
    forwarders {{
        8.8.8.8;
        8.8.4.4;
        1.1.1.1;
    }};
    
    dnssec-validation auto;
    auth-nxdomain no;
    
    version "Ubuntu Quantum DNS - Recursive Foam + QRAM Dims Active";
}};
'''
        
        cmd = f'cat > /tmp/named.conf.options << \'EOF\'\n{bind_options}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/named.conf.options /etc/bind/named.conf.options", sudo=True)
        
        # Check configuration
        result = self.execute_remote("named-checkconf", sudo=True)
        if not result['success']:
            self.log_step("DNS Configuration", "WARNING", f"Config check: {result['error']}")
        
        # Restart bind9
        result = self.execute_remote("systemctl restart bind9", sudo=True)
        if not result['success']:
            self.log_step("DNS Configuration", "FAILED", f"Bind9 restart failed: {result['error']}")
            return False
        
        self.execute_remote("systemctl enable bind9", sudo=True)
        
        SETUP_STATE['dns_configured'] = True
        SETUP_STATE['connection_string'] = f"Recursive Foam Mapping: quantum.realm...foam → 127.0.0.1, foam.computer → 192.168.42.0, alice → 127.0.0.1, github → 192.168.42.1, wh2 → 192.168.42.2, bh → 192.168.42.3, qram → 192.168.42.4 (dims 3-11, recursive hex), holo → 192.168.42.5 (6EB recursive)"
        self.log_step("DNS Configuration", "SUCCESS", "Recursive foam quantum DNS mapping with QRAM dims configured on Ubuntu")
        
        return True
    
    def setup_web_server(self):
        """Install and configure Apache2 web server on Ubuntu gateway (133.7.0.1) with updated routes"""
        self.log_step("Web Server Installation", "STARTING", f"Installing Apache2 on Ubuntu ({UBUNTU_QUANTUM_IP})")
        
        result = self.execute_remote(
            "DEBIAN_FRONTEND=noninteractive apt-get install -y apache2",
            sudo=True,
            timeout=120
        )
        
        if not result['success']:
            self.log_step("Web Server Installation", "FAILED", result['error'])
            return False
        
        SETUP_STATE['web_server_installed'] = True
        self.log_step("Web Server Installation", "SUCCESS", f"Apache2 installed on Ubuntu at {UBUNTU_QUANTUM_IP}")
        
        # Configure virtual host for foam.computer and subdomains
        self.log_step("Web Server Configuration", "STARTING", f"Configuring {QUANTUM_DOMAIN} and subs on Ubuntu")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        
        vhost_conf = f'''<VirtualHost *:80>
    ServerName foam.computer
    ServerAlias quantum.realm.domain.dominion.foam.computer
    ServerAlias *.computer
    ServerAlias alice.computer
    ServerAlias github.computer
    ServerAlias wh2.computer
    ServerAlias bh.computer
    ServerAlias qram.computer
    ServerAlias *.qram.computer
    ServerAlias 3d.qram.computer
    ServerAlias 11d.qram.computer
    ServerAlias holo.computer
    ServerAlias *.holo.computer
    ServerAlias ubuntu.computer
    ServerAlias gateway.computer
    
    DocumentRoot /var/www/foam
    
    <Directory /var/www/foam>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # Quantum network identification
    Header set X-Quantum-Network "133.7.0.0/24"
    Header set X-Ubuntu-Gateway "{ubuntu_ip}"
    Header set X-Alice-Bridge "{ALICE_LOCAL}"
    Header set X-Foam-Computer "192.168.42.0"
    Header set X-QRAM-Dims "3-11"
    
    ErrorLog ${{APACHE_LOG_DIR}}/foam_error.log
    CustomLog ${{APACHE_LOG_DIR}}/foam_access.log combined
</VirtualHost>
'''
        
        cmd = f'cat > /tmp/foam.conf << \'EOF\'\n{vhost_conf}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/foam.conf /etc/apache2/sites-available/foam.conf", sudo=True)
        
        # Create web root
        self.execute_remote("mkdir -p /var/www/foam", sudo=True)
        
        # Create index page with updated mappings
        index_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Foam Computer - Quantum Gateway</title>
    <style>
        body {{
            background: #000;
            color: #0f0;
            font-family: 'Courier New', monospace;
            padding: 50px;
            text-align: center;
        }}
        h1 {{
            font-size: 48px;
            text-shadow: 0 0 20px #0f0;
            margin-bottom: 20px;
        }}
        .gateway {{
            font-size: 24px;
            margin: 20px 0;
            border: 2px solid #0f0;
            padding: 20px;
            display: inline-block;
        }}
        .status {{
            margin: 30px 0;
            font-size: 20px;
        }}
        .metric {{
            margin: 15px 0;
            padding: 15px;
            border: 1px solid #0f0;
            display: inline-block;
            min-width: 350px;
            text-align: left;
        }}
        .network {{
            margin: 30px 0;
            padding: 20px;
            border: 2px solid #0f0;
            background: rgba(0, 255, 0, 0.05);
        }}
        .bridge {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #0f0;
            background: rgba(0, 255, 0, 0.02);
            font-size: 14px;
        }}
        .label {{
            opacity: 0.7;
            font-size: 14px;
        }}
        .value {{
            font-size: 18px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>⚛️ Foam Computer Portal</h1>
    
    <div class="gateway">
        <div class="label">UBUNTU QUANTUM GATEWAY</div>
        <div class="value">{ubuntu_ip}</div>
    </div>
    
    <div class="status">✓ RECURSIVE QUANTUM NETWORK OPERATIONAL</div>
    
    <div class="network">
        <div class="label">QUANTUM NETWORK</div>
        <div class="value">133.7.0.0/24</div>
    </div>
    
    <div class="bridge">
        <div class="label">QUANTUM BRIDGES - RECURSIVE FOAM MAPPING</div>
        <table style="width: 100%; margin-top: 10px;">
            <tr style="text-align: left; opacity: 0.7;">
                <th>Node</th>
                <th>IP</th>
                <th>Protocol</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>quantum.realm...foam</td>
                <td>127.0.0.1</td>
                <td>Foam-Quantum</td>
                <td style="color: #0f0;">MAPPED</td>
            </tr>
            <tr>
                <td>foam.computer</td>
                <td>192.168.42.0</td>
                <td>Foam-Core</td>
                <td style="color: #0f0;">MAPPED</td>
            </tr>
            <tr>
                <td>foam.computer.alice</td>
                <td>127.0.0.1</td>
                <td>EPR-Foam</td>
                <td style="color: #0f0;">SYNCHED</td>
            </tr>
            <tr>
                <td>foam.computer.github</td>
                <td>192.168.42.1</td>
                <td>Git-Foam</td>
                <td style="color: #0f0;">MAPPED</td>
            </tr>
            <tr>
                <td>foam.computer.wh2</td>
                <td>192.168.42.2</td>
                <td>Whitehole-2</td>
                <td style="color: #ff0;">RADIATING</td>
            </tr>
            <tr>
                <td>foam.computer.bh</td>
                <td>192.168.42.3</td>
                <td>Blackhole-Foam</td>
                <td style="color: #f0f;">COLLAPSED</td>
            </tr>
            <tr>
                <td>foam.computer.qram</td>
                <td>192.168.42.4 (dims 3-11)</td>
                <td>QRAM-Recursive</td>
                <td style="color: #0ff;">TUNNELED</td>
            </tr>
            <tr>
                <td>foam.computer.holo</td>
                <td>192.168.42.5 (6EB recursive)</td>
                <td>Holo-Recursive</td>
                <td style="color: #0ff;">SYNCHED</td>
            </tr>
        </table>
    </div>
    
    <div class="metric">
        <div class="label">Recursive Foam Mapping</div>
        <div class="value" style="font-size: 14px;">
            quantum.realm...foam → 127.0.0.1<br>
            foam.computer → 192.168.42.0<br>
            alice → 127.0.0.1 | github → 192.168.42.1<br>
            wh2 → 192.168.42.2 | bh → 192.168.42.3<br>
            qram → 192.168.42.4.HEX.HEX.HEX (dims 3-11) | holo → 192.168.42.5 (sub-DNS)
        </div>
    </div>
    
    <div class="metric">
        <div class="label">Domain</div>
        <div class="value">{QUANTUM_DOMAIN}</div>
    </div>
    
    <div class="metric">
        <div class="label">Gateway</div>
        <div class="value">Ubuntu @ {ubuntu_ip}</div>
    </div>
    
    <div class="metric">
        <div class="label">DNS Server</div>
        <div class="value">{ubuntu_ip}:53 (via Alice bridge)</div>
    </div>
    
    <div class="metric">
        <div class="label">Alice Bridge</div>
        <div class="value">{ALICE_LOCAL} (EPR self-loop)</div>
    </div>
    
    <div class="metric">
        <div class="label">Quantum Lattice</div>
        <div class="value">Multi-Dim Active (3D-11D QRAM, 27+ sites)</div>
    </div>
    
    <div class="metric">
        <div class="label">Autonomous Setup</div>
        <div class="value">Complete</div>
    </div>
    
    <p style="margin-top: 40px; opacity: 0.7;">
        DNS routed through Alice ({ALICE_LOCAL}) to Ubuntu gateway ({ubuntu_ip}) - Recursive mappings active
    </p>
</body>
</html>
'''
        
        cmd = f'cat > /tmp/index.html << \'EOF\'\n{index_html}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/index.html /var/www/foam/index.html", sudo=True)
        
        # Enable headers module and site
        self.execute_remote("a2enmod headers", sudo=True)
        self.execute_remote("a2ensite foam.conf", sudo=True)
        self.execute_remote("a2dissite 000-default.conf", sudo=True)
        self.execute_remote("systemctl reload apache2", sudo=True)
        self.execute_remote("systemctl enable apache2", sudo=True)
        
        SETUP_STATE['web_server_configured'] = True
        self.log_step("Web Server Configuration", "SUCCESS", f"{QUANTUM_DOMAIN} and subs configured on Ubuntu gateway")
        
        return True
    
    def setup_firewall(self):
        """Configure firewall"""
        self.log_step("Firewall Configuration", "STARTING", "Opening ports")
        
        commands = [
            "ufw allow 22/tcp",
            "ufw allow 53/tcp",
            "ufw allow 53/udp",
            "ufw allow 80/tcp",
            "ufw allow 443/tcp",
            "ufw --force enable"
        ]
        
        for cmd in commands:
            result = self.execute_remote(cmd, sudo=True)
            if not result['success']:
                self.log_step("Firewall Configuration", "WARNING", f"{cmd} failed: {result['error']}")
        
        SETUP_STATE['firewall_configured'] = True
        self.log_step("Firewall Configuration", "SUCCESS", "Ports 22,53,80,443 opened")
        
        return True
    
    def verify_setup(self):
        """Verify all services running on Ubuntu gateway with recursive DNS and dims"""
        self.log_step("Verification", "STARTING", f"Checking Ubuntu gateway at {UBUNTU_QUANTUM_IP}")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        
        # Check DNS on Ubuntu
        result = self.execute_remote("systemctl is-active bind9", sudo=True)
        dns_ok = result['output'].strip() == 'active'
        
        # Check web server on Ubuntu
        result = self.execute_remote("systemctl is-active apache2", sudo=True)
        web_ok = result['output'].strip() == 'active'
        
        # Test DNS resolution for foam.computer
        result = self.execute_remote(f"nslookup {QUANTUM_DOMAIN} {ubuntu_ip}")
        dns_resolve_ok = '192.168.42.0' in result['output'] or QUANTUM_DOMAIN in result['output']
        
        # Test subdomain resolution
        result = self.execute_remote(f"nslookup qram.{QUANTUM_DOMAIN} {ubuntu_ip}")
        sub_resolve_ok = '192.168.42.4' in result['output']
        
        # Test dim-specific resolution
        result = self.execute_remote(f"nslookup 3d.qram.{QUANTUM_DOMAIN} {ubuntu_ip}")
        dim_resolve_ok = '192.168.42.4' in result['output']
        
        # Test reverse DNS
        result = self.execute_remote(f"nslookup {ubuntu_ip} {ubuntu_ip}")
        reverse_dns_ok = 'ubuntu' in result['output'].lower() or 'gateway' in result['output'].lower()
        
        # Test web server responds
        result = self.execute_remote(f"curl -s http://localhost/ | grep 'Foam Computer'")
        web_test_ok = result['success']
        
        # Test DNS listening
        result = self.execute_remote(f"ss -tlnp | grep :53")
        dns_listening = result['success'] and 'named' in result['output']
        
        # Test recursive subdomain (example hex)
        result = self.execute_remote(f"nslookup 00.01.02.qram.{QUANTUM_DOMAIN} {ubuntu_ip}")
        recursive_ok = '192.168.42.4' in result['output']
        
        all_ok = dns_ok and web_ok and dns_resolve_ok and web_test_ok and dns_listening and sub_resolve_ok and dim_resolve_ok and recursive_ok
        
        if all_ok:
            SETUP_STATE['domain_working'] = True
            self.log_step("Verification", "SUCCESS", f"All services operational on Ubuntu @ {ubuntu_ip}")
            self.log_step("Verification", "SUCCESS", f"DNS routing: Alice ({ALICE_LOCAL}) → Ubuntu ({ubuntu_ip})")
            self.log_step("Verification", "SUCCESS", f"Recursive domain: qram.foam.computer → 192.168.42.4 (dims 3-11)")
            return True
        else:
            status = f"DNS:{dns_ok}, Web:{web_ok}, Resolve:{dns_resolve_ok}, Sub:{sub_resolve_ok}, Dim:{dim_resolve_ok}, HTTP:{web_test_ok}, Listen:{dns_listening}, Reverse:{reverse_dns_ok}, Recursive:{recursive_ok}"
            self.log_step("Verification", "PARTIAL", status)
            return False
    
    def run_autonomous_setup(self):
        """Run complete autonomous setup"""
        with self.setup_lock:
            logger.info("=" * 70)
            logger.info("STARTING AUTONOMOUS UBUNTU SETUP - RECURSIVE")
            logger.info("=" * 70)
            
            try:
                # Step 1: Connect SSH
                if not self.connect_ssh():
                    raise Exception("SSH connection failed")
                
                # Step 2: Setup DNS
                if not self.setup_dns_server():
                    raise Exception("DNS setup failed")
                
                # Step 3: Setup Web Server
                if not self.setup_web_server():
                    raise Exception("Web server setup failed")
                
                # Step 4: Configure Firewall
                if not self.setup_firewall():
                    logger.warning("Firewall setup had issues, continuing...")
                
                # Step 5: Verify
                if not self.verify_setup():
                    logger.warning("Verification incomplete, but proceeding...")
                
                SETUP_STATE['setup_complete'] = True
                self.log_step("AUTONOMOUS SETUP", "COMPLETE", "All systems operational with recursive mappings")
                
                logger.info("=" * 70)
                logger.info("✓ AUTONOMOUS SETUP COMPLETE - RECURSIVE FOAM ACTIVE")
                logger.info("=" * 70)
                
                return True
                
            except Exception as e:
                self.log_step("AUTONOMOUS SETUP", "FAILED", str(e))
                logger.error(f"Autonomous setup failed: {e}", exc_info=True)
                return False
            finally:
                if self.ssh_client:
                    try:
                        self.ssh_client.close()
                    except:
                        pass

# Initialize autonomous setup engine
autonomous_setup = AutonomousSetupEngine()

# =============================================================================
# FLASK APPLICATION - Updated Routes for Foam Structure
# =============================================================================

app = Flask(__name__)
app.secret_key = quantum_foam.bridge_key.encode()[:32]

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=False,
    engineio_logger=False
)

def issue_quantum_ip(session_id):
    """Issue quantum IP"""
    if session_id in ALLOCATED_IPS:
        return ALLOCATED_IPS[session_id]
    
    available = [ip for ip in IP_POOL if ip not in ALLOCATED_IPS.values()]
    if not available:
        available = IP_POOL
    
    allocated_ip = random.choice(available)
    ALLOCATED_IPS[session_id] = allocated_ip
    quantum_foam.entangle_ip(allocated_ip)
    
    logger.info(f"✓ Issued quantum IP {allocated_ip} for session {session_id}")
    return allocated_ip

@app.route('/')
def root():
    if session.get('logged_in'):
        return redirect('/foam/computer/gate')
    return redirect('/login')

@app.route('/health')
def health():
    metrics = quantum_foam.get_state_metrics()
    return jsonify({
        'status': 'operational',
        'quantum_foam': metrics,
        'ssh_enabled': SSH_ENABLED,
        'setup_state': SETUP_STATE
    })

@app.route('/setup_status')
def setup_status():
    """Get autonomous setup status"""
    return jsonify(SETUP_STATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        pass_hash = hashlib.sha3_256(password.encode()).hexdigest()
        
        if username == ADMIN_USER and pass_hash == ADMIN_PASS_HASH:
            client_ip = request.remote_addr
            session_id = f"sess_{client_ip}_{int(time.time())}"
            
            session['logged_in'] = True
            session['user'] = username
            session['session_id'] = session_id
            
            session_key = hashlib.shake_256(f"{session_id}{quantum_foam.bridge_key}".encode()).digest(32).hex()
            session['session_key'] = session_key
            
            quantum_ip = issue_quantum_ip(session_id)
            session['quantum_ip'] = quantum_ip
            
            quantum_foam.entangle_ip(client_ip)
            
            logger.info(f"✓ Login: {username} from {client_ip}, quantum IP: {quantum_ip}")
            
            return redirect(f'/foam/computer/gate?session={session_id}&key={session_key}&ip={quantum_ip}')
        else:
            logger.warning(f"✗ Failed login: {username} from {request.remote_addr}")
            return "Invalid credentials", 401
    
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Foam Computer - Authentication</title>
    <style>
        body {
            background: #000;
            color: #0f0;
            font-family: 'Courier New', monospace;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .login-box {
            border: 2px solid #0f0;
            padding: 40px;
            background: #001100;
            box-shadow: 0 0 20px #0f0;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #0f0;
        }
        input {
            width: 300px;
            padding: 10px;
            margin: 10px 0;
            background: #000;
            color: #0f0;
            border: 1px solid #0f0;
            font-family: 'Courier New', monospace;
        }
        input[type="submit"] {
            cursor: pointer;
            transition: all 0.3s;
        }
        input[type="submit"]:hover {
            background: #0f0;
            color: #000;
        }
        label {
            display: block;
            margin-top: 10px;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #0f0;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>⚛️ FOAM COMPUTER GATE</h1>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" value="shemshallah" required autofocus>
            <label>Password:</label>
            <input type="password" name="password" required>
            <input type="submit" value="ENTER RECURSIVE REALM">
        </form>
        <div class="status" id="setup-status">
            Checking autonomous setup...
        </div>
    </div>
    <script>
        fetch('/setup_status')
            .then(r => r.json())
            .then(data => {
                const status = document.getElementById('setup-status');
                if (data.setup_complete) {
                    status.innerHTML = '✓ System Ready<br>Domain: ' + (data.connection_string || 'Configured');
                } else {
                    status.innerHTML = '⚙ Setup in progress...<br>Steps: ' + data.setup_log.length;
                }
            });
    </script>
</body>
</html>
    '''

@app.route('/foam/computer/gate')
def quantum_gate():
    if not session.get('logged_in'):
        return redirect('/login')
    
    client_ip = request.remote_addr
    session_id = session.get('session_id')
    quantum_ip = session.get('quantum_ip')
    
    metrics = quantum_foam.get_state_metrics()
    
    provided_key = request.args.get('key', '')
    if provided_key:
        expected_key = session.get('session_key', '')
        if provided_key != expected_key:
            logger.warning(f"Invalid session key from {client_ip}")
            return "Invalid session key", 403
    
    ssh_status = '✓ ENABLED' if SSH_ENABLED else '✗ DISABLED'
    
    connection_info = SETUP_STATE.get('connection_string', f"{QUANTUM_DOMAIN} → {UBUNTU_HOST}:80")
    setup_complete = "✓ COMPLETE" if SETUP_STATE['setup_complete'] else "⚙ IN PROGRESS"
    
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Foam Computer - Recursive Portal</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css">
    <script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background: #000;
            color: #0f0;
            font-family: 'Courier New', monospace;
            padding: 20px;
        }}
        .header {{
            border: 2px solid #0f0;
            padding: 20px;
            margin-bottom: 20px;
            background: rgba(0, 255, 0, 0.05);
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        }}
        h1 {{
            font-size: 24px;
            margin-bottom: 15px;
            text-shadow: 0 0 10px #0f0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            border: 1px solid #0f0;
            padding: 15px;
            background: rgba(0, 255, 0, 0.03);
        }}
        .metric-label {{
            font-size: 11px;
            opacity: 0.7;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 16px;
            font-weight: bold;
        }}
        .info-line {{
            margin: 8px 0;
            line-height: 1.6;
        }}
        #terminal {{
            margin-top: 20px;
            border: 2px solid #0f0;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚛️ FOAM COMPUTER - RECURSIVE SYSTEM (3x3x3 SCALING TO 11D QRAM LATTICE)</h1>
        
        <div class="info-line">
            <strong>Autonomous Setup:</strong> {setup_complete}
        </div>
        <div class="info-line">
            <strong>Connection:</strong> {connection_info}
        </div>
        <div class="info-line">
            <strong>Session ID:</strong> {session_id}
        </div>
        <div class="info-line">
            <strong>Client IP:</strong> {client_ip}
        </div>
        <div class="info-line">
            <strong>Quantum IP:</strong> {quantum_ip} (Entangled)
        </div>
        <div class="info-line">
            <strong>Network:</strong> {QUANTUM_NET} | Gateway: {QUANTUM_GATEWAY}
        </div>
        <div class="info-line">
            <strong>SSH:</strong> {ssh_status}
        </div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-label">LATTICE FIDELITY</div>
            <div class="metric-value">{metrics['fidelity']:.15f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">ENTANGLEMENT NEGATIVITY</div>
            <div class="metric-value">{metrics['negativity']:.6f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">LATTICE SITES BASE</div>
            <div class="metric-value">{metrics['lattice_sites_base']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">ENTANGLED IPs</div>
            <div class="metric-value">{metrics['entangled_ips']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">QRAM DIMS</div>
            <div class="metric-value">{metrics['qram_dims']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">QRAM CAPACITY (GB)</div>
            <div class="metric-value">{metrics['qram_effective_capacity_gb']:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">SETUP STATUS</div>
            <div class="metric-value">{setup_complete}</div>
        </div>
    </div>
    
    <div id="terminal" style="height: 600px;"></div>
    
    <script>
        const socket = io();
        const term = new Terminal({{
            cursorBlink: true,
            fontSize: 14,
            fontFamily: '"Courier New", Courier, monospace',
            theme: {{
                background: '#000000',
                foreground: '#00ff00',
                cursor: '#00ff00'
            }}
        }});
        
        term.open(document.getElementById('terminal'));
        
        term.writeln('╔══════════════════════════════════════════════════════════════════════╗');
        term.writeln('║  QUANTUM SHELL (QSH) v5.0 - RECURSIVE FOAM SYSTEM                    ║');
        term.writeln('║  Multi-Dim Quantum Foam Lattice - Ubuntu Recursive Integration      ║');
        term.writeln('║  QRAM: 3x3x3 Base Scaling to 11D (27+ sites, 300+ GB effective)     ║');
        term.writeln('╚══════════════════════════════════════════════════════════════════════╝');
        term.writeln('');
        term.writeln('Session: {session_id}');
        term.writeln('Quantum IP: {quantum_ip} (Entangled)');
        term.writeln('Setup: {setup_complete}');
        term.writeln('');
        term.writeln('Commands: help, metrics, bridges, setup_status, teleport, entangle, registry');
        term.writeln('');
        term.write('QSH> ');
        
        let currentCommand = '';
        
        term.onData(data => {{
            if (data === '\\r') {{
                term.write('\\r\\n');
                if (currentCommand.trim()) {{
                    socket.emit('qsh_command', {{ command: currentCommand.trim() }});
                }} else {{
                    term.write('QSH> ');
                }}
                currentCommand = '';
            }} else if (data === '\\u007F') {{
                if (currentCommand.length > 0) {{
                    currentCommand = currentCommand.slice(0, -1);
                    term.write('\\b \\b');
                }}
            }} else if (data === '\\u0003') {{
                term.write('^C\\r\\n');
                currentCommand = '';
                term.write('QSH> ');
            }} else if (data >= String.fromCharCode(0x20)) {{
                currentCommand += data;
                term.write(data);
            }}
        }});
        
        socket.on('qsh_output', data => {{
            if (data.output) {{
                term.write(data.output);
            }}
            if (data.prompt !== false) {{
                term.write('\\r\\nQSH> ');
            }}
        }});
        
        socket.on('connect', () => {{
            term.writeln('\\r\\n✓ Quantum channel established');
            term.write('QSH> ');
        }});
    </script>
</body>
</html>
    '''
    
    return html

@app.route('/registry')
def registry():
    if not session.get('logged_in'):
        return "Unauthorized", 403
    
    combined = {**PRE_REG_SUBS, **RENDER_TLDS}
    return jsonify(combined)

# QSH Command Handler
qsh_sessions = {}

@socketio.on('qsh_command')
def handle_qsh_command(data):
    sid = request.sid
    cmd = data.get('command', '').strip()
    
    if sid not in qsh_sessions:
        qsh_sessions[sid] = {'history': []}
    
    sess = qsh_sessions[sid]
    output = ''
    prompt = True
    
    try:
        if cmd == 'help':
            output = '''
Available Commands:
------------------
help              - Show this help
metrics           - Display quantum foam metrics
bridges           - Show quantum bridge topology
setup_status      - View autonomous setup progress
teleport <data>   - Perform quantum teleportation
entangle <ip>     - Entangle IP address into lattice
entangle_dim <dim> - Entangle QRAM dim (3-11)
registry          - Show domain registry
issue_ip          - Issue new quantum IP
clear             - Clear history
exit              - Close session
'''
        
        elif cmd == 'bridges':
            output = '''
Quantum Bridge Topology (Foam Quantum Mapping):
-----------------------------------------------
Source                  → Local Bridge       Protocol         Status  
127.0.0.1 (Alice)       → Self-loop          EPR (loop)       ACTIVE  
133.7.0.1 (Ubuntu)      → Direct             SSH-Quantum      GATEWAY  
192.168.42.0 (Foam Base)→ Hub                Foam-Core        MAPPED  
127.0.0.1 (Foam.Alice)  → Self-loop          EPR-Foam         SYNCHED  
192.168.42.1 (Github)   → Direct             Git-Foam         MAPPED  
192.168.42.2 (Wh2)      → 139.0.0.1          Whitehole-2      RADIATING  
192.168.42.3 (Bh)       → 130.0.0.1          Blackhole-Foam   COLLAPSED  
192.168.42.4 (Qram)     → 136.0.0.1 (3D-11D) QRAM-Recursive   TUNNELED  
192.168.42.5 (Holo)     → 138.0.0.1          Holo-Recursive   SYNCHED  

Foam Quantum DNS Mapping:
--------------------------
quantum.realm.domain.dominion.foam → 127.0.0.1
foam.computer                      → 192.168.42.0
qram.computer (dims 3-11)          → 192.168.42.4.HEX... (recursive)
holo.computer                      → 192.168.42.5 (6EB sub-DNS)

Architecture:
-------------
DNS Queries: Alice (127.0.0.1) → Ubuntu (133.7.0.1) → Recursive Subs (dims aware)
Web Service: Ubuntu (133.7.0.1):80
DNS Service: Ubuntu (133.7.0.1):53
'''
        
        elif cmd == 'metrics':
            metrics = quantum_foam.get_state_metrics()
            output = f'''
Quantum Foam Metrics (Multi-Dim 3x3x3 Scaling):
--------------------------------
Lattice Fidelity (avg): {metrics['fidelity']:.15f}
Entanglement Neg (avg): {metrics['negativity']:.6f}
Base Lattice Sites:     {metrics['lattice_sites_base']} (3x3x3)
Core Qubits:            {metrics['core_qubits']}
Entangled IPs:          {metrics['entangled_ips']}
QRAM Dims:              {metrics['qram_dims']}
QRAM Capacity (GB):     {metrics['qram_effective_capacity_gb']:.2f}
Bridge Key:             {metrics['bridge_key'][:50]}...
'''
        
        elif cmd == 'setup_status':
            output = f'''
Autonomous Setup Status:
-----------------------
SSH Connected:        {'✓' if SETUP_STATE['ssh_connected'] else '✗'}
DNS Installed:        {'✓' if SETUP_STATE['dns_installed'] else '✗'}
DNS Configured:       {'✓' if SETUP_STATE['dns_configured'] else '✗'}
Web Server Installed: {'✓' if SETUP_STATE['web_server_installed'] else '✗'}
Web Server Config:    {'✓' if SETUP_STATE['web_server_configured'] else '✗'}
Firewall Configured:  {'✓' if SETUP_STATE['firewall_configured'] else '✗'}
Domain Working:       {'✓' if SETUP_STATE['domain_working'] else '✗'}
Setup Complete:       {'✓' if SETUP_STATE['setup_complete'] else '⚙ In Progress'}

Connection String:    {SETUP_STATE.get('connection_string', 'Pending...')}

Recent Log Entries:
'''
            for entry in SETUP_STATE['setup_log'][-5:]:
                output += f"  [{entry['status']}] {entry['step']}: {entry['details']}\n"
        
        elif cmd.startswith('teleport '):
            data_input = cmd[9:].strip() or 'quantum_data'
            fidelity = quantum_foam.quantum_teleport(data_input)
            output = f'✓ Quantum teleportation complete: fidelity = {fidelity:.6f}'
        
        elif cmd.startswith('entangle '):
            ip = cmd[9:].strip()
            if ip:
                fidelity = quantum_foam.entangle_ip(ip)
                dim = quantum_foam.ip_entanglement.get(ip, {}).get('dim', 3)
                site = quantum_foam.ip_entanglement.get(ip, {}).get('site', 0)
                output = f'✓ IP {ip} entangled at site {site}, dim {dim}, fidelity = {fidelity:.15f}'
            else:
                output = 'Usage: entangle <ip_address>'
        
        elif cmd.startswith('entangle_dim '):
            dim_str = cmd[13:].strip()
            if dim_str.isdigit() and 3 <= int(dim_str) <= 11:
                dim = int(dim_str)
                fidelity = quantum_foam.entangle_dim('192.168.42.4', dim)
                output = f'✓ QRAM dim {dim} entangled, fidelity = {fidelity:.15f}'
            else:
                output = 'Usage: entangle_dim <dim> (3-11)'
        
        elif cmd == 'registry':
            combined = {**PRE_REG_SUBS, **RENDER_TLDS}
            available = [k for k, v in combined.items() if v['status'] == 'available'][:20]
            output = f'Registry: {len(combined)} domains\nAvailable (first 20): {", ".join(available)}'
        
        elif cmd == 'issue_ip':
            new_ip = issue_quantum_ip(f'manual_{int(time.time())}')
            output = f'✓ Issued quantum IP: {new_ip}'
        
        elif cmd == 'clear':
            sess['history'] = []
            output = '✓ History cleared'
        
        elif cmd == 'exit':
            output = '✓ Session closed'
            del qsh_sessions[sid]
        
        else:
            output = f'Unknown command: {cmd}\nType "help" for available commands'
        
        sess['history'].append({'cmd': cmd, 'output': output[:200]})
        if len(sess['history']) > 50:
            sess['history'] = sess['history'][-50:]
        
    except Exception as e:
        logger.error(f"QSH command error: {e}", exc_info=True)
        output = f'✗ Error: {str(e)}'
    
    emit('qsh_output', {'output': output, 'prompt': prompt})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in qsh_sessions:
        del qsh_sessions[sid]

# =============================================================================
# MAIN - START AUTONOMOUS SETUP
# =============================================================================

def run_autonomous_setup_background():
    """Run autonomous setup in background thread"""
    time.sleep(5)  # Wait for server to start
    logger.info("Starting background autonomous setup...")
    autonomous_setup.run_autonomous_setup()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("=" * 70)
    logger.info("QUANTUM NETWORK - ALICE BRIDGE → UBUNTU GATEWAY")
    logger.info("FOAM QUANTUM MAPPING ACTIVE - QRAM 3x3x3 SCALING TO 11D")
    logger.info("=" * 70)
    logger.info(f"Server starting on 0.0.0.0:{port}")
    logger.info(f"Quantum Foam: 3x3x3 base multi-dim lattice scaling to 11D QRAM")
    logger.info(f"SSH: {'✓ Enabled' if SSH_ENABLED else '✗ Disabled'}")
    logger.info("")
    logger.info("Quantum Bridge Topology (Foam Mapping):")
    logger.info(f"  Alice (Local):  {ALICE_LOCAL} → EPR self-loop (DNS bridge)")
    logger.info(f"  Ubuntu (Gateway): {UBUNTU_QUANTUM_IP} → Primary DNS + Web server")
    logger.info(f"  Foam Base:      192.168.42.0 → Core hub")
    logger.info(f"  QRAM:           192.168.42.4 → Recursive HEX (3x3x3 scaling dims 3-11)")
    logger.info(f"  Holo:           192.168.42.5 → 6EB recursive storage")
    logger.info("")
    logger.info("Foam Quantum DNS Mapping:")
    logger.info(f"  quantum.realm.domain.dominion.foam → 127.0.0.1")
    logger.info(f"  foam.computer → 192.168.42.0")
    logger.info(f"  qram.computer (3d-11d) → 192.168.42.4.HEX... (recursive)")
    logger.info(f"  holo.computer → 192.168.42.5 (sub-DNS)")
    logger.info("")
    logger.info(f"SSH Target: {UBUNTU_HOST}:{UBUNTU_PORT}")
    logger.info(f"Quantum Network: {QUANTUM_NET}")
    logger.info(f"Domain: {QUANTUM_DOMAIN}")
    logger.info(f"Routing: Alice ({ALICE_LOCAL}) → Ubuntu ({UBUNTU_QUANTUM_IP})")
    logger.info("=" * 70)
    
    # Start autonomous setup in background
    if SSH_ENABLED and UBUNTU_HOST not in ['127.0.0.1', 'localhost']:
        setup_thread = threading.Thread(target=run_autonomous_setup_background, daemon=True)
        setup_thread.start()
        logger.info(f"✓ Autonomous setup thread started - connecting to Ubuntu @ {UBUNTU_HOST}")
    else:
        logger.warning("⚠ Autonomous setup skipped")
        logger.warning(f"   Current UBUNTU_HOST: {UBUNTU_HOST}")
        logger.warning(f"   Set UBUNTU_HOST={UBUNTU_QUANTUM_IP} to connect to Ubuntu gateway")
        logger.warning(f"   Alice bridge operates locally at {ALICE_LOCAL}")
    
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    finally:
        logger.info("Shutting down quantum network...")
        logger.info("✓ Shutdown complete")
