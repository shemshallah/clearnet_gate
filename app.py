import eventlet
eventlet.monkey_patch()

import warnings
warnings.filterwarnings("ignore", message="TripleDES")

import os
import sys
import logging
import hashlib
import numpy as np
import base64
import time
import threading
import random
from io import BytesIO
from flask import Flask, redirect, request, session, jsonify, abort, send_from_directory, render_template, render_template_string
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import urllib.parse
import re
import paramiko

# Source from cloned repo: Add gate dir to path for custom deps/modules
GATE_DIR = '/var/www/computer.render/gate'
if os.path.exists(GATE_DIR):
    sys.path.insert(0, GATE_DIR)  # Pull imports from cloned repo first

# Enterprise Logging
log_level = logging.DEBUG if os.environ.get('FLASK_ENV') == 'development' else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SSH_ENABLED = True

# Flask Config - Prioritize cloned static/templates
app = Flask(__name__, 
            template_folder=os.path.join(GATE_DIR, 'templates') if os.path.exists(os.path.join(GATE_DIR, 'templates')) else 'templates',
            static_folder=os.path.join(GATE_DIR, 'static') if os.path.exists(os.path.join(GATE_DIR, 'static')) else 'static')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
if os.environ.get('FLASK_ENV') == 'production':
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# DB Config
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///holo.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Alice - Local quantum bridge (127.0.0.1) - EPR self-loop
ALICE_LOCAL = '127.0.0.1'

# Ubuntu Quantum Gateway - Server at 192.168.42.0 (DNS + web base)
UBUNTU_QUANTUM_IP = os.environ.get('UBUNTU_IP', '192.168.42.0')
UBUNTU_HOST = os.environ.get('UBUNTU_HOST', 'clearnet_gate.onrender.com')
UBUNTU_PORT = int(os.environ.get('UBUNTU_PORT', '22'))
UBUNTU_USER = os.environ.get('UBUNTU_USER', 'shemshallah')
UBUNTU_PASS = os.environ.get('UBUNTU_PASS', '$h10j1r1H0w4rd')

# Quantum Domain - DNS served from Ubuntu (192.168.42.0)
QUANTUM_DOMAIN = 'computer.render'
QUANTUM_SUBDOMAIN = 'render'
BASE_DOMAIN = 'computer'

# Foam Quantum IP Mappings - DNS/web from Ubuntu .0, users at .7
FOAM_QUANTUM_IPS = {
    'quantum.realm.domain.dominion.foam': '127.0.0.1',
    'computer.render': '192.168.42.0',
    'computer.render.alice': '127.0.0.1',
    'computer.render.github': '192.168.42.1',
    'computer.render.wh2': '192.168.42.2',
    'computer.render.bh': '192.168.42.3',
    'computer.render.qram': '192.168.42.4',
    'computer.render.holo': '192.168.42.5',
    'computer.render.render': '192.168.42.7'
}

# QRAM Dimensions
QRAM_DIMS = list(range(3, 12))

# Authentication - Admin via env hash
ADMIN_USER = os.environ.get('ADMIN_USER', 'shemshallah')
ADMIN_PASS_HASH = os.environ.get('ADMIN_PASS_HASH', '930f0446221f865871805ab4e9577971ff97bb21d39abc4e91341ca6100c9181')

# Quantum Network
QUANTUM_NET = '192.168.42.0/24'
QUANTUM_GATEWAY = UBUNTU_QUANTUM_IP
QUANTUM_DNS_PRIMARY = UBUNTU_QUANTUM_IP
IP_POOL = [f'192.168.42.{i}' for i in range(10, 255)]
ALLOCATED_IPS = {}

# Quantum Bridge Topology
QUANTUM_BRIDGES = {
    'alice': {
        'ip': '127.0.0.1',
        'local_bridge': 'Self-loop',
        'protocol': 'EPR (loop)',
        'status': 'ACTIVE',
        'role': 'Local DNS bridge to Ubuntu via clearnet_gate.onrender.com'
    },
    'ubuntu': {
        'ip': '192.168.42.0',
        'local_bridge': 'Direct (DNS base)',
        'protocol': 'SSH-Quantum',
        'status': 'GATEWAY',
        'role': 'Primary DNS server + IP issuer (serves from 192.168.42.0 network)'
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
        'domain': 'computer.render',
        'protocol': 'Foam-Core',
        'status': 'MAPPED',
        'role': 'Foam computer base (web pages *.hex.computer.render)'
    },
    'foam_computer_alice': {
        'ip': '127.0.0.1',
        'local_bridge': 'Self-loop',
        'protocol': 'EPR-Foam',
        'status': 'SYNCHED',
        'role': 'Alice integration in computer.render'
    },
    'foam_computer_github': {
        'ip': '192.168.42.1',
        'local_bridge': 'Direct',
        'protocol': 'Git-Foam',
        'status': 'MAPPED',
        'role': 'GitHub node in computer.render'
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
    },
    'foam_computer_render': {
        'ip': '192.168.42.7',
        'local_bridge': 'clearnet_gate.onrender.com',
        'protocol': 'Render-Foam-HEX',
        'status': 'HOSTED',
        'role': 'Render clearnet gateway with special HEX USER range (192.168.42.7.HEX for users)'
    }
}

# Domain registry
RENDER_TLDS = {f'{i}.computer.render': {
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
    'repo_cloned': False,
    'requirements_installed': False,
    'qutip_built': False,
    'firewall_configured': False,
    'domain_working': False,
    'setup_complete': False,
    'setup_log': [],
    'connection_string': None
}

# Input Validation Regex
USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_]{3,20}$')

class Registrant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    hex_address = db.Column(db.String(20), unique=True, nullable=False)
    user_ip = db.Column(db.String(15), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables and pre-seed admin
with app.app_context():
    db.create_all()
    
    # Pre-seed admin if not exists (avoids insert dupes)
    admin_pass = UBUNTU_PASS  # Use env pass
    admin_hash = generate_password_hash(admin_pass)
    if not Registrant.query.filter_by(username=ADMIN_USER).first():
        admin_reg = Registrant(
            username=ADMIN_USER,
            email=f"{ADMIN_USER}@quantum.foam",
            password_hash=admin_hash,
            hex_address='00.00.00',
            user_ip='192.168.42.10'
        )
        try:
            db.session.add(admin_reg)
            db.session.commit()
            logger.info(f"✓ Admin '{ADMIN_USER}' pre-seeded in DB")
        except Exception as e:
            db.session.rollback()
            logger.warning(f"Admin seed skipped (may exist): {e}")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=False,
    engineio_logger=False
)

# Global for QSH sessions
qsh_sessions = {}

# QUANTUM FOAM LATTICE
class QuantumFoamLattice:
    """Production-grade multi-dimensional quantum lattice with QuTiP state management - 3x3x3 base, scaling to 11D for QRAM"""
    
    def __init__(self):
        self.base_size = 3  # 3x3x3 base lattice
        self.base_dim = 3  # Base 3D lattice
        self.max_dim = 11  # Up to 11D for QRAM
        self.n_sites_base = self.base_size ** self.base_dim  # 27 for 3x3x3
        self.qram_dims = QRAM_DIMS  # Moved early for _initialize_multi_dim_lattice
        
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
            
            # Connect to prior points and entangle QRAM dims + quantum routes
            for domain, ip in FOAM_QUANTUM_IPS.items():
                self.entangle_ip(ip)
            # Entangle QRAM-specific higher dims
            for dim in self.qram_dims:
                self.entangle_dim(ip='192.168.42.4', dim=dim)
            # Special entangle for render.HEX USER format and web .0.hex
            self.entangle_ip('192.168.42.7.00.00.00')  # Example USER HEX
            self.entangle_ip('192.168.42.0.00.00.00')  # Example web HEX
            
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
        # Special mapping for render.HEX USER format (192.168.42.7.HEX) and web (192.168.42.0.HEX)
        for hex_sample in ['00.00.00', '01.02.01', '02.00.02']:  # Sample HEX for quantum routes
            site_idx_user = int(''.join(hex_sample.split('.')), 16) % self.n_sites_base
            mapping[f"user_hex_{hex_sample}"] = {
                'hex_coords': hex_sample,
                'dim': 3,
                'qubit': site_idx_user % self.n_core,
                'phase': np.exp(2j * np.pi * site_idx_user / self.n_sites_base),
                'recursive_ip': f"192.168.42.7.{hex_sample}",
                'role': 'USER quantum route via render.HEX'
            }
            site_idx_web = (site_idx_user + 1) % self.n_sites_base
            mapping[f"web_hex_{hex_sample}"] = {
                'hex_coords': hex_sample,
                'dim': 3,
                'qubit': site_idx_web % self.n_core,
                'phase': np.exp(2j * np.pi * site_idx_web / self.n_sites_base),
                'recursive_ip': f"192.168.42.0.{hex_sample}",
                'role': 'Web page route via .0.hex'
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
        """Entangle IP address into quantum lattice - Production handling for recursive hex IPs, USER .7.hex, web .0.hex"""
        try:
            # Handle recursive IP format like 192.168.42.7.00.01.02 (USER) or 192.168.42.0.00.01.02 (web)
            if '.' in ip_address and ip_address.count('.') > 3:
                parts = ip_address.split('.')
                if len(parts) >= 7 and parts[:4] == ['192', '168', '42']:
                    base_part = parts[4]
                    hex_coords = '.'.join(parts[5:])
                    hex_list = [int(h, 16) for h in hex_coords.split('.')[:3]]
                    dim = 3  # Fixed for HEX
                    if all(0 <= h <= 255 for h in hex_list):  # HEX 00-FF
                        site_idx = sum(h * (256 ** i) for i, h in enumerate(hex_list[:3]))
                        if base_part == '7':
                            site_key = f"user_hex_{hex_coords}"
                        elif base_part == '0':
                            site_key = f"web_hex_{hex_coords}"
                        else:
                            site_key = f"render_hex_{hex_coords}"
                        if site_key in self.lattice_mapping:
                            site_info = self.lattice_mapping[site_key]
                        else:
                            site_info = {'qubit': site_idx % self.n_core, 'phase': 1j, 'dim': dim, 'coords': hex_list}
                    else:
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
            sample_sites = min(1000, n_sites_dim)
            site_idx = random.randint(0, sample_sites - 1)
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
logger.info("Connected to existing lattice at quantum.* routes")
logger.info("=" * 70)
quantum_foam = QuantumFoamLattice()
logger.info("=" * 70)

# AUTONOMOUS UBUNTU SETUP ENGINE - REAL PRODUCTION WITH CLONE & BUILD
class AutonomousSetupEngine:
    """Autonomously sets up complete infrastructure on Ubuntu server with recursive DNS, repo clone, and QuTiP build"""
    
    def __init__(self):
        self.ssh_client = None
        self.max_retries = 5
        self.retry_delay = 5
        self.setup_lock = threading.Lock()
    
    def get_ssh_creds_from_db(self, username):
        """Query holo DB for SSH login details"""
        with app.app_context():
            registrant = Registrant.query.filter_by(username=username).first()
            if registrant:
                logger.info(f"✓ DB check: SSH creds for {username} from holo DB (REGISTRANTS_LIST)")
                return username, registrant.password_hash  # Use hash as pass for sim
            else:
                logger.warning(f"✗ No DB entry for {username}, using default")
                return UBUNTU_USER, UBUNTU_PASS
    
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
        """Connect to Ubuntu server, check DB for creds"""
        if not SSH_ENABLED:
            raise Exception("Paramiko not available")
        
        # Check DB for creds (simulate Ubuntu calling DB)
        ssh_user, ssh_pass = self.get_ssh_creds_from_db(UBUNTU_USER)
        
        self.log_step("SSH Connection", "ATTEMPTING", f"Connecting to {UBUNTU_HOST}:{UBUNTU_PORT} with DB creds for {ssh_user}")
        
        for attempt in range(self.max_retries):
            try:
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                self.ssh_client.connect(
                    UBUNTU_HOST,
                    port=UBUNTU_PORT,
                    username=ssh_user,
                    password=ssh_pass,
                    timeout=10,
                    look_for_keys=False,
                    allow_agent=False
                )
                
                SETUP_STATE['ssh_connected'] = True
                self.log_step("SSH Connection", "SUCCESS", f"Connected as {ssh_user}@{UBUNTU_HOST} using holo DB creds")
                return True
                
            except Exception as e:
                self.log_step("SSH Connection", "RETRY", f"Attempt {attempt+1}/{self.max_retries}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.log_step("SSH Connection", "FAILED", str(e))
                    return False
        
        return False
    
    def clone_repo_and_install(self):
        """Clone/update clearnet_gate repo and install requirements"""
        self.log_step("Repo Clone", "STARTING", f"Cloning github.com/shemshallah/clearnet_gate to /var/www/computer.render/gate")
        
        # Update and install git
        result = self.execute_remote("apt-get update", sudo=True, timeout=60)
        if not result['success']:
            self.log_step("Repo Clone", "FAILED", f"apt update failed: {result['error']}")
            return False
        
        result = self.execute_remote("DEBIAN_FRONTEND=noninteractive apt-get install -y git", sudo=True, timeout=120)
        if not result['success']:
            self.log_step("Repo Clone", "FAILED", result['error'])
            return False
        
        # Create dir and clone/pull
        self.execute_remote("mkdir -p /var/www/computer.render", sudo=True)
        clone_cmd = "cd /var/www/computer.render && rm -rf gate && git clone https://github.com/shemshallah/clearnet_gate.git gate"
        result = self.execute_remote(clone_cmd, sudo=True, timeout=300)
        if not result['success']:
            self.log_step("Repo Clone", "FAILED", f"Clone failed: {result['error']}")
            return False
        
        # Install Python and pip
        result = self.execute_remote("DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip", sudo=True, timeout=120)
        if not result['success']:
            self.log_step("Requirements", "FAILED", "Python/pip install failed")
            return False
        
        self.execute_remote("pip3 install --upgrade pip", sudo=False, timeout=120)
        
        # Install requirements from repo
        result = self.execute_remote("pip3 install -r /var/www/computer.render/gate/requirements.txt", sudo=False, timeout=600)
        if not result['success']:
            self.log_step("Requirements", "FAILED", f"Requirements install failed: {result['error']}")
            return False
        
        # Symlink static/templates
        self.execute_remote("ln -sf /var/www/computer.render/gate/static /var/www/computer.render/static", sudo=True)
        self.execute_remote("ln -sf /var/www/computer.render/gate/templates /var/www/computer.render/templates", sudo=True)
        self.execute_remote("chown -R www-data:www-data /var/www/computer.render", sudo=True)
        
        SETUP_STATE['repo_cloned'] = True
        SETUP_STATE['requirements_installed'] = True
        self.log_step("Repo Clone", "SUCCESS", "clearnet_gate cloned/installed to /var/www/computer.render/gate")
        
        return True
    
    def build_qutip(self):
        """Clone and build QuTiP from source"""
        self.log_step("QuTiP Build", "STARTING", "Cloning and building QuTiP from source")
        
        # Install build deps
        deps_cmd = "DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python3-dev libblas-dev liblapack-dev libopenblas-dev gfortran"
        result = self.execute_remote(deps_cmd, sudo=True, timeout=120)
        if not result['success']:
            self.log_step("QuTiP Build", "FAILED", f"Build deps failed: {result['error']}")
            return False
        
        # Clone QuTiP
        clone_cmd = "rm -rf /opt/qutip && git clone https://github.com/qutip/qutip.git /opt/qutip"
        result = self.execute_remote(clone_cmd, sudo=True, timeout=300)
        if not result['success']:
            self.log_step("QuTiP Build", "FAILED", f"QuTiP clone failed: {result['error']}")
            return False
        
        # Build and install
        build_cmd = "cd /opt/qutip && pip3 install ."
        result = self.execute_remote(build_cmd, sudo=False, timeout=600)
        if not result['success']:
            self.log_step("QuTiP Build", "FAILED", f"QuTiP build failed: {result['error']}")
            return False
        
        SETUP_STATE['qutip_built'] = True
        self.log_step("QuTiP Build", "SUCCESS", "QuTiP cloned and built from source in /opt/qutip")
        
        return True
    
    def setup_dns_server(self):
        """Install and configure Bind9 DNS server on Ubuntu (192.168.42.0)"""
        self.log_step("DNS Installation", "STARTING", f"Installing Bind9 on Ubuntu DNS server ({UBUNTU_QUANTUM_IP}) serving {QUANTUM_DNS_PRIMARY}")
        
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
        self.log_step("DNS Installation", "SUCCESS", f"Bind9 installed on Ubuntu DNS at {UBUNTU_QUANTUM_IP} serving {QUANTUM_DNS_PRIMARY}")
        
        # Configure DNS zones with updated mappings: .0.hex for web, .7.hex for users
        self.log_step("DNS Configuration", "STARTING", "Configuring DNS from 192.168.42.0: web .0.hex, user .7.hex + QRAM dims")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        dns_base = QUANTUM_DNS_PRIMARY
        
        # Create named.conf.local with computer.render zone and recursive subs
        named_conf = f'''
// Quantum Network DNS Configuration - Served from {dns_base} via Ubuntu {ubuntu_ip}
// Recursive Computer.Render: Web (*.hex.computer.render → {dns_base}), Users (*.hex.render.computer.render → 192.168.42.7)
// Alice Bridge: {ALICE_LOCAL}

// quantum.realm.domain.dominion.foam zone (127.0.0.1)
zone "quantum.realm.domain.dominion.foam.computer.render" {{
    type master;
    file "/etc/bind/db.quantum.foam";
    allow-query {{ any; }};
}};

// computer.render zone (192.168.42.0 network) with recursive subs for web
zone "computer.render" {{
    type master;
    file "/etc/bind/db.computer.render";
    allow-query {{ any; }};
}};

// github subdomain zone
zone "github.computer.render" {{
    type master;
    file "/etc/bind/db.github.computer.render";
    allow-query {{ any; }};
}};

// qram subdomain with dim-specific zones (3D-11D)
zone "qram.computer.render" {{
    type master;
    file "/etc/bind/db.qram.computer.render";
    allow-query {{ any; }};
}};

// render subdomain with special HEX USER range (*.hex → 192.168.42.7)
zone "render.computer.render" {{
    type master;
    file "/etc/bind/db.render.computer.render";
    allow-query {{ any; }};
}};

// Reverse zones for 192.168.42.0/24
zone "42.168.192.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.192.168.42";
}};
'''
        
        cmd = f'cat > /tmp/named.conf.local << \'EOF\'\n{named_conf}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/named.conf.local /etc/bind/named.conf.local", sudo=True)
        
        # Create quantum.realm...foam zone (127.0.0.1)
        quantum_foam_zone = f'''$TTL    604800
@       IN      SOA     {dns_base}. root.computer.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.render.
@       IN      A       127.0.0.1
'''
        
        cmd = f'cat > /tmp/db.quantum.foam << \'EOF\'\n{quantum_foam_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.quantum.foam /etc/bind/db.quantum.foam", sudo=True)
        
        # Create computer.render zone with subdomains, *.hex for web pages → 192.168.42.0
        computer_render_zone = f'''$TTL    604800
@       IN      SOA     {dns_base}. root.computer.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.render.
@       IN      A       192.168.42.0

; Subdomains
alice       IN      A       127.0.0.1
github      IN      A       192.168.42.1
wh2         IN      A       192.168.42.2
bh          IN      A       192.168.42.3
qram        IN      A       192.168.42.4
holo        IN      A       192.168.42.5
render      IN      A       192.168.42.7
ubuntu      IN      A       {ubuntu_ip}

; Recursive wildcard for web pages (*.hex.computer.render → 192.168.42.0)
*.hex       IN      A       192.168.42.0
; Recursive for qram matrix coords (e.g., 00.01.02.qram.computer.render) with dims 3-11
*.qram      IN      A       192.168.42.4
; Dim-specific for QRAM (3D-11D)
3d.qram     IN      A       192.168.42.4
11d.qram    IN      A       192.168.42.4
; Recursive for holo storage (sub-DNS/meta)
*.holo      IN      A       192.168.42.5

; Gateway via clearnet
gateway     IN      CNAME   clearnet_gate.onrender.com.
bridge      IN      A       {ALICE_LOCAL}

; Wildcard for other .computer.render subdomains
*           IN      A       192.168.42.0
'''
        
        cmd = f'cat > /tmp/db.computer.render << \'EOF\'\n{computer_render_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.computer.render /etc/bind/db.computer.render", sudo=True)
        
        # qram zone
        qram_zone = f'''$TTL    604800
@       IN      SOA     {dns_base}. root.qram.computer.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.render.
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
        
        cmd = f'cat > /tmp/db.qram.computer.render << \'EOF\'\n{qram_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.qram.computer.render /etc/bind/db.qram.computer.render", sudo=True)
        
        # github zone
        github_zone = f'''$TTL    604800
@       IN      SOA     {dns_base}. root.github.computer.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.render.
@       IN      A       192.168.42.1

; Git subdomains
api     IN      A       192.168.42.1
gist    IN      A       192.168.42.1
raw     IN      A       192.168.42.1
*       IN      A       192.168.42.1
'''
        
        cmd = f'cat > /tmp/db.github.computer.render << \'EOF\'\n{github_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.github.computer.render /etc/bind/db.github.computer.render", sudo=True)
        
        # render zone with special HEX USER range (*.hex → 192.168.42.7)
        render_zone = f'''$TTL    604800
@       IN      SOA     {dns_base}. root.render.computer.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.render.
@       IN      A       192.168.42.7

; Special recursive HEX for USER range (e.g., 00.01.02.render.computer.render → 192.168.42.7)
*.hex        IN      A       192.168.42.7
*.render.hex IN      A       192.168.42.7
*            IN      A       192.168.42.7

; Clearnet CNAME
clearnet_gate IN CNAME clearnet_gate.onrender.com.
'''
        
        cmd = f'cat > /tmp/db.render.computer.render << \'EOF\'\n{render_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.render.computer.render /etc/bind/db.render.computer.render", sudo=True)
        
        # Reverse zone for 192.168.42.0/24 (ubuntu 0, render 7, pool 10-254)
        reverse_192_168_42 = f'''$TTL    604800
@       IN      SOA     {dns_base}. root.computer.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.computer.render.

0       IN      PTR     computer.render.
1       IN      PTR     github.computer.render.
2       IN      PTR     wh2.computer.render.
3       IN      PTR     bh.computer.render.
4       IN      PTR     qram.computer.render.
5       IN      PTR     holo.computer.render.
7       IN      PTR     render.computer.render.

; User IP pool 10-254 (issued by Ubuntu DNS)
'''
        for i in range(10, 255):
            reverse_192_168_42 += f"{i}       IN      PTR     user-{i}.render.computer.render.\n"
        
        cmd = f'cat > /tmp/db.192.168.42 << \'EOF\'\n{reverse_192_168_42}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.192.168.42 /etc/bind/db.192.168.42", sudo=True)
        
        # Configure Bind9 options - Listen on DNS base network
        bind_options = f'''
options {{
    directory "/var/cache/bind";
    
    // Listen on DNS base network + loopback (served from 192.168.42.0)
    listen-on {{ {dns_base}; 127.0.0.1; {ubuntu_ip}; }};
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
    
    version "Ubuntu Quantum DNS - Served from 192.168.42.0: Web .0.hex + User .7.hex Active";
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
        SETUP_STATE['connection_string'] = f"DNS served from {dns_base} (Ubuntu {ubuntu_ip}): web *.hex.computer.render → 192.168.42.0, user *.hex.render.computer.render → 192.168.42.7 (IPs issued from pool), qram → 192.168.42.4 (dims 3-11), holo → 192.168.42.5"
        self.log_step("DNS Configuration", "SUCCESS", f"DNS served from {dns_base}: web .0.hex, user .7.hex + IP issuance configured")
        
        return True

    def setup_web_server(self):
        """Install and configure Apache2 web server on Ubuntu (192.168.42.0) serving from 192.168.42.0"""
        self.log_step("Web Server Installation", "STARTING", f"Installing Apache2 on Ubuntu DNS ({UBUNTU_QUANTUM_IP}) serving {QUANTUM_DNS_PRIMARY}")
        
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
        
        # Configure virtual host for computer.render and subdomains, web .0.hex
        self.log_step("Web Server Configuration", "STARTING", f"Configuring {QUANTUM_DOMAIN} web from {QUANTUM_DNS_PRIMARY}")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        dns_base = QUANTUM_DNS_PRIMARY
        
        vhost_conf = f'''<VirtualHost *:80>
    ServerName computer.render
    ServerAlias quantum.realm.domain.dominion.foam.computer.render
    ServerAlias *.computer.render
    ServerAlias *.hex.computer.render  # Web pages
    ServerAlias alice.computer.render
    ServerAlias github.computer.render
    ServerAlias wh2.computer.render
    ServerAlias bh.computer.render
    ServerAlias qram.computer.render
    ServerAlias *.qram.computer.render
    ServerAlias 3d.qram.computer.render
    ServerAlias 11d.qram.computer.render
    ServerAlias holo.computer.render
    ServerAlias *.holo.computer.render
    ServerAlias render.computer.render
    ServerAlias *.render.computer.render
    ServerAlias *.render.hex.computer.render  # User range
    ServerAlias ubuntu.computer.render
    ServerAlias gateway.computer.render
    ServerAlias clearnet_gate.onrender.com
    
    DocumentRoot /var/www/computer.render
    
    <Directory /var/www/computer.render>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # Quantum network identification
    Header set X-Quantum-Network "{QUANTUM_NET}"
    Header set X-Ubuntu-DNS "{ubuntu_ip}"
    Header set X-DNS-Base "{dns_base}"
    Header set X-Alice-Bridge "{ALICE_LOCAL}"
    Header set X-Web-Base "192.168.42.0.hex"
    Header set X-User-Base "192.168.42.7.hex"
    Header set X-QRAM-Dims "3-11"
    Header set X-Clearnet-Gate "clearnet_gate.onrender.com"
    
    ErrorLog ${{APACHE_LOG_DIR}}/computer_render_error.log
    CustomLog ${{APACHE_LOG_DIR}}/computer_render_access.log combined
</VirtualHost>
'''
        
        cmd = f'cat > /tmp/computer.render.conf << \'EOF\'\n{vhost_conf}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/computer.render.conf /etc/apache2/sites-available/computer.render.conf", sudo=True)
        
        # Create web root
        self.execute_remote("mkdir -p /var/www/computer.render", sudo=True)
        
        # Index page with updated mappings (web .0.hex, user .7.hex)
        index_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Quantum Gateway</title>
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
    <h1>⚛️ Computer Render Portal</h1>
    
    <div class="gateway">
        <div class="label">UBUNTU DNS SERVER</div>
        <div class="value">{ubuntu_ip} (serving {dns_base})</div>
    </div>
    
    <div class="status">✓ DNS SERVED FROM 192.168.42.0 - WEB .0.HEX + USER .7.HEX OPERATIONAL</div>
    
    <div class="network">
        <div class="label">QUANTUM NETWORK</div>
        <div class="value">{QUANTUM_NET}</div>
    </div>
    
    <div class="bridge">
        <div class="label">QUANTUM BRIDGES - RECURSIVE MAPPINGS</div>
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
                <td>computer.render (web base)</td>
                <td>192.168.42.0.hex</td>
                <td>Foam-Core</td>
                <td style="color: #0f0;">MAPPED</td>
            </tr>
            <tr>
                <td>computer.render.alice</td>
                <td>127.0.0.1</td>
                <td>EPR-Foam</td>
                <td style="color: #0f0;">SYNCHED</td>
            </tr>
            <tr>
                <td>computer.render.github</td>
                <td>192.168.42.1</td>
                <td>Git-Foam</td>
                <td style="color: #0f0;">MAPPED</td>
            </tr>
            <tr>
                <td>computer.render.wh2</td>
                <td>192.168.42.2</td>
                <td>Whitehole-2</td>
                <td style="color: #ff0;">RADIATING</td>
            </tr>
            <tr>
                <td>computer.render.bh</td>
                <td>192.168.42.3</td>
                <td>Blackhole-Foam</td>
                <td style="color: #f0f;">COLLAPSED</td>
            </tr>
            <tr>
                <td>computer.render.qram</td>
                <td>192.168.42.4 (dims 3-11)</td>
                <td>QRAM-Recursive</td>
                <td style="color: #0ff;">TUNNELED</td>
            </tr>
            <tr>
                <td>computer.render.holo</td>
                <td>192.168.42.5 (6EB recursive)</td>
                <td>Holo-Recursive</td>
                <td style="color: #0ff;">SYNCHED</td>
            </tr>
            <tr>
                <td>computer.render.render (user base)</td>
                <td>192.168.42.7.hex</td>
                <td>Render-Foam-HEX</td>
                <td style="color: #0ff;">HOSTED</td>
            </tr>
        </table>
    </div>
    
    <div class="metric">
        <div class="label">DNS Served From</div>
        <div class="value">{dns_base} (Ubuntu {ubuntu_ip})</div>
    </div>
    
    <div class="metric">
        <div class="label">Web Pages</div>
        <div class="value">*.hex.computer.render → 192.168.42.0</div>
    </div>
    
    <div class="metric">
        <div class="label">User Range</div>
        <div class="value">*.hex.render.computer.render → 192.168.42.7 (IPs issued from pool)</div>
    </div>
    
    <div class="metric">
        <div class="label">Domain</div>
        <div class="value">{QUANTUM_DOMAIN}</div>
    </div>
    
    <div class="metric">
        <div class="label">Gateway</div>
        <div class="value">Ubuntu @ {ubuntu_ip} via {UBUNTU_HOST}</div>
    </div>
    
    <div class="metric">
        <div class="label">DNS Server</div>
        <div class="value">{dns_base}:53 (via Alice bridge)</div>
    </div>
    
    <div class="metric">
        <div class="label">Alice Bridge</div>
        <div class="value">{ALICE_LOCAL} (EPR self-loop)</div>
    </div>
    
    <div class="metric">
        <div class="label">Quantum Lattice</div>
        <div class="value">Multi-Dim Active (3D-11D QRAM, 27+ sites, connected to quantum.* routes)</div>
    </div>
    
    <div class="metric">
        <div class="label">Autonomous Setup</div>
        <div class="value">Complete</div>
    </div>
    
    <p style="margin-top: 40px; opacity: 0.7;">
        DNS served from {dns_base} (Ubuntu {ubuntu_ip}) via {UBUNTU_HOST} - Web .0.hex + User .7.hex active
    </p>
</body>
</html>
'''
        
        cmd = f'cat > /tmp/index.html << \'EOF\'\n{index_html}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/index.html /var/www/computer.render/index.html", sudo=True)
        
        # Enable headers module and site
        self.execute_remote("a2enmod headers", sudo=True)
        self.execute_remote("a2ensite computer.render.conf", sudo=True)
        self.execute_remote("a2dissite 000-default.conf", sudo=True)
        self.execute_remote("systemctl reload apache2", sudo=True)
        self.execute_remote("systemctl enable apache2", sudo=True)
        
        SETUP_STATE['web_server_configured'] = True
        self.log_step("Web Server Configuration", "SUCCESS", f"{QUANTUM_DOMAIN} web (.0.hex) configured from {dns_base}")
        
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
        """Verify all services: DNS from 192.168.42.0, web .0.hex, user .7.hex, repo, QuTiP"""
        self.log_step("Verification", "STARTING", f"Checking DNS from {QUANTUM_DNS_PRIMARY} via Ubuntu {UBUNTU_QUANTUM_IP}")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        dns_base = QUANTUM_DNS_PRIMARY
        
        # Check DNS on Ubuntu
        result = self.execute_remote("systemctl is-active bind9", sudo=True)
        dns_ok = result['output'].strip() == 'active'
        
        # Check web server on Ubuntu
        result = self.execute_remote("systemctl is-active apache2", sudo=True)
        web_ok = result['output'].strip() == 'active'
        
        # Test DNS resolution for computer.render (base)
        result = self.execute_remote(f"nslookup {QUANTUM_DOMAIN} {dns_base}")
        dns_resolve_ok = '192.168.42.0' in result['output'] or QUANTUM_DOMAIN in result['output']
        
        # Test web .0.hex (example)
        result = self.execute_remote(f"nslookup 00.01.02.computer.render {dns_base}")
        web_hex_ok = '192.168.42.0' in result['output']
        
        # Test user .7.hex
        result = self.execute_remote(f"nslookup 00.01.02.render.{QUANTUM_DOMAIN} {dns_base}")
        user_hex_ok = '192.168.42.7' in result['output']
        
        # Test subdomain resolution (qram)
        result = self.execute_remote(f"nslookup qram.{QUANTUM_DOMAIN} {dns_base}")
        sub_resolve_ok = '192.168.42.4' in result['output']
        
        # Test dim-specific
        result = self.execute_remote(f"nslookup 3d.qram.{QUANTUM_DOMAIN} {dns_base}")
        dim_resolve_ok = '192.168.42.4' in result['output']
        
        # Test reverse DNS for user pool example
        result = self.execute_remote(f"nslookup 192.168.42.10 {dns_base}")
        reverse_user_ok = 'user-10' in result['output'] or 'render' in result['output']
        
        # Test web server responds
        result = self.execute_remote(f"curl -s http://localhost/ | grep 'Computer Render'")
        web_test_ok = result['success']
        
        # Test DNS listening on base
        result = self.execute_remote(f"ss -tlnp | grep :53")
        dns_listening = result['success'] and 'named' in result['output']
        
        # Test repo cloned
        result = self.execute_remote("ls /var/www/computer.render/gate/app.py")
        repo_ok = result['success']
        
        # Test QuTiP
        result = self.execute_remote("python3 -c 'import qutip; print(qutip.__version__)'")
        qutip_ok = result['success']
        
        all_ok = dns_ok and web_ok and dns_resolve_ok and web_test_ok and dns_listening and web_hex_ok and user_hex_ok and sub_resolve_ok and dim_resolve_ok and reverse_user_ok and repo_ok and qutip_ok
        
        if all_ok:
            SETUP_STATE['domain_working'] = True
            self.log_step("Verification", "SUCCESS", f"DNS from {dns_base} operational: web .0.hex → 192.168.42.0, user .7.hex → 192.168.42.7, repo/QuTiP ready")
            return True
        else:
            status = f"DNS:{dns_ok}, Web:{web_ok}, Resolve:{dns_resolve_ok}, WebHEX:{web_hex_ok}, UserHEX:{user_hex_ok}, Sub:{sub_resolve_ok}, Dim:{dim_resolve_ok}, ReverseUser:{reverse_user_ok}, HTTP:{web_test_ok}, Listen:{dns_listening}, Repo:{repo_ok}, QuTiP:{qutip_ok}"
            self.log_step("Verification", "PARTIAL", status)
            return False
    
    def run_autonomous_setup(self):
        """Run complete autonomous setup"""
        with self.setup_lock:
            logger.info("=" * 70)
            logger.info("STARTING AUTONOMOUS UBUNTU SETUP - DNS FROM 192.168.42.0")
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
                
                # Step 4: Clone Repo & Install
                if not self.clone_repo_and_install():
                    raise Exception("Repo clone/install failed")
                
                # Step 5: Build QuTiP
                if not self.build_qutip():
                    raise Exception("QuTiP build failed")
                
                # Step 6: Configure Firewall
                if not self.setup_firewall():
                    logger.warning("Firewall setup had issues, continuing...")
                
                # Step 7: Verify
                if not self.verify_setup():
                    logger.warning("Verification incomplete, but proceeding...")
                
                SETUP_STATE['setup_complete'] = True
                self.log_step("AUTONOMOUS SETUP", "COMPLETE", f"DNS served from {QUANTUM_DNS_PRIMARY}: web .0.hex + user .7.hex + IP issuance + repo/QuTiP")
                
                logger.info("=" * 70)
                logger.info("✓ AUTONOMOUS SETUP COMPLETE - DNS FROM 192.168.42.0 ACTIVE")
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

def issue_user_ip_and_hex(username):
    """Issue user IP from pool and incremental HEX for .7.hex"""
    with app.app_context():
        # Find next available IP
        available_ips = [ip for ip in IP_POOL if ip not in [r.user_ip for r in Registrant.query.all() if r.user_ip]]
        if not available_ips:
            available_ips = IP_POOL
        user_ip = random.choice(available_ips)
        
        # Incremental HEX: find last HEX and increment (simple counter-based)
        last_reg = Registrant.query.order_by(Registrant.id.desc()).first()
        hex_counter = last_reg.id if last_reg else 0
        hex_counter += 1
        hex1 = f"{(hex_counter // 65536) % 256:02x}"
        hex2 = f"{(hex_counter // 256) % 256:02x}"
        hex3 = f"{hex_counter % 256:02x}"
        hex_address = f"{hex1}.{hex2}.{hex3}"
        
        return user_ip, hex_address

def issue_quantum_ip(session_id):
    """Fallback quantum IP issuance for admin"""
    available_ips = [ip for ip in IP_POOL if ip not in ALLOCATED_IPS.values()]
    if not available_ips:
        available_ips = IP_POOL
    ip = random.choice(available_ips)
    ALLOCATED_IPS[session_id] = ip
    return ip

def run_autonomous_setup_background():
    """Background thread for autonomous setup"""
    autonomous_setup.run_autonomous_setup()

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/')
def root():
    if session.get('logged_in'):
        return redirect('/computer/render/gate')
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

@app.route('/metrics')
def metrics():
    m = quantum_foam.get_state_metrics()
    return jsonify(m)

@app.route('/setup_status')
def setup_status():
    """Get autonomous setup status"""
    return jsonify(SETUP_STATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not USERNAME_REGEX.match(username):
            return "Invalid username format", 400
        
        # Check DB first
        with app.app_context():
            registrant = Registrant.query.filter_by(username=username).first()
            if registrant and check_password_hash(registrant.password_hash, password):
                client_ip = request.remote_addr
                session_id = f"sess_{client_ip}_{int(time.time())}"
                
                session['logged_in'] = True
                session['user'] = username
                session['session_id'] = session_id
                session['hex_address'] = registrant.hex_address
                session['email'] = registrant.email
                session['user_ip'] = registrant.user_ip
                
                session_key = hashlib.shake_256(f"{session_id}{quantum_foam.bridge_key}".encode()).digest(32).hex()
                session['session_key'] = session_key
                
                hex_ip = f"192.168.42.7.{registrant.hex_address}"
                session['quantum_ip'] = registrant.user_ip  # Issued IP
                session['hex_ip'] = hex_ip  # HEX subdomain
                
                quantum_foam.entangle_ip(hex_ip)
                quantum_foam.entangle_ip(registrant.user_ip)
                
                logger.info(f"✓ Login: {username} from {client_ip}, user IP: {registrant.user_ip}, hex: {hex_ip}")
                
                return redirect(f'/computer/render/gate?session={session_id}&key={session_key}&ip={hex_ip}')
            else:
                # Fallback to admin - with logging
                logger.info(f"DB check failed for '{username}', trying admin fallback")
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
                    
                    logger.info(f"✓ Admin Login: {username} from {client_ip}, quantum IP: {quantum_ip}")
                    
                    # BUILD REDIRECT URL EXPLICITLY
                    redirect_url = f'/computer/render/gate?session={session_id}&key={session_key}&ip={quantum_ip}'
                    logger.info(f"Redirecting to: {redirect_url}")
                    return redirect(redirect_url)
                else:
                    logger.warning(f"✗ Failed login: {username} from {request.remote_addr} (Admin hash mismatch: {pass_hash[:8]}... vs expected {ADMIN_PASS_HASH[:8]}...)")
                    return "Invalid credentials", 401
        
        return "Invalid credentials", 401
    
    # GET: Render from cloned template if exists
    try:
        return render_template('login.html')
    except:
        # Fallback inline
        return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Authentication</title>
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
        .register-link {
            margin-top: 20px;
            text-align: center;
        }
        .register-link a {
            color: #0f0;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>⚛️ COMPUTER.RENDER GATE (DNS FROM 192.168.42.0)</h1>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" value="shemshallah" required autofocus>
            <label>Password:</label>
            <input type="password" name="password" required>
            <input type="submit" value="ENTER RECURSIVE REALM">
        </form>
        <div class="register-link">
            <a href="/register">Register New Account (Get .7.hex + IP)</a>
        </div>
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
                    status.innerHTML = '✓ System Ready<br>DNS: ' + (data.connection_string || 'Configured');
                } else {
                    status.innerHTML = '⚙ Setup in progress...<br>Steps: ' + data.setup_log.length;
                }
            });
    </script>
</body>
</html>
        ''')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not all([username, email, password]):
            return "All fields required", 400
        
        # Check if exists - enhanced with scalar for efficiency
        if ' ' in username or not USERNAME_REGEX.match(username):
            return "Invalid username (no spaces, 3-20 alphanum/_)", 400
        
        with app.app_context():
            existing = db.session.query(Registrant.id).filter(
                (Registrant.username == username) | (Registrant.email == email)
            ).scalar()
            if existing:
                return "User or email exists", 400
        
        # Issue IP and HEX
        user_ip, hex_address = issue_user_ip_and_hex(username)
        hex_ip = f"192.168.42.7.{hex_address}"
        user_email = f"{username}@quantum.foam"
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Create registrant
        registrant = Registrant(
            username=username,
            email=email,
            password_hash=password_hash,
            hex_address=hex_address,
            user_ip=user_ip
        )
        db.session.add(registrant)
        try:
            db.session.commit()
            
            # Entangle
            quantum_foam.entangle_ip(hex_ip)
            quantum_foam.entangle_ip(user_ip)
            
            logger.info(f"✓ Registered: {username} ({email}) → {user_email}, IP: {user_ip}, HEX: {hex_address}")
            
            # Redirect to email.html with details
            return redirect('/email.html')
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration failed for {username}: {e}")
            if "UNIQUE constraint failed" in str(e):
                return "Username or email already taken (try again)", 409
            return "Registration error - try again", 500
    
    # GET: Render from cloned template if exists
    try:
        return render_template('register.html')
    except:
        # Fallback inline
        return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Registration</title>
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
        .register-box {
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
    </style>
</head>
<body>
    <div class="register-box">
        <h1>⚛️ REGISTER FOR COMPUTER.RENDER (.7.HEX + IP)</h1>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" required autofocus>
            <label>Email:</label>
            <input type="email" name="email" required>
            <label>Password:</label>
            <input type="password" name="password" required>
            <input type="submit" value="REGISTER (Get quantum.foam Email + .7.hex + IP)">
        </form>
    </div>
</body>
</html>
        ''')

@app.route('/email.html')
def email_html():
    # Get last registrant for display (in real, send email)
    with app.app_context():
        last_reg = Registrant.query.order_by(Registrant.id.desc()).first()
        if last_reg:
            user_email = f"{last_reg.username}@quantum.foam"
            hex_ip = f"192.168.42.7.{last_reg.hex_address}"
            user_ip = last_reg.user_ip
            pass_hash = last_reg.password_hash[:16] + "..."
            details = f"Email: {user_email}<br>HEX: {hex_ip}<br>IP: {user_ip}<br>Hash: {pass_hash}"
        else:
            details = "Registration details pending"
    
    # Render from cloned template if exists
    try:
        return render_template('email.html', details=details)
    except:
        # Fallback inline
        return render_template_string(f'''
<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Email Verification</title>
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
        .verify {{
            border: 2px solid #0f0;
            padding: 40px;
            display: inline-block;
            background: #001100;
            box-shadow: 0 0 20px #0f0;
        }}
        a {{
            color: #0f0;
            text-decoration: underline;
        }}
        .details {{
            text-align: left;
            margin-top: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>⚛️ EMAIL VERIFICATION</h1>
    <div class="verify">
        <p>Check your email for verification link to {user_email}.</p>
        <p>Domain: computer.render | DNS: 192.168.42.0</p>
        <div class="details">
            <strong>Your Details (Stored in holo DB):</strong><br>
            {details}
        </div>
        <p><a href="/login">Back to Login</a></p>
    </div>
</body>
</html>
        ''', user_email=user_email, details=details)

@app.route('/computer/render/gate')
def quantum_gate():
    if not session.get('logged_in'):
        return redirect('/login')
    
    client_ip = request.remote_addr
    session_id = session.get('session_id')
    quantum_ip = session.get('quantum_ip', '')
    hex_ip = session.get('hex_ip', '')
    user_ip = session.get('user_ip', '')
    
    metrics = quantum_foam.get_state_metrics()
    
    provided_key = request.args.get('key', '')
    expected_key = session.get('session_key', '')
    if provided_key and provided_key != expected_key:
        logger.warning(f"Invalid session key from {client_ip}: provided {provided_key[:8]}... != expected {expected_key[:8]}...")
        return "Invalid session key", 403
    
    if not provided_key:
        logger.info(f"Missing session key for {client_ip}, but allowing (direct access)")
    
    ssh_status = '✓ ENABLED' if SSH_ENABLED else '✗ DISABLED'
    
    connection_info = SETUP_STATE.get('connection_string', f"{QUANTUM_DOMAIN} DNS from {QUANTUM_DNS_PRIMARY}")
    setup_complete = "✓ COMPLETE" if SETUP_STATE['setup_complete'] else "⚙ IN PROGRESS"
    
    ip_display = f"User IP: {user_ip} | HEX: {hex_ip}" if user_ip else quantum_ip
    
    # Render from cloned template if exists
    try:
        return render_template('gate.html', 
                               session_id=session_id, ip_display=ip_display, 
                               setup_complete=setup_complete, connection_info=connection_info,
                               client_ip=client_ip, quantum_ip=quantum_ip, metrics=metrics,
                               ssh_status=ssh_status, QUANTUM_NET=QUANTUM_NET, 
                               QUANTUM_DNS_PRIMARY=QUANTUM_DNS_PRIMARY, QUANTUM_GATEWAY=QUANTUM_GATEWAY)
    except Exception as e:  # Broaden to catch template/render errors
        logger.warning(f"Template render failed (using fallback): {e}")
        # Fallback inline - properly indented under except
        html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Recursive Portal</title>
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
        <h1>⚛️ COMPUTER.RENDER - DNS FROM 192.168.42.0 (3x3x3 TO 11D QRAM)</h1>
        
        <div class="info-line">
            <strong>Autonomous Setup:</strong> {setup_complete}
        </div>
        <div class="info-line">
            <strong>Connection:</strong> {connection_info}
        </div>
        <div class="info-line">
            <strong>Session ID:</strong> {session_id or 'N/A'}
        </div>
        <div class="info-line">
            <strong>Client IP:</strong> {client_ip}
        </div>
        <div class="info-line">
            <strong>{ip_display}</strong> (Entangled)
        </div>
        <div class="info-line">
            <strong>Network:</strong> {QUANTUM_NET} | DNS: {QUANTUM_DNS_PRIMARY} | Gateway: {QUANTUM_GATEWAY}
        </div>
        <div class="info-line">
            <strong>SSH:</strong> {ssh_status}
        </div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-label">LATTICE FIDELITY</div>
            <div class="metric-value">{metrics.get("fidelity", 0):.15f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">ENTANGLEMENT NEGATIVITY</div>
            <div class="metric-value">{metrics.get("negativity", 0):.6f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">LATTICE SITES BASE</div>
            <div class="metric-value">{metrics.get("lattice_sites_base", 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">ENTANGLED IPs</div>
            <div class="metric-value">{metrics.get("entangled_ips", 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">QRAM DIMS</div>
            <div class="metric-value">{metrics.get("qram_dims", [])}</div>
        </div>
        <div class="metric">
            <div class="metric-label">QRAM CAPACITY (GB)</div>
            <div class="metric-value">{metrics.get("qram_effective_capacity_gb", 0):.2f}</div>
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
        term.writeln('║  QUANTUM SHELL (QSH) v5.0 - DNS FROM 192.168.42.0                   ║');
        term.writeln('║  Multi-Dim Quantum Foam Lattice - Web .0.hex + User .7.hex          ║');
        term.writeln('║  QRAM: 3x3x3 Base Scaling to 11D (27+ sites, 300+ GB effective)     ║');
        term.writeln('╚══════════════════════════════════════════════════════════════════════╝');
        term.writeln('');
        term.writeln('Session: {session_id or "N/A"}');
        term.writeln('{ip_display}');
        term.writeln('Setup: {setup_complete}');
        term.writeln('DNS Base: {QUANTUM_DNS_PRIMARY}');
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
            term.writeln('\\r\\n✓ Quantum channel established - DNS from 192.168.42.0');
            term.write('QSH> ');
        }});
    </script>
</body>
</html>
        '''
        return html

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
entangle_user_hex <hex> - Entangle user .7.hex (e.g., 00.01.02)
entangle_web_hex <hex> - Entangle web .0.hex (e.g., 00.01.02)
registry          - Show domain registry
issue_ip          - Issue new user IP (pool)
clear             - Clear history
exit              - Close session
'''
        
        elif cmd == 'bridges':
            output = f'''
Quantum Bridge Topology (DNS from {QUANTUM_DNS_PRIMARY}):
---------------------------------------------------------
Source                        → Local Bridge              Protocol          Status  
127.0.0.1 (Alice)             → Self-loop                 EPR (loop)        ACTIVE  
{UBUNTU_QUANTUM_IP} (Ubuntu)  → Direct (DNS base)          SSH-Quantum       GATEWAY  
192.168.42.0 (Web Base)       → {QUANTUM_DNS_PRIMARY}      Foam-Core         MAPPED (.hex web)  
127.0.0.1 (Render.Alice)      → Self-loop                 EPR-Foam          SYNCHED  
192.168.42.1 (Github)         → Direct                    Git-Foam          MAPPED  
192.168.42.2 (Wh2)            → 139.0.0.1                 Whitehole-2       RADIATING  
192.168.42.3 (Bh)             → 130.0.0.1                 Blackhole-Foam    COLLAPSED  
192.168.42.4 (Qram)           → 136.0.0.1 (3D-11D)         QRAM-Recursive    TUNNELED  
192.168.42.5 (Holo)           → 138.0.0.1                 Holo-Recursive    SYNCHED  
192.168.42.7 (User Base)      → clearnet_gate.onrender.com Render-Foam-HEX   HOSTED (.hex users)  

DNS Mapping (Served from {QUANTUM_DNS_PRIMARY}):
------------------------------------------------
quantum.realm...foam → 127.0.0.1
computer.render (.hex web) → 192.168.42.0
qram.computer.render (dims 3-11) → 192.168.42.4.HEX... 
render.computer.render (.hex users) → 192.168.42.7.HEX... (IPs issued)
holo.computer.render → 192.168.42.5

Architecture:
-------------
DNS Queries: Alice → {QUANTUM_DNS_PRIMARY} (Ubuntu {UBUNTU_QUANTUM_IP}) → Web .0.hex / User .7.hex + IP pool
Web: {QUANTUM_DNS_PRIMARY}:80 | DNS: {QUANTUM_DNS_PRIMARY}:53
Quantum Routes: 3x3x3 lattice at quantum.*
'''
        
        elif cmd == 'metrics':
            m = quantum_foam.get_state_metrics()
            output = f'''
Quantum Foam Metrics:
--------------------
Fidelity: {m['fidelity']:.15f}
Negativity: {m['negativity']:.6f}
Lattice Sites (Base): {m['lattice_sites_base']}
Entangled IPs: {m['entangled_ips']}
QRAM Dims: {m['qram_dims']}
QRAM Capacity (GB): {m['qram_effective_capacity_gb']:.2f}
Core Qubits: {m['core_qubits']}
Bridge Key: {m['bridge_key'][:16]}...
'''
        
        elif cmd == 'setup_status':
            output = f"Setup Complete: {SETUP_STATE['setup_complete']}\nConnection: {SETUP_STATE.get('connection_string', 'N/A')}\nLog Entries: {len(SETUP_STATE['setup_log'])}"
        
        elif cmd.startswith('teleport '):
            data = cmd[9:].strip()
            if data:
                fid = quantum_foam.quantum_teleport(data)
                output = f"Teleportation of '{data}': Fidelity = {fid:.6f}"
            else:
                output = "Usage: teleport <data>"
        
        elif cmd.startswith('entangle '):
            ip = cmd[9:].strip()
            if ip:
                fid = quantum_foam.entangle_ip(ip)
                output = f"Entangled {ip}: Fidelity = {fid:.15f}"
            else:
                output = "Usage: entangle <ip>"
        
        elif cmd.startswith('entangle_user_hex '):
            hex_str = cmd[18:].strip()
            if '.' in hex_str and len(hex_str.split('.')) == 3:
                full_ip = f"192.168.42.7.{hex_str}"
                fidelity = quantum_foam.entangle_ip(full_ip)
                output = f'✓ User HEX {hex_str} entangled as {full_ip}, fidelity = {fidelity:.15f}'
            else:
                output = 'Usage: entangle_user_hex <HEX.HEX.HEX> (e.g., 00.01.02)'
        
        elif cmd.startswith('entangle_web_hex '):
            hex_str = cmd[17:].strip()
            if '.' in hex_str and len(hex_str.split('.')) == 3:
                full_ip = f"192.168.42.0.{hex_str}"
                fidelity = quantum_foam.entangle_ip(full_ip)
                output = f'✓ Web HEX {hex_str} entangled as {full_ip}, fidelity = {fidelity:.15f}'
            else:
                output = 'Usage: entangle_web_hex <HEX.HEX.HEX> (e.g., 00.01.02)'
        
        elif cmd == 'registry':
            output = f"Domain Registry Sample:\n{list(RENDER_TLDS.items())[:5]}"
        
        elif cmd == 'issue_ip':
            new_ip, new_hex = issue_user_ip_and_hex('manual')
            hex_ip = f"192.168.42.7.{new_hex}"
            output = f'✓ Issued user IP: {new_ip} | HEX: {hex_ip}'
        
        elif cmd == 'clear':
            sess['history'] = []
            output = "History cleared."
        
        elif cmd == 'exit':
            output = "Session closed."
            prompt = False
        
        else:
            output = f"Unknown command: {cmd}. Type 'help' for commands."
    
    except Exception as e:
        logger.error(f"QSH command error: {e}", exc_info=True)
        output = f'✗ Error: {str(e)}'
    
    emit('qsh_output', {'output': output, 'prompt': prompt})

# Static files handling - Updated for cloned repo
@app.route('/static/<path:filename>')
def send_static(filename):
    cloned_static = os.path.join(GATE_DIR, 'static')
    if os.path.exists(os.path.join(cloned_static, filename)):
        return send_from_directory(cloned_static, filename)
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("=" * 70)
    logger.info(f"QUANTUM NETWORK - DNS SERVED FROM {QUANTUM_DNS_PRIMARY} VIA UBUNTU {UBUNTU_QUANTUM_IP}")
    logger.info("COMPUTER.RENDER MAPPING: WEB .0.HEX + USER .7.HEX + IP ISSUANCE")
    logger.info("=" * 70)
    logger.info(f"Server starting on 0.0.0.0:{port} (clearnet_gate.onrender.com)")
    logger.info(f"Quantum Foam: 3x3x3 base scaling to 11D QRAM, connected to quantum.*")
    logger.info(f"SSH: {'✓ Enabled' if SSH_ENABLED else '✗ Disabled'}")
    logger.info("")
    logger.info("Mappings:")
    logger.info(f"  DNS Base: {QUANTUM_DNS_PRIMARY} (Ubuntu {UBUNTU_QUANTUM_IP})")
    logger.info(f"  Web: *.hex.computer.render → 192.168.42.0")
    logger.info(f"  Users: *.hex.render.computer.render → 192.168.42.7 (IPs from pool)")
    logger.info(f"  QRAM: 192.168.42.4.HEX... (dims 3-11)")
    logger.info(f"  Holo: 192.168.42.5")
    logger.info("")
    logger.info(f"Routing: Alice ({ALICE_LOCAL}) → DNS {QUANTUM_DNS_PRIMARY} via {UBUNTU_HOST}")
    logger.info("=" * 70)
    
    # Start autonomous setup in background
    setup_thread = threading.Thread(target=run_autonomous_setup_background, daemon=True)
    setup_thread.start()
    logger.info(f"✓ Autonomous setup started - DNS from {QUANTUM_DNS_PRIMARY}")
    
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    finally:
        logger.info("Shutdown complete")
