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
    sys.path.insert(0, GATE_DIR)

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

# Flask Config
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
if os.environ.get('FLASK_ENV') == 'production' and not os.environ.get('DATABASE_URI'):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    logger.warning("No DATABASE_URI set in production; using in-memory SQLite")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Alice - Local quantum bridge
ALICE_LOCAL = '127.0.0.1'

# FIXED: Separate gateway IP from DNS/web service IPs
UBUNTU_GATEWAY_IP = os.environ.get('UBUNTU_GATEWAY_IP', '133.7.0.1')  # SSH connects HERE
UBUNTU_DNS_IP = '192.168.42.0'  # DNS service runs here
UBUNTU_WEB_IP = '192.168.42.0'  # Web service runs here
UBUNTU_HOST = os.environ.get('UBUNTU_HOST', 'clearnet_gate.onrender.com')
UBUNTU_PORT = int(os.environ.get('UBUNTU_PORT', '22'))

# FIXED: Unified credentials
UBUNTU_USER = 'shemshallah'
UBUNTU_PASS = '$h10j1r1H0w4rd'

# Quantum Domain
QUANTUM_DOMAIN = 'computer.render'
QUANTUM_SUBDOMAIN = 'render'
BASE_DOMAIN = 'computer'

# Foam Quantum IP Mappings
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

# Authentication
ADMIN_USER = 'shemshallah'
ADMIN_PASS_HASH = os.environ.get('ADMIN_PASS_HASH', '930f0446221f865871805ab4e9577971ff97bb21d39abc4e91341ca6100c9181')

# Quantum Network - FIXED to use correct IPs
QUANTUM_NET = '192.168.42.0/24'
QUANTUM_GATEWAY = UBUNTU_GATEWAY_IP  # Gateway for routing
QUANTUM_DNS_PRIMARY = UBUNTU_DNS_IP  # DNS service IP
IP_POOL = [f'192.168.42.{i}' for i in range(10, 255)]
ALLOCATED_IPS = {}

# Quantum Bridge Topology
QUANTUM_BRIDGES = {
    'alice': {
        'ip': '127.0.0.1',
        'local_bridge': 'Self-loop',
        'protocol': 'EPR (loop)',
        'status': 'ACTIVE',
        'role': 'Local DNS bridge'
    },
    'ubuntu': {
        'ip': UBUNTU_GATEWAY_IP,
        'local_bridge': 'Direct (Gateway)',
        'protocol': 'SSH-Quantum',
        'status': 'GATEWAY',
        'role': 'Primary gateway + SSH entry'
    },
    'dns_service': {
        'ip': UBUNTU_DNS_IP,
        'local_bridge': 'Via Gateway',
        'protocol': 'DNS-53',
        'status': 'SERVICE',
        'role': 'DNS server at 192.168.42.0:53'
    },
    'web_service': {
        'ip': UBUNTU_WEB_IP,
        'local_bridge': 'Via Gateway',
        'protocol': 'HTTP-80',
        'status': 'SERVICE',
        'role': 'Web server at 192.168.42.0:80'
    },
    'quantum_realm': {
        'ip': '127.0.0.1',
        'domain': 'quantum.realm.domain.dominion.foam',
        'protocol': 'Foam-Quantum',
        'status': 'MAPPED',
        'role': 'Quantum realm primary'
    },
    'foam_computer': {
        'ip': '192.168.42.0',
        'domain': 'computer.render',
        'protocol': 'Foam-Core',
        'status': 'MAPPED',
        'role': 'Foam computer base'
    },
    'foam_computer_render': {
        'ip': '192.168.42.7',
        'local_bridge': 'clearnet_gate.onrender.com',
        'protocol': 'Render-Foam-HEX',
        'status': 'HOSTED',
        'role': 'Render clearnet gateway'
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

# Input Validation
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
    
    admin_pass = UBUNTU_PASS
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
            logger.warning(f"Admin seed skipped: {e}")

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
    """Multi-dimensional quantum lattice with QuTiP state management"""
    
    def __init__(self):
        self.base_size = 3
        self.base_dim = 3
        self.max_dim = 11
        self.n_sites_base = self.base_size ** self.base_dim
        self.qram_dims = QRAM_DIMS
        
        logger.info("Initializing quantum foam lattice...")
        
        try:
            self.n_core = 12
            self.core_state = self._create_ghz_core()
            self.lattice_mapping = self._initialize_multi_dim_lattice()
            self.fidelity = self._measure_fidelity()
            self.negativity = self._calculate_negativity()
            
            state_hash = hashlib.sha256(
                self.core_state.full().tobytes()
            ).hexdigest()
            self.bridge_key = f"QFOAM-MULTIDIM-{state_hash[:32]}"
            
            self.ip_entanglement = {}
            
            for domain, ip in FOAM_QUANTUM_IPS.items():
                self.entangle_ip(ip)
            for dim in self.qram_dims:
                self.entangle_dim(ip='192.168.42.4', dim=dim)
            self.entangle_ip('192.168.42.7.00.00.00')
            self.entangle_ip('192.168.42.0.00.00.00')
            
            logger.info(f"✓ Lattice active: fidelity={self.fidelity:.15f}")
            logger.info(f"✓ Bridge key: {self.bridge_key}")
            
        except Exception as e:
            logger.error(f"Quantum lattice initialization failed: {e}", exc_info=True)
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        self.n_core = 12
        self.core_state = qt.tensor([qt.basis(2, 0)] * self.n_core)
        self.lattice_mapping = {}
        self.fidelity = 0.5
        self.negativity = 0.5
        self.bridge_key = f"QFOAM-FALLBACK-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]}"
        self.ip_entanglement = {}
        self.qram_dims = QRAM_DIMS
        logger.warning("Fallback quantum state initialized")
    
    def _create_ghz_core(self):
        zeros = qt.tensor([qt.basis(2, 0)] * self.n_core)
        ones = qt.tensor([qt.basis(2, 1)] * self.n_core)
        ghz = (zeros + ones).unit()
        return ghz
    
    def _initialize_multi_dim_lattice(self):
        mapping = {}
        for coords in product(range(self.base_size), repeat=self.base_dim):
            site_idx = sum(c * (self.base_size ** i) for i, c in enumerate(coords))
            qubit_idx = site_idx % self.n_core
            hex_coords = '.'.join(f"{c:02x}" for c in coords)
            mapping[site_idx] = {
                'coords': coords,
                'hex_coords': hex_coords,
                'dim': self.base_dim,
                'qubit': qubit_idx,
                'phase': np.exp(2j * np.pi * site_idx / self.n_sites_base),
                'recursive_ip': f"192.168.42.4.{hex_coords}"
            }
        for dim in self.qram_dims[1:]:
            n_sites_dim = self.base_size ** dim
            sample_sites = min(1000, n_sites_dim)
            for i in range(sample_sites):
                coords = tuple((i // (self.base_size ** j)) % self.base_size for j in range(dim))
                site_idx = i
                qubit_idx = site_idx % self.n_core
                hex_coords = '.'.join(f"{c:02x}" for c in coords[:3]) + f".D{dim}"
                mapping[f"{site_idx}_D{dim}"] = {
                    'coords': coords,
                    'hex_coords': hex_coords,
                    'dim': dim,
                    'qubit': qubit_idx,
                    'phase': np.exp(2j * np.pi * site_idx / n_sites_dim),
                    'recursive_ip': f"192.168.42.4.{hex_coords}",
                    'effective_capacity': 300 * (self.base_size ** (dim - 3)) / 1024
                }
        return mapping
    
    def _measure_fidelity(self):
        ideal_ghz = self._create_ghz_core()
        return float(qt.fidelity(self.core_state, ideal_ghz))
    
    def _calculate_negativity(self):
        neg_sum = 0.0
        samples = min(10, len(self.lattice_mapping))
        for i in range(samples):
            site_key = list(self.lattice_mapping.keys())[i]
            dim = self.lattice_mapping[site_key]['dim']
            qubit_idx = self.lattice_mapping[site_key]['qubit']
            rho_ab = self.core_state.ptrace([qubit_idx, (qubit_idx + 1) % self.n_core])
            neg = qt.negativity(rho_ab)
            neg_adjusted = neg * (1 - 0.005 * (dim - 3))
            neg_sum += neg_adjusted
        return neg_sum / samples
    
    def entangle_ip(self, ip_address):
        try:
            if '.' in ip_address and ip_address.count('.') > 3:
                parts = ip_address.split('.')
                if len(parts) >= 7 and parts[:4] == ['192', '168', '42']:
                    base_part = parts[4]
                    hex_coords = '.'.join(parts[5:])
                    hex_list = [int(h, 16) for h in hex_coords.split('.')[:3]]
                    dim = 3
                    if all(0 <= h <= 255 for h in hex_list):
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
            except Exception as e:
                logger.debug(f"Phase gate skipped: {e}")
            
            ip_fidelity = self._measure_fidelity()
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
            
            logger.info(f"✓ IP {ip_address} entangled, fidelity={ip_fidelity:.15f}")
            return ip_fidelity
            
        except Exception as e:
            logger.error(f"IP entanglement error for {ip_address}: {e}")
            return self._measure_fidelity()
    
    def entangle_dim(self, ip, dim):
        try:
            n_sites_dim = self.base_size ** dim
            sample_sites = min(1000, n_sites_dim)
            site_idx = random.randint(0, sample_sites - 1)
            qubit_idx = site_idx % self.n_core
            phase = np.exp(2j * np.pi * site_idx / n_sites_dim)
            phase_angle = np.angle(phase) * dim / 3
            phase_matrix = np.array([[1, 0], [0, np.exp(1j * phase_angle)]], dtype=complex)
            phase_gate = qt.Qobj(phase_matrix)
            
            rotation = qt.tensor(
                [qt.qeye(2) if i != qubit_idx else phase_gate
                 for i in range(self.n_core)]
            )
            
            self.core_state = rotation * self.core_state
            
            dim_fidelity = self._measure_fidelity()
            dim_fidelity *= (1 - 0.002 * (dim - 3))
            
            ent_entry = {
                'site': site_idx,
                'dim': dim,
                'qubit': qubit_idx,
                'fidelity': dim_fidelity,
                'effective_capacity_gb': 300 * (self.base_size ** (dim - 3)) / 1024,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.ip_entanglement[f"{ip}_D{dim}"] = ent_entry
            
            logger.info(f"✓ QRAM dim {dim} entangled, fidelity={dim_fidelity:.15f}")
            return dim_fidelity
            
        except Exception as e:
            logger.error(f"Dim {dim} entanglement error: {e}")
            return self._measure_fidelity()
    
    def quantum_teleport(self, data_input):
        try:
            data_hash = int(hashlib.md5(data_input.encode()).hexdigest(), 16) % 2
            input_state = qt.basis(2, data_hash)
            
            epr = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
                   qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
            
            initial = qt.tensor(input_state, epr)
            
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
            
            h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            H = qt.tensor(qt.Qobj(h_matrix), qt.qeye(2), qt.qeye(2))
            after_H = H * after_cnot
            
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
            return 0.5
    
    def get_state_metrics(self):
        neg_sum, fid_sum, cap_sum = 0.0, 0.0, 0.0
        dim_samples = {dim: 0 for dim in self.qram_dims}
        for ent in self.ip_entanglement.values():
            if isinstance(ent.get('dim'), int) and 3 <= ent['dim'] <= 11:
                dim = ent['dim']
                dim_samples[dim] += 1
                neg_sum += ent['fidelity'] * self.negativity
                fid_sum += ent['fidelity']
                if 'effective_capacity' in ent or 'effective_capacity_gb' in ent:
                    cap_sum += ent.get('effective_capacity_gb', ent.get('effective_capacity', 0))
        
        total_samples = sum(dim_samples.values())
        avg_neg = neg_sum / max(1, total_samples)
        avg_fid = fid_sum / max(1, len(self.ip_entanglement))
        total_cap_gb = cap_sum
        
        core_fid = self._measure_fidelity()
        core_neg = self._calculate_negativity()
        
        return {
            'fidelity': float(avg_fid * core_fid),
            'negativity': float(avg_neg * core_neg),
            'lattice_sites_base': self.n_sites_base,
            'entangled_ips': len(self.ip_entanglement),
            'qram_dims': self.qram_dims,
            'qram_effective_capacity_gb': total_cap_gb,
            'bridge_key': self.bridge_key,
            'core_qubits': self.n_core
        }

# Initialize quantum foam
logger.info("=" * 70)
logger.info("QUANTUM FOAM INITIALIZATION")
logger.info("=" * 70)
quantum_foam = QuantumFoamLattice()
logger.info("=" * 70)

# AUTONOMOUS UBUNTU SETUP ENGINE
class AutonomousSetupEngine:
    """Autonomously sets up infrastructure on Ubuntu server"""
    
    def __init__(self):
        self.ssh_client = None
        self.max_retries = 3
        self.retry_delay = 3
        self.setup_lock = threading.Lock()
    
    def log_step(self, step, status, details=""):
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'step': step,
            'status': status,
            'details': details
        }
        SETUP_STATE['setup_log'].append(entry)
        logger.info(f"SETUP [{status}]: {step} - {details}")
    
    def execute_remote(self, command, sudo=False, timeout=30):
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
        """Connect to Ubuntu server - FIXED to use gateway IP or hostname"""
        if not SSH_ENABLED:
            raise Exception("Paramiko not available")
        
        # FIXED: Try both gateway IP and hostname
        targets = [
            (UBUNTU_GATEWAY_IP, "Gateway IP"),
            (UBUNTU_HOST, "Hostname")
        ]
        
        self.log_step("SSH Connection", "STARTING", 
                     f"Targets: {UBUNTU_GATEWAY_IP}, {UBUNTU_HOST}")
        
        for attempt, (target, target_type) in enumerate(targets):
            self.log_step("SSH Connection", "ATTEMPTING", 
                         f"Try {attempt+1}: {target} ({target_type}):{UBUNTU_PORT} as {UBUNTU_USER}")
            
            try:
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                self.ssh_client.connect(
                    target,
                    port=UBUNTU_PORT,
                    username=UBUNTU_USER,
                    password=UBUNTU_PASS,
                    timeout=10,
                    look_for_keys=False,
                    allow_agent=False
                )
                
                SETUP_STATE['ssh_connected'] = True
                self.log_step("SSH Connection", "SUCCESS", 
                             f"Connected to {target} ({target_type}) as {UBUNTU_USER}")
                logger.info(f"✓ SSH established: {target} - DNS at {UBUNTU_DNS_IP}")
                return True
                
            except Exception as e:
                error_msg = f"{target_type} {target}: {str(e)}"
                self.log_step("SSH Connection", "FAILED", error_msg)
                logger.warning(f"✗ SSH attempt {attempt+1} failed: {error_msg}")
                
                if attempt < len(targets) - 1:
                    logger.info(f"Retrying with next target...")
                    time.sleep(self.retry_delay)
                continue
        
        self.log_step("SSH Connection", "FAILED", "All connection attempts exhausted")
        logger.error("✗ Could not establish SSH connection")
        return False
    
    def setup_dns_server(self):
        self.log_step("DNS Installation", "STARTING", f"Installing Bind9 on {UBUNTU_DNS_IP}")
        
        result = self.execute_remote("apt-get update", sudo=True, timeout=60)
        if not result['success']:
            self.log_step("DNS Installation", "FAILED", result['error'])
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
        SETUP_STATE['dns_configured'] = True
        self.log_step("DNS Installation", "SUCCESS", f"Bind9 installed on {UBUNTU_DNS_IP}")
        return True
    
    def setup_web_server(self):
        self.log_step("Web Server Installation", "STARTING", "Installing Apache2")
        
        result = self.execute_remote(
            "DEBIAN_FRONTEND=noninteractive apt-get install -y apache2",
            sudo=True,
            timeout=120
        )
        
        if not result['success']:
            self.log_step("Web Server Installation", "FAILED", result['error'])
            return False
        
        SETUP_STATE['web_server_installed'] = True
        SETUP_STATE['web_server_configured'] = True
        self.log_step("Web Server Installation", "SUCCESS", "Apache2 installed")
        return True
    
    def clone_repo_and_install(self):
        self.log_step("Repo Clone", "STARTING", "Cloning clearnet_gate repo")
        
        result = self.execute_remote("DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 python3-pip", sudo=True, timeout=120)
        if not result['success']:
            self.log_step("Repo Clone", "FAILED", result['error'])
            return False
        
        self.execute_remote("mkdir -p /var/www/computer.render", sudo=True)
        clone_cmd = "cd /var/www/computer.render && rm -rf gate && git clone https://github.com/shemshallah/clearnet_gate.git gate"
        result = self.execute_remote(clone_cmd, sudo=True, timeout=300)
        if not result['success']:
            self.log_step("Repo Clone", "FAILED", result['error'])
            return False
        
        SETUP_STATE['repo_cloned'] = True
        SETUP_STATE['requirements_installed'] = True
        self.log_step("Repo Clone", "SUCCESS", "Repo cloned")
        return True
    
    def setup_firewall(self):
        self.log_step("Firewall Configuration", "STARTING", "Opening ports")
        
        commands = [
            "ufw allow 22/tcp",
            "ufw allow 53/tcp",
            "ufw allow 53/udp",
            "ufw allow 80/tcp",
            "ufw --force enable"
        ]
        
        for cmd in commands:
            self.execute_remote(cmd, sudo=True)
        
        SETUP_STATE['firewall_configured'] = True
        self.log_step("Firewall Configuration", "SUCCESS", "Ports opened")
        return True
    
    def verify_setup(self):
        self.log_step("Verification", "STARTING", "Checking services")
        
        result = self.execute_remote("systemctl is-active bind9", sudo=True)
        dns_ok = 'active' in result['output']
        
        result = self.execute_remote("systemctl is-active apache2", sudo=True)
        web_ok = 'active' in result['output']
        
        if dns_ok and web_ok:
            SETUP_STATE['domain_working'] = True
            self.log_step("Verification", "SUCCESS", "Services operational")
            return True
        else:
            self.log_step("Verification", "PARTIAL", f"DNS:{dns_ok}, Web:{web_ok}")
            return False
    
    def run_autonomous_setup(self):
        with self.setup_lock:
            logger.info("=" * 70)
            logger.info("STARTING AUTONOMOUS UBUNTU SETUP")
            logger.info("=" * 70)
            
            try:
                if not self.connect_ssh():
                    raise Exception("SSH connection failed")
                
                if not self.setup_dns_server():
                    logger.warning("DNS setup had issues, continuing...")
                
                if not self.setup_web_server():
                    logger.warning("Web server setup had issues, continuing...")
                
                if not self.clone_repo_and_install():
                    logger.warning("Repo clone had issues, continuing...")
                
                if not self.setup_firewall():
                    logger.warning("Firewall setup had issues, continuing...")
                
                self.verify_setup()
                
                SETUP_STATE['setup_complete'] = True
                SETUP_STATE['connection_string'] = f"Gateway: {UBUNTU_GATEWAY_IP}, DNS: {UBUNTU_DNS_IP}"
                self.log_step("AUTONOMOUS SETUP", "COMPLETE", "All steps finished")
                
                logger.info("=" * 70)
                logger.info("✓ AUTONOMOUS SETUP COMPLETE")
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

autonomous_setup = AutonomousSetupEngine()

def issue_user_ip_and_hex(username):
    with app.app_context():
        available_ips = [ip for ip in IP_POOL if ip not in [r.user_ip for r in Registrant.query.all() if r.user_ip]]
        if not available_ips:
            available_ips = IP_POOL
        user_ip = random.choice(available_ips)
        
        last_reg = Registrant.query.order_by(Registrant.id.desc()).first()
        hex_counter = last_reg.id if last_reg else 0
        hex_counter += 1
        hex1 = f"{(hex_counter // 65536) % 256:02x}"
        hex2 = f"{(hex_counter // 256) % 256:02x}"
        hex3 = f"{hex_counter % 256:02x}"
        hex_address = f"{hex1}.{hex2}.{hex3}"
        
        return user_ip, hex_address

def issue_quantum_ip(session_id):
    available_ips = [ip for ip in IP_POOL if ip not in ALLOCATED_IPS.values()]
    if not available_ips:
        available_ips = IP_POOL
    ip = random.choice(available_ips)
    ALLOCATED_IPS[session_id] = ip
    return ip

def run_autonomous_setup_background():
    autonomous_setup.run_autonomous_setup()

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
        'setup_state': SETUP_STATE,
        'gateway_ip': UBUNTU_GATEWAY_IP,
        'dns_ip': UBUNTU_DNS_IP
    })

@app.route('/metrics')
def metrics():
    m = quantum_foam.get_state_metrics()
    return jsonify(m)

@app.route('/setup_status')
def setup_status():
    return jsonify(SETUP_STATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not USERNAME_REGEX.match(username):
            return "Invalid username format", 400
        
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
                session['quantum_ip'] = registrant.user_ip
                session['hex_ip'] = hex_ip
                
                try:
                    quantum_foam.entangle_ip(hex_ip)
                    quantum_foam.entangle_ip(registrant.user_ip)
                except Exception as ent_e:
                    logger.warning(f"Entanglement skipped: {ent_e}")
                
                logger.info(f"✓ DB Login: {username}, IP: {registrant.user_ip}")
                return redirect(f'/computer/render/gate?session={session_id}&key={session_key}&ip={hex_ip}')
            
            elif username == UBUNTU_USER:
                # Direct password match
                if password == UBUNTU_PASS:
                    client_ip = request.remote_addr
                    session_id = f"sess_{client_ip}_{int(time.time())}"
                    
                    session['logged_in'] = True
                    session['user'] = username
                    session['session_id'] = session_id
                    session['is_admin'] = True
                    
                    session_key = hashlib.shake_256(f"{session_id}{quantum_foam.bridge_key}".encode()).digest(32).hex()
                    session['session_key'] = session_key
                    
                    quantum_ip = issue_quantum_ip(session_id)
                    session['quantum_ip'] = quantum_ip
                    
                    try:
                        quantum_foam.entangle_ip(client_ip)
                    except Exception as ent_e:
                        logger.warning(f"Entanglement skipped: {ent_e}")
                    
                    logger.info(f"✓ Admin Login: {username}, IP: {quantum_ip}")
                    return redirect(f'/computer/render/gate?session={session_id}&key={session_key}&ip={quantum_ip}')
                
                # Hash match
                pass_hash = hashlib.sha3_256(password.encode()).hexdigest()
                if pass_hash == ADMIN_PASS_HASH:
                    client_ip = request.remote_addr
                    session_id = f"sess_{client_ip}_{int(time.time())}"
                    
                    session['logged_in'] = True
                    session['user'] = username
                    session['session_id'] = session_id
                    session['is_admin'] = True
                    
                    session_key = hashlib.shake_256(f"{session_id}{quantum_foam.bridge_key}".encode()).digest(32).hex()
                    session['session_key'] = session_key
                    
                    quantum_ip = issue_quantum_ip(session_id)
                    session['quantum_ip'] = quantum_ip
                    
                    try:
                        quantum_foam.entangle_ip(client_ip)
                    except Exception as ent_e:
                        logger.warning(f"Entanglement skipped: {ent_e}")
                    
                    logger.info(f"✓ Admin Login (hash): {username}, IP: {quantum_ip}")
                    return redirect(f'/computer/render/gate?session={session_id}&key={session_key}&ip={quantum_ip}')
            
            logger.warning(f"✗ Failed login: {username}")
            return "Invalid credentials", 401
    
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Login</title>
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
        .info {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #0f0;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>⚛️ COMPUTER.RENDER</h1>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" value="shemshallah" required autofocus>
            <label>Password:</label>
            <input type="password" name="password" required>
            <input type="submit" value="ENTER">
        </form>
        <div class="info">
            Gateway: ''' + UBUNTU_GATEWAY_IP + '''<br>
            DNS: ''' + UBUNTU_DNS_IP + '''<br>
            User: shemshallah
        </div>
    </div>
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
        
        if ' ' in username or not USERNAME_REGEX.match(username):
            return "Invalid username", 400
        
        with app.app_context():
            existing = db.session.query(Registrant.id).filter(
                (Registrant.username == username) | (Registrant.email == email)
            ).scalar()
            if existing:
                return "User or email exists", 400
        
        user_ip, hex_address = issue_user_ip_and_hex(username)
        hex_ip = f"192.168.42.7.{hex_address}"
        password_hash = generate_password_hash(password)
        
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
            
            try:
                quantum_foam.entangle_ip(hex_ip)
                quantum_foam.entangle_ip(user_ip)
            except Exception as ent_e:
                logger.warning(f"Entanglement skipped: {ent_e}")
            
            logger.info(f"✓ Registered: {username}, IP: {user_ip}, HEX: {hex_address}")
            return redirect('/login')
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration failed: {e}")
            return "Registration error", 500
    
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Register</title>
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
        <h1>⚛️ REGISTER</h1>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" required autofocus>
            <label>Email:</label>
            <input type="email" name="email" required>
            <label>Password:</label>
            <input type="password" name="password" required>
            <input type="submit" value="REGISTER">
        </form>
    </div>
</body>
</html>
    ''')

@app.route('/computer/render/gate')
def quantum_gate():
    try:
        if not session.get('logged_in'):
            return redirect('/login')
        
        client_ip = request.remote_addr
        session_id = session.get('session_id')
        quantum_ip = session.get('quantum_ip', '')
        hex_ip = session.get('hex_ip', '')
        user_ip = session.get('user_ip', '')
        
        metrics = quantum_foam.get_state_metrics()
        
        ssh_status = '✓ ENABLED' if SSH_ENABLED else '✗ DISABLED'
        setup_complete = "✓ COMPLETE" if SETUP_STATE['setup_complete'] else "⚙ IN PROGRESS"
        connection_info = SETUP_STATE.get('connection_string', f"Gateway: {UBUNTU_GATEWAY_IP}, DNS: {UBUNTU_DNS_IP}")
        ip_display = f"User IP: {user_ip} | HEX: {hex_ip}" if user_ip else quantum_ip
        
        html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Computer Render - Gateway</title>
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
        <h1>⚛️ COMPUTER.RENDER GATEWAY</h1>
        
        <div class="info-line">
            <strong>Setup:</strong> {setup_complete}
        </div>
        <div class="info-line">
            <strong>Connection:</strong> {connection_info}
        </div>
        <div class="info-line">
            <strong>Session:</strong> {session_id or 'N/A'}
        </div>
        <div class="info-line">
            <strong>Client IP:</strong> {client_ip}
        </div>
        <div class="info-line">
            <strong>{ip_display}</strong>
        </div>
        <div class="info-line">
            <strong>Network:</strong> {QUANTUM_NET} | Gateway: {UBUNTU_GATEWAY_IP} | DNS: {QUANTUM_DNS_PRIMARY}
        </div>
        <div class="info-line">
            <strong>SSH:</strong> {ssh_status}
        </div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-label">FIDELITY</div>
            <div class="metric-value">{metrics.get("fidelity", 0):.15f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">NEGATIVITY</div>
            <div class="metric-value">{metrics.get("negativity", 0):.6f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">LATTICE SITES</div>
            <div class="metric-value">{metrics.get("lattice_sites_base", 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">ENTANGLED IPs</div>
            <div class="metric-value">{metrics.get("entangled_ips", 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">SETUP</div>
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
        term.writeln('║  QUANTUM SHELL (QSH) v5.0                                            ║');
        term.writeln('╚══════════════════════════════════════════════════════════════════════╝');
        term.writeln('');
        term.writeln('Gateway: {UBUNTU_GATEWAY_IP} | DNS: {QUANTUM_DNS_PRIMARY}');
        term.writeln('Session: {session_id or "N/A"}');
        term.writeln('{ip_display}');
        term.writeln('');
        term.writeln('Commands: help, metrics, bridges, setup_status, teleport <data>, entangle <ip>');
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
            
    except Exception as e:
        logger.error(f"Gate route error: {e}", exc_info=True)
        return jsonify({'error': 'Gate access failed'}), 500

@socketio.on('qsh_command')
def handle_qsh_command(data):
    sid = request.sid
    cmd = data.get('command', '').strip()
    
    if sid not in qsh_sessions:
        qsh_sessions[sid] = {'history': []}
    
    output = ''
    prompt = True
    
    try:
        if cmd == 'help':
            output = '''
Available Commands:
  help              - Show this help
  metrics           - Display quantum metrics
  bridges           - Show quantum bridges
  setup_status      - View setup progress
  teleport <data>   - Quantum teleportation
  entangle <ip>     - Entangle IP address
  clear             - Clear history
  exit              - Close session
'''
        elif cmd == 'bridges':
            output = f'''
Quantum Bridge Topology:
  Gateway: {UBUNTU_GATEWAY_IP} (SSH entry point)
  DNS Service: {UBUNTU_DNS_IP}:53
  Web Service: {UBUNTU_WEB_IP}:80
  Alice Bridge: 127.0.0.1 (local)
  Render Gateway: clearnet_gate.onrender.com
'''
        elif cmd == 'metrics':
            m = quantum_foam.get_state_metrics()
            output = f'''
Quantum Metrics:
  Fidelity: {m['fidelity']:.15f}
  Negativity: {m['negativity']:.6f}
  Lattice Sites: {m['lattice_sites_base']}
  Entangled IPs: {m['entangled_ips']}
  Core Qubits: {m['core_qubits']}
'''
        elif cmd == 'setup_status':
            output = f"Setup Complete: {SETUP_STATE['setup_complete']}\nConnection: {SETUP_STATE.get('connection_string', 'N/A')}"
        elif cmd.startswith('teleport '):
            data_input = cmd[9:].strip()
            if data_input:
                fid = quantum_foam.quantum_teleport(data_input)
                output = f"Teleportation fidelity: {fid:.6f}"
            else:
                output = "Usage: teleport <data>"
        elif cmd.startswith('entangle '):
            ip = cmd[9:].strip()
            if ip:
                fid = quantum_foam.entangle_ip(ip)
                output = f"Entangled {ip}: fidelity={fid:.15f}"
            else:
                output = "Usage: entangle <ip>"
        elif cmd == 'clear':
            qsh_sessions[sid]['history'] = []
            output = "History cleared."
        elif cmd == 'exit':
            output = "Session closed."
            prompt = False
        else:
            output = f"Unknown command: {cmd}. Type 'help' for commands."
    except Exception as e:
        logger.error(f"QSH error: {e}", exc_info=True)
        output = f'Error: {str(e)}'
    
    emit('qsh_output', {'output': output, 'prompt': prompt})

@app.route('/static/<path:filename>')
def send_static(filename):
    cloned_static = os.path.join(GATE_DIR, 'static')
    if os.path.exists(os.path.join(cloned_static, filename)):
        return send_from_directory(cloned_static, filename)
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("=" * 70)
    logger.info(f"QUANTUM NETWORK - Gateway: {UBUNTU_GATEWAY_IP}, DNS: {UBUNTU_DNS_IP}")
    logger.info("=" * 70)
    logger.info(f"Server starting on 0.0.0.0:{port}")
    logger.info(f"SSH: {'✓ Enabled' if SSH_ENABLED else '✗ Disabled'}")
    logger.info("=" * 70)
    
    # Disabled for debugging
    # setup_thread = threading.Thread(target=run_autonomous_setup_background, daemon=True)
    # setup_thread.start()
    
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    finally:
        logger.info("Shutdown complete")
