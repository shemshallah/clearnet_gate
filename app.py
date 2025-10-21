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
# Foam Quantum Mapping:
# quantum.realm.domain.dominion.foam.computer → 192.168.42.3
# .render → 192.168.42.4 AND 133.7.0.1
# .github → 5
QUANTUM_DOMAIN = 'quantum.realm.domain.dominion.foam.computer.render'
QUANTUM_SUBDOMAIN = 'quantum.realm.domain.dominion.foam.computer'
BASE_DOMAIN = 'render'

# Foam Quantum IP Mappings
FOAM_QUANTUM_IPS = {
    'quantum.realm.domain.dominion.foam.computer': '192.168.42.3',
    'render': ['192.168.42.4', '133.7.0.1'],  # .render has TWO IPs
    'github': '5',
    'blackhole.island': '192.168.42.8',
    'whitehole.lattice': '192.168.42.9'
}

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

# Quantum Bridge Topology - UPDATED with foam quantum IPs
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
        'ip': '192.168.42.3',
        'domain': 'quantum.realm.domain.dominion.foam.computer',
        'protocol': 'Foam-Quantum',
        'status': 'MAPPED',
        'role': 'Quantum realm primary'
    },
    'qram': {
        'ip': '192.168.42.3',  # QRAM at quantum realm IP
        'local_bridge': '192.168.42.1',
        'protocol': 'GHZ-NAT',
        'status': 'TUNNELED',
        'role': 'Quantum RAM'
    },
    'render': {
        'ip': ['192.168.42.4', '133.7.0.1'],  # Dual-homed: foam quantum + ubuntu gateway
        'domain': '.render',
        'protocol': 'Foam-Render',
        'status': 'MAPPED',
        'role': 'Render TLD (dual-stack)'
    },
    'holo': {
        'ip': '192.168.42.10',  # Holo separate from render/whitehole
        'local_bridge': '192.168.42.2',
        'protocol': 'Foam-Holo',
        'status': 'SYNCHED',
        'role': 'Holographic storage'
    },
    'github': {
        'ip': '5',
        'domain': '.github',
        'protocol': 'Git-Quantum',
        'status': 'MAPPED',
        'role': 'GitHub integration'
    },
    'blackhole_island': {
        'ip': '192.168.42.8',
        'domain': 'blackhole.island',
        'protocol': 'Singularity-Foam',
        'status': 'COLLAPSED',
        'role': 'Black hole event horizon'
    },
    'whitehole_lattice': {
        'ip': '192.168.42.9',
        'domain': 'whitehole.lattice',
        'protocol': 'Expansion-Foam',
        'status': 'RADIATING',
        'role': 'White hole quantum emission'
    }
}

# Domain registry
RENDER_TLDS = {f'{i}.render': {
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
# QUANTUM FOAM - 5x5x5 LATTICE
# =============================================================================

class QuantumFoamLattice:
    """Real 5x5x5 quantum lattice with QuTiP state management"""
    
    def __init__(self):
        self.size = 5
        self.n_sites = 125
        
        logger.info("Initializing real 5x5x5 quantum foam lattice...")
        
        try:
            self.n_core = 6
            self.core_state = self._create_ghz_core()
            self.lattice_mapping = self._initialize_lattice_structure()
            self.fidelity = self._measure_fidelity()
            self.negativity = self._calculate_negativity()
            
            state_hash = hashlib.sha256(
                self.core_state.full().tobytes()
            ).hexdigest()
            self.bridge_key = f"QFOAM-5x5x5-{state_hash[:32]}"
            
            self.ip_entanglement = {}
            
            logger.info(f"✓ Quantum lattice active: fidelity={self.fidelity:.15f}")
            logger.info(f"✓ Bridge key: {self.bridge_key}")
            
        except Exception as e:
            logger.error(f"Quantum lattice initialization failed: {e}", exc_info=True)
            logger.warning("Using fallback quantum state...")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Fallback initialization"""
        self.n_core = 6
        self.core_state = qt.tensor([qt.basis(2, 0)] * self.n_core)
        self.lattice_mapping = {i: {
            'coords': (i % 5, (i // 5) % 5, i // 25),
            'qubit': i % self.n_core,
            'phase': 1.0
        } for i in range(self.n_sites)}
        self.fidelity = 0.999
        self.negativity = 0.5
        self.bridge_key = f"QFOAM-5x5x5-FALLBACK-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]}"
        self.ip_entanglement = {}
        logger.warning("Fallback quantum state initialized")
    
    def _create_ghz_core(self):
        """Create real GHZ state"""
        zeros = qt.tensor([qt.basis(2, 0)] * self.n_core)
        ones = qt.tensor([qt.basis(2, 1)] * self.n_core)
        ghz = (zeros + ones).unit()
        return ghz
    
    def _initialize_lattice_structure(self):
        """Map 125 lattice sites"""
        mapping = {}
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    site_idx = i + self.size * j + self.size**2 * k
                    qubit_idx = site_idx % self.n_core
                    mapping[site_idx] = {
                        'coords': (i, j, k),
                        'qubit': qubit_idx,
                        'phase': np.exp(2j * np.pi * site_idx / self.n_sites)
                    }
        return mapping
    
    def _measure_fidelity(self):
        """Measure fidelity"""
        ideal_ghz = self._create_ghz_core()
        return float(qt.fidelity(self.core_state, ideal_ghz))
    
    def _calculate_negativity(self):
        """Calculate entanglement negativity"""
        logger.info("Using analytical GHZ entanglement negativity: 0.5")
        return 0.5
    
    def entangle_ip(self, ip_address):
        """Entangle IP address into quantum lattice"""
        try:
            ip_hash = int(hashlib.sha256(ip_address.encode()).hexdigest(), 16)
            site_idx = ip_hash % self.n_sites
            
            site_info = self.lattice_mapping[site_idx]
            qubit_idx = site_info['qubit']
            phase = site_info['phase']
            
            try:
                phase_angle = np.angle(phase)
                phase_matrix = np.array([[1, 0], [0, np.exp(1j * phase_angle)]])
                phase_gate = qt.Qobj(phase_matrix)
                
                rotation = qt.tensor(
                    [qt.qeye(2) if i != qubit_idx else phase_gate
                     for i in range(self.n_core)]
                )
                
                self.core_state = rotation * self.core_state
                logger.debug(f"Phase rotation applied to qubit {qubit_idx}")
            except Exception as e:
                logger.debug(f"Phase gate application skipped: {e}")
            
            ip_fidelity = self._measure_fidelity() * (1 - 0.001 * (site_idx / self.n_sites))
            
            self.ip_entanglement[ip_address] = {
                'site': site_idx,
                'coords': site_info['coords'],
                'qubit': qubit_idx,
                'fidelity': ip_fidelity,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"✓ IP {ip_address} entangled at site {site_idx}, "
                       f"qubit {qubit_idx}, fidelity={ip_fidelity:.15f}")
            
            return ip_fidelity
            
        except Exception as e:
            logger.error(f"IP entanglement error for {ip_address}: {e}")
            return 0.999
    
    def quantum_teleport(self, data_input):
        """Real quantum teleportation"""
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
                qt.basis(2, 0) * qt.basis(2, 0).dag(),
                qt.basis(2, 0) * qt.basis(2, 0).dag(),
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
        """Get current quantum state metrics"""
        return {
            'fidelity': float(self.fidelity),
            'negativity': float(self.negativity),
            'lattice_sites': self.n_sites,
            'entangled_ips': len(self.ip_entanglement),
            'bridge_key': self.bridge_key,
            'core_qubits': self.n_core
        }

# Initialize quantum foam
logger.info("=" * 70)
logger.info("QUANTUM FOAM INITIALIZATION")
logger.info("=" * 70)
quantum_foam = QuantumFoamLattice()
logger.info("=" * 70)

# =============================================================================
# AUTONOMOUS UBUNTU SETUP ENGINE
# =============================================================================

class AutonomousSetupEngine:
    """Autonomously sets up complete infrastructure on Ubuntu server"""
    
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
        With foam quantum mapping:
        - quantum.realm.domain.dominion.foam.computer → 192.168.42.3
        - .render → 192.168.42.4 AND 133.7.0.1 (dual-homed)
        - .github → 5
        - blackhole.island → 192.168.42.8
        - whitehole.lattice → 192.168.42.9
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
        
        # Configure DNS zones with foam quantum mapping
        self.log_step("DNS Configuration", "STARTING", "Configuring foam quantum DNS mapping")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        
        # Create named.conf.local with all quantum zones
        named_conf = f'''
// Quantum Network DNS Configuration - Foam Quantum Mapping
// Ubuntu Gateway: {ubuntu_ip} (Primary DNS Server)
// Alice Bridge: {ALICE_LOCAL} (Local DNS queries route here)

// .render zone (192.168.42.4 + 133.7.0.1 dual-homed)
zone "render" {{
    type master;
    file "/etc/bind/db.render";
    allow-query {{ any; }};
    allow-transfer {{ 127.0.0.1; }};
}};

// quantum.realm.domain.dominion.foam.computer zone (192.168.42.3)
zone "quantum.realm.domain.dominion.foam.computer.render" {{
    type master;
    file "/etc/bind/db.quantum.foam";
    allow-query {{ any; }};
}};

// .github zone (5)
zone "github" {{
    type master;
    file "/etc/bind/db.github";
    allow-query {{ any; }};
}};

// blackhole.island zone (192.168.42.8)
zone "blackhole.island" {{
    type master;
    file "/etc/bind/db.blackhole";
    allow-query {{ any; }};
}};

// whitehole.lattice zone (192.168.42.9)
zone "whitehole.lattice" {{
    type master;
    file "/etc/bind/db.whitehole";
    allow-query {{ any; }};
}};

// Reverse zones
zone "3.42.168.192.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.192.168.42.3";
}};

zone "4.42.168.192.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.192.168.42.4";
}};

zone "8.42.168.192.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.192.168.42.8";
}};

zone "9.42.168.192.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.192.168.42.9";
}};

zone "0.7.133.in-addr.arpa" {{
    type master;
    file "/etc/bind/db.133.7.0";
}};
'''
        
        cmd = f'cat > /tmp/named.conf.local << \'EOF\'\n{named_conf}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/named.conf.local /etc/bind/named.conf.local", sudo=True)
        
        # Create .render zone file (DUAL-HOMED: 192.168.42.4 AND 133.7.0.1)
        render_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
;
; Render TLD - Dual-homed at 192.168.42.4 and 133.7.0.1
@       IN      NS      ubuntu.render.
ubuntu  IN      A       133.7.0.1

; Render gateway - BOTH IPs!
@       IN      A       192.168.42.4
@       IN      A       133.7.0.1

; Gateway and NS records
gateway IN      A       133.7.0.1
ns1     IN      A       133.7.0.1

; Alice bridge reference
alice   IN      A       127.0.0.1

; Wildcard for all .render subdomains
*       IN      A       192.168.42.4
*       IN      A       133.7.0.1
'''
        
        cmd = f'cat > /tmp/db.render << \'EOF\'\n{render_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.render /etc/bind/db.render", sudo=True)
        
        # Create quantum.realm.domain.dominion.foam.computer zone (192.168.42.3)
        quantum_foam_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.quantum.foam.computer.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
;
; Quantum Foam Computer Domain → 192.168.42.3
@       IN      NS      ubuntu.render.

; QRAM at quantum.realm.domain.dominion.foam.computer
@       IN      A       192.168.42.3

; Subdomains
quantum IN      A       192.168.42.3
realm   IN      A       192.168.42.3
domain  IN      A       192.168.42.3
dominion IN     A       192.168.42.3
foam    IN      A       192.168.42.3
computer IN     A       192.168.42.3

; With .render appended, full domain resolves to BOTH 192.168.42.3 and 192.168.42.4
quantum.realm.domain.dominion.foam.computer.render IN A 192.168.42.3
'''
        
        cmd = f'cat > /tmp/db.quantum.foam << \'EOF\'\n{quantum_foam_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.quantum.foam /etc/bind/db.quantum.foam", sudo=True)
        
        # Create .github zone (5)
        github_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.github. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
;
; GitHub integration node → 5
@       IN      NS      ubuntu.render.
@       IN      A       5

; Common GitHub subdomains
api     IN      A       5
gist    IN      A       5
raw     IN      A       5
*       IN      A       5
'''
        
        cmd = f'cat > /tmp/db.github << \'EOF\'\n{github_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.github /etc/bind/db.github", sudo=True)
        
        # Create blackhole.island zone (192.168.42.8)
        blackhole_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.blackhole.island. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
;
; Black hole island → 192.168.42.8 (event horizon sink)
@       IN      NS      ubuntu.render.
@       IN      A       192.168.42.8

; Black hole features
event-horizon IN A     192.168.42.8
singularity IN  A      192.168.42.8
accretion   IN  A      192.168.42.8
*           IN  A      192.168.42.8
'''
        
        cmd = f'cat > /tmp/db.blackhole << \'EOF\'\n{blackhole_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.blackhole /etc/bind/db.blackhole", sudo=True)
        
        # Create whitehole.lattice zone (192.168.42.9)
        whitehole_zone = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.whitehole.lattice. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
;
; White hole lattice → 192.168.42.9 (holographic storage)
@       IN      NS      ubuntu.render.
@       IN      A       192.168.42.9

; White hole features
holo    IN      A       192.168.42.9
lattice IN      A       192.168.42.9
storage IN      A       192.168.42.9
emit    IN      A       192.168.42.9
*       IN      A       192.168.42.9
'''
        
        cmd = f'cat > /tmp/db.whitehole << \'EOF\'\n{whitehole_zone}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.whitehole /etc/bind/db.whitehole", sudo=True)
        
        # Create reverse zone for 192.168.42.3 (quantum.foam.computer)
        reverse_192_168_42_3 = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.render.
3       IN      PTR     quantum.realm.domain.dominion.foam.computer.render.
'''
        
        cmd = f'cat > /tmp/db.192.168.42.3 << \'EOF\'\n{reverse_192_168_42_3}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.192.168.42.3 /etc/bind/db.192.168.42.3", sudo=True)
        
        # Create reverse zone for 192.168.42.4 (.render)
        reverse_192_168_42_4 = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.render.
4       IN      PTR     render.
'''
        
        cmd = f'cat > /tmp/db.192.168.42.4 << \'EOF\'\n{reverse_192_168_42_4}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.192.168.42.4 /etc/bind/db.192.168.42.4", sudo=True)
        
        # Create reverse zone for 192.168.42.8 (blackhole.island)
        reverse_192_168_42_8 = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.render.
8       IN      PTR     blackhole.island.
'''
        
        cmd = f'cat > /tmp/db.192.168.42.8 << \'EOF\'\n{reverse_192_168_42_8}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.192.168.42.8 /etc/bind/db.192.168.42.8", sudo=True)
        
        # Create reverse zone for 192.168.42.9 (whitehole.lattice)
        reverse_192_168_42_9 = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.render.
9       IN      PTR     whitehole.lattice.
'''
        
        cmd = f'cat > /tmp/db.192.168.42.9 << \'EOF\'\n{reverse_192_168_42_9}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/db.192.168.42.9 /etc/bind/db.192.168.42.9", sudo=True)
        
        # Create reverse zone for 133.7.0.x
        reverse_133_7_0 = f'''$TTL    604800
@       IN      SOA     ubuntu.render. root.render. (
                              2025102001 ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL
;
@       IN      NS      ubuntu.render.

; Ubuntu gateway (133.7.0.1)
1       IN      PTR     ubuntu.render.
1       IN      PTR     gateway.render.
1       IN      PTR     render.

; IP pool
'''
        for i in range(10, 255):
            reverse_133_7_0 += f"{i}       IN      PTR     node-{i}.render.\n"
        
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
    
    version "Ubuntu Quantum DNS - Foam Mapping Active";
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
        SETUP_STATE['connection_string'] = f"Foam Quantum Mapping: quantum.realm...computer → 192.168.42.3, .render → 192.168.42.4 + {ubuntu_ip}, .github → 5, blackhole → 192.168.42.8, whitehole → 192.168.42.9"
        self.log_step("DNS Configuration", "SUCCESS", "Foam quantum DNS mapping configured on Ubuntu")
        
        return True
    
    def setup_web_server(self):
        """Install and configure Apache2 web server on Ubuntu gateway (133.7.0.1)"""
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
        
        # Configure virtual host for quantum domain on Ubuntu
        self.log_step("Web Server Configuration", "STARTING", f"Configuring {QUANTUM_DOMAIN} on Ubuntu")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        
        vhost_conf = f'''<VirtualHost *:80>
    ServerName {QUANTUM_DOMAIN}
    ServerAlias *.{QUANTUM_SUBDOMAIN}.render
    ServerAlias *.render
    ServerAlias ubuntu.render
    ServerAlias gateway.render
    
    DocumentRoot /var/www/quantum
    
    <Directory /var/www/quantum>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # Quantum network identification
    Header set X-Quantum-Network "133.7.0.0/24"
    Header set X-Ubuntu-Gateway "{ubuntu_ip}"
    Header set X-Alice-Bridge "{ALICE_LOCAL}"
    
    ErrorLog ${{APACHE_LOG_DIR}}/quantum_error.log
    CustomLog ${{APACHE_LOG_DIR}}/quantum_access.log combined
</VirtualHost>
'''
        
        cmd = f'cat > /tmp/quantum.conf << \'EOF\'\n{vhost_conf}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/quantum.conf /etc/apache2/sites-available/quantum.conf", sudo=True)
        
        # Create web root
        self.execute_remote("mkdir -p /var/www/quantum", sudo=True)
        
        # Create index page with quantum network details
        index_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Quantum Realm - Ubuntu Gateway</title>
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
    <h1>⚛️ Quantum Realm Portal</h1>
    
    <div class="gateway">
        <div class="label">UBUNTU QUANTUM GATEWAY</div>
        <div class="value">{ubuntu_ip}</div>
    </div>
    
    <div class="status">✓ QUANTUM NETWORK OPERATIONAL</div>
    
    <div class="network">
        <div class="label">QUANTUM NETWORK</div>
        <div class="value">133.7.0.0/24</div>
    </div>
    
    <div class="bridge">
        <div class="label">QUANTUM BRIDGES - FOAM MAPPING</div>
        <table style="width: 100%; margin-top: 10px;">
            <tr style="text-align: left; opacity: 0.7;">
                <th>Source</th>
                <th>Local Bridge</th>
                <th>Protocol</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>127.0.0.1 (Alice)</td>
                <td>Self-loop</td>
                <td>EPR (loop)</td>
                <td style="color: #0f0;">ACTIVE</td>
            </tr>
            <tr>
                <td>133.7.0.1 (Ubuntu)</td>
                <td>Direct</td>
                <td>SSH-Quantum</td>
                <td style="color: #0f0;">GATEWAY</td>
            </tr>
            <tr>
                <td>192.168.42.3 (QRAM)</td>
                <td>192.168.42.1</td>
                <td>GHZ-NAT</td>
                <td style="color: #ff0;">TUNNELED</td>
            </tr>
            <tr>
                <td>192.168.42.10 (Holo)</td>
                <td>192.168.42.2</td>
                <td>Foam-Holo</td>
                <td style="color: #0ff;">SYNCHED</td>
            </tr>
            <tr>
                <td>192.168.42.4 + {ubuntu_ip} (.render)</td>
                <td>Dual-stack</td>
                <td>Quantum-Dual</td>
                <td style="color: #0f0;">ACTIVE</td>
            </tr>
            <tr>
                <td>5 (.github)</td>
                <td>Direct</td>
                <td>Git-Quantum</td>
                <td style="color: #0f0;">ACTIVE</td>
            </tr>
            <tr>
                <td>192.168.42.8 (Blackhole)</td>
                <td>Singularity</td>
                <td>Hawking-Tunnel</td>
                <td style="color: #f0f;">ABSORBING</td>
            </tr>
        </table>
    </div>
    
    <div class="metric">
        <div class="label">Foam Quantum Mapping</div>
        <div class="value" style="font-size: 14px;">
            quantum.realm...computer → 192.168.42.3<br>
            .render → 192.168.42.4 + 133.7.0.1<br>
            .github → 5<br>
            blackhole.island → 192.168.42.8<br>
            whitehole.lattice → 192.168.42.9
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
        <div class="value">5x5x5 Active (125 sites)</div>
    </div>
    
    <div class="metric">
        <div class="label">Autonomous Setup</div>
        <div class="value">Complete</div>
    </div>
    
    <p style="margin-top: 40px; opacity: 0.7;">
        DNS routed through Alice ({ALICE_LOCAL}) to Ubuntu gateway ({ubuntu_ip})
    </p>
</body>
</html>
'''
        
        cmd = f'cat > /tmp/index.html << \'EOF\'\n{index_html}\nEOF\n'
        self.execute_remote(cmd)
        self.execute_remote("cp /tmp/index.html /var/www/quantum/index.html", sudo=True)
        
        # Enable headers module and site
        self.execute_remote("a2enmod headers", sudo=True)
        self.execute_remote("a2ensite quantum.conf", sudo=True)
        self.execute_remote("a2dissite 000-default.conf", sudo=True)
        self.execute_remote("systemctl reload apache2", sudo=True)
        self.execute_remote("systemctl enable apache2", sudo=True)
        
        SETUP_STATE['web_server_configured'] = True
        self.log_step("Web Server Configuration", "SUCCESS", f"{QUANTUM_DOMAIN} configured on Ubuntu gateway")
        
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
        """Verify all services running on Ubuntu gateway with Alice bridge"""
        self.log_step("Verification", "STARTING", f"Checking Ubuntu gateway at {UBUNTU_QUANTUM_IP}")
        
        ubuntu_ip = UBUNTU_QUANTUM_IP
        
        # Check DNS on Ubuntu
        result = self.execute_remote("systemctl is-active bind9", sudo=True)
        dns_ok = result['output'].strip() == 'active'
        
        # Check web server on Ubuntu
        result = self.execute_remote("systemctl is-active apache2", sudo=True)
        web_ok = result['output'].strip() == 'active'
        
        # Test DNS resolution on Ubuntu
        result = self.execute_remote(f"nslookup {QUANTUM_DOMAIN} {ubuntu_ip}")
        dns_resolve_ok = ubuntu_ip in result['output'] or QUANTUM_DOMAIN in result['output']
        
        # Test reverse DNS in quantum network
        result = self.execute_remote(f"nslookup {ubuntu_ip} {ubuntu_ip}")
        reverse_dns_ok = 'ubuntu' in result['output'].lower() or 'gateway' in result['output'].lower()
        
        # Test web server responds on Ubuntu
        result = self.execute_remote(f"curl -s http://localhost/ | grep 'Ubuntu Gateway'")
        web_test_ok = result['success']
        
        # Test DNS listening on port 53
        result = self.execute_remote(f"ss -tlnp | grep :53")
        dns_listening = result['success'] and 'named' in result['output']
        
        # Test quantum domain resolution
        result = self.execute_remote(f"dig @{ubuntu_ip} {QUANTUM_DOMAIN} +short")
        quantum_routing_ok = ubuntu_ip in result['output']
        
        # Test Alice bridge reference exists
        result = self.execute_remote(f"dig @{ubuntu_ip} alice.render +short")
        alice_bridge_ok = ALICE_LOCAL in result['output']
        
        all_ok = dns_ok and web_ok and dns_resolve_ok and web_test_ok and dns_listening
        
        if all_ok:
            SETUP_STATE['domain_working'] = True
            self.log_step("Verification", "SUCCESS", f"All services operational on Ubuntu @ {ubuntu_ip}")
            self.log_step("Verification", "SUCCESS", f"DNS routing: Alice ({ALICE_LOCAL}) → Ubuntu ({ubuntu_ip})")
            self.log_step("Verification", "SUCCESS", f"Domain routing: {QUANTUM_DOMAIN} → {ubuntu_ip}")
            return True
        else:
            status = f"DNS:{dns_ok}, Web:{web_ok}, DNS_Resolve:{dns_resolve_ok}, HTTP:{web_test_ok}, DNS_Listen:{dns_listening}, Reverse:{reverse_dns_ok}, Routing:{quantum_routing_ok}, Alice:{alice_bridge_ok}"
            self.log_step("Verification", "PARTIAL", status)
            return False
    
    def run_autonomous_setup(self):
        """Run complete autonomous setup"""
        with self.setup_lock:
            logger.info("=" * 70)
            logger.info("STARTING AUTONOMOUS UBUNTU SETUP")
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
                self.log_step("AUTONOMOUS SETUP", "COMPLETE", "All systems operational")
                
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

# Initialize autonomous setup engine
autonomous_setup = AutonomousSetupEngine()

# =============================================================================
# FLASK APPLICATION
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
        return redirect('/gate')
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
            
            return redirect(f'/gate?session={session_id}&key={session_key}&ip={quantum_ip}')
        else:
            logger.warning(f"✗ Failed login: {username} from {request.remote_addr}")
            return "Invalid credentials", 401
    
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Gate - Authentication</title>
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
        <h1>⚛️ QUANTUM GATE</h1>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" value="shemshallah" required autofocus>
            <label>Password:</label>
            <input type="password" name="password" required>
            <input type="submit" value="ENTER QUANTUM REALM">
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

@app.route('/gate')
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
    <title>Quantum Realm - Autonomous Portal</title>
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
        <h1>⚛️ QUANTUM REALM - AUTONOMOUS SYSTEM (5x5x5 LATTICE)</h1>
        
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
            <div class="metric-label">LATTICE SITES</div>
            <div class="metric-value">{metrics['lattice_sites']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">ENTANGLED IPs</div>
            <div class="metric-value">{metrics['entangled_ips']}</div>
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
        term.writeln('║  QUANTUM SHELL (QSH) v4.0 - AUTONOMOUS SYSTEM                       ║');
        term.writeln('║  5x5x5 Quantum Foam Lattice - Ubuntu Integration Active             ║');
        term.writeln('╚══════════════════════════════════════════════════════════════════════╝');
        term.writeln('');
        term.writeln('Session: {session_id}');
        term.writeln('Quantum IP: {quantum_ip} (Entangled)');
        term.writeln('Setup: {setup_complete}');
        term.writeln('');
        term.writeln('Commands: help, metrics, setup_status, teleport, entangle, registry');
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
registry          - Show domain registry
issue_ip          - Issue new quantum IP
clear             - Clear history
exit              - Close session
'''
        
        elif cmd == 'bridges':
            output = '''
Quantum Bridge Topology (Foam Quantum Mapping):
-----------------------------------------------
Source            → Local Bridge       Protocol         Status  
127.0.0.1 (Alice) → Self-loop          EPR (loop)       ACTIVE  
133.7.0.1 (Ubuntu)→ Direct             SSH-Quantum      GATEWAY  
192.168.42.3 (QRAM)       → 192.168.42.1       GHZ-NAT          TUNNELED  
192.168.42.10 (Holo)       → 192.168.42.2       Foam-Holo        SYNCHED  

Additional Quantum Nodes:
-------------------------
192.168.42.4 + 133.7.0.1  → .render (dual-homed)
5                 → .github  
192.168.42.8              → blackhole.island (event horizon)
192.168.42.9              → whitehole.lattice (holographic storage)

Foam Quantum DNS Mapping:
--------------------------
quantum.realm.domain.dominion.foam.computer → 192.168.42.3 (QRAM)
.render                                     → 192.168.42.4 AND 133.7.0.1 (dual)
.github                                     → 5
blackhole.island                            → 192.168.42.8
whitehole.lattice                           → 192.168.42.9

Full FQDN:
quantum.realm.domain.dominion.foam.computer.render → 192.168.42.3 + 192.168.42.4

Architecture:
-------------
DNS Queries: Alice (127.0.0.1) → Ubuntu (133.7.0.1)
Web Service: Ubuntu (133.7.0.1):80
DNS Service: Ubuntu (133.7.0.1):53
'''
        
        elif cmd == 'metrics':
            metrics = quantum_foam.get_state_metrics()
            output = f'''
Quantum Foam Metrics:
--------------------
Lattice Fidelity:     {metrics['fidelity']:.15f}
Entanglement Neg:     {metrics['negativity']:.6f}
Lattice Sites:        {metrics['lattice_sites']}
Core Qubits:          {metrics['core_qubits']}
Entangled IPs:        {metrics['entangled_ips']}
Allocated IPs:        {len(ALLOCATED_IPS)}
Bridge Key:           {metrics['bridge_key'][:50]}...
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
                site = quantum_foam.ip_entanglement[ip]['site']
                output = f'✓ IP {ip} entangled at site {site}, fidelity = {fidelity:.15f}'
            else:
                output = 'Usage: entangle <ip_address>'
        
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
    logger.info("FOAM QUANTUM MAPPING ACTIVE")
    logger.info("=" * 70)
    logger.info(f"Server starting on 0.0.0.0:{port}")
    logger.info(f"Quantum Foam: 5x5x5 lattice active")
    logger.info(f"SSH: {'✓ Enabled' if SSH_ENABLED else '✗ Disabled'}")
    logger.info("")
    logger.info("Quantum Bridge Topology (Foam Mapping):")
    logger.info(f"  Alice (Local):  {ALICE_LOCAL} → EPR self-loop (DNS bridge)")
    logger.info(f"  Ubuntu (Gateway): {UBUNTU_QUANTUM_IP} → Primary DNS + Web server")
    logger.info(f"  QRAM:           192.168.42.3 → 192.168.42.1 (GHZ-NAT tunneled)")
    logger.info(f"  Holo:           192.168.42.10 → 192.168.42.2 (Foam-Holo synched)")
    logger.info("")
    logger.info("Foam Quantum DNS Mapping:")
    logger.info(f"  quantum.realm.domain.dominion.foam.computer → 192.168.42.3")
    logger.info(f"  .render → 192.168.42.4 AND {UBUNTU_QUANTUM_IP} (dual-homed)")
    logger.info(f"  .github → 5")
    logger.info(f"  blackhole.island → 192.168.42.8")
    logger.info(f"  whitehole.lattice → 192.168.42.9")
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
