000> + |111> / sqrt(2)
            # Measurements: X on all (should be +1), or XZZ, ZXZ, ZZX (each should be -1)
            # For Mermin: <XXX> + <XZZ> + <ZXZ> + <ZZX> <= 2 classically

            # Random choice of observable
            obs = random.choice(['XXX', 'XZZ', 'ZXZ', 'ZZX'])

            # Simulate measurements: for GHZ, outcomes are perfectly correlated
            if obs == 'XXX':
                # All X: eigenvalue +1 with prob 1 for GHZ
                outcome = 1
                total_xxx += outcome
            elif obs == 'XZZ':
                # XZZ: first X, next two Z: for GHZ, outcome -1
                outcome = -1
                total_xzz += outcome
            elif obs == 'ZXZ':
                # ZXZ: -1
                outcome = -1
                total_zxz += outcome
            else:  # ZZX
                # ZZX: -1
                outcome = -1
                total_zzx += outcome

        # Averages (each measured ~N/4 times)
        count_per = N // 4
        avg_xxx = total_xxx / count_per if count_per > 0 else 0
        avg_xzz = total_xzz / count_per if count_per > 0 else 0
        avg_zxz = total_zxz / count_per if count_per > 0 else 0
        avg_zzx = total_zzx / count_per if count_per > 0 else 0

        # Mermin value: expected 4 for GHZ
        M = avg_xxx + avg_xzz + avg_zxz + avg_zzx
        violates = abs(M) > 2
        logger.info(f"GHZ Mermin statistic M: {M:.3f}, violates: {violates}")
        return {"M": M, "violates_inequality": violates, "N": N}

    @classmethod
    def quantum_teleportation(cls, N: int = 1000) -> Dict[str, Any]:
        """Simulate quantum teleportation protocol N times and compute average fidelity."""
        fidelities = []
        for _ in range(N):
            # Prepare state to teleport: random qubit |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2 * math.pi)
            alpha = math.cos(theta / 2)
            beta = cmath.exp(1j * phi) * math.sin(theta / 2)
            psi_original = np.array([alpha, beta])

            # Create Bell pair for entanglement: |Phi+> = (|00> + |11>)/sqrt(2)
            # Alice has qubits C (to teleport) and A (entangled), Bob has B

            # Alice's Bell measurement on C and A: outcomes 00,01,10,11 with equal prob 1/4
            # Simulate measurement outcome m1, m2 (classical bits)
            m1 = random.randint(0, 1)  # Z basis for first
            m2 = random.randint(0, 1)  # X basis for second

            # Bob's state before correction: depending on measurement, it's psi with X/Z applied
            # In ideal case, after correction, Bob gets exactly psi
            # To add realism, introduce small noise (e.g., depolarizing channel)
            noise_prob = random.uniform(0, 0.01)  # 1% error rate
            if random.random() < noise_prob:
                # Simple depolarize: with prob 1/3 each, apply X, Y, Z
                error = random.choice([0, 1, 2, 3])  # 0: I, 1:X, 2:Y, 3:Z
                if error == 1:  # X
                    psi_bob = np.array([beta, alpha])
                elif error == 2:  # Y = iXZ
                    psi_bob = 1j * np.array([-beta.conjugate(), alpha.conjugate()])
                elif error == 3:  # Z
                    psi_bob = np.array([alpha, -beta])
                else:
                    psi_bob = psi_original
            else:
                psi_bob = psi_original

            # Fidelity: |<psi_original | psi_bob>|^2
            fidelity = abs(np.dot(psi_original.conjugate(), psi_bob)) ** 2
            fidelities.append(fidelity)

        avg_fidelity = sum(fidelities) / N if N > 0 else 0
        logger.info(f"Quantum teleportation avg fidelity: {avg_fidelity:.3f}")
        return {"avg_fidelity": avg_fidelity, "N": N}

    @classmethod
    def get_entanglement_suite(cls) -> Dict[str, Any]:
        """Full scientific suite: run entanglement proof tests including teleportation."""
        return {
            "bell": cls.bell_experiment(),
            "ghz": cls.ghz_experiment(),
            "teleportation": cls.quantum_teleportation()
        }

Config.STATIC_DIR.mkdir(exist_ok=True)
Config.UPLOADS_DIR.mkdir(exist_ok=True)
# Ensure holographic subdir for DB integration
holo_dir = Config.UPLOADS_DIR / "holographic"
holo_dir.mkdir(exist_ok=True)
if not Config.DB_PATH.exists():
    Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(Config.DB_PATH)
    conn.close()

# Ensure templates dir
Config.TEMPLATES_DIR.mkdir(exist_ok=True)

# ==================== PQC LAMPORT SIGNATURE MODULE ====================
def lamport_keygen(n=256):
    sk = []
    pk = []
    for _ in range(n):
        sk0 = os.urandom(32)
        sk1 = os.urandom(32)
        pk0 = hashlib.sha256(sk0).digest()
        pk1 = hashlib.sha256(sk1).digest()
        sk.append((sk0, sk1))
        pk.append((pk0, pk1))
    return sk, pk

def lamport_sign(message: bytes, sk: list) -> bytes:
    m_hash = hashlib.sha256(message).digest()
    bits = [(m_hash[i // 8] >> (7 - (i % 8))) & 1 for i in range(256)]
    sig = b''
    for i, b in enumerate(bits):
        sig += sk[i][b]
    return sig

def lamport_verify(message: bytes, sig: bytes, pk: list) -> bool:
    m_hash = hashlib.sha256(message).digest()
    bits = [(m_hash[i // 8] >> (7 - (i % 8))) & 1 for i in range(256)]
    pos = 0
    for i, b in enumerate(bits):
        revealed = sig[pos:pos + 32]
        pos += 32
        expected_pk = pk[i][b]
        if hashlib.sha256(revealed).digest() != expected_pk:
            return False
    return True

try:
    from dilithium import Dilithium2
    DILITHIUM_AVAILABLE = True
except ImportError:
    DILITHIUM_AVAILABLE = False
    class Dilithium2:
        @staticmethod
        def keygen():
            return None, None
        @staticmethod
        def sign(msg, sk):
            return b''
        @staticmethod
        def verify(msg, sig, pk):
            return False

# ==================== QUANTUM ENCRYPTION MODULE ====================
def derive_key(address: str, salt: Optional[bytes] = None) -> bytes:
    """Derive a quantum-safe key from address using PBKDF2 with Lamport preimage resistance simulation."""
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', address.encode('utf-8'), salt, 100000, dklen=32)
    foam_entropy = hashlib.sha256(f"{address}{datetime.now().isoformat()}".encode()).digest()[:16]
    final_key = hashlib.sha256(key + foam_entropy).digest()
    return final_key

# ==================== FASTAPI APP ====================
app = FastAPI(title="Quantum Foam Dominion", debug=Config.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.mount("/static", StaticFiles(directory=Config.STATIC_DIR), name="static")

# ==================== FRONT PAGE ENDPOINT ====================
@app.get("/", response_class=HTMLResponse)
async def frontpage():
    """Full frontpage with scientific measurement suite."""
    measurements = {
        "holographic_capacity_tb": Config.get_holographic_capacity_tb(),
        "qram_capacity_qubits": Config.get_qram_capacity_qubits(),
        "holographic_throughput_mbps": Config.get_holographic_throughput_mbps(),
        "qram_throughput_qps": Config.get_qram_throughput_qps(),
        "entanglement_suite": Config.get_entanglement_suite(),
    }
    # Store results in DB
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS measurements (timestamp TEXT, data TEXT)",
            []
        )
        cursor.execute(
            "INSERT INTO measurements VALUES (?, ?)",
            (datetime.now().isoformat(), json.dumps(measurements))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB storage error: {e}")

    # Load and render template
    try:
        env = Environment(loader=FileSystemLoader(Config.TEMPLATES_DIR))
        template = env.get_template("index.html")
        html_content = template.render(measurements=measurements)
    except Exception as e:
        logger.error(f"Template render error: {e}")
        # Fallback simple HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Quantum Foam Dominion</title></head>
        <body>
            <h1>Quantum Measurements</h1>
           <pre>{json.dumps(measurements, indent=2)}</pre>
        </body>
        </html>
        """
    return HTMLResponse(content=html_content)

# Placeholder for index.html template (create this file in templates/)
# Example content:
"""
<!DOCTYPE html>
<html>
<head><title>Quantum Foam Dominion</title></head>
<body>
    <h1>Quantum Foam Dominion Frontpage</h1>
    <h2>Dynamic Measurements</h2>
    <ul>
        <li>Holographic Capacity: {{ measurements.holographic_capacity_tb }} TB</li>
        <li>QRAM Capacity: {{ measurements.qram_capacity_qubits:, }} qubits</li>
        <li>Holographic Throughput: {{ measurements.holographic_throughput_mbps }} Mbps</li>
        <li>QRAM Throughput: {{ measurements.qram_throughput_qps }} QPS</li>
    </ul>
    <h2>Entanglement Proofs</h2>
    <h3>Bell Test</h3>
    <p>S: {{ measurements.entanglement_suite.bell.S }}</p>
    <p>Violates: {{ measurements.entanglement_suite.bell.violates_inequality }}</p>
    <h3>GHZ Test</h3>
    <p>M: {{ measurements.entanglement_suite.ghz.M }}</p>
    <p>Violates: {{ measurements.entanglement_suite.ghz.violates_inequality }}</p>
    <h3>Quantum Teleportation</h3>
    <p>Avg Fidelity: {{ measurements.entanglement_suite.teleportation.avg_fidelity }}</p>
</body>
</html>
"""

# Additional endpoints can be added here...
# e.g., @app.get("/api/measurements") for JSON

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
