
    def get_recent_measurements(limit: int = 10) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp, measurement_type, data, lattice_anchor, entanglement_fidelity FROM measurements ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "timestamp": row[0],
                    "type": row[1],
                    "data": json.loads(row[2]),
                    "lattice": row[3],
                    "fidelity": row[4]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Database retrieval error: {e}")
            return []
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Production password hashing with salt"""
        salt = hashlib.sha256(Config.SECRET_KEY.encode()).hexdigest().encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return hashlib.sha256(kdf.derive(password.encode())).hexdigest()
    
    @staticmethod
    def create_user(username: str, password: str) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            email = f"{username}@{Config.QUANTUM_EMAIL_DOMAIN}"
            password_hash = Database.hash_password(password)
            created_at = datetime.now().isoformat()
            quantum_key = secrets.token_urlsafe(32)
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, email, created_at, quantum_key) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, email, created_at, quantum_key)
            )
            
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"User created: {email}")
            return {"id": user_id, "username": username, "email": email, "created_at": created_at}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Username already exists")
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail="Error creating user")
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, username, password_hash, email FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            user_id, username, password_hash, email = row
            
            if Database.hash_password(password) != password_hash:
                return None
            
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(), user_id)
            )
            conn.commit()
            conn.close()
            
            return {"id": user_id, "username": username, "email": email}
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    @staticmethod
    def create_session(user_email: str) -> str:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            token = secrets.token_urlsafe(32)
            created_at = datetime.now()
            expires_at = created_at + timedelta(days=7)
            
            cursor.execute(
                "INSERT INTO sessions (token, user_email, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_email, created_at.isoformat(), expires_at.isoformat())
            )
            
            conn.commit()
            conn.close()
            
            return token
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail="Error creating session")
    
    @staticmethod
    def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT user_email, expires_at FROM sessions WHERE token = ?",
                (token,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            user_email, expires_at = row
            
            if datetime.fromisoformat(expires_at) < datetime.now():
                cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
                conn.commit()
                conn.close()
                return None
            
            cursor.execute(
                "SELECT id, username, email FROM users WHERE email = ?",
                (user_email,)
            )
            user_row = cursor.fetchone()
            conn.close()
            
            if user_row:
                return {"id": user_row[0], "username": user_row[1], "email": user_row[2]}
            return None
        except Exception as e:
            logger.error(f"Error getting user from token: {e}")
            return None
    
    @staticmethod
    def send_email(from_email: str, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            # Encrypt body via Sagittarius A* lattice
            encrypted_body = QuantumEncryption.encrypt_via_sagittarius_lattice(body)
            
            # Lattice routing path
            lattice_route = f"{Config.SAGITTARIUS_A_LATTICE} -> {Config.WHITE_HOLE_LATTICE} -> {Config.QUANTUM_DOMAIN}"
            
            sent_at = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO emails (from_user, to_user, subject, body, encrypted_body, lattice_route, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (from_email, to_email, subject, body, encrypted_body, lattice_route, sent_at)
            )
            
            conn.commit()
            email_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Email sent from {from_email} to {to_email} via lattice route: {lattice_route}")
            return {
                "id": email_id,
                "from": from_email,
                "to": to_email,
                "subject": subject,
                "sent_at": sent_at,
                "lattice_route": lattice_route
            }
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")
    
    @staticmethod
    def get_inbox(user_email: str) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, from_user, to_user, subject, encrypted_body, sent_at, read, starred, lattice_route 
                   FROM emails 
                   WHERE to_user = ? AND deleted_receiver = 0
                   ORDER BY sent_at DESC""",
                (user_email,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            emails = []
            for row in rows:
                # Decrypt body via white hole lattice
                try:
                    decrypted_body = QuantumEncryption.decrypt_via_whitehole_lattice(row[4])
                except:
                    decrypted_body = "[Decryption Error]"
                
                emails.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "subject": row[3],
                    "body": decrypted_body,
                    "sent_at": row[5],
                    "read": bool(row[6]),
                    "starred": bool(row[7]),
                    "lattice_route": row[8]
                })
            
            return emails
        except Exception as e:
            logger.error(f"Error getting inbox: {e}")
            return []
    
    @staticmethod
    def get_sent(user_email: str) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT id, from_user, to_user, subject, encrypted_body, sent_at, read, starred, lattice_route 
                   FROM emails 
                   WHERE from_user = ? AND deleted_sender = 0 
                   ORDER BY sent_at DESC""",
                (user_email,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            emails = []
            for row in rows:
                # Decrypt body via white hole lattice
                try:
                    decrypted_body = QuantumEncryption.decrypt_via_whitehole_lattice(row[4])
                except:
                    decrypted_body = "[Decryption Error]"
                
                emails.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "subject": row[3],
                    "body": decrypted_body,
                    "sent_at": row[5],
                    "read": bool(row[6]),
                    "starred": bool(row[7]),
                    "lattice_route": row[8]
                })
            
            return emails
        except Exception as e:
            logger.error(f"Error getting sent emails: {e}")
            return []
    
    @staticmethod
    def mark_as_read(email_id: int, user_email: str):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE emails SET read = 1 WHERE id = ? AND to_user = ?",
                (email_id, user_email)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error marking email as read: {e}")
    
    @staticmethod
    def toggle_star(email_id: int, user_email: str):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT starred FROM emails WHERE id = ? AND (from_user = ? OR to_user = ?)",
                (email_id, user_email, user_email)
            )
            row = cursor.fetchone()
            
            if row:
                new_starred = 0 if row[0] else 1
                cursor.execute(
                    "UPDATE emails SET starred = ? WHERE id = ?",
                    (new_starred, email_id)
                )
                conn.commit()
            
            conn.close()
        except Exception as e:
            logger.error(f"Error toggling star: {e}")
    
    @staticmethod
    def delete_emails(email_ids: List[int], user_email: str):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            for email_id in email_ids:
                cursor.execute(
                    "SELECT from_user, to_user FROM emails WHERE id = ?",
                    (email_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    from_user, to_user = row
                    if from_user == user_email:
                        cursor.execute(
                            "UPDATE emails SET deleted_sender = 1 WHERE id = ?",
                            (email_id,)
                        )
                    if to_user == user_email:
                        cursor.execute(
                            "UPDATE emails SET deleted_receiver = 1 WHERE id = ?",
                            (email_id,)
                        )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error deleting emails: {e}")

# ==================== SECURITY MODULE ====================
security = HTTPBearer(auto_error=False)

class SecurityManager:
    """Production authentication and authorization"""
    
    @staticmethod
    def generate_token() -> str:
        return secrets.token_urlsafe(32)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"authenticated": True}

async def get_current_user_email(session_token: Optional[str] = Cookie(None)):
    if not session_token:
        return None
    
    user = Database.get_user_from_token(session_token)
    if not user:
        return None
    
    return user

# ==================== RATE LIMITING ====================
rate_limit_store = defaultdict(list)

async def check_rate_limit(request: Request):
    client_ip = request.client.host
    now = datetime.now()
    
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip]
        if now - ts < timedelta(minutes=1)
    ]
    
    if len(rate_limit_store[client_ip]) >= Config.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    rate_limit_store[client_ip].append(now)

# ==================== QSH FOAM REPL (WebSocket) ====================
repl_sessions = {}

async def repl_exec(code: str, session_id: str):
    ns = repl_sessions.get(session_id, {
        'QuantumPhysics': QuantumPhysics,
        'SystemMetrics': SystemMetrics,
        'NetInterface': NetInterface,
        'AliceNode': AliceNode,
        'TorinoQuantumBackend': TorinoQuantumBackend,
        'Config': Config,
        'np': np,
        'math': math,
        'random': random,
        'basis': basis,
        'bell_state': bell_state,
        'tensor': tensor,
        'sigmax': sigmax,
        'sigmay': sigmay,
        'sigmaz': sigmaz,
        'qeye': qeye,
        'print': print,
        '__builtins__': {}
    })
    
    code = code.strip()
    
    # Handle network commands
    if code.startswith(('ping ', 'resolve ', 'whois ')):
        cmd, arg = code.split(' ', 1)
        if cmd == 'ping':
            result = NetInterface.ping(arg)
            return f"Ping to {arg}: {result} ms" if result is not None else f"Ping to {arg}: Unreachable"
        elif cmd == 'resolve':
            result = NetInterface.resolve(arg)
            return f"{arg} resolves to: {result}"
        elif cmd == 'whois':
            result = NetInterface.whois(arg)
            return f"WHOIS for {arg}: {result}"
    
    # Handle special commands
    if code == 'alice status':
        return json.dumps(AliceNode.status(), indent=2)
    
    if code == 'torino status':
        result = await TorinoQuantumBackend.get_backend_status()
        return json.dumps(result, indent=2)
    
    if code == 'lattice map':
        return json.dumps({
            "sagittarius_a": Config.SAGITTARIUS_A_LATTICE,
            "white_hole": Config.WHITE_HOLE_LATTICE,
            "alice_node": Config.ALICE_NODE_IP,
            "storage": Config.STORAGE_IP,
            "quantum_domain": Config.QUANTUM_DOMAIN
        }, indent=2)
    
    # Execute Python code
    old_stdout = sys.stdout
    output = []
    try:
        from io import StringIO
        sys.stdout = mystdout = StringIO()
        
        # Try eval first for expressions
        try:
            result = eval(code, ns)
            if result is not None:
                print(result)
        except SyntaxError:
            # If eval fails, use exec
            exec(code, ns)
        
        output.append(mystdout.getvalue())
    except Exception:
        output.append(traceback.format_exc())
    finally:
        sys.stdout = old_stdout
    
    repl_sessions[session_id] = ns
    return '\n'.join(output)

# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="QSH Foam Dominion - Production Quantum System",
    description="Production quantum email, blockchain integration with IBM Torino backend",
    version="3.0.0",
    debug=Config.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.on_event("startup")
async def startup_event():
    logger.info(f"QSH Foam Production System starting on {Config.HOST}:{Config.PORT}")
    logger.info(f"Sagittarius A* lattice anchor: {Config.SAGITTARIUS_A_LATTICE}")
    logger.info(f"White hole lattice: {Config.WHITE_HOLE_LATTICE}")
    logger.info(f"IBM Torino backend: {Config.IBM_BACKEND}")

# ==================== MAIN DASHBOARD ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    # Fetch live Torino metrics
    torino_status = await TorinoQuantumBackend.get_backend_status()
    torino_html = ""
    
    if 'error' not in torino_status:
        lattice_resonance = TorinoQuantumBackend.calculate_lattice_resonance(torino_status)
        torino_html = f"""
        <div class="status-item">
            <div class="label">IBM Torino</div>
            <div class="value">{torino_status.get('num_qubits', 0)} qubits • QV{torino_status.get('quantum_volume', 0)}</div>
        </div>
        <div class="status-item">
            <div class="label">Lattice Resonance</div>
            <div class="value">{lattice_resonance:.4f}</div>
        </div>
        <div class="status-item">
            <div class="label">T1 Coherence</div>
            <div class="value">{torino_status.get('t1_avg_us', 0):.2f} μs</div>
        </div>
        <div class="status-item">
            <div class="label">Gate Error</div>
            <div class="value">{torino_status.get('cx_error_avg', 0):.4f}</div>
        </div>"""
    else:
        torino_html = f"""
        <div class="status-item">
            <div class="label">IBM Torino</div>
            <div class="value">Configure Token</div>
        </div>"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QSH Foam Dominion v3.0 - Production</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #0f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            width: 100%;
        }}
        
        h1 {{
            text-align: center;
            color: #00ff9d;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 255, 157, 0.8);
        }}
        
        .subtitle {{
            text-align: center;
            color: #00ffff;
            margin-bottom: 40px;
            font-size: 1.2em;
        }}
        
        .lattice-info {{
            text-align: center;
            background: rgba(26, 26, 46, 0.8);
            border: 1px solid #ff6b35;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 30px;
        }}
        
        .lattice-info h3 {{
            color: #ff6b35;
            margin-bottom: 10px;
        }}
        
        .lattice-info p {{
            color: #aaa;
            font-size: 0.9em;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 25px;
            margin: 40px 0;
        }}
        
        .card {{
            background: rgba(26, 26, 46, 0.9);
            border: 2px solid #00ff9d;
            border-radius: 15px;
            padding: 30px;
            transition: all 0.3s;
            cursor: pointer;
        }}
        
        .card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0, 255, 157, 0.5);
            border-color: #00ffff;
        }}
        
        .card h2 {{
            color: #00ff9d;
            margin-bottom: 15px;
            font-size: 1.8em;
        }}
        
        .card p {{
            color: #ccc;
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        
        .card .features {{
            list-style: none;
            padding: 0;
        }}
        
        .card .features li {{
            color: #00ffff;
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }}
        
        .card .features li:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #ff6b35;
        }}
        
        .btn {{
            display: inline-block;
            background: #00ff9d;
            color: #000;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            text-decoration: none;
            transition: all 0.3s;
            cursor: pointer;
            font-family: 'Courier New', monospace;
        }}
        
        .btn:hover {{
            background: #00ffff;
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.5);
        }}
        
        .footer {{
            text-align: center;
            margin-top: 60px;
            color: #666;
        }}
        
        .status {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: rgba(0, 255, 157, 0.1);
            border: 1px solid #00ff9d;
            border-radius: 10px;
        }}
        
        .status h3 {{
            color: #00ff9d;
            margin-bottom: 15px;
        }}
        
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .status-item {{
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
        }}
        
        .status-item .label {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .status-item .value {{
            color: #00ffff;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .production-badge {{
            display: inline-block;
            background: #ff6b35;
            color: #000;
            padding: 5px 15px;
            border-radius: 5px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚛️ QSH Foam Dominion v3.0 <span class="production-badge">PRODUCTION</span></h1>
        <p class="subtitle">IBM Torino Backend • Sagittarius A* Lattice • Real Quantum Cryptography</p>
        
        <div class="lattice-info">
            <h3>🌌 Conceptual Lattice Network</h3>
            <p>Sagittarius A* Black Hole: {Config.SAGITTARIUS_A_LATTICE} (Encryption) ⇄ White Hole: {Config.WHITE_HOLE_LATTICE} (Decryption)</p>
            <p>IBM Torino Anchored via {Config.QUANTUM_DOMAIN} • QuTiP Resonance Entanglement</p>
        </div>
        
        <div class="status">
            <h3>Live System Status</h3>
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">Alice Node</div>
                    <div class="value">{Config.ALICE_NODE_IP} ✓</div>
                </div>
                <div class="status-item">
                    <div class="label">Sagittarius A*</div>
                    <div class="value">{Config.SAGITTARIUS_A_LATTICE}</div>
                </div>
                <div class="status-item">
                    <div class="label">White Hole</div>
                    <div class="value">{Config.WHITE_HOLE_LATTICE}</div>
                </div>
                <div class="status-item">
                    <div class="label">Holo Storage</div>
                    <div class="value">{Config.STORAGE_IP} (6EB)</div>
                </div>
                {torino_html}
            </div>
        </div>
        
        <div class="grid">
            <div class="card" onclick="location.href='/email'">
                <h2>📧 Quantum Email</h2>
                <p>Production cryptographic email via Sagittarius A* lattice routing</p>
                <ul class="features">
                    <li>Real Fernet encryption</li>
                    <li>Black hole → White hole routing</li>
                    <li>Lattice anchor: {Config.SAGITTARIUS_A_LATTICE}</li>
                    <li>QuTiP entanglement verification</li>
                </ul>
                <br>
                <a href="/email" class="btn">Open Email Client</a>
            </div>
            
            <div class="card" onclick="location.href='/blockchain'">
                <h2>₿ Bitcoin Client</h2>
                <p>Bitcoin Core integration with QSH Foam REPL</p>
                <ul class="features">
                    <li>Full Bitcoin RPC</li>
                    <li>Quantum-resistant routing</li>
                    <li>Real-time blockchain sync</li>
                    <li>Network diagnostics</li>
                </ul>
                <br>
                <a href="/blockchain" class="btn">Open Bitcoin Client</a>
            </div>
            
            <div class="card" onclick="location.href='/qsh'">
                <h2>🖥️ QSH Shell</h2>
                <p>Production quantum shell with IBM Torino integration</p>
                <ul class="features">
                    <li>QuTiP quantum operations</li>
                    <li>Real Torino backend access</li>
                    <li>Lattice routing commands</li>
                    <li>Python + network tools</li>
                </ul>
                <br>
                <a href="/qsh" class="btn">Open Shell</a>
            </div>
            
            <div class="card" onclick="location.href='/encryption'">
                <h2>🔐 Encryption Lab</h2>
                <p>Test black hole/white hole encryption routing</p>
                <ul class="features">
                    <li>Live encryption demo</li>
                    <li>Sagittarius A* routing</li>
                    <li>Fernet cryptography</li>
                    <li>Lattice visualization</li>
                </ul>
                <br>
                <a href="/encryption" class="btn">Open Encryption Lab</a>
            </div>
            
            <div class="card" onclick="location.href='/holo_search'">
                <h2>🔍 Holo Search</h2>
                <p>Holographic storage search @ {Config.STORAGE_IP}</p>
                <ul class="features">
                    <li>6 EB holographic capacity</li>
                    <li>Quantum-indexed search</li>
                    <li>Real-time lattice queries</li>
                    <li>Multi-dimensional indexing</li>
                </ul>
                <br>
                <a href="/holo_search" class="btn">Open Holo Search</a>
            </div>
            
            <div class="card" onclick="location.href='/networking'">
                <h2>🌐 Network Monitor</h2>
                <p>*.computer.networking domain routing</p>
                <ul class="features">
                    <li>Real-time ping/traceroute</li>
                    <li>Lattice node status</li>
                    <li>WHOIS lookups</li>
                    <li>Alice node @ {Config.ALICE_NODE_IP}</li>
                </ul>
                <br>
                <a href="/networking" class="btn">Open Network Monitor</a>
            </div>
        </div>
        
        <div class="footer">
           ```python
            <p>QSH Foam Dominion v3.0.0 | Production Quantum System</p>
            <p>IBM Torino: {Config.IBM_BACKEND} | Sagittarius A*: {Config.SAGITTARIUS_A_LATTICE} | quantum.realm.domain.dominion.foam.computer</p>
            <p>Real QuTiP Entanglement • Production Cryptography • Live Backend Metrics</p>
        </div>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# ==================== HTML PAGE ROUTES ====================

@app.get("/email", response_class=HTMLResponse)
async def email_page():
    html_path = Path(__file__).resolve().parent / "static" / "email.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Foam Email</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 15px 30px;
            border-bottom: 2px solid #00ff9d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5em;
            color: #00ff9d;
            font-weight: bold;
        }
        
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
            text-decoration: none;
        }
        
        .btn-primary {
            background: #00ff9d;
            color: #000;
        }
        
        .btn-secondary {
            background: transparent;
            border: 1px solid #00ffff;
            color: #00ffff;
        }
        
        .btn:hover {
            opacity: 0.8;
            transform: translateY(-2px);
        }
        
        .container {
            display: flex;
            height: calc(100vh - 65px);
        }
        
        .sidebar {
            width: 200px;
            background: #1a1a2e;
            border-right: 1px solid #333;
            padding: 20px 0;
        }
        
        .nav-item {
            padding: 12px 25px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .nav-item:hover {
            background: rgba(0, 255, 157, 0.1);
        }
        
        .nav-item.active {
            background: rgba(0, 255, 157, 0.2);
            color: #00ff9d;
            border-left: 3px solid #00ff9d;
        }
        
        .main-content {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            text-align: center;
        }
        
        .welcome {
            max-width: 600px;
            margin: 100px auto;
            padding: 40px;
            background: rgba(26, 26, 46, 0.9);
            border: 2px solid #00ff9d;
            border-radius: 15px;
        }
        
        .welcome h1 {
            color: #00ff9d;
            margin-bottom: 20px;
        }
        
        .info {
            color: #00ffff;
            margin: 15px 0;
            line-height: 1.8;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">⚛️ Quantum Foam Email</div>
        <div class="nav-buttons">
            <a href="/" class="btn btn-secondary">← Dashboard</a>
            <a href="/blockchain" class="btn btn-secondary">₿ Bitcoin</a>
            <a href="/qsh" class="btn btn-secondary">🔬 REPL</a>
            <button class="btn btn-primary" onclick="alert('Login coming soon')">Login</button>
        </div>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="nav-item active">📥 Inbox</div>
            <div class="nav-item">📤 Sent</div>
            <div class="nav-item">⭐ Starred</div>
            <div class="nav-item">🗑️ Trash</div>
        </div>

        <div class="main-content">
            <div class="welcome">
                <h1>📧 Quantum Foam Email</h1>
                <div class="info">
                    <strong>Storage:</strong> Holographic @ 138.0.0.1<br>
                    <strong>Domain:</strong> @quantum.foam<br>
                    <strong>Capacity:</strong> 10GB per user<br>
                    <strong>Network:</strong> 137.0.0.x blocks<br>
                </div>
                <br>
                <p style="color: #888;">Login system and full email client interface will be implemented here.</p>
                <br>
                <button class="btn btn-primary" onclick="alert('Registration will be integrated with /api/register endpoint')">Create Account</button>
            </div>
        </div>
    </div>
</body>
</html>
    """)

@app.get("/blockchain", response_class=HTMLResponse)
async def blockchain_page():
    html_path = Path(__file__).resolve().parent / "static" / "blockchain.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin Client - QSH Foam</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }
        .back-btn {
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .info {
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>₿ Bitcoin Client</h1>
        <div class="info">
            <h2>Bitcoin Core Integration</h2>
            <p>Full Bitcoin RPC client interface coming soon</p>
        </div>
    </div>
</body>
</html>
    """)

@app.get("/encryption", response_class=HTMLResponse)
async def encryption_page():
    html_path = Path(__file__).resolve().parent / "static" / "encryption.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Encryption Lab - QSH Foam</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }
        .back-btn {
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .info {
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>🔐 Encryption Lab</h1>
        <div class="info">
            <h2>Black Hole / White Hole Encryption</h2>
            <p>Sagittarius A* Lattice Encryption Lab coming soon</p>
        </div>
    </div>
</body>
</html>
    """)

@app.get("/holo_search", response_class=HTMLResponse)
async def holo_search_page():
    html_path = Path(__file__).resolve().parent / "static" / "holo_search.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Holo Search - QSH Foam</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }
        .back-btn {
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .info {
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>🔍 Holo Search</h1>
        <div class="info">
            <h2>Holographic Storage Search</h2>
            <p>6 EB Holographic Storage @ 138.0.0.1</p>
            <p>Search interface coming soon</p>
        </div>
    </div>
</body>
</html>
    """)

@app.get("/networking", response_class=HTMLResponse)
async def networking_page():
    html_path = Path(__file__).resolve().parent / "static" / "networking.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Monitor - QSH Foam</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff9d;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00ff9d;
            margin-bottom: 30px;
        }
        .back-btn {
            display: inline-block;
            background: #00ffff;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .info {
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #ff6b35;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h1>🌐 Network Monitor</h1>
        <div class="info">
            <h2>*.computer.networking Domain Routing</h2>
            <p>Network monitoring interface coming soon</p>
        </div>
    </div>
</body>
</html>
    """)

# ==================== 404 HANDLER ====================
@app.exception_handler(status.HTTP_404_NOT_FOUND)
async def not_found_handler(request: Request, exc: HTTPException):
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Not Found | QSH Foam Dominion</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #0f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            text-align: center;
            max-width: 600px;
        }}
        h1 {{
            color: #ff6b35;
            font-size: 4em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(255, 107, 53, 0.8);
        }}
        p {{
            color: #00ffff;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        a {{
            display: inline-block;
            background: #00ff9d;
            color: #000;
            padding: 12px 24px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
        }}
        a:hover {{
            background: #00ffff;
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.5);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>404</h1>
        <p>Quantum Entanglement Lost - The requested lattice route does not exist in the foam.</p>
        <p>Lattice Anchor: {Config.SAGITTARIUS_A_LATTICE}</p>
        <a href="/">Return to QSH Foam Dominion</a>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=404)

# ==================== API ROUTES ====================

@app.post("/api/register", tags=["auth"])
async def register(user: UserRegister):
    result = Database.create_user(user.username, user.password)
    return result

@app.post("/api/login", tags=["auth"])
async def login(user: UserLogin):
    auth_user = Database.authenticate_user(user.username, user.password)
    
    if not auth_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = Database.create_session(auth_user['email'])
    
    response = JSONResponse(content={"message": "Login successful", "user": auth_user})
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=7*24*60*60,
        samesite="lax"
    )
    
    return response

@app.post("/api/logout", tags=["auth"])
async def logout():
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie("session_token")
    return response

@app.get("/api/emails/inbox", tags=["email"])
async def get_inbox(user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.get_inbox(user['email'])

@app.get("/api/emails/sent", tags=["email"])
async def get_sent(user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.get_sent(user['email'])

@app.post("/api/emails/send", tags=["email"])
async def send_email(email: EmailCreate, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return Database.send_email(user['email'], email.to, email.subject, email.body)

@app.post("/api/emails/{email_id}/read", tags=["email"])
async def mark_read(email_id: int, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.mark_as_read(email_id, user['email'])
    return {"message": "Marked as read"}

@app.post("/api/emails/{email_id}/star", tags=["email"])
async def toggle_star_email(email_id: int, user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.toggle_star(email_id, user['email'])
    return {"message": "Star toggled"}

@app.post("/api/emails/delete", tags=["email"])
async def delete_emails_route(email_ids: Dict[str, List[int]], user: dict = Depends(get_current_user_email)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    Database.delete_emails(email_ids['email_ids'], user['email'])
    return {"message": "Emails deleted"}

# ==================== ENCRYPTION API ====================

@app.post("/api/encrypt", tags=["encryption"])
async def encrypt_text(data: Dict[str, str]):
    plaintext = data.get('plaintext', '')
    encrypted = QuantumEncryption.encrypt_via_sagittarius_lattice(plaintext)
    return {
        "encrypted": encrypted.hex(),
        "lattice_route": Config.SAGITTARIUS_A_LATTICE,
        "algorithm": "Fernet"
    }

@app.post("/api/decrypt", tags=["encryption"])
async def decrypt_text(data: Dict[str, str]):
    try:
        ciphertext = bytes.fromhex(data.get('ciphertext', ''))
        decrypted = QuantumEncryption.decrypt_via_whitehole_lattice(ciphertext)
        return {
            "decrypted": decrypted,
            "lattice_route": Config.WHITE_HOLE_LATTICE
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")

# ==================== QSH REPL ROUTES ====================

@app.get("/qsh", tags=["repl"])
async def qsh_repl():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>QSH Foam REPL v3.0</title>
    <script src="https://cdn.jsdelivr.net/npm/xterm@5.5.0/lib/xterm.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.5.0/css/xterm.css" />
    <style> 
        body { margin: 0; padding: 0; background: #000; } 
        #terminal { width: 100vw; height: 100vh; }
        .header {
            background: #1a1a2e;
            padding: 10px 20px;
            color: #00ff9d;
            font-family: monospace;
            border-bottom: 2px solid #00ff9d;
        }
        .prod-badge {
            background: #ff6b35;
            color: #000;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        QSH Foam REPL v3.0 <span class="prod-badge">PRODUCTION</span> | 
        <a href="/" style="color: #00ffff; text-decoration: none;">← Dashboard</a> | 
        IBM Torino Connected | Sagittarius A* Lattice Active
    </div>
    <div id="terminal"></div>
    <script>
        const term = new Terminal({ cols: 120, rows: 40, theme: { background: '#000000', foreground: '#00ff00' } });
        term.open(document.getElementById('terminal'));
        term.write('QSH Foam REPL v3.0 [PRODUCTION]\\r\\n');
        term.write('IBM Torino Backend Connected\\r\\n');
        term.write('Lattice: Sagittarius A* (130.0.0.1) <-> White Hole (139.0.0.1)\\r\\n');
        term.write('Commands: alice status | torino status | lattice map | ping <ip> | QuTiP operations\\r\\n');
        term.write('QSH> ');

        const ws = new WebSocket('ws://' + location.host + '/ws/repl');
        ws.onopen = () => term.write('[Connected]\\r\\nQSH> ');
        ws.onmessage = (event) => term.write(event.data + '\\r\\nQSH> ');

        let buffer = '';
        term.onData(data => {
            if (data === '\\r') {
                if (buffer.trim()) ws.send(buffer.trim());
                term.write('\\r\\n');
                buffer = '';
            } else if (data === '\\u007F') {
                if (buffer.length > 0) {
                    buffer = buffer.slice(0, -1);
                    term.write('\\b \\b');
                }
            } else {
                buffer += data;
                term.write(data);
            }
        });
    </script>
</body>
</html>
    """)

@app.websocket("/ws/repl")
async def websocket_repl(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    repl_sessions[session_id] = {}
    
    try:
        while True:
            data = await websocket.receive_text()
            output = await repl_exec(data, session_id)
            await websocket.send_text(output)
    except WebSocketDisconnect:
        logger.info(f"QSH REPL session {session_id} disconnected")
        del repl_sessions[session_id]

# ==================== QUANTUM & METRICS ROUTES ====================

@app.get("/quantum/suite", tags=["quantum"])
async def get_quantum_suite(request: Request):
    await check_rate_limit(request)
    return await QuantumPhysics.run_full_suite()

@app.get("/quantum/bell", tags=["quantum"])
async def get_bell_test(request: Request, shots: int = Query(8192)):
    await check_rate_limit(request)
    return QuantumPhysics.bell_experiment_qutip(shots)

@app.get("/quantum/ghz", tags=["quantum"])
async def get_ghz_test(request: Request, shots: int = Query(8192)):
    await check_rate_limit(request)
    return QuantumPhysics.ghz_experiment_qutip(shots)

@app.get("/quantum/teleportation", tags=["quantum"])
async def get_teleportation(request: Request, shots: int = Query(4096)):
    await check_rate_limit(request)
    return QuantumPhysics.quantum_teleportation_qutip(shots)

@app.get("/quantum/torino", tags=["quantum"])
async def get_torino_status(request: Request):
    await check_rate_limit(request)
    return await TorinoQuantumBackend.get_backend_status()

@app.get("/metrics", tags=["system"])
async def get_metrics(request: Request):
    await check_rate_limit(request)
    return await SystemMetrics.get_all_metrics()

@app.get("/metrics/lattice", tags=["system"])
async def get_lattice_map():
    return {
        "sagittarius_a_black_hole": {
            "ip": Config.SAGITTARIUS_A_LATTICE,
            "function": "Encryption ingestion",
            "backend": "IBM Torino conceptual anchor"
        },
        "white_hole": {
            "ip": Config.WHITE_HOLE_LATTICE,
            "function": "Decryption expansion"
        },
        "alice_node": {
            "ip": Config.ALICE_NODE_IP,
            "function": "Local quantum operations"
        },
        "storage": {
            "ip": Config.STORAGE_IP,
            "capacity_eb": Config.HOLOGRAPHIC_CAPACITY_EB
        },
        "quantum_domain": Config.QUANTUM_DOMAIN,
        "network_domain": Config.COMPUTER_NETWORK_DOMAIN
    }

@app.get("/health", tags=["info"])
async def health():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "environment": Config.ENVIRONMENT,
        "lattice_active": True,
        "torino_configured": bool(Config.IBM_QUANTUM_TOKEN)
    }

# ==================== START SERVER ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting QSH Foam Dominion on 0.0.0.0:{port}")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info" if not Config.DEBUG else "debug"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)
