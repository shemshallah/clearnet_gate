import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import httpx
import asyncio
from contextlib import asynccontextmanager

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if os.path.exists('/app') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment
CHAT_BACKEND = os.getenv("CHAT_BACKEND_URL", "https://clearnet-chat-4bal.onrender.com")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# Global state management
class AppState:
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.backend_health: bool = False
        self.last_health_check: datetime = datetime.now()
        self.request_counts: Dict[str, int] = {}
        self.active_connections: int = 0
        
    async def initialize(self):
        """Initialize application state"""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(TIMEOUT),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        await self.check_backend_health()
        logger.info("Application state initialized")
    
    async def shutdown(self):
        """Cleanup application state"""
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Application state cleaned up")
    
    async def check_backend_health(self) -> bool:
        """Check if backend is healthy"""
        try:
            if self.http_client:
                response = await self.http_client.get(f"{CHAT_BACKEND}/health", timeout=5.0)
                self.backend_health = response.status_code == 200
                self.last_health_check = datetime.now()
                return self.backend_health
        except Exception as e:
            logger.error(f"Backend health check failed: {e}")
            self.backend_health = False
        return False

app_state = AppState()

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting application...")
    await app_state.initialize()
    
    # Start background tasks
    background_task = asyncio.create_task(periodic_health_check())
    
    yield
    
    # Cleanup
    background_task.cancel()
    await app_state.shutdown()
    logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Clearnet Gate - Quantum Foam Gateway",
    description="Production-grade frontend proxy and static server for Quantum Foam Network",
    version="3.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if DEBUG else [CHAT_BACKEND],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if not DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure properly in production
    )

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting"""
    client_ip = request.client.host
    current_minute = datetime.now().replace(second=0, microsecond=0)
    key = f"{client_ip}:{current_minute}"
    
    if key not in app_state.request_counts:
        app_state.request_counts[key] = 0
    
    app_state.request_counts[key] += 1
    
    if app_state.request_counts[key] > RATE_LIMIT_PER_MINUTE:
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

# Logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} in {duration:.3f}s")
    
    return response

# Background tasks
async def periodic_health_check():
    """Periodically check backend health"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            healthy = await app_state.check_backend_health()
            status = "healthy" if healthy else "unhealthy"
            logger.info(f"Backend health check: {status}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health check error: {e}")

# Proxy helper with retry logic
async def proxy_to_backend(request: Request, path: str, max_retries: int = MAX_RETRIES) -> JSONResponse:
    """Proxy requests to backend with retry logic"""
    if not app_state.http_client:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    backend_url = f"{CHAT_BACKEND}/{path}"
    
    # Prepare request parameters
    params = dict(request.query_params)
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ["host", "content-length", "connection"]
    }
    
    # Get request body if applicable
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
    
    # Retry logic
    last_error = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"Proxying {request.method} to {backend_url} (attempt {attempt + 1}/{max_retries})")
            
            response = await app_state.http_client.request(
                method=request.method,
                url=backend_url,
                params=params,
                content=body,
                headers=headers
            )
            
            # Success - return response
            return JSONResponse(
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else {"data": response.text},
                status_code=response.status_code,
                headers={k: v for k, v in response.headers.items() if k.lower() not in ["content-encoding", "content-length", "transfer-encoding"]}
            )
            
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        except httpx.RequestError as e:
            last_error = e
            logger.error(f"Request error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                
        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            break
    
    # All retries failed
    logger.error(f"All {max_retries} attempts failed for {backend_url}: {last_error}")
    raise HTTPException(
        status_code=502,
        detail=f"Backend unavailable after {max_retries} attempts: {str(last_error)}"
    )

# Main routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main chat HTML page with full featured interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clearnet Chat - Quantum Foam Chatroom</title>
    <meta name="description" content="Secure quantum-encrypted chat platform">
    <meta name="theme-color" content="#0f0f23">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            width: 100%;
            background: rgba(0, 0, 0, 0.85);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 0 40px rgba(0, 255, 255, 0.2);
            border: 1px solid rgba(0, 255, 255, 0.3);
        }
        
        h1 {
            text-align: center;
            color: #00ffff;
            text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff;
            font-size: 2.5em;
            margin-bottom: 10px;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff; }
            to { text-shadow: 0 0 30px #00ffff, 0 0 60px #00ffff, 0 0 80px #00ffff; }
        }
        
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(0, 255, 255, 0.1);
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.85em;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #00ff00;
            animation: pulse 2s infinite;
        }
        
        .status-dot.disconnected { background: #ff0000; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        input, button, textarea {
            padding: 12px 15px;
            border: 1px solid #00ffff;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.5);
            color: #e0e0e0;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        input:focus, textarea:focus {
            outline: none;
            border-color: #00ffff;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
            background: rgba(0, 0, 0, 0.7);
        }
        
        button {
            cursor: pointer;
            background: linear-gradient(135deg, #00ffff, #00cccc);
            color: #000;
            font-weight: bold;
            border: none;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        button:hover {
            background: linear-gradient(135deg, #00cccc, #009999);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
            transform: translateY(-2px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button.secondary {
            background: rgba(0, 255, 255, 0.2);
            color: #00ffff;
            border: 1px solid #00ffff;
        }
        
        button.secondary:hover {
            background: rgba(0, 255, 255, 0.3);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .chat-container {
            border: 1px solid #00ffff;
            height: 500px;
            overflow-y: auto;
            padding: 15px;
            background: rgba(0, 0, 0, 0.6);
            border-radius: 8px;
            margin-bottom: 15px;
            scroll-behavior: smooth;
        }
        
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.3);
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: #00ffff;
            border-radius: 4px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            animation: slideIn 0.3s ease;
            position: relative;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.sent {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.2), rgba(0, 255, 255, 0.1));
            text-align: right;
            margin-left: 20%;
            border: 1px solid rgba(0, 255, 255, 0.3);
        }
        
        .message.received {
            background: linear-gradient(135deg, rgba(255, 0, 255, 0.2), rgba(255, 0, 255, 0.1));
            margin-right: 20%;
            border: 1px solid rgba(255, 0, 255, 0.3);
        }
        
        .message.ai {
            background: linear-gradient(135deg, rgba(255, 165, 0, 0.3), rgba(255, 140, 0, 0.2));
            font-style: italic;
            border: 1px solid rgba(255, 165, 0, 0.4);
            margin-left: 10%;
            margin-right: 10%;
        }
        
        .message.system {
            background: rgba(128, 128, 128, 0.2);
            text-align: center;
            font-size: 0.85em;
            color: #888;
            border: 1px dashed #555;
        }
        
        .message-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.85em;
        }
        
        .message-sender {
            font-weight: bold;
            color: #00ffff;
        }
        
        .message-time {
            color: #888;
            font-size: 0.9em;
        }
        
        .message-content {
            word-wrap: break-word;
            line-height: 1.5;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        #messageInput {
            flex: 1;
        }
        
        #sendButton {
            width: 120px;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .button-group button {
            flex: 1;
        }
        
        .hidden {
            display: none !important;
        }
        
        .error {
            color: #ff6b6b;
            background: rgba(255, 0, 0, 0.1);
            padding: 10px;
            border-radius: 5px;
            border: 1px solid rgba(255, 0, 0, 0.3);
            margin-top: 10px;
            text-align: center;
        }
        
        .success {
            color: #51cf66;
            background: rgba(0, 255, 0, 0.1);
            padding: 10px;
            border-radius: 5px;
            border: 1px solid rgba(0, 255, 0, 0.3);
            margin-top: 10px;
            text-align: center;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px;
            color: #888;
            font-style: italic;
            font-size: 0.9em;
        }
        
        .typing-indicator.active {
            display: block;
        }
        
        .user-count {
            display: flex;
            align-items: center;
            gap: 5px;
            color: #00ffff;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 1.8em;
            }
            
            .chat-container {
                height: 400px;
            }
            
            .message.sent,
            .message.received {
                margin-left: 0;
                margin-right: 0;
            }
            
            .input-group {
                flex-direction: column;
            }
            
            #sendButton {
                width: 100%;
            }
        }
        
        .loader {
            border: 3px solid rgba(0, 255, 255, 0.3);
            border-top: 3px solid #00ffff;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌊 Quantum Foam Chatroom</h1>
        <p class="subtitle">Secure End-to-End Encrypted Communication</p>
        
        <!-- Status Bar -->
        <div class="status-bar">
            <div class="status-indicator">
                <span class="status-dot" id="connectionStatus"></span>
                <span id="connectionText">Connecting...</span>
            </div>
            <div class="user-count">
                <span>👥</span>
                <span id="userCount">0</span>
                <span>online</span>
            </div>
        </div>
        
        <!-- Login/Register Form -->
        <div id="authForm">
            <h2 id="formTitle">Login</h2>
            <form id="authFormElement" onsubmit="return false;">
                <input type="text" id="username" placeholder="Username" required autocomplete="username">
                <input type="password" id="password" placeholder="Password" required autocomplete="current-password">
                <input type="email" id="email" placeholder="Email (for registration)" class="hidden" autocomplete="email">
                <label style="display: flex; align-items: center; gap: 10px; color: #888;">
                    <input type="checkbox" id="rememberMe" style="width: auto;">
                    <span>Remember Me (30 days)</span>
                </label>
                <button type="submit" id="submitBtn">Login</button>
                <button type="button" class="secondary" id="toggleForm">Need an account? Register</button>
            </form>
            <div id="authMessage"></div>
        </div>

        <!-- Chat Interface -->
        <div id="chatInterface" class="hidden">
            <div class="chat-container" id="messages">
                <div class="message system">Welcome to Quantum Foam Chat! Your messages are encrypted.</div>
            </div>
            <div class="typing-indicator" id="typingIndicator">Someone is typing...</div>
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Type a message..." autocomplete="off">
                <button id="sendButton">Send 📤</button>
            </div>
            <div class="button-group">
                <button class="secondary" id="clearChat">Clear Chat</button>
                <button class="secondary" id="logoutBtn">Logout 🚪</button>
            </div>
        </div>
    </div>

    <script>
        // Configuration
        const API_BASE = '/api';
        const WS_URL = 'wss://clearnet-chat-4bal.onrender.com/ws/chat';
        
        // State management
        let ws = null;
        let token = localStorage.getItem('token') || null;
        let currentUser = null;
        let reconnectAttempts = 0;
        let maxReconnectAttempts = 5;
        let reconnectDelay = 1000;
        let messageQueue = [];
        let isTyping = false;
        let typingTimeout = null;
        
        // DOM elements
        const elements = {
            authForm: document.getElementById('authForm'),
            chatInterface: document.getElementById('chatInterface'),
            formTitle: document.getElementById('formTitle'),
            username: document.getElementById('username'),
            password: document.getElementById('password'),
            email: document.getElementById('email'),
            rememberMe: document.getElementById('rememberMe'),
            submitBtn: document.getElementById('submitBtn'),
            toggleForm: document.getElementById('toggleForm'),
            authMessage: document.getElementById('authMessage'),
            messages: document.getElementById('messages'),
            messageInput: document.getElementById('messageInput'),
            sendButton: document.getElementById('sendButton'),
            logoutBtn: document.getElementById('logoutBtn'),
            clearChat: document.getElementById('clearChat'),
            connectionStatus: document.getElementById('connectionStatus'),
            connectionText: document.getElementById('connectionText'),
            userCount: document.getElementById('userCount'),
            typingIndicator: document.getElementById('typingIndicator')
        };
        
        // Utility functions
        function showMessage(message, type = 'error') {
            elements.authMessage.className = type;
            elements.authMessage.textContent = message;
            elements.authMessage.classList.remove('hidden');
            setTimeout(() => {
                elements.authMessage.classList.add('hidden');
            }, 5000);
        }
        
        function updateConnectionStatus(connected) {
            if (connected) {
                elements.connectionStatus.classList.remove('disconnected');
                elements.connectionText.textContent = 'Connected';
            } else {
                elements.connectionStatus.classList.add('disconnected');
                elements.connectionText.textContent = 'Disconnected';
            }
        }
        
        function formatTimestamp(date = new Date()) {
            return date.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit'
            });
        }
        
        function addMessage(msg) {
            const messageDiv = document.createElement('div');
            const isOwn = currentUser && (msg.sender === currentUser.username || msg.sender_id === currentUser.id);
            
            let messageClass = 'message ';
            if (msg.type === 'system') {
                messageClass += 'system';
            } else if (msg.is_ai) {
                messageClass += 'ai';
            } else if (isOwn) {
                messageClass += 'sent';
            } else {
                messageClass += 'received';
            }
            
            messageDiv.className = messageClass;
            
            if (msg.type === 'system') {
                messageDiv.innerHTML = `<div class="message-content">${escapeHtml(msg.content || msg.body)}</div>`;
            } else {
                messageDiv.innerHTML = `
                    <div class="message-header">
                        <span class="message-sender">${escapeHtml(msg.sender || 'Unknown')}</span>
                        <span class="message-time">${msg.timestamp || formatTimestamp()}</span>
                    </div>
                    <div class="message-content">${escapeHtml(msg.content || msg.body)}</div>
                `;
            }
            
            elements.messages.appendChild(messageDiv);
            elements.messages.scrollTop = elements.messages.scrollHeight;
            
            // Limit messages in DOM to prevent memory issues
            const messages = elements.messages.querySelectorAll('.message');
            if (messages.length > 100) {
                messages[0].remove();
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Authentication functions
        elements.toggleForm.onclick = () => {
            const isLogin = elements.formTitle.textContent === 'Login';
            elements.formTitle.textContent = isLogin ? 'Register' : 'Login';
            elements.email.classList.toggle('hidden');
            elements.submitBtn.textContent = isLogin ? 'Register' : 'Login';
            elements.toggleForm.textContent = isLogin ? 'Already have an account? Login' : 'Need an account? Register';
            elements.authMessage.classList.add('hidden');
        };
        
        elements.authFormElement.onsubmit = async (e) => {
            e.preventDefault();
            await handleAuth();
        };
        
        elements.submitBtn.onclick = handleAuth;
        
        async function handleAuth() {
            const isRegister = elements.formTitle.textContent === 'Register';
            const username = elements.username.value.trim();
            const password = elements.password.value.trim();
            const email = elements.email.value.trim();
            const rememberMe = elements.rememberMe.checked;
            
            if (!username || !password) {
                showMessage('Please fill in all required fields', 'error');
                return;
            }
            
            if (isRegister && !email) {
                showMessage('Email is required for registration', 'error');
                return;
            }
            
            elements.submitBtn.disabled = true;
            elements.submitBtn.innerHTML = 'Processing... <span class="loader"></span>';
            
            try {
                const endpoint = isRegister ? '/register' : '/login';
                const body = isRegister 
                    ? { username, password, email }
                    : { username, password, remember_me: rememberMe };
                
                const response = await fetch(`${API_BASE}${endpoint}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    token = data.token;
                    currentUser = {
                        id: data.user_id || data.id,
                        username: data.username || username,
                        email: data.email || email
                    };
                    
                    if (rememberMe) {
                        localStorage.setItem('token', token);
                        localStorage.setItem('user', JSON.stringify(currentUser));
                    } else {
                        sessionStorage.setItem('token', token);
                        sessionStorage.setItem('user', JSON.stringify(currentUser));
                    }
                    
                    showMessage(isRegister ? 'Registration successful!' : 'Login successful!', 'success');
                    setTimeout(() => showChat(), 1000);
                } else {
                    showMessage(data.message || data.detail || 'Authentication failed', 'error');
                }
            } catch (error) {
                console.error('Auth error:', error);
                showMessage('Network error: ' + error.message, 'error');
            } finally {
                elements.submitBtn.disabled = false;
                elements.submitBtn.textContent = isRegister ? 'Register' : 'Login';
            }
        }
        
        // Chat functions
        function showChat() {
            elements.authForm.classList.add('hidden');
            elements.chatInterface.classList.remove('hidden');
            connectWebSocket();
        }
        
        function connectWebSocket() {
            if (!token) {
                console.error('No token available');
                return;
            }
            
            try {
                ws = new WebSocket(`${WS_URL}?token=${token}`);
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                    updateConnectionStatus(true);
                    reconnectAttempts = 0;
                    reconnectDelay = 1000;
                    
                    addMessage({
                        type: 'system',
                        content: 'Connected to chat server',
                        timestamp: formatTimestamp()
                    });
                    
                    // Send queued messages
                    while (messageQueue.length > 0) {
                        const msg = messageQueue.shift();
                        ws.send(JSON.stringify(msg));
                    }
                };
                
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'user_count') {
                            elements.userCount.textContent = data.count || 0;
                        } else if (data.type === 'typing') {
                            if (data.username !== currentUser.username) {
                                elements.typingIndicator.textContent = `${data.username} is typing...`;
                                elements.typingIndicator.classList.add('active');
                                setTimeout(() => {
                                    elements.typingIndicator.classList.remove('active');
                                }, 3000);
                            }
                        } else {
                            addMessage(data);
                        }
                    } catch (error) {
                        console.error('Error parsing message:', error);
                    }
                };
                
                ws.onclose = (event) => {
                    console.log('WebSocket disconnected:', event.code, event.reason);
                    updateConnectionStatus(false);
                    
                    if (!event.wasClean && reconnectAttempts < maxReconnectAttempts) {
                        addMessage({
                            type: 'system',
                            content: `Connection lost. Reconnecting in ${reconnectDelay / 1000}s...`,
                            timestamp: formatTimestamp()
                        });
                        
                        setTimeout(() => {
                            reconnectAttempts++;
                            reconnectDelay *= 2;
                            connectWebSocket();
                        }, reconnectDelay);
                    } else if (reconnectAttempts >= maxReconnectAttempts) {
                        addMessage({
                            type: 'system',
                            content: 'Failed to reconnect. Please refresh the page.',
                            timestamp: formatTimestamp()
                        });
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    updateConnectionStatus(false);
                };
                
            } catch (error) {
                console.error('Error creating WebSocket:', error);
                updateConnectionStatus(false);
            }
        }
        
        function sendMessage() {
            const content = elements.messageInput.value.trim();
            
            if (!content) return;
            
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                showMessage('Not connected to server', 'error');
                messageQueue.push({ content });
                return;
            }
            
            try {
                ws.send(JSON.stringify({ content }));
                elements.messageInput.value = '';
                
                // Stop typing indicator
                clearTimeout(typingTimeout);
                isTyping = false;
            } catch (error) {
                console.error('Error sending message:', error);
                showMessage('Failed to send message', 'error');
                messageQueue.push({ content });
            }
        }
        
        // Event listeners
        elements.sendButton.onclick = sendMessage;
        
        elements.messageInput.onkeypress = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        };
        
        elements.messageInput.oninput = () => {
            if (!isTyping && ws && ws.readyState === WebSocket.OPEN) {
                isTyping = true;
                ws.send(JSON.stringify({ type: 'typing' }));
            }
            
            clearTimeout(typingTimeout);
            typingTimeout = setTimeout(() => {
                isTyping = false;
            }, 3000);
        };
        
        elements.clearChat.onclick = () => {
            if (confirm('Are you sure you want to clear all messages?')) {
                elements.messages.innerHTML = '<div class="message system">Chat cleared</div>';
            }
        };
        
        elements.logoutBtn.onclick = () => {
            if (confirm('Are you sure you want to logout?')) {
                localStorage.removeItem('token');
                localStorage.removeItem('user');
                sessionStorage.removeItem('token');
                sessionStorage.removeItem('user');
                
                token = null;
                currentUser = null;
                
                if (ws) {
                    ws.close();
                    ws = null;
                }
                
                elements.chatInterface.classList.add('hidden');
                elements.authForm.classList.remove('hidden');
                elements.messages.innerHTML = '<div class="message system">Welcome to Quantum Foam Chat! Your messages are encrypted.</div>';
                elements.username.value = '';
                elements.password.value = '';
                elements.email.value = '';
                
                updateConnectionStatus(false);
            }
        };
        
        // Auto-login if token exists
        if (token) {
            try {
                currentUser = JSON.parse(localStorage.getItem('user') || sessionStorage.getItem('user'));
                showChat();
            } catch (error) {
                console.error('Error parsing stored user:', error);
                localStorage.removeItem('token');
                localStorage.removeItem('user');
                sessionStorage.removeItem('token');
                sessionStorage.removeItem('user');
                token = null;
            }
        }
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && token && (!ws || ws.readyState !== WebSocket.OPEN)) {
                console.log('Page visible, reconnecting...');
                connectWebSocket();
            }
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (ws) {
                ws.close();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    backend_healthy = await app_state.check_backend_health()
    
    return {
        "status": "healthy" if backend_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "environment": ENVIRONMENT,
        "backend": {
            "url": CHAT_BACKEND,
            "healthy": backend_healthy,
            "last_check": app_state.last_health_check.isoformat()
        },
        "metrics": {
            "active_connections": app_state.active_connections,
            "rate_limited_ips": len(app_state.request_counts)
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    if not DEBUG:
        raise HTTPException(status_code=404)
    
    return {
        "active_connections": app_state.active_connections,
        "backend_healthy": app_state.backend_health,
        "rate_limit_entries": len(app_state.request_counts),
        "last_health_check": app_state.last_health_check.isoformat()
    }

# API proxy routes
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_api(path: str, request: Request):
    """Proxy all API calls to chat backend with retry logic"""
    try:
        return await proxy_to_backend(request, f"api/{path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected proxy error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# WebSocket proxy info endpoint
@app.get("/ws/info")
async def websocket_info():
    """Provide WebSocket connection information"""
    return {
        "websocket_url": f"wss://{CHAT_BACKEND.replace('https://', '')}/ws/chat",
        "protocol": "wss",
        "authentication": "token",
        "description": "Connect with ?token=YOUR_TOKEN parameter"
    }

# Fallback routes
@app.get("/favicon.ico")
async def favicon():
    """Return 204 for favicon requests"""
    return JSONResponse(content={}, status_code=204)

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={"error": "Endpoint not found", "path": request.url.path}
        )
    return RedirectResponse(url="/", status_code=302)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc) if DEBUG else "An error occurred"}
    )

# Startup message
@app.on_event("startup")
async def startup_message():
    """Log startup information"""
    logger.info("=" * 60)
    logger.info("🚀 Clearnet Gate - Quantum Foam Gateway")
    logger.info(f"📍 Version: 3.0.0")
    logger.info(f"🌍 Environment: {ENVIRONMENT}")
    logger.info(f"🔗 Backend: {CHAT_BACKEND}")
    logger.info(f"🐛 Debug: {DEBUG}")
    logger.info("=" * 60)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="debug" if DEBUG else "info",
        access_log=True,
        use_colors=True
    )
