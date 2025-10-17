from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import json
from datetime import timedelta
from app.database import get_db, engine, SessionLocal.
from .models import Base, User, Message, Email, Contact, collider_black_hole_hash, encrypt_with_collider
from .auth import create_access_token, get_current_user
from .collider import create_black_hole_hash
import random
import os

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Quantum Foam Chatroom")

# CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static for React build (prod)
app.mount("/static", StaticFiles(directory="../frontend/dist"), name="static")

# Templates for server-rendered pages (login/register)
templates = Jinja2Templates(directory="templates")

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}  # user_id -> websocket

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def broadcast_to_matches(self, message: str, sender_labels: str):
        for uid, ws in self.active_connections.items():
            # Entanglement match
            if any(label in sender_labels for label in db.query(User).filter(User.id == uid).first().labels.split(',')):
                await ws.send_text(message)

manager = ConnectionManager()

# Front page / login
@app.get("/", response_class=HTMLResponse)
async def front_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# Registration
@app.post("/api/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    domain: str = Form("quantum"),  # Choice of domains
    labels: str = Form(""),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username taken")
    user = User(username=username, domain=domain, labels=labels.strip())
    user.set_password(password)
    user.email = user.full_email
    db.add(user)
    db.commit()
    db.refresh(user)
    # Token
    token = create_access_token(data={"sub": username}, expires_delta=timedelta(days=7))
    return {"token": token, "email": user.email, "message": "Welcome to Foam Computer! Your messages are secured via quantum colliders—black hole hashes retrieved from white holes."}

# Login
@app.post("/api/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False),
    request: Request = None,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.verify_password(password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    expires = timedelta(days=7) if remember_me else timedelta(minutes=30)
    token = create_access_token(data={"sub": username}, expires_delta=expires)
    response = RedirectResponse(url="/chat", status_code=303)
    if remember_me:
        response.set_cookie(key="remember_ip", value=request.client.host, httponly=True, max_age=604800)  # 7 days
    response.headers["Authorization"] = f"Bearer {token}"
    return response

# Forget password
@app.post("/api/forget-password")
async def forget_password(username: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_pass = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$', k=16))
    user.set_password(new_pass)
    db.commit()
    return {"new_password": new_pass, "message": "Password reset. Change it immediately in settings. Secured via Argon2 quantum-resistant hashing."}

# WebSocket for IM/Chat (entanglement spawn)
@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket, token: str = Depends(get_current_user)):
    user = get_current_user(token=token)  # From auth
    await manager.connect(websocket, user.id)
    try:
        while True:
            data = await websocket.receive_text()
            parsed = json.loads(data)
            content = parsed["content"]
            receiver_id = parsed.get("receiver_id")
            # Encrypt with collider
            encrypted, bhash = encrypt_with_collider(content)
            message = Message(sender_id=user.id, receiver_id=receiver_id, encrypted_content=encrypted, black_hole_hash=bhash)
            db = next(get_db())
            db.add(message)
            db.commit()
            # AI spawn if label match or /ai command
            if " /ai" in content or any("ai" in label for label in user.labels.split(',')):
                ai_resp = f"Grok Clone: Resonating with your query through foam... {content.upper()}"  # Scrubbed autonomous AI
                await manager.send_personal(json.dumps({"content": ai_resp, "is_ai": True}), websocket)
            # Send to receiver or broadcast to matches
            if receiver_id:
                await manager.send_personal(json.dumps({"content": content, "sender": user.id}), receiver_id)
            else:
                await manager.broadcast_to_matches(json.dumps({"content": content, "sender": user.id}), user.labels)
    except WebSocketDisconnect:
        manager.disconnect(user.id)

# Inbox API (searchable)
@app.get("/api/inbox")
async def get_inbox(
    search: str = "",
    folder: str = "Inbox",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Email).filter(Email.receiver_id == current_user.id, Email.folder == folder, Email.is_deleted == False)
    if search:
        query = query.filter((Email.subject.like(f"%{search}%")) | (Email.encrypted_body.like(f"%{search}%")))  # Full-text
    emails = query.order_by(Email.timestamp.desc()).all()
    return [{"id": e.id, "subject": e.subject, "body": e.body[:100], "folder": e.folder, "label": e.label, "is_starred": e.is_starred, "timestamp": e.timestamp, "hash_explain": "Secured by black hole hash from white hole retrieval"} for e in emails]

@app.post("/api/inbox/send")
async def send_email(
    receiver_email: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    receiver = db.query(User).filter(User.email == receiver_email).first()
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")
    encrypted, bhash = encrypt_with_collider(body)
    email = Email(sender_id=current_user.id, receiver_id=receiver.id, subject=subject, encrypted_body=encrypted, black_hole_hash=bhash)
    db.add(email)
    db.commit()
    return {"message": "Email sent to Holo storage, encrypted via collider."}

# Delete
@app.delete("/api/inbox/{email_id}")
async def delete_email(email_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    email = db.query(Email).filter(Email.id == email_id, Email.receiver_id == current_user.id).first()
    if not email:
        raise HTTPException(status_code=404)
    email.is_deleted = True
    db.commit()
    return {"message": "Deleted from Holo (soft delete for audit)"}

# Contacts
@app.post("/api/contacts")
async def add_contact(contact_email: str = Form(...), name: str = Form(""), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    contact = Contact(user_id=current_user.id, contact_email=contact_email, name=name)
    db.add(contact)
    db.commit()
    return {"message": "Contact added"}

@app.post("/api/contacts/import")
async def import_contacts(data: str = Form(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Simulate CSV import
    lines = data.split('\n')
    for line in lines:
        if ',' in line:
            email, name = line.split(',', 1)
            contact = Contact(user_id=current_user.id, contact_email=email.strip(), name=name.strip())
            db.add(contact)
    db.commit()
    return {"message": f"Imported {len(lines)} contacts"}

# Chat page (React served)
@app.get("/chat")
async def chat_route(current_user: User = Depends(get_current_user)):
    return RedirectResponse(url="/static/index.html")  # React app

# Inbox page
@app.get("/inbox")
async def inbox_route(current_user: User = Depends(get_current_user)):
    return RedirectResponse(url="/static/index.html")  # React

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
