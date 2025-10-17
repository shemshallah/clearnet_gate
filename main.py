from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List
import random
import json
from datetime import timedelta
from .database import get_db, engine, SessionLocal
from .models import Base, User, Message, Email
from .auth import create_access_token, get_current_user
import os

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Quantum Chatroom")

# Mount static files if any
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Front page / login
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Registration
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), labels: str = Form(""), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    user = User(username=username, labels=labels)
    user.set_password(password)
    user.email = f"{username}@hackah.edu"
    db.add(user)
    db.commit()
    db.refresh(user)
    # Generate token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "email": user.email}

# Login
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), remember_me: bool = Form(False), request: Request = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.verify_password(password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES * (7*24*60 if remember_me else 1))  # 1 week if remember
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    # Store remember me IP in cookie
    if remember_me:
        request.cookies["remember_ip"] = request.client.host  # Simulate eth0/wlan0 as IP
    return {"access_token": access_token, "token_type": "bearer"}

# Forget password - simple reset (in prod, email link)
@app.post("/forget-password")
async def forget_password(username: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Simulate reset - in prod, send email
    new_password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12))
    user.set_password(new_password)
    db.commit()
    return {"new_password": new_password}  # In prod, email this

# Chat WebSocket
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            parsed = json.loads(data)
            content = parsed.get("content")
            receiver_id = parsed.get("receiver_id")
            is_ai = parsed.get("is_ai", False)
            # Save message
            message = Message(sender_id=user_id, receiver_id=receiver_id, content=content, is_ai=is_ai)
            db.add(message)
            db.commit()
            # If AI, generate response
            if is_ai:
                ai_response = f"AI: Echoing quantum entanglement: {content[::-1]}"  # Simple AI
                await manager.send_personal_message(json.dumps({"content": ai_response, "is_ai": True}), websocket)
            # Broadcast or personal
            if receiver_id:
                # Personal IM
                await manager.send_personal_message(json.dumps({"content": content, "sender": user_id}), websocket)
            else:
                await manager.broadcast(json.dumps({"content": content, "sender": user_id}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Inbox
@app.get("/inbox", response_class=HTMLResponse)
async def inbox_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("inbox.html", {"request": request, "user": current_user})

@app.get("/api/emails")
async def get_emails(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    emails = db.query(Email).filter(Email.receiver_id == current_user.id).all()
    return [{"id": e.id, "sender": e.sender, "subject": e.subject, "body": e.body, "folder": e.folder, "label": e.label, "is_starred": e.is_starred, "timestamp": e.timestamp} for e in emails]

@app.post("/api/emails")
async def send_email(sender_id: int = Form(...), receiver_email: str = Form(...), subject: str = Form(...), body: str = Form(...), db: Session = Depends(get_db)):
    receiver = db.query(User).filter(User.email == receiver_email).first()
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")
    email = Email(sender=current_user.email, receiver_id=receiver.id, subject=subject, body=body)
    db.add(email)
    db.commit()
    return {"message": "Email sent"}

# Folders, labels, stars - simple CRUD
@app.post("/api/emails/{email_id}/folder")
async def update_folder(email_id: int, folder: str = Form(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    email = db.query(Email).filter(Email.id == email_id, Email.receiver_id == current_user.id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    email.folder = folder
    db.commit()
    return {"message": "Folder updated"}

# Similar for label, star

# Chat room page
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, current_user: User = Depends(get_current_user)):
    # Match users by labels for "entanglement"
    matches = db.query(User).filter(User.labels.like(f"%{current_user.labels}%")).all()
    return templates.TemplateResponse("chat.html", {"request": request, "user": current_user, "matches": matches})

# Import contacts - simple
@app.post("/api/contacts/import")
async def import_contacts(data: str = Form(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Simulate import from another email
    # Parse data, add to user labels or separate contacts table
    return {"message": "Contacts imported"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
