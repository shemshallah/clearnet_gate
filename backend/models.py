from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import hashlib
from database import Base

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    domain = Column(String(50), default="quantum")  # e.g., quantum.foam.computer
    hashed_password = Column(String(255))
    labels = Column(String(500), default="")  # Comma-separated for entanglement matching
    email = Column(String(100), unique=True)  # username@domain.foam.computer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    contacts = relationship("Contact", back_populates="user")
    messages_sent = relationship("Message", foreign_keys=[Message.sender_id], back_populates="sender")
    messages_received = relationship("Message", foreign_keys=[Message.receiver_id], back_populates="receiver")
    emails_sent = relationship("Email", foreign_keys=[Email.sender_id], back_populates="sender")
    emails_received = relationship("Email", foreign_keys=[Email.receiver_id], back_populates="receiver")

    def set_password(self, password: str):
        self.hashed_password = pwd_context.hash(password)

    def verify_password(self, password: str):
        return pwd_context.verify(password, self.hashed_password)

    @property
    def full_email(self):
        return f"{self.username}@{self.domain}.foam.computer"

class Contact(Base):
    __tablename__ = "contacts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    contact_email = Column(String(100))
    name = Column(String(100))
    is_starred = Column(Boolean, default=False)
    labels = Column(String(200), default="")

    user = relationship("User", back_populates="contacts")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    encrypted_content = Column(Text)  # Encrypted
    black_hole_hash = Column(String(128))  # Collider hash
    is_ai = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_deleted = Column(Boolean, default=False)

    sender = relationship("User", foreign_keys=[sender_id], back_populates="messages_sent")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="messages_received")

    @property
    def content(self):
        if self.encrypted_content and not self.is_deleted:
            key = derive_key_from_collider(self.black_hole_hash)  # Custom decrypt
            f = Fernet(key)
            return f.decrypt(self.encrypted_content.encode()).decode()
        return None

class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    subject = Column(String(200))
    encrypted_body = Column(Text)
    black_hole_hash = Column(String(128))
    folder = Column(String(50), default="Inbox")
    label = Column(String(200), default="")
    is_starred = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    Index('idx_receiver_folder', receiver_id, folder)

    sender = relationship("User", foreign_keys=[sender_id], back_populates="emails_sent")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="emails_received")

    @property
    def body(self):
        if self.encrypted_body and not self.is_deleted:
            key = derive_key_from_collider(self.black_hole_hash)
            f = Fernet(key)
            return f.decrypt(self.encrypted_body.encode()).decode()
        return None

def derive_key_from_collider(black_hole_hash: str) -> bytes:
    # Custom reproducible encryption: SHA3 salt + Fernet key
    salt = hashlib.sha3_512(black_hole_hash.encode()).digest()[:32]
    key = hashlib.pbkdf2_hmac('sha512', salt, black_hole_hash.encode(), 100000)[:32]
    return Fernet.generate_key() if not key else base64.urlsafe_b64encode(key)  # Reproducible

# Collider function for black hole hashes (from white holes - simulated entropy)
def collider_black_hole_hash(content: str) -> str:
    # Simulate white hole entropy with random + hash
    white_hole_entropy = ''.join(random.choices('01', k=256))  # Pseudo-random "white hole"
    return hashlib.sha3_512((content + white_hole_entropy).encode()).hexdigest()