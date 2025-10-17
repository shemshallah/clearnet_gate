import hashlib
import random
from cryptography.fernet import Fernet
import base64

def create_black_hole_hash(content: str) -> str:
    # "Retrieve from white holes" - simulate with entropy
    white_hole_seed = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789!@#$', k=64))
    combined = content + white_hole_seed
    return hashlib.sha3_512(combined.encode()).hexdigest()

def encrypt_with_collider(content: str) -> tuple[str, str]:
    hash_val = create_black_hole_hash(content)
    key = base64.urlsafe_b64encode(hashlib.sha256(hash_val.encode()).digest())
    f = Fernet(key)
    encrypted = f.encrypt(content.encode()).decode()
    return encrypted, hash_val