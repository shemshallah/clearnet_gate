import os
import logging
import hashlib
import base64
import json
import uuid
import traceback  # Added for error handling
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx
import asyncio
from contextlib import asynccontextmanager
import secrets
from collections import defaultdict
import random  # Retained only for non-metric quantum variance
import psutil  # For real-time system network and storage metrics
import subprocess  # For real routing table extraction and QSH commands
from jinja2 import Template
import socket  # For AF_INET constant
import sys
import types  # For sandboxing
from io import StringIO

# ==================== CONFIGURATION MODULE ====================
class Config:
    """Centralized configuration management"""
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
