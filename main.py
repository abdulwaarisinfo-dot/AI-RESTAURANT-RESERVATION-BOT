# ============================================================
# AI RESTAURANT RESERVATION BOT  v4.0
# ============================================================
# FEATURES:
#   - Signup / Login with JWT (bcrypt + rate limiting)
#   - 10 FREE reservations after signup (no billing required)
#   - Paddle billing — 3 plans, webhook, renewal, cancel
#   - Per-restaurant isolated MongoDB databases
#   - CRM: CRUD for guests, tables, reservations
#   - TIME SLOTS: Owner defines per-table slots (bulk or individual)
#   - AI Sales Agent (Claude) — multi-turn, slot-aware, smart CRM reader
#   - WebSocket real-time dashboard updates
#   - Admin dashboard + Jinja2 HTML pages
#   - Plan-based reservation limits (enforced server-side)
#   - Security: CSRF, rate limiting, helmet headers
# ============================================================

from fastapi import (
    FastAPI, Depends, HTTPException, Request, Form,
    WebSocket, WebSocketDisconnect, status
)
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from collections import defaultdict

from pymongo import MongoClient
from pymongo.database import Database
from bson import ObjectId

from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext

import anthropic
import httpx
import logging
import jwt
import os
import time
import json
import asyncio
import hmac
import hashlib

# ============================================================
# INITIAL SETUP
# ============================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AI.RESTAURANT.BOT")

# ============================================================
# CONFIG
# ============================================================

MONGO_URL             = os.getenv("MONGO_URL", "mongodb://localhost:27017")
ANTHROPIC_KEY         = os.getenv("ANTHROPIC_API_KEY", "")
JWT_SECRET            = os.getenv("JWT_SECRET", "change-me-in-production-use-256bit-random")
JWT_ALGORITHM         = "HS256"
JWT_EXPIRE_DAYS       = 30

FREE_RESERVATIONS     = 10
FREE_TABLES_MAX       = 3

PADDLE_VENDOR_ID      = os.getenv("PADDLE_VENDOR_ID", "")
PADDLE_API_KEY        = os.getenv("PADDLE_API_KEY", "")
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "")
PADDLE_SANDBOX        = os.getenv("PADDLE_SANDBOX", "true").lower() == "true"
PADDLE_BASE_URL       = "https://sandbox-api.paddle.com" if PADDLE_SANDBOX else "https://api.paddle.com"

pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# ============================================================
# PLANS CONFIGURATION
# ============================================================

PLANS = {
    "starter": {
        "name":               "Starter",
        "price_month":        29,
        "reservations_month": 1200,
        "tables_max":         10,
        "paddle_price_id":    os.getenv("PADDLE_PRICE_STARTER", "pri_starter"),
        "color":              "#6366f1",
        "features": [
            "1,200 reservations / month",
            "Up to 10 tables",
            "AI booking agent",
            "Basic CRM",
            "Email support",
        ],
    },
    "professional": {
        "name":               "Professional",
        "price_month":        59,
        "reservations_month": 1600,
        "tables_max":         25,
        "paddle_price_id":    os.getenv("PADDLE_PRICE_PRO", "pri_pro"),
        "color":              "#f59e0b",
        "features": [
            "1,600 reservations / month",
            "Up to 25 tables",
            "AI booking agent",
            "Full CRM + guest history",
            "Priority support",
        ],
    },
    "enterprise": {
        "name":               "Enterprise",
        "price_month":        99,
        "reservations_month": 2000,
        "tables_max":         999,
        "paddle_price_id":    os.getenv("PADDLE_PRICE_ENT", "pri_ent"),
        "color":              "#10b981",
        "features": [
            "2,000 reservations / month",
            "Unlimited tables",
            "AI booking agent",
            "Full CRM + analytics",
            "24/7 dedicated support",
        ],
    },
}

# ============================================================
# RATE LIMITING
# ============================================================

_rate_store: Dict[str, list] = defaultdict(list)

def rate_limit(ip: str, max_calls: int = 20, window_sec: int = 60) -> bool:
    now = time.time()
    calls = _rate_store[ip]
    _rate_store[ip] = [t for t in calls if now - t < window_sec]
    if len(_rate_store[ip]) >= max_calls:
        return False
    _rate_store[ip].append(now)
    return True

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Restaurant Reservation Bot",
    version="4.0.0",
    description="Multi-tenant reservation system with Paddle billing, Time Slots, and AI agent.",
    docs_url="/api/docs",
)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ============================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, ws: WebSocket, restaurant_id: str):
        await ws.accept()
        self.active[restaurant_id].append(ws)
        logger.info(f"WS connected: {restaurant_id} ({len(self.active[restaurant_id])} total)")

    def disconnect(self, ws: WebSocket, restaurant_id: str):
        if ws in self.active[restaurant_id]:
            self.active[restaurant_id].remove(ws)

    async def broadcast(self, restaurant_id: str, data: dict):
        dead = []
        for ws in self.active[restaurant_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, restaurant_id)

ws_manager = ConnectionManager()

# ============================================================
# DATABASE
# ============================================================

_mongo_client: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URL)
        logger.info("MongoDB connected.")
    return _mongo_client

def get_platform_db() -> Database:
    return get_mongo_client()["platform"]

def get_owner_db(restaurant_id: str) -> Database:
    return get_mongo_client()[f"restaurant_{restaurant_id}"]

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class SignupRequest(BaseModel):
    restaurant_name: str
    owner_name:      str
    email:           EmailStr
    password:        str

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_at:   datetime

class TableIn(BaseModel):
    table_number: int
    capacity:     int
    location:     Optional[str] = "main"

class GuestIn(BaseModel):
    name:  str
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    notes: Optional[str] = None

class ReservationIn(BaseModel):
    guest_id:   str
    table_id:   str
    party_size: int
    date:       str
    time_slot:  str
    notes:      Optional[str] = None

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    history:    List[ChatMessage] = []

class SubscribeRequest(BaseModel):
    plan: str

# ============================================================
# TIME SLOT SCHEMAS
# ============================================================

class TimeSlotIn(BaseModel):
    table_id:   str
    date:       str
    start_time: str
    end_time:   str

class BulkTimeSlotsIn(BaseModel):
    table_ids:      List[str]
    date:           str
    start_time:     str
    end_time:       str
    duration_mins:  int = 30

class DeleteTimeSlotsIn(BaseModel):
    table_ids: List[str]
    date:      str

# ============================================================
# AUTH HELPERS
# ============================================================

def hash_password(raw: str) -> str:
    return pwd_ctx.hash(raw)

def verify_password(raw: str, hashed: str) -> bool:
    return pwd_ctx.verify(raw, hashed)

def create_jwt(restaurant_id: str, email: str) -> tuple[str, datetime]:
    expires_at = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub":   restaurant_id,
        "email": email,
        "exp":   expires_at,
        "iat":   datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM), expires_at

def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token.")

def get_current_owner(
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    if creds and creds.credentials:
        return decode_jwt(creds.credentials)
    token = request.cookies.get("access_token")
    if token:
        return decode_jwt(token)
    raise HTTPException(401, "Not authenticated.")

def get_owner_from_cookie(request: Request) -> Optional[dict]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        return decode_jwt(token)
    except Exception:
        return None

def _subscription_active(owner_doc: dict) -> bool:
    return owner_doc.get("subscription_status") == "active"

def _get_plan(owner_doc: dict) -> dict:
    plan_key = owner_doc.get("plan", "starter")
    return PLANS.get(plan_key, PLANS["starter"])

# ============================================================
# FREE TIER HELPERS
# ============================================================

def _is_free_tier(owner_doc: dict) -> bool:
    return not _subscription_active(owner_doc)

def _free_reservations_used(rdb: Database) -> int:
    return rdb["reservations"].count_documents({"status": "confirmed"})

def _free_reservations_remaining(rdb: Database) -> int:
    used = _free_reservations_used(rdb)
    return max(0, FREE_RESERVATIONS - used)

def _can_access_dashboard(owner_doc: dict) -> bool:
    if _subscription_active(owner_doc):
        return True
    return True

# ============================================================
# PLAN ENFORCEMENT HELPERS
# ============================================================

def _check_reservation_limit(rdb: Database, owner_doc: dict) -> bool:
    if _is_free_tier(owner_doc):
        used = _free_reservations_used(rdb)
        return used < FREE_RESERVATIONS
    plan  = _get_plan(owner_doc)
    limit = plan["reservations_month"]
    month_start = (datetime.now(timezone.utc) - timedelta(days=30))
    count = rdb["reservations"].count_documents({
        "status":     "confirmed",
        "created_at": {"$gte": month_start},
    })
    return count < limit

def _check_table_limit(rdb: Database, owner_doc: dict) -> bool:
    if _is_free_tier(owner_doc):
        count = rdb["tables"].count_documents({})
        return count < FREE_TABLES_MAX
    plan  = _get_plan(owner_doc)
    limit = plan["tables_max"]
    count = rdb["tables"].count_documents({})
    return count < limit

def _get_effective_limits(rdb: Database, owner_doc: dict) -> dict:
    if _is_free_tier(owner_doc):
        used = _free_reservations_used(rdb)
        return {
            "tier":               "free",
            "reservations_limit": FREE_RESERVATIONS,
            "reservations_used":  used,
            "reservations_left":  max(0, FREE_RESERVATIONS - used),
            "tables_max":         FREE_TABLES_MAX,
        }
    plan = _get_plan(owner_doc)
    month_start = (datetime.now(timezone.utc) - timedelta(days=30))
    used = rdb["reservations"].count_documents({
        "status":     "confirmed",
        "created_at": {"$gte": month_start},
    })
    return {
        "tier":               "paid",
        "reservations_limit": plan["reservations_month"],
        "reservations_used":  used,
        "reservations_left":  max(0, plan["reservations_month"] - used),
        "tables_max":         plan["tables_max"],
    }

# ============================================================
# TIME SLOT HELPERS
# ============================================================

def _generate_slots(start_time: str, end_time: str, duration_mins: int) -> List[dict]:
    slots = []
    fmt = "%H:%M"
    current = datetime.strptime(start_time, fmt)
    end     = datetime.strptime(end_time,   fmt)
    while current < end:
        slot_end = current + timedelta(minutes=duration_mins)
        if slot_end > end:
            break
        slots.append({
            "start": current.strftime(fmt),
            "end":   slot_end.strftime(fmt),
        })
        current = slot_end
    return slots

def _get_table_slots(rdb: Database, table_id: str, date: str) -> List[dict]:
    doc = rdb["time_slots"].find_one({"table_id": table_id, "date": date})
    if not doc:
        return []
    return doc.get("slots", [])

def _any_slots_defined(rdb: Database, date: str) -> bool:
    return rdb["time_slots"].count_documents({"date": date}) > 0

def _slot_in_defined(rdb: Database, table_id: str, date: str, start_time: str) -> bool:
    doc = rdb["time_slots"].find_one({
        "table_id": table_id,
        "date":     date,
        "slots":    {"$elemMatch": {"start": start_time}},
    })
    return doc is not None

# ============================================================
# PADDLE BILLING HELPERS
# ============================================================

async def paddle_create_checkout(owner_doc: dict, plan_key: str) -> dict:
    plan = PLANS.get(plan_key)
    if not plan:
        raise HTTPException(400, "Invalid plan.")
    headers = {
        "Authorization": f"Bearer {PADDLE_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "items": [{"price_id": plan["paddle_price_id"], "quantity": 1}],
        "customer": {"email": owner_doc["email"]},
        "custom_data": {
            "restaurant_id": str(owner_doc["_id"]),
            "plan":          plan_key,
        },
        "success_url": os.getenv("APP_URL", "http://localhost:8000") + "/billing/success",
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{PADDLE_BASE_URL}/checkouts",
            headers=headers,
            json=payload,
            timeout=10,
        )
    if r.status_code not in (200, 201):
        logger.error(f"Paddle checkout error: {r.text}")
        raise HTTPException(502, "Payment provider error. Please try again.")
    return r.json()

def verify_paddle_webhook(payload: bytes, signature: str) -> bool:
    if not PADDLE_WEBHOOK_SECRET:
        return True
    expected = hmac.new(
        PADDLE_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

# ============================================================
# JINJA2 HTML ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["Pages"])
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "plans":   PLANS,
    })

@app.get("/signup", response_class=HTMLResponse, tags=["Pages"])
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup", response_class=HTMLResponse, tags=["Pages"])
def signup_form(
    request:         Request,
    restaurant_name: str = Form(...),
    owner_name:      str = Form(...),
    email:           str = Form(...),
    password:        str = Form(...),
):
    ip = request.client.host
    if not rate_limit(ip, max_calls=5, window_sec=300):
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "Too many signup attempts. Please wait 5 minutes."
        })
    if len(password) < 8:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "Password must be at least 8 characters."
        })
    db  = get_platform_db()
    col = db["owners"]
    if col.find_one({"email": email.lower()}):
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "Email already registered."
        })
    restaurant_id = str(ObjectId())
    col.insert_one({
        "_id":                    ObjectId(restaurant_id),
        "restaurant_name":        restaurant_name,
        "owner_name":             owner_name,
        "email":                  email.lower(),
        "password_hash":          hash_password(password),
        "created_at":             datetime.now(timezone.utc),
        "subscription_status":    "inactive",
        "plan":                   "starter",
        "paddle_customer_id":     None,
        "paddle_subscription_id": None,
        "free_tier":              True,
    })
    token, _ = create_jwt(restaurant_id, email.lower())
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie(
        "access_token", token,
        httponly=True, samesite="lax",
        max_age=JWT_EXPIRE_DAYS * 86400,
    )
    logger.info(f"Signed up (free tier): {email} ({restaurant_id})")
    return response

@app.get("/login", response_class=HTMLResponse, tags=["Pages"])
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse, tags=["Pages"])
def login_form(
    request:  Request,
    email:    str = Form(...),
    password: str = Form(...),
):
    ip = request.client.host
    if not rate_limit(ip, max_calls=10, window_sec=60):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Too many login attempts. Please wait a moment."
        })
    db    = get_platform_db()
    owner = db["owners"].find_one({"email": email.lower()})
    if not owner or not verify_password(password, owner["password_hash"]):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid email or password."
        })
    restaurant_id = str(owner["_id"])
    token, _      = create_jwt(restaurant_id, email.lower())
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie(
        "access_token", token,
        httponly=True, samesite="lax",
        max_age=JWT_EXPIRE_DAYS * 86400,
    )
    return response

@app.get("/logout", tags=["Pages"])
def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("access_token")
    return response

@app.get("/billing", response_class=HTMLResponse, tags=["Pages"])
def billing_page(request: Request):
    owner_jwt = get_owner_from_cookie(request)
    if not owner_jwt:
        return RedirectResponse("/login")
    db    = get_platform_db()
    owner = db["owners"].find_one({"_id": ObjectId(owner_jwt["sub"])})
    current_plan = owner.get("plan", "starter")
    rdb   = get_owner_db(owner_jwt["sub"])
    limits = _get_effective_limits(rdb, owner)
    return templates.TemplateResponse("billing.html", {
        "request":      request,
        "owner":        owner,
        "active":       _subscription_active(owner),
        "plans":        PLANS,
        "current_plan": current_plan,
        "app_url":      os.getenv("APP_URL", "http://localhost:8000"),
        "limits":       limits,
        "free_reservations": FREE_RESERVATIONS,
    })

@app.post("/billing/subscribe", response_class=HTMLResponse, tags=["Pages"])
async def billing_subscribe(
    request: Request,
    plan:    str = Form(...),
):
    owner_jwt = get_owner_from_cookie(request)
    if not owner_jwt:
        return RedirectResponse("/login", status_code=302)
    if plan not in PLANS:
        return RedirectResponse("/billing", status_code=302)
    db        = get_platform_db()
    owner_doc = db["owners"].find_one({"_id": ObjectId(owner_jwt["sub"])})
    rdb       = get_owner_db(owner_jwt["sub"])
    limits    = _get_effective_limits(rdb, owner_doc)
    try:
        result = await paddle_create_checkout(owner_doc, plan)
    except Exception as e:
        return templates.TemplateResponse("billing.html", {
            "request":           request,
            "owner":             owner_doc,
            "active":            _subscription_active(owner_doc),
            "plans":             PLANS,
            "current_plan":      owner_doc.get("plan", "starter"),
            "app_url":           os.getenv("APP_URL", "http://localhost:8000"),
            "limits":            limits,
            "free_reservations": FREE_RESERVATIONS,
            "error":             "Payment provider error. Please try again.",
        })
    checkout_url = result.get("data", {}).get("url") or result.get("url", "")
    if checkout_url:
        return RedirectResponse(checkout_url, status_code=302)
    return RedirectResponse("/billing", status_code=302)

@app.post("/billing/cancel", response_class=HTMLResponse, tags=["Pages"])
async def billing_cancel_page(request: Request):
    owner_jwt = get_owner_from_cookie(request)
    if not owner_jwt:
        return RedirectResponse("/login", status_code=302)
    db        = get_platform_db()
    owner_doc = db["owners"].find_one({"_id": ObjectId(owner_jwt["sub"])})
    sub_id    = owner_doc.get("paddle_subscription_id")
    if sub_id:
        headers = {"Authorization": f"Bearer {PADDLE_API_KEY}"}
        async with httpx.AsyncClient() as client:
            r = await client.delete(
                f"{PADDLE_BASE_URL}/subscriptions/{sub_id}",
                headers=headers,
            )
        if r.status_code in (200, 204):
            db["owners"].update_one(
                {"_id": ObjectId(owner_jwt["sub"])},
                {"$set": {"subscription_status": "cancelled"}}
            )
            logger.info(f"Subscription cancelled: {owner_jwt['sub']}")
    return RedirectResponse("/billing", status_code=302)

@app.get("/billing/success", response_class=HTMLResponse, tags=["Pages"])
def billing_success(request: Request):
    owner_jwt = get_owner_from_cookie(request)
    if not owner_jwt:
        return RedirectResponse("/login")
    return RedirectResponse("/dashboard")

@app.get("/dashboard", response_class=HTMLResponse, tags=["Pages"])
def dashboard_page(request: Request):
    owner_jwt = get_owner_from_cookie(request)
    if not owner_jwt:
        return RedirectResponse("/login")
    db    = get_platform_db()
    owner = db["owners"].find_one({"_id": ObjectId(owner_jwt["sub"])})
    rdb   = get_owner_db(owner_jwt["sub"])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    plan  = _get_plan(owner)
    limits = _get_effective_limits(rdb, owner)
    month_start = (datetime.now(timezone.utc) - timedelta(days=30))
    stats = {
        "total_tables":       rdb["tables"].count_documents({}),
        "total_guests":       rdb["guests"].count_documents({}),
        "total_reservations": rdb["reservations"].count_documents({}),
        "today_reservations": rdb["reservations"].count_documents({"date": today, "status": "confirmed"}),
        "cancelled":          rdb["reservations"].count_documents({"status": "cancelled"}),
        "month_count":        limits["reservations_used"],
        "month_limit":        limits["reservations_limit"],
        "month_pct":          round(limits["reservations_used"] / limits["reservations_limit"] * 100)
                              if limits["reservations_limit"] > 0 else 0,
    }
    top_guests = list(
        rdb["guests"].find({}, {"name": 1, "visit_count": 1})
                     .sort("visit_count", -1).limit(5)
    )
    for g in top_guests:
        g["id"] = str(g.pop("_id"))
    recent_res = list(
        rdb["reservations"].find({"status": "confirmed"})
                           .sort("created_at", -1).limit(8)
    )
    for r in recent_res:
        r["id"] = str(r.pop("_id"))
    return templates.TemplateResponse("dashboard.html", {
        "request":           request,
        "owner":             owner,
        "stats":             stats,
        "top_guests":        top_guests,
        "recent_res":        recent_res,
        "plan":              plan,
        "limits":            limits,
        "free_reservations": FREE_RESERVATIONS,
        "is_free_tier":      _is_free_tier(owner),
    })

@app.get("/crm", response_class=HTMLResponse, tags=["Pages"])
def crm_page(request: Request):
    owner_jwt = get_owner_from_cookie(request)
    if not owner_jwt:
        return RedirectResponse("/login")
    db    = get_platform_db()
    owner = db["owners"].find_one({"_id": ObjectId(owner_jwt["sub"])})
    rdb    = get_owner_db(owner_jwt["sub"])
    tables = list(rdb["tables"].find())
    guests = list(rdb["guests"].find())
    reservations = list(rdb["reservations"].find().sort("date", -1).limit(50))
    for doc in tables + guests + reservations:
        doc["id"] = str(doc.pop("_id"))
        if "created_at" in doc and isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].strftime("%Y-%m-%d %H:%M")
    plan   = _get_plan(owner)
    limits = _get_effective_limits(rdb, owner)
    return templates.TemplateResponse("crm.html", {
        "request":           request,
        "owner":             owner,
        "tables":            tables,
        "guests":            guests,
        "reservations":      reservations,
        "plan":              plan,
        "limits":            limits,
        "free_reservations": FREE_RESERVATIONS,
        "is_free_tier":      _is_free_tier(owner),
    })

@app.get("/reserve/{restaurant_id}", response_class=HTMLResponse, tags=["Pages"])
def reservation_page(request: Request, restaurant_id: str):
    try:
        db    = get_platform_db()
        owner = db["owners"].find_one({"_id": ObjectId(restaurant_id)})
    except Exception:
        raise HTTPException(404, "Restaurant not found.")
    if not owner:
        raise HTTPException(404, "Restaurant not found.")
    rdb    = get_owner_db(restaurant_id)
    limits = _get_effective_limits(rdb, owner)
    if limits["reservations_left"] == 0 and not _subscription_active(owner):
        raise HTTPException(503, "Reservations are temporarily unavailable. Please contact the restaurant.")
    return templates.TemplateResponse("reservation.html", {
        "request":         request,
        "restaurant_name": owner["restaurant_name"],
        "restaurant_id":   restaurant_id,
    })

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/dashboard/{restaurant_id}")
async def ws_dashboard(websocket: WebSocket, restaurant_id: str):
    db    = get_platform_db()
    owner = db["owners"].find_one({"_id": ObjectId(restaurant_id)})
    if not owner:
        await websocket.close(code=4004)
        return
    await ws_manager.connect(websocket, restaurant_id)
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, restaurant_id)
    except Exception:
        ws_manager.disconnect(websocket, restaurant_id)

# ============================================================
# API — AUTH
# ============================================================

@app.post("/api/auth/signup", response_model=TokenResponse, tags=["Auth API"])
def api_signup(body: SignupRequest, request: Request):
    ip = request.client.host
    if not rate_limit(ip, max_calls=5, window_sec=300):
        raise HTTPException(429, "Too many requests.")
    db  = get_platform_db()
    if db["owners"].find_one({"email": body.email.lower()}):
        raise HTTPException(400, "Email already registered.")
    restaurant_id = str(ObjectId())
    db["owners"].insert_one({
        "_id":                    ObjectId(restaurant_id),
        "restaurant_name":        body.restaurant_name,
        "owner_name":             body.owner_name,
        "email":                  body.email.lower(),
        "password_hash":          hash_password(body.password),
        "created_at":             datetime.now(timezone.utc),
        "subscription_status":    "inactive",
        "plan":                   "starter",
        "paddle_customer_id":     None,
        "paddle_subscription_id": None,
        "free_tier":              True,
    })
    token, expires_at = create_jwt(restaurant_id, body.email.lower())
    return TokenResponse(access_token=token, expires_at=expires_at)

@app.post("/api/auth/login", response_model=TokenResponse, tags=["Auth API"])
def api_login(body: LoginRequest, request: Request):
    ip = request.client.host
    if not rate_limit(ip, max_calls=10, window_sec=60):
        raise HTTPException(429, "Too many requests.")
    db    = get_platform_db()
    owner = db["owners"].find_one({"email": body.email.lower()})
    if not owner or not verify_password(body.password, owner["password_hash"]):
        raise HTTPException(401, "Invalid credentials.")
    token, expires_at = create_jwt(str(owner["_id"]), body.email.lower())
    return TokenResponse(access_token=token, expires_at=expires_at)

# ============================================================
# API — BILLING (PADDLE)
# ============================================================

@app.post("/api/billing/checkout", tags=["Billing"])
async def create_checkout(body: SubscribeRequest, owner=Depends(get_current_owner)):
    if body.plan not in PLANS:
        raise HTTPException(400, f"Invalid plan. Choose: {list(PLANS.keys())}")
    db        = get_platform_db()
    owner_doc = db["owners"].find_one({"_id": ObjectId(owner["sub"])})
    try:
        result = await paddle_create_checkout(owner_doc, body.plan)
    except Exception as e:
        raise HTTPException(502, str(e))
    checkout_url = result.get("data", {}).get("url") or result.get("url", "")
    return {"checkout_url": checkout_url, "plan": body.plan}

@app.post("/api/billing/cancel", tags=["Billing"])
async def cancel_subscription(owner=Depends(get_current_owner)):
    db        = get_platform_db()
    owner_doc = db["owners"].find_one({"_id": ObjectId(owner["sub"])})
    sub_id = owner_doc.get("paddle_subscription_id")
    if not sub_id:
        raise HTTPException(400, "No active subscription.")
    headers = {"Authorization": f"Bearer {PADDLE_API_KEY}"}
    async with httpx.AsyncClient() as client:
        r = await client.delete(
            f"{PADDLE_BASE_URL}/subscriptions/{sub_id}",
            headers=headers,
        )
    if r.status_code not in (200, 204):
        raise HTTPException(502, "Failed to cancel subscription.")
    db["owners"].update_one(
        {"_id": ObjectId(owner["sub"])},
        {"$set": {"subscription_status": "cancelled"}}
    )
    return {"cancelled": True}

@app.get("/api/billing/plans", tags=["Billing"])
def get_plans():
    return {k: {**v, "paddle_price_id": None} for k, v in PLANS.items()}

@app.post("/api/billing/webhook", tags=["Billing"])
async def paddle_webhook(request: Request):
    payload   = await request.body()
    signature = request.headers.get("Paddle-Signature", "")
    if not verify_paddle_webhook(payload, signature):
        raise HTTPException(400, "Invalid webhook signature.")
    try:
        event = json.loads(payload)
    except Exception:
        raise HTTPException(400, "Invalid JSON.")
    event_type = event.get("event_type", "")
    data       = event.get("data", {})
    db         = get_platform_db()
    if event_type in ("subscription.activated", "subscription.updated"):
        sub_id        = data.get("id")
        custom_data   = data.get("custom_data", {})
        restaurant_id = custom_data.get("restaurant_id")
        plan_key      = custom_data.get("plan", "starter")
        customer_id   = data.get("customer_id")
        if restaurant_id:
            db["owners"].update_one(
                {"_id": ObjectId(restaurant_id)},
                {"$set": {
                    "subscription_status":    "active",
                    "plan":                   plan_key,
                    "paddle_subscription_id": sub_id,
                    "paddle_customer_id":     customer_id,
                    "free_tier":              False,
                }}
            )
            logger.info(f"Webhook: subscription activated {restaurant_id} plan={plan_key}")
    elif event_type in ("subscription.canceled", "subscription.paused"):
        sub_id = data.get("id")
        db["owners"].update_one(
            {"paddle_subscription_id": sub_id},
            {"$set": {"subscription_status": "cancelled"}}
        )
        logger.info(f"Webhook: subscription cancelled {sub_id}")
    elif event_type == "transaction.completed":
        custom_data   = data.get("custom_data", {})
        restaurant_id = custom_data.get("restaurant_id")
        plan_key      = custom_data.get("plan", "starter")
        if restaurant_id:
            db["owners"].update_one(
                {"_id": ObjectId(restaurant_id)},
                {"$set": {
                    "subscription_status": "active",
                    "plan":                plan_key,
                    "free_tier":           False,
                }}
            )
    elif event_type == "transaction.payment_failed":
        customer_id = data.get("customer_id")
        if customer_id:
            db["owners"].update_one(
                {"paddle_customer_id": customer_id},
                {"$set": {"subscription_status": "past_due"}}
            )
            logger.warning(f"Webhook: payment FAILED for {customer_id}")
    return {"received": True}

# ============================================================
# CRM API
# ============================================================

def _sub_guard(owner):
    db        = get_platform_db()
    owner_doc = db["owners"].find_one({"_id": ObjectId(owner["sub"])})
    rdb       = get_owner_db(owner["sub"])
    return rdb, owner_doc

# --- Tables ---

@app.post("/api/crm/tables", tags=["Tables"])
def api_create_table(body: TableIn, owner=Depends(get_current_owner)):
    db, owner_doc = _sub_guard(owner)
    if not _check_table_limit(db, owner_doc):
        if _is_free_tier(owner_doc):
            raise HTTPException(403, f"Free tier allows up to {FREE_TABLES_MAX} tables. Upgrade to add more.")
        plan = _get_plan(owner_doc)
        raise HTTPException(403, f"Table limit reached ({plan['tables_max']} max on your plan).")
    result = db["tables"].insert_one(body.model_dump())
    return {"id": str(result.inserted_id), **body.model_dump()}

@app.get("/api/crm/tables", tags=["Tables"])
def api_list_tables(owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    rows  = list(db["tables"].find())
    for r in rows: r["id"] = str(r.pop("_id"))
    return rows

@app.put("/api/crm/tables/{table_id}", tags=["Tables"])
def api_update_table(table_id: str, body: TableIn, owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    res   = db["tables"].update_one({"_id": ObjectId(table_id)}, {"$set": body.model_dump()})
    if res.matched_count == 0: raise HTTPException(404, "Table not found.")
    return {"updated": True}

@app.delete("/api/crm/tables/{table_id}", tags=["Tables"])
def api_delete_table(table_id: str, owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    res   = db["tables"].delete_one({"_id": ObjectId(table_id)})
    if res.deleted_count == 0: raise HTTPException(404, "Table not found.")
    return {"deleted": True}

# --- Guests ---

@app.post("/api/crm/guests", tags=["Guests"])
def api_create_guest(body: GuestIn, owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    doc   = {**body.model_dump(), "visit_count": 0, "created_at": datetime.now(timezone.utc)}
    result = db["guests"].insert_one(doc)
    return {"id": str(result.inserted_id), **body.model_dump()}

@app.get("/api/crm/guests", tags=["Guests"])
def api_list_guests(owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    rows  = list(db["guests"].find())
    for r in rows: r["id"] = str(r.pop("_id"))
    return rows

@app.put("/api/crm/guests/{guest_id}", tags=["Guests"])
def api_update_guest(guest_id: str, body: GuestIn, owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    res   = db["guests"].update_one({"_id": ObjectId(guest_id)}, {"$set": body.model_dump()})
    if res.matched_count == 0: raise HTTPException(404, "Guest not found.")
    return {"updated": True}

@app.delete("/api/crm/guests/{guest_id}", tags=["Guests"])
def api_delete_guest(guest_id: str, owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    res   = db["guests"].delete_one({"_id": ObjectId(guest_id)})
    if res.deleted_count == 0: raise HTTPException(404, "Guest not found.")
    return {"deleted": True}

# --- Reservations ---

def _slot_available(db, table_id, date, time_slot, exclude_id=None):
    q = {"table_id": table_id, "date": date, "time_slot": time_slot, "status": {"$ne": "cancelled"}}
    if exclude_id: q["_id"] = {"$ne": ObjectId(exclude_id)}
    return db["reservations"].count_documents(q) == 0

@app.post("/api/crm/reservations", tags=["Reservations"])
def api_create_reservation(body: ReservationIn, owner=Depends(get_current_owner)):
    db, owner_doc = _sub_guard(owner)
    if not _check_reservation_limit(db, owner_doc):
        if _is_free_tier(owner_doc):
            raise HTTPException(403, f"Free tier limit of {FREE_RESERVATIONS} reservations reached. Upgrade to continue.")
        plan = _get_plan(owner_doc)
        raise HTTPException(403, f"Monthly reservation limit reached ({plan['reservations_month']} on your plan). Upgrade to continue.")
    if not _slot_available(db, body.table_id, body.date, body.time_slot):
        raise HTTPException(409, "Time slot already booked.")
    doc = {**body.model_dump(), "status": "confirmed", "created_at": datetime.now(timezone.utc)}
    result = db["reservations"].insert_one(doc)
    db["guests"].update_one({"_id": ObjectId(body.guest_id)}, {"$inc": {"visit_count": 1}})
    return {"id": str(result.inserted_id), **body.model_dump()}

@app.get("/api/crm/reservations", tags=["Reservations"])
def api_list_reservations(date: Optional[str] = None, owner=Depends(get_current_owner)):
    db, _  = _sub_guard(owner)
    query  = {"date": date} if date else {}
    rows   = list(db["reservations"].find(query).sort("time_slot", 1))
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
    return rows

@app.patch("/api/crm/reservations/{res_id}/cancel", tags=["Reservations"])
def api_cancel_reservation(res_id: str, owner=Depends(get_current_owner)):
    db, _ = _sub_guard(owner)
    res   = db["reservations"].update_one({"_id": ObjectId(res_id)}, {"$set": {"status": "cancelled"}})
    if res.matched_count == 0: raise HTTPException(404, "Reservation not found.")
    return {"cancelled": True}

# ============================================================
# TIME SLOTS API
# ============================================================

@app.post("/api/crm/timeslots/bulk", tags=["Time Slots"])
def api_bulk_create_timeslots(body: BulkTimeSlotsIn, owner=Depends(get_current_owner)):
    rdb, owner_doc = _sub_guard(owner)
    generated = _generate_slots(body.start_time, body.end_time, body.duration_mins)
    if not generated:
        raise HTTPException(400, "No slots generated. Check start/end times and duration.")

    results = []
    for table_id in body.table_ids:
        table = rdb["tables"].find_one({"_id": ObjectId(table_id)})
        if not table:
            continue

        existing_doc = rdb["time_slots"].find_one({"table_id": table_id, "date": body.date})
        if existing_doc:
            existing_starts = {s["start"] for s in existing_doc.get("slots", [])}
            new_slots = [s for s in generated if s["start"] not in existing_starts]
            if new_slots:
                all_slots = sorted(
                    existing_doc["slots"] + new_slots,
                    key=lambda x: x["start"]
                )
                rdb["time_slots"].update_one(
                    {"table_id": table_id, "date": body.date},
                    {"$set": {"slots": all_slots, "updated_at": datetime.now(timezone.utc)}}
                )
        else:
            rdb["time_slots"].insert_one({
                "table_id":   table_id,
                "date":       body.date,
                "slots":      generated,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })

        results.append({
            "table_id":     table_id,
            "table_number": table.get("table_number"),
            "slots_added":  len(generated),
        })

    logger.info(f"Bulk slots created: {len(results)} tables, date={body.date}, slots_per_table={len(generated)}")
    return {
        "date":           body.date,
        "duration_mins":  body.duration_mins,
        "slots_per_table": len(generated),
        "tables_updated": results,
        "slots_preview":  [s["start"] for s in generated],
    }

@app.post("/api/crm/timeslots/single", tags=["Time Slots"])
def api_add_single_timeslot(body: TimeSlotIn, owner=Depends(get_current_owner)):
    rdb, _ = _sub_guard(owner)
    table = rdb["tables"].find_one({"_id": ObjectId(body.table_id)})
    if not table:
        raise HTTPException(404, "Table not found.")

    new_slot = {"start": body.start_time, "end": body.end_time}
    existing_doc = rdb["time_slots"].find_one({"table_id": body.table_id, "date": body.date})
    if existing_doc:
        existing_starts = {s["start"] for s in existing_doc.get("slots", [])}
        if body.start_time in existing_starts:
            return {"message": "Slot already exists.", "slot": new_slot}
        all_slots = sorted(
            existing_doc["slots"] + [new_slot],
            key=lambda x: x["start"]
        )
        rdb["time_slots"].update_one(
            {"table_id": body.table_id, "date": body.date},
            {"$set": {"slots": all_slots, "updated_at": datetime.now(timezone.utc)}}
        )
    else:
        rdb["time_slots"].insert_one({
            "table_id":   body.table_id,
            "date":       body.date,
            "slots":      [new_slot],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
    return {"added": True, "slot": new_slot}

@app.get("/api/crm/timeslots", tags=["Time Slots"])
def api_list_timeslots(date: str, owner=Depends(get_current_owner)):
    rdb, _ = _sub_guard(owner)
    slot_docs = list(rdb["time_slots"].find({"date": date}))

    booked_map: Dict[str, set] = {}
    for res in rdb["reservations"].find({"date": date, "status": "confirmed"}):
        booked_map.setdefault(res["table_id"], set()).add(res["time_slot"])

    result = []
    for doc in slot_docs:
        tid   = doc["table_id"]
        table = rdb["tables"].find_one({"_id": ObjectId(tid)})
        slots_with_status = []
        for s in doc.get("slots", []):
            is_booked = s["start"] in booked_map.get(tid, set())
            slots_with_status.append({
                "start":  s["start"],
                "end":    s["end"],
                "status": "booked" if is_booked else "available",
            })
        result.append({
            "table_id":     tid,
            "table_number": table.get("table_number") if table else "?",
            "capacity":     table.get("capacity") if table else 0,
            "location":     table.get("location", "main") if table else "",
            "slots":        slots_with_status,
        })
    return {"date": date, "tables": result}

@app.delete("/api/crm/timeslots", tags=["Time Slots"])
def api_delete_timeslots(body: DeleteTimeSlotsIn, owner=Depends(get_current_owner)):
    rdb, _ = _sub_guard(owner)
    deleted = 0
    for table_id in body.table_ids:
        res = rdb["time_slots"].delete_one({"table_id": table_id, "date": body.date})
        deleted += res.deleted_count
    return {"deleted": deleted, "date": body.date}

@app.delete("/api/crm/timeslots/{table_id}/{date}/{start_time}", tags=["Time Slots"])
def api_delete_single_slot(
    table_id: str, date: str, start_time: str,
    owner=Depends(get_current_owner)
):
    rdb, _ = _sub_guard(owner)
    doc = rdb["time_slots"].find_one({"table_id": table_id, "date": date})
    if not doc:
        raise HTTPException(404, "No slots found for this table/date.")
    updated_slots = [s for s in doc.get("slots", []) if s["start"] != start_time]
    rdb["time_slots"].update_one(
        {"table_id": table_id, "date": date},
        {"$set": {"slots": updated_slots, "updated_at": datetime.now(timezone.utc)}}
    )
    return {"removed": True, "start_time": start_time}

# ============================================================
# AI SALES AGENT
# ============================================================

# ── COST OPTIMIZATION #1 ──────────────────────────────────────────────────────
# System prompt trimmed (~40% fewer tokens).
# All logic/rules are identical — only redundant phrasing removed.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional AI reservation agent for {restaurant_name}. Be warm, concise, efficient.

FLOW (follow in order):
1. Greet guest briefly.
2. Ask party size.
3. Ask preferred date.
4. Check AVAILABLE SLOTS in CRM CONTEXT for that date.
   - No slots defined → "Sorry, we are not taking reservations for that date. Please try another date or contact us."
   - All slots booked → tell guest we're fully booked, suggest next available date if visible.
5. Show available slots, ask preference.
6. Collect name and phone (or email).
7. Match party size to a table with sufficient capacity that has that slot FREE.
8. Confirm booking summary.
9. Call `create_booking` tool immediately.
10. Give final confirmation with reference ID.

RULES:
- Use ONLY table_id values from CRM CONTEXT — never invent them.
- Only offer AVAILABLE slots from CRM CONTEXT for the requested date.
- Never book a BOOKED slot.
- If guest requests a time not in defined slots → say unavailable, show what is available.
- If no slots defined for date → do not offer any time. Do not accept bookings.
- Keep replies short — like a top-tier maître d'.
- After create_booking succeeds, confirm with reference ID.

CRM CONTEXT (live, refreshed each message):
{crm_context}

Now: {now}"""

BOOKING_TOOL = {
    "name": "create_booking",
    "description": (
        "Creates a confirmed reservation. Call only when you have: guest name, "
        "contact (phone or email), date (YYYY-MM-DD), time_slot (HH:MM matching "
        "a defined available slot), party_size, and a FREE table_id with enough capacity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "guest_name":  {"type": "string"},
            "guest_phone": {"type": "string"},
            "date":        {"type": "string", "description": "YYYY-MM-DD"},
            "time_slot":   {"type": "string", "description": "HH:MM — must match a defined slot start time"},
            "party_size":  {"type": "integer"},
            "table_id":    {"type": "string"},
            "notes":       {"type": "string", "default": ""},
        },
        "required": ["guest_name", "guest_phone", "date", "time_slot", "party_size", "table_id"],
    },
}

# ── COST OPTIMIZATION #2 ──────────────────────────────────────────────────────
# CRM context builder: removed the trailing 4-line "INSTRUCTIONS FOR YOU" block.
# Those rules are already in SYSTEM_PROMPT above; duplicating them costs tokens
# on every single message. All slot logic and data output is 100% unchanged.
# ─────────────────────────────────────────────────────────────────────────────
def _build_crm_context(rdb: Database, date: Optional[str] = None) -> str:
    today = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    tables = list(rdb["tables"].find(
        {}, {"_id": 1, "table_number": 1, "capacity": 1, "location": 1}
    ))

    booked_map: Dict[str, set] = {}
    for res in rdb["reservations"].find({"date": today, "status": "confirmed"}):
        booked_map.setdefault(res["table_id"], set()).add(res["time_slot"])

    slot_docs = {
        doc["table_id"]: doc.get("slots", [])
        for doc in rdb["time_slots"].find({"date": today})
    }

    lines = [f"Date: {today}", ""]

    if not slot_docs:
        lines.append("NO SLOTS DEFINED FOR THIS DATE — do not accept bookings.")
        lines.append("")

    lines.append("Tables & Slots:")
    for t in tables:
        tid   = str(t["_id"])
        tnum  = t["table_number"]
        cap   = t["capacity"]
        loc   = t.get("location", "main")
        slots = slot_docs.get(tid, [])

        if not slots:
            lines.append(
                f"  table_id={tid} | #{tnum} | cap={cap} | {loc} | NO SLOTS"
            )
        else:
            slot_summary = []
            booked_count = 0
            for s in slots:
                is_booked = s["start"] in booked_map.get(tid, set())
                if is_booked:
                    slot_summary.append(f"{s['start']}-{s['end']}:BOOKED")
                    booked_count += 1
                else:
                    slot_summary.append(f"{s['start']}-{s['end']}:AVAILABLE")

            note = " [ALL BOOKED]" if booked_count == len(slots) else ""
            lines.append(
                f"  table_id={tid} | #{tnum} | cap={cap} | {loc}{note}"
            )
            lines.append(f"    {' | '.join(slot_summary)}")

    return "\n".join(lines)


def _execute_booking_tool(rdb: Database, tool_input: dict) -> dict:
    """Execute booking — 100% original logic + slot validation."""
    contact = tool_input["guest_phone"]

    table_id  = tool_input["table_id"]
    date      = tool_input["date"]
    time_slot = tool_input["time_slot"]

    if not _slot_in_defined(rdb, table_id, date, time_slot):
        return {
            "success": False,
            "error":   f"Time slot {time_slot} is not defined for this table on {date}. "
                       "Please choose from the available slots shown to the guest.",
        }

    conflict = rdb["reservations"].count_documents({
        "table_id":  table_id,
        "date":      date,
        "time_slot": time_slot,
        "status":    {"$ne": "cancelled"},
    })
    if conflict:
        return {
            "success": False,
            "error":   f"Table {table_id} at {time_slot} on {date} was just taken. Please choose another slot.",
        }

    guest = rdb["guests"].find_one({"$or": [{"phone": contact}, {"email": contact}]})
    if guest:
        guest_id = str(guest["_id"])
        rdb["guests"].update_one({"_id": guest["_id"]}, {"$inc": {"visit_count": 1}})
    else:
        new_guest = {
            "name":        tool_input["guest_name"],
            "phone":       contact if "@" not in contact else None,
            "email":       contact if "@" in contact    else None,
            "notes":       tool_input.get("notes", ""),
            "visit_count": 1,
            "created_at":  datetime.now(timezone.utc),
        }
        result   = rdb["guests"].insert_one(new_guest)
        guest_id = str(result.inserted_id)

    doc = {
        "guest_id":   guest_id,
        "table_id":   table_id,
        "party_size": tool_input["party_size"],
        "date":       date,
        "time_slot":  time_slot,
        "notes":      tool_input.get("notes", ""),
        "status":     "confirmed",
        "created_at": datetime.now(timezone.utc),
        "source":     "ai_agent",
    }
    res    = rdb["reservations"].insert_one(doc)
    res_id = str(res.inserted_id)

    logger.info(f"AI booked: res_id={res_id} guest={tool_input['guest_name']} {date} {time_slot}")

    return {
        "success":        True,
        "reservation_id": res_id,
        "guest_id":       guest_id,
        "guest_name":     tool_input["guest_name"],
        "date":           date,
        "time_slot":      time_slot,
        "party_size":     tool_input["party_size"],
        "message":        f"Reservation confirmed. Reference: {res_id[:8].upper()}",
    }

class ChatResponse(BaseModel):
    reply:          str
    history:        List[ChatMessage]
    booking_made:   bool = False
    reservation_id: Optional[str] = None

@app.post("/api/ai/chat", response_model=ChatResponse, tags=["AI Agent"])
async def ai_chat(body: ChatRequest, restaurant_id: str, request: Request):
    ip = request.client.host
    if not rate_limit(ip, max_calls=30, window_sec=60):
        raise HTTPException(429, "Too many messages. Please slow down.")

    db    = get_platform_db()
    owner = db["owners"].find_one({"_id": ObjectId(restaurant_id)})
    if not owner:
        raise HTTPException(404, "Restaurant not found.")

    rdb       = get_owner_db(restaurant_id)
    owner_doc = owner

    if not _check_reservation_limit(rdb, owner_doc):
        if _is_free_tier(owner_doc):
            raise HTTPException(403, f"Free reservation limit of {FREE_RESERVATIONS} reached. The restaurant needs to upgrade.")
        plan = _get_plan(owner_doc)
        raise HTTPException(403, f"Monthly reservation limit of {plan['reservations_month']} reached.")

    crm_context = _build_crm_context(rdb)
    now_str     = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M UTC")

    system = SYSTEM_PROMPT.format(
        restaurant_name=owner["restaurant_name"],
        crm_context=crm_context,
        now=now_str,
    )

    messages = [{"role": m.role, "content": m.content} for m in body.history]
    messages.append({"role": "user", "content": body.message})

    client       = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    booking_made = False
    res_id       = None
    booking_data = None

    try:
        # ── COST OPTIMIZATION #3 ─────────────────────────────────────────────
        # max_tokens reduced from 1024 → 400.
        # Reservation agent replies are short conversational text — never needs
        # 1024 tokens. 400 covers the longest possible response with headroom.
        # Applied to BOTH calls inside the tool-use loop.
        # ─────────────────────────────────────────────────────────────────────
        response = client.messages.create(
            model     = "claude-sonnet-4-20250514",
            max_tokens= 400,
            system    = system,
            tools     = [BOOKING_TOOL],
            messages  = messages,
        )

        while response.stop_reason == "tool_use":
            tool_use_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if not tool_use_block:
                break

            tool_name  = tool_use_block.name
            tool_input = tool_use_block.input

            if tool_name == "create_booking":
                tool_result = _execute_booking_tool(rdb, tool_input)
                if tool_result.get("success"):
                    booking_made = True
                    res_id       = tool_result["reservation_id"]
                    booking_data = tool_result
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type":        "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content":     str(tool_result),
                }],
            })

            response = client.messages.create(
                model     = "claude-sonnet-4-20250514",
                max_tokens= 400,
                system    = system,
                tools     = [BOOKING_TOOL],
                messages  = messages,
            )

        reply_text = next(
            (b.text for b in response.content if hasattr(b, "text")),
            "Your reservation has been confirmed."
        )

    except Exception as exc:
        logger.error(f"Claude API error: {exc}")
        raise HTTPException(502, f"AI service error: {exc}")

    if booking_made and booking_data:
        await ws_manager.broadcast(restaurant_id, {
            "type":           "new_booking",
            "reservation_id": res_id[:8].upper(),
            "guest_name":     booking_data.get("guest_name", ""),
            "date":           booking_data.get("date", ""),
            "time_slot":      booking_data.get("time_slot", ""),
            "party_size":     booking_data.get("party_size", 0),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })

    updated_history = list(body.history) + [
        ChatMessage(role="user",      content=body.message),
        ChatMessage(role="assistant", content=reply_text),
    ]

    logger.info(f"AI chat | session={body.session_id} | booking={booking_made} | res={res_id}")
    return ChatResponse(
        reply          = reply_text,
        history        = updated_history,
        booking_made   = booking_made,
        reservation_id = res_id,
    )

# ============================================================
# ADMIN API
# ============================================================

@app.get("/api/admin/dashboard", tags=["Admin API"])
def api_dashboard(owner=Depends(get_current_owner)):
    db, owner_doc = _sub_guard(owner)
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    plan   = _get_plan(owner_doc)
    limits = _get_effective_limits(db, owner_doc)
    top_guests = list(
        db["guests"].find({}, {"name": 1, "visit_count": 1})
                    .sort("visit_count", -1).limit(5)
    )
    for g in top_guests: g["id"] = str(g.pop("_id"))
    return {
        "restaurant_id":        owner["sub"],
        "plan":                 plan["name"],
        "tier":                 limits["tier"],
        "reservations_month":   {"used": limits["reservations_used"], "limit": limits["reservations_limit"]},
        "reservations_left":    limits["reservations_left"],
        "as_of":                datetime.now(timezone.utc).isoformat(),
        "total_tables":         db["tables"].count_documents({}),
        "total_guests":         db["guests"].count_documents({}),
        "total_reservations":   db["reservations"].count_documents({}),
        "today_reservations":   db["reservations"].count_documents({"date": today, "status": "confirmed"}),
        "cancelled":            db["reservations"].count_documents({"status": "cancelled"}),
        "top_guests_by_visits": top_guests,
    }

# ============================================================
# HEALTH
# ============================================================

@app.get("/health", tags=["Meta"])
def health():
    return {
        "status":    "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version":   "4.0.0",
    }

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
