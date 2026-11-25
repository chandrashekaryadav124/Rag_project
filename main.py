
# ==========================================
# AI COMMERCE CUSTOMER RAG - FASTAPI
# ==========================================

import os
import time
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.generation import generate_answer
from app.core.embeddings import preload_embeddings
from app.core.supabase_client import check_supabase_health
from app.database import check_mysql_health

from app.orchestration.hybrid_generation import generate_hybrid_answer


from fastapi import Depends
from app.core.auth import get_current_principal
from app.core.permissions import require_roles
from app.core.security import UserPrincipal, UserRole

from app.api.role_data import router as role_data_router
from app.api.session_routes import router as session_router


# ==========================================
# APPLICATION LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n=================================")
    print("AI COMMERCE CUSTOMER RAG STARTED")
    print("=================================")

    print("Chat UI: /ui")
    print("API Docs: /docs")
    print("Health: /health")

    if os.getenv("PRELOAD_EMBEDDINGS", "true").lower() in ("true", "1"):
        try:
            preload_embeddings()
            print("Embedding model preloaded.")
        except Exception as error:
            print("Embedding preload skipped:", str(error))

    yield

    print("Customer RAG API stopped.")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Commerce Customer RAG API",
    description="Customer Role-Based RAG Assistant",
    version="3.0.0",
    lifespan=lifespan
)


# ==========================================
# CORS
# ==========================================
app.include_router(role_data_router)
app.include_router(session_router)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in allowed_origins
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )



class HybridChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )
    session_id: str | None = None
    include_own_orders: bool = False

class SellerHybridChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    include_own_products: bool = True
    session_id: str | None = None

class SupportHybridChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    order_id: str | None = None
    include_support_tickets: bool = True
    session_id: str | None = None

class OperationsHybridChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    order_id: str | None = None
    include_inventory_overview: bool = True
    session_id: str | None = None

class AdminHybridChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    include_system_analytics: bool = True
    session_id: str | None = None
# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
@app.head("/", include_in_schema=False)
def home():

    return {
        "application": "AI Commerce Customer RAG",
        "status": "running",
        "version": "3.0.0",
        "ui": "/ui",
        "docs": "/docs"
    }


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    try:

        return {
            "status": "healthy",
            "vector_store": "supabase",
            "supabase": check_supabase_health(),
            "mysql": check_mysql_health(),
        }

    except Exception as error:
        print("Health check error:", str(error))
        is_prod = (
            os.getenv("APP_ENV", "development").lower() == "production"
            or os.getenv("RENDER", "").lower() in ("true", "1")
        )
        detail = "Service unhealthy." if is_prod else f"Service unhealthy: {error}"

        raise HTTPException(
            status_code=503,
            detail=detail
        )


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat")
def customer_chat(request: ChatRequest):

    start_time = time.perf_counter()

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = generate_answer(question)

        response_time = round(
            time.perf_counter() - start_time,
            2
        )

        print(
            f"Customer RAG response time: {response_time} seconds"
        )

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "found_context": result["found_context"],
            "response_time": response_time,
            "role": "customer"
            
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        error_message=str(error)
        print("Customer RAG Error:", error_message)

        if "504" in error_message or "DEADLINE_EXCEEDED" in error_message:

            raise HTTPException(
                status_code=504,
                detail="The AI service is taking too long. Please try again."
            )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer. Please try again."
        )




def _run_hybrid_chat(
    principal: UserPrincipal,
    raw_question: str,
    task: str | None = None,
    order_id: str | None = None,
    session_id: str | None = None,
    error_label: str = "Hybrid RAG",
    failure_detail: str = "Unable to generate a hybrid answer. Please try again.",
) -> dict:
    """Validate input, run hybrid generation, and map errors consistently."""
    start_time = time.perf_counter()

    question = raw_question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = generate_hybrid_answer(
            principal=principal,
            question=question,
            task=task,
            order_id=order_id,
            session_id=session_id,
        )

        response_time = round(
            time.perf_counter() - start_time,
            2
        )

        return {
            "question": question,
            "answer": result["answer"],
            "role": result["role"],
            "session_id": result["session_id"],
            "sources": result["sources"],
            "found_context": result["found_context"],
            "retrieval_time": result["retrieval_time"],
            "generation_time": result["generation_time"],
            "response_time": response_time,
        }

    except PermissionError as error:
        raise HTTPException(
            status_code=403,
            detail=str(error)
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        ) from error

    except Exception as error:
        print(f"{error_label} Error:", str(error))
        raise HTTPException(
            status_code=500,
            detail=failure_detail
        ) from error


# ==========================================
# AUTHENTICATED CUSTOMER HYBRID CHAT
# ==========================================

@app.post("/api/v1/hybrid-chat")
def authenticated_hybrid_chat(
    request: HybridChatRequest,
    user: UserPrincipal = Depends(
        require_roles(UserRole.CUSTOMER)
    )
):
    task = "own_orders" if request.include_own_orders else None
    return _run_hybrid_chat(
        principal=user,
        raw_question=request.question,
        task=task,
        session_id=request.session_id,
        error_label="Hybrid generation",
    )


@app.post("/api/v1/seller/hybrid-chat")
def authenticated_seller_hybrid_chat(
    request: SellerHybridChatRequest,
    user: UserPrincipal = Depends(
        require_roles(UserRole.SELLER)
    )
):
    task = "own_products" if request.include_own_products else None
    return _run_hybrid_chat(
        principal=user,
        raw_question=request.question,
        task=task,
        session_id=request.session_id,
        error_label="Seller Hybrid RAG",
        failure_detail="Unable to generate Seller AI answer.",
    )


# ==========================================
# AUTHENTICATED SUPPORT HYBRID CHAT
# ==========================================

@app.post("/api/v1/support/hybrid-chat")
def authenticated_support_hybrid_chat(
    request: SupportHybridChatRequest,
    user: UserPrincipal = Depends(
        require_roles(UserRole.SUPPORT)
    )
):
    order_id = (
        request.order_id.strip()
        if request.order_id and request.order_id.strip()
        else None
    )

    if order_id and request.include_support_tickets:
        task = "order_and_tickets"
    elif order_id:
        task = "order_summary"
    elif request.include_support_tickets:
        task = "support_tickets"
    else:
        task = None

    return _run_hybrid_chat(
        principal=user,
        raw_question=request.question,
        task=task,
        order_id=order_id,
        session_id=request.session_id,
        error_label="Support Hybrid RAG",
        failure_detail="Unable to generate Support AI answer.",
    )


# ==========================================
# AUTHENTICATED OPERATIONS HYBRID CHAT
# ==========================================

@app.post("/api/v1/operations/hybrid-chat")
def authenticated_operations_hybrid_chat(
    request: OperationsHybridChatRequest,
    user: UserPrincipal = Depends(
        require_roles(UserRole.OPERATIONS)
    )
):
    order_id = (
        request.order_id.strip()
        if request.order_id and request.order_id.strip()
        else None
    )

    if order_id:
        task = "shipment_summary"
    elif request.include_inventory_overview:
        task = "inventory_overview"
    else:
        task = None

    return _run_hybrid_chat(
        principal=user,
        raw_question=request.question,
        task=task,
        order_id=order_id,
        session_id=request.session_id,
        error_label="Operations Hybrid RAG",
        failure_detail="Unable to generate Operations AI answer.",
    )


# ==========================================
# AUTHENTICATED ADMIN HYBRID CHAT
# ==========================================

@app.post("/api/v1/admin/hybrid-chat")
def authenticated_admin_hybrid_chat(
    request: AdminHybridChatRequest,
    user: UserPrincipal = Depends(
        require_roles(UserRole.ADMIN)
    )
):
    task = "system_analytics" if request.include_system_analytics else None
    return _run_hybrid_chat(
        principal=user,
        raw_question=request.question,
        task=task,
        session_id=request.session_id,
        error_label="Admin Hybrid RAG",
        failure_detail="Unable to generate Admin AI answer.",
    )



@app.get("/auth/test")
def test_authentication(
    user: UserPrincipal = Depends(get_current_principal)
):
    return {
        "message": "Authentication successful",
        "user_id": user.user_id,
        "role": user.role.value
    }


@app.get("/auth/customer-test")
def customer_test(
    user: UserPrincipal = Depends(
        require_roles(UserRole.CUSTOMER)
    )
):
    return {
        "message": "Customer access granted",
        "user_id": user.user_id
    }


@app.get("/auth/admin-test")
def admin_test(
    user: UserPrincipal = Depends(
        require_roles(UserRole.ADMIN)
    )
):
    return {
        "message": "Admin access granted",
        "user_id": user.user_id
    }





# ==========================================
# CUSTOMER CHAT UI
# ==========================================

@app.get("/ui", response_class=HTMLResponse)
def customer_ui():

    html = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#f5f7fb">
<title>AI Commerce | Customer Support</title>
<style>
:root {
    color-scheme: light;
    --ink: #17233c;
    --muted: #78859d;
    --line: #e8edf5;
    --blue: #526ee5;
    --blue-dark: #405bd0;
    --soft-blue: #eef2ff;
    --green: #17865b;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    min-width: 320px;
    min-height: 100vh;
    background:
        radial-gradient(ellipse at 86% 0%, rgba(205, 217, 255, .36), transparent 34rem),
        #f5f7fb;
    color: var(--ink);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}

button, textarea { font: inherit; }
button { -webkit-tap-highlight-color: transparent; }

.topbar {
    height: 72px;
    padding: 0 max(28px, calc((100vw - 1240px) / 2));
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255, 255, 255, .88);
    border-bottom: 1px solid rgba(226, 232, 242, .85);
}

.brand, .topbar-actions, .brand-mark, .online, .panel-title,
.new-chat, .topic-button, .composer-meta, .send-button {
    display: flex;
    align-items: center;
}

.brand { gap: 11px; text-decoration: none; color: inherit; }

.brand-mark {
    width: 39px;
    height: 39px;
    justify-content: center;
    border-radius: 13px;
    color: white;
    background: linear-gradient(145deg, #7189f4, #4a62d8);
    box-shadow: 0 6px 15px rgba(76, 99, 215, .23);
}

.brand-name { font-size: 15px; font-weight: 750; letter-spacing: -.3px; }
.brand-name span { color: var(--blue); }
.brand-caption { margin-top: 3px; color: var(--muted); font-size: 11px; }
.topbar-actions { gap: 18px; }
.top-link { color: #65728a; font-size: 12px; text-decoration: none; }
.top-link:hover { color: var(--blue); }

.online {
    gap: 8px;
    padding: 8px 12px;
    border: 1px solid #dcefe5;
    border-radius: 24px;
    color: #347b5c;
    background: #f1faf5;
    font-size: 11px;
    font-weight: 650;
}

.online-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #29a66d;
    box-shadow: 0 0 0 3px rgba(41, 166, 109, .12);
}

.workspace {
    width: min(1240px, calc(100% - 48px));
    min-height: calc(100vh - 108px);
    margin: 18px auto;
    display: grid;
    grid-template-columns: 252px minmax(0, 1fr);
    gap: 18px;
}

.sidebar, .chat-panel {
    border: 1px solid rgba(229, 234, 243, .95);
    border-radius: 19px;
    background: rgba(255, 255, 255, .93);
    box-shadow: 0 12px 40px rgba(41, 57, 91, .055);
}

.sidebar { padding: 25px 17px; }
.sidebar-kicker {
    margin: 0 10px 9px;
    color: #9aa5b7;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.15px;
    text-transform: uppercase;
}

.sidebar-title {
    margin: 0 10px 21px;
    color: #34425c;
    font-size: 14px;
    font-weight: 700;
}

.topic-list { display: grid; gap: 7px; }

.topic-button {
    width: 100%;
    gap: 11px;
    padding: 11px 10px;
    border: 1px solid transparent;
    border-radius: 11px;
    color: #596780;
    background: transparent;
    text-align: left;
    cursor: pointer;
    transition: .18s ease;
}

.topic-button:hover, .topic-button:focus-visible {
    border-color: #e2e8fb;
    background: #f5f7ff;
    color: var(--blue-dark);
    outline: none;
}

.topic-icon {
    display: grid;
    width: 31px;
    height: 31px;
    flex: 0 0 31px;
    place-items: center;
    border-radius: 10px;
    color: #6075d7;
    background: #f0f3ff;
    font-size: 15px;
}

.topic-label { font-size: 12px; font-weight: 620; }
.sidebar-divider { height: 1px; margin: 22px 4px; background: var(--line); }

.knowledge-card {
    padding: 15px 14px;
    border: 1px solid #e7ebfa;
    border-radius: 13px;
    background: linear-gradient(145deg, #f8f9ff, #f1f4ff);
}

.knowledge-card-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #43557f;
    font-size: 11px;
    font-weight: 700;
}

.knowledge-card p {
    margin: 9px 0 0;
    color: #7d89a0;
    font-size: 11px;
    line-height: 1.65;
}

.sidebar-note {
    margin: 18px 9px 0;
    color: #939daf;
    font-size: 10px;
    line-height: 1.65;
}

.auth-card {
    padding: 13px;
    margin-bottom: 18px;
    border: 1px solid #e3e8f6;
    border-radius: 13px;
    background: #f8faff;
}

.auth-card-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    color: #34425c;
    font-size: 11px;
    font-weight: 720;
}

.auth-input {
    width: 100%;
    padding: 8px 10px;
    border: 1px solid #dce3f2;
    border-radius: 8px;
    color: var(--ink);
    background: #fff;
    font-family: inherit;
    font-size: 11px;
    outline: none;
}

.auth-input:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 2px rgba(82, 110, 229, .12);
}

.auth-actions {
    display: flex;
    gap: 6px;
    margin-top: 8px;
}

.auth-btn {
    flex: 1;
    padding: 7px 10px;
    border: 0;
    border-radius: 8px;
    color: #fff;
    background: var(--blue);
    font-size: 11px;
    font-weight: 650;
    cursor: pointer;
}

.auth-btn:hover { background: var(--blue-dark); }
.auth-btn:disabled { opacity: .6; cursor: not-allowed; }

.auth-clear {
    padding: 7px 10px;
    border: 1px solid #dce3f2;
    border-radius: 8px;
    color: #596780;
    background: #fff;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
}

.auth-status {
    margin-top: 8px;
    padding: 6px 8px;
    border-radius: 8px;
    font-size: 10px;
    line-height: 1.45;
    color: #596780;
    background: #eef2fb;
    overflow-wrap: anywhere;
}

.auth-status.verified {
    color: #156d49;
    background: #eaf8f1;
    border: 1px solid #cdecd9;
}

.auth-status.error {
    color: #9e2a2b;
    background: #fdeeee;
    border: 1px solid #f6d0d0;
}

.order-filter-wrap {
    margin-top: 9px;
    padding-top: 9px;
    border-top: 1px solid #e4e9f5;
}

.order-filter-label {
    display: block;
    margin-bottom: 4px;
    color: #65728a;
    font-size: 10px;
    font-weight: 650;
}

.meta-timing {
    margin-top: 6px;
    color: #8894ab;
    font-size: 10px;
}

.chat-panel {
    min-width: 0;
    min-height: 650px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.chat-header {
    min-height: 76px;
    padding: 0 25px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #edf0f6;
}

.panel-title { gap: 12px; }

.assistant-avatar {
    display: grid;
    width: 40px;
    height: 40px;
    flex: 0 0 40px;
    place-items: center;
    border-radius: 14px;
    color: #fff;
    background: linear-gradient(145deg, #7087f0, #4b62d5);
    box-shadow: 0 5px 12px rgba(76, 99, 215, .19);
}

.panel-heading { font-size: 13px; font-weight: 720; }
.panel-subheading { margin-top: 4px; color: #929db0; font-size: 10px; }

.new-chat {
    gap: 7px;
    padding: 9px 12px;
    border: 1px solid #e6eaf2;
    border-radius: 9px;
    color: #68758d;
    background: white;
    font-size: 11px;
    font-weight: 620;
    cursor: pointer;
    transition: .18s ease;
}

.new-chat:hover { border-color: #cbd4f5; color: var(--blue-dark); background: #f8f9ff; }

.conversation {
    width: 100%;
    max-width: 820px;
    min-height: 0;
    flex: 1;
    margin: 0 auto;
    padding: 26px 34px 18px;
    display: flex;
    flex-direction: column;
}

#messages {
    min-height: 0;
    flex: 1;
    overflow-y: auto;
    padding: 4px 3px 14px;
    scroll-behavior: smooth;
}

.welcome {
    max-width: 610px;
    margin: auto;
    padding: 24px 0 31px;
    text-align: center;
    animation: appear .35s ease both;
}

.welcome-icon {
    display: grid;
    width: 61px;
    height: 61px;
    margin: 0 auto 19px;
    place-items: center;
    border: 1px solid #e4e9ff;
    border-radius: 20px;
    color: #5e73dc;
    background: linear-gradient(145deg, #f4f6ff, #e9edff);
    box-shadow: 0 10px 25px rgba(81, 103, 210, .11);
}

.eyebrow {
    margin: 0 0 10px;
    color: #7788d5;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.35px;
    text-transform: uppercase;
}

.welcome h1 {
    margin: 0;
    color: #202d48;
    font-size: clamp(24px, 3vw, 31px);
    font-weight: 730;
    letter-spacing: -1.05px;
}

.welcome-copy {
    max-width: 475px;
    margin: 12px auto 25px;
    color: #8792a6;
    font-size: 13px;
    line-height: 1.7;
}

.suggestion-heading {
    margin: 0 0 11px;
    color: #929db0;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .45px;
    text-align: left;
}

.suggestions {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px;
    text-align: left;
}

.suggestion {
    min-height: 58px;
    padding: 11px 13px;
    border: 1px solid #e8ecf5;
    border-radius: 11px;
    color: #596781;
    background: #fff;
    cursor: pointer;
    text-align: left;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}

.suggestion:hover, .suggestion:focus-visible {
    transform: translateY(-2px);
    border-color: #cbd5ff;
    box-shadow: 0 7px 17px rgba(51, 72, 132, .08);
    outline: none;
}

.suggestion strong { display: block; color: #3f4e69; font-size: 11px; font-weight: 680; }
.suggestion span { display: block; margin-top: 4px; color: #929db0; font-size: 10px; }

.message-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 15px 0;
    animation: appear .22s ease both;
}

.message-row.user { justify-content: flex-end; }

.message-avatar {
    display: grid;
    width: 29px;
    height: 29px;
    flex: 0 0 29px;
    place-items: center;
    border-radius: 10px;
    color: #fff;
    background: linear-gradient(145deg, #7087f0, #4b62d5);
    font-size: 9px;
    font-weight: 750;
}

.message-row.user .message-avatar {
    order: 2;
    color: #5268ce;
    background: #edf0ff;
}

.message-content { max-width: min(78%, 610px); min-width: 0; }
.message-row.user .message-content { display: flex; flex-direction: column; align-items: flex-end; }

.bubble {
    padding: 12px 15px;
    border: 1px solid #e9edf4;
    border-radius: 4px 14px 14px 14px;
    color: #49566f;
    background: #fff;
    font-size: 12px;
    line-height: 1.75;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.message-row.assistant .bubble {
    white-space: normal;
}

.bubble h3, .bubble h4 {
    margin: 12px 0 6px;
    color: var(--ink);
    font-size: 13px;
    font-weight: 700;
}

.bubble h3:first-child, .bubble h4:first-child { margin-top: 0; }
.bubble p { margin: 0 0 8px; }
.bubble p:last-child { margin-bottom: 0; }
.bubble ul, .bubble ol { margin: 6px 0 8px; padding-left: 18px; }
.bubble li { margin: 3px 0; }
.bubble hr { border: 0; border-top: 1px solid #edf0f6; margin: 10px 0; }
.bubble code {
    padding: 1px 5px;
    border-radius: 5px;
    background: #f1f4fb;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 11px;
}

.bubble table {
    width: 100%;
    margin: 8px 0;
    border-collapse: collapse;
    font-size: 11px;
    display: block;
    overflow-x: auto;
}

.bubble th, .bubble td {
    padding: 6px 9px;
    border: 1px solid #e6ebf5;
    text-align: left;
}

.bubble th {
    background: #f6f8fd;
    color: #34425c;
    font-weight: 680;
}

.message-row.user .bubble {
    border: 0;
    border-radius: 14px 4px 14px 14px;
    color: #fff;
    background: linear-gradient(135deg, #637be9, #4c65d9);
    box-shadow: 0 5px 13px rgba(76, 99, 215, .15);
}

.sources {
    margin-top: 8px;
    padding: 10px 12px;
    border: 1px solid #e8ecf6;
    border-radius: 10px;
    color: #72809a;
    background: #f7f8fc;
    font-size: 10px;
    line-height: 1.75;
}

.sources-heading { display: block; margin-bottom: 4px; color: #566580; font-weight: 720; }
.source-item { overflow-wrap: anywhere; }

.loading {
    display: flex;
    min-width: 52px;
    align-items: center;
    gap: 5px;
    padding: 5px 2px;
}

.loading span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #7890e8;
    animation: pulse 1s infinite;
}

.loading span:nth-child(2) { animation-delay: .15s; }
.loading span:nth-child(3) { animation-delay: .3s; }

@keyframes pulse {
    0%, 100% { opacity: .35; transform: translateY(0); }
    50% { opacity: 1; transform: translateY(-4px); }
}

@keyframes appear {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

.composer-wrap { padding: 12px 0 0; }

.composer {
    padding: 9px 10px 8px 15px;
    border: 1px solid #e1e7f1;
    border-radius: 15px;
    background: #fff;
    box-shadow: 0 6px 22px rgba(41, 57, 91, .06);
    transition: border-color .18s ease, box-shadow .18s ease;
}

.composer:focus-within {
    border-color: #aebcf6;
    box-shadow: 0 0 0 3px rgba(82, 110, 229, .08), 0 6px 22px rgba(41, 57, 91, .06);
}

.composer-row { display: flex; align-items: flex-end; gap: 10px; }

#question {
    width: 100%;
    min-height: 42px;
    max-height: 130px;
    flex: 1;
    resize: none;
    border: 0;
    outline: 0;
    padding: 11px 0 8px;
    color: #35425b;
    background: transparent;
    font-size: 12px;
    line-height: 1.55;
}

#question::placeholder { color: #a0a9b9; }

.send-button {
    width: 42px;
    height: 42px;
    flex: 0 0 42px;
    justify-content: center;
    border: 0;
    border-radius: 11px;
    color: #fff;
    background: var(--blue);
    cursor: pointer;
    transition: background .18s ease, transform .18s ease;
}

.send-button:hover { transform: translateY(-1px); background: var(--blue-dark); }
.send-button:disabled { opacity: .55; cursor: not-allowed; transform: none; }

.composer-meta {
    justify-content: space-between;
    margin-top: 4px;
    color: #a2abba;
    font-size: 9px;
}

.hint {
    margin: 11px 0 0;
    color: #9aa4b5;
    font-size: 10px;
    line-height: 1.5;
    text-align: center;
}

@media (max-width: 800px) {
    .workspace { width: min(100% - 28px, 760px); grid-template-columns: 1fr; }
    .chat-panel { min-height: 520px; }
    .topbar { padding: 0 20px; }
    .conversation { padding: 22px 24px 17px; }
}

@media (max-width: 520px) {
    .topbar { height: 64px; padding: 0 15px; }
    .brand-mark { width: 35px; height: 35px; border-radius: 11px; }
    .brand-caption, .top-link { display: none; }
    .online { padding: 7px 9px; font-size: 10px; }
    .workspace { width: 100%; min-height: calc(100vh - 64px); margin: 0; }
    .chat-panel { min-height: calc(100vh - 64px); border-right: 0; border-left: 0; border-radius: 0; }
    .chat-header { min-height: 68px; padding: 0 16px; }
    .assistant-avatar { width: 36px; height: 36px; flex-basis: 36px; }
    .panel-heading { font-size: 12px; }
    .panel-subheading { font-size: 9px; }
    .new-chat { padding: 8px; font-size: 0; }
    .new-chat svg { width: 16px; height: 16px; }
    .conversation { padding: 15px 15px 14px; }
    .welcome { padding: 18px 0 23px; }
    .welcome h1 { font-size: 25px; }
    .welcome-copy { font-size: 12px; }
    .suggestions { grid-template-columns: 1fr; gap: 7px; }
    .suggestion { min-height: 53px; }
    .message-content { max-width: 85%; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
</style>
</head>
<body>
<header class="topbar">
    <a class="brand" href="/" aria-label="AI Commerce home">
        <span class="brand-mark" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 3.5 14.3 9.7 20.5 12l-6.2 2.3L12 20.5l-2.3-6.2L3.5 12l6.2-2.3L12 3.5Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
                <path d="m19 3 .7 2.3L22 6l-2.3.7L19 9l-.7-2.3L16 6l2.3-.7L19 3Z" fill="currentColor"/>
            </svg>
        </span>
        <span>
            <span class="brand-name">AI <span>Commerce</span></span>
            <span class="brand-caption">Customer support, made simple</span>
        </span>
    </a>
    <div class="topbar-actions">
        <a class="top-link" href="/docs">API documentation</a>
        <div class="online"><span class="online-dot"></span>Assistant online</div>
    </div>
</header>

<main class="workspace">
    <aside class="sidebar" aria-label="Support topics and authentication">
        <div class="auth-card">
            <div class="auth-card-title">
                <span>Phase I Token Tester</span>
                <span id="roleBadge">Guest</span>
            </div>
            <input class="auth-input" id="authTokenInput" type="password" placeholder="Paste Bearer JWT from Phase I..." aria-label="Phase I JWT Token" autocomplete="off" />
            <div class="auth-actions">
                <button class="auth-btn" id="verifyTokenBtn" type="button">Verify Token</button>
                <button class="auth-clear" id="clearTokenBtn" type="button" hidden>Clear</button>
            </div>
            <div class="auth-status" id="authStatus">Guest mode · Public Customer KB (/chat)</div>
            <div class="order-filter-wrap" id="orderFilterWrap" hidden>
                <label class="order-filter-label" for="orderIdInput">Optional Order ID (Support / Ops)</label>
                <input class="auth-input" id="orderIdInput" type="text" placeholder="e.g. ORD25B99B4" autocomplete="off" />
            </div>
        </div>
        <p class="sidebar-kicker">Quick prompts</p>
        <h2 class="sidebar-title">What can we help with?</h2>
        <nav class="topic-list" id="topicList"></nav>
        <div class="sidebar-divider"></div>
        <div class="knowledge-card">
            <div class="knowledge-card-title">
                <span aria-hidden="true">✦</span> Answers you can trust
            </div>
            <p id="knowledgeCopy">Responses are grounded in the approved customer knowledge base, with source references when available.</p>
        </div>
        <p class="sidebar-note" id="sidebarNote">Paste a Phase I JWT above to test role-based MySQL + RAG + memory.</p>
    </aside>

    <section class="chat-panel" aria-label="Customer support chat">
        <div class="chat-header">
            <div class="panel-title">
                <span class="assistant-avatar" aria-hidden="true">
                    <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
                        <path d="M12 3.5 14.3 9.7 20.5 12l-6.2 2.3L12 20.5l-2.3-6.2L3.5 12l6.2-2.3L12 3.5Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
                    </svg>
                </span>
                <span>
                    <span class="panel-heading" id="panelHeading">Customer support assistant</span>
                    <span class="panel-subheading" id="panelSubheading">Public Knowledge Base mode · Paste a Phase I token to unlock role data</span>
                </span>
            </div>
            <button class="new-chat" id="newChatButton" type="button" aria-label="Start a new chat" title="Start a new chat">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                </svg>
                New chat
            </button>
        </div>

        <div class="conversation">
            <div id="messages" role="log" aria-live="polite" aria-relevant="additions text">
                <section class="welcome" id="welcome">
                    <div class="welcome-icon" aria-hidden="true">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                            <path d="M12 3.5 14.3 9.7 20.5 12l-6.2 2.3L12 20.5l-2.3-6.2L3.5 12l6.2-2.3L12 3.5Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
                            <path d="m19 3 .7 2.3L22 6l-2.3.7L19 9l-.7-2.3L16 6l2.3-.7L19 3Z" fill="currentColor"/>
                        </svg>
                    </div>
                    <p class="eyebrow">Your personal support assistant</p>
                    <h1>How can we help today?</h1>
                    <p class="welcome-copy">Ask a question about our products, delivery, returns, or policies. We’ll find the most helpful answer for you.</p>
                    <p class="suggestion-heading">POPULAR QUESTIONS</p>
                    <div class="suggestions" id="suggestionsGrid"></div>
                </section>
            </div>

            <form class="composer-wrap" id="chatForm">
                <div class="composer">
                    <div class="composer-row">
                        <textarea id="question" rows="1" maxlength="2000" placeholder="Ask us anything..." aria-label="Your question" autocomplete="off"></textarea>
                        <button class="send-button" id="sendButton" type="submit" aria-label="Send message" title="Send message">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                <path d="M5 12h14M13 5l7 7-7 7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                    <div class="composer-meta"><span>Enter to send · Shift + Enter for a new line</span><span id="charCount">0 / 2000</span></div>
                </div>
                <p class="hint">AI answers are based on the Customer Knowledge Base. Verify live order details in the application.</p>
            </form>
        </div>
    </section>
</main>

<script>
let isLoading = false;
let activeSessionId = null;
const authState = { token: "", userId: null, role: null };
const welcomePanel = document.getElementById("welcome");

const ROLE_CONFIG = {
    guest: {
        heading: "Customer support assistant",
        subheading: "Public Knowledge Base mode · Paste a Phase I token to unlock role data",
        topics: [
            { icon: "↩", label: "Returns & exchanges", question: "What is the return policy?" },
            { icon: "₹", label: "Payments & refunds", question: "How long do refunds take?" },
            { icon: "⌁", label: "Shipping & delivery", question: "Who delivers customer orders?" },
            { icon: "◇", label: "Products & availability", question: "What products are available?" }
        ],
        suggestions: [
            { title: "What is your return policy?", desc: "Learn about returns and exchanges", question: "What is the return policy?" },
            { title: "When will my refund arrive?", desc: "Get details on refund timelines", question: "How long do refunds take?" },
            { title: "How are orders delivered?", desc: "Find out about shipping and delivery", question: "Who delivers customer orders?" },
            { title: "What products do you offer?", desc: "Explore products and availability", question: "What products are available?" }
        ]
    },
    customer: {
        endpoint: "/api/v1/hybrid-chat",
        heading: "Customer Hybrid Assistant",
        subheading: "Authenticated · Live MySQL Orders + Customer Knowledge Base + Memory",
        topics: [
            { icon: "📦", label: "My recent orders", question: "What are my recent orders and what is your return policy?" },
            { icon: "₹", label: "Highest order total", question: "Which of my orders had the highest amount?" },
            { icon: "↩", label: "Return eligibility", question: "How do I request a return for an order?" },
            { icon: "⏱", label: "Refund timeline", question: "How long do refunds take after approval?" }
        ],
        suggestions: [
            { title: "My orders & return policy", desc: "Live MySQL orders + KB rules", question: "What are my recent orders and what is your return policy?" },
            { title: "Order status overview", desc: "Check processing or dispatched orders", question: "Summarize the current status of my orders." },
            { title: "How to request a return", desc: "Step-by-step return guide", question: "How can I submit a return request?" },
            { title: "Refund timeline", desc: "Bank & refund processing details", question: "How long do refunds take?" }
        ]
    },
    seller: {
        endpoint: "/api/v1/seller/hybrid-chat",
        heading: "Seller Hybrid Assistant",
        subheading: "Authenticated · Approved Seller Catalog & Stock Analytics",
        topics: [
            { icon: "🏷", label: "My product catalog", question: "List my approved products, selling prices in INR, and current stock quantities." },
            { icon: "⚠", label: "Low stock alerts", question: "Which of my products have low stock or inactive status?" },
            { icon: "₹", label: "Pricing & discounts", question: "Summarize my product selling prices and discount prices." }
        ],
        suggestions: [
            { title: "List my approved products", desc: "SKUs, prices in INR & stock", question: "List my approved products, selling prices in INR, and current stock quantities." },
            { title: "Check low stock items", desc: "Identify products needing restock", question: "Which of my products have the lowest stock?" },
            { title: "Discount pricing overview", desc: "Review selling vs discount prices", question: "Summarize my product selling prices and discount prices." },
            { title: "Listing status summary", desc: "Active vs inactive listings", question: "Show the listing status breakdown of my products." }
        ]
    },
    support: {
        endpoint: "/api/v1/support/hybrid-chat",
        heading: "Support Hybrid Assistant",
        subheading: "Authenticated · Customer Support Tickets & Order Lookup",
        topics: [
            { icon: "🎫", label: "Latest support tickets", question: "Summarize the latest customer support tickets and their statuses." },
            { icon: "🔎", label: "Order ticket lookup", question: "What issue was reported for this order and what is the resolution guidance?" },
            { icon: "📋", label: "Pending tickets", question: "Which support tickets are currently open or unresolved?" }
        ],
        suggestions: [
            { title: "Summarize latest tickets", desc: "Recent customer issues & statuses", question: "Summarize the latest customer support tickets and their statuses." },
            { title: "Open / unresolved tickets", desc: "Prioritize pending customer issues", question: "Which support tickets are still open and need action?" },
            { title: "Inspect specific order", desc: "Enter an Order ID on the left first", question: "What is the order summary and ticket history for this order?" },
            { title: "Resolution guidance", desc: "Review recommended support actions", question: "Summarize the resolution guidance for recent tickets." }
        ]
    },
    operations: {
        endpoint: "/api/v1/operations/hybrid-chat",
        heading: "Operations Hybrid Assistant",
        subheading: "Authenticated · Inventory Snapshots & Shipment Tracking",
        topics: [
            { icon: "📊", label: "Inventory snapshot", question: "Summarize the latest inventory snapshot across all inventory statuses." },
            { icon: "🚚", label: "Order shipment lookup", question: "What is the shipment status, courier, and delayed flag for this order?" },
            { icon: "⚠", label: "Reserved vs available", question: "Compare total available and reserved inventory quantities." }
        ],
        suggestions: [
            { title: "Latest inventory snapshot", desc: "Available & reserved counts by status", question: "Summarize the latest inventory snapshot across all inventory statuses." },
            { title: "Track order shipment", desc: "Enter an Order ID on the left first", question: "What is the shipment status, courier, and delayed flag for this order?" },
            { title: "Stock health summary", desc: "Check inventory availability", question: "Compare total available and reserved inventory quantities." },
            { title: "Delivery status overview", desc: "Courier & dispatch details", question: "Give me an operational summary of inventory readiness." }
        ]
    },
    admin: {
        endpoint: "/api/v1/admin/hybrid-chat",
        heading: "Admin Hybrid Assistant",
        subheading: "Authenticated · Platform-Wide System Analytics",
        topics: [
            { icon: "📈", label: "Platform analytics", question: "Give me an executive summary of total customers, orders, products, sellers, returns, and support tickets." },
            { icon: "👥", label: "Customers vs sellers", question: "Compare total registered customers and sellers on the platform." },
            { icon: "↩", label: "Returns & tickets", question: "Summarize return and support ticket volumes relative to total orders." }
        ],
        suggestions: [
            { title: "Executive platform summary", desc: "All core entity totals", question: "Give me an executive summary of total customers, orders, products, sellers, returns, and support tickets." },
            { title: "Marketplace scale", desc: "Customers, sellers & products", question: "Compare total customers, sellers, and listed products." },
            { title: "Post-order volume", desc: "Orders, returns & support tickets", question: "Summarize order volume alongside returns and support tickets." },
            { title: "Platform health check", desc: "High-level operational metrics", question: "Provide a concise overview of platform metrics." }
        ]
    }
};

function renderRoleUI(roleKey) {
    const cfg = ROLE_CONFIG[roleKey] || ROLE_CONFIG.guest;
    document.getElementById("panelHeading").textContent = cfg.heading;
    document.getElementById("panelSubheading").textContent = cfg.subheading;

    const topicList = document.getElementById("topicList");
    topicList.replaceChildren();
    cfg.topics.forEach(item => {
        const btn = document.createElement("button");
        btn.className = "topic-button";
        btn.type = "button";
        const icon = document.createElement("span");
        icon.className = "topic-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = item.icon;
        const label = document.createElement("span");
        label.className = "topic-label";
        label.textContent = item.label;
        btn.appendChild(icon);
        btn.appendChild(label);
        btn.addEventListener("click", () => ask(item.question));
        topicList.appendChild(btn);
    });

    const suggestionsGrid = document.getElementById("suggestionsGrid");
    suggestionsGrid.replaceChildren();
    cfg.suggestions.forEach(item => {
        const btn = document.createElement("button");
        btn.className = "suggestion";
        btn.type = "button";
        const strong = document.createElement("strong");
        strong.textContent = item.title;
        const span = document.createElement("span");
        span.textContent = item.desc;
        btn.appendChild(strong);
        btn.appendChild(span);
        btn.addEventListener("click", () => ask(item.question));
        suggestionsGrid.appendChild(btn);
    });

    const orderWrap = document.getElementById("orderFilterWrap");
    orderWrap.hidden = !(roleKey === "support" || roleKey === "operations");

    const knowledgeCopy = document.getElementById("knowledgeCopy");
    knowledgeCopy.textContent = roleKey === "guest"
        ? "Responses are grounded in the approved customer knowledge base, with source references when available."
        : "Authenticated hybrid mode active: answers combine role-scoped MySQL data, Knowledge Base docs, and Supabase memory.";
}

function appendInlineMarkdown(parent, text) {
    const tokenPattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let lastIndex = 0;
    let match;
    while ((match = tokenPattern.exec(text)) !== null) {
        if (match.index > lastIndex) {
            parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
        }
        const token = match[0];
        if (token.startsWith("**")) {
            const strong = document.createElement("strong");
            strong.textContent = token.slice(2, -2);
            parent.appendChild(strong);
        } else {
            const code = document.createElement("code");
            code.textContent = token.slice(1, -1);
            parent.appendChild(code);
        }
        lastIndex = tokenPattern.lastIndex;
    }
    if (lastIndex < text.length) {
        parent.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
}

function parseTableCells(line) {
    return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
}

function isTableSeparator(line) {
    return /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$/.test(line.trim());
}

function renderFormattedMarkdown(container, text) {
    const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        const trimmed = line.trim();
        if (!trimmed) { i++; continue; }

        if (/^---+$/.test(trimmed)) {
            container.appendChild(document.createElement("hr"));
            i++;
            continue;
        }

        const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
        if (headingMatch) {
            const tag = headingMatch[1].length <= 3 ? "h3" : "h4";
            const heading = document.createElement(tag);
            appendInlineMarkdown(heading, headingMatch[2]);
            container.appendChild(heading);
            i++;
            continue;
        }

        if (trimmed.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
            const table = document.createElement("table");
            const thead = document.createElement("thead");
            const headRow = document.createElement("tr");
            parseTableCells(lines[i]).forEach(cell => {
                const th = document.createElement("th");
                appendInlineMarkdown(th, cell);
                headRow.appendChild(th);
            });
            thead.appendChild(headRow);
            table.appendChild(thead);
            i += 2;

            const tbody = document.createElement("tbody");
            while (i < lines.length && lines[i].trim().includes("|")) {
                const tr = document.createElement("tr");
                parseTableCells(lines[i]).forEach(cell => {
                    const td = document.createElement("td");
                    appendInlineMarkdown(td, cell);
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
                i++;
            }
            table.appendChild(tbody);
            container.appendChild(table);
            continue;
        }

        if (/^[-*•]\s+/.test(trimmed)) {
            const ul = document.createElement("ul");
            while (i < lines.length && /^[-*•]\s+/.test(lines[i].trim())) {
                const li = document.createElement("li");
                appendInlineMarkdown(li, lines[i].trim().replace(/^[-*•]\s+/, ""));
                ul.appendChild(li);
                i++;
            }
            container.appendChild(ul);
            continue;
        }

        if (/^\d+\.\s+/.test(trimmed)) {
            const ol = document.createElement("ol");
            while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
                const li = document.createElement("li");
                appendInlineMarkdown(li, lines[i].trim().replace(/^\d+\.\s+/, ""));
                ol.appendChild(li);
                i++;
            }
            container.appendChild(ol);
            continue;
        }

        const p = document.createElement("p");
        appendInlineMarkdown(p, trimmed);
        container.appendChild(p);
        i++;
    }
}

function addMessage(text, type, sources = [], timing = null) {
    const row = document.createElement("div");
    row.className = "message-row " + type;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = type === "user" ? "YOU" : "AI";
    avatar.setAttribute("aria-hidden", "true");
    const content = document.createElement("div");
    content.className = "message-content";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (type === "assistant") {
        renderFormattedMarkdown(bubble, text);
    } else {
        bubble.textContent = text;
    }
    content.appendChild(bubble);
    if (sources.length > 0) {
        const sourceBox = document.createElement("section");
        sourceBox.className = "sources";
        const heading = document.createElement("strong");
        heading.className = "sources-heading";
        heading.textContent = "Sources";
        sourceBox.appendChild(heading);
        sources.forEach(source => {
            const item = document.createElement("div");
            item.className = "source-item";
            const prefix = source.type === "structured_data" ? "[SQL] " : "[KB] ";
            let label = prefix + (source.source || "Customer Knowledge Base");
            if (source.page !== null && source.page !== undefined)
                label += " · Page " + (Number(source.page) + 1);
            item.textContent = "• " + label;
            sourceBox.appendChild(item);
        });
        content.appendChild(sourceBox);
    }
    if (timing) {
        const timingDiv = document.createElement("div");
        timingDiv.className = "meta-timing";
        timingDiv.textContent = timing;
        content.appendChild(timingDiv);
    }
    row.appendChild(avatar);
    row.appendChild(content);
    const messages = document.getElementById("messages");
    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
    return row;
}

function showLoading() {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.setAttribute("aria-label", "Assistant is responding");
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "AI";
    avatar.setAttribute("aria-hidden", "true");
    const content = document.createElement("div");
    content.className = "message-content";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const loading = document.createElement("div");
    loading.className = "loading";
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        loading.appendChild(dot);
    }
    bubble.appendChild(loading);
    content.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(content);
    document.getElementById("messages").appendChild(row);
    row.scrollIntoView({ behavior: "smooth", block: "end" });
    return row;
}

async function verifyToken(tokenInput) {
    const token = (tokenInput || "").trim().replace(/^Bearer\s+/i, "");
    const statusBox = document.getElementById("authStatus");
    const roleBadge = document.getElementById("roleBadge");
    const clearBtn = document.getElementById("clearTokenBtn");
    const verifyBtn = document.getElementById("verifyTokenBtn");

    if (!token) {
        statusBox.className = "auth-status error";
        statusBox.textContent = "Please paste a Phase I JWT token first.";
        return;
    }

    verifyBtn.disabled = true;
    statusBox.className = "auth-status";
    statusBox.textContent = "Verifying token with Phase I...";

    try {
        const response = await fetch("/auth/test", {
            method: "GET",
            headers: { "Authorization": "Bearer " + token }
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Authentication failed");

        authState.token = token;
        authState.userId = data.user_id;
        authState.role = String(data.role || "").toLowerCase();
        activeSessionId = null;
        sessionStorage.setItem("rag_demo_jwt", token);

        roleBadge.textContent = authState.role.toUpperCase();
        statusBox.className = "auth-status verified";
        statusBox.textContent = "Verified · " + authState.role.toUpperCase() + " (" + authState.userId + ")";
        clearBtn.hidden = false;
        renderRoleUI(authState.role);
        document.getElementById("messages").replaceChildren(welcomePanel);
    } catch (error) {
        clearAuth(false);
        statusBox.className = "auth-status error";
        statusBox.textContent = error.message;
    } finally {
        verifyBtn.disabled = false;
    }
}

function clearAuth(resetMessage = true) {
    authState.token = "";
    authState.userId = null;
    authState.role = null;
    activeSessionId = null;
    sessionStorage.removeItem("rag_demo_jwt");
    document.getElementById("authTokenInput").value = "";
    document.getElementById("orderIdInput").value = "";
    document.getElementById("roleBadge").textContent = "Guest";
    document.getElementById("clearTokenBtn").hidden = true;
    const statusBox = document.getElementById("authStatus");
    if (resetMessage) {
        statusBox.className = "auth-status";
        statusBox.textContent = "Guest mode · Public Customer KB (/chat)";
    }
    renderRoleUI("guest");
    document.getElementById("messages").replaceChildren(welcomePanel);
}

function buildRequestForActiveMode(question) {
    if (!authState.token || !authState.role) {
        return {
            url: "/chat",
            headers: { "Content-Type": "application/json" },
            body: { question: question }
        };
    }

    const cfg = ROLE_CONFIG[authState.role];
    const body = { question: question, session_id: activeSessionId };
    const orderId = document.getElementById("orderIdInput").value.trim();

    if (authState.role === "customer") body.include_own_orders = true;
    if (authState.role === "seller") body.include_own_products = true;
    if (authState.role === "support") {
        body.include_support_tickets = true;
        if (orderId) body.order_id = orderId;
    }
    if (authState.role === "operations") {
        body.include_inventory_overview = true;
        if (orderId) body.order_id = orderId;
    }
    if (authState.role === "admin") body.include_system_analytics = true;

    return {
        url: cfg.endpoint,
        headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + authState.token
        },
        body: body
    };
}

async function sendMessage() {
    if (isLoading) return;
    const input = document.getElementById("question");
    const button = document.getElementById("sendButton");
    const question = input.value.trim();
    if (!question) return;
    if (welcomePanel.isConnected) welcomePanel.remove();
    addMessage(question, "user");
    input.value = "";
    input.style.height = "auto";
    updateCharCount();
    isLoading = true;
    button.disabled = true;
    button.setAttribute("aria-label", "Waiting for a response");
    const loadingRow = showLoading();
    try {
        const req = buildRequestForActiveMode(question);
        const response = await fetch(req.url, {
            method: "POST",
            headers: req.headers,
            body: JSON.stringify(req.body)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Request failed");
        if (data.session_id) activeSessionId = data.session_id;
        loadingRow.remove();

        const timingParts = [];
        if (data.retrieval_time !== undefined) timingParts.push("Retrieval " + data.retrieval_time + "s");
        if (data.generation_time !== undefined) timingParts.push("LLM " + data.generation_time + "s");
        if (data.response_time !== undefined) timingParts.push("Total " + data.response_time + "s");
        if (data.session_id) timingParts.push("Session " + data.session_id.slice(0, 8));

        addMessage(data.answer, "assistant", data.sources || [], timingParts.join(" · "));
    } catch (error) {
        loadingRow.remove();
        addMessage("Sorry, I could not process your request. " + error.message, "assistant");
        console.error("Chat error:", error);
    } finally {
        isLoading = false;
        button.disabled = false;
        button.setAttribute("aria-label", "Send message");
        input.focus();
    }
}

function ask(question) {
    const input = document.getElementById("question");
    input.value = question;
    updateCharCount();
    sendMessage();
}

function updateCharCount() {
    const input = document.getElementById("question");
    document.getElementById("charCount").textContent = input.value.length + " / 2000";
}

document.getElementById("verifyTokenBtn").addEventListener("click", function() {
    verifyToken(document.getElementById("authTokenInput").value);
});

document.getElementById("authTokenInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        verifyToken(this.value);
    }
});

document.getElementById("clearTokenBtn").addEventListener("click", function() {
    clearAuth(true);
});

document.getElementById("chatForm").addEventListener("submit", function(event) {
    event.preventDefault();
    sendMessage();
});

document.getElementById("question").addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 130) + "px";
    updateCharCount();
});

document.getElementById("question").addEventListener("keydown", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

document.getElementById("newChatButton").addEventListener("click", function() {
    if (isLoading) return;
    activeSessionId = null;
    const messages = document.getElementById("messages");
    messages.replaceChildren(welcomePanel);
    document.getElementById("question").focus();
});

renderRoleUI("guest");
const savedToken = sessionStorage.getItem("rag_demo_jwt");
if (savedToken) {
    document.getElementById("authTokenInput").value = savedToken;
    verifyToken(savedToken);
}
</script>
</body>
</html>
"""

    return HTMLResponse(content=html)
