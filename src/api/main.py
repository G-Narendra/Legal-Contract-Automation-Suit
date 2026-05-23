"""
FastAPI Backend for Legal Contract Automation Suite.

Endpoints:
- POST /api/v1/contract/analyze - Analyze a contract
- POST /api/v1/contract/draft - Draft a new contract
- POST /api/v1/contract/review - Full review with risk assessment
- POST /api/v1/research - Legal research
- GET /api/v1/contracts - List contracts
- GET /api/v1/health - Health check
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import uvicorn

app = FastAPI(
    title="Legal Contract Automation Suite API",
    version="1.0.0",
    description="AI-powered legal contract automation for UAE law firms",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "subsystems": {
            "analysis": "operational",
            "research": "operational",
            "drafting": "operational",
            "risk_compliance": "operational",
            "lifecycle": "operational",
        },
    }


@app.post("/api/v1/contract/analyze")
async def analyze_contract(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
):
    """Analyze an uploaded contract document."""
    if file.filename and not file.filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(400, "Only PDF, DOCX, and TXT files supported")

    content = await file.read()
    file_type = file.filename.split(".")[-1] if file.filename else "txt"

    from src.core.contract_processor import ContractProcessor
    processor = ContractProcessor()

    result = processor.process_contract(
        contract_file=content,
        file_type=file_type,
        task="analyze",
        user_context={"user": user_id},
    )

    return result


@app.post("/api/v1/contract/draft")
async def draft_contract(
    contract_type: str = Form(...),
    language: str = Form("english"),
    params: str = Form("{}"),
    user_id: str = Form("anonymous"),
):
    """Draft a new contract from template parameters."""
    import json
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in params")

    from src.core.contract_processor import ContractProcessor, ContractType
    processor = ContractProcessor()

    # Validate contract type
    try:
        ctype = ContractType(contract_type)
    except ValueError:
        raise HTTPException(400, f"Unknown contract type: {contract_type}")

    result = processor.process_contract(
        contract_file=b"",
        file_type="txt",
        task="draft",
        user_context={"user": user_id},
        params={"contract_type": ctype, **params_dict, "language": language},
    )

    return result


@app.post("/api/v1/contract/review")
async def review_contract(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
):
    """Full contract review with risk assessment."""
    if file.filename and not file.filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(400, "Only PDF, DOCX, and TXT files supported")

    content = await file.read()
    file_type = file.filename.split(".")[-1] if file.filename else "txt"

    from src.core.contract_processor import ContractProcessor
    processor = ContractProcessor()

    result = processor.process_contract(
        contract_file=content,
        file_type=file_type,
        task="review",
        user_context={"user": user_id},
    )

    return result


@app.post("/api/v1/research")
async def legal_research(
    query: str = Form(...),
    context: str = Form(""),
):
    """Conduct legal research."""
    from src.subsystems.legal_research import LegalResearchSystem
    research = LegalResearchSystem()

    result = research.search(query=query, contract_context=context)

    return result


@app.get("/api/v1/contracts")
async def list_contracts(
    status: str = "",
    contract_type: str = "",
):
    """List registered contracts."""
    from src.subsystems.lifecycle_management import LifecycleManagementSystem
    lifecycle = LifecycleManagementSystem()

    contracts = lifecycle.list_contracts(status=status, contract_type=contract_type)

    return {"contracts": contracts, "total": len(contracts)}


@app.get("/api/v1/contracts/{contract_id}")
async def get_contract(contract_id: str):
    """Get contract details."""
    from src.subsystems.lifecycle_management import LifecycleManagementSystem
    lifecycle = LifecycleManagementSystem()

    contract = lifecycle.get_contract(contract_id)
    if not contract:
        raise HTTPException(404, "Contract not found")

    obligations = lifecycle.get_obligations(contract_id)
    alerts = lifecycle.get_alerts(contract_id)

    return {"contract": contract, "obligations": obligations, "alerts": alerts}


@app.get("/api/v1/dashboard")
async def dashboard():
    """Get lifecycle dashboard statistics."""
    from src.subsystems.lifecycle_management import LifecycleManagementSystem
    lifecycle = LifecycleManagementSystem()

    return lifecycle.get_dashboard_stats()
