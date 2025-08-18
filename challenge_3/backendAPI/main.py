from fastapi import FastAPI
from pydantic import BaseModel
from backendAPI.utils import random_case
from backendAPI.rag_agent import rag_response
from backendAPI.chaos_lawyer import chaos_response
from backendAPI.judge import judge_decision

app = FastAPI(title="Courtroom Clash API")

class CaseRequest(BaseModel):
    case_text: str = None

class JudgeRequest(BaseModel):
    winner: str = None
    new_evidence: str = None

@app.post("/generate_case")
def generate_case():
    case = random_case()
    return {"case": case}

@app.post("/debate")
def debate(case_req: CaseRequest):
    case = case_req.case_text
    rag = rag_response(case)
    chaos = chaos_response(case)
    return {"case": case, "rag_lawyer": rag, "chaos_lawyer": chaos, "judge_decision": "Pending"}

@app.post("/judge_decision")
def judge(judge_req: JudgeRequest):
    result = judge_decision(judge_req)
    return result
