import httpx
import os
import re
import sys
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import StringIO
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        namespace = {"__name__": "__main__"}
        exec(code, namespace)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception as e:
        extracted = traceback.extract_tb(e.__traceback__)
        lines = ["Traceback (most recent call last):"]
        for frame in extracted:
            if frame.filename == "<string>":
                lines.append(f'  File "<string>", line {frame.lineno}, in {frame.name}')
                if frame.line:
                    lines.append(f"    {frame.line}")
        lines.append(f"{type(e).__name__}: {e}")
        output = "\n".join(lines) + "\n"
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    token = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
    prompt = f"Analyze this Python code and its error traceback. Identify the exact line number(s) where the error occurred. Return ONLY a JSON object with key error_lines containing an array of integers.\n\nCODE:\n{code}\n\nTRACEBACK:\n{tb}"
    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                "https://aipipe.org/openrouter/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            result = ErrorAnalysis.model_validate_json(content)
            return result.error_lines
    except Exception:
        matches = re.findall(r'File\s+"[^"]*",\s+line\s+(\d+)', tb)
        seen = set()
        out = []
        for m in matches:
            n = int(m)
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    res = execute_python_code(request.code)
    if res["success"]:
        return CodeResponse(error=[], result=res["output"])
    lines = analyze_error_with_ai(request.code, res["output"])
    return CodeResponse(error=lines, result=res["output"])


@app.get("/")
def root():
    return {"status": "ok"}
