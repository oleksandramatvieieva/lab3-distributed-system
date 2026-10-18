import os, time, uuid, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from lab2_kernel import gen_inputs, compute, run_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lab3")
app = FastAPI(title="Lab 3 Distributed Computation API", version="1.0")

class ComputeRequest(BaseModel):
    n: int = Field(default=80, ge=2, le=400)
    blocks: int = Field(default=1, ge=1, le=16)
    run_stage: str = "all"
    requestId: str | None = None

@app.get("/")
def root():
    return {"service": "Lab3 REST API", "status": "ok", "endpoints": ["/health", "/compute", "/benchmark"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/compute")
def compute_endpoint(req: ComputeRequest):
    rid = req.requestId or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        inputs = gen_inputs(req.n, seed=11)
        server_start = time.perf_counter()
        block_list = [(i * req.n // req.blocks, (i + 1) * req.n // req.blocks) for i in range(req.blocks)]
        block_list = [(a, b) for a, b in block_list if a < b]
        def run_stage(tasks):
            return [run_task(inputs, *task) for task in tasks]
        result = compute(inputs, run_stage, block_list)
        server_ms = (time.perf_counter() - server_start) * 1000
        result_summary = summarize(result)
        total_ms = (time.perf_counter() - start) * 1000
        log.info("requestId=%s serverComputeMs=%.3f n=%s blocks=%s", rid, server_ms, req.n, req.blocks)
        return {"requestId": rid, "n": req.n, "blocks": req.blocks, "run_stage": req.run_stage,
                "result": result_summary, "serverComputeMs": round(server_ms, 3),
                "serverTotalMs": round(total_ms, 3)}
    except Exception as e:
        log.exception("requestId=%s failed", rid)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

def summarize(obj):
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return {"shape": list(obj.shape), "sum": float(np.sum(obj)), "norm": float(np.linalg.norm(obj))}
        if isinstance(obj, (list, tuple)):
            return [summarize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): summarize(v) for k, v in obj.items()}
        if hasattr(obj, "shape"):
            return {"shape": list(obj.shape)}
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return repr(obj)
    except Exception:
        return repr(obj)
