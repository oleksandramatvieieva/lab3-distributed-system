import argparse, time, uuid
import numpy as np
import requests
from lab2_kernel import gen_inputs, compute, run_task

def run_rest(base_url, n, k, blocks):
    rid = str(uuid.uuid4())
    g = gen_inputs(n, seed=11)
    local_blocks = [(0, k)]
    def stage(tasks): return [run_task(g, *task) for task in tasks]
    t0 = time.perf_counter()
    stage([(kind, 0, n) for kind in ("y1", "y2", "D")])
    stage([("Y3", 0, k)])
    resp = requests.post(base_url.rstrip("/") + "/compute_y3_block",
                         json={"n": n, "k": k, "requestId": rid}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    g["Y3"][k:n] = np.asarray(data["y3_block"], dtype=float)
    work_blocks = [(i*n//blocks, (i+1)*n//blocks) for i in range(blocks)]
    work_blocks = [(a,b) for a,b in work_blocks if a < b]
    parts = stage([("w1u", a, b) for a,b in work_blocks])
    g["v"][:] = np.sum(parts, axis=0)
    s1 = float(g["y2"] @ g["y2"])
    stage([("w2", a, b) for a,b in work_blocks])
    s3 = float(g["y1"] @ g["w2"])
    stage([("x", a, b, (s1, s3)) for a,b in work_blocks])
    elapsed = (time.perf_counter()-t0)*1000
    ref = gen_inputs(n, seed=11)
    def refstage(tasks): return [run_task(ref, *task) for task in tasks]
    ref_blocks = [(i*n//blocks, (i+1)*n//blocks) for i in range(blocks)]
    ref_blocks = [(a,b) for a,b in ref_blocks if a < b]
    reference = compute(ref, refstage, ref_blocks)
    ok = bool(np.allclose(g["x"], reference, rtol=1e-10, atol=1e-8))
    print(f"H3 REST n={n} k={k} blocks={blocks}")
    print(f"requestId={rid}")
    print(f"T_hybrid_ms={elapsed:.3f}")
    print(f"T_client_local_part_ms=measured within T_hybrid")
    print(f"T_server_compute_ms={data['serverComputeMs']:.3f}")
    print(f"result_shape={list(g['x'].shape)} sum={np.sum(g['x']):.6f} norm={np.linalg.norm(g['x']):.6f}")
    print(f"validation_allclose={ok}")
    if not ok: raise RuntimeError("Hybrid result does not match local reference")

def run_grpc(target, n, k, blocks):
    import grpc, lab3_pb2, lab3_pb2_grpc
    rid = str(uuid.uuid4())
    g = gen_inputs(n, seed=11)
    local_blocks = [(0,k)]
    def stage(tasks): return [run_task(g,*task) for task in tasks]
    t0 = time.perf_counter()
    stage([(kind,0,n) for kind in ("y1","y2","D")])
    stage([("Y3",0,k)])
    channel = grpc.insecure_channel(target)
    stub = lab3_pb2_grpc.ComputeServiceStub(channel)
    res = stub.ComputeY3Block(lab3_pb2.Y3BlockRequest(request_id=rid,n=n,k=k), timeout=60)
    g["Y3"][k:n] = np.asarray(res.values,dtype=float).reshape(res.rows,res.cols)
    work_blocks=[(i*n//blocks,(i+1)*n//blocks) for i in range(blocks)]
    work_blocks=[(a,b) for a,b in work_blocks if a<b]
    parts=stage([("w1u",a,b) for a,b in work_blocks]); g["v"][:]=np.sum(parts,axis=0)
    s1=float(g["y2"]@g["y2"]); stage([("w2",a,b) for a,b in work_blocks]); s3=float(g["y1"]@g["w2"])
    stage([("x",a,b,(s1,s3)) for a,b in work_blocks])
    elapsed=(time.perf_counter()-t0)*1000
    ref=gen_inputs(n,seed=11)
    def refstage(tasks): return [run_task(ref,*task) for task in tasks]
    rb=[(i*n//blocks,(i+1)*n//blocks) for i in range(blocks)]; rb=[(a,b) for a,b in rb if a<b]
    reference=compute(ref,refstage,rb); ok=bool(np.allclose(g["x"],reference,rtol=1e-10,atol=1e-8))
    print(f"H3 gRPC n={n} k={k} blocks={blocks}")
    print(f"requestId={rid}")
    print(f"T_hybrid_ms={elapsed:.3f}")
    print(f"T_server_compute_ms={res.server_compute_ms:.3f}")
    print(f"result_shape={list(g['x'].shape)} sum={np.sum(g['x']):.6f} norm={np.linalg.norm(g['x']):.6f}")
    print(f"validation_allclose={ok}")
    channel.close()
    if not ok: raise RuntimeError("Hybrid result does not match local reference")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",choices=["rest","grpc"],required=True)
    p.add_argument("--n",type=int,default=20)
    p.add_argument("--k",type=int,default=None)
    p.add_argument("--blocks",type=int,default=1)
    p.add_argument("--url",default="https://lab3-distributed-system.onrender.com")
    p.add_argument("--target",default="127.0.0.1:50051")
    a=p.parse_args(); k=a.k if a.k is not None else a.n//2
    if not 1 <= k < a.n: p.error("k must satisfy 1 <= k < n")
    if a.protocol=="rest": run_rest(a.url,a.n,k,a.blocks)
    else: run_grpc(a.target,a.n,k,a.blocks)
