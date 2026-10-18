import argparse, time, uuid, statistics, json
import grpc
import lab3_pb2, lab3_pb2_grpc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="127.0.0.1:50051")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--timeout", type=float, default=10)
    a = p.parse_args()
    channel = grpc.insecure_channel(a.target)
    stub = lab3_pb2_grpc.ComputeServiceStub(channel)
    vals = []
    for i in range(3 + a.runs):
        rid = str(uuid.uuid4())
        req = lab3_pb2.ComputeRequest(request_id=rid, n=a.n, blocks=a.blocks)
        t0 = time.perf_counter()
        try:
            res = stub.Compute(req, timeout=a.timeout)
            e2e = (time.perf_counter()-t0)*1000
            if i >= 3:
                vals.append(e2e)
                print(json.dumps({"requestId": res.request_id, "clientE2eMs": round(e2e,3),
                    "serverComputeMs": round(res.server_compute_ms,3),
                    "overheadApproxMs": round(e2e-res.server_compute_ms,3),
                    "result_shape": res.result_shape}))
        except grpc.RpcError as e:
            print("gRPC error:", e.code(), e.details())
    if vals: print("gRPC mean client E2E ms:", round(statistics.mean(vals),3))
    channel.close()
if __name__ == "__main__": main()
