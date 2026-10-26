import time, uuid, logging
from concurrent import futures
import grpc
import lab3_pb2
import lab3_pb2_grpc
from lab2_kernel import gen_inputs, compute, run_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class ComputeService(lab3_pb2_grpc.ComputeServiceServicer):
    def Compute(self, request, context):
        rid = request.request_id or str(uuid.uuid4())
        try:
            inputs = gen_inputs(request.n, seed=11)
            t0 = time.perf_counter()
            blocks = max(1, min(request.blocks, request.n))
            block_list = [(i * request.n // blocks, (i + 1) * request.n // blocks) for i in range(blocks)]
            block_list = [(a, b) for a, b in block_list if a < b]
            def run_stage(tasks): return [run_task(inputs, *task) for task in tasks]
            result = compute(inputs, run_stage, block_list)
            server_ms = (time.perf_counter() - t0) * 1000
            import numpy as np
            arr = np.asarray(result)
            logging.info("requestId=%s serverComputeMs=%.3f", rid, server_ms)
            return lab3_pb2.ComputeReply(request_id=rid, n=request.n,
                result_shape="x".join(map(str, arr.shape)), result_sum=float(np.sum(arr)),
                result_norm=float(np.linalg.norm(arr)), server_compute_ms=server_ms)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL); context.set_details(str(exc)); raise

    def ComputeY3Block(self, request, context):
        rid = request.request_id or str(uuid.uuid4())
        try:
            if request.n < 2 or request.n > 400 or request.k < 1 or request.k >= request.n:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT); context.set_details("Require 1 <= k < n <= 400"); return lab3_pb2.Y3BlockReply()
            inputs = gen_inputs(request.n, seed=11)
            def run_stage(tasks): return [run_task(inputs, *task) for task in tasks]
            t0 = time.perf_counter()
            run_stage([("D", 0, request.n)])
            run_stage([("Y3", request.k, request.n)])
            server_ms = (time.perf_counter() - t0) * 1000
            block = inputs["Y3"][request.k:request.n]
            logging.info("requestId=%s serverComputeMs=%.3f n=%s rows=%s:%s", rid, server_ms, request.n, request.k, request.n)
            return lab3_pb2.Y3BlockReply(request_id=rid, n=request.n, k=request.k,
                rows=block.shape[0], cols=block.shape[1], values=block.ravel().tolist(), server_compute_ms=server_ms)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL); context.set_details(str(exc)); raise

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    lab3_pb2_grpc.add_ComputeServiceServicer_to_server(ComputeService(), server)
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("gRPC listening on 0.0.0.0:50051")
    server.wait_for_termination()

if __name__ == "__main__": serve()
