# Lab 3 — Distributed client-server system (variant 11)

This project reuses `lab2_kernel.py` from Lab 2 and provides:
- REST/JSON API (`main.py`) for deployment on Render;
- gRPC/Protobuf server/client (`grpc_server.py`, `grpc_client.py`);
- REST measurement client (`client_benchmark.py`);
- request IDs, server compute time, client end-to-end time, approximate overhead.

## Deploy REST on Render
Render Web Services expose one public HTTP port. This configuration deploys the REST API.
Build command: `pip install -r requirements.txt && python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. lab3.proto`
Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
Health check: `/health`

## REST test
`https://YOUR-SERVICE.onrender.com/health`
`python client_benchmark.py --url https://YOUR-SERVICE.onrender.com --n 80 --blocks 1 --runs 10`

## gRPC local test
Generate stubs:
`python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. lab3.proto`
Terminal 1: `python grpc_server.py`
Terminal 2: `python grpc_client.py --target 127.0.0.1:50051 --n 80 --runs 10`

## Important limitation
Render's public web service forwards public traffic to one HTTP port. The included gRPC server is ready for a host that exposes a public gRPC/TCP port, but this free Render web-service configuration does not expose port 50051 publicly. Do not report remote gRPC measurements unless the client really reaches a gRPC server on a separate host. Record real measurements only.
