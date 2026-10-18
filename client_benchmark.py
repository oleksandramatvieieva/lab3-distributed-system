import argparse, json, time, uuid, statistics
import requests
from lab2_kernel import gen_inputs, compute

def local_once(n, blocks):
    data = gen_inputs(n, seed=11)
    t0 = time.perf_counter()
    compute(*data, run_stage="all", blocks=blocks)
    return (time.perf_counter() - t0) * 1000

def remote_once(url, n, blocks, timeout):
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    r = requests.post(url.rstrip("/") + "/compute",
                      json={"n": n, "blocks": blocks, "run_stage": "all", "requestId": rid},
                      timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    e2e = (time.perf_counter() - t0) * 1000
    payload["clientE2eMs"] = round(e2e, 3)
    payload["overheadApproxMs"] = round(e2e - payload["serverComputeMs"], 3)
    return payload

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Base URL, e.g. https://your-service.onrender.com")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--timeout", type=float, default=15)
    a = p.parse_args()
    print("Warm-up (3 requests)...")
    for _ in range(3):
        try: remote_once(a.url, a.n, a.blocks, a.timeout)
        except Exception as e: print("Warm-up warning:", repr(e))
    rows = []
    for i in range(a.runs):
        try:
            row = remote_once(a.url, a.n, a.blocks, a.timeout)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        except requests.Timeout:
            print(json.dumps({"run": i+1, "error": "timeout"}))
        except Exception as e:
            print(json.dumps({"run": i+1, "error": repr(e)}))
    if rows:
        e2e = [x["clientE2eMs"] for x in rows]
        srv = [x["serverComputeMs"] for x in rows]
        ov = [x["overheadApproxMs"] for x in rows]
        print("\nSUMMARY (measured, milliseconds)")
        print(json.dumps({"successfulRuns": len(rows), "e2eMeanMs": round(statistics.mean(e2e),3),
                          "serverMeanMs": round(statistics.mean(srv),3),
                          "overheadMeanMs": round(statistics.mean(ov),3)}, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
