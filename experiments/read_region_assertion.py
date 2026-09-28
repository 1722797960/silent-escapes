"""Read com.example.region_assertion from a (possibly edited) C2PA file.
Runs under the integrity-clash venv. Prints JSON:
  {manifest_valid, region_assertion_present, region: {...}|null}
"""
import json
import sys

import c2pa


def main():
    path = sys.argv[1]
    out = {"manifest_valid": False, "region_assertion_present": False,
           "region": None}
    try:
        r = c2pa.Reader(path)
        out["manifest_valid"] = r.get_validation_state() == "Valid"
        d = json.loads(r.detailed_json())
        man = list(d["manifests"].values())[0]
        store = man.get("assertion_store", {})
        ra = store.get("com.example.region_assertion")
        if ra:
            out["region_assertion_present"] = True
            out["region"] = ra
    except Exception as e:  # noqa: BLE001 — audit must never crash on one asset
        out["error"] = str(e)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
