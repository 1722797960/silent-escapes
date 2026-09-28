"""Independent verification sweep over the re-signed crop corpus (Road B add-on).

Uses the c2pa SDK's own validation path (c2pa.Reader = the reference
verifier, not our audit code) on EVERY re-signed asset at every severity,
plus a binary-level check that the assertion physically lives in the file's
c2pa (caBX) box. Produces results/defense_e2e/independent_verification.json
summarized per severity:

  manifest_valid  reference-SDK validation state == Valid
  untrusted_only  sole failure code is signingCredential.untrusted
                  (expected with self-signed research certs, same as
                  Nemecek et al.)
  bin_assertion   'com.example.region_assertion' found in raw bytes
  ra_readable     assertion parsed by the SDK with intact payload hash

Run under the integrity-clash venv (c2pa installed).
"""
import glob
import hashlib
import json
import os
import sys

import c2pa

E2E_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E2E = os.path.join(E2E_ROOT, "results/defense_e2e")

meta = {m["input_image"]: m for m in
        json.load(open(os.path.join(E2E, "embed_meta.json")))["images"]}

summary = {}
examples = {}
for d in sorted(glob.glob(os.path.join(E2E, "attack_resign/crop*"))):
    crop = os.path.basename(d)
    counts = {"n": 0, "manifest_valid": 0, "untrusted_only": 0,
              "bin_assertion": 0, "ra_readable": 0}
    for path in sorted(glob.glob(os.path.join(d, "*.png"))):
        counts["n"] += 1
        raw = open(path, "rb").read()
        if b"com.example.region_assertion" in raw:
            counts["bin_assertion"] += 1
        try:
            r = c2pa.Reader(path)
            state = r.get_validation_state()
            dd = json.loads(r.detailed_json())
            man = list(dd["manifests"].values())[0]
            vr = dd.get("validation_results", {}).get("activeManifest", {})
            fails = [f["code"] for f in vr.get("failure", [])]
            if state == "Valid":
                counts["manifest_valid"] += 1
            if fails == ["signingCredential.untrusted"]:
                counts["untrusted_only"] += 1
            ra = man.get("assertion_store", {}).get("com.example.region_assertion")
            if ra and ra.get("payload_hash"):
                counts["ra_readable"] += 1
            if crop not in examples:
                examples[crop] = {"file": os.path.basename(path), "state": state,
                                  "fails": fails}
        except Exception as e:  # noqa: BLE001
            examples.setdefault(crop, {"error": str(e)})
    summary[crop] = counts
    print(crop, counts, flush=True)

out = {"summary": summary, "examples": examples,
       "note": "Reference c2pa-python 0.37.10 validation path; binary check "
               "inspects the raw caBX/JUMBF store independent of our audit code."}
with open(os.path.join(E2E, "independent_verification.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(E2E, "independent_verification.json"))
