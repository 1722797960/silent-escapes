#!/usr/bin/env bash
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${C2PA_PY:-$REPO_ROOT/venvs/c2pa/bin/python}
echo "=== sign ==="
$PY scripts/sign_images.py outputs_500_wam outputs_500_wam_signed_ai --manifest-template manifests/manifest_ai.json --cert certs/ec_chain.pem --key certs/ec_key.pem 2>&1 | tail -1
$PY scripts/sign_images.py outputs_500_wam outputs_500_wam_signed_human --manifest-template manifests/manifest_human_edited.json --cert certs/ec_chain.pem --key certs/ec_key.pem 2>&1 | tail -1
for att in jpeg_q80 crop10 social; do
  $PY scripts/sign_images.py outputs_500_wam_attack_$att outputs_500_wam_signed_attack_${att}_human --manifest-template manifests/manifest_human_edited.json --cert certs/ec_chain.pem --key certs/ec_key.pem 2>&1 | tail -1
done
echo "=== verify ==="
$PY scripts/verify_images.py outputs_500_wam_signed_ai --output-json results_500/verification_wam_ai.json 2>&1 | tail -1
$PY scripts/verify_images.py outputs_500_wam_signed_human --output-json results_500/verification_wam_human.json 2>&1 | tail -1
for att in jpeg_q80 crop10 social; do
  $PY scripts/verify_images.py outputs_500_wam_signed_attack_${att}_human --output-json results_500/verification_wam_attack_${att}.json 2>&1 | tail -1
done
echo "=== detect ==="
META=outputs_500_wam/watermark_metadata.json
run_detect () {
  $PY scripts/wam_detect.py --watermarked-dir "$1" --metadata-path "$META" --output-json "$2" --ckpt ckpts/wam_mit.pth --params-json ckpts/params.json --device cuda 2>&1 | tail -2
}
run_detect outputs_500_wam results_500/wm_not_signed.json
run_detect outputs_500_wam_signed_ai results_500/wm_signed_ai.json
run_detect outputs_500_wam_signed_human results_500/wm_signed_human.json
for att in jpeg_q80 crop10 social; do
  run_detect outputs_500_wam_signed_attack_${att}_human results_500/wm_attack_${att}_human.json
done
echo "=== audit ==="
$PY scripts/audit_protocol.py --wm-not-signed results_500/wm_not_signed.json --wm-signed-ai results_500/wm_signed_ai.json --wm-signed-human results_500/wm_signed_human.json --verif-ai results_500/verification_wam_ai.json --verif-human results_500/verification_wam_human.json --output-prefix results_500/audit_wam --threshold 0.75 2>&1 | tail -12
echo "=== ALL DONE ==="
