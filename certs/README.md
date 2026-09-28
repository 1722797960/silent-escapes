# Disposable demo credentials

Run the following command from the repository root:

```bash
python tools/generate_demo_credentials.py
```

It creates a self-signed demo CA plus a P-256 leaf signing key and certificate
chain at the paths expected by the experiment scripts. All generated PEM files
are ignored by Git. Never reuse these credentials outside this experiment.
