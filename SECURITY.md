# Security and trust boundary

## Intended execution model

This repository contains local, command-line research programs. It does not run
a network service, accept remote requests, manage user accounts, or execute
untrusted multi-tenant jobs. Command-line file and directory arguments are
chosen by the researcher running the experiment and are therefore inside the
local trust boundary.

## Credentials

No signing private key is committed. `tools/generate_demo_credentials.py`
creates short-lived, self-signed demonstration credentials in `certs/`, and the
repository `.gitignore` excludes all generated PEM/key files. These credentials
must never be used as a production identity.

## Reproducible randomness

Python and NumPy pseudo-random generators are intentionally used for seeded
image sampling, augmentation, and crop placement. They are not used to produce
cryptographic keys or authentication tokens. Demo signing keys use the
`cryptography` library's elliptic-curve key generator.

## Local paths

Several experiment scripts accept input and output paths from the command line.
This is required for reproducibility across machines. They should only be run
with paths supplied by the local researcher. Do not expose these commands
directly as a web service without adding path confinement and request isolation.

## Dependency downloads

`setup_env.sh` downloads VideoSeal from its official GitHub repository. Set
`VIDEOSEAL_REF` to a reviewed commit SHA when preparing a frozen environment.
The historical environment retained its package freeze but not the originating
VideoSeal commit, so the default remains `main` and is explicitly not claimed to
be bit-for-bit reproducible.

## Reporting a vulnerability

Please open a GitHub issue that describes the affected file, the execution
conditions, and a minimal reproduction. Do not include real credentials or
private datasets in a public issue.
