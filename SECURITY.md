# Security policy

## Supported versions

This is a small, single-maintainer project; only the latest `main` is supported.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Instead, report privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):
go to the repository's **Security** tab → **Report a vulnerability**.

Please include enough detail to reproduce the issue. You can expect an initial
response within a couple of weeks; fixes are best-effort given this is a hobby
project.

## Scope notes

These are local command-line tools. Worth keeping in mind:

- `metadata_ui.py` runs a local HTTP server bound to `127.0.0.1` only and, by
  design, lets you edit any `.safetensors` path you point it at — don't expose it
  to untrusted networks.
- The tools parse untrusted files (safetensors headers, images). Reports about
  crashes or unsafe parsing of malformed inputs are in scope and appreciated.
