# Contributing
- Text only: code, configs, workflow JSON and docs. Never commit weights, images, video, datasets or outputs (enforced by `.gitignore`).
- Never commit secrets, IPs, hostnames, usernames, emails or absolute home paths. Machine-specific settings go in `config.local.yaml` / `.env` (gitignored); update `config.example.yaml` with placeholders instead.
- Run `scripts/check_secrets.sh` on your staged changes before every commit.
- Log each experiment in `docs/FINDINGS.md`, using one dated entry with goal, config, measured results, verdict and next step.
- Follow RESPONSIBLE_USE.md.
