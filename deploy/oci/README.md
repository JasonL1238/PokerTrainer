# Oracle Cloud Always Free deployment

This is the free-first deployment path for PokerTrainer's private, completed-session study workflow. It runs one Streamlit/CV container behind Caddy and keeps SQLite, uploaded videos, timelines, logs, and backups on an attached block volume.

## Release gates

Do not use this host until all gates pass:

1. An Always Free Ampere A1 instance can be provisioned in the tenancy's home region.
2. `docker buildx build --platform linux/arm64 .` succeeds without changing or unpinning CV dependencies.
3. The application healthcheck passes on the A1 VM.
4. A short fixture and one representative session video finish within the configured one-hour timeout without exhausting memory.
5. Restart, backup, and restore drills pass.

If architecture compatibility, capacity, or runtime reliability fails, record the result and use the checked-in Fly configuration instead.

## Provision the host

1. Create one Ubuntu `VM.Standard.A1.Flex` instance using the Always Free allocation: 2 OCPUs and 12GB RAM.
2. Use a 50GB boot volume. Create and attach a separate Always Free block volume sized for retained videos, then mount it at `/srv/pokertrainer`.
3. Reserve a public IP and point the hostname in `POKERTRAINER_HOST` to it.
4. Allow inbound TCP 22 from the administrator's IP and TCP/UDP 80 and 443 from the internet. Do not expose port 8501.
5. Install Docker Engine plus the Compose plugin from Docker's supported Ubuntu repository.

Create the durable directory:

```bash
sudo mkdir -p /srv/pokertrainer/data
sudo chown -R 10001:10001 /srv/pokertrainer
```

## Configure and deploy

From the repository root on the VM:

```bash
cp deploy/oci/.env.example deploy/oci/.env
chmod 600 deploy/oci/.env
```

Set a long random `APP_PASSWORD`, the real hostname, and any model-provider keys in `.env`. Never commit `.env`.

```bash
docker compose -f deploy/oci/compose.yaml build
docker compose -f deploy/oci/compose.yaml up -d
docker compose -f deploy/oci/compose.yaml ps
docker compose -f deploy/oci/compose.yaml logs --tail=200 app caddy
```

Caddy obtains and renews TLS certificates automatically after DNS and ports are correct.

## Upgrade and roll back

Before every upgrade, copy `/srv/pokertrainer/data/poker_tracker.db` using SQLite's backup API or retain the latest file in `/srv/pokertrainer/data/backups`. Then:

```bash
git pull --ff-only
docker compose -f deploy/oci/compose.yaml build
docker compose -f deploy/oci/compose.yaml up -d
```

To roll back, check out the previous known-good revision, rebuild, and restore a compatible database backup only when required. Never overwrite the only database copy.

## Backup and recovery

- The app keeps the five newest consistent pre-import SQLite backups under `/srv/pokertrainer/data/backups`.
- Configure OCI block-volume backups and copy compact SQLite backups to an Always Free Object Storage bucket.
- Keep secrets outside the repository and in a separate password manager.
- Videos are not copied to Object Storage automatically. The block volume remains their system of record.

Recovery drill:

1. Stop the Compose project.
2. Preserve the failed database under a different filename.
3. Restore the newest verified backup to `/srv/pokertrainer/data/poker_tracker.db`.
4. Start Compose and verify authentication, session counts, `PRAGMA journal_mode`, and one completed hand.
5. Restart the VM during a test reconstruction and confirm the job becomes `failed` rather than remaining stuck.

Oracle may have no free A1 capacity and may reclaim idle Always Free instances. The deployment must therefore remain reproducible from this repository, `.env` secrets, the mounted data, and an off-instance database backup.
