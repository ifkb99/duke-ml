# Auto-deploy

Every push to `main` runs `.github/workflows/deploy.yml`:

1. **test**: `npm ci`, build, `npm test`. A failure stops the deploy.
2. **image**: builds the Dockerfile and pushes `ghcr.io/ifkb99/duke-ml:<sha>` and `:latest`.
   The network files come from Git LFS (`packages/web/public/models`).
3. **deploy**: SSHes to the server through the Cloudflare Tunnel with a key that can
   only run `deploy.sh`, which pulls that image and restarts the container on port 3001.
   Then it checks https://duke.nyaaa.fun/ answers.

The deploy job is skipped until the `DEPLOY_HOST` variable exists, so steps 1–2 run
on their own until the setup below is done.

## One-time setup

### 0. Check the server

```sh
uname -m                 # x86_64 expected; anything else: the workflow's `platforms` must change
docker compose version   # Compose v2
groups | grep docker     # the deploy user must be able to run docker
```

### 1. Deploy files

```sh
mkdir -p ~/duke-deploy && cd ~/duke-deploy
curl -fsSLO https://raw.githubusercontent.com/ifkb99/duke-ml/main/deploy/compose.yml
curl -fsSLO https://raw.githubusercontent.com/ifkb99/duke-ml/main/deploy/deploy.sh
chmod +x deploy.sh
```

### 2. Deploy key (generated on the server, locked to `deploy.sh`)

```sh
ssh-keygen -t ed25519 -N '' -C github-actions-duke -f ~/duke-deploy/actions_key
echo "command=\"$HOME/duke-deploy/deploy.sh\",restrict $(cat ~/duke-deploy/actions_key.pub)" >> ~/.ssh/authorized_keys
cat ~/duke-deploy/actions_key                  # -> GitHub secret DEPLOY_SSH_KEY (whole thing, BEGIN to END)
echo "ssh-duke.nyaaa.fun $(cut -d' ' -f1,2 /etc/ssh/ssh_host_ed25519_key.pub)"   # -> secret DEPLOY_KNOWN_HOSTS
rm ~/duke-deploy/actions_key ~/duke-deploy/actions_key.pub   # once the secret is saved
```

`restrict` turns off forwarding and terminals; `command=` means the key runs
`deploy.sh` whatever the client asks for.

### 3. Cloudflare (Zero Trust dashboard)

1. **SSH through the tunnel.** Networks → Tunnels → your tunnel → Public hostnames → Add:
   subdomain `ssh-duke`, domain `nyaaa.fun`, type **SSH**, URL `localhost:22`.
   If cloudflared runs in Docker, `localhost` is the container: use the host's address
   (e.g. `172.17.0.1:22`) instead. For a tunnel configured in `config.yml`, add
   `- hostname: ssh-duke.nyaaa.fun` / `service: ssh://localhost:22` above the catch-all
   rule and restart cloudflared.
2. **Service token.** Access → Service credentials → Service Tokens → Create, name it
   `github-deploy`. Copy the Client ID and Client Secret (the secret is shown once).
3. **Lock the hostname to that token.** Access → Applications → Add → Self-hosted,
   domain `ssh-duke.nyaaa.fun`; one policy with action **Service Auth**, include
   **Service Token → github-deploy**. Without this, anyone with cloudflared can reach
   sshd on that hostname (keys still protect it, but this keeps it closed).

### 4. GitHub (repo → Settings → Secrets and variables → Actions)

| Secrets                   | Value                           |
|---------------------------|---------------------------------|
| `DEPLOY_SSH_KEY`          | private key from step 2         |
| `DEPLOY_KNOWN_HOSTS`      | host key line from step 2       |
| `CF_ACCESS_CLIENT_ID`     | service token Client ID         |
| `CF_ACCESS_CLIENT_SECRET` | service token Client Secret     |

| Variables     | Value                         |
|---------------|-------------------------------|
| `DEPLOY_HOST` | `ssh-duke.nyaaa.fun`          |
| `DEPLOY_USER` | the server user from step 2   |

### 5. First deploy

1. Actions → Deploy → Run workflow (or push to main). Wait for **image** to finish.
2. Make the image public so the server can pull it without logging in: github.com/ifkb99
   → Packages → duke-ml → Package settings → Change visibility → Public.
3. Stop the old hand-built container (it holds port 3001): in its checkout,
   `docker compose down`.
4. Re-run the **deploy** job.

## Day to day

- **New model**: copy the export (`duke-zero.json`, `.q4.onnx`, `.q4.bin`) into
  `packages/web/public/models`, commit, push to main. Git LFS stores them.
- **Roll back**: Actions → the Deploy run of the commit you want → Re-run the **deploy**
  job. Or on the server: `cd ~/duke-deploy && SSH_ORIGINAL_COMMAND="deploy <sha>" ./deploy.sh`.
- **By hand**: `cd ~/duke-deploy && SSH_ORIGINAL_COMMAND="deploy latest" ./deploy.sh`.
