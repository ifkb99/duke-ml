# Auto-deploy

Every push to `main` runs `.github/workflows/deploy.yml`:

1. **test**: `npm ci`, build, `npm test`. A failure stops the deploy.
2. **image**: builds the Dockerfile and pushes `ghcr.io/ifkb99/duke-ml:<sha>` and `:latest`.
   The network files come from Git LFS (`packages/web/public/models`).
3. **deploy**: SSHes to the server as `djneko@72.62.114.183` (host key pinned in the
   workflow) with the `Srv` secret and runs `~/duke-deploy/deploy.sh <sha>`, which pulls that
   image and restarts the container on port 3001. Then it checks
   https://duke.nyaaa.fun/ answers.

Until the setup below is done, the deploy job fails; the image still builds.

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

### 2. The Actions key (`Srv` secret)

The `Srv` repository secret holds the private key; its public half must be in
`~/.ssh/authorized_keys` of `djneko`.

If `Srv` is a key used only for deploys, lock it to the deploy script: prefix its
line in `authorized_keys` with

```
command="/home/djneko/duke-deploy/deploy.sh",restrict ssh-ed25519 AAAA... 
```

`restrict` turns off forwarding and terminals; `command=` makes the key run
`deploy.sh` whatever the client asks, and `deploy.sh` only accepts a commit sha.
Don't do this to a key you also log in with: it would only ever deploy.

Host key check: the workflow pins `SHA256:gNZUcA/2b/7QOYTs3WECNAMka4kzfKBGM31WgGaeDi4`.
Compare with `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the server.

### 3. First deploy

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
  job. Or on the server: `~/duke-deploy/deploy.sh <sha>`.
- **By hand**: `~/duke-deploy/deploy.sh latest`.
