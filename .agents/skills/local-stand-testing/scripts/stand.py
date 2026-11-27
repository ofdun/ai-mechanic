"""Manage a private Compose project for the checkout in the current directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from zipfile import BadZipFile

from check_inputs import check

HARNESS = Path(__file__).resolve().parents[4]


def output(cmd, **kwargs):
    return subprocess.check_output(cmd, text=True, **kwargs).strip()


def password_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", opener=lambda name, flags: os.open(name, flags, 0o600)) as f:
            f.write(secrets.token_urlsafe(24) + "\n")
    except FileExistsError:
        pass
    return path.read_text().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["up", "e2e", "status", "down", "logs", "config"])
    parser.add_argument("--instance", default="dev", help="Independent slot within this checkout")
    parser.add_argument("--baseline", action="store_true", help="Use offline agent instead of stand profile A on up")
    parser.add_argument("--test", help="Browser test under the feature checkout's tests/ (for e2e)")
    args = parser.parse_args()
    root = Path(output(["git", "rev-parse", "--show-toplevel"])).resolve()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", args.instance):
        parser.error("Имя экземпляра: до 24 строчных латинских букв, цифр или дефисов.")
    identity = hashlib.sha256(os.fsencode(root)).hexdigest()[:12]
    project = f"mech-{identity}-{args.instance}"
    private = root / "build" / "local-stand" / args.instance
    if not all((root / f).is_file() for f in ("compose.yaml", "compose.stand.yaml", "app/speech/Dockerfile")):
        parser.error("Нужна рабочая копия ai-mechanic с Compose и сервисом speech.")
    env = dict(os.environ, LOCAL_HARNESS=str(HARNESS), LOCAL_FLEET_IMAGE=project + "-fleet:local",
               STAND_HOST=project + ".localhost", STAND_USER="mechanic",
               STAND_PASSWORD=password_file(private / "password"), STAND_PASSWORD_HASH="",
               COMPOSE_PROFILES="mocks", STATE_DIR="state", FLEET_USER="mechanic", WEB_LISTEN_PORT="8080",
               AGENT_URL="http://agent:8000", SPEECH_URL="http://speech:8003",
               ML_API_URL="http://ml:8001", MAINTENANCE_API_URL="http://maintenance:8002",
               PLANNER_MODE="local", AGENT_ARCHITECTURE="baseline" if args.baseline else "a",
               LOCAL_EXTERNAL_EVIDENCE="false" if args.baseline else "true", WEB_RETRIEVAL_MODE="team",
               RETRIEVAL_CONFIG_PATH="/app/configs/retrieval-web.yaml")
    if args.baseline:
        env["LLM_API_KEY"] = ""
    dc = ["docker", "compose", "-p", project, "--project-directory", str(root),
          "-f", str(root/"compose.yaml"), "-f", str(root/"compose.stand.yaml"),
          "-f", str(HARNESS/"compose.local.yaml"), "--profile", "mocks"]

    def run(*args):
        subprocess.run(dc + list(args), env=env, check=True)

    def read(*args):
        return output(dc + list(args), env=env)

    if args.action in {"config", "up"}:
        config = json.loads(read("config", "--format", "json"))
        services = config["services"]
        mounts = {m["target"]: Path(m["source"]) for svc in ("agent", "ml")
                  for m in services[svc]["volumes"] if m["type"] == "bind"}
        check(mounts["/fleet"], mounts["/mocks"])
        print(f"{project}: {env['AGENT_ARCHITECTURE']}, SQLite volume {config['volumes']['state']['name']}", flush=True)
        if args.action == "config":
            return
        if not args.baseline and not services["agent"]["environment"].get("LLM_API_KEY"):
            parser.error("Профилю A нужен LLM_API_KEY в локальном .env или окружении. Для автономного теста укажите --baseline.")

    endpoint = os.environ.get("DOCKER_HOST") if not os.environ.get("DOCKER_CONTEXT") else None
    if not endpoint:
        endpoint = output(["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}", output(["docker","context","show"])])
    if not endpoint.startswith(("unix://", "npipe://")):
        parser.error("Для локального стенда выберите Docker context с локальным сокетом.")
    if args.action == "down":
        run("down")
    elif args.action == "logs":
        run("logs", "--tail=60", "gateway", "web", "agent", "index", "speech")
    elif args.action == "e2e":
        test = "/tests/smoke.py"
        if args.test:
            path = (root / "tests" / args.test).resolve()
            if not path.is_file() or not path.is_relative_to(root / "tests") or path.suffix != ".py":
                parser.error("--test должен указывать на Python-файл внутри tests/")
            test = "/feature-tests/" + path.relative_to(root / "tests").as_posix()
        read("port", "gateway", "443")
        run("--profile", "test", "run", "--rm", "--no-deps", "--build", "browser", test)
    else:
        if args.action == "up":
            run("build", "agent", "web", "speech")
            run("up", "-d", "--wait", "--wait-timeout", "600")
            ca = read("exec", "-T", "gateway", "cat", "/data/caddy/pki/authorities/local/root.crt")
            (private / "ca.crt").write_text(ca + "\n")
        run("ps", "-a")
        port = read("port", "gateway", "443").rsplit(":", 1)[1]
        print(f"https://{env['STAND_HOST']}:{port}/\nЛогин: mechanic\nФайл пароля: {private / 'password'}\nCA: {private / 'ca.crt'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, BadZipFile, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
