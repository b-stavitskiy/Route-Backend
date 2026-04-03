import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml


async def pull_config_from_github(
    github_token: str,
    repo_url: str,
    branch: str = "main",
    config_files: list[str] | None = None,
) -> dict[str, Any]:
    import git

    if config_files is None:
        config_files = ["provider.yaml", "plans.yaml", "settings.yaml"]

    auth_url = repo_url.replace("https://", f"https://{github_token}@")

    loop = asyncio.get_event_loop()

    def _clone_and_read():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = git.Repo.clone_from(
                auth_url,
                tmpdir,
                branch=branch,
                depth=1,
            )

            configs = {}
            for config_file in config_files:
                file_path = Path(tmpdir) / config_file
                if file_path.exists():
                    with open(file_path) as f:
                        configs[config_file] = yaml.safe_load(f)

            return configs

    configs = await loop.run_in_executor(None, _clone_and_read)
    return configs


def load_local_config(config_dir: str = "config") -> dict[str, Any]:
    config_path = Path(config_dir)
    configs = {}

    for config_file in ["provider.yaml", "plans.yaml", "settings.yaml"]:
        file_path = config_path / config_file
        if file_path.exists():
            with open(file_path) as f:
                configs[config_file] = yaml.safe_load(f)

    return configs


def find_route_configs_dir() -> Path | None:
    current = Path(__file__).parent.parent.parent
    for parent in [current] + list(current.parents):
        route_configs = parent / "Route-Configs"
        if route_configs.exists() and route_configs.is_dir():
            provider_yaml = route_configs / "provider.yaml"
            if provider_yaml.exists():
                return route_configs
    return None


async def get_configs(
    use_remote: bool = True,
    github_token: str = "",
    repo_url: str = "https://github.com/RoutingRun/Route-Configs.git",
    branch: str = "main",
    config_dir: str = "config",
) -> dict[str, Any]:
    if use_remote and github_token:
        try:
            configs = await pull_config_from_github(
                github_token=github_token,
                repo_url=repo_url,
                branch=branch,
            )
            if configs:
                return configs
            print(
                f"[config_puller] WARNING: Remote config returned empty dict, falling back to local"
            )
        except Exception as e:
            print(f"[config_puller] ERROR pulling from github: {e}")

    route_configs = find_route_configs_dir()
    if route_configs:
        print(f"[config_puller] Using local config from: {route_configs}")
        return load_local_config(str(route_configs))

    print(f"[config_puller] WARNING: No config found, returning empty dict")
    return load_local_config(config_dir)
