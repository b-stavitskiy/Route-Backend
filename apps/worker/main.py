import asyncio

from arq import run_worker
from arq.connections import RedisSettings
from apps.worker.tasks.main import (
    WORKER_FUNCTIONS,
    shutdown,
    startup,
)


async def main():
    await run_worker(
        functions=WORKER_FUNCTIONS,
        redis_settings=RedisSettings(),
        on_startup=startup,
        on_shutdown=shutdown,
        max_jobs=100,
        job_timeout=300,
        keep_result=3600,
    )


if __name__ == "__main__":
    asyncio.run(main())
