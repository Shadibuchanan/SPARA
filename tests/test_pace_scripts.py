import subprocess
from pathlib import Path

import pytest

SCRIPTS = sorted(Path("scripts/pace").glob("*.sbatch")) + sorted(Path("scripts/pace").glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_job_scripts_parse(script):
    # A quoting slip fails only once Slurm starts the job (jobs 6911236 and 7105700).
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
