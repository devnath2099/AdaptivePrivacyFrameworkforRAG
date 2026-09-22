import sys
import pytest
from review2_final.notebook_runner import run_logged


def test_subprocess_failure_preserves_actual_traceback(tmp_path):
    with pytest.raises(RuntimeError, match="specific child failure"):
        run_logged([sys.executable, "-u", "-c", "raise ValueError('specific child failure')"], tmp_path)
    log, = tmp_path.glob("*.log")
    assert "ValueError: specific child failure" in log.read_text()
    assert "Exit code: 1" in log.read_text()


def test_success_streams_and_retains_output(tmp_path, capsys):
    path = run_logged([sys.executable, "-u", "-c", "print('completed')"], tmp_path)
    assert "completed" in capsys.readouterr().out
    assert "Exit code: 0" in path.read_text()
