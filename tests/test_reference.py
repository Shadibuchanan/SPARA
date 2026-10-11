import os
from pathlib import Path
import pytest
from spara.data import audit, records
from spara.evaluation import ReferenceEvaluator


@pytest.fixture
def data_root():
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    if not (root / "manifest.json").is_file():
        pytest.skip("Run spara prepare-data first")
    return root


@pytest.mark.integration
def test_original_split_and_training_answers(data_root):
    report = audit(data_root)
    assert report["splits"]["train"]["per_grid"] == {3: 100, 4: 200, 5: 300, 6: 400}
    assert report["splits"]["test"]["per_grid"] == {3: 100, 4: 100, 5: 100, 6: 100}
    assert report["shared_layouts"] == 7 and report["test_rows_with_train_layout"] == 8
    evaluator = ReferenceEvaluator(data_root)
    assert all(evaluator.score(row, row["text_output"])["success"] for row in records(data_root, "train"))


@pytest.mark.integration
@pytest.mark.parametrize("answer,success,status", [
    (r"\boxed{R}", True, "Reached goal"),
    (r"\boxed{L,R}", True, "Reached goal"),
    (r"\boxed{R,L}", False, "Did not reach goal"),
    (r"\boxed{R,D,U}", False, "Fell in hole"),
    (r"\boxed{WRONG}", True, "Reached goal"),
    ("no answer", False, "Did not reach goal"),
    (r"\boxed{R", False, "Did not reach goal"),
    (r"\boxed{{R}}", True, "Reached goal"),
    (r"\boxed{R} then \boxed{", False, "Did not reach goal"),
    (r"\boxed{R} then \boxed{L}", False, "Did not reach goal"),
])
def test_reference_quirks(data_root, answer, success, status):
    row = {"map_id": 1, "map_desc": [[1, 2], [0, -1]]}
    result = ReferenceEvaluator(data_root).score(row, answer)
    assert result["success"] == success
    assert result["status"] == status


@pytest.mark.integration
def test_tampered_reference_rejected(data_root, tmp_path):
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference/task.py").write_text("raise RuntimeError('should never execute')")
    with pytest.raises(ValueError, match="checksum"):
        ReferenceEvaluator(tmp_path)
