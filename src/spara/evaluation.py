"""Execute the hash-pinned reference scorer, shared by every experiment."""
import ast
import inspect
import hashlib
from importlib.metadata import version
from pathlib import Path

from mathruler.grader import extract_boxed_content

from .data import REFERENCE_SHA, sha256


class ReferenceEvaluator:
    def __init__(self, data_root):
        source = Path(data_root) / "reference" / "task.py"
        if sha256(source) != REFERENCE_SHA:
            raise ValueError("Reference evaluator source checksum mismatch")
        # Isolate the exact upstream scorer without importing image/helper training code.
        tree = ast.parse(source.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if
                     isinstance(node, ast.FunctionDef) and node.name in {"parse_action_sequence", "simulate_vsp"}
                     or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ACTION_MAP" for t in node.targets)]
        namespace = {}
        exec(compile(tree, str(source), "exec"), namespace)
        self.simulate = namespace["simulate_vsp"]
        if version("mathruler") != "0.1.0":
            raise ValueError("The reproduction pins mathruler==0.1.0")

    @property
    def provenance(self):
        return {"task_sha256": REFERENCE_SHA, "mathruler": version("mathruler"),
                "extractor_sha256": hashlib.sha256(inspect.getsource(extract_boxed_content).encode()).hexdigest()}

    def score(self, row, decoded_output):
        answer = decoded_output.split("<|im_start|>assistant")[-1]
        extracted = extract_boxed_content(answer)
        result = self.simulate(row["map_desc"], extracted)
        return {"map_id": row["map_id"], "grid_size": len(row["map_desc"]),
                "raw_response": decoded_output, "answer": answer,
                "extracted_answer": extracted, **result}


def summarize(outputs):
    def stats(items):
        n = len(items)
        return {"count": n, "successes": sum(bool(x["success"]) for x in items),
                "accuracy": sum(bool(x["success"]) for x in items) / n if n else None}
    return {"overall": stats(outputs), "per_grid": {
        str(size): stats([x for x in outputs if x["grid_size"] == size])
        for size in sorted({x["grid_size"] for x in outputs})}}
