# orchestrator/output_validator.py
# Validates spec dicts against the SpecJSON Pydantic schema.
# Also handles file save/load and JSON Schema export.

import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))

import json
import os
from typing import Dict, Any
from pydantic import ValidationError
from JAY.shared.models import SpecJSON

def validate_spec(spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a spec dictionary. Sets validation_passed=True on success.
    Raises ValueError with all field errors listed if validation fails.
    """
    try:
        spec_obj = SpecJSON.model_validate(spec_dict)
        spec_obj.validation_passed = True
        return spec_obj.model_dump(mode="json")
    except ValidationError as e:
        errors = [f"  • {' → '.join(str(x) for x in err['loc'])}: {err['msg']}"
                  for err in e.errors()]
        raise ValueError(
            f"Spec validation failed ({len(e.errors())} errors):\n" + "\n".join(errors)
        )


def serialize_to_json(spec_dict: Dict[str, Any], pretty: bool = True) -> str:
    """Serialize a validated spec dict to a JSON string."""
    return json.dumps(spec_dict, indent=2 if pretty else None,
                      ensure_ascii=False, default=str)


def save_spec(spec_dict: Dict[str, Any], output_path: str = "spec.json") -> None:
    """Write a validated spec to disk."""
    json_str = serialize_to_json(spec_dict)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"[OUTPUT] spec.json saved → {output_path}  ({len(json_str):,} bytes)")


def load_spec(path: str) -> SpecJSON:
    """Load and re-validate a spec.json from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SpecJSON.model_validate(data)


def export_json_schema(output_path: str = "docs/spec_schema.json") -> None:
    """Export the SpecJSON Pydantic model as a JSON Schema file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    schema = SpecJSON.model_json_schema()
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[SCHEMA] Exported JSON Schema → {output_path}")


if __name__ == "__main__":
    export_json_schema()
    print("Schema exported.")
