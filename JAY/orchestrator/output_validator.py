# orchestrator/output_validator.py
# Validates spec dicts against the SpecJSON Pydantic schema.
# Handles file I/O with atomic writes and comprehensive logging.

import logging
import json
import os
import tempfile
from typing import Dict, Any
from pydantic import ValidationError
from JAY.shared.models import SpecJSON

logger = logging.getLogger(__name__)


def validate_spec(spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a spec dictionary. Sets validation_passed=True on success.
    Raises ValueError with all field errors listed if validation fails.
    """
    try:
        spec_obj = SpecJSON.model_validate(spec_dict)
        spec_obj.validation_passed = True
        logger.info(f"✅ Spec validation passed for {spec_obj.pipeline_id}")
        return spec_obj.model_dump(mode="json")
    except ValidationError as e:
        errors = [
            f"  • {' → '.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        ]
        error_msg = (
            f"Spec validation failed ({len(e.errors())} errors):\n" 
            + "\n".join(errors)
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def serialize_to_json(spec_dict: Dict[str, Any], pretty: bool = True) -> str:
    """Serialize a validated spec dict to a JSON string."""
    return json.dumps(
        spec_dict,
        indent=2 if pretty else None,
        ensure_ascii=False,
        default=str
    )


def save_spec(spec_dict: Dict[str, Any], output_path: str = "spec.json") -> None:
    """Write a validated spec to disk with atomic semantics (temp file + rename)."""
    # Validate output path parent exists
    output_dir = os.path.dirname(output_path) or "."
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            raise

    # Serialize to JSON
    json_str = serialize_to_json(spec_dict)

    # Atomic write: write to temp file, then rename
    try:
        temp_fd, temp_path = tempfile.mkstemp(dir=output_dir, suffix=".json")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(json_str)
            os.replace(temp_path, output_path)
            logger.info(
                f"✅ Spec saved to {output_path} "
                f"({len(json_str):,} bytes)"
            )
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise
    except Exception as e:
        logger.error(f"Failed to save spec to {output_path}: {e}")
        raise


def load_spec(path: str) -> SpecJSON:
    """Load and re-validate a spec.json from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spec file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded spec from {path}")
        return SpecJSON.model_validate(data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        raise
    except ValidationError as e:
        logger.error(f"Validation failed for {path}: {e}")
        raise


def export_json_schema(output_path: str = "docs/spec_schema.json") -> None:
    """Export the SpecJSON Pydantic model as a JSON Schema file."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create schema directory {output_dir}: {e}")
            raise

    try:
        schema = SpecJSON.model_json_schema()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        logger.info(f"✅ JSON Schema exported to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export schema to {output_path}: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_json_schema()
    logger.info("Schema exported successfully.")
