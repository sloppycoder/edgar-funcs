from typing import Optional

from pydantic import BaseModel

from edgar_funcs.rag.extract.llm import (
    _convert_json_schema_to_google_schema,
)


class NestdType(BaseModel):
    is_nested: bool
    name: str


class MainType(BaseModel):
    id: int
    name: str
    nested: NestdType
    maybe_nested: Optional[NestdType]


def test_convert_json_schema_to_google_schema():
    google_schema = _convert_json_schema_to_google_schema(MainType.model_json_schema())
    # print(json.dumps(google_schema, indent=2))
    props = google_schema["properties"]
    required = google_schema.get("required", [])
    assert props["id"]["type"] == "INTEGER"
    assert props["nested"]["type"] == "OBJECT"
    assert props["maybe_nested"]["type"] == "OBJECT"
    assert "id" in required and "nested" in required
    assert props["nested"]["properties"]["is_nested"]["type"] == "BOOLEAN"
