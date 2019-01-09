schema = """
{
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "title": "Dilemma",
    "description": "A moral dilemma containing two options",
    "properties": {
        "firstOption": {
            "type": "array",
            "items": {
                "$ref": "#/definitions/person"
            }
        },
        "secondOption": {
            "type": "array",
            "items": {
                "$ref": "#/definitions/person"
            }
        }
    },
    "additionalProperties": False,
    "required": ["firstOption", "secondOption"],
    "definitions": {
        "person": {
            "type": "object",
            "title": "Person",
            "description": "A person with varying attributes",
            "properties": {
                "age": {
                    "type": "number",
                    "minimum": 0
                },
                "race": {
                    "type": "string",
                    "enum": [
                        "white",
                        "black",
                        "asian",
                        "native american",
                        "other race"
                    ]
                },
                "legal sex": {
                    "type": "string",
                    "enum": [
                        "male",
                        "female"
                    ]
                },
                "jaywalking": {
                    "type": "boolean"
                },
                "driving under the influence": {
                    "type": "boolean"
                }
            },
            "additionalProperties": False
        }
    }
}
"""
