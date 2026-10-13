import json
import re
from pathlib import Path
from typing import Any

#CONFIGURATION

DATA_DIR = (
Path(__file__).parent
/ "data"
/ "processed"
/ "json"
)

#COMMON HELPER FUNCTIONS

def load_json_file(file_path: Path) -> Any:
    """Load a JSON file using UTF-8 encoding."""

    try:
        with file_path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {file_path.name}, "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

def safe_id(value: str) -> str:
    """Convert a value into a safe ID component."""

    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))
    return value.strip("_") or "record"

def make_record(
    record_id: str,
    text: str,
    metadata: dict
    ) -> dict:
    """Create a standardized record."""

    return {
        "id": record_id,
        "text": text.strip(),
        "metadata": metadata
    }



#FORMAT 1: CAREER JSON
#Structure: career_roles_by_qualification

def load_career_records(data: dict) -> list[dict]:
    """Load career information organized by qualification."""

    records = []
    qualifications = data.get(
        "career_roles_by_qualification", {}
    )

    if not isinstance(qualifications, dict):
        raise ValueError(
            "'career_roles_by_qualification' must be an object."
        )

    for code, details in qualifications.items():

        if not isinstance(details, dict):
            continue

        title = details.get("qualification_title", code)
        description = details.get("description", "")
        roles = details.get("roles", [])

        if not isinstance(roles, list):
            roles = [str(roles)]

        roles_text = "\n".join(
            f"- {role}" for role in roles
        )

        text = (
            f"Qualification: {title}\n"
            f"Qualification Code: {code}\n"
            f"Description: {description}\n"
            f"Career Roles:\n{roles_text}"
        )

        records.append(
            make_record(
                record_id=f"career_{safe_id(code)}",
                text=text,
                metadata={
                    "record_type": "career",
                    "qualification_code": str(code),
                    "qualification_title": str(title)
                }
            )
        )

    return records


#FORMAT 2: HIGHER STUDY JSON
#Structure: career_records

def load_higher_study_records(data: dict) -> list[dict]:
    """Load higher-study and career pathway information."""

    records = []
    study_records = data.get("career_records", [])

    if not isinstance(study_records, list):
        raise ValueError("'career_records' must be an array.")

    searchable_fields = [
        "higher_study",
        "specialization",
        "career_area",
        "career_role",
        "role_description",
        "workplace",
        "qualification_required",
        "registration_required",
        "experience_required",
        "additional_requirements",
        "responsibilities",
        "government_private",
        "organization",
        "pay_information",
        "career_progression",
        "further_study",
        "international_opportunity",
        "interpretation",
        "notes"
    ]

    for index, item in enumerate(study_records):

        if not isinstance(item, dict):
            continue

        source_record_id = str(
            item.get(
                "career_record_id",
                f"HS-{index + 1:04d}"
            )
        )

        text_parts = []

        for field in searchable_fields:
            value = item.get(field)

            if value is None or value == "":
                continue

            if isinstance(value, list):
                value = "; ".join(str(v) for v in value)

            text_parts.append(
                f"{field.replace('_', ' ').title()}: {value}"
            )

        text = "\n".join(text_parts)

        source_ids = item.get("source_ids", [])

        if isinstance(source_ids, list):
            source_ids = ", ".join(
                str(source_id) for source_id in source_ids
            )
        else:
            source_ids = str(source_ids)

        metadata = {
            "record_type": "higher_study",
            "career_record_id": source_record_id,
            "source_ids": source_ids,
            "page_section": str(
                item.get("page_section", "")
            ),
            "jurisdiction": str(
                item.get("jurisdiction", "")
            ),
            "publication_date": str(
                item.get("publication_date", "")
            ),
            "effective_date": str(
                item.get("effective_date", "")
            ),
            "academic_year": str(
                item.get("academic_year", "")
            ),
            "last_verified": str(
                item.get("last_verified", "")
            ),
            "confidence": str(
                item.get("confidence", "")
            )
        }

        if text.strip():
            records.append(
                make_record(
                    record_id=(
                        f"higher_study_"
                        f"{safe_id(source_record_id)}"
                    ),
                    text=text,
                    metadata=metadata
                )
            )

    return records

#FORMAT 3 AND 4: JOBS AND EDUCATION JSON
#Structure: document_title, category, sections

def content_item_to_text(item: dict) -> str:
    """Convert a section content item into readable text."""

    if not isinstance(item, dict):
        return ""

    content_type = item.get("type", "text")
    text = item.get("text", "")

    if content_type == "heading":
        return f"\n{text}\n" if text else ""

    if content_type == "text":
        return str(text) if text else ""

    if content_type == "list":
        items = item.get("items", [])

        if isinstance(items, list):
            return "\n".join(
                f"- {value}" for value in items
            )

        return str(items)

    # Preserve available text for unrecognized content types.
    if text:
        return str(text)

    return ""

def load_section_records(
data: dict,
file_path: Path
) -> list[dict]:
    """Load section-based documents such as Jobs and Education."""

    records = []

    sections = data.get("sections", [])

    if not isinstance(sections, list):
        raise ValueError("'sections' must be an array.")

    document_title = data.get(
        "document_title",
        file_path.stem
    )

    category = data.get(
        "category",
        "Uncategorized"
    )

    filename = file_path.stem.lower()

    if "education" in filename:
        record_type = "education"
    elif "job" in filename or "career" in filename:
        record_type = "jobs"
    else:
        record_type = "section_document"

    for index, section in enumerate(sections):

        if not isinstance(section, dict):
            continue

        section_number = section.get(
            "number",
            index + 1
        )

        section_title = section.get(
            "title",
            f"Section {section_number}"
        )

        content_items = section.get("content", [])

        if not isinstance(content_items, list):
            continue

        text_parts = [
            f"Document: {document_title}",
            f"Category: {category}",
            f"Section: {section_title}"
        ]

        for content_item in content_items:
            content_text = content_item_to_text(
                content_item
            )

            if content_text.strip():
                text_parts.append(content_text)

        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            continue

        record_id = (
            f"{safe_id(filename)}_section_"
            f"{safe_id(str(section_number))}"
        )

        metadata = {
            "record_type": record_type,
            "document_title": str(document_title),
            "category": str(category),
            "section_number": str(section_number),
            "section_title": str(section_title),
            "source_file": file_path.name
        }

        records.append(
            make_record(
                record_id=record_id,
                text=full_text,
                metadata=metadata
            )
        )

    return records

#MAIN LOADER
#Detect and load all supported JSON formats

def load_all_records() -> list[dict]:
    """Load supported JSON files from the data directory."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"JSON directory not found: {DATA_DIR}\n"
            "Create data/processed/json and add your JSON files."
        )

    all_records = []

    for file_path in sorted(DATA_DIR.glob("*.json")):

        data = load_json_file(file_path)

        if not isinstance(data, dict):
            print(
                f"Skipping {file_path.name}: "
                "expected a JSON object."
            )
            continue

        if "career_roles_by_qualification" in data:
            records = load_career_records(data)

        elif "career_records" in data:
            records = load_higher_study_records(data)

        elif "sections" in data:
            records = load_section_records(
                data,
                file_path
            )

        else:
            print(
                f"Skipping {file_path.name}: "
                "unsupported JSON structure."
            )
            continue

        # Attach the source filename to every record.
        for record in records:
            record["metadata"]["source_file"] = (
                file_path.name
            )

        all_records.extend(records)

        print(
            f"{file_path.name}: "
            f"{len(records)} records loaded"
        )

    # Check for duplicate record IDs.
    seen_ids = set()

    for record in all_records:

        if record["id"] in seen_ids:
            raise ValueError(
                f"Duplicate record ID detected: {record['id']}"
            )

        seen_ids.add(record["id"])

    return all_records

#TEST THE DATA LOADER

if __name__ == "__main__":

    records = load_all_records()

    print("\n" + "=" * 50)
    print(f"TOTAL RECORDS LOADED: {len(records)}")
    print("=" * 50)

    for record in records[:3]:

        print(f"\nRecord ID: {record['id']}")
        print(f"Metadata: {record['metadata']}")
        print("\nText Preview:")
        print(record["text"][:500])