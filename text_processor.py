from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_loader import load_all_records


def split_records(records):
    """Split loaded records into smaller searchable chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for record in records:
        text = record["text"]

        if not text.strip():
            continue

        record_chunks = splitter.split_text(text)

        for index, chunk_text in enumerate(record_chunks):
            chunks.append({
                "id": f"{record['id']}_chunk_{index}",
                "text": chunk_text,
                "metadata": {
                    **record["metadata"],
                    "parent_record_id": record["id"],
                    "chunk_index": index
                }
            })

    return chunks


if __name__ == "__main__":
    records = load_all_records()
    chunks = split_records(records)

    print("\n" + "=" * 50)
    print("TEXT CHUNKING RESULTS")
    print("=" * 50)

    print(f"Records loaded: {len(records)}")
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk ID:", chunks[0]["id"])
        print("First chunk metadata:", chunks[0]["metadata"])
        print("\nFirst chunk text:")
        print(chunks[0]["text"][:500])