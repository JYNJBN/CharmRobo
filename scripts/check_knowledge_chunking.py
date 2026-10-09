"""本地切块回归检查；不连接数据库、COS 或 Milvus。"""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.models import KnowledgeChunk, KnowledgeDocument, KnowledgeIngestJob
from app.services import knowledge_ingest_job_service as worker
from app.services.knowledge_chunk_service import (
    KnowledgeChunkError,
    build_element_chunks,
)


def make_document(values):
    text = ""
    elements = []
    for kind, body, path in values:
        start = len(text) + (2 if text else 0)
        text += ("\n\n" if text else "") + body
        elements.append({
            "text": body,
            "element_type": kind,
            "heading_path": path,
            "title": path[-1] if path else "demo",
            "start_offset": start,
            "end_offset": len(text),
            "metadata": {},
        })
    return KnowledgeDocument(
        id=1, user_id=1, knowledge_base_id=1, ingest_generation=1,
        title="demo", parsed_text=text, parsed_elements_json=elements,
    )


class ChunkChecks(unittest.TestCase):
    def test_paragraphs_merge_and_preserve_offsets(self):
        document = make_document([
            ("paragraph", "first paragraph", ["A"]),
            ("paragraph", "second paragraph", ["A"]),
        ])
        chunks = build_element_chunks(
            document, target_chars=100, overlap_chars=10,
        )
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.metadata_json["heading_path"], ["A"])
        self.assertEqual(
            chunk.content,
            document.parsed_text[chunk.start_offset:chunk.end_offset],
        )

    def test_different_sections_do_not_merge(self):
        document = make_document([
            ("paragraph", "a " * 150, ["A"]),
            ("paragraph", "b " * 100, ["B"]),
        ])
        chunks = build_element_chunks(
            document, target_chars=100, overlap_chars=20,
        )
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 100)
            self.assertFalse("a" in chunk.content and "b" in chunk.content)
            self.assertEqual(
                chunk.content,
                document.parsed_text[chunk.start_offset:chunk.end_offset],
            )

    def test_large_table_keeps_headers_and_all_rows(self):
        header = "| Name | Price |\n| --- | --- |\n"
        table = header + "\n".join(
            f"| Room{i:02d} | {300 + i} |" for i in range(20)
        )
        document = make_document([("table", table, ["Rooms"])])
        chunks = build_element_chunks(
            document, target_chars=100, overlap_chars=20,
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.element_type, "table")
            self.assertLessEqual(len(chunk.content), 100)
            body = document.parsed_text[
                chunk.start_offset:chunk.end_offset
            ].rstrip()
            self.assertEqual(chunk.content, header + body)
        for index in range(20):
            self.assertEqual(
                sum(f"| Room{index:02d} |" in c.content for c in chunks),
                1,
            )

    def test_oversized_table_row_is_rejected(self):
        document = make_document([
            ("table", "| N |\n| --- |\n| " + "x" * 200 + " |", ["A"]),
        ])
        with self.assertRaises(KnowledgeChunkError) as caught:
            build_element_chunks(
                document, target_chars=100, overlap_chars=20,
            )
        self.assertEqual(caught.exception.code, "TABLE_ROW_TOO_LARGE")


class FakeSession:
    def __init__(self, document, job, stored):
        self.document = document
        self.job = job
        self.stored = stored
        self.additions = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def begin(self):
        return self

    async def scalar(self, statement):
        entity = statement.column_descriptions[0].get("entity")
        if entity is KnowledgeIngestJob:
            return self.job
        if entity is KnowledgeDocument:
            return self.document
        return datetime.now(timezone.utc)

    async def scalars(self, statement):
        entity = statement.column_descriptions[0].get("entity")
        if entity is not KnowledgeChunk:
            raise AssertionError("Unexpected query")
        return SimpleNamespace(all=lambda: self.stored)

    def add_all(self, chunks):
        self.additions.append(list(chunks))


class WorkerChunkChecks(unittest.IsolatedAsyncioTestCase):
    async def run_stage(self, version, stored=None):
        document = make_document([("paragraph", "hello world", [])])
        job = KnowledgeIngestJob(
            id=1, document_id=1, knowledge_base_id=1, user_id=1,
            ingest_generation=1, status="running", stage="chunk",
            locked_until=datetime.now(timezone.utc) + timedelta(seconds=120),
            config_snapshot={
                "chunker_version": version,
                "chunk_target_chars": 100,
                "chunk_overlap_chars": 10,
            },
            result_json={},
        )
        session = FakeSession(document, job, stored or [])
        with patch.object(worker, "AsyncSessionLocal", return_value=session):
            result = await worker.process_chunk_stage({
                "job_id": 1,
                "lease_token": "test",
            })
        return result, document, job, session

    async def test_new_chunks_are_inserted_only_once(self):
        result, document, job, session = await self.run_stage(
            "element-overlap-v1",
        )
        self.assertTrue(result)
        self.assertEqual(len(session.additions), 1)
        self.assertEqual(session.additions[0][0].start_offset, 0)
        self.assertEqual(job.stage, "embed")
        self.assertEqual(document.chunk_count, 1)

    async def test_existing_ids_are_reused(self):
        _, _, _, first = await self.run_stage("element-overlap-v1")
        stored = first.additions[0]
        stored[0].id = 900
        result, document, job, session = await self.run_stage(
            "element-overlap-v1", stored,
        )
        self.assertTrue(result)
        self.assertEqual(session.additions, [])
        self.assertEqual(stored[0].id, 900)
        self.assertEqual(document.chunk_count, 1)
        self.assertEqual(job.stage, "embed")

    async def test_snapshot_mismatch_does_not_overwrite_chunks(self):
        _, _, _, first = await self.run_stage("element-overlap-v1")
        stored = first.additions[0]
        stored[0].embedding_text_hash = "changed"
        result, document, job, session = await self.run_stage(
            "element-overlap-v1", stored,
        )
        self.assertFalse(result)
        self.assertEqual(session.additions, [])
        self.assertEqual(document.error_code, "CHUNK_SNAPSHOT_MISMATCH")
        self.assertEqual(job.status, "failed")

    async def test_legacy_version_remains_supported(self):
        result, _, _, session = await self.run_stage("char-overlap-v1")
        self.assertTrue(result)
        self.assertEqual(len(session.additions), 1)
        self.assertIsNone(session.additions[0][0].start_offset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
