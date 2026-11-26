from unittest.mock import MagicMock

from pypdf import PdfReader

from app.role_kb_ingestion import (
    ROLE_KB_SPECS,
    extract_pdf_chunks,
    generate_role_pdfs,
    ingest_role_kbs_to_supabase,
)


def test_generate_role_pdfs_creates_readable_pdfs_for_all_four_roles(tmp_path):
    paths = generate_role_pdfs(base_data_dir=tmp_path)

    assert len(paths) == 4
    expected_roles = {"SELLER", "SUPPORT", "OPERATIONS", "ADMIN"}
    assert {spec.role_scope for spec in ROLE_KB_SPECS} == expected_roles

    for spec, pdf_path in zip(ROLE_KB_SPECS, paths):
        assert pdf_path.is_file()
        assert pdf_path.parent.name == spec.folder_name

        reader = PdfReader(str(pdf_path))
        assert len(reader.pages) >= 1
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        assert f"Role Scope: {spec.role_scope}" in extracted

        chunks = extract_pdf_chunks(pdf_path, spec.role_scope)
        assert len(chunks) >= 1
        assert all(chunk["role_scope"] == spec.role_scope for chunk in chunks)
        assert all(chunk["source"] == spec.filename for chunk in chunks)


def test_ingest_role_kbs_to_supabase_upserts_role_scoped_vectors(tmp_path, monkeypatch):
    generate_role_pdfs(base_data_dir=tmp_path)

    fake_table = MagicMock()
    fake_table.upsert.return_value.execute.return_value = MagicMock(data=[])
    fake_supabase = MagicMock()
    fake_supabase.table.return_value = fake_table

    monkeypatch.setattr(
        "app.role_kb_ingestion.get_supabase_client",
        lambda: fake_supabase,
    )
    monkeypatch.setattr(
        "app.role_kb_ingestion.embed_query",
        lambda text: [0.05] * 384,
    )

    summary = ingest_role_kbs_to_supabase(base_data_dir=tmp_path)

    assert set(summary.keys()) == {"SELLER", "SUPPORT", "OPERATIONS", "ADMIN"}
    assert all(count >= 1 for count in summary.values())
    assert fake_table.upsert.call_count == 4
