from sqlalchemy.orm import configure_mappers

from app.core.config import get_settings
from app.db.base import Base
from app.models.document import Document
from app.models.document_chunk import DocumentChunk


def test_database_metadata_contains_expected_tables() -> None:
    assert Document.__tablename__ in Base.metadata.tables
    assert DocumentChunk.__tablename__ in Base.metadata.tables


def test_document_chunk_embedding_dimension_matches_settings() -> None:
    embedding_column = DocumentChunk.__table__.columns["embedding"]

    assert embedding_column.type.dim == get_settings().embedding_dimension


def test_sqlalchemy_relationships_can_be_configured() -> None:
    configure_mappers()
