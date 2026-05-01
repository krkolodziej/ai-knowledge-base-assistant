import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, document: Document) -> Document:
        self.db.add(document)
        self.db.flush()
        self.db.refresh(document)
        return document

    def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return self.db.get(Document, document_id)

    def list(self) -> list[Document]:
        statement = select(Document).order_by(Document.created_at.desc())
        return list(self.db.scalars(statement).all())

    def delete(self, document: Document) -> None:
        self.db.delete(document)
        self.db.flush()
