from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentCreate


class DocumentServiceError(Exception):
    pass


class DocumentService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.documents = DocumentRepository(db)

    def create_document(self, payload: DocumentCreate) -> Document:
        document = Document(
            title=payload.title,
            content=payload.content,
            content_type=payload.content_type,
            status="pending",
            document_metadata=payload.metadata,
        )

        try:
            created_document = self.documents.add(document)
            self.db.commit()
            self.db.refresh(created_document)
        except SQLAlchemyError as exc:
            self.db.rollback()
            raise DocumentServiceError("Could not save document.") from exc

        return created_document

    def list_documents(self) -> list[Document]:
        try:
            return self.documents.list()
        except SQLAlchemyError as exc:
            raise DocumentServiceError("Could not list documents.") from exc
