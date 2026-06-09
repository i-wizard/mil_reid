"""
Bridging uploaded files to the path-based inference seam.

The ML core's inference functions all take image *file paths* (they were written
for offline use over a dataset on disk). FastAPI hands us in-memory ``UploadFile``
streams instead. This module is the single adapter between the two: it validates
each upload is a not-too-large image, writes it to a temp file, yields the path,
and guarantees cleanup — so routers never juggle temp files or re-implement
validation, and no partially-written temp files leak.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterator, List, Sequence

from fastapi import HTTPException, UploadFile, status


def _validate_is_image(upload: UploadFile) -> None:
    """
    Reject anything that is not declared as an image.

    We check ``content_type`` up front so a wrong file type fails with a clear
    415 before we spend effort decoding it deeper in the model.
    """
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected an image upload, got content_type='{upload.content_type}'.",
        )


def _persist_one(upload: UploadFile, directory: str, max_bytes: int) -> str:
    """
    Stream one upload to a temp file under ``directory``, enforcing the size cap.

    Streamed in chunks (rather than reading the whole file into memory) so a
    surprise large upload is caught at ``max_bytes`` without first buffering it
    all. Returns the written path.
    """
    _validate_is_image(upload)
    suffix = os.path.splitext(upload.filename or "")[1] or ".img"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=directory)

    written = 0
    upload.file.seek(0)
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                )
            out.write(chunk)
    return path


@contextmanager
def saved_uploads(uploads: Sequence[UploadFile], max_bytes: int) -> Iterator[List[str]]:
    """
    Persist one or more uploads to temp paths for the duration of the request.

    Yields the list of temp paths and removes the whole temp directory on exit —
    success or error — so the model gets real file paths and nothing lingers on
    disk afterwards. Used by every endpoint that accepts image uploads.
    """
    directory = tempfile.mkdtemp(prefix="reid_upload_")
    try:
        paths = [_persist_one(upload=u, directory=directory, max_bytes=max_bytes) for u in uploads]
        yield paths
    finally:
        shutil.rmtree(directory, ignore_errors=True)
