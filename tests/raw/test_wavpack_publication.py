"""Keep archive publication safe on removable filesystems without hard links."""

from contextlib import contextmanager
import errno
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, Iterator

import pytest

import neuro_py.raw.wavpack as wavpack


@pytest.fixture
def no_hard_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the POSIX removable-drive path on every operating system."""

    def unsupported_link(source: Path, output: Path) -> None:
        raise OSError(errno.ENOTSUP, "Filesystem does not support hard links")

    # Replacing this module's os reference avoids changing pathlib's platform
    # detection, so these POSIX filesystem checks also run on Windows CI.
    monkeypatch.setattr(
        wavpack, "os", SimpleNamespace(name="posix", link=unsupported_link)
    )


def test_removable_drive_publication_copies_complete_archive(
    tmp_path: Path, no_hard_links: None
) -> None:
    staged = tmp_path / "staged.wv"
    content = b"complete verified archive\x00\xff" * 128
    staged.write_bytes(content)
    output = tmp_path / "archive.wv"

    wavpack._publish(staged, output)

    assert output.read_bytes() == content
    assert staged.read_bytes() == content


def test_removable_drive_publication_never_overwrites_existing_archive(
    tmp_path: Path, no_hard_links: None
) -> None:
    staged = tmp_path / "staged.wv"
    staged.write_bytes(b"new archive")
    output = tmp_path / "archive.wv"
    output.write_bytes(b"valuable existing archive")

    with pytest.raises(FileExistsError):
        wavpack._publish(staged, output)

    assert output.read_bytes() == b"valuable existing archive"
    assert staged.read_bytes() == b"new archive"


def test_removable_drive_close_failure_removes_incomplete_archive(
    tmp_path: Path, no_hard_links: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged.wv"
    content = b"complete verified archive" * 128
    staged.write_bytes(content)
    output = tmp_path / "archive.wv"
    original_open = Path.open

    @contextmanager
    def delayed_disk_full(path: Path) -> Iterator[BinaryIO]:
        # Buffered filesystem errors can arrive during close after all write
        # calls have succeeded. Model that separately from a copy-time error.
        with original_open(path, "xb") as stream:
            yield stream
            stream.truncate(7)
        raise OSError(errno.ENOSPC, "Delayed disk-full error on close")

    def open_with_delayed_failure(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        if path == output and mode == "xb":
            return delayed_disk_full(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_delayed_failure)

    with pytest.raises(OSError, match="Delayed disk-full"):
        wavpack._publish(staged, output)

    assert not output.exists()
    assert staged.read_bytes() == content
