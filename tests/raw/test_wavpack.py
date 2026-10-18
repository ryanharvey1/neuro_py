"""Safety checks and real-codec lossless tests for Intan raw archives."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neuro_py.raw.wavpack import compress_dat, decompress_dat


@pytest.fixture(scope="module")
def wavpack_tools() -> tuple[str, str]:
    """Use actual codecs; dedicated CI must fail rather than silently skip."""
    encoder = shutil.which("wavpack")
    decoder = shutil.which("wvunpack")
    if encoder is None or decoder is None:
        message = "The wavpack and wvunpack command-line tools are required."
        if os.environ.get("NEURO_PY_REQUIRE_WAVPACK") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return encoder, decoder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_recording(
    path: Path, kind: str, n_channels: int, n_samples: int = 2048
) -> bytes:
    """Write sample-major, little-endian signed 16-bit Intan frames."""
    rng = np.random.default_rng(4319)
    if kind == "extrema":
        values = np.resize(
            np.array([-32768, 32767, -1, 0, 1, -256, 256], dtype="<i2"),
            n_samples * n_channels,
        )
    elif kind == "zeros":
        values = np.zeros((n_samples, n_channels), dtype="<i2")
    elif kind == "noise":
        values = rng.integers(
            -32768, 32768, size=(n_samples, n_channels), dtype=np.int16
        ).astype("<i2")
    else:
        # Smooth, shared signal with distinct channel offsets and small noise.
        signal = 12000 * np.sin(np.arange(n_samples) / 17)
        values = (
            signal[:, None]
            + np.arange(n_channels)[None, :] * 7
            + rng.integers(-20, 21, size=(n_samples, n_channels))
        ).astype("<i2")
    content = values.tobytes(order="C")
    path.write_bytes(content)
    return content


@pytest.mark.parametrize(
    ("kind", "n_channels", "sample_rate"),
    [
        ("extrema", 1, 20000),
        ("extrema", 3, 30000),
        ("zeros", 4, 30000),
        ("noise", 7, 25000),
        ("correlated", 256, 30000),
    ],
)
@pytest.mark.parametrize("verify", [True, False])
def test_real_codec_preserves_every_byte(
    tmp_path: Path,
    wavpack_tools: tuple[str, str],
    kind: str,
    n_channels: int,
    sample_rate: int,
    verify: bool,
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / "amplifier.dat"
    original = _write_recording(source, kind, n_channels)
    original_hash = _sha256(source)
    archive = compress_dat(
        source,
        n_channels,
        sample_rate,
        executable=encoder,
        verify=verify,
    )
    assert archive == source.with_suffix(".wv")
    assert archive.is_file()
    if kind == "zeros":
        assert archive.stat().st_size < source.stat().st_size
    assert source.read_bytes() == original
    archive_hash = _sha256(archive)
    metadata = (
        subprocess.run(
            [decoder, "-f", str(archive)], check=True, capture_output=True, text=True
        )
        .stdout.strip()
        .split(";")
    )
    assert int(metadata[0]) == sample_rate
    assert int(metadata[1]) == 16
    assert int(metadata[3]) == n_channels
    assert int(metadata[5]) == 2048

    destination = tmp_path / "restored.dat"
    restored = decompress_dat(archive, destination, executable=decoder, verify=verify)
    assert restored == destination
    assert restored.read_bytes() == original
    assert _sha256(restored) == original_hash
    assert _sha256(archive) == archive_hash
    assert source.read_bytes() == original


def test_default_restore_path_and_unicode_paths(
    tmp_path: Path, wavpack_tools: tuple[str, str]
) -> None:
    encoder, decoder = wavpack_tools
    recording_dir = tmp_path / "session with spaces \u03b1 \u9f20"
    recording_dir.mkdir()
    source = recording_dir / "amplifier recording.dat"
    original = _write_recording(source, "extrema", 3)
    # Exercise both string inputs and explicit archive destinations.
    archive_path = recording_dir / "archive with spaces \u03b2.wv"
    archive = compress_dat(str(source), 3, 30000, str(archive_path), executable=encoder)
    restored = decompress_dat(str(archive), executable=decoder)
    assert restored == archive_path.with_suffix(".dat")
    assert source.read_bytes() == original
    assert restored.read_bytes() == original


def test_compression_functions_are_exported_from_raw() -> None:
    from neuro_py import raw

    assert raw.compress_dat is compress_dat
    assert raw.decompress_dat is decompress_dat


@pytest.mark.parametrize("n_channels", [257, 384, 512, 1024, 4096])
def test_large_channel_counts_restore_exactly(
    tmp_path: Path, wavpack_tools: tuple[str, str], n_channels: int
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / "amplifier.dat"
    original = _write_recording(source, "noise", n_channels, n_samples=64)
    original_hash = _sha256(source)
    archive = compress_dat(source, n_channels, 30000, executable=encoder)
    metadata = (
        subprocess.run(
            [decoder, "-f", str(archive)], check=True, capture_output=True, text=True
        )
        .stdout.strip()
        .split(";")
    )
    assert int(metadata[0]) == 30000
    assert int(metadata[1]) == 16
    assert int(metadata[3]) == n_channels
    assert int(metadata[5]) == 64
    restored = decompress_dat(archive, tmp_path / "restored.dat", executable=decoder)
    assert restored.read_bytes() == original
    assert _sha256(restored) == original_hash
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    ("filename", "dtype", "n_channels", "values"),
    [
        ("amplifier.dat", "int16", 3, [-32768, -1, 0, 1, 32767]),
        ("time.dat", "int32", 1, [-2147483648, -1, 0, 1, 2147483647]),
        ("digitalin.dat", "uint16", 1, [0, 1, 3, 32768, 65535, 21845, 43690]),
        ("digitalout.dat", "uint16", 1, [0, 1, 2, 32767, 32768, 65535]),
        ("analogin.dat", "uint16", 3, [0, 1, 32767, 32768, 65535]),
        ("auxiliary.dat", "uint16", 3, [0, 1, 32767, 32768, 65535]),
        ("supply.dat", "uint16", 2, [0, 1, 32767, 32768, 65535]),
        ("unsigned32.dat", "uint32", 1, [0, 1, 2147483647, 2147483648, 4294967295]),
    ],
    ids=[
        "amplifier-int16",
        "timestamps-int32",
        "packed-digital-input-uint16",
        "packed-digital-output-uint16",
        "analog-input-uint16",
        "auxiliary-uint16",
        "supply-uint16",
        "uint32-high-bit",
    ],
)
def test_all_intan_dat_streams_restore_exact_bytes(
    tmp_path: Path,
    wavpack_tools: tuple[str, str],
    filename: str,
    dtype: str,
    n_channels: int,
    values: list[int],
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / filename
    samples = np.resize(np.array(values), 2048 * n_channels)
    original = samples.astype(np.dtype(dtype).newbyteorder("<")).tobytes()
    source.write_bytes(original)
    original_hash = _sha256(source)
    archive = compress_dat(source, n_channels, 30000, dtype=dtype, executable=encoder)
    metadata = (
        subprocess.run(
            [decoder, "-f", str(archive)], check=True, capture_output=True, text=True
        )
        .stdout.strip()
        .split(";")
    )
    assert int(metadata[0]) == 30000
    assert int(metadata[1]) == np.dtype(dtype).itemsize * 8
    assert int(metadata[3]) == n_channels
    assert int(metadata[5]) == 2048
    restored = decompress_dat(archive, tmp_path / "restored.dat", executable=decoder)
    assert restored.read_bytes() == original
    assert _sha256(restored) == original_hash
    assert source.read_bytes() == original


@pytest.mark.parametrize("dtype", ["int8", "uint8", "float32", "float64", "int64"])
def test_rejects_unsupported_data_types(tmp_path: Path, dtype: str) -> None:
    source = tmp_path / "amplifier.dat"
    source.write_bytes(b"\x00" * 48)
    with pytest.raises(ValueError):
        compress_dat(source, 3, 30000, dtype=dtype, executable=sys.executable)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["amplifier.dat"]


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_rejects_incomplete_32_bit_frames(tmp_path: Path, dtype: str) -> None:
    source = tmp_path / "time.dat"
    source.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError):
        compress_dat(source, 1, 30000, dtype=dtype, executable=sys.executable)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["time.dat"]


@pytest.mark.parametrize("argument", [0, -1, True, 1.5, "4"])
@pytest.mark.parametrize("parameter", ["n_channels", "sample_rate"])
def test_rejects_invalid_acquisition_parameters(
    tmp_path: Path, parameter: str, argument: object
) -> None:
    source = tmp_path / "amplifier.dat"
    source.write_bytes(b"\x00\x00" * 12)
    parameters = {"n_channels": 3, "sample_rate": 30000}
    parameters[parameter] = argument
    with pytest.raises(ValueError):
        compress_dat(source, **parameters, executable=sys.executable)
    assert source.read_bytes() == b"\x00\x00" * 12
    assert sorted(path.name for path in tmp_path.iterdir()) == ["amplifier.dat"]


@pytest.mark.parametrize("content", [b"", b"\x00", b"\x00\x00" * 5])
def test_rejects_empty_or_incomplete_frames(tmp_path: Path, content: bytes) -> None:
    source = tmp_path / "amplifier.dat"
    source.write_bytes(content)
    with pytest.raises(ValueError):
        compress_dat(source, 3, 30000, executable=sys.executable)
    assert source.read_bytes() == content
    assert sorted(path.name for path in tmp_path.iterdir()) == ["amplifier.dat"]


@pytest.mark.parametrize(
    ("n_channels", "sample_rate"), [(4097, 30000), (1, 1_000_000_001)]
)
def test_rejects_values_outside_codec_limits(
    tmp_path: Path, n_channels: int, sample_rate: int
) -> None:
    source = tmp_path / "amplifier.dat"
    source.write_bytes(b"\x00\x00" * 257)
    with pytest.raises(ValueError):
        compress_dat(source, n_channels, sample_rate, executable=sys.executable)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["amplifier.dat"]


@pytest.mark.parametrize("operation", ["compress", "decompress"])
@pytest.mark.parametrize("same_path", [True, False])
def test_never_overwrites_an_existing_file(
    tmp_path: Path, operation: str, same_path: bool
) -> None:
    source = tmp_path / ("amplifier.dat" if operation == "compress" else "amplifier.wv")
    source.write_bytes(b"\x00\x00" * 12)
    output = source if same_path else tmp_path / "already exists.dat"
    if not same_path:
        output.write_bytes(b"valuable existing output")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        if operation == "compress":
            compress_dat(source, 3, 30000, output, executable=sys.executable)
        else:
            decompress_dat(source, output, executable=sys.executable)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_default_restore_refuses_to_overwrite_original(
    tmp_path: Path, wavpack_tools: tuple[str, str]
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / "amplifier.dat"
    original = _write_recording(source, "correlated", 3)
    archive = compress_dat(source, 3, 30000, executable=encoder)
    with pytest.raises(FileExistsError):
        decompress_dat(archive, executable=decoder)
    assert source.read_bytes() == original


@pytest.mark.parametrize("operation", ["compress", "decompress"])
def test_missing_source(tmp_path: Path, operation: str) -> None:
    source = tmp_path / "missing.dat"
    with pytest.raises(FileNotFoundError):
        if operation == "compress":
            compress_dat(source, 3, 30000, executable=sys.executable)
        else:
            decompress_dat(source, executable=sys.executable)


@pytest.mark.parametrize("operation", ["compress", "decompress"])
def test_missing_executable(tmp_path: Path, operation: str) -> None:
    source = tmp_path / ("amplifier.dat" if operation == "compress" else "amplifier.wv")
    source.write_bytes(b"\x00\x00" * 12)
    missing = tmp_path / "missing-codec"
    with pytest.raises(FileNotFoundError):
        if operation == "compress":
            compress_dat(source, 3, 30000, executable=missing)
        else:
            decompress_dat(source, executable=missing)


@pytest.mark.parametrize("corruption", ["truncated", "invalid_header", "damaged_audio"])
def test_corrupt_archive_never_publishes_a_restored_file(
    tmp_path: Path, wavpack_tools: tuple[str, str], corruption: str
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / "amplifier.dat"
    original = _write_recording(source, "noise", 7)
    archive = compress_dat(source, 7, 30000, executable=encoder)
    content = bytearray(archive.read_bytes())
    if corruption == "truncated":
        content = content[: len(content) // 2]
    elif corruption == "invalid_header":
        content[:4] = b"xxxx"
    else:
        content[len(content) // 2] ^= 0xFF
    archive.write_bytes(content)
    before = {path.name for path in tmp_path.iterdir()}
    restored = tmp_path / "restored.dat"
    with pytest.raises(RuntimeError):
        decompress_dat(archive, restored, executable=decoder)
    assert not restored.exists()
    assert {path.name for path in tmp_path.iterdir()} == before
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    ("raw_format", "options"),
    [
        ("30000,8u,1,le", []),
        ("30000,24s,1,le", []),
        ("30000,16s,3,le", ["-b2"]),
    ],
    ids=["8-bit", "24-bit", "hybrid-lossy"],
)
def test_rejects_other_sample_formats_and_lossy_archives(
    tmp_path: Path,
    wavpack_tools: tuple[str, str],
    raw_format: str,
    options: list[str],
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / "amplifier.dat"
    _write_recording(source, "noise", 3)
    archive = tmp_path / "unsupported.wv"
    subprocess.run(
        [
            encoder,
            "-q",
            "-m",
            f"--raw-pcm={raw_format}",
            *options,
            str(source),
            "-o",
            str(archive),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    before = {path.name for path in tmp_path.iterdir()}
    with pytest.raises(RuntimeError):
        decompress_dat(archive, tmp_path / "restored.dat", executable=decoder)
    assert {path.name for path in tmp_path.iterdir()} == before


def test_verification_requires_stored_checksum(
    tmp_path: Path, wavpack_tools: tuple[str, str]
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / "amplifier.dat"
    original = _write_recording(source, "extrema", 3)
    archive = tmp_path / "no-checksum.wv"
    subprocess.run(
        [encoder, "-q", "--raw-pcm=30000,16s,3,le", str(source), "-o", str(archive)],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    restored = tmp_path / "restored.dat"
    with pytest.raises(RuntimeError):
        decompress_dat(archive, restored, executable=decoder)
    assert not restored.exists()
    # Explicitly disabling verification also supports externally made archives.
    assert (
        decompress_dat(archive, restored, executable=decoder, verify=False).read_bytes()
        == original
    )


@pytest.mark.parametrize("raw_format", ["30000,16s,1,be", "30000,16u,1,le"])
def test_verification_detects_external_pcm_byte_conversion(
    tmp_path: Path, wavpack_tools: tuple[str, str], raw_format: str
) -> None:
    encoder, decoder = wavpack_tools
    source = tmp_path / "external-format.dat"
    if raw_format.endswith("be"):
        values = np.resize(np.array([-32768, 32767, -1, 0, 1]), 2048)
        source.write_bytes(values.astype(">i2").tobytes())
        expected_little_endian = values.astype("<i2").tobytes()
    else:
        values = np.resize(np.array([0, 65535, 32768, 32767, 1]), 2048)
        source.write_bytes(values.astype("<u2").tobytes())
        expected_little_endian = (values - 32768).astype("<i2").tobytes()
    archive = tmp_path / "external-format.wv"
    subprocess.run(
        [
            encoder,
            "-q",
            "-m",
            f"--raw-pcm={raw_format}",
            str(source),
            "-o",
            str(archive),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    restored = tmp_path / "restored.dat"
    before = {path.name for path in tmp_path.iterdir()}
    # WvUnpack raw output is signed little-endian. An archive created from a
    # different PCM layout must not be claimed to restore the original bytes.
    with pytest.raises(RuntimeError):
        decompress_dat(archive, restored, executable=decoder)
    assert {path.name for path in tmp_path.iterdir()} == before
    assert (
        decompress_dat(archive, restored, executable=decoder, verify=False).read_bytes()
        == expected_little_endian
    )


def test_subprocess_failure_does_not_publish_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "amplifier.dat"
    source.write_bytes(b"\x00\x00" * 12)
    output = tmp_path / "amplifier.wv"

    def fail_codec(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert isinstance(command, list)
        assert kwargs.get("shell", False) is False
        assert kwargs["stdin"] == subprocess.DEVNULL
        Path(command[command.index("-o") + 1]).write_bytes(b"incomplete archive")
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="simulated encoder failure"
        )

    monkeypatch.setattr(subprocess, "run", fail_codec)
    with pytest.raises(RuntimeError, match="simulated encoder failure"):
        compress_dat(source, 3, 30000, output, executable=sys.executable)
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["amplifier.dat"]


def test_changing_source_does_not_publish_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "amplifier.dat"
    source.write_bytes(b"\x00\x00" * 12)
    output = tmp_path / "amplifier.wv"

    def modify_during_compression(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        Path(command[command.index("-o") + 1]).write_bytes(b"staged archive")
        with source.open("ab") as stream:
            stream.write(b"\x00\x00" * 3)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", modify_during_compression)
    with pytest.raises(RuntimeError, match="changed"):
        compress_dat(source, 3, 30000, output, executable=sys.executable)
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["amplifier.dat"]


def test_file_created_during_compression_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "amplifier.dat"
    source.write_bytes(b"\x00\x00" * 12)
    output = tmp_path / "amplifier.wv"

    def create_competing_output(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        Path(command[command.index("-o") + 1]).write_bytes(b"new archive")
        output.write_bytes(b"another process created this")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", create_competing_output)
    with pytest.raises(FileExistsError):
        compress_dat(source, 3, 30000, output, executable=sys.executable)
    assert output.read_bytes() == b"another process created this"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "amplifier.dat",
        "amplifier.wv",
    ]
