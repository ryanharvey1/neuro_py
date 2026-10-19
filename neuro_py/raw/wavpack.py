"""Lossless archives of Intan integer data streams using the WavPack CLI.

Install the external ``wavpack`` and ``wvunpack`` programs separately. Archives
are standard WavPack files, with no Python-specific container or sidecar.
"""

import errno
import hashlib
import operator
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def _positive_integer(value: int, name: str, maximum: int) -> int:
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.") from exc
    if isinstance(value, bool) or not 1 <= number <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
    return number


def _source(path: str | Path) -> Path:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise IsADirectoryError(source)
    return source


def _destination(source: Path, path: str | Path | None, suffix: str) -> Path:
    output = Path(path).absolute() if path is not None else source.with_suffix(suffix)
    # lexists also protects dangling symlinks, which Path.exists() overlooks.
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() != suffix:
        raise ValueError(f"Output must have a {suffix} extension.")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    return output


def _executable(name: str, executable: str | Path | None) -> str:
    requested = str(executable) if executable is not None else name
    found = shutil.which(requested)
    if found is None:
        raise FileNotFoundError(
            f"Cannot find {requested!r}. Install the WavPack command-line tools "
            "and add them to PATH, or pass executable= with the program's path."
        )
    return str(Path(found).resolve())


def _run(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "No diagnostic output.").strip()
        raise RuntimeError(
            f"WavPack failed (exit {result.returncode}): {detail[-8000:]}"
        )
    return result.stdout


def _signature(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _md5(path: Path) -> str:
    # MD5 is WavPack's file-integrity format, not a security/authentication check.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(staged: Path, output: Path) -> None:
    # Both operations fail rather than replace an existing destination. Windows
    # rename also supports exFAT, a common format for external archive drives.
    if os.name == "nt":
        staged.rename(output)
        return
    try:
        os.link(staged, output)
    except OSError as exc:
        if exc.errno not in {
            errno.EPERM,
            errno.ENOTSUP,
            errno.ENOSYS,
            errno.EXDEV,
        }:
            raise
        # Some removable/network filesystems do not support hard links. An
        # exclusive, bounded-memory copy still prevents overwriting other data.
        created = False
        try:
            with output.open("xb") as target:
                created = True
                with staged.open("rb") as source:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        except BaseException:
            if created:
                output.unlink(missing_ok=True)
            raise


def compress_dat(
    input_path: str | Path,
    n_channels: int,
    sample_rate: int,
    output_path: str | Path | None = None,
    *,
    dtype: str = "int16",
    executable: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Compress an Intan .dat data stream into a lossless .wv archive.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Headerless, little-endian integer data, interleaved by sample:
        channel 1 at sample 1, channel 2 at sample 1, ..., channel 1 at sample 2.
        Pass the original binary file directly, without scaling or conversion.
    n_channels : int
        Number of values per sample frame actually saved in the file (1 to
        4096). Use 1 for time.dat and bit-packed digitalin.dat/digitalout.dat,
        not the number of active digital lines.
    sample_rate : int
        Sampling rate of this stream in Hz, for example 20000 or 30000.
    output_path : str or pathlib.Path, optional
        New .wv file. Defaults to input_path with its suffix replaced by .wv.
        The parent directory must exist. Existing files are never overwritten.
    dtype : {"int16", "uint16", "int32", "uint32"}, optional
        Original little-endian sample type. Default is "int16" for amplifier
        data. Use "int32" for time.dat and "uint16" for digital, analog,
        auxiliary, and supply streams. Unsigned values are reinterpreted as
        signed integers of the same width for encoding; their bits are never
        changed. There is no scaling, offset, or sample reordering.
    executable : str or pathlib.Path, optional
        Path to wavpack (wavpack.exe on Windows). By default, search PATH.
    verify : bool, optional
        Re-read and decode the archive after writing to verify recovery using
        WavPack's MD5 check. Default is True. The original data's MD5 is stored
        even if this extra verification pass is disabled.

    Returns
    -------
    pathlib.Path
        Absolute path to the completed .wv archive.

    Raises
    ------
    ValueError
        Invalid recording parameters, empty input, incomplete sample frames,
        or an output filename without the .wv extension.
    FileNotFoundError
        Input, output directory, or WavPack executable is missing.
    FileExistsError
        The destination already exists, including the input itself.
    RuntimeError
        Compression or verification fails, or the source changes during work.

    Notes
    -----
    The source is never deleted. WavPack streams the recording using bounded
    memory; no entire-recording array is loaded into Python. A failed operation
    removes its temporary output. The archive stores channel count, sample rate,
    bit depth, and MD5. Keep info.rhd, session metadata, and each stream's dtype:
    WavPack stores unsigned streams as signed bit patterns; interpretation still
    uses the original Intan metadata. Compress each .dat stream separately.
    Structured spike.dat and .rhd/.rhs containers are not fixed-width streams
    supported by this function. Lossy and hybrid modes are not exposed.
    """
    n_channels = _positive_integer(n_channels, "n_channels", 4096)
    sample_rate = _positive_integer(sample_rate, "sample_rate", 1_000_000_000)
    bit_depths = {"int16": 16, "uint16": 16, "int32": 32, "uint32": 32}
    if dtype not in bit_depths:
        raise ValueError("dtype must be int16, uint16, int32, or uint32.")
    bits = bit_depths[dtype]
    bytes_per_frame = n_channels * (bits // 8)
    source = _source(input_path)
    output = _destination(source, output_path, ".wv")
    before = _signature(source)
    size = before[2]
    if size == 0 or size % bytes_per_frame:
        raise ValueError(
            "Input must contain complete, nonempty integer channel frames."
        )
    if size // bytes_per_frame > (1 << 40) - 257:
        raise ValueError("Recording exceeds WavPack's maximum sample count.")
    program = _executable("wavpack", executable)
    with tempfile.TemporaryDirectory(
        prefix=".wavpack-", dir=output.parent
    ) as directory:
        staged = Path(directory) / output.name
        arguments = [
            program,
            # Extended raw mode bypasses the CLI's conservative 256-channel
            # default while retaining the format's real channel count.
            f"--raw-pcm{'-ex' if n_channels > 256 else ''}="
            f"{sample_rate},{bits}s,{n_channels},le",
            "--channel-order=...",
            "--pair-unassigned-chans",
            "-m",
            "-q",
        ]
        if verify:
            arguments.append("-v")
        arguments.extend([str(source), "-o", str(staged)])
        _run(arguments)
        if _signature(source) != before:
            raise RuntimeError(
                "Source changed during compression; archive not published."
            )
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError("WavPack did not create a nonempty archive.")
        _publish(staged, output)
    return output


def decompress_dat(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    executable: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Restore an Intan .dat recording from a standard lossless WavPack archive.

    Parameters
    ----------
    input_path : str or pathlib.Path
        A lossless .wv archive of 16- or 32-bit integer samples, such as one
        created by compress_dat. Hybrid/lossy and other bit depths are rejected.
    output_path : str or pathlib.Path, optional
        New .dat file. Defaults to input_path with its suffix replaced by .dat.
        Use a different filename if the original .dat is still present. The
        parent directory must exist. Existing files are never overwritten.
    executable : str or pathlib.Path, optional
        Path to wvunpack (wvunpack.exe on Windows). By default, search PATH.
    verify : bool, optional
        Check restored bytes against the MD5 stored in the archive. Default
        is True, which requires an embedded MD5 (always stored by compress_dat).
        WavPack's block checks remain active when False.

    Returns
    -------
    pathlib.Path
        Absolute path to the restored, headerless .dat, with the original bytes.
        No dtype, channel count, or rate is needed for restoration.

    Raises
    ------
    ValueError
        The output filename does not have the .dat extension.
    FileNotFoundError
        Input, output directory, or WvUnpack executable is missing.
    FileExistsError
        The destination already exists, including the archive itself.
    RuntimeError
        The archive is unsupported, corrupt, changed during work, lacks an MD5
        required for verification, or cannot be completely decoded.

    Notes
    -----
    The archive is never deleted. Raw-output mode prevents insertion of a WAV
    header. Decoding streams through WvUnpack using bounded memory. Temporary
    output is cleaned on failure; completed output is never overwritten.
    """
    source = _source(input_path)
    output = _destination(source, output_path, ".dat")
    program = _executable("wvunpack", executable)
    before = _signature(source)
    fields = _run([program, "-f", str(source)]).strip().split(";", maxsplit=9)
    try:
        bits, channels, samples = int(fields[1]), int(fields[3]), int(fields[5])
        mode = int(fields[8], 0)
        supported = (
            bits in (16, 32)
            and fields[2] == "int"
            and 1 <= channels <= 4096
            and samples > 0
            and bool(mode & 0x2)  # MODE_LOSSLESS, from WavPack's public API.
            and not mode & 0x4  # MODE_HYBRID: reject even with a correction file.
        )
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Cannot read WavPack recording metadata.") from exc
    if not supported:
        raise RuntimeError(
            "Archive must contain lossless, non-hybrid 16/32-bit integers."
        )
    if verify and re.fullmatch(r"[0-9a-fA-F]{32}", fields[6]) is None:
        raise RuntimeError("Archive has no stored MD5; cannot verify exact recovery.")
    with tempfile.TemporaryDirectory(
        prefix=".wavpack-", dir=output.parent
    ) as directory:
        staged = Path(directory) / output.name
        arguments = [program, "-r", "-q"]
        if verify:
            arguments.append("-m")
        arguments.extend([str(source), "-o", str(staged)])
        _run(arguments)
        if _signature(source) != before:
            raise RuntimeError(
                "Archive changed during decompression; output not published."
            )
        expected_size = samples * channels * (bits // 8)
        if not staged.is_file() or staged.stat().st_size != expected_size:
            raise RuntimeError(
                "Restored recording size does not match archive metadata."
            )
        # Native -m can report a mismatch without failing when an external file
        # was encoded with different byte order, signedness, or channel order.
        # Comparing the raw output itself makes exact recovery unambiguous.
        if verify and _md5(staged) != fields[6].lower():
            raise RuntimeError(
                "Restored recording MD5 does not match the original bytes."
            )
        _publish(staged, output)
    return output
