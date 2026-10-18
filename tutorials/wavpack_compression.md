# Compress Intan recordings with WavPack

Use this to archive standard headerless Intan `.dat` streams as smaller `.wv` files.
Compression is **lossless**: restoring an archive recreates the original file's
bytes. These functions support little-endian integer samples (`int16`, `uint16`,
`int32`, or `uint32`) with 1–4096 saved channels. Multichannel data must be
interleaved by sample, as in Intan's "One File Per Signal Type" format.

## 1. Install the tools

Install neuro_py in your Python environment:

```bash
pip install neuro-analysis-py
```

You also need the **WavPack command-line tools**; installing the Python package
does not install them. Choose your operating system:

- **Windows:** download the 64-bit command-line ZIP from the
  [official WavPack downloads page](https://www.wavpack.com/downloads.html), then
  extract it. Add the folder containing `wavpack.exe` and `wvunpack.exe` to your
  Windows `PATH`, then restart your terminal or notebook. Alternatively, pass
  the full executable paths as shown in the comments below.
- **macOS:** run `brew install wavpack` in Terminal, using
  [Homebrew](https://brew.sh/).
- **Ubuntu/Debian Linux:** run `sudo apt install wavpack` in a terminal.

Run `wavpack --version` and `wvunpack --version` in your terminal to check that
both tools are available. **For recordings with more than 256 saved channels, use
WavPack 5.6 or newer.** If your Linux package manager provides an older version,
install a current release from the
[official source downloads](https://www.wavpack.com/downloads.html).

## 2. Compress a recording

Change the path, saved channel count, and sample rate to match your recording.
The channel count is the number of channels **actually saved in this file**, not
the capacity of your acquisition system. Check your acquisition metadata, such
as `info.rhd`, if you are unsure. The values below are only examples.
For recordings with more than 256 channels, pass the real saved channel count
as usual; the function selects the extended format automatically.

Choose `dtype` for the type of file you are compressing:

| Intan file | `dtype` | `n_channels` |
| --- | --- | --- |
| `amplifier.dat` | `"int16"` (default) | Saved amplifier channel count |
| `time.dat` | `"int32"` | `1` |
| `digitalin.dat`, `digitalout.dat` | `"uint16"` | `1` (all TTL lines share one packed word) |
| `auxiliary.dat`, `supply.dat`, `analogin.dat` | `"uint16"` | Saved channel count for that file |

These settings follow the
[Intan RHD file format](https://intantech.com/files/Intan_RHD2000_data_file_formats.pdf).
For these standard files, use the amplifier sample rate: auxiliary and supply
samples are repeated in the `.dat` files to align with `time.dat`. Compress the
original integer files directly, without converting values to volts or seconds.
If another tool has resampled a file, use that file's actual stored rate instead.

```python
from neuro_py.raw import compress_dat

archive = compress_dat(
    input_path=r"D:\recordings\session1\amplifier.dat",
    n_channels=128,
    sample_rate=30_000,  # Hz
    # executable=r"C:\Tools\WavPack\wavpack.exe",  # If not on PATH
)
print(archive)  # D:\recordings\session1\amplifier.wv
```

On macOS or Linux, use your recording's path, such as
`"/data/session1/amplifier.dat"`.

Use the same function for timestamps and digital inputs, changing `dtype`:

```python
time_archive = compress_dat(
    r"D:\recordings\session1\time.dat",
    n_channels=1,
    sample_rate=30_000,
    dtype="int32",
)
digital_archive = compress_dat(
    r"D:\recordings\session1\digitalin.dat",
    n_channels=1,  # Packed digital word, even when several TTL lines are saved
    sample_rate=30_000,
    dtype="uint16",
)
```

Compress each file separately. There is no automatic folder scan or type
detection. Unsigned values are preserved byte for byte; digital bit patterns are
not unpacked or changed. Read restored files using their original Intan `dtype`;
the archive's internal PCM metadata uses signed integers even for unsigned input.

The original `.dat` stays in place. Verification is enabled by default: WavPack
stores a checksum and fully reads and decodes the archive to check the samples.
This extra verification pass takes time. Existing output files are never
overwritten. Leave enough free space for the original and archive while
compressing.

## 3. Restore the recording when needed

Restore to a different filename while your original is still present:

```python
from neuro_py.raw import decompress_dat

restored = decompress_dat(
    input_path=r"D:\recordings\session1\amplifier.wv",
    output_path=r"D:\recordings\session1\amplifier_restored.dat",
    # executable=r"C:\Tools\WavPack\wvunpack.exe",  # If not on PATH
)
print(restored)
```

Restoration works the same way for every file type and verifies the stored
checksum by default. You do not need to supply the channel count, sample rate,
or `dtype` again. If you omit `output_path`, the restored file is named
`amplifier.dat`; this fails safely if that file already exists.

You can also restore without Python using WavPack's raw-output mode:

```bash
wvunpack -r -m amplifier.wv -o amplifier_restored.dat
```

## What to keep

Keep every `.wv` archive **and the rest of the recording folder**: `info.rhd`,
tracking, behavior, and other session metadata. Keep any `.dat` streams you
have not compressed. Each archive contains only the file you passed to it, not
the rest of the session. Keep a copy of these instructions and the WavPack tools
with your archives. Before removing an original recording yourself, restore and
check your first archive and verify the copies on your destination drive.

For the function options, see the
[WavPack API reference](../reference/neuro_py/raw/wavpack.md).
