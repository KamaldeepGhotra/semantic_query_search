"""Extract AudioCaps clips from Hugging Face parquet shards into 16 kHz mono FLAC files.

Usage: python scoring/prepare_audio.py data/audiocaps_parquet/data data/youtube/audio
Skips clips already extracted, so it can be re-run as more shards finish downloading.
"""

import io
import sys
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000  # Gemma 3n audio encoder expects 16 kHz mono.


def main(parquet_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out.glob("*.flac")}
    for shard in sorted(Path(parquet_dir).glob("*.parquet")):
        table = pq.read_table(shard, columns=["youtube_id", "audio"])
        for ytid, audio in zip(table.column("youtube_id").to_pylist(), table.column("audio").to_pylist()):
            if ytid in done:
                continue
            wave, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=SR, mono=True)
            sf.write(out / f"{ytid}.flac", wave, SR)
            done.add(ytid)
        print(f"{shard.name}: {len(done)} clips total", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
