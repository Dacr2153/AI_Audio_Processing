"""End-to-end pipeline example using the public Python API.

Run directly (writes ``examples/demo_output.wav``)::

    python examples/api_demo.py
    # or:  python examples/api_demo.py /path/to/noisy.wav -o /tmp/clean.wav

Generate and restore a synthetic "noisy recording" by default so the example
is fully self-contained (no audio files required).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from audio_restoration import PipelineConfig, RestorationPipeline


def make_demo_input(
    path: str, sample_rate: int = 44_100, duration_s: float = 2.0
) -> str:
    """Write a synthetic stereo 'tape recording': a tone + clicks + hiss."""
    import soundfile as sf

    rng = np.random.default_rng(7)
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    signal = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    hiss = rng.normal(0.0, 0.03, size=len(t))
    signal += hiss
    for _ in range(6):
        idx = rng.integers(1000, len(t) - 1000)
        signal[idx : idx + 30] += 0.8 * np.sin(
            2 * np.pi * 2000 * np.arange(30) / sample_rate
        )
    stereo = np.column_stack([signal, signal * 0.9]).astype(np.float32)
    sf.write(path, stereo, sample_rate, subtype="PCM_16")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        help="Input audio file (creates a synthetic one if omitted).",
    )
    parser.add_argument(
        "-o", "--output", default="demo_output.wav", help="Output path."
    )
    args = parser.parse_args()

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input or make_demo_input(os.path.join(demo_dir, "demo_input.wav"))
    output_path = os.path.abspath(args.output)

    config = PipelineConfig(
        denoise_method="music",
        denoise_prop_decrease=0.7,
        enable_declicker=True,
        declicker_threshold=4.0,
        enable_ms=True,
        ms_side_gain=0.85,
        enable_wow_flutter=True,
        wow_flutter_max_cents=60.0,
        lufs_target=-14.0,
        output_sample_rate=None,
        save_comparison_plot=True,
        print_metrics=True,
    )

    report = RestorationPipeline(config).restore(input_path, output_path)
    print(f"\nDone. Report keys: {sorted(report)}")


if __name__ == "__main__":
    main()
