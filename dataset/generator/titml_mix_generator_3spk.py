# -*- coding: utf-8 -*-
"""TITML-IDN 3-Speaker Dataset Generator for Local Environment.

This script generates synthetic 3-speaker mixtures from the TITML-IDN dataset
for speech separation training. Optimized for local machine execution.
"""

import os
import sys
import random
import numpy as np
import soundfile as sf
import librosa
from librosa import effects as librosa_effects
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json
import argparse
from datetime import datetime

# Make project root importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.paths import get_raw_dir, get_synthetic_dir


class TITMLMixGenerator3Spk:
    """Generator for 3-speaker speech separation dataset."""

    def __init__(self, titml_dir, output_dir, target_sr=16000, seed=42):
        self.titml_dir = Path(titml_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr

        random.seed(seed)
        np.random.seed(seed)

        print("=" * 60)
        print("TITML-IDN 3-Speaker Speech Separation Dataset Generator")
        print("=" * 60)

        self.speakers = self._collect_speakers()
        self._print_statistics()

    def _collect_speakers(self):
        """Collect audio files organized by speaker."""
        speakers = defaultdict(list)
        speech_dir = self.titml_dir / "Speech"

        if not speech_dir.exists():
            raise FileNotFoundError(f"Speech directory not found: {speech_dir}")

        # TITML structure: Speech/[speaker_id]/[speaker_id]-[sentence_id].wav
        for speaker_dir in sorted(speech_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name
            audio_files = sorted(speaker_dir.glob("*.wav"))

            if len(audio_files) > 0:
                speakers[speaker_id] = audio_files

        return speakers

    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\nDataset Statistics:")
        print(f"  Total speakers: {len(self.speakers)}")

        male_speakers = [s for s in self.speakers.keys() if s.startswith('m')]
        female_speakers = [s for s in self.speakers.keys() if s.startswith('f')]

        print(f"  Male speakers: {len(male_speakers)}")
        print(f"  Female speakers: {len(female_speakers)}")

        total_utterances = sum(len(files) for files in self.speakers.values())
        avg_utterances = total_utterances / len(self.speakers)

        print(f"  Total utterances: {total_utterances}")
        print(f"  Avg utterances per speaker: {avg_utterances:.1f}")

        print(f"\nSample speakers:")
        for speaker_id, files in list(self.speakers.items())[:5]:
            print(f"  {speaker_id}: {len(files)} files")

    def load_audio(self, file_path, target_duration=None):
        """Load and preprocess audio."""
        try:
            audio, sr = sf.read(file_path)

            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Trim silence
            audio, _ = librosa_effects.trim(y=audio, top_db=30)

            # Skip if too short (less than 1 second)
            if len(audio) < self.target_sr:
                return None

            # Normalize to [-0.9, 0.9]
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9

            # Cut or pad to specific duration
            if target_duration:
                target_len = int(target_duration * self.target_sr)
                if len(audio) > target_len:
                    start = random.randint(0, len(audio) - target_len)
                    audio = audio[start:start + target_len]
                elif len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))

            return audio

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def mix_three_sources(self, audio1, audio2, audio3, snr_range=(-5, 5)):
        """
        Mix three audio sources with random SNR and time offsets.

        Returns:
            mixture, source1, source2, source3 (all same length, normalized)
        """
        max_offset = int(1.0 * self.target_sr)

        # Apply random time offsets to spk2 and spk3
        offset2 = random.randint(0, max_offset)
        offset3 = random.randint(0, max_offset)

        audio2_shifted = np.pad(audio2, (offset2, 0))
        audio3_shifted = np.pad(audio3, (offset3, 0))

        # Align all three to the same length
        use_min_length = random.choice([True, False])
        lengths = [len(audio1), len(audio2_shifted), len(audio3_shifted)]

        if use_min_length:
            target_len = min(lengths)
        else:
            target_len = max(lengths)

        def fit_audio(a, length):
            if len(a) > length:
                return a[:length]
            elif len(a) < length:
                return np.pad(a, (0, length - len(a)))
            return a

        audio1 = fit_audio(audio1, target_len)
        audio2_shifted = fit_audio(audio2_shifted, target_len)
        audio3_shifted = fit_audio(audio3_shifted, target_len)

        # Scale spk2 and spk3 relative to spk1 using random SNR
        snr2_db = random.uniform(*snr_range)
        snr3_db = random.uniform(*snr_range)

        audio1_power = np.mean(audio1 ** 2) + 1e-10
        audio2_power = np.mean(audio2_shifted ** 2) + 1e-10
        audio3_power = np.mean(audio3_shifted ** 2) + 1e-10

        scale2 = np.sqrt(audio1_power / (audio2_power * 10 ** (snr2_db / 10)))
        scale3 = np.sqrt(audio1_power / (audio3_power * 10 ** (snr3_db / 10)))

        audio2_scaled = audio2_shifted * scale2
        audio3_scaled = audio3_shifted * scale3

        # Mix all three
        mixture = audio1 + audio2_scaled + audio3_scaled

        # Normalize mixture to prevent clipping; apply same scale to sources
        max_val = np.max(np.abs(mixture))
        if max_val > 1.0:
            scale_factor = 0.9 / max_val
            mixture = mixture * scale_factor
            audio1 = audio1 * scale_factor
            audio2_scaled = audio2_scaled * scale_factor
            audio3_scaled = audio3_scaled * scale_factor

        return mixture, audio1, audio2_scaled, audio3_scaled

    def split_speakers(self, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
        """
        Split speakers into train/dev/test sets (speaker-independent).
        Requires at least 3 speakers per split for 3-speaker mixing.
        """
        speaker_ids = list(self.speakers.keys())
        total = len(speaker_ids)

        # Ensure minimum 3 speakers per split
        min_speakers = 3
        if total < min_speakers * 3:
            raise ValueError(f"Need at least {min_speakers * 3} speakers, got {total}")

        # Calculate split sizes ensuring minimum 3 per split
        n_test = max(min_speakers, int(total * test_ratio))
        n_dev = max(min_speakers, int(total * dev_ratio))
        n_train = total - n_dev - n_test

        # Ensure train has enough speakers too
        if n_train < min_speakers:
            # Adjust by taking from test first, then dev
            deficit = min_speakers - n_train
            if n_test > min_speakers:
                take_from_test = min(deficit, n_test - min_speakers)
                n_test -= take_from_test
                n_train += take_from_test
                deficit -= take_from_test
            if deficit > 0 and n_dev > min_speakers:
                take_from_dev = min(deficit, n_dev - min_speakers)
                n_dev -= take_from_dev
                n_train += take_from_dev

        # Shuffle and split
        random.shuffle(speaker_ids)
        train_speakers = speaker_ids[:n_train]
        dev_speakers = speaker_ids[n_train:n_train + n_dev]
        test_speakers = speaker_ids[n_train + n_dev:]

        print(f"\nSpeaker Split (Speaker-Independent, 3-Speaker):")
        print(f"  Train: {len(train_speakers)} speakers {sorted(train_speakers)}")
        print(f"  Dev:   {len(dev_speakers)} speakers {sorted(dev_speakers)}")
        print(f"  Test:  {len(test_speakers)} speakers {sorted(test_speakers)}")

        return train_speakers, dev_speakers, test_speakers

    def split_utterances(self, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
        """
        Split utterances per speaker into train/dev/test sets.
        All 20 speakers appear in every split, but with non-overlapping
        utterances. This avoids the speaker-mismatch generalisation gap
        that occurs with only 20 source speakers.

        Returns:
            Three dicts mapping speaker_id -> list of audio file paths.
        """
        train_utts = defaultdict(list)
        dev_utts = defaultdict(list)
        test_utts = defaultdict(list)

        for speaker_id, files in self.speakers.items():
            shuffled = list(files)
            random.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_dev = int(n * dev_ratio)
            # rest goes to test

            train_utts[speaker_id] = shuffled[:n_train]
            dev_utts[speaker_id] = shuffled[n_train:n_train + n_dev]
            test_utts[speaker_id] = shuffled[n_train + n_dev:]

        # Summary
        total_train = sum(len(v) for v in train_utts.values())
        total_dev = sum(len(v) for v in dev_utts.values())
        total_test = sum(len(v) for v in test_utts.values())
        print(f"\nUtterance Split (all {len(self.speakers)} speakers in every split):")
        print(f"  Train: {total_train} utterances across {len(train_utts)} speakers")
        print(f"  Dev:   {total_dev} utterances across {len(dev_utts)} speakers")
        print(f"  Test:  {total_test} utterances across {len(test_utts)} speakers")

        return train_utts, dev_utts, test_utts

    def generate_mixtures_from_utterances(self, utterances_by_speaker, split_name,
                                          num_mixtures, target_duration=5.0,
                                          gender_balance=True):
        """
        Generate 3-speaker mixtures drawing only from the given
        per-speaker utterance lists (utterance-level split).
        """
        speaker_list = [sid for sid, utts in utterances_by_speaker.items() if len(utts) > 0]
        if len(speaker_list) < 3:
            raise ValueError(f"Need at least 3 speakers with utterances, got {len(speaker_list)}")

        output_split = self.output_dir / split_name

        for subdir in ('mix', 's1', 's2', 's3'):
            (output_split / subdir).mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {num_mixtures} 3-speaker mixtures for '{split_name}' split...")
        print(f"   Using {len(speaker_list)} speakers")
        print(f"   Target duration: {target_duration}s")

        males = [s for s in speaker_list if s.startswith('m')]
        females = [s for s in speaker_list if s.startswith('f')]

        success_count = 0
        attempt = 0
        max_attempts = num_mixtures * 5

        pbar = tqdm(total=num_mixtures, desc=f"{split_name}")

        while success_count < num_mixtures and attempt < max_attempts:
            attempt += 1

            # Speaker selection (same gender-balance logic)
            if gender_balance and len(males) >= 1 and len(females) >= 1:
                r = random.random()
                if r < 0.5 and len(males) >= 2 and len(females) >= 1:
                    chosen = random.sample(males, 2) + random.sample(females, 1)
                elif r < 0.8 and len(females) >= 2 and len(males) >= 1:
                    chosen = random.sample(males, 1) + random.sample(females, 2)
                else:
                    chosen = random.sample(speaker_list, 3)
            else:
                chosen = random.sample(speaker_list, 3)

            spk1, spk2, spk3 = chosen

            # Pick utterances from the split-specific pool
            audio1 = self.load_audio(random.choice(utterances_by_speaker[spk1]),
                                     target_duration=target_duration)
            audio2 = self.load_audio(random.choice(utterances_by_speaker[spk2]),
                                     target_duration=target_duration)
            audio3 = self.load_audio(random.choice(utterances_by_speaker[spk3]),
                                     target_duration=target_duration)

            if audio1 is None or audio2 is None or audio3 is None:
                continue

            try:
                mixture, src1, src2, src3 = self.mix_three_sources(audio1, audio2, audio3)
            except Exception as e:
                print(f"Error mixing: {e}")
                continue

            filename = f"{split_name}_{success_count:05d}.wav"
            sf.write(output_split / 'mix' / filename, mixture, self.target_sr)
            sf.write(output_split / 's1' / filename, src1, self.target_sr)
            sf.write(output_split / 's2' / filename, src2, self.target_sr)
            sf.write(output_split / 's3' / filename, src3, self.target_sr)

            success_count += 1
            pbar.update(1)

        pbar.close()

        if success_count < num_mixtures:
            print(f"Warning: Generated {success_count}/{num_mixtures} mixtures")
        else:
            print(f"Generated {success_count} mixtures")

        self._save_metadata(output_split, success_count, speaker_list, target_duration)
        return success_count

    def generate_mixtures(self, speaker_list, split_name, num_mixtures,
                          target_duration=5.0, gender_balance=True):
        """
        Generate 3-speaker mixed speech dataset.

        Args:
            speaker_list: List of speaker IDs to draw from
            split_name: 'train', 'dev', or 'test'
            num_mixtures: Number of mixtures to generate
            target_duration: Fixed clip length in seconds
            gender_balance: Favour gender-diverse triplets when possible
        """
        if len(speaker_list) < 3:
            raise ValueError(f"Need at least 3 speakers, got {len(speaker_list)}")

        output_split = self.output_dir / split_name

        # Create output directories
        for subdir in ('mix', 's1', 's2', 's3'):
            (output_split / subdir).mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {num_mixtures} 3-speaker mixtures for '{split_name}' split...")
        print(f"   Using {len(speaker_list)} speakers")
        print(f"   Target duration: {target_duration}s")

        males = [s for s in speaker_list if s.startswith('m')]
        females = [s for s in speaker_list if s.startswith('f')]

        success_count = 0
        attempt = 0
        max_attempts = num_mixtures * 5

        pbar = tqdm(total=num_mixtures, desc=f"{split_name}")

        while success_count < num_mixtures and attempt < max_attempts:
            attempt += 1

            # Speaker selection
            if gender_balance and len(males) >= 1 and len(females) >= 1:
                r = random.random()
                if r < 0.5 and len(males) >= 2 and len(females) >= 1:
                    # 2 males + 1 female
                    chosen = random.sample(males, 2) + random.sample(females, 1)
                elif r < 0.8 and len(females) >= 2 and len(males) >= 1:
                    # 1 male + 2 females
                    chosen = random.sample(males, 1) + random.sample(females, 2)
                else:
                    # Fully random (fallback / same-gender)
                    chosen = random.sample(speaker_list, 3)
            else:
                chosen = random.sample(speaker_list, 3)

            spk1, spk2, spk3 = chosen

            # Load audio
            audio1 = self.load_audio(random.choice(self.speakers[spk1]), target_duration=target_duration)
            audio2 = self.load_audio(random.choice(self.speakers[spk2]), target_duration=target_duration)
            audio3 = self.load_audio(random.choice(self.speakers[spk3]), target_duration=target_duration)

            if audio1 is None or audio2 is None or audio3 is None:
                continue

            # Mix
            try:
                mixture, src1, src2, src3 = self.mix_three_sources(audio1, audio2, audio3)
            except Exception as e:
                print(f"Error mixing: {e}")
                continue

            # Save
            filename = f"{split_name}_{success_count:05d}.wav"
            sf.write(output_split / 'mix' / filename, mixture, self.target_sr)
            sf.write(output_split / 's1' / filename, src1, self.target_sr)
            sf.write(output_split / 's2' / filename, src2, self.target_sr)
            sf.write(output_split / 's3' / filename, src3, self.target_sr)

            success_count += 1
            pbar.update(1)

        pbar.close()

        if success_count < num_mixtures:
            print(f"Warning: Generated {success_count}/{num_mixtures} mixtures")
        else:
            print(f"Generated {success_count} mixtures")

        self._save_metadata(output_split, success_count, speaker_list, target_duration)

        return success_count

    def _save_metadata(self, output_path, num_mixtures, speakers, target_duration):
        """Save dataset metadata."""
        metadata = {
            'num_mixtures': num_mixtures,
            'sample_rate': self.target_sr,
            'num_sources': 3,
            'num_speakers_total': len(speakers),
            'speakers': speakers,
            'source': 'TITML-IDN',
            'duration_seconds': target_duration,
            'total_duration_hours': (num_mixtures * target_duration) / 3600,
        }
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_dataset_info(self, train_count, dev_count, test_count, target_duration):
        """Generate comprehensive dataset info file."""
        info = {
            'dataset_name': 'TITML-3spk',
            'description': '3-speaker speech separation dataset from TITML-IDN',
            'created_at': datetime.now().isoformat(),
            'source_dataset': 'TITML-IDN (Indonesian Speech Dataset)',
            'configuration': {
                'sample_rate': self.target_sr,
                'clip_duration_seconds': target_duration,
                'num_speakers': 3,
                'snr_range_db': [-5, 5],
                'time_offset_max_seconds': 1.0,
            },
            'splits': {
                'train': {
                    'num_mixtures': train_count,
                    'duration_hours': (train_count * target_duration) / 3600,
                },
                'dev': {
                    'num_mixtures': dev_count,
                    'duration_hours': (dev_count * target_duration) / 3600,
                },
                'test': {
                    'num_mixtures': test_count,
                    'duration_hours': (test_count * target_duration) / 3600,
                },
            },
            'total': {
                'num_mixtures': train_count + dev_count + test_count,
                'duration_hours': ((train_count + dev_count + test_count) * target_duration) / 3600,
            },
            'directory_structure': {
                'train': {
                    'mix': '3-speaker mixtures',
                    's1': 'Speaker 1 (reference)',
                    's2': 'Speaker 2 (SNR-scaled, time-offset)',
                    's3': 'Speaker 3 (SNR-scaled, time-offset)',
                },
                'dev': 'Same structure as train',
                'test': 'Same structure as train',
            },
            'speaker_info': {
                'total_speakers': len(self.speakers),
                'male_speakers': len([s for s in self.speakers.keys() if s.startswith('m')]),
                'female_speakers': len([s for s in self.speakers.keys() if s.startswith('f')]),
            },
        }

        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\nDataset info saved to: {info_path}")
        return info


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description='Generate TITML-3spk dataset for speech separation'
    )
    parser.add_argument(
        '--titml-dir',
        type=str,
        default=None,
        help='Path to TITML-IDN raw dataset directory. '
             'Falls back to $TSS_RAW_DIR or <project_root>/dataset/raw/TTML-IDN.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for generated dataset. '
             'Falls back to $TSS_SYNTHETIC_DIR/TITML-3spk or '
             '<project_root>/dataset/synthetic/TITML-3spk.'
    )
    parser.add_argument(
        '--target-duration',
        type=float,
        default=5.0,
        help='Duration of each clip in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--target-hours',
        type=float,
        default=50.0,
        help='Target total dataset size in hours (default: 50.0)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8)'
    )
    parser.add_argument(
        '--dev-ratio',
        type=float,
        default=0.1,
        help='Ratio of dev data (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--split-mode',
        type=str,
        choices=['speaker', 'utterance'],
        default='utterance',
        help='Split mode: "speaker" (speaker-independent) or "utterance" '
             '(all speakers in every split, non-overlapping utterances). '
             'Default: utterance'
    )

    args = parser.parse_args()

    # Resolve paths via env-aware helper
    args.titml_dir = str(get_raw_dir(args.titml_dir))
    args.output_dir = str(get_synthetic_dir('TITML-3spk', args.output_dir))

    # Calculate number of mixtures needed
    total_mixtures = int((args.target_hours * 3600) / args.target_duration)
    train_mixtures = int(total_mixtures * args.train_ratio)
    dev_mixtures = int(total_mixtures * args.dev_ratio)
    test_mixtures = total_mixtures - train_mixtures - dev_mixtures
    
    # Ensure minimum mixtures per split (at least 100 for dev/test to be useful)
    min_mixtures = 100
    if dev_mixtures < min_mixtures:
        deficit = min_mixtures - dev_mixtures
        if train_mixtures > min_mixtures * 10:
            train_mixtures -= deficit
            dev_mixtures += deficit
    if test_mixtures < min_mixtures:
        deficit = min_mixtures - test_mixtures
        if train_mixtures > min_mixtures * 10:
            train_mixtures -= deficit
            test_mixtures += deficit

    print("\n" + "=" * 60)
    print("Dataset Generation Configuration")
    print("=" * 60)
    print(f"TITML source: {args.titml_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split mode: {args.split_mode}")
    print(f"Target duration: {args.target_duration}s")
    print(f"Target total hours: {args.target_hours}h")
    print(f"Total mixtures: {total_mixtures}")
    print(f"  Train: {train_mixtures} ({args.train_ratio*100:.0f}%)")
    print(f"  Dev:   {dev_mixtures} ({args.dev_ratio*100:.0f}%)")
    print(f"  Test:  {test_mixtures} ({(1-args.train_ratio-args.dev_ratio)*100:.0f}%)")
    print("=" * 60)

    # Initialize generator
    generator = TITMLMixGenerator3Spk(
        titml_dir=args.titml_dir,
        output_dir=args.output_dir,
        target_sr=16000,
        seed=args.seed
    )

    if args.split_mode == 'utterance':
        # Utterance-level split: all 20 speakers in every split,
        # non-overlapping utterances per speaker
        train_utts, dev_utts, test_utts = generator.split_utterances(
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=1 - args.train_ratio - args.dev_ratio
        )

        train_count = generator.generate_mixtures_from_utterances(
            utterances_by_speaker=train_utts,
            split_name='train',
            num_mixtures=train_mixtures,
            target_duration=args.target_duration,
            gender_balance=True
        )
        dev_count = generator.generate_mixtures_from_utterances(
            utterances_by_speaker=dev_utts,
            split_name='dev',
            num_mixtures=dev_mixtures,
            target_duration=args.target_duration,
            gender_balance=True
        )
        test_count = generator.generate_mixtures_from_utterances(
            utterances_by_speaker=test_utts,
            split_name='test',
            num_mixtures=test_mixtures,
            target_duration=args.target_duration,
            gender_balance=True
        )
    else:
        # Speaker-level split (original behaviour)
        train_spk, dev_spk, test_spk = generator.split_speakers(
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=1 - args.train_ratio - args.dev_ratio
        )

        train_count = generator.generate_mixtures(
            speaker_list=train_spk,
            split_name='train',
            num_mixtures=train_mixtures,
            target_duration=args.target_duration,
            gender_balance=True
        )
        dev_count = generator.generate_mixtures(
            speaker_list=dev_spk,
            split_name='dev',
            num_mixtures=dev_mixtures,
            target_duration=args.target_duration,
            gender_balance=True
        )
        test_count = generator.generate_mixtures(
            speaker_list=test_spk,
            split_name='test',
            num_mixtures=test_mixtures,
            target_duration=args.target_duration,
            gender_balance=True
        )

    # Generate comprehensive dataset info
    generator.generate_dataset_info(train_count, dev_count, test_count, args.target_duration)

    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nDataset location: {args.output_dir}")
    print("\nDataset structure:")
    print(f"  TITML-3spk/")
    print(f"  ├── train/ ({train_count} mixtures, ~{train_count * args.target_duration / 3600:.1f} hours)")
    print(f"  │   ├── mix/   <- 3-speaker mixture")
    print(f"  │   ├── s1/    <- speaker 1 (reference)")
    print(f"  │   ├── s2/    <- speaker 2 (SNR-scaled)")
    print(f"  │   └── s3/    <- speaker 3 (SNR-scaled)")
    print(f"  ├── dev/   ({dev_count} mixtures, ~{dev_count * args.target_duration / 3600:.1f} hours)")
    print(f"  ├── test/  ({test_count} mixtures, ~{test_count * args.target_duration / 3600:.1f} hours)")
    print(f"  └── dataset_info.json")
    print(f"\nTotal: {train_count + dev_count + test_count} mixtures")
    print(f"Total duration: ~{(train_count + dev_count + test_count) * args.target_duration / 3600:.1f} hours")


if __name__ == "__main__":
    main()
