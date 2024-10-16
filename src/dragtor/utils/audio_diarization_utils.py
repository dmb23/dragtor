import re
from logging import raiseExceptions
from xml.dom import ValidationErr

import torch

from pathlib import Path
from loguru import logger
from dragtor import config
from pyannote.audio import Pipeline


def audio_diarize(audio_path: str, diarize_model: str = "pyannote/speaker-diarization-3.1", num_speakers: int = 1,
                  min_speakers: int = 1, max_speakers: int = 1, device: str = "mps") -> str:
    logger.info(f"Diarize {audio_path}")
    output_dir = Path(config.conf.base_path) / config.conf.data.diarize_cache
    output_dir.mkdir(exist_ok=True)

    # Load pre-trained model
    pipeline = Pipeline.from_pretrained(
        diarize_model,
        use_auth_token=config.conf.creds.hf,
    )
    pipeline.to(torch.device(device))

    diarization = pipeline(audio_path, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)

    diarization_results = []
    output_file = output_dir / f"{Path(audio_path).stem}.txt"
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_results.append(f"Speaker {speaker}: {turn.start:.1f}s to {turn.end:.1f}s")

    diarization_str = "\n".join(diarization_results)
    output_file.write_text(diarization_str)
    logger.debug(f"Diarization saved at {output_file}")

    return diarization_str


def parse_audio_transcript(transcript: str):
    # Regex to match [hh:mm:ss.xxx --> hh:mm:ss.xxx] and the following text
    pattern = r"\[(\d+):(\d+):([\d.]+)\.\d+ --> (\d+):(\d+):([\d.]+)\.\d+\]\s+(.+)"

    transcript_segments = []
    for match in re.finditer(pattern, transcript):
        start_sec = int(match.group(1)) * 60 * 60 + int(match.group(2)) * 60 + float(
            match.group(3))  # Convert start to seconds
        end_sec = int(match.group(4)) * 60 * 60 + int(match.group(5)) * 60 + float(
            match.group(6))  # Convert end to seconds
        text = match.group(7).strip()

        transcript_segments.append({
            "start": start_sec,
            "end": end_sec,
            "text": text
        })

    return transcript_segments


def parse_diarization(diarization: str):
    # Regex to match [Speaker SPEAKER_xx: ss.x to ss.x]
    pattern = r"Speaker (\S+): (\d+\.\ds) to (\d+\.\ds)"

    speaker_segments = []
    for match in re.finditer(pattern, diarization):
        speaker = match.group(1)
        start_time = float(match.group(2)[:-1])  # Convert start to seconds
        end_time = float(match.group(3)[:-1])  # Convert end to seconds
        speaker_segments.append({
            "start": start_time,
            "end": end_time,
            "speaker": speaker
        })

    return speaker_segments


def align_transcription_with_speakers(audio_path: str, audio_segments, diarization_segments):
    logger.info(f"Combine audio transcript & diarization of {audio_path}")
    output_dir = Path(config.conf.base_path) / config.conf.data.audio_cache
    output_dir.mkdir(exist_ok=True)

    aligned_transcription = []

    for audio_segment in audio_segments:
        audio_start = audio_segment['start']
        audio_end = audio_segment['end']
        audio_text = audio_segment['text']

        # Find the corresponding speaker segment
        speaker_in_segment = None

        for diarization_segment in diarization_segments:
            diarization_start = diarization_segment['start']
            diarization_end = diarization_segment['end']
            speaker = diarization_segment['speaker']

            # Check if whisper segment falls within the speaker segment
            if (audio_start >= diarization_start) and (audio_end <= diarization_end):
                speaker_in_segment = speaker
                break

        # Interpolate: if no direct speaker match is found, find the closest speaker
        if not speaker_in_segment:
            closest_speaker = None
            closest_distance = float('inf')  # for now infinite, to make sure each segment has speaker

            for diarization_segment in diarization_segments:
                diarization_start = diarization_segment['start']
                diarization_end = diarization_segment['end']
                speaker = diarization_segment['speaker']

                # Calculate distances to the start and end of the current whisper segment
                if audio_start > diarization_end:  # Before the current speaker segment
                    distance = audio_start - diarization_end
                elif audio_end < diarization_start:  # After the current speaker segment
                    distance = diarization_start - audio_end
                else:
                    distance = 0  # Overlap case already handled

                # Keep track of the closest speaker
                if distance < closest_distance:
                    closest_distance = distance
                    closest_speaker = speaker

            speaker_in_segment = closest_speaker

        # Add the aligned transcription with speaker
        aligned_transcription.append({
            "start": audio_start,
            "end": audio_end,
            "text": audio_text,
            "speaker": speaker_in_segment if speaker_in_segment else "Unknown"
        })

    grouped_transcription = group_transcription_by_speaker(aligned_transcription)
    final_transcription = []

    output_file = output_dir / f"diarization_{Path(audio_path).stem}.txt"
    for segment in grouped_transcription:
        final_transcription.append(
            f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['speaker']}: {segment['text']}"
        )

    output_file.write_text("\n".join(final_transcription))
    logger.debug(f"Transcript & diarization saved at {output_file}")

    return grouped_transcription


def group_transcription_by_speaker(aligned_transcription):
    grouped_transcription = []
    current_speaker = None
    current_text = ""
    current_start = None
    current_end = None

    for segment in aligned_transcription:
        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        end = segment['end']

        # If the speaker changes or this is the first segment, append the previous block
        if speaker != current_speaker:
            if current_speaker is not None:
                # Append the grouped segment
                grouped_transcription.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "text": current_text.strip()
                })

            # Start a new block for the new speaker
            current_speaker = speaker
            current_text = text
            current_start = start
            current_end = end
        else:
            # Concatenate text for the current speaker
            current_text += " " + text
            current_end = end

    # Append the last block
    if current_speaker is not None:
        grouped_transcription.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "text": current_text.strip()
        })

    return grouped_transcription
