import re

import torch

from pathlib import Path
from loguru import logger
from dragtor import config
from pyannote.audio import Pipeline


def audio_diarize(audio_path: str, diarize_model: str = None, num_speakers: int = None,
                  min_speakers: int = None, max_speakers: int = None, device: str = None) -> str:
    """
    Run diarize function to differentiate audio transcription between different speakers.

    Input:
        audio_path (str): Path/URL to the audio to be transcribed.
        diarize_model (str, optional): The model used for diarization. Defaults to "pyannote/speaker-diarization-3.1".
        num_speakers (int, optional): The number of speaker. Prioritized over the other speaker inputs. Defaults to 1.
        min_speakers (int, optional): The minimum number of speaker. Defaults to 1.
        max_speakers (int, optional): The maximum number of speaker. Defaults to 1.
        device (str, optional): mps, cuda, cpu, or other devices. Defaults to mps.

    Returns:
        String: Speakers' audio timestamps.
    """
    diarize_model = diarize_model or config.conf.audio.diarization_model
    device = device or config.conf.device

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


def parse_audio_transcript(transcript: str) -> list[dict()]:
    """Reformat timestamp from raw transcription into seconds."""
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


def parse_diarization(diarization: str) -> list[dict()]:
    """Reformat timestamp from raw diarization into seconds."""
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


def align_transcription_with_speakers(audio_path: str, audio_segments, diarization_segments) -> list[dict()]:
    """
    Matches audio transcription and diarization using timestamp. Missing intervals will follow the nearest timestamp between segments.

    Input:
        audio_path (str): Path/URL to the audio to be transcribed.
        audio_segments (str): Raw audio transcription.
        diarization_segments (str): Raw audio diarization.

    Return:
        Dialogue between speaker(s) and its audio transcription.
    """
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

            # Check if raw audio transcript segment falls within the speaker segment
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

    output_file = output_dir / f"diarization_{Path(audio_path).parts[-2]}_{Path(audio_path).stem}.txt"
    for segment in grouped_transcription:
        final_transcription.append(
            f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['speaker']}: {segment['text']}"
        )

    output_file.write_text("\n".join(final_transcription))
    logger.debug(f"Transcript & diarization saved at {output_file}")

    return grouped_transcription


def group_transcription_by_speaker(aligned_transcription) -> list[dict()]:
    """Groups multiple transcript segments with the same speaker into one line of dialogue"""
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