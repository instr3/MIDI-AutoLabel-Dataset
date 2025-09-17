import os
from typing import BinaryIO, Optional, Tuple, Union

import torch
import torchaudio

InputType = Union[BinaryIO, str, os.PathLike]

def _get_load_filter(
        frame_offset: int = 0,
        num_frames: int = -1,
        convert: bool = True,
) -> Optional[str]:
    if frame_offset < 0:
        raise RuntimeError("Invalid argument: frame_offset must be non-negative. Found: {}".format(frame_offset))
    if num_frames == 0 or num_frames < -1:
        raise RuntimeError("Invalid argument: num_frames must be -1 or greater than 0. Found: {}".format(num_frames))

    # All default values -> no filter
    if frame_offset == 0 and num_frames == -1 and not convert:
        return None
    # Only convert
    aformat = "aformat=sample_fmts=fltp"
    if frame_offset == 0 and num_frames == -1 and convert:
        return aformat
    # At least one of frame_offset or num_frames has non-default value
    if num_frames > 0:
        atrim = "atrim=start_sample={}:end_sample={}".format(frame_offset, frame_offset + num_frames)
    else:
        atrim = "atrim=start_sample={}".format(frame_offset)
    if not convert:
        return atrim
    return "{},{}".format(atrim, aformat)


def _load_audio(
    s: "torchaudio.io.StreamReader",
    seek_time: Union[int, float] = 0,
    chunk_size: int = 1024,
    filter: Optional[str] = None,
    channels_first: bool = True,
) -> torch.Tensor:
    s.add_audio_stream(chunk_size, -1, filter_desc=filter)
    s.seek(seek_time)
    s.fill_buffer()
    chunk = s.pop_chunks()[0]
    if chunk is None:
        raise RuntimeError("Failed to decode audio.")
    waveform = chunk._elem
    return waveform.T if channels_first else waveform

def load_audio(
    src: InputType,
    frame_offset: int = 0,
    num_frames: int = -1,
    convert: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
) -> Tuple[torch.Tensor, int]:
    if hasattr(src, "read") and format == "vorbis":
        format = "ogg"
    s = torchaudio.io.StreamReader(src, format, None, buffer_size)
    sample_rate = int(s.get_src_stream_info(s.default_audio_stream).sample_rate)
    filter = _get_load_filter(0, num_frames, convert)
    waveform = _load_audio(s, frame_offset / sample_rate, num_frames, filter, channels_first)
    return waveform, sample_rate

