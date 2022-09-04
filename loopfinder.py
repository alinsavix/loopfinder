#!/usr/bin/env python3
# FIX HERE (see readme): If for some reason matplotlib isn't correctly
# detecting the interactive backend for your system, but you know you have
# one installed, you can specify it explicitly here. You need to call
# matplotlib.use() before importing matplotlib.pyplot and friends.  I'm not
# sure if there's a better way to do this. --A
# import matplotlib
# matplotlib.use('Qt5Agg')

# Remaining imports
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"  # shut up a NUMBA warning

import warnings
warnings.filterwarnings('ignore')  # shut up a librosa warning

# typical python bits
import argparse
import atexit
import dataclasses
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Union, Text, Sequence, Any, Optional, List

# mathy python bits
import matplotlib.pyplot as plt
import librosa
import numpy
import numpy as np
import numpy.typing as npt
from scipy import signal
import soundfile as sf


@dataclasses.dataclass
class OffsetDetail():
    peak_sample: int
    peak_value: float
    avg_diff: float
    peak_prominence: float


# original request parameters
@dataclasses.dataclass
class ArgsDetail():
    start_offset: float
    search_offset: float
    searchlength: int
    window: int
    realstart: float
    fft: bool


# class to hold our analysis data, and wrangle it to disk
@dataclasses.dataclass
class OffsetInfo():
    file: str  # without path
    start: float
    start_samples: int
    end: float
    end_samples: int
    length: float
    length_samples: int
    samplerate: int

    correlation: Optional[npt.ArrayLike]

    duration: Optional[float] = None

    detail: Optional[OffsetDetail] = None
    args: Optional[ArgsDetail] = None

    def dump_to(self, file: Path) -> None:
        trimmed = self
        trimmed.correlation = None

        with file.open("w") as fp:
            json.dump(dataclasses.asdict(self), fp, indent=4)


# A very basic HH:MM:SS.SSS format to seconds conversion. We could
# use strptime here, but really, who in their right mind wants to use
# strptime? This is simple enough and straightforward. Also handles the
# case of just specifying some number of seconds without the HH or MM parts.
def hms_to_sec(hms: str) -> float:
    timesplit = hms.split(":")

    if len(timesplit) == 3:
        h, m, s = timesplit
    elif len(timesplit) == 2:
        h = "0"
        m, s = timesplit
    elif len(timesplit) == 1:
        h = "0"
        m = "0"
        s = timesplit[0]
    else:
        print(f"ERROR: too many fields ({len(timesplit)}) in hh:mm:ss string")
        sys.exit(1)

    return (int(h) * 60 * 60) + (int(m) * 60) + float(s)


# Convert seconds to HH:MM:SS.SSS format. Sure, this could use strftime
# or datetime.timedelta, but both of those have their own issues when
# you want a consistent format involving milliseconds.
def sec_to_hms(secs: float) -> str:
    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    ms = int((secs % 1) * 1000)
    secs = int(secs)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


# For whatever reason, the `soundfile` library doesn't let us load
# the audio file direct from a video, but we want to use that library
# for processing our audio data a chunk at a time... so extract the
# audio as a wav we can then wrangle.
#
# I'd love to do this via librosa (so we don't need another program),
# but best I can tell the only way to do that involves loading the
# entire (potentially long) audio completely into memory, which I'd
# rather avoid.
#
# Maybe we can do that as a fallback if ffmpeg isn't available?
#
# If no filename is provided, a temp file will be created, and cleaned up when
# the program exits.
def wav_extract(file: Path, output: Optional[Path] = None) -> Optional[Path]:
    if not output:
        fd, newfile = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        outpath = Path(newfile)

        atexit.register(lambda: outpath.unlink())

    else:
        outpath = output

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", str(file),
        "-vn",
        "-f", "wav",
        "-y",
        str(output),
    ]

    try:
        subprocess.run(ffmpeg_cmd, stdin=None, capture_output=True, shell=False, check=True)
    except FileNotFoundError as e:
        print("WARNING: ffmpeg not found in path, so no layer diffs", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"WARNING: ffmpeg exited code {e.returncode}, so no layer diffs", file=sys.stderr)
        return None

    # else
    return output


# make some diffs. Assumes everything is stereo
def wav_diff(wav: Path, info: OffsetInfo, output: str, layers: int = 4) -> None:
    prev: List[Optional[npt.ArrayLike]] = [None] * layers

    outputs = []
    for i in reversed(range(1, layers + 1)):
        # outputs.append(sf.SoundFile(f"{output}-{i}.wav", "w",
        #                             samplerate=info.samplerate, channels=2))
        of = sf.SoundFile(f"{output}-{i}.wav", "w", samplerate=info.samplerate, channels=2)
        # of.write(np.zeros((info.start_samples, 2)))
        outputs.append(of)

    for block in sf.blocks(str(wav), blocksize=info.length_samples, overlap=0, start=info.start_samples, dtype='int16'):
        # print(f"read block size: {len(block)}    shape: {block.shape}")
        for i in range(0, layers):
            # if prev[i] is not None and len(prev[i]) == len(block):
            if prev[i] is not None:
                if len(prev[i]) == len(block):
                    outputs[i].write(block - prev[i])
                else:
                    outputs[i].write(block - np.resize(prev[i], block.shape))
            else:
                outputs[i].write(np.zeros(block.shape))

        prev = prev[1:]
        prev.append(block)

    pass


# The file argument here is the full path, so that we know where to write out
# the correlation graph.
def graph(oi: OffsetInfo, file: Path, skip_graph: bool = False):
    x_scale = np.arange(oi.start_samples, oi.start_samples + len(oi.correlation)) / 48000.0
    plt.figure(figsize=(14, 5))
    plt.title(f"Correlation for '{oi.file}'")
    plt.xlabel("seconds")
    plt.plot(x_scale, oi.correlation)
    plt.savefig(file.with_name("correlation.png"))

    if not skip_graph:
        try:
            plt.show()
        except Exception:  # FIXME: figure out what this actually throws, and catch that
            print("ERROR: can't show graph. Make sure you have a matplotlib backend installed.")


# Take a bit of audio, and some offsets, run some correlation on it, and
# figure out where the best correlation happens, which is probably where our
# loop repeats. Not guaranteed to work, but it seems to work reasonably well.
def find_offset(file: Path, start_offset: float, search_offset: float, search_length: int,
                window: int, skip_graph: bool = False, use_fft: bool = False) -> OffsetInfo:
    is_reversed = True if search_offset < start_offset else False
    # load in the parts of the intput file that we need (as numpy arrays of floats)
    right, samplerate = librosa.load(
        str(file), sr=None, offset=start_offset, duration=window, mono=True)

    if search_length <= 0:
        search_length = 4 * window

    if is_reversed:
        left, _ = librosa.load(str(file), sr=samplerate, offset=search_offset,
                               duration=min(start_offset - search_offset, search_length), mono=True)
    else:
        left, _ = librosa.load(str(file), sr=samplerate, offset=search_offset,
                               duration=search_length, mono=True)


    # do some multiplying to make the very low (usually) audio signal a bit
    # louder, for better correlations. We'll also remove things in the area
    # to be searched that would be clipped, because they can't possibly be
    # part of an good correlation. We're using 0.8 as the volume we want to
    # reach, just to give ourselves some wiggle room.
    src_max = np.max(np.abs(right))
    mult = 0.8 / src_max

    right *= mult
    left *= mult

    # And then on the audio to be searched, turn anything that's too loud to
    # be part of the correlation into NANs, which should (I think) basically
    # amount to a negative correlation at that point.
    #
    # FIXME: collapse this again when use_fft is sorted better
    if use_fft:
        left[abs(left) > 1.0] = 0.0
    else:
        left[abs(left) > 1.0] = np.NAN

    # I won't lie, I don't entirely understand the specifics of why this
    # particular set of options works. From what I can tell, doing the
    # correlation actually gives you an array that's len(left) + len(right)
    # in length, and the array slice we use from this selects the right part
    # of it for the correlation we want to measure. There may even be a
    # better way to do this overall, but my math skills have long since
    # atrophied far too hard to figure out what that way would be.  --A
    if use_fft:
        c = signal.correlate(left, right, mode='full', method='fft')[len(right) - 1:]
    else:
        c = numpy.correlate(left, right, mode='full')[len(right) - 1:]

    # find the index & offset of the single highest correlation
    peak = np.nanargmax(c)
    match_offset = round(peak / samplerate, 3)


    # I've absolutely zero clue what a good way to tell if the correlation was
    # actually a GOOD one, other than looking at the correlation graph with my
    # Mk. I human eyeball. You'd think the level of correlation (as represented)
    # by a number) would be an indicator, but I've seen good correlations of > 3.0,
    # and obviously bad correlations show the same, so... I dunno. These bits here
    # are an attempt at gathering some possibly useful information that may or
    # may not be useful in the future for figuring out correlation quality.
    match_audio = left[peak:peak + (window * samplerate)]  # the actual audio we matched

    if len(match_audio) != (window * samplerate):
        print("ERROR: match failed (best correlation is incomplete)", file=sys.stderr)
        sys.exit(1)

    # So at this point we have...
    # - the snippet of audio we're looking for, which starts at 0.000s, and
    #   lasts for `window` seconds. The -actual- start time is `args.start + 0.000s``
    # - offset, which is where the match we found is, relative to search_offset
    #
    # From this, we can generate a struct that has:
    #   - start: the start time of the loop within the provided file
    #   - end: the end time of the loop within the file
    #   - length: the length of the loop

    # if it's reversed, we have to flipflop everything
    if is_reversed:
        starttime = float(search_offset + match_offset)
        start_samples = int(search_offset * samplerate) + int(peak)
        endtime = float(start_offset)
        end_samples = int(endtime * samplerate)
        length = float(endtime - starttime)
        length_samples = int(end_samples - start_samples)
    else:
        starttime = float(start_offset)
        start_samples = int(starttime * samplerate)
        endtime = float(search_offset + match_offset)
        end_samples = int(search_offset * samplerate) + int(peak)
        length = float(endtime - start_offset)
        length_samples = int(end_samples - start_samples)


    oi = OffsetInfo(file=file.name, start=starttime, start_samples=start_samples, end=endtime,
                    end_samples=end_samples, length=length, length_samples=length_samples,
                    samplerate=samplerate, correlation=c)


    # subtract our original audio signal from our match, under the theory that these
    # numbers should be far smaller for an actual good correlation. Multiplying by
    # 2**15 gives us a number that corresponds roughly to 16-bit audio values
    diff = numpy.multiply(match_audio, 2**15) - numpy.multiply(right, 2**15)

    # measure that difference overall, under the idea that it should be pretty small for
    # a good correlation
    avg = numpy.average(numpy.abs(diff)) / mult

    # prominence theoretically measures how, uh, prominent a peak is, for some
    # mathematical definition of such. Not sure this is the most useful measure
    # of this, but time will (or may, at least) tell.
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html
    prom = float(signal.peak_prominences(c, [peak], wlen=samplerate)[0])
    print(f"Found! {peak} samples ({match_offset:0.3f}s) from search offset {search_offset:0.3f}s")
    print(f"       peak corr. {c[peak]:0.3f}, avg diff. {avg:0.3f}, prominence {prom:0.3f}")

    oi.detail = OffsetDetail(peak_sample=int(peak), peak_value=float(c[peak]), avg_diff=float(avg),
                             peak_prominence=float(prom))


    graph(oi, file, skip_graph)

    # plt.figure(figsize=(14, 5))
    # plt.title("Correlation")
    # plt.plot(c)
    # plt.savefig(file.with_name("correlation.png"))

    # if not skip_graph:
    #     try:
    #         plt.show()
    #     except Exception:  # FIXME: figure out what this actually throws, and catch that
    #         print("ERROR: can't show graph. Make sure you have a matplotlib backend installed.")


    return oi


# helper function to validate and parse a provided time string
def offset_str(arg_value: str) -> float:
    offset_re = re.compile(r"^(\d+:)?(\d+:)?(\d+)(\.\d+)?$")

    if not offset_re.match(arg_value):
        raise argparse.ArgumentTypeError

    # else
    return hms_to_sec(arg_value)


class NegateAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Union[Text, Sequence[Any], None], option_string: Optional[Text] = "") -> None:
        assert option_string is not None  # n
        setattr(namespace, self.dest, option_string[2:4] != 'no')

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        metavar="timestring",
        type=offset_str,
        default=0.0,
        help="Start time of first loop (for matching)",
    )

    parser.add_argument(
        "--realstart",
        metavar="timestring",
        type=offset_str,
        default=0.0,
        help="Actual start time of first loop (for marking)",
    )

    # no default, can't do anything without a real time being specified
    parser.add_argument(
        "--searchat", "--search_at",
        metavar="timestring",
        type=offset_str,
        help="Time to start looking for matches",
    )

    parser.add_argument(
        "--searchlength", "--searchlen", "--slength", "--slen",
        metavar="seconds",
        type=int,
        default=0,
        help="amount of data to be compared (starting at --searchat",
    )

    parser.add_argument(
        "--window",
        metavar="seconds",
        type=int,
        default=5,
        help="Use n seconds of audio from start postion for matching",
    )

    # experimental support for correlating w/ a FFT
    parser.add_argument(
        "--fft",
        default=False,
        action='store_true',
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--markers",
        "--no-markers",
        default=True,
        action=NegateAction,
        nargs=0,
        help='Generate marker data cut & paste block',
    )

    # hidden because it's a real fricking mess to use, and is useful in
    # limited situations.
    parser.add_argument(
        "--diffs",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--no-graph", "--nograph",
        default=False,
        action="store_true",
        help="Don't display correlation graph",
    )

    parser.add_argument(
        "file",
        metavar="audio_file",
        type=Path,
        help="File to process for loop markers",
    )

    parsed_args = parser.parse_args()

    return parsed_args


def main():
    # os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    args = parse_arguments()
    info = find_offset(args.file, args.start, args.searchat,
                       args.searchlength, args.window, args.no_graph, args.fft)

    # Figure out how long the file is, total, mostly for creating markers
    info.duration = librosa.get_duration(filename=str(args.file))

    # If we're not using the actual loop start time in our match parameters,
    # take the offsets we calculated and shift them so that we end up reporting
    # the actual loop start time rather than the shifted one
    if args.realstart > 0:
        real_offset = info.start - args.realstart
        info.start -= real_offset
        info.start_samples -= int(real_offset * info.samplerate)
        info.end -= real_offset
        info.end_samples -= int(real_offset * info.samplerate)

    info.args = ArgsDetail(
        start_offset=args.start,
        search_offset=args.searchat,
        searchlength=args.searchlength,
        window=args.window,
        realstart=args.realstart,
        fft=args.fft
    )

    print(
        "\n"
        f"Loop start:  {info.start:0.3f}s ({sec_to_hms(info.start)}; {info.start_samples} samples)\n"
        f"Loop end:    {info.end:0.3f}s ({sec_to_hms(info.end)})\n"
        f"Loop length: {info.length:0.3f}s ({sec_to_hms(info.length)}; {info.length_samples} samples)\n"
        f"Sample rate: {info.samplerate}Hz\n"
    )

    info.dump_to(args.file.with_name("correlation.json"))

    if args.diffs > 0:
        wavfile = wav_extract(args.file)

        # if there's an error, it'll be displayed by wav_extract, otherwise...
        if wavfile:
            wav_diff(wavfile, info, str(args.file.with_name("audiodiff")), layers=args.diffs)
            print(f"Wrote {args.diffs} audio diff layers")

    if args.markers > 0:
        t = info.start
        i = 1

        print()

        # I hate long lines and I cannot lie (...sigh)
        with args.file.with_name("markers_premiere.txt").open("w") as markers_prem, \
                args.file.with_name("markers_audacity.txt").open("w") as markers_aud:
            while True:
                print(f"{t:0.6f},{t:0.6f},1,segmentation,Loop_{i}", file=markers_prem)
                print(f"{t:0.6f}\t{t:0.6f}\tLoop_{i}", file=markers_aud)
                i += 1
                t += info.length
                if t > (info.duration + info.length):
                    break


if __name__ == '__main__':
    main()
