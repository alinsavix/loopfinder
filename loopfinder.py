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
from dataclasses import dataclass
from pathlib import Path
import re

# mathy python bits
import matplotlib.pyplot as plt
import librosa
import numpy
import numpy as np
from scipy import signal


@dataclass
class OffsetInfo(object):
    start: float
    start_samples: int
    end: float
    length: float
    length_samples: int
    samplerate: int


# A very basic HH:MM:SS.SSS format to seconds conversion. We could
# use strptime here, but really, who in their right mind wants to use
# strptime? This is simple enough and straightforward. Also handles the
# case of just specifying some number of seconds without the HH or MM parts.
def hms_to_sec(hms: str) -> float:
    timesplit = hms.split(":")

    if len(timesplit) == 3:
        h, m, s = timesplit
    elif len(timesplit) == 2:
        h = 0
        m, s = timesplit
    elif len(timesplit) == 1:
        h = 0
        m = 0
        s = timesplit[0]

    return (int(h) * 60 * 60) + (int(m) * 60) + float(s)


# Convert seconds to HH:MM:SS.SSS format. Sure, this could use strftime
# or datetime.timedelta, but both of those have their own issues when
# you want a consistent format involving milliseconds.
def sec_to_hms(secs: float) -> str:
    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    return f"{hours:02d}:{minutes:02d}:{secs:02.3f}"


# Take a bit of audio, and some offsets, run some correlation on it, and
# figure out where the best correlation happens, which is probably where our
# loop repeats. Not guaranteed to work, but it seems to work reasonably well.
def find_offset(file: Path, start_offset: float, search_offset: float, window: int, skip_graph: bool = False) -> OffsetInfo:
    # load in the parts of the intput file that we need (as numpy arrays of floats)
    right, samplerate = librosa.load(
        str(file), sr=None, offset=start_offset, duration=window, mono=True)
    left, _ = librosa.load(str(file), sr=samplerate, offset=search_offset,
                           duration=4 * window, mono=True)

    # I won't lie, I don't entirely understand the specifics of why this
    # particular set of options works. From what I can tell, doing the
    # correlation actually gives you an array that's len(left) + len(right)
    # in length, and the array slice we use from this selects the right part
    # of it for the correlation we want to measure. There may even be a
    # better way to do this overall, but my math skills have long since
    # atrophied far too hard to figure out what that way would be.  --A
    c = numpy.correlate(left, right, mode='full')[len(right) - 1:]

    # find the index & offset of the single highest correlation
    peak = np.argmax(c)
    match_offset = round(peak / samplerate, 3)

    # I've absolutely zero clue what a good way to tell if the correlation was
    # actually a GOOD one, other than looking at the correlation graph with my
    # Mk. I human eyeball. You'd think the level of correlation (as represented)
    # by a number) would be an indicator, but I've seen good correlations of > 3.0,
    # and obviously bad correlations show the same, so... I dunno. These bits here
    # are an attempt at gathering some possibly useful information that may or
    # may not be useful in the future for figuring out correlation quality.
    match_audio = left[peak:peak + (window * samplerate)]  # the actual audio we matched

    # subtract our original audio signal from our match, under the theory that these
    # numbers should be far smaller for an actual good correlation. Multiplying by
    # 2**15 gives us a number that corresponds roughly to 16-bit audio values
    diff = numpy.multiply(match_audio, 2**15) - numpy.multiply(right, 2**15)

    # measure that difference overall, under the idea that it should be pretty small for
    # a good correlation
    avg = numpy.average(numpy.abs(diff))

    # prominence theoretically measures how, uh, prominent a peak is, for some
    # mathematical definition of such. Not sure this is the most useful measure
    # of this, but time will (or may, at least) tell.
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html
    prom = float(signal.peak_prominences(c, [peak], wlen=samplerate)[0])
    print(f"Found! {peak} samples ({match_offset:0.3f}s) from search offset {search_offset:0.3f}s")
    print(f"       peak corr. {c[peak]:0.3f}, avg diff. {avg:0.3f}, prominence {prom:0.3f}")

    plt.figure(figsize=(14, 5))
    plt.title("Correlation")
    plt.plot(c)
    plt.savefig("cross-correlation.png")

    if not skip_graph:
        try:
            plt.show()
        except Exception:  # FIXME: figure out what this actually throws, and catch that
            print("ERROR: can't show graph. Make sure you have a matplotlib backend installed.")



    # So at this point we have...
    # - the snippet of audio we're looking for, which starts at 0.000s, and
    #   lasts for `window` seconds. The -actual- start time is `args.start + 0.000s``
    # - offset, which is where the match we found is, relative to search_offset
    #
    # From this, we can generate a struct that has:
    #   - start: the start time of the loop within the provided file
    #   - end: the end time of the loop within the file
    #   - length: the length of the loop
    start_samples = int(start_offset * samplerate)
    endtime = search_offset + match_offset
    endtime_samples = int(search_offset * samplerate) + peak
    length = endtime - start_offset
    length_samples = endtime_samples - start_samples
    return OffsetInfo(start=start_offset, start_samples=start_samples, end=endtime, length=length, length_samples=length_samples, samplerate=samplerate)


# helper function to validate and parse a provided time string
def offset_str(arg_value: str) -> float:
    offset_re = re.compile(r"^(\d+:)?(\d+:)?(\d+)(\.\d+)?$")

    if not offset_re.match(arg_value):
        raise argparse.ArgumentTypeError

    # else
    return hms_to_sec(arg_value)


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
        "--window",
        metavar="seconds",
        type=int,
        default=5,
        help="Use n seconds of audio from start postion for matching",
    )

    parser.add_argument(
        "--markers",
        default=False,
        action='store_true',
        help='Generate marker data cut & paste block',
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
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

    args = parse_arguments()
    info = find_offset(args.file, args.start, args.searchat, args.window, args.no_graph)

    # Figure out how long the file is, total, mostly for creating markers
    duration = librosa.get_duration(filename=str(args.file))

    # If we're not using the actual loop start time in our match parameters,
    # take the offsets we calculated and shift them so that we end up reporting
    # the actual loop start time rather than the shifted one
    if args.realstart > 0:
        real_offset = info.start - args.realstart
        info.start -= real_offset
        info.end -= real_offset

    print(
        "\n"
        f"Loop start:  {info.start:0.3f}s ({sec_to_hms(info.start)}; {info.start_samples} samples)\n"
        f"Loop end:    {info.end:0.3f}s ({sec_to_hms(info.end)})\n"
        f"Loop length: {info.length:0.3f}s ({sec_to_hms(info.length)}; {info.length_samples} samples)\n"
        f"Sample rate: {info.samplerate}Hz\n"
        # "\n"
        # f"Clip length: {duration:0.3f}s\n"
    )

    #     print(
    #         f"Loop start: {info['start']:0.3f}s\nEnds at: {info['end']:0.3f}\nLoop length: {info['length']:0.3f}s\nTrack duration: {duration:0.3f}s")

    # else:
    #     print(
    #         f"Loop start: {info['start']:0.3f}s\nEnds at: {info['end']:0.3f}\nLoop length: {info['length']:0.3f}s\nTrack duration: {duration:0.3f}s")

    if args.markers > 0:
        t = info.start
        i = 1

        print()
        while True:
            print(f"{t:0.6f},{t:0.6f},1,segmentation,Loop_{i}")
            i += 1
            t += info.length
            if t > (duration + info.length):
                break


if __name__ == '__main__':
    main()
