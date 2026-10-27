"""Test double for soundfile: writes the raw samples."""


def write(file, data, samplerate, format=None, subtype=None):
    file.write(b"RIFF" + bytes(data))
