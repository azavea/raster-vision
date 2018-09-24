import json

import numpy as np
from npstreams import imean, istd, last

from rastervision.utils.files import str_to_file, file_to_str


class RasterStats():
    def __init__(self):
        self.means = None
        self.stds = None

    def compute(self, raster_sources):
        chip_size = 300
        stride = chip_size

        def chip_stream(channel):
            for raster_source in raster_sources:
                windows = raster_source.get_extent().get_windows(
                    chip_size, stride)
                for window in windows:
                    chip = raster_source.get_raw_chip(window).astype(
                        np.float32)
                    chip = chip[:, :, channel].ravel()
                    # Ignore NODATA values.
                    chip[chip == 0.0] = np.nan
                    yield chip

        # Sniff the number of channels.
        window = raster_sources[0].get_extent().get_windows(chip_size,
                                                            stride)[0]
        nb_channels = raster_sources[0].get_raw_chip(window).shape[2]

        self.means = []
        self.stds = []
        for channel in range(nb_channels):
            mean = last(
                imean(chip_stream(channel), axis=None, ignore_nan=True))
            self.means.append(mean)
            std = last(istd(chip_stream(channel), axis=None, ignore_nan=True))
            self.stds.append(std)

    def save(self, stats_uri):
        # Ensure lists
        means = list(self.means)
        stds = list(self.stds)
        stats = {'means': means, 'stds': stds}
        str_to_file(json.dumps(stats), stats_uri)

    @staticmethod
    def load(stats_uri):
        stats_json = json.loads(file_to_str(stats_uri))
        stats = RasterStats()
        stats.means = stats_json['means']
        stats.stds = stats_json['stds']
        return stats
