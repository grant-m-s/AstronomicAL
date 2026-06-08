from dataclasses import dataclass, field

import copy
import warnings
import numpy as np
from astropy.visualization import PowerStretch, SqrtStretch, LogStretch
from astropy.visualization import AsinhStretch, LinearStretch, AsymmetricPercentileInterval, MinMaxInterval, PercentileInterval
from astropy.wcs import WCS, FITSFixedWarning
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.nddata import Cutout2D
from astropy.modeling.functional_models import Gaussian2D


@dataclass
class BandState:
    raw: np.ndarray
    wcs: WCS | None = None
    stretched: np.ndarray = None
    clipped: np.ndarray = None
    scaled: np.ndarray = None
    _resampled_raw: np.ndarray = None
    _resampled_wcs: WCS | None = None
    config: dict = field(default_factory=dict)

    def invalidate_from(self, level: str):
        if level == "band":
            self.stretched = None
            self.clipped = None
            self.scaled = None
            self._resampled_raw = None
            self._resampled_wcs = None

        elif level == "stretch":
            self.clipped = None
            self.scaled = None

        elif level == "clip":
            self.scaled = None

        elif level == "resample":
            self.stretched = None
            self.clipped = None
            self.scaled = None

        elif level == "scale":
            pass


class ImageVisualizationClass:
    _STRETCH_MAP = {
        "Linear": lambda scale: LinearStretch(slope=scale if scale is not None else 1),
        "Sqrt": lambda scale: SqrtStretch(),
        "Log": lambda scale: LogStretch(a=scale if scale is not None else 1000),
        "Asinh": lambda scale: AsinhStretch(a=scale if scale is not None else 0.1),
        "PowerLaw": lambda scale: PowerStretch(a=scale if scale is not None else 2),
    }

    _INTERVAL_MAP = {
        "Asymmetric": lambda perc: AsymmetricPercentileInterval(
            lower_percentile=perc[0],
            upper_percentile=perc[1],
        ),
        "MinMax": lambda _: MinMaxInterval(),
        "Percentile": lambda perc: PercentileInterval(perc),
    }

    _INTERVAL_DEFAULTS = {
        "Asymmetric": (0.1, 100),
        "MinMax": None,
        "Percentile": 99.5,
    }

    def __init__(
        self,
        images,
        wcs=None,
        band_names=None,
        color_image=False,
        color_bands=None,
        color_name=None,
        target_wcs=None,
    ):
        """ images : 2D np.ndarray or list of 2D np.ndarrays.
            The image(s) which will be transformed and rendered
        wcs : WCS file
        band_names : string or list of strings. Band identifiers
        color_image : bool. True if a color image must be created
        color_bands : list of strings. names of the images used as R,G and B
        """
        if isinstance(images, np.ndarray):
            if images.ndim != 2:
                raise ValueError("Single image must be a 2D numpy array")
            images = [images]
        elif isinstance(images, list):
            if not all(isinstance(img, np.ndarray) and img.ndim == 2 for img in images):
                raise ValueError("All images must be 2D numpy arrays")
        else:
            raise TypeError("'images' must be a 2D numpy array or list of 2D numpy arrays")

        if wcs is not None:
            if isinstance(wcs, WCS):
                wcs = [wcs]
            elif isinstance(wcs, (list, tuple)):
                if not all(isinstance(x, WCS) for x in wcs):
                    raise TypeError("All elements in wcs list must be WCS objects")
            else:
                raise TypeError("wcs must be a WCS object or a list/tuple of WCS objects")

            if len(wcs) != len(images):
                raise ValueError("Length of wcs must match number of images")

        if band_names is None:
            band_names = [str(i) for i in range(len(images))]
        elif isinstance(band_names, str):
            band_names = [band_names]

        if len(band_names) != len(images):
            raise ValueError("'images' and 'image_names' must have the same length")

        self.data = dict(zip(band_names, images))
        self.target_wcs = target_wcs
        self._bands = {}

        for idx, (band_name, img) in enumerate(self.data.items()):
            # This dataclass stores the image properties by band so that i do not need to apply all the transformation every time
            band_wcs = None
            if wcs is not None:
                band_wcs = wcs[idx]

            self._bands[band_name] = BandState(
                raw=img,
                wcs=band_wcs,
            )

        self.has_color_img = color_image
        if self.has_color_img:
            if len(self.data) < 3:
                raise ValueError(
                    "Color image can be created only if at least three images are given"
                )
            if color_bands is None:
                raise ValueError(
                    "'color_bands' must be provided when color_image=True"
                )
            if len(color_bands) != 3:
                raise ValueError(
                    "'color_bands' must contain exactly three bands (R, G, B)"
                )
            if not all(band in self.data for band in color_bands):
                raise KeyError(
                    "Bands used for the color image must be among band_names"
                )
            self.color_bands = color_bands
        else:
            self.color_bands = None

        self.color_name = color_name if color_name is not None else "ColorImage"

    @property
    def band_names(self):
        return list(self._bands.keys())

    @property
    def has_color(self):
        return bool(
            self.has_color_img
            and self.color_bands is not None
            and len(self.color_bands) == 3
            and all(band in self._bands for band in self.color_bands)
        )

    @property
    def available_bands(self):
        if self.has_color:
            return [self.color_name] + self.band_names
        return self.band_names

    def resolve_wcs_band(self, band):
        """Map virtual display bands to a real band with WCS.

        This is only for plugin compatibility. The original visualization logic
        is unchanged; this method just lets overlays/scale bars use the Color
        image by resolving it to one of the real bands, usually VIS.
        """
        if band != self.color_name:
            return band
        for candidate in ["VIS"]:
            if candidate in self._bands:
                return candidate
        if self.color_bands is not None:
            for candidate in reversed(self.color_bands):
                if candidate in self._bands:
                    return candidate
        for candidate in self.band_names:
            if candidate in self._bands:
                return candidate
        return band

    def _get_aligned_image(self, band_state: BandState):
        if self.target_wcs is None:
            return band_state.raw, band_state.wcs

        # Already computed
        if (
            band_state._resampled_raw is not None
            and band_state._resampled_wcs == self.target_wcs
        ):
            return band_state._resampled_raw, band_state._resampled_wcs

        # Image has the same WCS
        if band_state.wcs == self.target_wcs:
            band_state._resampled_raw = band_state.raw
            band_state._resampled_wcs = band_state.wcs
            return band_state.raw, band_state.wcs

        # Compute reprojection
        resampled, _ = reproject_interp(
            (band_state.raw, band_state.wcs),
            self.target_wcs,
            shape_out=self.target_wcs.array_shape,
        )
        band_state._resampled_raw = resampled
        band_state._resampled_wcs = self.target_wcs
        return resampled, self.target_wcs

    def _get_stretch(self, stretch_type, stretch_scale=None):
        if stretch_type not in self._STRETCH_MAP:
            raise ValueError(f"Unknown stretch type: {stretch_type}")
        return self._STRETCH_MAP[stretch_type](stretch_scale)

    def _get_interval(self, interval_type, interval_param=None):
        if interval_type not in self._INTERVAL_MAP:
            raise ValueError(f"Unknown interval type: {interval_type}")

        if interval_param is None:
            interval_param = self._INTERVAL_DEFAULTS.get(interval_type)

        return self._INTERVAL_MAP[interval_type](interval_param)

    @staticmethod
    def _stretch_image(image, stretch, stretch_interval):
        if stretch is None:
            stretch = LinearStretch()

        if stretch_interval is None:
            transform = stretch
        else:
            transform = stretch + stretch_interval

        return transform(image)

    @staticmethod
    def _clip_image(image, low, high, image_min=None, image_max=None):
        arr = np.asarray(image, dtype=np.float32)

        if (low == 0) and (high == 1):
            return arr

        image_min, image_max = ImageVisualizationClass._finite_min_max(
            arr,
            image_min=image_min,
            image_max=image_max,
        )

        if image_min is None or image_max is None:
            return np.zeros(arr.shape, dtype=np.float32)

        image_range = image_max - image_min
        absolute_low = image_min + float(low) * image_range
        absolute_high = image_min + float(high) * image_range

        return np.clip(arr, absolute_low, absolute_high)
    
    @staticmethod
    def _finite_min_max(image, image_min=None, image_max=None):
        arr = np.asarray(image, dtype=np.float32)
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            return None, None

        if image_min is None:
            image_min = float(finite.min())
        else:
            image_min = float(image_min)

        if image_max is None:
            image_max = float(finite.max())
        else:
            image_max = float(image_max)

        if not np.isfinite(image_min) or not np.isfinite(image_max):
            return None, None

        if image_max <= image_min:
            return None, None

        return image_min, image_max


    @staticmethod
    def _scale_image(image, scale_method="minmax", image_min=None, image_max=None):
        arr = np.asarray(image, dtype=np.float32)
        scaled = np.zeros(arr.shape, dtype=np.float32)
        finite_mask = np.isfinite(arr)

        if not finite_mask.any():
            return scaled

        method = str(scale_method or "minmax").lower()

        if method == "minmax":
            image_min, image_max = ImageVisualizationClass._finite_min_max(
                arr,
                image_min=image_min,
                image_max=image_max,
            )

            if image_min is None or image_max is None:
                return scaled

            image_range = image_max - image_min

            np.subtract(arr, image_min, out=scaled, where=finite_mask)
            np.divide(scaled, image_range, out=scaled, where=finite_mask)
            np.clip(scaled, 0.0, 1.0, out=scaled)
            scaled[~finite_mask] = 0.0
            return scaled

        if method == "expand":
            finite_values = arr[finite_mask]
            mid_value = float(np.nanmedian(finite_values))
            sigma = float(np.nanstd(finite_values))

            if not np.isfinite(sigma) or sigma <= 0:
                return ImageVisualizationClass._scale_image(arr, scale_method="minmax")

            expanded = np.where(arr > mid_value + sigma, arr * 2.0, arr / 2.0)
            return ImageVisualizationClass._scale_image(expanded, scale_method="minmax")

        raise ValueError(f"Unknown scale_method: {scale_method}")

    @staticmethod
    def _unzip(value):
        if np.iterable(value) and not isinstance(value, (str, bytes)):
            return value
        return [value, value, value]

    def get_current_plot_config(self):
        if getattr(self, "_plot_config", None) is None:
            return None  # clearer than {}
        return copy.deepcopy(self._plot_config)

    def get_plot_data(
        self,
        band,
        stretch="Linear",
        stretch_scale=None,
        stretch_interval="Asymmetric",
        low_clip=0,
        high_clip=1,
        gamma_color=(1, 1, 1),
        scale_method="MinMax",
        _internal=False,
    ):
        if not _internal:
            self._plot_config = {
                "band": band,
                "stretch": stretch,
                "stretch_scale": stretch_scale,
                "stretch_interval": stretch_interval,
                "low_clip": low_clip,
                "high_clip": high_clip,
                "gamma_color": gamma_color,
                "scale_method": scale_method,
            }

        if band == self.color_name:
            if not self.has_color_img:
                raise ValueError("Color mode not enabled")

            low_clips = self._unzip(low_clip)
            high_clips = self._unzip(high_clip)
            gamma = self._unzip(gamma_color)

            channels = []
            for i, color_band in enumerate(self.color_bands):
                channel = self.get_plot_data(
                    color_band,
                    stretch=stretch,
                    stretch_scale=stretch_scale,
                    stretch_interval=stretch_interval,
                    low_clip=low_clips[i],
                    high_clip=high_clips[i],
                    scale_method=scale_method,
                    _internal=True,
                )

                channels.append(channel ** gamma[i])
            return np.stack(channels, axis=2)

        band_state = self._bands[band]

        if band_state.config is None:
            band_state.config = {}
        band_config = band_state.config

        image, _ = self._get_aligned_image(band_state)

        if (
            band_state.stretched is None
            or band_config.get("stretch") != stretch
            or band_config.get("stretch_scale") != stretch_scale
            or band_config.get("stretch_interval") != stretch_interval
        ):
            stretch_func = self._get_stretch(stretch, stretch_scale)
            stretch_interval_func = self._get_interval(stretch_interval)

            band_state.stretched = self._stretch_image(
                image,
                stretch=stretch_func,
                stretch_interval=stretch_interval_func,
            )
            band_state.invalidate_from("stretch")
            band_config.update(
                {
                    "stretch": stretch,
                    "stretch_scale": stretch_scale,
                    "stretch_interval": stretch_interval,
                }
            )

        if (
            band_state.clipped is None
            or band_config.get("low_clip") != low_clip
            or band_config.get("high_clip") != high_clip
        ):
            band_state.clipped = self._clip_image(
                band_state.stretched,
                low=low_clip,
                high=high_clip,
            )
            band_state.invalidate_from("clip")
            band_config.update({"low_clip": low_clip, "high_clip": high_clip})

        if band_state.scaled is None or band_config.get("scale_method") != scale_method:
            band_state.scaled = self._scale_image(
                band_state.clipped,
                scale_method=scale_method,
            )
            band_state.invalidate_from("scale")  # in case something is added later
            band_config.update({"scale_method": scale_method})

        if band_state.scaled is None:
            raise RuntimeError(f"Scaling failed for band {band}")

        band_state.config = band_config
        return band_state.scaled

    def world2pixel(self, ra, dec, band):
        """Converts from Sky Coordinates to pixel coordinates if available
        returns the pixel coordinates in the resampled_frame
        """
        band = self.resolve_wcs_band(band)
        if band not in self._bands:
            raise ValueError(f"Band {band} not found")

        band_state = self._bands[band]
        wcs = band_state._resampled_wcs or band_state.wcs

        if wcs is None:
            raise ValueError(f"Band {band} has no WCS")

        coords = SkyCoord(ra=np.atleast_1d(ra) * u.deg, dec=np.atleast_1d(dec) * u.deg, frame="icrs")
        xpix, ypix = wcs.world_to_pixel(coords)
        return xpix, ypix

    def get_arcsec_per_pixel(self, band, scalar=True):
        """ Return the pixel scale (arcsec/pixel) for a given band.
        If the band has been resampled, use the resampled WCS
        """
        band = self.resolve_wcs_band(band)
        if band not in self._bands:
            raise ValueError(f"Band {band} not found")

        band_state = self._bands[band]
        wcs = band_state._resampled_wcs or band_state.wcs

        if wcs is None:
            raise ValueError(f"Band {band} has no WCS defined")

        scales_deg = proj_plane_pixel_scales(wcs)
        scales_arcsec = scales_deg * 3600.0

        if scalar:
            return np.mean(scales_arcsec)
        return scales_arcsec[0], scales_arcsec[1]


class ALMAPlotClass:
    def __init__(self, header, data, wcs=None, radius=None, coordinates=None):
        self.data = data
        self.header = header
        self.wcs = wcs

        if self.wcs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                self.wcs = WCS(self.header).celestial

        self.ndim = self.data.ndim
        self.coordinates = coordinates or SkyCoord(self.header["CRVAL1"], self.header["CRVAL2"], unit="deg")

        if radius is not None:
            self.image, self.image_wcs = self.make_cutout(radius=radius)
        else:
            self.image = self.data.squeeze()
            self.image_wcs = self.wcs

    def make_cutout(self, radius=5 * u.arcsec):
        if self.ndim == 3:
            plane = self.data[0, ...]
        elif self.ndim == 4:
            plane = self.data[0, 0, ...]
        else:
            plane = self.data

        cutout = Cutout2D(plane, position=self.coordinates, size=radius, wcs=self.wcs)
        y_slice = cutout.slices_original[0]
        x_slice = cutout.slices_original[1]

        if self.ndim == 3:  # (frequency, y, x)
            image = self.data[:, y_slice, x_slice].squeeze()
        elif self.ndim == 4:  # (stokes, frequency, y, x)
            image = self.data[:, :, y_slice, x_slice].squeeze()
        else:
            image = cutout.data

        return image, cutout.wcs

    def get_frequency_range(self):
        if self.ndim == 3:
            pixels = np.arange(0, self.data.shape[0], 1)
        elif self.ndim == 4:
            pixels = np.arange(0, self.data.shape[1], 1)
        else:
            raise ValueError("Data has no frequency range")

        self.frequencies = self.header["CRVAL3"] + (pixels - self.header["CRPIX3"]) * self.header["CDELT3"]
        return self.frequencies

    def get_wavelength_range(self):
        wavelength_range = 2.998e14 / self.get_frequency_range()
        return wavelength_range

    def map_wavelength_indexes(self, wav_start, wav_end):
        """Given a starting wavelength wav_start and and ending one wav_end (in microns) returns the indexes along the first axis which correspond to that interval"""
        f_start, f_end = 2.998e14 / wav_end, 2.998e14 / wav_start
        return self.map_frequency_indexes(f_start, f_end)

    def map_frequency_indexes(self, f_start, f_end):
        """Given a starting frequency f_start and and ending one f_end returns the indexes along the first axis which correspond to that interval"""
        if not hasattr(self, "frequencies"):
            _ = self.get_frequency_range()

        freq_array = self.frequencies
        low, high = (f_start, f_end) if f_start <= f_end else (f_end, f_start)
        idx = np.where((freq_array >= low) & (freq_array <= high))[0]
        if len(idx) == 0:
            raise ValueError("No spectral channels found in the requested frequency range.")
        return idx

    def get_beam_model(self, x0=0, y0=0):
        sigma_x = self.header['BMAJ'] / (2 * np.sqrt(2 * np.log(2))) / np.abs(self.header['CDELT1'])
        sigma_y = self.header['BMIN'] / (2 * np.sqrt(2 * np.log(2))) / np.abs(self.header['CDELT2'])
        theta = np.deg2rad(self.header['BPA'])
        beam = Gaussian2D(1, x0, y0, sigma_x, sigma_y, theta)
        return beam

    def get_weighted_spectrum(self, x0=None, y0=None, coordinates=None):
        if (x0 is None) or (y0 is None):
            coordinates = coordinates or self.coordinates
            x0, y0 = self.wcs.world_to_pixel(coordinates)
            x0, y0 = int(x0), int(y0)

        beam_model = self.get_beam_model()
        size_x = int(3 * beam_model.x_stddev.value)
        size_y = int(3 * beam_model.y_stddev.value)

        y_min, y_max = y0 - size_y, y0 + size_y + 1
        x_min, x_max = x0 - size_x, x0 + size_x + 1

        if self.ndim == 3:
            subcube = self.data[:, y_min:y_max, x_min:x_max]
        elif self.ndim == 4:
            subcube = self.data[0, :, y_min:y_max, x_min:x_max]
        else:
            raise ValueError("Unsupported data dimensions")

        yy, xx = np.mgrid[-size_y:size_y + 1, -size_x:size_x + 1]
        beam = beam_model(xx, yy)
        beam /= beam.sum()

        weighted_spectrum = (subcube * beam).sum(axis=(1, 2))
        return weighted_spectrum

