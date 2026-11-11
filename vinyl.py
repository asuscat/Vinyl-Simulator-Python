"""
Fluance RT85 Ultimate - Vinyl Physics Emulator  (v3: physically modelled + web-simulator controls)

Signal path (all time constants are in SECONDS, all state is carried across blocks):

  source -> RIAA record pre-emphasis (baked into the "groove")
         -> position-based playback with cubic interpolation (varispeed, reverse, scratching)
            * groove noise texture and click/pop/scratch defects live IN the groove
              (they pitch-shift, slow down, reverse and fade with stylus velocity)
            * Ghost Echo (print-through): loud passages bleed into the neighbouring grooves,
              one revolution before and after, so they are heard early and late
         -> cartridge = velocity sensor (output voltage proportional to stylus speed)
         -> stylus: radius-dependent tracing loss, mistracking, pinch / tangential /
            inner-groove distortion (vertical = L-R), bass-mono, crosstalk, HF peak
         -> groove wear (Vinyl Age), stylus resonance, worn stylus
         -> + rumble, warp, needle-drop thump (through arm resonance) + mains hum
         -> RIAA playback preamp (so everything at cartridge level is shaped correctly)
         -> subsonic filter, loudness EQ, Warmth (low-mid lift, top cut, soft saturation),
            RIAA mismatch tilt, preamp saturation, electronic noise floor
         -> optional suitcase-player stages and low-fidelity sample-rate profile
         -> volume, mute, soft limiter

The "Classic Controls" tab carries the same ten controls as the web Vinyl Simulator:
Warmth, Surface Noise, Wow & Flutter, Crackle, Vinyl Age, Stylus Resonance, Worn Stylus,
Turntable Rumble, Ghost Echo and RIAA Mismatch, each with a "?" explanation.

Dependencies: numpy scipy sounddevice soundfile PyQt6   (pydub optional for odd formats)
"""
import sys
import math
import os
import time
import copy
import json
import threading
import traceback
from math import gcd

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import (
    butter, sosfilt, sosfreqz, bilinear, tf2sos, resample_poly, lfilter,
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFrame, QMessageBox, QGridLayout, QCheckBox,
    QProgressDialog, QFileDialog, QComboBox, QTabWidget, QInputDialog,
)


# ============================================================
# CRASH HANDLER
# ============================================================
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    err = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(err, file=sys.stderr)
    app = QApplication.instance()
    if app:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("Application Error")
        msg.setDetailedText(err)
        msg.setWindowTitle("Crash Report")
        msg.exec()


sys.excepthook = handle_exception

# DSP ENGINE START ===========================================================
TWO_PI = 2.0 * math.pi
REV_HZ_33 = (100.0 / 3.0) / 60.0          # 0.5556 rev/s  (1.8 s per revolution)
LEAD_IN_S = 2.0                           # silent lead-in groove
RUN_OUT_S = 4.0                           # run-out groove (locks in the last revolution)
LOCK_REV_S = 1.0 / REV_HZ_33
R_OUTER_MM = 146.0                        # 12" LP geometry
R_INNER_MM = 60.3
OFF_DB = -99.5                            # dB sliders at/below this value mean "off"
COG_HZ = 7.4                              # motor pulley ripple frequency at 33 1/3
HUM_W = np.array([1.0, 0.55, 0.30, 0.18, 0.10])
HUM_PH = np.array([0.3, 1.9, 4.1, 2.2, 5.0])
HUM_NORM = 1.0 / math.sqrt(np.sum(HUM_W ** 2) / 2.0)


def db2lin(db):
    return 10.0 ** (db / 20.0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-30))


def rbj(kind, f0, fs, q=0.70710678, gain_db=0.0):
    """RBJ cookbook biquad, returned as a (1,6) second-order-section row."""
    f0 = float(np.clip(f0, 1.0, fs * 0.49))
    w0 = TWO_PI * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq),
             2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq,
             -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq),
             -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq,
             2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def riaa_filters(fs):
    """Exact RIAA playback filter and its (band-limited) exact inverse, both 0 dB @ 1 kHz.
    The record-side filter has an extra pole at 300 kHz so it is realisable and stable;
    the pair is flat to within ~0.3 dB up to 20 kHz."""
    t1, t2, t3 = 3180e-6, 318e-6, 75e-6
    t4 = 1.0 / (TWO_PI * 300e3)
    b, a = bilinear([t2, 1.0], np.polymul([t1, 1.0], [t3, 1.0]), fs)
    play = tf2sos(b, a)
    b, a = bilinear(np.polymul([t1, 1.0], [t3, 1.0]), np.polymul([t2, 1.0], [t4, 1.0]), fs)
    rec = tf2sos(b, a)

    def g(sos):
        return abs(sosfreqz(sos, worN=np.array([TWO_PI * 1000.0 / fs]))[1][0])

    play[0, :3] /= g(play)
    rec[0, :3] /= g(rec)
    return play, rec


class SOSFilter:
    """Cascaded biquads with state carried across blocks; coefficients can change on the fly."""

    def __init__(self, channels=2):
        self.ch = channels
        self.sos = None
        self.zi = None
        self.key = "__unset__"

    def set(self, key, design):
        if key == self.key:
            return
        self.key = key
        sos = design()
        if sos is None:
            self.sos = None
            self.zi = None
            return
        sos = np.atleast_2d(sos)
        if self.zi is None or self.zi.shape[0] != sos.shape[0]:
            self.zi = np.zeros((sos.shape[0], 2, self.ch))
        self.sos = sos

    def process(self, x):
        if self.sos is None:
            return x
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class NoiseGen:
    """Stateful filtered Gaussian noise, normalised to unit RMS (exactly, via impulse response)."""

    def __init__(self, sos, channels, rng, impulse_len):
        self.sos = sos
        self.ch = channels
        self.rng = rng
        self.zi = np.zeros((sos.shape[0], 2, channels))
        imp = np.zeros(int(impulse_len))
        imp[0] = 1.0
        h = sosfilt(sos, imp)
        self.norm = 1.0 / math.sqrt(float(np.sum(h * h)))

    def next(self, n):
        w = self.rng.standard_normal((n, self.ch))
        y, self.zi = sosfilt(self.sos, w, axis=0, zi=self.zi)
        return y * self.norm


def make_noise_texture(sr, rng, seconds=20.0):
    """Looping stereo groove-roughness texture (unit RMS). Pink + white, vertical-dominant."""
    T = int(seconds * sr)
    F = int(0.25 * sr)
    pre = int(sr)
    pb = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    pa = [1.0, -2.494956002, 2.017265875, -0.522189400]
    hp = butter(2, 150.0, "highpass", fs=sr, output="sos")

    def one():
        w = rng.standard_normal(T + F + pre)
        pink = lfilter(pb, pa, w)
        pink /= np.std(pink[pre:])
        return sosfilt(hp, (0.8 * pink + 0.6 * w))[pre:]

    m, s = one(), one()
    x = np.column_stack([0.6 * m + 1.0 * s, 0.6 * m - 1.0 * s])
    fade = np.linspace(0.0, 1.0, F)[:, None]
    y = x[:T].copy()
    y[:F] = x[:F] * np.sqrt(fade) + x[T:T + F] * np.sqrt(1.0 - fade)   # seamless loop
    y /= rms(y[sr:])
    return y.astype(np.float32)


class DefectClass:
    """Scratches, dust and pressing flaws fixed at positions in the groove.
    Waveforms are functions of groove distance, so they stretch, reverse and
    fade with stylus velocity exactly like real defects."""

    def __init__(self, kind, sr, arrays, max_rate, noise_tbl):
        self.kind = kind
        self.sr = sr
        self.inv_sr = 1.0 / sr
        self.p = arrays
        self.max_rate = max_rate
        self.tbl = noise_tbl
        self._rate = None
        self.a = {k: v[:0] for k, v in arrays.items()}
        self.max_dur = 0.0

    @classmethod
    def random(cls, kind, sr, end, max_rate, rng, noise_tbl):
        n = max(1, int(end / sr * max_rate))
        pos = np.sort(rng.uniform(0, end, n))
        sign = rng.choice([-1.0, 1.0], n)
        if kind == "tick":
            amp = rng.lognormal(math.log(0.05), 0.8, n)
            f0 = rng.uniform(2500, 7000, n)
            tau = rng.uniform(0.00008, 0.0003, n)
            dur = tau * 7
            tau_n = tau
        elif kind == "pop":
            amp = rng.lognormal(math.log(0.10), 0.7, n)
            f0 = rng.uniform(300, 1500, n)
            tau = rng.uniform(0.0006, 0.0025, n)
            dur = tau * 7
            tau_n = tau
        elif kind == "thump":
            amp = rng.lognormal(math.log(0.012), 0.6, n)
            f0 = rng.uniform(40, 110, n)
            tau = rng.uniform(0.010, 0.035, n)
            dur = tau * 6
            tau_n = tau
        else:  # scratch: sharp tick followed by a burst of rough noise
            amp = rng.lognormal(math.log(0.12), 0.6, n)
            f0 = rng.uniform(1500, 4000, n)
            tau = rng.uniform(0.0003, 0.001, n)
            dur = rng.uniform(0.008, 0.04, n)
            tau_n = dur / 3.0
        phi = rng.uniform(0, TWO_PI, n)
        arrays = dict(pos=pos, amp=amp * sign, f0=f0, tau=tau, dur=dur,
                      cl=np.cos(phi), cr=np.sin(phi), tau_n=tau_n,
                      off=rng.integers(0, len(noise_tbl), n), rank=rng.random(n))
        return cls(kind, sr, arrays, max_rate, noise_tbl)

    @classmethod
    def fixed(cls, kind, sr, pos, amp, f0, tau, noise_tbl):
        one = lambda v: np.array([float(v)])
        arrays = dict(pos=one(pos), amp=one(amp), f0=one(f0), tau=one(tau), dur=one(tau * 7),
                      cl=one(0.7), cr=one(-0.7), tau_n=one(tau), off=np.zeros(1, dtype=np.int64),
                      rank=one(0.0))
        return cls(kind, sr, arrays, 1.0, noise_tbl)

    def clone(self):
        c = copy.copy(self)
        c._rate = None
        return c

    def set_rate(self, rate):
        if rate == self._rate:
            return
        self._rate = rate
        frac = min(1.0, max(0.0, rate / self.max_rate))
        mask = self.p["rank"] < frac
        self.a = {k: v[mask] for k, v in self.p.items()}
        self.max_dur = float(self.a["dur"].max()) if mask.any() else 0.0

    def render(self, ph, out, gain):
        a = self.a
        pos = a["pos"]
        if gain <= 0.0 or len(pos) == 0:
            return
        i0 = np.searchsorted(pos, ph.min() - self.max_dur * self.sr, "left")
        i1 = np.searchsorted(pos, ph.max(), "right")
        scratch = self.kind == "scratch"
        for k in range(i0, i1):
            u = (ph - pos[k]) * self.inv_sr
            m = (u >= 0.0) & (u < a["dur"][k])
            if not m.any():
                continue
            uu = u[m]
            w = np.exp(-uu / a["tau"][k]) * np.sin(TWO_PI * a["f0"][k] * uu)
            if scratch:
                ni = ((uu * self.sr).astype(np.int64) + a["off"][k]) % len(self.tbl)
                w = w + 0.6 * self.tbl[ni] * np.exp(-uu / a["tau_n"][k])
            w *= a["amp"][k] * gain
            out[m, 0] += w * a["cl"][k]
            out[m, 1] += w * a["cr"][k]


class TurntableDSP:
    FILTER_SPECS = {
        "lp": 2, "bassmono": 1, "hp_sq": 1, "xtalk": 2, "peak": 2, "res": 2,
        "riaa": 2, "subsonic": 2, "loud": 2, "honk": 2, "r_hp": 2, "r_lp": 2,
        "r_box": 2, "lofi": 2,
        # web-simulator controls
        "age": 2, "stres": 2, "worn": 2, "warm_pk": 2, "warm_hs": 2, "mm_lo": 2, "mm_hi": 2,
    }

    def __init__(self, audio_data, sample_rate, device_sr=None, riaa=True):
        src_sr = int(sample_rate)
        self.sr = int(device_sr or src_sr)
        a = np.nan_to_num(np.asarray(audio_data, dtype=np.float32))
        if a.ndim == 1:
            a = np.column_stack([a, a])
        elif a.shape[1] == 1:
            a = np.column_stack([a[:, 0], a[:, 0]])
        elif a.shape[1] > 2:
            a = a[:, :2]
        if src_sr != self.sr:
            g = gcd(src_sr, self.sr)
            a = resample_poly(a, self.sr // g, src_sr // g, axis=0).astype(np.float32)
        self._raw = np.ascontiguousarray(a)
        self.n_frames = len(a)
        self.lead = int(LEAD_IN_S * self.sr)
        self.n_total = self.lead + self.n_frames + int(RUN_OUT_S * self.sr)
        self.lock_len = int(LOCK_REV_S * self.sr)
        self.lock_start = self.n_total - self.lock_len

        self._lock = threading.Lock()
        self._set_defaults()
        self.riaa_play, self.riaa_rec = riaa_filters(self.sr)
        self.riaa_on = bool(riaa)
        self.music = self._build_music(self.riaa_on)

        rng = np.random.default_rng()
        self.tex = make_noise_texture(self.sr, rng)
        tbl = rng.standard_normal(1 << 15)
        self.classes = {
            "tick": DefectClass.random("tick", self.sr, self.n_total, 80.0, rng, tbl),
            "pop": DefectClass.random("pop", self.sr, self.n_total, 20.0, rng, tbl),
            "thump": DefectClass.random("thump", self.sr, self.n_total, 2.0, rng, tbl),
            "scratch": DefectClass.random("scratch", self.sr, self.n_total, 1.0, rng, tbl),
            # once-per-revolution thump of the locked run-out groove
            "lock": DefectClass.fixed("thump", self.sr, self.lock_start, 0.02, 55.0, 0.03, tbl),
        }
        self._rumble_sos = butter(2, [12.0, 110.0], "bandpass", fs=self.sr, output="sos")
        self._init_state()
        self._recalibrate()

    # ---------------------------------------------------------- defaults / state
    def _set_defaults(self):
        # transport
        self.motor_on = True
        self.speed_select = 1.0            # 1.0 = 33 1/3, 1.35 = 45
        self.pitch_pct = 0.0
        self.tonearm_lifted = True
        self.needle_event = None
        self.contact = 0.0
        self.speed = 1.0
        self.playhead = 0.0
        self.platter_rev = 0.0
        self.is_holding = False
        self.hand_target = 0.0
        self.slider_pressed = False
        self.slider_scrub_mode = False
        self.slider_seek_pct = 0.0
        # motor / platter
        self.spinup_time = 1.8
        self.wow_amt = 0.0006
        self.flutter_amt = 0.0002
        self.drift_amt = 0.00025
        self.cogging_amt = 0.00003
        self.hum_db = -74.0
        self.mains_hz = 60.0
        self.rumble_db = -80.0
        self.warp_amt = 0.25
        self.res_freq = 10.0
        self.res_db = 8.0
        # stylus / cartridge
        self.stylus_fc = 19000.0
        self.igd_amount = 0.40
        self.tracking_limit = 3.0
        self.pinch_amount = 0.08
        self.tangential_deg = 1.5
        self.bass_mono_hz = 120.0
        self.crosstalk_amt = 0.05
        self.cart_peak_db = 2.0
        # surface
        self.hiss_db = -66.0
        self.crackle_rate, self.crackle_level_db = 12.0, 0.0
        self.pop_rate, self.pop_level_db = 0.6, 0.0
        self.thump_rate, self.thump_level_db = 0.2, 0.0
        self.scratch_rate, self.scratch_level_db = 0.03, 0.0
        # electronics
        self.subsonic_hz = 18.0
        self.sat_drive = 0.08
        self.fm_strength = 0.0
        self.noise_floor_db = -96.0
        self.output_gain = 0.85
        # web-simulator controls (all 0-1; riaa_mismatch is neutral at 0.5)
        self.warmth = 0.0
        self.vinyl_age = 0.0
        self.stylus_res = 0.0
        self.worn_stylus = 0.0
        self.ghost_echo = 0.0
        self.riaa_mismatch = 0.5
        # player profile
        self.fidelity_mode = 0
        self.r608_distortion_mode = False
        self.r608_distortion_intensity = 0.5
        self.r608_comp_mode = False
        self.r608_comp_intensity = 0.5

    def _init_state(self):
        sr = self.sr
        rng = np.random.default_rng()
        self._rng = rng
        self.filters = {k: SOSFilter(c) for k, c in self.FILTER_SPECS.items()}
        self._flutter = NoiseGen(butter(2, [3.0, 25.0], "bandpass", fs=sr, output="sos"), 1, rng, 2 * sr)
        self._drift = NoiseGen(butter(1, 0.06, "lowpass", fs=sr, output="sos"), 1, rng,
                               int(12 * sr / (TWO_PI * 0.06)))
        self._rumble = NoiseGen(self._rumble_sos, 2, rng, 2 * sr)
        self.cog_phase = 0.0
        self.hum_phase = 0.0
        self._hold_hist = np.zeros((8, 2))
        self._hold_pos = 0
        self._vol_prev = None
        self._ghost_g = 0.0
        self.block_counter = 0

    def clone_for_render(self):
        """Independent engine for offline export: shares the big read-only arrays."""
        c = copy.copy(self)
        c._lock = threading.Lock()
        c.classes = {k: v.clone() for k, v in self.classes.items()}
        c._init_state()
        c.playhead = 0.0
        c.tonearm_lifted = False
        c.contact = 1.0
        c.speed = c.speed_select * (1.0 + c.pitch_pct / 100.0) if c.motor_on else 0.0
        c.is_holding = False
        c.slider_pressed = False
        c.slider_scrub_mode = False
        c.needle_event = "drop"
        return c

    # ---------------------------------------------------------- source / RIAA
    def _preemphasis(self, x):
        out = np.empty(x.shape, dtype=np.float32)
        zi = np.zeros((self.riaa_rec.shape[0], 2, 2))
        step = 1 << 20
        for s in range(0, len(x), step):
            y, zi = sosfilt(self.riaa_rec, x[s:s + step].astype(np.float64), axis=0, zi=zi)
            out[s:s + len(y)] = y
        return out

    def _build_music(self, riaa_on):
        x = self._preemphasis(self._raw) if riaa_on else self._raw
        m = np.zeros((self.n_total, 2), dtype=np.float32)
        m[self.lead:self.lead + self.n_frames] = x
        return m

    def set_riaa(self, enabled):
        enabled = bool(enabled)
        if enabled == self.riaa_on:
            return
        new = self._build_music(enabled)          # heavy part outside the lock
        with self._lock:
            self.music = new
            self.riaa_on = enabled
            self._recalibrate()

    def _mix_rumble(self, r):
        a, b = r[:, 0], r[:, 1]
        return np.column_stack([0.55 * a + 0.85 * b, 0.55 * a - 0.85 * b]) / 1.012

    def _recalibrate(self):
        """Cartridge-level generators are scaled so their sliders read dBFS AFTER the RIAA stage."""
        sr = self.sr
        self._cal_key = (self.riaa_on, self.mains_hz)
        if not self.riaa_on:
            self.cal_tex = self.cal_rumble = self.cal_lf = 1.0
            self.hum_cal = np.ones(len(HUM_W))
            return
        H = self.riaa_play

        def gain(f):
            if f >= 0.48 * sr:
                return 1.0
            return abs(sosfreqz(H, worN=np.array([TWO_PI * f / sr]))[1][0])

        seg = self.tex[:4 * sr].astype(np.float64)
        y = sosfilt(H, seg, axis=0)
        self.cal_tex = rms(seg[sr // 2:]) / rms(y[sr // 2:])
        g = NoiseGen(self._rumble_sos, 2, np.random.default_rng(1), 2 * sr)
        x = self._mix_rumble(g.next(4 * sr))
        y = sosfilt(H, x, axis=0)
        self.cal_rumble = rms(x[sr:]) / rms(y[sr:])
        self.cal_lf = 1.0 / gain(1.5)
        self.hum_cal = np.array([1.0 / gain(self.mains_hz * (k + 1)) for k in range(len(HUM_W))])

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _cubic(arr, idx, mu, wrap):
        L = len(arr)
        if wrap:
            j = [(idx + k) % L for k in (-1, 0, 1, 2)]
        else:
            j = [np.clip(idx + k, 0, L - 1) for k in (-1, 0, 1, 2)]
        ym1, y0, y1, y2 = (arr[i] for i in j)
        a0 = -0.5 * ym1 + 1.5 * y0 - 1.5 * y1 + 0.5 * y2
        a1 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2
        a2 = -0.5 * ym1 + 0.5 * y1
        return ((a0 * mu + a1) * mu + a2) * mu + y0

    def _print_through(self, idx, mu, n):
        """Ghost Echo. A loud passage presses a faint copy of itself into the walls of the
        neighbouring grooves, one revolution (1.8 s) away. The copy one groove outward is
        heard BEFORE the loud sound (pre-echo, the stronger one); the one inward AFTER it.
        It only appears when the passage is loud, and it is read from the groove itself,
        so it slows, stops and runs backwards with the record."""
        L = self.lock_len
        pre = self._cubic(self.music, idx + L, mu, False)
        post = self._cubic(self.music, idx - L, mu, False)
        level = math.sqrt(0.5 * (float(np.mean(pre * pre)) + float(np.mean(post * post))))
        target = min(1.0, level / 0.2) * self.ghost_echo
        g0 = self._ghost_g
        self._ghost_g = target
        g = np.linspace(g0, target, n)[:, None]
        return (0.15 * pre + 0.06 * post) * g

    @staticmethod
    def _warm_saturate(x, w):
        """Soft saturation with a loudness-neutral reference: k grows with Warmth, quiet
        passages are lifted and peaks are rounded off. y = (pi+k)x / (pi+k|x|)."""
        k = w * 30.0 + 1.0
        y = (math.pi + k) * x / (math.pi + k * np.abs(x))
        ref = 0.3
        return y * ((math.pi + k * ref) / (math.pi + k))

    def _hold(self, x, tsr):
        """Zero-order sample-and-hold at a lower rate (imaging/aliasing of cheap players)."""
        n = len(x)
        r = self.sr / tsr
        need = int(math.ceil(r)) + 2
        h = self._hold_hist
        if len(h) < need:
            h = np.vstack([np.zeros((need - len(h), 2)), h])
        g = self._hold_pos + np.arange(n)
        src = np.floor(np.floor(g / r) * r).astype(np.int64)
        buf = np.vstack([h, x])
        rel = np.clip(src - (self._hold_pos - len(h)), 0, len(buf) - 1)
        y = buf[rel]
        self._hold_hist = buf[-need:]
        self._hold_pos += n
        return y

    @property
    def progress(self):
        return float(np.clip((self.playhead - self.lead) / max(1, self.n_frames), 0.0, 1.0))

    @property
    def in_runout(self):
        return self.playhead >= self.lock_start

    # ---------------------------------------------------------- the engine
    def process_block(self, n, volume_val):
        with self._lock:
            return self._process(int(n), volume_val)

    def _process(self, n, volume_val):
        sr = self.sr
        dt = n / sr
        F = self.filters
        self.block_counter += 1
        if self._cal_key != (self.riaa_on, self.mains_hz):
            self._recalibrate()
        vol_norm = float(np.clip(volume_val / 100.0, 0.0, 1.0))
        C = self.classes
        C["tick"].set_rate(self.crackle_rate)
        C["pop"].set_rate(self.pop_rate)
        C["thump"].set_rate(self.thump_rate)
        C["scratch"].set_rate(self.scratch_rate)
        C["lock"].set_rate(1.0)

        # ---- tonearm contact (0.15 s lift / lower)
        c0 = self.contact
        c_t = 0.0 if self.tonearm_lifted else 1.0
        c1 = c0 + float(np.clip(c_t - c0, -dt / 0.15, dt / 0.15))
        self.contact = c1
        contact = np.linspace(c0, c1, n)
        ev, self.needle_event = self.needle_event, None

        # ---- platter speed: first-order motor/inertia response, exact per-sample
        motor_target = self.speed_select * (1.0 + self.pitch_pct / 100.0) if self.motor_on else 0.0
        seeking = self.slider_pressed and not self.slider_scrub_mode
        hand = self.is_holding or (self.slider_pressed and self.slider_scrub_mode)
        s0 = self.speed
        if hand:
            self.hand_target *= math.exp(-dt / 0.07)      # a still hand stops the platter
            tgt, tau = self.hand_target, 0.025
        else:
            tgt = motor_target
            tau = max(0.02, self.spinup_time if tgt >= s0 else self.spinup_time * 0.6)
        t = np.arange(1, n + 1) / sr
        s_base = tgt + (s0 - tgt) * np.exp(-t / tau)
        self.speed = float(s_base[-1])

        # ---- speed errors, locked to platter rotation (wow) or real time (flutter, drift)
        cum = np.cumsum(s_base)
        rev = self.platter_rev + cum * (REV_HZ_33 / sr)
        cph = self.cog_phase + TWO_PI * COG_HZ * cum / sr
        mod = (self.wow_amt * (np.sin(TWO_PI * rev + 0.7) + 0.35 * np.sin(2 * TWO_PI * rev + 2.1))
               + self.flutter_amt * self._flutter.next(n)[:, 0]
               + self.drift_amt * self._drift.next(n)[:, 0]
               + self.cogging_amt * np.sin(cph))
        self.platter_rev = float(rev[-1] % 1.0)
        self.cog_phase = float(cph[-1] % TWO_PI)
        s_true = s_base * (1.0 + mod)

        # ---- stylus position in the groove (stationary while the arm is up)
        if seeking:
            self.playhead = float(np.clip(self.lead + self.slider_seek_pct * self.n_frames,
                                          0, self.n_total - 1))
            sv = np.zeros(n)
        elif c1 > 0.01 or c0 > 0.01:
            sv = s_true
        else:
            sv = np.zeros(n)
        ph = self.playhead + np.cumsum(sv)
        np.maximum(ph, 0.0, out=ph)
        over = ph >= self.lock_start                       # locked run-out groove
        if over.any():
            ph[over] = self.lock_start + np.mod(ph[over] - self.lock_start, self.lock_len)
        self.playhead = float(ph[-1])
        p = self.progress

        # ---- groove read-out: music + roughness texture + defects, all position based
        idx = np.floor(ph).astype(np.int64)
        mu = (ph - idx)[:, None]
        groove = self._cubic(self.music, idx, mu, False)
        if self.ghost_echo > 0.0 or self._ghost_g > 1e-4:
            groove = groove + self._print_through(idx, mu, n)
        hiss_g = 0.0 if self.hiss_db <= OFF_DB else db2lin(self.hiss_db) * self.cal_tex * (1.0 + 0.35 * p)
        if hiss_g > 0.0:
            groove = groove + self._cubic(self.tex, idx, mu, True) * hiss_g
        defects = np.zeros((n, 2))
        C["tick"].render(ph, defects, db2lin(self.crackle_level_db))
        C["pop"].render(ph, defects, db2lin(self.pop_level_db))
        C["thump"].render(ph, defects, db2lin(self.thump_level_db))
        C["scratch"].render(ph, defects, db2lin(self.scratch_level_db))
        C["lock"].render(ph, defects, 1.0)
        groove = groove + defects
        # velocity-sensing cartridge: output is proportional to stylus speed (zero when stopped)
        cart = groove * (sv * contact)[:, None]

        # ---- stylus & cartridge mechanics
        fc = self.stylus_fc * (1.0 - self.igd_amount * 0.586 * p)     # tracing loss ~ groove radius
        if fc < 0.45 * sr:
            F["lp"].set(("lp", int(fc / 100)), lambda: rbj("lp", fc, sr, 0.7071))
        else:
            F["lp"].set(("off",), lambda: None)
        cart = F["lp"].process(cart)
        T = max(0.3, self.tracking_limit * (1.0 - 0.35 * self.igd_amount * p))
        cart = T * np.tanh(cart / T)                                  # mistracking

        m = 0.5 * (cart[:, 0] + cart[:, 1])                           # lateral
        s = 0.5 * (cart[:, 0] - cart[:, 1])                           # vertical
        if self.bass_mono_hz >= 20.0:
            bm = self.bass_mono_hz
            F["bassmono"].set(("bm", round(bm)), lambda: butter(2, bm, "highpass", fs=sr, output="sos"))
            s = F["bassmono"].process(s[:, None])[:, 0]
        else:
            F["bassmono"].set(("off",), lambda: None)
        r_mm = R_OUTER_MM - (R_OUTER_MM - R_INNER_MM) * p
        eps = math.radians(self.tangential_deg * (r_mm - 66.0) * (r_mm - 120.0) / r_mm / 14.25)
        c2 = self.pinch_amount * 0.15 + abs(math.tan(eps)) * 0.5
        if c2 > 0.0:
            F["hp_sq"].set(("hp60",), lambda: butter(1, 60.0, "highpass", fs=sr, output="sos"))
            s = s + c2 * F["hp_sq"].process((m * m)[:, None])[:, 0]   # 2nd harmonic, vertical
        s = s + math.sin(eps) * m                                     # axis rotation leakage
        cart = np.column_stack([m + s, m - s])
        d3 = self.igd_amount * 0.05 * p * p
        if d3 > 0.0:
            cart = cart + d3 * cart ** 3                              # inner-groove distortion

        if self.crosstalk_amt > 0.0:
            F["xtalk"].set(("hp3k5",), lambda: butter(1, 3500.0, "highpass", fs=sr, output="sos"))
            hf = F["xtalk"].process(cart)
            c = self.crosstalk_amt
            cart = np.column_stack([cart[:, 0] + c * (cart[:, 1] + 2.5 * hf[:, 1]),
                                    cart[:, 1] + c * (cart[:, 0] + 2.5 * hf[:, 0])])
        pk = self.cart_peak_db
        if pk > 0.05 and 14500.0 < 0.46 * sr:
            F["peak"].set(("pk", round(pk * 10)), lambda: rbj("peak", 14500.0, sr, 1.2, pk))
        else:
            F["peak"].set(("off",), lambda: None)
        cart = F["peak"].process(cart)

        # ---- groove wear and stylus condition (the web simulator's Vinyl Age / Stylus
        # Resonance / Worn Stylus). All three act on the signal the stylus picks up, so the
        # surface noise and defects are dulled along with the music; rumble is not.
        age = self.vinyl_age
        fa = 14500.0 - age * 6500.0                    # played-to-death records lose their top
        if age > 0.005 and fa < 0.45 * sr:
            F["age"].set(("age", round(age * 200)), lambda: rbj("lp", fa, sr, 0.5))
        else:
            F["age"].set(("off",), lambda: None)
        cart = F["age"].process(cart)

        sres = self.stylus_res
        if sres > 0.005 and 10000.0 < 0.46 * sr:       # cantilever/stylus-mass resonance
            F["stres"].set(("sres", round(sres * 100)), lambda: rbj("peak", 10000.0, sr, 2.5, 6.0 * sres))
        else:
            F["stres"].set(("off",), lambda: None)
        cart = F["stres"].process(cart)

        worn = self.worn_stylus
        if worn > 0.005 and 8000.0 < 0.46 * sr:        # rounded tip can't trace fine detail
            F["worn"].set(("worn", round(worn * 100)), lambda: rbj("highshelf", 8000.0, sr, 0.7071, -8.0 * worn))
        else:
            F["worn"].set(("off",), lambda: None)
        cart = F["worn"].process(cart)

        # ---- low-frequency mechanical stream: rumble, warp, needle drop, via arm resonance
        lf = np.zeros((n, 2))
        spin = np.abs(s_base) * contact
        if self.rumble_db > OFF_DB:
            lf += self._mix_rumble(self._rumble.next(n)) * (db2lin(self.rumble_db) * self.cal_rumble * spin)[:, None]
        if self.warp_amt > 0.0:
            w = (np.sin(TWO_PI * rev + 0.3) + 0.6 * np.sin(2 * TWO_PI * rev + 1.1)
                 + 0.3 * np.sin(3 * TWO_PI * rev + 2.4))
            wv = w * self.warp_amt * 0.02 * self.cal_lf * spin
            lf[:, 0] += wv
            lf[:, 1] -= wv
        if ev:
            amp = (1.6 if ev == "drop" else 0.4) * self.cal_lf
            lf[0, 0] += amp
            lf[0, 1] -= 0.8 * amp
        rf, rdb = self.res_freq, self.res_db
        F["res"].set(("res", round(rf * 10), round(rdb * 10)),
                     lambda: rbj("peak", rf, sr, 3.0, rdb) if rdb > 0.05 else None)
        cart = cart + F["res"].process(lf)

        # ---- mains hum (electromagnetic pickup, present even with the arm lifted)
        if self.hum_db > OFF_DB and (self.motor_on or self.speed > 0.05):
            ang = self.hum_phase + TWO_PI * self.mains_hz * np.arange(n) / sr
            hum = np.zeros(n)
            for k in range(len(HUM_W)):
                hum += HUM_W[k] * self.hum_cal[k] * np.sin((k + 1) * ang + HUM_PH[k])
            hum *= HUM_NORM * db2lin(self.hum_db)
            self.hum_phase = float((self.hum_phase + TWO_PI * self.mains_hz * n / sr) % TWO_PI)
            cart[:, 0] += hum
            cart[:, 1] += 0.85 * hum

        # ---- phono preamp
        if self.riaa_on:
            F["riaa"].set(("riaa",), lambda: self.riaa_play)
        else:
            F["riaa"].set(("off",), lambda: None)
        out = F["riaa"].process(cart)
        hz = self.subsonic_hz
        F["subsonic"].set(("sub", round(hz * 2)),
                          lambda: butter(2, hz, "highpass", fs=sr, output="sos") if hz >= 1.0 else None)
        out = F["subsonic"].process(out)
        k = self.fm_strength * float(np.clip((0.8 - vol_norm) / 0.8, 0.0, 1.0))
        F["loud"].set(("fm", round(k * 40)),
                      lambda: np.vstack([rbj("lowshelf", 150.0, sr, 0.6, 10.0 * k),
                                         rbj("highshelf", 6500.0, sr, 0.6, 4.0 * k)]) if k > 0.02 else None)
        out = F["loud"].process(out)

        # ---- Warmth: low-mid lift at 220 Hz, treble roll-off above 6 kHz, soft saturation
        wm = self.warmth
        if wm > 0.005:
            F["warm_pk"].set(("wpk", round(wm * 100)), lambda: rbj("peak", 220.0, sr, 0.8, 5.0 * wm))
            F["warm_hs"].set(("whs", round(wm * 100)), lambda: rbj("highshelf", 6000.0, sr, 0.7071, -3.0 * wm))
        else:
            F["warm_pk"].set(("off",), lambda: None)
            F["warm_hs"].set(("off",), lambda: None)
        out = F["warm_hs"].process(F["warm_pk"].process(out))

        # ---- RIAA mismatch: below the middle = warm/lush, above = bright/punchy
        mm = (self.riaa_mismatch - 0.5) * 2.0
        if abs(mm) > 0.01:
            F["mm_lo"].set(("mml", round(mm * 100)), lambda: rbj("lowshelf", 200.0, sr, 0.7071, -3.0 * mm))
            F["mm_hi"].set(("mmh", round(mm * 100)), lambda: rbj("highshelf", 3000.0, sr, 0.7071, 4.0 * mm))
        else:
            F["mm_lo"].set(("off",), lambda: None)
            F["mm_hi"].set(("off",), lambda: None)
        out = F["mm_hi"].process(F["mm_lo"].process(out))
        if wm > 0.005:
            out = self._warm_saturate(out, wm)

        if self.sat_drive > 0.0:
            g = 1.0 + self.sat_drive * 2.0
            out = np.tanh(g * out) / g
        if self.noise_floor_db > OFF_DB:
            out = out + self._rng.standard_normal((n, 2)) * db2lin(self.noise_floor_db)

        # ---- R608 suitcase player
        if self.r608_distortion_mode and self.r608_distortion_intensity > 0.0:
            a = self.r608_distortion_intensity
            F["honk"].set(("honk",), lambda: rbj("bp", 4200.0, sr, 1.2))
            out = out + F["honk"].process(out) * (0.6 * a)
            d = 1.0 + a * 6.0
            out = np.sign(out) * (1.0 - np.exp(-np.abs(out) * d)) / d * (1.0 + 0.45 * (d - 1.0))
        else:
            F["honk"].set(("off",), lambda: None)
        if self.r608_comp_mode and self.r608_comp_intensity > 0.0:
            a = self.r608_comp_intensity
            g = 1.0 + a * 4.0
            out = np.tanh(g * out) / np.tanh(g)
            hp_f, lp_f = 100.0 + 450.0 * a, 9000.0 - 5500.0 * a
            F["r_hp"].set(("rhp", round(hp_f)), lambda: butter(2, hp_f, "highpass", fs=sr, output="sos"))
            F["r_lp"].set(("rlp", round(lp_f)),
                          lambda: butter(2, lp_f, "lowpass", fs=sr, output="sos") if lp_f < 0.45 * sr else None)
            F["r_box"].set(("rbx", round(a * 30)), lambda: rbj("peak", 800.0, sr, 1.0, 3.0 * a))
            out = F["r_box"].process(F["r_lp"].process(F["r_hp"].process(out)))
        else:
            for nm in ("r_hp", "r_lp", "r_box"):
                F[nm].set(("off",), lambda: None)

        # ---- player fidelity profile (band-limit + sample-and-hold)
        mode = self.fidelity_mode
        tsr = {1: 44100, 2: 22050, 3: 11025}.get(mode)
        if tsr is not None and tsr < sr * 0.999:
            F["lofi"].set(("lofi", mode, sr),
                          lambda: butter(4 if mode == 1 else 2, min(0.42 * tsr, 0.45 * sr), "lowpass",
                                         fs=sr, output="sos"))
            out = self._hold(F["lofi"].process(out), tsr)
        else:
            F["lofi"].set(("off",), lambda: None)

        # ---- volume (smoothed), soft limiter
        vg = vol_norm ** 2 * self.output_gain
        v0 = vg if self._vol_prev is None else self._vol_prev
        out = out * np.linspace(v0, vg, n)[:, None]
        self._vol_prev = vg
        ax = np.abs(out)
        out = np.where(ax <= 0.8, out, np.sign(out) * (0.8 + 0.2 * np.tanh((ax - 0.8) / 0.2)))
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out, -1.0, 1.0).astype(np.float32)
# DSP ENGINE END =============================================================


# ============================================================
# AUDIO FILE HELPERS
# ============================================================
def load_audio_file(path):
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return data, sr
    except Exception:
        from pydub import AudioSegment
        song = AudioSegment.from_file(path)
        raw = np.array(song.get_array_of_samples(), dtype=np.float32)
        raw /= float(2 ** (8 * song.sample_width - 1))
        if song.channels >= 2:
            data = raw.reshape(-1, song.channels)[:, :2]
        else:
            data = np.column_stack([raw, raw])
        return data, song.frame_rate


def synth_demo(sr=44100, seconds=40):
    t = np.arange(int(sr * seconds)) / sr
    out = np.zeros_like(t)
    chords = [(220.0, 277.18, 329.63), (196.0, 246.94, 293.66),
              (174.61, 220.0, 261.63), (196.0, 246.94, 293.66)]
    step = 0.25
    for i in range(int(seconds / step)):
        f = chords[(i // 16) % 4][i % 3]
        s = int(i * step * sr)
        e = min(len(t), s + int(0.6 * sr))
        tt = t[s:e] - t[s]
        env = np.exp(-tt * 5.0) * (1 - np.exp(-tt * 400.0))
        out[s:e] += 0.25 * env * (np.sin(TWO_PI * f * tt) + 0.4 * np.sin(2 * TWO_PI * f * tt)
                                  + 0.15 * np.sin(3 * TWO_PI * f * tt))
    out *= 0.7 / max(1e-9, np.max(np.abs(out)))
    return np.column_stack([out, np.roll(out, 90) * 0.95]).astype(np.float32), sr


# ============================================================
# EXPORT WORKER
# ============================================================
class ExportWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dsp, output_path, volume_val, subtype, parent=None):
        super().__init__(parent)
        self.dsp = dsp
        self.output_path = output_path
        self.volume_val = volume_val
        self.subtype = subtype

    def run(self):
        try:
            c = self.dsp.clone_for_render()
            if c.speed_select * (1.0 + c.pitch_pct / 100.0) <= 0.0 or not c.motor_on:
                raise RuntimeError("Motor is stopped - start the motor before exporting.")
            block = 4096
            end = c.lead + c.n_frames + 1.0 * c.sr          # music + 1 s of run-out
            est = max(1, int(end / (max(c.speed, 0.1) * block)))
            chunks = []
            i = 0
            while c.playhead < end and i < est * 3 + 20:
                if self.isInterruptionRequested():
                    return
                chunks.append(c.process_block(block, self.volume_val))
                i += 1
                self.progress.emit(int(min(99, c.playhead / end * 100)))
            rendered = np.concatenate(chunks, axis=0).astype(np.float64)
            if self.subtype == "PCM_16":                     # TPDF dither
                rng = np.random.default_rng()
                rendered += (rng.random(rendered.shape) - rng.random(rendered.shape)) / 32768.0
            rendered = np.clip(np.nan_to_num(rendered), -1.0, 1.0).astype(np.float32)
            sf.write(self.output_path, rendered, c.sr, subtype=self.subtype)
            self.progress.emit(100)
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


# ============================================================
# TURNTABLE VISUAL WIDGET
# ============================================================
class VirtualTurntableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(340, 340)
        self.angle_deg = 0.0
        self.progress = 0.0
        self.contact = 0.0
        self.is_dragging = False
        self.last_angle = 0.0
        self.last_t = 0.0
        self.acc = 0.0
        self.scratch_callback = None      # (kind, deg_per_second)

    def set_state(self, angle, progress, contact):
        self.angle_deg = angle % 360.0
        self.progress = progress
        self.contact = contact
        self.update()

    def _angle_from_center(self, pos):
        return math.degrees(math.atan2(pos.y() - self.height() / 2.0, pos.x() - self.width() / 2.0))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.last_angle = self._angle_from_center(event.position())
            self.last_t = time.perf_counter()
            self.acc = 0.0
            if self.scratch_callback:
                self.scratch_callback("press", 0.0)

    def mouseMoveEvent(self, event):
        if not self.is_dragging:
            return
        ang = self._angle_from_center(event.position())
        d = ang - self.last_angle
        if d > 180:
            d -= 360
        if d < -180:
            d += 360
        self.last_angle = ang
        self.acc += d
        now = time.perf_counter()
        if now - self.last_t >= 0.004:
            if self.scratch_callback:
                self.scratch_callback("move", self.acc / (now - self.last_t))
            self.acc = 0.0
            self.last_t = now

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            if self.scratch_callback:
                self.scratch_callback("release", 0.0)

    def _draw_arm(self, p, cx, cy, r):
        vr = r - 10.0
        px, py = cx + r * 1.0, cy - r * 0.72
        dx, dy = px - cx, py - cy
        d = math.hypot(dx, dy)
        arm = 1.15 * r
        rs_play = vr * (0.96 - 0.58 * self.progress)
        rs_rest = vr * 1.10
        rs = rs_rest + (rs_play - rs_rest) * self.contact
        a = (rs * rs - arm * arm + d * d) / (2 * d)
        hh = math.sqrt(max(rs * rs - a * a, 0.0))
        ux, uy = dx / d, dy / d
        bx, by = cx + a * ux, cy + a * uy
        c1 = (bx - uy * hh, by + ux * hh)
        c2 = (bx + uy * hh, by - ux * hh)
        sx, sy = c1 if c1[1] > c2[1] else c2
        p.setPen(QPen(QColor(195, 200, 212), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(px, py), QPointF(sx, sy))
        ex, ey = sx - (sx - px) / arm * 14, sy - (sy - py) / arm * 14
        p.setPen(QPen(QColor(40, 44, 52), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(ex, ey), QPointF(sx, sy))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(90, 96, 108)))
        p.drawEllipse(QPointF(px, py), 12.0, 12.0)
        p.setBrush(QBrush(QColor(225, 70, 60) if self.contact > 0.5 else QColor(150, 150, 150)))
        p.drawEllipse(QPointF(sx, sy), 2.5, 2.5)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r = min(w, h) / 2.0 - 10.0
        p.setBrush(QBrush(QColor(25, 27, 32)))
        p.setPen(QPen(QColor(12, 14, 18), 3))
        p.drawRoundedRect(self.rect().adjusted(5, 5, -5, -5), 16, 16)
        p.save()
        p.translate(QPointF(cx, cy))
        plat = QRadialGradient(0, 0, r)
        plat.setColorAt(0.68, QColor(190, 220, 250, 55))
        plat.setColorAt(1.0, QColor(8, 12, 18, 240))
        p.setBrush(QBrush(plat))
        p.setPen(QPen(QColor(160, 195, 230, 210), 2))
        p.drawEllipse(QPointF(0, 0), r, r)
        p.rotate(self.angle_deg)
        vr = r - 10.0
        vg = QRadialGradient(0, 0, vr)
        vg.setColorAt(0.0, QColor(8, 8, 8))
        vg.setColorAt(0.72, QColor(35, 37, 42))
        vg.setColorAt(1.0, QColor(3, 3, 3))
        p.setBrush(QBrush(vg))
        p.setPen(QPen(QColor(40, 40, 40), 1))
        p.drawEllipse(QPointF(0, 0), vr, vr)
        p.setPen(QPen(QColor(25, 25, 25), 0.9))
        for rr in np.linspace(vr * 0.38, vr * 0.96, 26):
            p.drawEllipse(QPointF(0, 0), float(rr), float(rr))
        lr = vr * 0.32
        lg = QRadialGradient(0, 0, lr)
        lg.setColorAt(0.0, QColor(14, 70, 150))
        lg.setColorAt(1.0, QColor(6, 30, 75))
        p.setBrush(QBrush(lg))
        p.setPen(QPen(QColor(200, 200, 200), 1.5))
        p.drawEllipse(QPointF(0, 0), lr, lr)
        p.setBrush(QBrush(QColor(230, 180, 50)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(0, 0), 5.0, 5.0)
        p.restore()
        self._draw_arm(p, cx, cy, r)


# ============================================================
# PRESETS
# ============================================================
RECORD_PRESETS = {
    # raw slider units: crackle/s, pops/s*10, thumps/s*10, scratches/s*100, hiss dBFS
    "Mint (new pressing)":   dict(crackle_rate=1, pop_rate=0, thump_rate=0, scratch_rate=0, hiss_db=-74),
    "Very Good (VG+)":       dict(crackle_rate=12, pop_rate=6, thump_rate=2, scratch_rate=3, hiss_db=-66),
    "Good":                  dict(crackle_rate=30, pop_rate=15, thump_rate=4, scratch_rate=10, hiss_db=-62),
    "Worn":                  dict(crackle_rate=55, pop_rate=40, thump_rate=8, scratch_rate=30, hiss_db=-57),
    "Trashed":               dict(crackle_rate=80, pop_rate=120, thump_rate=20, scratch_rate=80, hiss_db=-50),
}


# Presets are written in raw slider units and applied on top of the factory state,
# so every preset fully defines the sound (nothing leaks from the previous preset).
# Extra keys: fidelity (0-3), mains (0=60Hz, 1=50Hz), riaa (bool),
# r608_distortion_mode / r608_comp_mode (bool), volume (0-100), _desc (text).
# Web-simulator keys (all 0-100 sliders): warmth, vinyl_age, stylus_res, worn_stylus,
# ghost_echo, riaa_mismatch (50 = neutral).
_CLEAN = dict(
    wow_amt=10, flutter_amt=5, drift_amt=5, cogging_amt=5, hum_db=-100, rumble_db=-100, warp_amt=3,
    res_db=4, igd_amount=8, tracking_limit=45, pinch_amount=2, tangential_deg=6, crosstalk_amt=2,
    cart_peak_db=0, hiss_db=-82, crackle_rate=0, pop_rate=0, thump_rate=0, scratch_rate=0,
    sat_drive=0, noise_floor_db=-110, fidelity=0, r608_distortion_mode=False, r608_comp_mode=False,
)

_TRANSPARENT = dict(
    _desc="Every mechanical and noise flaw off. Just the RIAA chain - ideal for checking the source.",
    wow_amt=0, flutter_amt=0, drift_amt=0, cogging_amt=0, hum_db=-100, rumble_db=-100, warp_amt=0,
    res_db=0, stylus_fc=22000, igd_amount=0, tracking_limit=60, pinch_amount=0, tangential_deg=0,
    bass_mono_hz=0, crosstalk_amt=0, cart_peak_db=0, hiss_db=-90, crackle_rate=0, pop_rate=0,
    thump_rate=0, scratch_rate=0, subsonic_hz=0, sat_drive=0, noise_floor_db=-110, fidelity=0,
    r608_distortion_mode=False, r608_comp_mode=False)


def _classic(warmth, noise, wow, crackle, age, res, worn, rumble, ghost, riaa, desc):
    """A preset from the web Vinyl Simulator: its ten 0-100 controls, translated to this
    engine's detailed controls (the same mapping the Classic Controls tab uses), on top of
    a flaw-free deck so that nothing else colours the sound."""
    return {**_TRANSPARENT, "_desc": desc,
            "warmth": warmth, "hiss_db": round(-90 + 0.6 * noise),
            "wow_amt": round(3 * wow), "flutter_amt": round(2 * wow),
            "crackle_rate": round(0.8 * crackle), "pop_rate": round(1.2 * crackle),
            "rumble_db": round(-100 + 0.6 * rumble),
            "vinyl_age": age, "stylus_res": res, "worn_stylus": worn,
            "ghost_echo": ghost, "riaa_mismatch": riaa}


PRESET_GROUPS = [
    ("Reference / Clean", {
        "Transparent (all flaws off)": dict(_TRANSPARENT),
        "Audiophile Reference": {**_CLEAN, "_desc": "High-end deck, new stylus, mint pressing. Barely-there vinyl character.",
                                 "stylus_fc": 21000, "bass_mono_hz": 100},
        "Modern 180g Audiophile Pressing": {**_CLEAN, "_desc": "Quiet modern pressing with a whisper of surface noise and a few ticks.",
                                            "hiss_db": -76, "crackle_rate": 1, "stylus_fc": 20000, "igd_amount": 15,
                                            "hum_db": -90},
        "Direct-Drive DJ Deck": dict(
            _desc="Instant spin-up, rock-steady speed, tight tracking. Technics-style club deck.",
            spinup_time=3, wow_amt=4, flutter_amt=3, drift_amt=2, cogging_amt=0, rumble_db=-88, hum_db=-90,
            warp_amt=5, tracking_limit=40, tangential_deg=8, stylus_fc=21000, crosstalk_amt=4, cart_peak_db=10,
            hiss_db=-70, crackle_rate=8, pop_rate=3, sat_drive=10, bass_mono_hz=140, output_gain=95),
    }),
    ("Web Simulator Presets", {
        "Web: Digital": _classic(0, 0, 0, 0, 0, 0, 0, 0, 0, 50,
                                 "Everything off - a clean digital reference."),
        "Web: Audiophile": _classic(40, 0, 0, 0, 0, 60, 0, 0, 0, 42,
                                    "A touch of warmth and cartridge sparkle, no noise: a well-kept hi-fi system."),
        "Web: Fresh Press": _classic(25, 8, 12, 3, 0, 45, 0, 10, 8, 47,
                                     "A new pressing: faint surface noise, hint of rumble, almost no wear."),
        "Web: Well Played": _classic(50, 40, 35, 30, 20, 25, 15, 20, 15, 45,
                                     "A much-loved record: audible crackle, wobble and a softened top end."),
        "Web: Worn Classic": _classic(75, 70, 50, 75, 80, 10, 65, 45, 55, 30,
                                      "Decades of play: heavy crackle, dull highs, ghost echoes and rumble."),
        "Web: Lo-Fi": _classic(90, 85, 75, 90, 55, 5, 80, 65, 50, 25,
                               "Maximum character: thick warmth, wobble, dust and a worn stylus."),
    }),
    ("Vintage Hi-Fi", {
        "1970s Hi-Fi Console (warm)": dict(
            _desc="Warm, slightly wobbly, gently saturated. Belt-drive console with an aging cartridge.",
            spinup_time=25, wow_amt=90, flutter_amt=30, drift_amt=40, cogging_amt=40, hum_db=-68, rumble_db=-72,
            warp_amt=30, res_freq=90, res_db=10, stylus_fc=16500, igd_amount=55, tracking_limit=25,
            pinch_amount=14, tangential_deg=20, bass_mono_hz=150, crosstalk_amt=8, cart_peak_db=25,
            hiss_db=-62, crackle_rate=20, pop_rate=8, thump_rate=3, scratch_rate=4, subsonic_hz=14,
            sat_drive=28, noise_floor_db=-92, warmth=55, vinyl_age=15, riaa_mismatch=44),
        "1950s Mono Console": dict(
            _desc="Narrow, boxy, near-mono sound from a tube-era console.",
            spinup_time=30, wow_amt=130, flutter_amt=45, drift_amt=60, cogging_amt=50, hum_db=-62, rumble_db=-68,
            warp_amt=40, stylus_fc=13000, igd_amount=60, tracking_limit=20, pinch_amount=16, tangential_deg=25,
            bass_mono_hz=400, crosstalk_amt=30, cart_peak_db=30, hiss_db=-58, crackle_rate=28, pop_rate=12,
            thump_rate=4, scratch_rate=6, subsonic_hz=0, sat_drive=30, fm_strength=60, fidelity=1,
            r608_comp_mode=True, r608_comp_intensity=15, warmth=65, vinyl_age=40, ghost_echo=15, riaa_mismatch=40),
        "Budget Belt-Drive Turntable": dict(
            _desc="Entry-level deck: slow spin-up, audible wow/flutter, hum and rumble.",
            spinup_time=35, wow_amt=150, flutter_amt=55, drift_amt=70, cogging_amt=60, hum_db=-66, rumble_db=-68,
            warp_amt=35, res_freq=85, res_db=12, stylus_fc=17000, igd_amount=60, tracking_limit=18,
            pinch_amount=15, tangential_deg=25, crosstalk_amt=10, cart_peak_db=30, hiss_db=-63,
            crackle_rate=18, pop_rate=8, sat_drive=12, warmth=20, vinyl_age=10),
    }),
    ("Worn / Damaged", {
        "Thrift-Store Find (worn)": dict(
            _desc="A played-to-death record: constant crackle, pops, a little warp and tired highs.",
            wow_amt=120, flutter_amt=40, drift_amt=40, warp_amt=55, igd_amount=70, pinch_amount=20,
            tracking_limit=22, stylus_fc=15000, crosstalk_amt=9, hum_db=-70, rumble_db=-72, hiss_db=-57,
            crackle_rate=55, pop_rate=40, thump_rate=8, scratch_rate=30,
            vinyl_age=45, worn_stylus=30, ghost_echo=25, warmth=30),
        "Dusty Attic Record": dict(
            _desc="Scratched, dusty, warped and rumbly. Everything wrong at once, but still playable.",
            wow_amt=160, flutter_amt=50, drift_amt=60, warp_amt=50, hum_db=-66, rumble_db=-66, stylus_fc=11000,
            igd_amount=80, tracking_limit=16, pinch_amount=25, tangential_deg=30, hiss_db=-52, crackle_rate=75,
            pop_rate=80, thump_rate=12, scratch_rate=50, sat_drive=20,
            vinyl_age=70, worn_stylus=60, ghost_echo=40, warmth=45),
        "Warped Record": dict(
            _desc="Big dished warp: slow pitch wobble and woofer-pumping infrasonics through the arm resonance.",
            warp_amt=100, wow_amt=220, flutter_amt=30, res_freq=85, res_db=14, rumble_db=-70),
        "Worn Stylus / Cheap Cartridge": dict(
            _desc="Dull, distorted highs, harsh peak, poor tracking and heavy channel leakage.",
            stylus_fc=9500, igd_amount=90, tracking_limit=14, pinch_amount=35, tangential_deg=30,
            crosstalk_amt=14, cart_peak_db=60, sat_drive=15, hiss_db=-63, worn_stylus=70, vinyl_age=30,
            stylus_res=35),
        "Cold Basement Rumble": dict(
            _desc="Heavy bearing rumble, hum and ringing arm - a bad turntable on a bad shelf.",
            rumble_db=-52, hum_db=-60, warp_amt=70, res_db=16, res_freq=80, subsonic_hz=5, wow_amt=80),
        "Broken Turntable (chaos)": dict(
            _desc="Everything maxed. For stress-testing and sound design.",
            wow_amt=300, flutter_amt=200, drift_amt=300, cogging_amt=100, warp_amt=100, hum_db=-52, rumble_db=-50,
            res_db=20, igd_amount=100, pinch_amount=70, tangential_deg=50, tracking_limit=6, stylus_fc=9000,
            crosstalk_amt=30, cart_peak_db=100, hiss_db=-48, crackle_rate=80, pop_rate=150, thump_rate=20,
            scratch_rate=90, sat_drive=60, fidelity=1, warmth=80, vinyl_age=90, worn_stylus=90, ghost_echo=80,
            stylus_res=60),
    }),
    ("Creative / Character", {
        "Lo-Fi Hip-Hop Dust": dict(
            _desc="Wobbly, saturated, dusty sample-source vibe with a rolled-off top and mono low end.",
            wow_amt=220, flutter_amt=80, drift_amt=120, cogging_amt=40, warp_amt=60, hiss_db=-58, crackle_rate=45,
            crackle_level_db=3, pop_rate=20, thump_rate=5, scratch_rate=8, stylus_fc=12000, igd_amount=50,
            sat_drive=45, bass_mono_hz=220, crosstalk_amt=12, cart_peak_db=0, hum_db=-72, rumble_db=-70, res_db=10,
            warmth=70, vinyl_age=40, ghost_echo=20, riaa_mismatch=38),
        "Dreamy Tape-Wobble (extreme wow)": dict(
            _desc="Seasick pitch drift for ambient and dream-pop textures.",
            wow_amt=300, flutter_amt=150, drift_amt=280, cogging_amt=80, warp_amt=80, crackle_rate=8,
            hiss_db=-70, sat_drive=20, stylus_fc=15000, warmth=40, ghost_echo=30),
        "Pure Crackle Bed (producer)": {**_CLEAN, "_desc": "Clean playback plus just hiss and crackle - for layering vinyl texture.",
                                        "hiss_db": -66, "crackle_rate": 40, "crackle_level_db": 2, "pop_rate": 12,
                                        "thump_rate": 3, "scratch_rate": 5},
        "Shellac 78 Sound": dict(
            _desc="Band-limited, noisy, narrow and honky, in the spirit of a 78 RPM shellac disc.",
            stylus_fc=8500, bass_mono_hz=400, subsonic_hz=30, hiss_db=-46, crackle_rate=80, crackle_level_db=4,
            pop_rate=70, thump_rate=10, scratch_rate=30, wow_amt=140, flutter_amt=50, drift_amt=50, warp_amt=40,
            igd_amount=70, tracking_limit=15, pinch_amount=25, tangential_deg=25, crosstalk_amt=30,
            cart_peak_db=60, sat_drive=30, fidelity=2, r608_comp_mode=True, r608_comp_intensity=25,
            warmth=60, vinyl_age=60, worn_stylus=50, ghost_echo=35),
    }),
    ("Cheap Players", {
        "R608 Suitcase Player": dict(
            _desc="Ceramic cartridge, tiny speaker, cheap amp. Honky, compressed and mono-ish.",
            spinup_time=20, wow_amt=180, flutter_amt=70, drift_amt=80, cogging_amt=70, warp_amt=40, hum_db=-58,
            rumble_db=-62, res_db=14, tracking_limit=12, igd_amount=75, pinch_amount=30, tangential_deg=35,
            stylus_fc=12000, bass_mono_hz=400, crosstalk_amt=30, cart_peak_db=70, hiss_db=-56, crackle_rate=35,
            pop_rate=15, thump_rate=4, scratch_rate=8, sat_drive=35, fidelity=2, r608_distortion_mode=True,
            r608_distortion_intensity=45, r608_comp_mode=True, r608_comp_intensity=55,
            warmth=30, vinyl_age=25, worn_stylus=30),
        "Toy Record Player (ultra cheap)": dict(
            _desc="11 kHz sampling, crushed dynamics, huge wobble. The worst player imaginable.",
            spinup_time=30, wow_amt=280, flutter_amt=140, drift_amt=150, cogging_amt=100, warp_amt=70, hum_db=-52,
            rumble_db=-55, res_db=18, tracking_limit=8, igd_amount=95, pinch_amount=50, tangential_deg=50,
            stylus_fc=8000, bass_mono_hz=400, crosstalk_amt=30, cart_peak_db=100, hiss_db=-50, crackle_rate=60,
            pop_rate=40, thump_rate=8, scratch_rate=20, sat_drive=50, fidelity=3, r608_distortion_mode=True,
            r608_distortion_intensity=85, r608_comp_mode=True, r608_comp_intensity=85,
            vinyl_age=50, worn_stylus=60, ghost_echo=20),
    }),
    ("Listening Situations", {
        "Quiet Night (loudness comp)": dict(
            _desc="Low volume with Fletcher-Munson loudness boost so bass and treble stay audible.",
            fm_strength=100, volume=30, bass_mono_hz=140, sat_drive=0, crackle_rate=6, hiss_db=-72, hum_db=-84),
        "Club / Warehouse Loud": dict(
            _desc="Driven, punchy and tight - turned-up system with extra saturation.",
            fm_strength=40, sat_drive=35, output_gain=115, bass_mono_hz=160, wow_amt=20, tracking_limit=40,
            hum_db=-80, hiss_db=-70, volume=85),
    }),
]
PRESETS = {name: p for _, grp in PRESET_GROUPS for name, p in grp.items()}
PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt85_presets")


# ============================================================
# CLASSIC CONTROLS (the web simulator's ten knobs)
# ============================================================
# Four of the ten are macros: they move the detailed sliders on the other tabs together.
# The other six are engine controls of their own. Keys are the web simulator's.
MACROS = {
    "noise":   lambda v: {"hiss_db": -90 + 0.6 * v},
    "wow":     lambda v: {"wow_amt": 3.0 * v, "flutter_amt": 2.0 * v},
    "crackle": lambda v: {"crackle_rate": 0.8 * v, "pop_rate": 1.2 * v},
    "rumble":  lambda v: {"rumble_db": -100 + 0.6 * v},
}

EFFECT_INFO = {
    "warmth": (
        "Warmth",
        "Lifts the low-mids around 220 Hz, rolls off the treble above 6 kHz, and adds soft "
        "saturation that lifts quiet passages and rounds off peaks. The saturation is set so "
        "the overall loudness stays about the same.",
        "Vinyl's whole analogue chain - the cutting lathe, the lacquer, the phono preamp - "
        "colours the sound. The result is a rounder, slightly saturated midrange and a softer "
        "top end that digital recordings don't have.",
        "Use a bright or clinical recording. Play it at 0%, then bring Warmth up to 70%. The "
        "sound gets fuller and rounder, with a slight grit on loud peaks."),
    "noise": (
        "Surface Noise",
        "Sets the level of the continuous hiss from the groove walls. This is a macro: it moves "
        "the Surface Noise slider on the Surface && Noise tab. The hiss is stored in the groove "
        "itself, so it slows, stops and reverses with the record, and it grows slightly toward "
        "the inner grooves.",
        "The groove wall is never perfectly smooth. Dust, pressing imperfections and wear create "
        "a constant low-level hiss under the music on every record, even a clean one.",
        "Listen in a quiet passage or before the music starts, and raise it slowly from 0%. A "
        "soft tape-like hiss emerges; at high values it becomes a clear noise floor."),
    "wow": (
        "Wow && Flutter",
        "Varies the playback speed. This is a macro: it moves the Wow and Flutter sliders on "
        "the Motor && Platter tab. Wow is slow drift locked to the platter's rotation (once per "
        "turn); flutter is a faster, random wobble from the bearing and motor.",
        "No turntable runs at perfectly constant speed. Wow comes from an off-centre hole or an "
        "uneven belt; flutter from fast mechanical irregularities in the motor and bearing.",
        "Play a sustained note - a piano chord, a long vocal, a synth pad. At 0% the pitch is "
        "steady. Raise it and the note starts to waver. Percussive music hides it."),
    "crackle": (
        "Crackle",
        "Sets how many ticks and pops fall on the record. This is a macro: it moves the Crackle "
        "and Pop Density sliders on the Surface && Noise tab. Each defect sits at a fixed spot "
        "in the groove, so it is the same every time you play that part, slows down with the "
        "platter, and plays backwards when you scratch.",
        "Dust, fingerprints and tiny scratches make the stylus jolt, producing the ticks and "
        "pops of a well-used record.",
        "Best heard in quiet passages. Set Crackle to 60% and listen to a soft section: pops "
        "appear at the same places on each play. Try the Pop Level and Scratch sliders too."),
    "age": (
        "Vinyl Age",
        "Rolls off the top end, from 14.5 kHz at 0% down to 8 kHz at 100%, the way a heavily "
        "played groove loses its fine detail. It dulls the surface noise along with the music.",
        "Every play grinds the stylus along the groove walls and slowly erodes the fine detail "
        "that carries high frequencies. After hundreds of plays a record loses its air.",
        "Play something with hi-hats or cymbals. At 0% the top is crisp; toward 100% the "
        "cymbals go dull and muffled."),
    "resonance": (
        "Stylus Resonance",
        "A peak around 10 kHz (Q 2.5), up to +6 dB, from the cartridge's moving parts.",
        "Every cartridge has a natural resonance where the stylus mass and the cantilever "
        "springiness interact, giving a slight presence peak - the 'zing' of an analogue top end.",
        "Play something with clear highs - cymbals, sibilant vocals, acoustic guitar. At 0% the "
        "top end is flat; raise it and an airy brightness builds around 10 kHz."),
    "worn": (
        "Worn Stylus",
        "A high-frequency cut above 8 kHz, up to -8 dB.",
        "A diamond tip wears down over hundreds of hours. A rounded tip can't follow the finest "
        "groove detail, so highs lose resolution and transients go soft.",
        "Play something bright and crisp. At 0% it is clear and detailed; toward 100% hi-hats are "
        "veiled and vocals lose their air, as if a thin cloth covered the speakers."),
    "rumble": (
        "Turntable Rumble",
        "Adds low-frequency noise (12-110 Hz) from the motor and bearing. This is a macro: it "
        "moves Platter Rumble on the Motor && Platter tab. The rumble only appears while the "
        "platter is turning and the needle is down, and it passes through the tonearm's "
        "resonance.",
        "The motor vibrates at low frequencies, and the vibration travels through the chassis and "
        "tonearm into the stylus. Even well-built decks never remove it completely.",
        "You need headphones or speakers that reproduce bass below 80 Hz - laptop speakers won't. "
        "Raise it slowly and a low continuous thudding appears beneath the music."),
    "ghost": (
        "Ghost Echo",
        "Print-through: a faint copy of loud passages bleeds into the neighbouring grooves, one "
        "revolution (1.8 s) away. You hear a ghost slightly before the loud sound (pre-echo, "
        "the stronger one) and a fainter one after it. Only loud passages cause it, and it is "
        "read from the groove, so it slows, stops and reverses with the record.",
        "A very loud passage can press a faint impression into the adjacent groove wall. On "
        "playback the neighbouring revolution carries a quiet shadow of it.",
        "Find a track with a sudden loud entry after a quiet part - a snare hit or a big "
        "chorus. Listen to the quiet second before it with Ghost Echo raised: you'll hear a "
        "faint shadow of the loud sound arriving early."),
    "riaa": (
        "RIAA Mismatch",
        "Tilts the tonal balance with a bass shelf (200 Hz) and a treble shelf (3 kHz). Below "
        "50% is warm and lush (bass up, treble down); above 50% is bright and punchy (bass down, "
        "treble up). 50% is neutral.",
        "Records are cut with a standard EQ curve that the phono preamp reverses on playback. "
        "Mastering engineers of different eras interpreted the curve slightly differently, so a "
        "small mismatch shifts the tonal balance.",
        "Play a full mix with bass, vocals and hi-hats. Drag the slider slowly from 0% to 100%: "
        "at 0% the low end is prominent and lush; at 100% the mids and highs push forward and the "
        "bass recedes."),
}


# ============================================================
# MAIN WINDOW
# ============================================================
class TurntableWindow(QMainWindow):
    def __init__(self, dsp_engine, device_sr):
        super().__init__()
        self.dsp = dsp_engine
        self.device_sr = device_sr
        self.block_size = 1024
        self.stream = None
        self.volume_val = 80
        self._muted = False
        self._mute_g = 1.0
        self.visual_angle = 0.0
        self.last_slider_val = 0
        self._tl_last_t = 0.0
        self._speed_sel = 1.0
        self._motor_on = True
        self._export_worker = None
        self._cb_err = False
        self._last_tick = time.perf_counter()
        self._sliders = {}
        self._checks = {}
        self._macros = {}
        self._appliers = []
        self._ab = {"A": None, "B": None}
        self._ab_cur = None
        self._factory = None
        self._user_presets = {}

        self.setWindowTitle("Fluance RT85 Ultimate - Vinyl Physics Emulator")
        self.resize(820, 1260)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #121418; color: #f1f5f9; font-family: 'Segoe UI', Arial; }
            QPushButton { background-color: #334155; color: #f8fafc; border: none; border-radius: 6px;
                          padding: 6px 12px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background-color: #4a5568; }
            QPushButton:pressed { background-color: #2d3748; }
            QLabel { font-size: 11px; color: #cbd5e1; }
            QSlider::groove:horizontal { height: 4px; background: #334155; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; height: 14px; background: #38bdf8; border-radius: 7px; margin: -5px 0; }
            QSlider::sub-page:horizontal { background: #0284c7; border-radius: 2px; }
            QComboBox { background: #334155; color: #f8fafc; border: none; border-radius: 4px; padding: 3px 8px; }
            QTabWidget::pane { border: 1px solid #1f2937; border-radius: 6px; }
            QTabBar::tab { background: #1a1d24; color: #cbd5e1; padding: 6px 12px; margin-right: 2px;
                           border-top-left-radius: 6px; border-top-right-radius: 6px; font-size: 11px; }
            QTabBar::tab:selected { background: #0284c7; color: white; }
        """)
        self._init_ui()
        self._init_audio()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._gui_tick)
        self._timer.start(16)

    # ------------------------------------------------------------ widgets helpers
    def _add_slider(self, grid, row, col, title, attr, lo, hi, default, conv, fmt):
        lbl = QLabel()
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(default)

        def update(v):
            lbl.setText(f"{title}: {fmt(v)}")
            setattr(self.dsp, attr, conv(v))

        sld.valueChanged.connect(update)
        update(default)
        grid.addWidget(lbl, row * 2, col)
        grid.addWidget(sld, row * 2 + 1, col)
        self._sliders[attr] = sld
        self._appliers.append(lambda d, a=attr, s=sld, c=conv: setattr(d, a, c(s.value())))

    def _add_check(self, grid, row, col, text, attr, default=False):
        chk = QCheckBox(text)
        chk.setChecked(default)
        chk.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        chk.stateChanged.connect(lambda s: setattr(self.dsp, attr, bool(s)))
        grid.addWidget(chk, row * 2, col, 1, 1)
        self._appliers.append(lambda d, a=attr, c=chk: setattr(d, a, c.isChecked()))
        self._checks[attr] = chk
        setattr(self.dsp, attr, default)
        return chk

    def _add_classic(self, grid, row, col, key, title, attr=None, default=0):
        """One of the web simulator's ten 0-100% controls, with its '?' explanation.
        With `attr` it drives an engine control of its own (and is saved in presets);
        without, it is a macro that moves the detailed sliders on the other tabs."""
        box = QWidget()
        hl = QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"{title}: {default}%")
        hl.addWidget(lbl)
        hl.addStretch(1)
        info = QPushButton("?")
        info.setFixedSize(22, 22)
        info.setStyleSheet("QPushButton { padding: 0; border-radius: 11px; border: 1px solid #0284c7;"
                           " background: #0f1923; color: #38bdf8; } QPushButton:hover { background: #0284c7; color: white; }")
        info.clicked.connect(lambda _=False, k=key: self._show_info(k))
        hl.addWidget(info)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(0, 100)
        sld.setValue(default)
        conv = lambda v: v / 100.0

        def update(v):
            lbl.setText(f"{title}: {v}%")
            if attr:
                setattr(self.dsp, attr, conv(v))
            else:
                for a, val in MACROS[key](v).items():
                    self._sliders[a].setValue(int(round(val)))

        sld.valueChanged.connect(update)
        grid.addWidget(box, row * 2, col)
        grid.addWidget(sld, row * 2 + 1, col)
        if attr:
            self._sliders[attr] = sld
            self._appliers.append(lambda d, a=attr, s=sld, c=conv: setattr(d, a, c(s.value())))
        else:
            self._macros[key] = sld

    def _show_info(self, key):
        title, what, why, how = EFFECT_INFO[key]
        clean = lambda s: s.replace("&&", "&")
        box = QMessageBox(self)
        box.setWindowTitle(clean(title))
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(clean(title).upper())
        box.setInformativeText(
            f"WHAT IT DOES\n{clean(what)}\n\nWHY VINYL PLAYERS HAVE THIS\n{clean(why)}\n\nHOW TO HEAR IT\n{clean(how)}")
        box.exec()

    def _tab(self, tabs, name):
        w = QWidget()
        g = QGridLayout(w)
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(2)
        tabs.addTab(w, name)
        return g

    def _init_ui(self):
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)
        ttl = QLabel("FLUANCE RT85 - VINYL PHYSICS EMULATOR")
        ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ttl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ttl.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        lay.addWidget(ttl)

        self.platter = VirtualTurntableWidget()
        self.platter.scratch_callback = self._on_scratch
        lay.addWidget(self.platter, alignment=Qt.AlignmentFlag.AlignCenter)

        top = QHBoxLayout()
        top.addWidget(QLabel("Track Position"))
        self.chk_scrub = QCheckBox("Scrub Mode")
        self.chk_scrub.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.chk_scrub.stateChanged.connect(lambda s: setattr(self.dsp, "slider_scrub_mode", bool(s)))
        top.addWidget(self.chk_scrub)
        lay.addLayout(top)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderPressed.connect(self._tl_pressed)
        self.timeline.sliderMoved.connect(self._tl_moved)
        self.timeline.sliderReleased.connect(self._tl_released)
        lay.addWidget(self.timeline)

        frm = QFrame()
        frm.setStyleSheet("QFrame { background:#1a1d24; border-radius:12px; }")
        fl = QVBoxLayout(frm)
        row1 = QHBoxLayout()
        sp = QVBoxLayout()
        sp.addWidget(QLabel("Speed", alignment=Qt.AlignmentFlag.AlignCenter))
        sb = QHBoxLayout()
        self.btn_33 = QPushButton("33 1/3 RPM")
        self.btn_45 = QPushButton("45 RPM")
        self.btn_33.clicked.connect(lambda: self._set_speed(1.0, self.btn_33, self.btn_45))
        self.btn_45.clicked.connect(lambda: self._set_speed(1.35, self.btn_45, self.btn_33))
        self.btn_33.setStyleSheet("background:#0284c7;color:white;")
        sb.addWidget(self.btn_33)
        sb.addWidget(self.btn_45)
        sp.addLayout(sb)
        row1.addLayout(sp)
        mt = QVBoxLayout()
        mt.addWidget(QLabel("Motor", alignment=Qt.AlignmentFlag.AlignCenter))
        self.btn_stop = QPushButton("STOP / BRAKE")
        self.btn_stop.clicked.connect(self._toggle_stop)
        mt.addWidget(self.btn_stop)
        row1.addLayout(mt)
        cu = QVBoxLayout()
        cu.addWidget(QLabel("Tonearm", alignment=Qt.AlignmentFlag.AlignCenter))
        self.btn_cue = QPushButton("LOWER NEEDLE")
        self.btn_cue.clicked.connect(self._toggle_cue)
        cu.addWidget(self.btn_cue)
        row1.addLayout(cu)
        vl = QVBoxLayout()
        self.lbl_vol = QLabel("Volume: 80%")
        vl.addWidget(self.lbl_vol)
        vrow = QHBoxLayout()
        self.sld_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_vol.setRange(0, 100)
        self.sld_vol.setValue(80)
        self.sld_vol.valueChanged.connect(self._on_volume)
        vrow.addWidget(self.sld_vol, 1)
        self.btn_mute = QPushButton("MUTE")
        self.btn_mute.setFixedWidth(64)
        self.btn_mute.clicked.connect(self._toggle_mute)
        vrow.addWidget(self.btn_mute)
        vl.addLayout(vrow)
        row1.addLayout(vl)
        fl.addLayout(row1)
        lay.addWidget(frm)

        # ---------------- preset bar
        pf = QFrame()
        pf.setStyleSheet("QFrame { background:#1a1d24; border-radius:10px; }")
        pl = QVBoxLayout(pf)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("PRESET:"))
        self.cbo_presets = QComboBox()
        self.cbo_presets.setMinimumWidth(300)
        self.cbo_presets.activated.connect(self._on_preset_chosen)
        pr.addWidget(self.cbo_presets, 1)
        self.btn_p_save = QPushButton("SAVE AS...")
        self.btn_p_save.clicked.connect(self._save_user_preset)
        self.btn_p_del = QPushButton("DELETE")
        self.btn_p_del.clicked.connect(self._delete_user_preset)
        self.btn_p_reset = QPushButton("RESET")
        self.btn_p_reset.clicked.connect(lambda: self._apply_state(self._preset_to_state({})) or
                                         self.lbl_preset_desc.setText("Factory defaults restored."))
        for b in (self.btn_p_save, self.btn_p_del, self.btn_p_reset):
            pr.addWidget(b)
        pl.addLayout(pr)
        ab = QHBoxLayout()
        ab.addWidget(QLabel("A/B COMPARE:"))
        self.btn_sa = QPushButton("STORE -> A")
        self.btn_sb = QPushButton("STORE -> B")
        self.btn_ab = QPushButton("SWITCH A <-> B")
        self.btn_sa.clicked.connect(lambda: self._ab_store("A"))
        self.btn_sb.clicked.connect(lambda: self._ab_store("B"))
        self.btn_ab.clicked.connect(self._ab_switch)
        for b in (self.btn_sa, self.btn_sb, self.btn_ab):
            ab.addWidget(b)
        pl.addLayout(ab)
        self.lbl_preset_desc = QLabel("Pick a preset - every preset sets ALL sound controls.")
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color:#94a3b8; font-size:11px;")
        pl.addWidget(self.lbl_preset_desc)
        lay.addWidget(pf)

        tabs = QTabWidget()
        self._build_tabs(tabs)
        self._rebuild_preset_combo()
        lay.addWidget(tabs)

        io = QFrame()
        io.setStyleSheet("QFrame { background: #0f1923; border: 1px solid #0284c7; border-radius: 8px; }")
        il = QHBoxLayout(io)
        self.btn_load = QPushButton("LOAD NEW AUDIO")
        self.btn_load.clicked.connect(self._load_new_audio)
        il.addWidget(self.btn_load)
        il.addWidget(QLabel("Format:"))
        self.cbo_format = QComboBox()
        self.cbo_format.addItems(["WAV (PCM 24-bit)", "WAV (PCM 16-bit, dithered)",
                                  "WAV (32-bit Float)", "FLAC (24-bit)"])
        il.addWidget(self.cbo_format)
        self.btn_export = QPushButton("EXPORT RENDERED VINYL")
        self.btn_export.setStyleSheet("background: #0284c7; color: white;")
        self.btn_export.clicked.connect(self._start_export)
        il.addWidget(self.btn_export)
        lay.addWidget(io)

        self.lbl_status = QLabel("Status: Tonearm Lifted")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #38bdf8; font-size: 11px;")
        lay.addWidget(self.lbl_status)
        self.lbl_rpm = QLabel("")
        self.lbl_rpm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl_rpm)
        self.setCentralWidget(root)

    def _build_tabs(self, tabs):
        S = self._add_slider
        pct = lambda f: (lambda v: f"{v * f:.3f}%")

        # ---------------- Classic controls (the web simulator's ten knobs)
        g = self._tab(tabs, "Classic Controls")
        C = self._add_classic
        C(g, 0, 0, "warmth", "Warmth", attr="warmth")
        C(g, 0, 1, "noise", "Surface Noise")
        C(g, 1, 0, "wow", "Wow & Flutter")
        C(g, 1, 1, "crackle", "Crackle")
        C(g, 2, 0, "age", "Vinyl Age", attr="vinyl_age")
        C(g, 2, 1, "resonance", "Stylus Resonance", attr="stylus_res")
        C(g, 3, 0, "worn", "Worn Stylus", attr="worn_stylus")
        C(g, 3, 1, "rumble", "Turntable Rumble")
        C(g, 4, 0, "ghost", "Ghost Echo", attr="ghost_echo")
        C(g, 4, 1, "riaa", "RIAA Mismatch", attr="riaa_mismatch", default=50)
        note = QLabel("These are the same ten controls as the web Vinyl Simulator. Surface Noise, Wow & Flutter, "
                      "Crackle and Turntable Rumble are macros: moving one sets the detailed sliders on the "
                      "other tabs, which you can then fine-tune. Presets set the detailed sliders directly, so "
                      "those four slider positions are not updated when a preset is loaded. Press ? on any "
                      "control for an explanation.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#94a3b8; font-size:10px;")
        g.addWidget(note, 10, 0, 1, 2)

        # ---------------- Motor & platter
        g = self._tab(tabs, "Motor && Platter")
        S(g, 0, 0, "Platter Inertia (spin-up)", "spinup_time", 3, 80, 18, lambda v: v / 10.0, lambda v: f"{v / 10:.1f} s")
        S(g, 0, 1, "Pitch Control", "pitch_pct", -80, 80, 0, lambda v: v / 10.0, lambda v: f"{v / 10:+.1f}%")
        S(g, 1, 0, "Wow / Eccentricity (peak)", "wow_amt", 0, 300, 60, lambda v: v / 100000.0, pct(0.001))
        S(g, 1, 1, "Flutter (RMS)", "flutter_amt", 0, 200, 20, lambda v: v / 100000.0, pct(0.001))
        S(g, 2, 0, "Speed Drift (RMS)", "drift_amt", 0, 300, 25, lambda v: v / 100000.0, pct(0.001))
        S(g, 2, 1, "Motor Cogging (pitch ripple)", "cogging_amt", 0, 100, 30, lambda v: v / 1e6, lambda v: f"{v / 10000:.4f}%")
        off_db = lambda v: "Off" if v <= -100 else f"{v} dBFS"
        S(g, 3, 0, "Motor Hum", "hum_db", -100, -40, -74, float, off_db)
        S(g, 3, 1, "Platter Rumble", "rumble_db", -100, -40, -80, float, off_db)
        S(g, 4, 0, "Record Warp", "warp_amt", 0, 100, 25, lambda v: v / 100.0, lambda v: f"{v}%")
        S(g, 4, 1, "Arm Resonance Frequency", "res_freq", 60, 160, 100, lambda v: v / 10.0, lambda v: f"{v / 10:.1f} Hz")
        S(g, 5, 0, "Arm Resonance Peak", "res_db", 0, 20, 8, float, lambda v: f"{v} dB")
        self.cbo_mains = QComboBox()
        self.cbo_mains.addItems(["Mains 60 Hz (Americas)", "Mains 50 Hz (Europe / Asia)"])
        self.cbo_mains.currentIndexChanged.connect(lambda i: setattr(self.dsp, "mains_hz", 60.0 if i == 0 else 50.0))
        g.addWidget(self.cbo_mains, 11, 1)
        self._appliers.append(lambda d: setattr(d, "mains_hz", 60.0 if self.cbo_mains.currentIndex() == 0 else 50.0))

        # ---------------- Stylus & cartridge
        g = self._tab(tabs, "Stylus && Cartridge")
        S(g, 0, 0, "Stylus HF Limit (outer groove)", "stylus_fc", 8000, 22000, 19000, float, lambda v: f"{v} Hz")
        S(g, 0, 1, "Inner Groove Distortion / Loss", "igd_amount", 0, 100, 40, lambda v: v / 100.0, lambda v: f"{v}%")
        S(g, 1, 0, "Tracking Margin (higher = cleaner)", "tracking_limit", 5, 60, 30, lambda v: v / 10.0, lambda v: f"{v / 10:.1f}x")
        S(g, 1, 1, "Pinch Effect", "pinch_amount", 0, 100, 8, lambda v: v / 100.0, lambda v: f"{v}%")
        S(g, 2, 0, "Tangential Tracking Error", "tangential_deg", 0, 50, 15, lambda v: v / 10.0, lambda v: f"{v / 10:.1f} deg")
        S(g, 2, 1, "Bass Mono Crossover", "bass_mono_hz", 0, 400, 120, float, lambda v: "Off" if v < 20 else f"{v} Hz")
        S(g, 3, 0, "Stereo Crosstalk (HF-rising)", "crosstalk_amt", 0, 30, 5, lambda v: v / 100.0, lambda v: f"{v}%")
        S(g, 3, 1, "Cartridge HF Peak (14.5 kHz)", "cart_peak_db", 0, 120, 20, lambda v: v / 10.0, lambda v: f"{v / 10:.1f} dB")

        # ---------------- Surface & noise
        g = self._tab(tabs, "Surface && Noise")
        g.addWidget(QLabel("Record Condition Preset:"), 0, 0)
        self.cbo_preset = QComboBox()
        self.cbo_preset.addItems(list(RECORD_PRESETS.keys()))
        self.cbo_preset.setCurrentIndex(1)
        self.cbo_preset.currentTextChanged.connect(self._apply_preset)
        g.addWidget(self.cbo_preset, 0, 1)
        S(g, 1, 0, "Surface Noise", "hiss_db", -90, -30, -66, float, lambda v: f"{v} dBFS")
        S(g, 1, 1, "Crackle Density", "crackle_rate", 0, 80, 12, float, lambda v: f"{v}/s")
        S(g, 2, 0, "Crackle Level", "crackle_level_db", -20, 20, 0, float, lambda v: f"{v:+d} dB")
        S(g, 2, 1, "Pop Density", "pop_rate", 0, 200, 6, lambda v: v / 10.0, lambda v: f"{v / 10:.1f}/s")
        S(g, 3, 0, "Pop Level", "pop_level_db", -20, 20, 0, float, lambda v: f"{v:+d} dB")
        S(g, 3, 1, "Dust Thump Density", "thump_rate", 0, 20, 2, lambda v: v / 10.0, lambda v: f"{v / 10:.1f}/s")
        S(g, 4, 0, "Dust Thump Level", "thump_level_db", -20, 20, 0, float, lambda v: f"{v:+d} dB")
        S(g, 4, 1, "Scratch Density", "scratch_rate", 0, 100, 3, lambda v: v / 100.0, lambda v: f"{v / 100:.2f}/s")
        S(g, 5, 0, "Scratch Level", "scratch_level_db", -20, 20, 0, float, lambda v: f"{v:+d} dB")

        # ---------------- Electronics
        g = self._tab(tabs, "Electronics")
        self.chk_riaa = self._add_check(g, 0, 0, "RIAA phono stage (record cut + playback EQ)", "_riaa_dummy", True)
        self.chk_riaa.stateChanged.disconnect()
        self.chk_riaa.stateChanged.connect(self._on_riaa)
        self._appliers.pop()                                  # RIAA is applied at engine construction
        self._checks.pop("_riaa_dummy", None)
        S(g, 1, 0, "Subsonic Filter", "subsonic_hz", 0, 40, 18, float, lambda v: "Off" if v == 0 else f"{v} Hz")
        S(g, 1, 1, "Preamp Saturation", "sat_drive", 0, 100, 8, lambda v: v / 100.0, lambda v: f"{v}%")
        S(g, 2, 0, "Fletcher-Munson Loudness", "fm_strength", 0, 100, 0, lambda v: v / 100.0, lambda v: f"{v}%")
        S(g, 2, 1, "Electronic Noise Floor", "noise_floor_db", -110, -60, -96, float,
          lambda v: "Off" if v <= -110 else f"{v} dBFS")
        S(g, 3, 0, "Output Trim", "output_gain", 20, 150, 85, lambda v: v / 100.0, lambda v: f"{v}%")

        # ---------------- Player profile
        g = self._tab(tabs, "Player Profile")
        g.addWidget(QLabel("Player Fidelity Profile (band-limit + sample-and-hold):"), 0, 0, 1, 2)
        self.cbo_fidelity = QComboBox()
        self.cbo_fidelity.addItems([
            "Native (device rate, unchanged)",
            "Audiophile / Real RT85 Player (44.1 kHz)",
            "Lofi Suitcase / Shittier (22.05 kHz)",
            "Ultra Shitty / Vintage Toy (11.025 kHz)"])
        self.cbo_fidelity.currentIndexChanged.connect(lambda i: setattr(self.dsp, "fidelity_mode", i))
        self._appliers.append(lambda d: setattr(d, "fidelity_mode", self.cbo_fidelity.currentIndex()))
        g.addWidget(self.cbo_fidelity, 1, 0, 1, 2)
        self._add_check(g, 1, 0, "R608: Ceramic Distortion && Grit", "r608_distortion_mode")
        S(g, 2, 0, "Distortion Grit Level", "r608_distortion_intensity", 0, 100, 50, lambda v: v / 100.0, lambda v: f"{v}%")
        self._add_check(g, 3, 0, "R608: Suitcase Compression, Bass Roll-Off && Speaker", "r608_comp_mode")
        S(g, 4, 0, "Compression / Speaker Level", "r608_comp_intensity", 0, 100, 50, lambda v: v / 100.0, lambda v: f"{v}%")
        self._factory = self._capture_state()          # defaults as built = the "Factory" preset

    # ------------------------------------------------------------ presets
    def _capture_state(self):
        return {
            "sliders": {a: s.value() for a, s in self._sliders.items()},
            "checks": {a: c.isChecked() for a, c in self._checks.items()},
            "riaa": self.chk_riaa.isChecked(),
            "fidelity": self.cbo_fidelity.currentIndex(),
            "mains": self.cbo_mains.currentIndex(),
        }

    def _apply_state(self, st):
        for a, v in st.get("sliders", {}).items():
            if a in self._sliders:
                self._sliders[a].setValue(int(v))
        for a, v in st.get("checks", {}).items():
            if a in self._checks:
                self._checks[a].setChecked(bool(v))
        if "fidelity" in st:
            self.cbo_fidelity.setCurrentIndex(int(st["fidelity"]))
        if "mains" in st:
            self.cbo_mains.setCurrentIndex(int(st["mains"]))
        if "volume" in st:
            self.sld_vol.setValue(int(st["volume"]))
        if "riaa" in st:
            self.chk_riaa.setChecked(bool(st["riaa"]))      # heavy only if it actually changes

    def _preset_to_state(self, preset):
        st = copy.deepcopy(self._factory)
        for k, v in preset.items():
            if k == "_desc":
                continue
            if k in self._sliders:
                st["sliders"][k] = v
            elif k in self._checks:
                st["checks"][k] = v
            else:
                st[k] = v                                   # fidelity / mains / riaa / volume
        return st

    def _load_user_presets(self):
        out = {}
        if os.path.isdir(PRESET_DIR):
            for fn in sorted(os.listdir(PRESET_DIR)):
                if fn.lower().endswith(".json"):
                    try:
                        with open(os.path.join(PRESET_DIR, fn), "r", encoding="utf-8") as f:
                            out[os.path.splitext(fn)[0]] = json.load(f)
                    except Exception:
                        pass
        return out

    def _rebuild_preset_combo(self, select=None):
        self._user_presets = self._load_user_presets()
        cb = self.cbo_presets
        cb.clear()
        cb.addItem("Select a preset...")

        def header(text):
            cb.addItem(f"-- {text} --")
            cb.model().item(cb.count() - 1).setEnabled(False)

        for gname, grp in PRESET_GROUPS:
            header(gname)
            for name in grp:
                cb.addItem(name)
        if self._user_presets:
            header("My Presets")
            for name in self._user_presets:
                cb.addItem(name)
        if select:
            i = cb.findText(select)
            if i >= 0:
                cb.setCurrentIndex(i)

    def _on_preset_chosen(self, i):
        name = self.cbo_presets.itemText(i)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if name in PRESETS:
                self._apply_state(self._preset_to_state(PRESETS[name]))
                self.lbl_preset_desc.setText(PRESETS[name].get("_desc", ""))
            elif name in self._user_presets:
                self._apply_state(self._user_presets[name])
                self.lbl_preset_desc.setText("Your saved preset.")
            else:
                return
        finally:
            QApplication.restoreOverrideCursor()
        self.lbl_status.setText(f"Status: Preset '{name}' applied")

    def _save_user_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        name = "".join(c for c in (name or "") if c.isalnum() or c in " -_()").strip()
        if not ok or not name:
            return
        try:
            os.makedirs(PRESET_DIR, exist_ok=True)
            path = os.path.join(PRESET_DIR, name + ".json")
            if os.path.exists(path):
                r = QMessageBox.question(self, "Overwrite?", f"Preset '{name}' exists. Overwrite?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
                if r != QMessageBox.StandardButton.Yes:
                    return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._capture_state(), f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))
            return
        self._rebuild_preset_combo(select=name)
        self.lbl_preset_desc.setText(f"Saved to {PRESET_DIR}")

    def _delete_user_preset(self):
        name = self.cbo_presets.currentText()
        if name not in self._user_presets:
            self.lbl_preset_desc.setText("Only your own saved presets can be deleted.")
            return
        r = QMessageBox.question(self, "Delete Preset", f"Delete '{name}'?",
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                 QMessageBox.StandardButton.No)
        if r == QMessageBox.StandardButton.Yes:
            try:
                os.remove(os.path.join(PRESET_DIR, name + ".json"))
            except Exception as e:
                QMessageBox.critical(self, "Delete Failed", str(e))
            self._rebuild_preset_combo()

    def _ab_store(self, slot):
        self._ab[slot] = self._capture_state()
        self._ab_cur = slot
        self.lbl_preset_desc.setText(f"Current settings stored in slot {slot}.")

    def _ab_switch(self):
        if not (self._ab["A"] and self._ab["B"]):
            self.lbl_preset_desc.setText("Store both A and B first (set up a sound, press STORE).")
            return
        self._ab_cur = "B" if self._ab_cur == "A" else "A"
        self._apply_state(self._ab[self._ab_cur])
        self.lbl_preset_desc.setText(f"Now playing slot {self._ab_cur}.")

    # ------------------------------------------------------------ audio stream
    def _init_audio(self):
        def callback(outdata, frames, time_info, status):
            try:
                blk = self.dsp.process_block(frames, self.volume_val)
                tgt = 0.0 if self._muted else 1.0
                if tgt != 1.0 or self._mute_g != 1.0:          # short ramp: no click on mute/unmute
                    blk = blk * np.linspace(self._mute_g, tgt, frames)[:, None]
                    self._mute_g = tgt
                outdata[:] = blk
            except Exception:
                outdata.fill(0)
                if not self._cb_err:
                    self._cb_err = True
                    traceback.print_exc()
        try:
            self.stream = sd.OutputStream(samplerate=self.device_sr, channels=2, callback=callback,
                                          blocksize=self.block_size, dtype="float32")
            self.stream.start()
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Cannot open audio stream:\n{e}")
            raise

    # ------------------------------------------------------------ callbacks
    def _on_volume(self, v):
        self.volume_val = v
        # moving the slider off zero is an implicit unmute
        if self._muted and v > 0:
            self._set_muted(False)
        self.lbl_vol.setText(f"Volume: {v}%" + (" (muted)" if self._muted else ""))

    def _set_muted(self, muted):
        self._muted = bool(muted)
        self.btn_mute.setText("UNMUTE" if self._muted else "MUTE")
        self.btn_mute.setStyleSheet("background:#991b1b;color:white;" if self._muted else "")
        self.lbl_vol.setText(f"Volume: {self.volume_val}%" + (" (muted)" if self._muted else ""))

    def _toggle_mute(self):
        self._set_muted(not self._muted)

    def _apply_preset(self, name):
        for attr, raw in RECORD_PRESETS[name].items():
            self._sliders[attr].setValue(raw)

    def _on_riaa(self, state):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.dsp.set_riaa(bool(state))
        finally:
            QApplication.restoreOverrideCursor()
        self.lbl_status.setText("Status: RIAA stage " + ("ON" if state else "OFF (flat cartridge)"))

    def _load_new_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data, sr = load_audio_file(path)
            new = TurntableDSP(data, sr, self.device_sr, riaa=self.chk_riaa.isChecked())
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Load Error", f"Could not load audio file:\n{e}")
            return
        QApplication.restoreOverrideCursor()
        for f in self._appliers:
            f(new)
        new.speed_select = self._speed_sel
        new.motor_on = self._motor_on
        new.speed = self._speed_sel if self._motor_on else 0.0
        new.slider_scrub_mode = self.chk_scrub.isChecked()
        self.dsp = new
        self.btn_cue.setText("LOWER NEEDLE")
        self.btn_cue.setStyleSheet("")
        self.timeline.setValue(0)
        self.lbl_status.setText(f"Status: Loaded {os.path.basename(path)} - tonearm lifted")

    def _set_speed(self, ratio, active, inactive):
        self._speed_sel = ratio
        self.dsp.speed_select = ratio
        active.setStyleSheet("background:#0284c7;color:white;")
        inactive.setStyleSheet("")
        self.lbl_status.setText(f"Status: {'33 1/3' if ratio == 1.0 else '45'} RPM selected")

    def _toggle_stop(self):
        self._motor_on = not self._motor_on
        self.dsp.motor_on = self._motor_on
        if self._motor_on:
            self.btn_stop.setText("STOP / BRAKE")
            self.btn_stop.setStyleSheet("")
            self.lbl_status.setText("Status: Motor Starting")
        else:
            self.btn_stop.setText("START MOTOR")
            self.btn_stop.setStyleSheet("background:#991b1b;color:white;")
            self.lbl_status.setText("Status: Motor Braking")

    def _toggle_cue(self):
        self.dsp.tonearm_lifted = not self.dsp.tonearm_lifted
        if self.dsp.tonearm_lifted:
            self.btn_cue.setText("LOWER NEEDLE")
            self.btn_cue.setStyleSheet("")
            self.dsp.needle_event = "lift"
            self.lbl_status.setText("Status: Tonearm Lifted (position held)")
        else:
            self.btn_cue.setText("LIFT NEEDLE")
            self.btn_cue.setStyleSheet("background:#0284c7;color:white;")
            self.dsp.needle_event = "drop"
            self.lbl_status.setText("Status: Needle Down")

    def _tl_pressed(self):
        self.dsp.slider_pressed = True
        self.last_slider_val = self.timeline.value()
        self._tl_last_t = time.perf_counter()
        self.dsp.slider_seek_pct = self.timeline.value() / 1000.0

    def _tl_moved(self, val):
        if self.dsp.slider_scrub_mode:
            now = time.perf_counter()
            dt = max(1e-3, now - self._tl_last_t)
            frames = (val - self.last_slider_val) / 1000.0 * self.dsp.n_frames
            ratio = float(np.clip(frames / (self.dsp.sr * dt), -12.0, 12.0))
            self.dsp.hand_target = 0.5 * ratio + 0.5 * self.dsp.hand_target
            self._tl_last_t = now
        else:
            self.dsp.slider_seek_pct = val / 1000.0
        self.last_slider_val = val

    def _tl_released(self):
        if not self.dsp.slider_scrub_mode:
            self.dsp.playhead = float(self.dsp.lead + self.dsp.slider_seek_pct * self.dsp.n_frames)
            if not self.dsp.tonearm_lifted:
                self.dsp.needle_event = "drop"
        self.dsp.slider_pressed = False
        self.dsp.hand_target = 0.0

    def _on_scratch(self, kind, deg_per_s):
        d = self.dsp
        if kind == "press":
            d.is_holding = True
            d.hand_target = 0.0
        elif kind == "move":
            ratio = float(np.clip((deg_per_s / 360.0) / REV_HZ_33, -15.0, 15.0))
            d.hand_target = 0.65 * ratio + 0.35 * d.hand_target
        else:
            d.is_holding = False
            d.hand_target = 0.0

    def _gui_tick(self):
        now = time.perf_counter()
        dt = min(0.1, now - self._last_tick)
        self._last_tick = now
        d = self.dsp
        self.visual_angle += d.speed * REV_HZ_33 * 360.0 * dt
        self.platter.set_state(self.visual_angle, d.progress, d.contact)
        if not self.timeline.isSliderDown():
            v = int(d.progress * 1000)
            self.last_slider_val = v
            self.timeline.setValue(v)
        txt = f"Platter: {d.speed * 100.0 / 3.0:5.2f} RPM"
        if d.in_runout and not d.tonearm_lifted:
            txt += "   |   Run-out groove (locked)"
        self.lbl_rpm.setText(txt)

    # ------------------------------------------------------------ export
    def _start_export(self):
        fmt = {0: ("PCM_24", ".wav"), 1: ("PCM_16", ".wav"), 2: ("FLOAT", ".wav"), 3: ("PCM_24", ".flac")}
        subtype, ext = fmt[self.cbo_format.currentIndex()]
        path, _ = QFileDialog.getSaveFileName(self, "Export Rendered Vinyl", f"vinyl_export{ext}",
                                              f"Audio Files (*{ext});;All Files (*)")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        if os.path.exists(path):
            r = QMessageBox.question(self, "File Exists",
                                     f"{os.path.basename(path)} already exists.\nOverwrite?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        self.btn_export.setEnabled(False)
        self._progress_dlg = QProgressDialog(f"Rendering vinyl DSP...\n{path}", "Cancel", 0, 100, self)
        self._progress_dlg.setWindowTitle("Exporting")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setValue(0)
        self._export_worker = ExportWorker(self.dsp, path, self.volume_val, subtype)
        self._export_worker.progress.connect(self._progress_dlg.setValue)
        self._export_worker.finished.connect(self._export_done)
        self._export_worker.error.connect(self._export_error)
        self._progress_dlg.canceled.connect(self._cancel_export)
        self._export_worker.start()

    def _cancel_export(self):
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.requestInterruption()
            self._export_worker.wait()
        self.btn_export.setEnabled(True)

    def _export_done(self, path):
        self._progress_dlg.close()
        self.btn_export.setEnabled(True)
        QMessageBox.information(self, "Export Complete",
                                f"Vinyl render saved!\n\nFile: {path}\n"
                                f"Size: {os.path.getsize(path) / 1_048_576:.1f} MB")

    def _export_error(self, msg):
        self._progress_dlg.close()
        self.btn_export.setEnabled(True)
        QMessageBox.critical(self, "Export Failed", f"Export error:\n\n{msg}")

    def closeEvent(self, event):
        self._timer.stop()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.requestInterruption()
            self._export_worker.wait()
        event.accept()


# ============================================================
# ENTRY POINT
# ============================================================
def pick_device_rate(preferred):
    """Prefer the file's own rate (no resampling needed); fall back to the device default."""
    try:
        sd.check_output_settings(samplerate=preferred, channels=2)
        return int(preferred)
    except Exception:
        pass
    try:
        return int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        return 48000


def main():
    print("Starting Fluance RT85 emulator...", flush=True)
    app = QApplication(sys.argv)
    print("Opening file picker (check your taskbar - it may be behind this window)...", flush=True)
    path, _ = QFileDialog.getOpenFileName(
        None, "Select Audio Track to Play (Cancel = demo tone)", "",
        "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)")
    if path:
        print(f"Loading {path} ...", flush=True)
        data, sample_rate = load_audio_file(path)
    else:
        print("No file chosen - using built-in demo.", flush=True)
        data, sample_rate = synth_demo()
    dev_sr = pick_device_rate(sample_rate)
    print(f"Audio device rate: {dev_sr} Hz (source {sample_rate} Hz)", flush=True)
    print("Building vinyl model (can take several seconds for long tracks)...", flush=True)
    engine = TurntableDSP(data, sample_rate, dev_sr, riaa=True)
    print("Starting audio + window...", flush=True)
    win = TurntableWindow(engine, dev_sr)
    win.show()
    win.raise_()
    win.activateWindow()
    print("Ready.", flush=True)
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        input("\nSomething went wrong (see above). Press Enter to close...")
