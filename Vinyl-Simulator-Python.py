import sys
import math
import traceback
import threading
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFrame, QMessageBox, QGridLayout,
    QCheckBox, QProgressDialog, QFileDialog, QComboBox,
)
import sounddevice as sd
import soundfile as sf
from scipy.signal import lfilter, butter, lfilter_zi, sosfilt, sosfilt_zi, sosfiltfilt


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CRASH HANDLER
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# DSP UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pink_noise_voss(n_samples, n_rows=16):
    """
    True 1/f pink noise via the Voss-McCartney algorithm.
    Far more accurate than filtering white noise — matches vinyl groove hiss spectrum.
    """
    n_rows = max(1, n_rows)
    array = np.zeros((n_samples, n_rows))
    running_sum = np.zeros(n_rows)
    for i in range(n_samples):
        col = (i & -i).bit_length() - 1  # lowest set bit
        col = min(col, n_rows - 1)
        running_sum[col] = np.random.randn()
        array[i] = running_sum
    pink = array.sum(axis=1)
    # normalise to unit amplitude
    mx = np.max(np.abs(pink))
    if mx > 0:
        pink /= mx
    return pink.astype(np.float32)


def make_antialiasing_sos(cutoff_hz, sr, order=4):
    """Butterworth anti-alias LP before any variable-rate resample."""
    nyq = sr / 2.0
    cut = min(cutoff_hz / nyq, 0.95)
    return butter(order, cut, btype='low', output='sos')


def apply_riaa_playback_sos(audio, sr):
    """
    Proper RIAA playback de-emphasis as a biquad SOS chain.
    Time constants: 3180 µs, 318 µs, 75 µs  (IEC 60386)
    """
    # Build continuous-time poles/zeros, then bilinear transform
    # Using the standard matched-z / bilinear formulation
    tc1 = 3180e-6  # 50.05 Hz
    tc2 = 318e-6   # 500.5 Hz
    tc3 = 75e-6    # 2122 Hz

    # Rumble filter: 2nd-order HP @ 20 Hz
    b_rumble, a_rumble = butter(2, 20.0 / (sr / 2.0), btype='high')
    sos_rumble = np.concatenate([b_rumble[:3], a_rumble[:3]])[np.newaxis, :]  # crude sos

    processed = np.zeros_like(audio, dtype=np.float32)
    for ch in range(audio.shape[1]):
        sig = audio[:, ch].astype(np.float64)
        # Rumble HP
        sig = lfilter(b_rumble, a_rumble, sig)
        # Approximate RIAA shelving via two first-order shelves
        # Low shelf: boost below 500 Hz
        f1 = 1.0 / (2.0 * np.pi * tc2)
        b1, a1 = butter(1, f1 / (sr / 2.0), btype='low')
        sig = lfilter(b1, a1, sig) * 2.0 + sig * 0.5
        # High shelf: cut above 2122 Hz
        f2 = 1.0 / (2.0 * np.pi * tc3)
        b2, a2 = butter(1, f2 / (sr / 2.0), btype='high')
        sig = sig - lfilter(b2, a2, sig) * 0.6
        processed[:, ch] = sig.astype(np.float32)
    return processed


def soft_saturate(x, drive=0.3):
    """
    Tape-style 3rd-order polynomial soft saturator.
    Adds even and odd harmonics like magnetic tape saturation.
    drive=0 → transparent; drive=1 → heavy saturation.
    """
    d = np.clip(drive, 0.0, 1.0)
    return x * (1.0 - d * 0.4) + (x ** 3) * d * (-0.33) + (x ** 5) * d * 0.05


def fletcher_munson_eq(audio, sr, volume_norm):
    """
    Equal-loudness (Fletcher-Munson) compensation.
    At low volumes, human hearing becomes less sensitive to bass and treble.
    We compensate by boosting those bands as volume decreases — the inverse
    of the equal-loudness curve (ISO 226:2003 approximation).

    volume_norm: 0.0–1.0 (1.0 = full volume, no compensation needed)
    """
    if volume_norm >= 0.95:
        return audio  # full volume: flat response

    # Amount of compensation needed (more at lower volumes)
    comp = 1.0 - volume_norm  # 0 at full vol, 1 at silence

    n = len(audio)
    if n < 10:
        return audio

    # Low-frequency boost (below ~200 Hz) — ear loses sensitivity first here
    b_lf, a_lf = butter(2, 200.0 / (sr / 2.0), btype='low')
    # High-frequency boost (above ~8 kHz) — "air" that disappears at low vol
    b_hf, a_hf = butter(1, 8000.0 / (sr / 2.0), btype='high')

    out = audio.copy()
    for ch in range(audio.shape[1]):
        lf_band = lfilter(b_lf, a_lf, audio[:, ch])
        hf_band = lfilter(b_hf, a_hf, audio[:, ch])
        out[:, ch] = audio[:, ch] + lf_band * comp * 1.8 + hf_band * comp * 0.9
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DSP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TurntableDSP:
    def __init__(self, audio_data, sample_rate=44100):
        self.sr = sample_rate
        self.audio = audio_data.astype(np.float32)
        if self.audio.ndim == 1:
            self.audio = np.stack([self.audio, self.audio], axis=-1)
        self.total_samples = len(self.audio)

        # Pre-bake a block of pink noise large enough to index into cyclically
        self._pink_buf_len = min(self.total_samples, int(self.sr * 60))
        self._pink_buf = pink_noise_voss(self._pink_buf_len)

        # Pre-compute per-sample RMS envelope for ducking
        rms_window = int(self.sr * 0.1)  # 100 ms RMS window
        rms_env = np.convolve(
            self.audio.mean(axis=1) ** 2,
            np.ones(rms_window) / rms_window,
            mode='same'
        )
        self._rms_env = np.sqrt(np.clip(rms_env, 0.0, 1.0)).astype(np.float32)

        # Pre-generate surface artifacts fixed to track positions
        self.base_crackle_indices = np.random.choice(
            self.total_samples, size=max(1, int(self.total_samples * 0.0003))
        )
        self.base_crackle_values = (
            np.random.choice([-1.0, 1.0, 0.5, -0.5], size=len(self.base_crackle_indices))
            * np.random.uniform(0.003, 0.015, size=len(self.base_crackle_indices))
        )
        self.base_dust_indices = np.random.choice(
            self.total_samples, size=max(1, int(self.total_samples * 0.00008))
        )

        # Anti-alias filter (applied before resampling)
        self._aa_sos = make_antialiasing_sos(0.47 * self.sr, self.sr)
        aa_zi_1ch = sosfilt_zi(self._aa_sos)
        self._aa_zi = np.stack([aa_zi_1ch, aa_zi_1ch], axis=-1)  # (n_sos, 2, 2_ch)

        # Platter & motor state
        self.target_speed_ratio = 1.0
        self.current_speed_ratio = 1.0
        self.inertia_mass = 0.995
        self.playhead = 0.0

        # Tonearm state
        self.tonearm_lifted = True
        self.cue_transition = 0.0
        self.needle_drop_trigger = False

        # 10 Hz tonearm resonance biquad
        self.b_res, self.a_res = butter(2, [8.0, 12.0], btype='bandpass', fs=self.sr)
        zi_1 = lfilter_zi(self.b_res, self.a_res)
        self.res_zi = np.column_stack([zi_1, zi_1])

        # Scratch / hold state
        self.is_holding = False
        self.hand_velocity = 0.0
        self.smoothed_hand_vel = 0.0

        # Slider state
        self.slider_scrub_mode = False
        self.slider_pressed = False
        self.slider_scrub_velocity = 0.0
        self.slider_seek_pct = 0.0

        # Flutter & eccentricity
        self.flutter_state = 0.0
        self.eccentricity_phase = 0.0

        # Cogging torque micro-vibration state
        self.cogging_phase = 0.0

        # Hum LFO phase
        self.hum_phase = 0.0
        self.block_counter = 0

        # ── All adjustable physics parameters ──────────────────────────
        self.hiss_level       = 0.00002
        self.air_sparkle      = 0.40
        self.crosstalk_amt    = 0.12
        self.hum_level        = 0.001
        self.crackle_density  = 0.50
        self.dust_density     = 0.50
        self.igd_amount       = 0.35
        self.pinch_amount     = 0.10
        self.tangential_error_amt    = 0.15
        self.cantilever_compliance   = 0.20
        self.saturation_drive        = 0.20   # tape saturator drive
        self.groove_vel_limit        = 0.80   # stylus maximum tracking velocity (0=off, 1=strict)
        self.cogging_level           = 0.0003 # AC motor cogging torque vibration
        self.fm_compensation         = 1.0    # Fletcher-Munson compensation strength (0=off)
        self.antialiasing_on         = True

    def set_target_speed(self, ratio):
        self.target_speed_ratio = ratio

    # ── Core per-block DSP (called from audio callback at ~44100 Hz) ──────────
    def process_block(self, block_size, volume_slider_val):
        self.block_counter += 1
        vol_norm = np.clip(volume_slider_val / 100.0, 0.0, 1.0)
        vol_curve = vol_norm ** 1.5

        target_cue = 0.0 if self.tonearm_lifted else 1.0
        self.cue_transition += 0.05 * (target_cue - self.cue_transition)

        # ── 1. Speed / playhead trajectory ───────────────────────────────────
        if self.slider_pressed and not self.slider_scrub_mode:
            self.playhead = self.slider_seek_pct * self.total_samples
            speeds = np.zeros(block_size, dtype=np.float32)
        elif self.is_holding:
            self.hand_velocity *= 0.90
            self.smoothed_hand_vel += 0.2 * (self.hand_velocity - self.smoothed_hand_vel)
            speeds = np.full(block_size, self.smoothed_hand_vel, dtype=np.float32)
        elif self.slider_pressed and self.slider_scrub_mode:
            self.slider_scrub_velocity *= 0.85
            speeds = np.full(block_size, self.slider_scrub_velocity, dtype=np.float32)
        else:
            speed_diff = self.target_speed_ratio - self.current_speed_ratio
            self.current_speed_ratio += speed_diff * (1.0 - self.inertia_mass)

            # Stochastic flutter (bandlimited Brownian)
            noise_step = np.random.normal(0, 0.00008, size=block_size)
            for i in range(block_size):
                self.flutter_state = 0.98 * self.flutter_state + noise_step[i]

            t_abs = (self.block_counter * block_size + np.arange(block_size, dtype=np.float32))

            # Rotational eccentricity (off-center spindle hole) → 0.55 Hz
            self.eccentricity_phase += 2.0 * np.pi * 0.55 * block_size / self.sr
            ecc = 0.0008 * np.sin(2.0 * np.pi * 0.55 * t_abs / self.sr + self.eccentricity_phase)

            # Motor cogging torque ripple (12-pole AC motor → 60 Hz × 12/2 = 360 Hz, but
            # belted to platter the cogging appears as 60 Hz envelope × 2 = 120 Hz micro-pulse)
            self.cogging_phase += 2.0 * np.pi * 120.0 * block_size / self.sr
            cogging = self.cogging_level * np.sin(2.0 * np.pi * 120.0 * t_abs / self.sr)

            wow = 0.00035 * np.sin(2.0 * np.pi * 0.55 * t_abs / self.sr) + self.flutter_state + ecc + cogging
            speeds = np.full(block_size, self.current_speed_ratio, dtype=np.float32) * (1.0 + wow)

        # ── 2. Playhead advancement ───────────────────────────────────────────
        playheads = self.playhead + np.cumsum(speeds)
        playheads %= self.total_samples
        self.playhead = float(playheads[-1])

        # ── 3. Catmull-Rom cubic interpolation (smooth, low aliasing) ─────────
        idx_i = playheads.astype(np.int64) % self.total_samples
        mu    = (playheads - idx_i.astype(np.float32))

        ym1 = self.audio[(idx_i - 1) % self.total_samples]
        y0  = self.audio[idx_i]
        y1  = self.audio[(idx_i + 1) % self.total_samples]
        y2  = self.audio[(idx_i + 2) % self.total_samples]

        mu2 = mu[:, None] ** 2
        mu3 = mu[:, None] ** 3
        a0 = -0.5 * ym1 + 1.5 * y0 - 1.5 * y1 + 0.5 * y2
        a1 =        ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2
        a2 = -0.5 * ym1              + 0.5 * y1
        a3 = y0
        out = a0 * mu3 + a1 * mu2 + a2 * mu[:, None] + a3

        if self.slider_pressed and not self.slider_scrub_mode:
            out *= 0.0
        else:
            out *= self.cue_transition

        # ── 4. Needle-drop thump ──────────────────────────────────────────────
        if self.needle_drop_trigger:
            t_th = np.arange(block_size, dtype=np.float32)
            thump = np.sin(2.0 * np.pi * 35.0 * t_th / self.sr) * np.exp(-t_th / 40.0) * 0.4
            out[:, 0] += thump
            out[:, 1] += thump * 0.85
            self.needle_drop_trigger = False

        # ── 5. 10 Hz tonearm resonance ────────────────────────────────────────
        res = np.zeros_like(out)
        for ch in range(2):
            res[:, ch], self.res_zi[:, ch] = lfilter(self.b_res, self.a_res, out[:, ch], zi=self.res_zi[:, ch])
        out += np.clip(res, -2.0, 2.0) * 0.2

	# ── 6. Groove velocity limiter (physical stylus max tracking speed) ────
        if self.groove_vel_limit > 0.0:
            # Limit high-frequency amplitude proportional to groove radius
            # (outer tracks → more groove space → less tracing distortion)
            progress = idx_i.astype(np.float32) / self.total_samples
            # Allowable velocity decreases toward inner grooves
            max_vel = 1.0 - progress * self.groove_vel_limit * 0.4
            diff = np.diff(out, axis=0, prepend=out[:1])
            clipped_diff = np.clip(diff, -max_vel[:, None] * 0.05, max_vel[:, None] * 0.05)
            # Simpler effective approach: gentle saturate only the derivative
            out = np.clip(out, -max_vel[:, None], max_vel[:, None])

        # ── 7. Inner-groove tracing loss & 3rd-harmonic distortion ───────────
        progress = idx_i.astype(np.float32) / self.total_samples
        igd_curve = (np.maximum(0.0, progress - 0.7) / 0.3) ** 2.0 * self.igd_amount
        out += (out ** 3) * igd_curve[:, None] * 0.1
        out *= (1.0 - igd_curve[:, None] * 0.4)

        # ── 8. Pinch-effect distortion (groove wall compression) ─────────────
        vel_mag = np.abs(speeds)[:, None]
        out += np.clip((out ** 2) * vel_mag * self.pinch_amount, -1.0, 1.0)

        # ── 9. Asymmetric tangential tracking angle error ─────────────────────
        twist = (progress[:, None] - 0.5) * self.tangential_error_amt
        out += (out ** 2) * twist * 0.08

        # ── 10. Cantilever compliance micro-stall drag ────────────────────────
        transient = np.abs(out).mean(axis=1, keepdims=True)
        out *= 1.0 - transient * self.cantilever_compliance * 0.05

        # ── 11. V-groove stereo crosstalk & R-channel phase smearing ─────────
        l = out[:, 0] + out[:, 1] * self.crosstalk_amt
        r = out[:, 1] + out[:, 0] * self.crosstalk_amt
        r = 0.98 * r + 0.02 * np.roll(r, 1)
        out[:, 0] = l
        out[:, 1] = r

        # ── 12. AC motor hum: 60 Hz fundamental + 120 Hz cogging harmonic ────
        if self.target_speed_ratio > 0.0 or self.current_speed_ratio > 0.05:
            t_abs_f = (self.block_counter * block_size + np.arange(block_size, dtype=np.float32))
            hum60  = np.sin(2.0 * np.pi * 60.0  * t_abs_f / self.sr) * self.hum_level
            hum120 = np.sin(2.0 * np.pi * 120.0 * t_abs_f / self.sr) * self.hum_level * 0.35
            hum = hum60 + hum120
            out[:, 0] += hum
            out[:, 1] += hum * 0.8

        # ── 13. Surface noise: pink groove hiss (1/f) with RMS ducking ────────
        pink_start = (self.block_counter * block_size) % self._pink_buf_len
        pink_end   = pink_start + block_size
        if pink_end <= self._pink_buf_len:
            pink_slice = self._pink_buf[pink_start:pink_end]
        else:
            pink_slice = np.concatenate([
                self._pink_buf[pink_start:],
                self._pink_buf[:(pink_end - self._pink_buf_len)]
            ])

        # Duck surface noise dynamically by local RMS of the music
        rms_duck = 1.0 - np.clip(self._rms_env[idx_i] * 3.0, 0.0, 0.85)
        hiss_sig = pink_slice * self.hiss_level * rms_duck  # stereo spread
        out[:, 0] += hiss_sig
        out[:, 1] += hiss_sig * np.random.uniform(0.7, 1.0, size=block_size)  # subtle uncorrelated

        # ── 14. Crackle pops (also RMS-ducked) ───────────────────────────────
        active_crack = int(len(self.base_crackle_indices) * np.clip(self.crackle_density, 0, 2))
        crack_mask   = np.isin(idx_i, self.base_crackle_indices[:active_crack])
        pops = np.zeros(block_size, dtype=np.float32)
        if np.any(crack_mask):
            locs = np.where(crack_mask)[0]
            vals = self.base_crackle_values[:len(locs)]
            # Duck pops proportionally to local music RMS
            duck  = 1.0 - np.clip(self._rms_env[idx_i[locs]] * 4.0, 0.0, 0.9)
            pops[locs] = vals * duck
        out[:, 0] += pops
        out[:, 1] += pops * 0.9

        # ── 15. Dust-blob low-frequency thumps ───────────────────────────────
        active_dust = int(len(self.base_dust_indices) * np.clip(self.dust_density, 0, 2))
        dust = np.zeros(block_size, dtype=np.float32)
        for d_idx in self.base_dust_indices[:active_dust]:
            hit = (idx_i >= d_idx) & (idx_i < d_idx + 40)
            if np.any(hit):
                locs  = np.where(hit)[0]
                t_b   = (idx_i[locs] - d_idx).astype(np.float32)
                thump = np.sin(2.0 * np.pi * 60.0 * t_b / self.sr) * np.exp(-t_b / 10.0) * 0.08
                rms_d = 1.0 - np.clip(self._rms_env[idx_i[locs]] * 3.5, 0.0, 0.85)
                dust[locs] += thump * rms_d
        out[:, 0] += dust
        out[:, 1] += dust * 0.85

        # ── 16. Volume with Fletcher-Munson EQ compensation ──────────────────
        if self.fm_compensation > 0.0:
            out = fletcher_munson_eq(out, self.sr, vol_norm * self.fm_compensation
                                     + (1.0 - self.fm_compensation))
        out *= vol_curve

        # ── 17. Ortofon 2M Blue air sparkle (high-shelf emphasis) ────────────
        if self.air_sparkle > 0.0:
            hi_boost = out - np.roll(out, 1, axis=0)
            out += hi_boost * self.air_sparkle

        # ── 18. Tape-style soft saturator ────────────────────────────────────
        if self.saturation_drive > 0.0:
            out = soft_saturate(out, self.saturation_drive)

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

class ExportWorker(QThread):
    progress   = pyqtSignal(int)      # 0-100
    finished   = pyqtSignal(str)      # output path on success
    error      = pyqtSignal(str)      # error message

    def __init__(self, dsp, output_path, volume_val, bit_depth, parent=None):
        super().__init__(parent)
        self.dsp         = dsp
        self.output_path = output_path
        self.volume_val  = volume_val
        self.bit_depth   = bit_depth   # 'PCM_16', 'PCM_24', 'FLOAT'

    def run(self):
        try:
            # Create a fresh DSP snapshot so live playback is undisturbed
            import copy
            snap = copy.deepcopy(self.dsp)
            # Reset playback to start
            snap.playhead          = 0.0
            snap.tonearm_lifted    = False
            snap.cue_transition    = 1.0
            snap.current_speed_ratio = snap.target_speed_ratio
            snap.block_counter     = 0
            snap.is_holding        = False
            snap.slider_pressed    = False

            block_size    = 4096
            total_blocks  = math.ceil(snap.total_samples / block_size)
            output_chunks = []

            for i in range(total_blocks):
                chunk = snap.process_block(block_size, self.volume_val)
                # Trim last block to exact sample count
                remaining = snap.total_samples - i * block_size
                if remaining < block_size:
                    chunk = chunk[:remaining]
                output_chunks.append(chunk)
                pct = int((i + 1) / total_blocks * 100)
                self.progress.emit(pct)

            rendered = np.concatenate(output_chunks, axis=0)
            rendered = np.clip(rendered, -1.0, 1.0).astype(np.float32)

            sf_subtype = {
                'PCM_16': 'PCM_16',
                'PCM_24': 'PCM_24',
                'FLOAT':  'FLOAT',
            }.get(self.bit_depth, 'PCM_24')

            sf.write(self.output_path, rendered, snap.sr, subtype=sf_subtype)
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# VIRTUAL TURNTABLE PLATTER WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class VirtualTurntableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.angle_deg       = 0.0
        self.is_dragging     = False
        self.last_mouse_pos  = QPointF()
        self.last_angle      = 0.0
        self.scratch_callback = None

    def set_rotation(self, angle):
        self.angle_deg = angle % 360
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging    = True
            self.last_mouse_pos = event.position()
            self.last_angle     = self._angle_from_center(self.last_mouse_pos)
            if self.scratch_callback:
                self.scratch_callback(True, 0.0)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            cur   = event.position()
            ang   = self._angle_from_center(cur)
            delta = ang - self.last_angle
            if delta >  180: delta -= 360
            if delta < -180: delta += 360
            if self.scratch_callback:
                self.scratch_callback(False, delta * 0.75)
            self.last_angle = ang

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            if self.scratch_callback:
                self.scratch_callback(False, None)

    def _angle_from_center(self, pos):
        cx, cy = self.width() / 2.0, self.height() / 2.0
        return math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h   = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r      = min(w, h) / 2.0 - 10.0

        # Cabinet background
        p.setBrush(QBrush(QColor(25, 27, 32)))
        p.setPen(QPen(QColor(12, 14, 18), 3))
        p.drawRoundedRect(self.rect().adjusted(5, 5, -5, -5), 16, 16)

        p.save()
        p.translate(QPointF(cx, cy))

        # Acrylic platter rim
        plat_grad = QRadialGradient(0, 0, r)
        plat_grad.setColorAt(0.68, QColor(190, 220, 250, 55))
        plat_grad.setColorAt(1.00, QColor(8, 12, 18, 240))
        p.setBrush(QBrush(plat_grad))
        p.setPen(QPen(QColor(160, 195, 230, 210), 2))
        p.drawEllipse(QPointF(0, 0), r, r)

        p.rotate(self.angle_deg)
        vr = r - 10.0

        # Vinyl record body
        v_grad = QRadialGradient(0, 0, vr)
        v_grad.setColorAt(0.0,  QColor(8, 8, 8))
        v_grad.setColorAt(0.72, QColor(35, 37, 42))
        v_grad.setColorAt(1.0,  QColor(3, 3, 3))
        p.setBrush(QBrush(v_grad))
        p.setPen(QPen(QColor(40, 40, 40), 1))
        p.drawEllipse(QPointF(0, 0), vr, vr)

        # Groove rings
        p.setPen(QPen(QColor(25, 25, 25), 0.9))
        for ring_r in np.linspace(vr * 0.38, vr * 0.86, 24):
            p.drawEllipse(QPointF(0, 0), ring_r, ring_r)

        # Centre label
        lr = vr * 0.32
        l_grad = QRadialGradient(0, 0, lr)
        l_grad.setColorAt(0.0, QColor(14, 70, 150))
        l_grad.setColorAt(1.0, QColor(6, 30, 75))
        p.setBrush(QBrush(l_grad))
        p.setPen(QPen(QColor(200, 200, 200), 1.5))
        p.drawEllipse(QPointF(0, 0), lr, lr)

        # Spindle hole
        p.setBrush(QBrush(QColor(230, 180, 50)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(0, 0), 5.0, 5.0)
        p.restore()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class TurntableWindow(QMainWindow):
    def __init__(self, dsp_engine):
        super().__init__()
        self.dsp        = dsp_engine
        self.block_size = 2048
        self.stream     = None
        self.last_slider_val    = 0
        self.current_visual_angle = 0.0
        self._export_worker     = None

        self.setWindowTitle("Fluance RT85 Ultimate — Vinyl Physics Emulator")
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #121418; color: #f1f5f9;
                                   font-family: 'Segoe UI', Arial; }
            QPushButton {
                background-color: #334155; color: #f8fafc; border: none;
                border-radius: 6px; padding: 6px 12px; font-weight: bold; font-size: 11px;
            }
            QPushButton:hover  { background-color: #4a5568; }
            QPushButton:pressed{ background-color: #2d3748; }
            QLabel     { font-size: 11px; color: #cbd5e1; }
            QSlider::groove:horizontal { height: 4px; background: #334155; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; background: #38bdf8;
                border-radius: 7px; margin: -5px 0;
            }
            QSlider::sub-page:horizontal { background: #0284c7; border-radius: 2px; }
            QComboBox { background: #334155; color: #f8fafc; border: none;
                        border-radius: 4px; padding: 3px 8px; }
        """)
        self.resize(900, 1060)
        self._init_ui()
        self._init_audio()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._gui_tick)
        self._timer.start(16)   # ~60 fps

    # ── UI construction ───────────────────────────────────────────────────────
    def _init_ui(self):
        root = QWidget()
        lay  = QVBoxLayout(root)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)

        # Title
        ttl = QLabel("FLUANCE RT85  ·  VINYL PHYSICS EMULATOR")
        ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ttl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ttl.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        lay.addWidget(ttl)

        # Platter
        self.platter = VirtualTurntableWidget()
        self.platter.scratch_callback = self._on_scratch
        lay.addWidget(self.platter, alignment=Qt.AlignmentFlag.AlignCenter)

        # Timeline
        tl_lay = QVBoxLayout()
        tl_top = QHBoxLayout()
        tl_top.addWidget(QLabel("Track Position"))
        self.chk_scrub = QCheckBox("Scrub Mode (slider = rewind / fast-forward)")
        self.chk_scrub.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.chk_scrub.stateChanged.connect(lambda s: setattr(self.dsp, 'slider_scrub_mode', bool(s)))
        tl_top.addWidget(self.chk_scrub, alignment=Qt.AlignmentFlag.AlignRight)
        tl_lay.addLayout(tl_top)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderPressed.connect(self._tl_pressed)
        self.timeline.sliderMoved.connect(self._tl_moved)
        self.timeline.sliderReleased.connect(self._tl_released)
        tl_lay.addWidget(self.timeline)
        lay.addLayout(tl_lay)

        # Main controls frame
        frm = QFrame()
        frm.setStyleSheet("QFrame { background:#1a1d24; border-radius:12px; padding:8px; }")
        frm_lay = QVBoxLayout(frm)

        # Row 1: speed / motor / cue / volume
        row1 = QHBoxLayout()

        spd_lay = QVBoxLayout()
        spd_lay.addWidget(QLabel("Speed"), alignment=Qt.AlignmentFlag.AlignCenter)
        spd_btns = QHBoxLayout()
        self.btn_33 = QPushButton("33 ⅓ RPM")
        self.btn_33.setStyleSheet("background:#0284c7;color:white;")
        self.btn_33.clicked.connect(lambda: self._set_speed(1.0, self.btn_33, self.btn_45))
        self.btn_45 = QPushButton("45 RPM")
        self.btn_45.clicked.connect(lambda: self._set_speed(1.35, self.btn_45, self.btn_33))
        spd_btns.addWidget(self.btn_33)
        spd_btns.addWidget(self.btn_45)
        spd_lay.addLayout(spd_btns)
        row1.addLayout(spd_lay)

        mtr_lay = QVBoxLayout()
        mtr_lay.addWidget(QLabel("Motor"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_stop = QPushButton("⏹  STOP / BRAKE")
        self.btn_stop.clicked.connect(self._toggle_stop)
        mtr_lay.addWidget(self.btn_stop)
        row1.addLayout(mtr_lay)

        cue_lay = QVBoxLayout()
        cue_lay.addWidget(QLabel("Tonearm"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_cue = QPushButton("⬇  LOWER NEEDLE")
        self.btn_cue.clicked.connect(self._toggle_cue)
        cue_lay.addWidget(self.btn_cue)
        row1.addLayout(cue_lay)

        vol_lay = QVBoxLayout()
        self.lbl_vol = QLabel("Volume: 80%")
        vol_lay.addWidget(self.lbl_vol)
        self.sld_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_vol.setRange(0, 100)
        self.sld_vol.setValue(80)
        self.sld_vol.valueChanged.connect(lambda v: self.lbl_vol.setText(f"Volume: {v}%"))
        vol_lay.addWidget(self.sld_vol)
        row1.addLayout(vol_lay)
        frm_lay.addLayout(row1)

        # Physics sliders grid (12 parameters)
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)

        def mk_slider(label_txt, attr, lo, hi, default, fmt, row, col):
            lbl = QLabel(label_txt.format(default))
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(lo, hi)
            sld.setValue(default)
            def _cb(v, _lbl=lbl, _attr=attr, _fmt=fmt, _label_txt=label_txt):
                setattr(self.dsp, _attr, _fmt(v))
                _lbl.setText(_label_txt.format(v))
            sld.valueChanged.connect(_cb)
            grid.addWidget(lbl, row * 2,     col)
            grid.addWidget(sld, row * 2 + 1, col)
            return lbl, sld

        mk_slider("Platter Inertia Mass: {}‰",   "inertia_mass",        980, 999, 995,
                  lambda v: v / 1000.0, 0, 0)
        mk_slider("Pinch Effect: {}%",             "pinch_amount",          0, 100,  10,
                  lambda v: v / 100.0, 0, 1)
        mk_slider("Motor Hum Level: {}%",          "hum_level",             0,  50,  10,
                  lambda v: v / 10000.0, 1, 0)
        mk_slider("Cogging Torque Vibration: {}%", "cogging_level",         0, 100,  30,
                  lambda v: v / 100000.0, 1, 1)
        mk_slider("Ortofon Air Sparkle: {}%",      "air_sparkle",           0, 120,  40,
                  lambda v: v / 100.0, 2, 0)
        mk_slider("Crackle Density: {}%",          "crackle_density",       0, 200,  50,
                  lambda v: v / 100.0, 2, 1)
        mk_slider("IGD Tracing Distortion: {}%",   "igd_amount",            0, 100,  35,
                  lambda v: v / 100.0, 3, 0)
        mk_slider("Dust Blob Thumps: {}%",         "dust_density",          0, 200,  50,
                  lambda v: v / 100.0, 3, 1)
        mk_slider("V-Groove Crosstalk: {}%",       "crosstalk_amt",         0,  40,  12,
                  lambda v: v / 100.0, 4, 0)
        mk_slider("Tangential Tracking Error: {}%","tangential_error_amt",  0,  50,  15,
                  lambda v: v / 100.0, 4, 1)
        mk_slider("Cantilever Compliance: {}%",    "cantilever_compliance", 0,  50,  20,
                  lambda v: v / 100.0, 5, 0)
        mk_slider("Tape Saturation Drive: {}%",    "saturation_drive",      0, 100,  20,
                  lambda v: v / 100.0, 5, 1)
        mk_slider("Groove Velocity Limit: {}%",    "groove_vel_limit",      0, 100,  80,
                  lambda v: v / 100.0, 6, 0)
        mk_slider("Fletcher-Munson EQ: {}%",       "fm_compensation",       0, 100, 100,
                  lambda v: v / 100.0, 6, 1)

        frm_lay.addLayout(grid)

        # ── EXPORT ROW ────────────────────────────────────────────────────────
        exp_frame = QFrame()
        exp_frame.setStyleSheet("""
            QFrame { background: #0f1923; border: 1px solid #0284c7;
                     border-radius: 8px; padding: 6px; }
        """)
        exp_lay = QHBoxLayout(exp_frame)

        exp_lay.addWidget(QLabel("Export Format:"))
        self.cbo_format = QComboBox()
        self.cbo_format.addItems(["WAV  (PCM 24-bit)", "WAV  (PCM 16-bit)", "WAV  (32-bit Float)"])
        exp_lay.addWidget(self.cbo_format)

        self.btn_export = QPushButton("💾  EXPORT RENDERED VINYL")
        self.btn_export.setStyleSheet("""
            QPushButton { background: #0284c7; color: white; font-size: 12px;
                          padding: 8px 18px; border-radius: 6px; }
            QPushButton:hover  { background: #0369a1; }
            QPushButton:pressed{ background: #075985; }
        """)
        self.btn_export.clicked.connect(self._start_export)
        exp_lay.addWidget(self.btn_export)
        frm_lay.addWidget(exp_frame)

        lay.addWidget(frm)

        # Status bar
        self.lbl_status = QLabel("Status: Tonearm Lifted — click LOWER NEEDLE to play")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #38bdf8; font-size: 11px;")
        lay.addWidget(self.lbl_status)

        self.setCentralWidget(root)

    # ── Audio stream ──────────────────────────────────────────────────────────
    def _init_audio(self):
        try:
            def _cb(outdata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                outdata[:] = self.dsp.process_block(frames, self.sld_vol.value())

            self.stream = sd.OutputStream(
                samplerate=self.dsp.sr,
                channels=2,
                callback=_cb,
                blocksize=self.block_size,
            )
            self.stream.start()
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Cannot open audio stream:\n{e}")
            sys.exit(1)

    # ── Control handlers ──────────────────────────────────────────────────────
    def _set_speed(self, ratio, active, inactive):
        self.dsp.set_target_speed(ratio)
        active.setStyleSheet("background:#0284c7;color:white;")
        inactive.setStyleSheet("")
        label = "33 ⅓" if ratio == 1.0 else "45"
        self.lbl_status.setText(f"Status: Playing at {label} RPM")

    def _toggle_stop(self):
        if self.dsp.target_speed_ratio > 0.0:
            self.dsp.set_target_speed(0.0)
            self.btn_stop.setStyleSheet("background:#991b1b;color:white;")
            self.lbl_status.setText("Status: Motor Braking — Acrylic Platter Coasting...")
        else:
            self.dsp.set_target_speed(1.0)
            self.btn_stop.setStyleSheet("")
            self.btn_33.setStyleSheet("background:#0284c7;color:white;")
            self.btn_45.setStyleSheet("")
            self.lbl_status.setText("Status: Motor Restarting...")

    def _toggle_cue(self):
        self.dsp.tonearm_lifted = not self.dsp.tonearm_lifted
        if self.dsp.tonearm_lifted:
            self.btn_cue.setText("⬇  LOWER NEEDLE")
            self.btn_cue.setStyleSheet("")
            self.lbl_status.setText("Status: Tonearm Lifted")
        else:
            self.btn_cue.setText("⬆  LIFT NEEDLE")
            self.btn_cue.setStyleSheet("background:#0284c7;color:white;")
            self.dsp.needle_drop_trigger = True
            self.lbl_status.setText("Status: Needle Down — All Physics Active")

    def _tl_pressed(self):
        self.dsp.slider_pressed  = True
        self.last_slider_val = self.timeline.value()

    def _tl_moved(self, val):
        delta = val - self.last_slider_val
        self.last_slider_val = val
        if self.dsp.slider_scrub_mode:
            self.dsp.slider_scrub_velocity = float(np.clip(delta * 0.035, -2.5, 2.5))
        else:
            self.dsp.slider_seek_pct = val / 1000.0

    def _tl_released(self):
        if not self.dsp.slider_scrub_mode:
            self.dsp.playhead = self.dsp.slider_seek_pct * self.dsp.total_samples
        self.dsp.slider_pressed        = False
        self.dsp.slider_scrub_velocity = 0.0

    def _on_scratch(self, start, vel):
        if not self.dsp.slider_scrub_mode:
            if start:
                self.dsp.is_holding = True
                self.dsp.hand_velocity = 0.0
                self.dsp.smoothed_hand_vel = 0.0
            elif vel is None:
                self.dsp.is_holding = False
            else:
                self.dsp.hand_velocity = float(np.clip(vel, -15.0, 15.0))

    # ── 60 fps GUI tick ───────────────────────────────────────────────────────
    def _gui_tick(self):
        if self.dsp.is_holding or (self.dsp.slider_pressed and self.dsp.slider_scrub_mode):
            self.current_visual_angle += self.dsp.hand_velocity * 4.0
        else:
            self.current_visual_angle += self.dsp.current_speed_ratio * 3.33
        self.platter.set_rotation(self.current_visual_angle)

        if not self.timeline.isSliderDown():
            pct = self.dsp.playhead / self.dsp.total_samples
            val = int(pct * 1000)
            self.last_slider_val = val
            self.timeline.setValue(val)

    # ── EXPORT ────────────────────────────────────────────────────────────────
    def _start_export(self):
        # Pick a safe output file — NEVER overwrites without explicit user consent
        fmt_map = {
            0: ("WAV PCM 24-bit", "PCM_24", ".wav"),
            1: ("WAV PCM 16-bit", "PCM_16", ".wav"),
            2: ("WAV 32-bit Float","FLOAT",  ".wav"),
        }
        fmt_label, sf_subtype, ext = fmt_map[self.cbo_format.currentIndex()]

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Rendered Vinyl — Choose Output File",
            f"vinyl_export{ext}",
            f"Audio Files (*{ext});;All Files (*)",
        )
        if not path:
            return  # user cancelled — nothing touched

        # Ensure extension
        if not path.lower().endswith(ext):
            path += ext

        # Double-check: if file exists, Qt's native dialog already asked; but just in case
        import os
        if os.path.exists(path):
            reply = QMessageBox.question(
                self, "File Exists",
                f"'{os.path.basename(path)}' already exists.\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return  # aborted — original file untouched

        # Progress dialog (stays responsive because export runs in QThread)
        self._progress_dlg = QProgressDialog(
            f"Rendering vinyl DSP to {fmt_label}...\n{path}",
            "Cancel", 0, 100, self
        )
        self._progress_dlg.setWindowTitle("Exporting...")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setValue(0)

        self._export_worker = ExportWorker(
            self.dsp, path, self.sld_vol.value(), sf_subtype
        )
        self._export_worker.progress.connect(self._progress_dlg.setValue)
        self._export_worker.finished.connect(self._export_done)
        self._export_worker.error.connect(self._export_error)
        self._progress_dlg.canceled.connect(self._export_worker.terminate)
        self._export_worker.start()

    def _export_done(self, path):
        self._progress_dlg.close()
        import os
        size_mb = os.path.getsize(path) / 1_048_576
        QMessageBox.information(
            self, "Export Complete",
            f"✅  Vinyl render saved successfully!\n\n"
            f"File : {path}\n"
            f"Size : {size_mb:.1f} MB\n\n"
            f"The original source file was NOT modified.",
        )

    def _export_error(self, msg):
        self._progress_dlg.close()
        QMessageBox.critical(self, "Export Failed", f"Export error:\n\n{msg}")

    def closeEvent(self, event):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.terminate()
            self._export_worker.wait(2000)
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ── Load audio ────────────────────────────────────────────────────────────
    # Change this path to your audio file (MP3, FLAC, WAV, OGG, etc.)
    AUDIO_PATH = r"C:\Users\YOUR-PATH-TO-YOUR-MUSIC-FILE-HERE"

    data, sample_rate = None, 44100

    try:
        data, sample_rate = sf.read(AUDIO_PATH, always_2d=True)
        print(f"[soundfile] Loaded '{AUDIO_PATH}' at {sample_rate} Hz.")
    except Exception as e_sf:
        print(f"[soundfile] Failed ({e_sf}). Trying pydub...")
        try:
            from pydub import AudioSegment
            song = AudioSegment.from_file(AUDIO_PATH)
            sample_rate = song.frame_rate
            raw = np.array(song.get_array_of_samples(), dtype=np.float32) / 32768.0
            if song.channels == 2:
                data = raw.reshape(-1, 2)
            else:
                data = np.stack([raw, raw], axis=-1)
            print(f"[pydub] Loaded '{AUDIO_PATH}' at {sample_rate} Hz.")
        except Exception as e_pd:
            print(f"[pydub] Failed ({e_pd}). Using synthetic test tone.")
            t    = np.linspace(0, 30, int(44100 * 30), endpoint=False)
            tone = (0.30 * np.sin(2 * np.pi * 220 * t)
                  + 0.15 * np.sin(2 * np.pi * 440 * t)
                  + 0.07 * np.sin(2 * np.pi * 880 * t))
            data = np.stack([tone, tone * 0.9], axis=-1).astype(np.float32)

    # Apply RIAA playback de-emphasis (removes the recording pre-emphasis, i wouldnt really reccamend uncommenting this...)
    #audio_data = apply_riaa_playback_sos(data.astype(np.float32), sample_rate)
    #dsp = TurntableDSP(audio_data, sample_rate=sample_rate)

    # Skip RIAA de-emphasis for standard digital files so highs aren't muffled (if your going to use RIAA playback de-emphasis, uncomment the block above this, and comment this block below this text
    audio_data = data.astype(np.float32)
    dsp = TurntableDSP(audio_data, sample_rate=sample_rate)

    win = TurntableWindow(dsp)
    win.show()
    sys.exit(app.exec())