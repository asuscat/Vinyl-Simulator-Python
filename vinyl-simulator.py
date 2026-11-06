import sys
import math
import traceback
import threading
import copy
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi, sosfiltfilt, resample
from PyQt6.QtCore import Qt, QTimer, QPointF, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QFrame,
    QMessageBox,
    QGridLayout,
    QCheckBox,
    QProgressDialog,
    QFileDialog,
    QComboBox,
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

# ============================================================
# DSP UTILITIES
# ============================================================
def pink_noise_voss(n_samples, n_rows=16):
    n_samples = max(1, int(n_samples))
    n_rows = max(1, int(n_rows))
    rng = np.random.default_rng()
    array = np.zeros((n_samples, n_rows), dtype=np.float32)
    running_sum = np.zeros(n_rows, dtype=np.float32)
    for i in range(n_samples):
        col = min((i & -i).bit_length() - 1, n_rows - 1)
        running_sum[col] = rng.normal()
        array[i] = running_sum
    pink = array.sum(axis=1)
    pink -= np.mean(pink)
    rms = np.sqrt(np.mean(pink ** 2))
    if rms > 0:
        pink /= rms
    return pink.astype(np.float32)

def soft_saturate(x, drive=0.08):
    drive = float(np.clip(drive, 0.0, 1.0))
    if drive <= 0:
        return x
    gain = 1.0 + drive * 2.0
    return np.tanh(x * gain) / np.tanh(gain)

def make_riaa_playback_sos(sr):
    sr = float(sr)
    numerator = np.array([318e-6, 1.0], dtype=np.float64)
    denominator = np.polymul(
        [3180e-6, 1.0], [75e-6, 1.0]
    )
    from scipy.signal import bilinear, tf2sos
    b, a = bilinear(numerator, denominator, fs=sr)
    return tf2sos(b, a)

def apply_riaa_playback(audio, sr):
    sos = make_riaa_playback_sos(sr)
    result = np.zeros_like(audio, dtype=np.float32)
    for ch in range(audio.shape[1]):
        result[:, ch] = sosfiltfilt(
            sos, audio[:, ch].astype(np.float64)
        ).astype(np.float32)
    return result

def dc_block(audio, sr):
    b, a = butter(2, 5.0, btype="highpass", fs=sr)
    result = np.zeros_like(audio, dtype=np.float32)
    for ch in range(audio.shape[1]):
        result[:, ch] = lfilter(b, a, audio[:, ch])
    return result

def fletcher_munson_eq(audio, sr, volume_norm):
    if volume_norm >= 0.95:
        return audio
    comp = 1.0 - volume_norm
    b_lf, a_lf = butter(2, 200.0, btype="lowpass", fs=sr)
    b_hf, a_hf = butter(1, 8000.0, btype="highpass", fs=sr)
    out = audio.copy()
    for ch in range(audio.shape[1]):
        lf = lfilter(b_lf, a_lf, audio[:, ch])
        hf = lfilter(b_hf, a_hf, audio[:, ch])
        out[:, ch] = (
            audio[:, ch] + lf * comp * 0.5 + hf * comp * 0.2
        )
    return out.astype(np.float32)

def gentle_lowpass(audio, sr, cutoff=18000):
    cutoff = min(cutoff, sr * 0.45)
    sos = butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)

# ============================================================
# MAIN TURNTABLE DSP ENGINE
# ============================================================
class TurntableDSP:
    def __init__(self, audio_data, sample_rate=44100):
        self.native_sr = int(sample_rate)
        self.sr = self.native_sr
        self.raw_audio = np.asarray(audio_data, dtype=np.float32)
        if self.raw_audio.ndim == 1:
            self.raw_audio = np.column_stack([self.raw_audio, self.raw_audio])
        if self.raw_audio.shape[1] == 1:
            self.raw_audio = np.column_stack([
                self.raw_audio[:, 0], self.raw_audio[:, 0]
            ])
        if self.raw_audio.shape[1] > 2:
            self.raw_audio = self.raw_audio[:, :2]
        self.raw_audio = np.nan_to_num(self.raw_audio)
        
        # Audio resampling / fidelity mode tracking
        self.fidelity_mode = 0  # 0: Native, 1: Audiophile (44.1k), 2: Lofi Shitty (22.05k), 3: Ultra Shitty (11.025k)
        self._resample_audio()

        self._pink_buf_len = max(
            1, min(self.total_samples, self.sr * 60)
        )
        self._pink_buf = pink_noise_voss(self._pink_buf_len)
        
        rms_window = max(1, int(self.sr * 0.1))
        rms_env = np.convolve(
            self.audio.mean(axis=1) ** 2,
            np.ones(rms_window, dtype=np.float32) / rms_window,
            mode="same"
        )
        self._rms_env = np.sqrt(
            np.clip(rms_env, 0, 1)
        ).astype(np.float32)
        rng = np.random.default_rng()
        
        crackle_count = max(1, int(self.total_samples * 0.00015))
        self.base_crackle_indices = np.sort(
            rng.choice(self.total_samples, size=crackle_count, replace=False)
        )
        self.base_crackle_values = (
            rng.choice([-1.0, 1.0, -0.5, 0.5], size=crackle_count)
            * rng.uniform(0.0005, 0.004, crackle_count)
        ).astype(np.float32)
        
        dust_count = max(1, int(self.total_samples * 0.00002))
        self.base_dust_indices = np.sort(
            rng.choice(self.total_samples, size=dust_count, replace=False)
        )
        
        self.target_speed_ratio = 1.0
        self.current_speed_ratio = 1.0
        self.inertia_mass = 0.995
        self.playhead = 0.0
        
        self.tonearm_lifted = True
        self.cue_transition = 0.0
        self.needle_drop_trigger = False
        
        self.b_res, self.a_res = butter(
            2, [8.0, 12.0], btype="bandpass", fs=self.sr
        )
        zi = np.zeros((max(len(self.a_res), len(self.b_res)) - 1, 2))
        self.res_zi = zi
        
        self.is_holding = False
        self.hand_velocity = 0.0
        self.smoothed_hand_vel = 0.0
        self.slider_scrub_mode = False
        self.slider_pressed = False
        self.slider_scrub_velocity = 0.0
        self.slider_seek_pct = 0.0
        
        self.flutter_state = 0.0
        self.eccentricity_phase = 0.0
        self.cogging_phase = 0.0
        self.hum_phase = 0.0
        self.block_counter = 0
        
        # Original RT85 defaults
        self.hiss_level = 0.00002
        self.air_sparkle = 0.08
        self.crosstalk_amt = 0.04
        self.hum_level = 0.0001
        self.crackle_density = 0.35
        self.dust_density = 0.25
        self.igd_amount = 0.12
        self.pinch_amount = 0.05
        self.tangential_error_amt = 0.05
        self.cantilever_compliance = 0.08
        self.saturation_drive = 0.08
        self.groove_vel_limit = 0.15
        self.cogging_level = 0.00003
        self.fm_compensation = 0.0
        self.antialiasing_on = True
        self.riaa_enabled = False
        self.vinyl_high_frequency_rolloff = 18000
        self.vinyl_warmth = 0.08
        self.output_gain = 0.85
        self.riaa_sos = make_riaa_playback_sos(self.sr)

        # R608 Options
        self.r608_distortion_mode = False
        self.r608_distortion_intensity = 0.5
        self.r608_comp_mode = False
        self.r608_comp_intensity = 0.5

        # Pre-allocated scratch buffers for memory optimization
        self._scratch_speeds = np.zeros(2048, dtype=np.float32)
        self._scratch_t = np.zeros(2048, dtype=np.float32)
        self._scratch_pops = np.zeros(2048, dtype=np.float32)
        self._scratch_dust = np.zeros(2048, dtype=np.float32)

    def set_fidelity_mode(self, mode_index):
        self.fidelity_mode = mode_index
        self._resample_audio()

    def _resample_audio(self):
        target_srs = {
            0: self.native_sr,
            1: 44100,
            2: 22050,
            3: 11025
        }
        new_sr = target_srs.get(self.fidelity_mode, self.native_sr)
        if new_sr != self.native_sr and new_sr > 0:
            num_samples = int(len(self.raw_audio) * new_sr / self.native_sr)
            self.audio = resample(self.raw_audio, num_samples, axis=0).astype(np.float32)
            self.sr = new_sr
        else:
            self.audio = self.raw_audio.copy()
            self.sr = self.native_sr
        self.total_samples = len(self.audio)

    def set_target_speed(self, ratio):
        self.target_speed_ratio = float(ratio)

    def process_block(self, block_size, volume_slider_val):
        self.block_counter += 1
        vol_norm = np.clip(volume_slider_val / 100.0, 0.0, 1.0)
        vol_curve = vol_norm ** 1.5
        target_cue = 0.0 if self.tonearm_lifted else 1.0
        self.cue_transition += (0.05 * (target_cue - self.cue_transition))
        
        if block_size != len(self._scratch_speeds):
            self._scratch_speeds = np.zeros(block_size, dtype=np.float32)
            self._scratch_t = np.zeros(block_size, dtype=np.float32)
            self._scratch_pops = np.zeros(block_size, dtype=np.float32)
            self._scratch_dust = np.zeros(block_size, dtype=np.float32)

        if self.slider_pressed and not self.slider_scrub_mode:
            self.playhead = self.slider_seek_pct * self.total_samples
            speeds = np.zeros(block_size, dtype=np.float32)
        elif self.is_holding:
            self.hand_velocity *= 0.90
            self.smoothed_hand_vel += (
                0.2 * (self.hand_velocity - self.smoothed_hand_vel)
            )
            speeds = np.full(block_size, self.smoothed_hand_vel, dtype=np.float32)
        elif self.slider_pressed and self.slider_scrub_mode:
            self.slider_scrub_velocity *= 0.85
            speeds = np.full(block_size, self.slider_scrub_velocity, dtype=np.float32)
        else:
            speed_diff = self.target_speed_ratio - self.current_speed_ratio
            self.current_speed_ratio += (speed_diff * (1.0 - self.inertia_mass))
            noise_step = np.random.normal(0, 0.000025, size=block_size)
            for i in range(block_size):
                self.flutter_state = (0.995 * self.flutter_state + noise_step[i])
            
            np.copyto(self._scratch_t, self.block_counter * block_size + np.arange(block_size, dtype=np.float32))
            
            self.eccentricity_phase += (2.0 * np.pi * 0.55 * block_size / self.sr)
            ecc = (0.00025 * np.sin(2.0 * np.pi * 0.55 * self._scratch_t / self.sr))
            self.cogging_phase += (2.0 * np.pi * 120.0 * block_size / self.sr)
            cogging = (
                self.cogging_level
                * np.sin(2.0 * np.pi * 120.0 * self._scratch_t / self.sr)
            )
            wow = (
                0.00015 * np.sin(2.0 * np.pi * 0.55 * self._scratch_t / self.sr)
                + self.flutter_state
                + ecc
                + cogging
            )
            speeds = (
                np.full(block_size, self.current_speed_ratio, dtype=np.float32)
                * (1.0 + wow)
            )
            
        playheads = self.playhead + np.cumsum(speeds)
        playheads %= self.total_samples
        self.playhead = float(playheads[-1])
        
        idx_i = (playheads.astype(np.int64) % self.total_samples)
        mu = (playheads - idx_i.astype(np.float32))
        ym1 = self.audio[(idx_i - 1) % self.total_samples]
        y0 = self.audio[idx_i]
        y1 = self.audio[(idx_i + 1) % self.total_samples]
        y2 = self.audio[(idx_i + 2) % self.total_samples]
        
        mu2 = mu[:, None] ** 2
        mu3 = mu[:, None] ** 3
        a0 = (-0.5 * ym1 + 1.5 * y0 - 1.5 * y1 + 0.5 * y2)
        a1 = (ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2)
        a2 = (-0.5 * ym1 + 0.5 * y1)
        a3 = y0
        out = (a0 * mu3 + a1 * mu2 + a2 * mu[:, None] + a3)
        
        if self.slider_pressed and not self.slider_scrub_mode:
            out *= 0.0
        else:
            out *= self.cue_transition
            
        if self.needle_drop_trigger:
            t_drop = np.arange(block_size, dtype=np.float32)
            thump = (
                np.sin(2.0 * np.pi * 35.0 * t_drop / self.sr)
                * np.exp(-t_drop / 40.0)
                * 0.04
            )
            out[:, 0] += thump
            out[:, 1] += thump * 0.85
            self.needle_drop_trigger = False
            
        res = np.zeros_like(out)
        for ch in range(2):
            res[:, ch], self.res_zi[:, ch] = lfilter(
                self.b_res, self.a_res, out[:, ch], zi=self.res_zi[:, ch]
            )
        out += res * 0.08
        
        if self.groove_vel_limit > 0:
            progress = (idx_i.astype(np.float32) / self.total_samples)
            max_level = (1.0 - progress * self.groove_vel_limit * 0.12)
            threshold = np.maximum(0.85, max_level)
            out = (np.tanh(out / threshold[:, None]) * threshold[:, None])
            
        progress = (idx_i.astype(np.float32) / self.total_samples)
        igd_curve = (np.maximum(0.0, progress - 0.72) / 0.28) ** 2
        igd_curve *= self.igd_amount
        if self.igd_amount > 0:
            hf = out - np.roll(out, 1, axis=0)
            out -= (hf * igd_curve[:, None] * 0.04)
            
        if self.pinch_amount > 0:
            velocity = np.abs(np.diff(out, axis=0, prepend=out[:1]))
            pinch = (velocity * self.pinch_amount * 0.3)
            out *= (1.0 - np.clip(pinch, 0, 0.05))
            
        twist = (progress[:, None] - 0.5) * self.tangential_error_amt
        out += (out ** 2 * twist * 0.02)
        
        transient = np.abs(out).mean(axis=1, keepdims=True)
        out *= (1.0 - transient * self.cantilever_compliance * 0.02)
        
        left = (out[:, 0] + out[:, 1] * self.crosstalk_amt)
        right = (out[:, 1] + out[:, 0] * self.crosstalk_amt)
        out[:, 0] = left
        out[:, 1] = right
        
        if self.target_speed_ratio > 0 or self.current_speed_ratio > 0.05:
            hum = (np.sin(2.0 * np.pi * 60.0 * self._scratch_t / self.sr) * self.hum_level)
            hum += (
                np.sin(2.0 * np.pi * 120.0 * self._scratch_t / self.sr)
                * self.hum_level
                * 0.15
            )
            out[:, 0] += hum
            out[:, 1] += hum * 0.8
            
        pink_start = (self.block_counter * block_size) % self._pink_buf_len
        indices = (pink_start + np.arange(block_size)) % self._pink_buf_len
        pink_slice = self._pink_buf[indices]
        rms_duck = (1.0 - np.clip(self._rms_env[idx_i] * 1.5, 0, 0.65))
        hiss = (pink_slice * self.hiss_level * rms_duck)
        out[:, 0] += hiss
        out[:, 1] += hiss * 0.9
        
        active_crack = int(
            len(self.base_crackle_indices) * np.clip(self.crackle_density, 0, 2)
        )
        crack_mask = np.isin(idx_i, self.base_crackle_indices[:active_crack])
        np.copyto(self._scratch_pops, 0.0)
        if np.any(crack_mask):
            locs = np.where(crack_mask)[0]
            vals = self.base_crackle_values[
                np.searchsorted(self.base_crackle_indices, idx_i[locs])
            ]
            self._scratch_pops[locs] = vals
            
        # R608 Distortion & Crackle
        if self.r608_distortion_mode and self.r608_distortion_intensity > 0.0:
            self._scratch_pops *= (1.0 + (self.r608_distortion_intensity * 2.5))
            res_amt = 0.6 * self.r608_distortion_intensity
            b_honk, a_honk = butter(2, [3000.0, 5500.0], btype="bandpass", fs=self.sr)
            honk = np.zeros_like(out)
            for ch in range(2):
                honk[:, ch] = lfilter(b_honk, a_honk, out[:, ch])
            out += honk * res_amt
            fizz_drive = 1.0 + (self.r608_distortion_intensity * 5.0)
            out = np.sign(out) * (1.0 - np.exp(-np.abs(out) * fizz_drive))

        out[:, 0] += self._scratch_pops
        out[:, 1] += self._scratch_pops * 0.9
        
        active_dust = int(
            len(self.base_dust_indices) * np.clip(self.dust_density, 0, 2)
        )
        np.copyto(self._scratch_dust, 0.0)
        for d_idx in self.base_dust_indices[:active_dust]:
            hit = ((idx_i >= d_idx) & (idx_i < d_idx + 40))
            if np.any(hit):
                locs = np.where(hit)[0]
                t_dust = (idx_i[locs] - d_idx).astype(np.float32)
                thump = (
                    np.sin(2.0 * np.pi * 60.0 * t_dust / self.sr)
                    * np.exp(-t_dust / 10.0)
                    * 0.015
                )
                self._scratch_dust[locs] += thump
        out[:, 0] += self._scratch_dust
        out[:, 1] += self._scratch_dust * 0.85
        
        if self.vinyl_high_frequency_rolloff < self.sr * 0.45:
            sos = butter(
                2,
                self.vinyl_high_frequency_rolloff,
                btype="lowpass",
                fs=self.sr,
                output="sos"
            )
            out = sosfilt(sos, out, axis=0)
            
        if self.vinyl_warmth > 0:
            b, a = butter(1, 250.0, btype="lowpass", fs=self.sr)
            low = lfilter(b, a, out, axis=0)
            out += (low * self.vinyl_warmth * 0.08)
            
        if self.air_sparkle > 0:
            transient = (out - np.roll(out, 1, axis=0))
            out += (transient * self.air_sparkle * 0.05)
            
        if self.saturation_drive > 0:
            out = soft_saturate(out, self.saturation_drive)
            
        if self.fm_compensation > 0:
            out = fletcher_munson_eq(
                out, self.sr, 1.0 - self.fm_compensation * (1.0 - vol_norm)
            )

        # R608 Compression & Bass Stripping
        if self.r608_comp_mode and self.r608_comp_intensity > 0.0:
            comp_gain = 1.0 + (self.r608_comp_intensity * 3.0)
            out = np.tanh(out * comp_gain) / np.tanh(comp_gain)
            cutoff_freq = 80.0 + (self.r608_comp_intensity * 320.0)
            if cutoff_freq < self.sr * 0.45:
                sos_hp = butter(2, cutoff_freq, btype="highpass", fs=self.sr, output="sos")
                out = sosfilt(sos_hp, out, axis=0)

        out *= (vol_curve * self.output_gain)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out, -1.0, 1.0).astype(np.float32)

# ============================================================
# EXPORT WORKER (WITH CANCELLATION SUPPORT)
# ============================================================
class ExportWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dsp, output_path, volume_val, bit_depth, parent=None):
        super().__init__(parent)
        self.dsp = dsp
        self.output_path = output_path
        self.volume_val = volume_val
        self.bit_depth = bit_depth

    def run(self):
        try:
            snap = copy.deepcopy(self.dsp)
            snap.playhead = 0.0
            snap.tonearm_lifted = False
            snap.cue_transition = 1.0
            snap.current_speed_ratio = snap.target_speed_ratio
            snap.block_counter = 0
            snap.is_holding = False
            snap.slider_pressed = False
            snap.slider_scrub_mode = False
            block_size = 4096
            total_blocks = math.ceil(snap.total_samples / block_size)
            output_chunks = []
            
            for i in range(total_blocks):
                if self.isInterruptionRequested():
                    return
                chunk = snap.process_block(block_size, self.volume_val)
                remaining = (snap.total_samples - i * block_size)
                if remaining < block_size:
                    chunk = chunk[:remaining]
                output_chunks.append(chunk)
                self.progress.emit(int((i + 1) / total_blocks * 100))
                
            if self.isInterruptionRequested():
                return
                
            rendered = np.concatenate(output_chunks, axis=0)
            rendered = np.nan_to_num(rendered)
            rendered = np.clip(rendered, -1.0, 1.0).astype(np.float32)
            sf.write(self.output_path, rendered, snap.sr, subtype=self.bit_depth)
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())

# ============================================================
# TURNTABLE VISUAL
# ============================================================
class VirtualTurntableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.angle_deg = 0.0
        self.is_dragging = False
        self.last_mouse_pos = QPointF()
        self.last_angle = 0.0
        self.scratch_callback = None

    def set_rotation(self, angle):
        self.angle_deg = angle % 360
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = event.position()
            self.last_angle = self._angle_from_center(self.last_mouse_pos)
            if self.scratch_callback:
                self.scratch_callback(True, 0.0)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            cur = event.position()
            ang = self._angle_from_center(cur)
            delta = ang - self.last_angle
            if delta > 180:
                delta -= 360
            if delta < -180:
                delta += 360
            if self.scratch_callback:
                self.scratch_callback(False, delta * 0.75)
            self.last_angle = ang

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            if self.scratch_callback:
                self.scratch_callback(False, None)

    def _angle_from_center(self, pos):
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        return math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0
        r = min(w, h) / 2.0 - 10.0
        
        p.setBrush(QBrush(QColor(25, 27, 32)))
        p.setPen(QPen(QColor(12, 14, 18), 3))
        p.drawRoundedRect(self.rect().adjusted(5, 5, -5, -5), 16, 16)
        
        p.save()
        p.translate(QPointF(cx, cy))
        
        plat_grad = QRadialGradient(0, 0, r)
        plat_grad.setColorAt(0.68, QColor(190, 220, 250, 55))
        plat_grad.setColorAt(1.0, QColor(8, 12, 18, 240))
        p.setBrush(QBrush(plat_grad))
        p.setPen(QPen(QColor(160, 195, 230, 210), 2))
        p.drawEllipse(QPointF(0, 0), r, r)
        
        p.rotate(self.angle_deg)
        vr = r - 10.0
        v_grad = QRadialGradient(0, 0, vr)
        v_grad.setColorAt(0.0, QColor(8, 8, 8))
        v_grad.setColorAt(0.72, QColor(35, 37, 42))
        v_grad.setColorAt(1.0, QColor(3, 3, 3))
        p.setBrush(QBrush(v_grad))
        p.setPen(QPen(QColor(40, 40, 40), 1))
        p.drawEllipse(QPointF(0, 0), vr, vr)
        
        p.setPen(QPen(QColor(25, 25, 25), 0.9))
        for ring_r in np.linspace(vr * 0.38, vr * 0.86, 24):
            p.drawEllipse(QPointF(0, 0), ring_r, ring_r)
            
        lr = vr * 0.32
        l_grad = QRadialGradient(0, 0, lr)
        l_grad.setColorAt(0.0, QColor(14, 70, 150))
        l_grad.setColorAt(1.0, QColor(6, 30, 75))
        p.setBrush(QBrush(l_grad))
        p.setPen(QPen(QColor(200, 200, 200), 1.5))
        p.drawEllipse(QPointF(0, 0), lr, lr)
        
        p.setBrush(QBrush(QColor(230, 180, 50)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(0, 0), 5.0, 5.0)
        p.restore()

# ============================================================
# MAIN WINDOW
# ============================================================
class TurntableWindow(QMainWindow):
    def __init__(self, dsp_engine):
        super().__init__()
        self.dsp = dsp_engine
        self.block_size = 2048
        self.stream = None
        self.last_slider_val = 0
        self.current_visual_angle = 0.0
        self._export_worker = None
        
        self.setWindowTitle("Fluance RT85 Ultimate - Vinyl Physics Emulator")
        self.resize(900, 1280)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121418;
                color: #f1f5f9;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4a5568;
            }
            QPushButton:pressed {
                background-color: #2d3748;
            }
            QLabel {
                font-size: 11px;
                color: #cbd5e1;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #334155;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                background: #38bdf8;
                border-radius: 7px;
                margin: -5px 0;
            }
            QSlider::sub-page:horizontal {
                background: #0284c7;
                border-radius: 2px;
            }
            QComboBox {
                background: #334155;
                color: #f8fafc;
                border: none;
                border-radius: 4px;
                padding: 3px 8px;
            }
        """)
        self._init_ui()
        self._init_audio()
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._gui_tick)
        self._timer.start(16)

    def _init_ui(self):
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)
        
        ttl = QLabel("FLUANCE RT85 - VINYL PHYSICS EMULATOR")
        ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ttl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ttl.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        lay.addWidget(ttl)
        
        self.platter = VirtualTurntableWidget()
        self.platter.scratch_callback = self._on_scratch
        lay.addWidget(self.platter, alignment=Qt.AlignmentFlag.AlignCenter)
        
        tl_lay = QVBoxLayout()
        tl_top = QHBoxLayout()
        tl_top.addWidget(QLabel("Track Position"))
        self.chk_scrub = QCheckBox("Scrub Mode")
        self.chk_scrub.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.chk_scrub.stateChanged.connect(
            lambda s: setattr(self.dsp, "slider_scrub_mode", bool(s))
        )
        tl_top.addWidget(self.chk_scrub)
        tl_lay.addLayout(tl_top)
        
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderPressed.connect(self._tl_pressed)
        self.timeline.sliderMoved.connect(self._tl_moved)
        self.timeline.sliderReleased.connect(self._tl_released)
        tl_lay.addWidget(self.timeline)
        lay.addLayout(tl_lay)
        
        frm = QFrame()
        frm.setStyleSheet("QFrame { background:#1a1d24; border-radius:12px; padding:8px; }")
        frm_lay = QVBoxLayout(frm)
        row1 = QHBoxLayout()
        
        spd_lay = QVBoxLayout()
        spd_lay.addWidget(QLabel("Speed", alignment=Qt.AlignmentFlag.AlignCenter))
        spd_btns = QHBoxLayout()
        self.btn_33 = QPushButton("33 1/3 RPM")
        self.btn_45 = QPushButton("45 RPM")
        self.btn_33.clicked.connect(lambda: self._set_speed(1.0, self.btn_33, self.btn_45))
        self.btn_45.clicked.connect(lambda: self._set_speed(1.35, self.btn_45, self.btn_33))
        spd_btns.addWidget(self.btn_33)
        spd_btns.addWidget(self.btn_45)
        spd_lay.addLayout(spd_btns)
        row1.addLayout(spd_lay)
        
        mtr_lay = QVBoxLayout()
        mtr_lay.addWidget(QLabel("Motor", alignment=Qt.AlignmentFlag.AlignCenter))
        self.btn_stop = QPushButton("STOP / BRAKE")
        self.btn_stop.clicked.connect(self._toggle_stop)
        mtr_lay.addWidget(self.btn_stop)
        row1.addLayout(mtr_lay)
        
        cue_lay = QVBoxLayout()
        cue_lay.addWidget(QLabel("Tonearm", alignment=Qt.AlignmentFlag.AlignCenter))
        self.btn_cue = QPushButton("LOWER NEEDLE")
        self.btn_cue.clicked.connect(self._toggle_cue)
        cue_lay.addWidget(self.btn_cue)
        row1.addLayout(cue_lay)
        
        vol_lay = QVBoxLayout()
        self.lbl_vol = QLabel("Volume: 80%")
        vol_lay.addWidget(self.lbl_vol)
        self.sld_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_vol.setRange(0, 100)
        self.sld_vol.setValue(80)
        self.sld_vol.valueChanged.connect(
            lambda v: self.lbl_vol.setText(f"Volume: {v}%")
        )
        vol_lay.addWidget(self.sld_vol)
        row1.addLayout(vol_lay)
        frm_lay.addLayout(row1)
        
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)
        self.physics_sliders = {}

        def mk_slider(label_txt, attr, lo, hi, default, fmt, row, col):
            lbl = QLabel(label_txt.format(default))
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(lo, hi)
            sld.setValue(default)
            def callback(v):
                setattr(self.dsp, attr, fmt(v))
                lbl.setText(label_txt.format(v))
            sld.valueChanged.connect(callback)
            grid.addWidget(lbl, row * 2, col)
            grid.addWidget(sld, row * 2 + 1, col)
            self.physics_sliders[attr] = sld

        mk_slider("Platter Inertia: {}‰", "inertia_mass", 980, 999, 995, lambda v: v / 1000.0, 0, 0)
        mk_slider("Pinch Effect: {}%", "pinch_amount", 0, 100, 5, lambda v: v / 100.0, 0, 1)
        mk_slider("Motor Hum: {}%", "hum_level", 0, 50, 1, lambda v: v / 10000.0, 1, 0)
        mk_slider("Cogging Vibration: {}%", "cogging_level", 0, 100, 3, lambda v: v / 100000.0, 1, 1)
        mk_slider("Air Sparkle: {}%", "air_sparkle", 0, 120, 8, lambda v: v / 100.0, 2, 0)
        mk_slider("Crackle Density: {}%", "crackle_density", 0, 200, 35, lambda v: v / 100.0, 2, 1)
        mk_slider("IGD Distortion: {}%", "igd_amount", 0, 100, 12, lambda v: v / 100.0, 3, 0)
        mk_slider("Dust Thumps: {}%", "dust_density", 0, 200, 25, lambda v: v / 100.0, 3, 1)
        mk_slider("Stereo Crosstalk: {}%", "crosstalk_amt", 0, 40, 4, lambda v: v / 100.0, 4, 0)
        mk_slider("Tangential Error: {}%", "tangential_error_amt", 0, 50, 5, lambda v: v / 100.0, 4, 1)
        mk_slider("Cantilever Compliance: {}%", "cantilever_compliance", 0, 50, 8, lambda v: v / 100.0, 5, 0)
        mk_slider("Analogue Saturation: {}%", "saturation_drive", 0, 100, 8, lambda v: v / 100.0, 5, 1)
        mk_slider("Groove Velocity Limit: {}%", "groove_vel_limit", 0, 100, 15, lambda v: v / 100.0, 6, 0)
        mk_slider("Fletcher-Munson EQ: {}%", "fm_compensation", 0, 100, 0, lambda v: v / 100.0, 6, 1)
        mk_slider("Vinyl Warmth: {}%", "vinyl_warmth", 0, 100, 8, lambda v: v / 100.0, 7, 0)
        mk_slider("Surface Hiss: {}%", "hiss_level", 0, 100, 2, lambda v: v / 100000.0, 7, 1)
        
        # Fidelity & Sample Rate Selector
        grid.addWidget(QLabel("Output Sample Rate / Player Fidelity Profile:"), 16, 0, 1, 2)
        self.cbo_fidelity = QComboBox()
        self.cbo_fidelity.addItems([
            "Native Source Sample Rate (Unchanged)",
            "Audiophile / Real RT85 Player Profile (44.1 kHz)",
            "Lofi Suitcase / Shittier Profile (22.05 kHz)",
            "Ultra Shitty / Vintage Toy Player Profile (11.025 kHz)"
        ])
        self.cbo_fidelity.currentIndexChanged.connect(self._on_fidelity_changed)
        grid.addWidget(self.cbo_fidelity, 17, 0, 1, 2)

        # R608 Distortion Checkbox & Slider
        self.chk_r608_dist = QCheckBox("R608: Ceramic Distortion & Crackle Grit")
        self.chk_r608_dist.setChecked(False)
        self.chk_r608_dist.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        self.chk_r608_dist.stateChanged.connect(
            lambda s: setattr(self.dsp, "r608_distortion_mode", bool(s))
        )
        grid.addWidget(self.chk_r608_dist, 18, 0, 1, 2)

        self.lbl_dist_int = QLabel("Distortion Grit Level: 50%")
        self.sld_dist_int = QSlider(Qt.Orientation.Horizontal)
        self.sld_dist_int.setRange(0, 100)
        self.sld_dist_int.setValue(50)
        self.sld_dist_int.valueChanged.connect(
            lambda v: (setattr(self.dsp, "r608_distortion_intensity", v / 100.0),
                       self.lbl_dist_int.setText(f"Distortion Grit Level: {v}%"))
        )
        grid.addWidget(self.lbl_dist_int, 19, 0)
        grid.addWidget(self.sld_dist_int, 20, 0, 1, 2)

        # R608 Compression & Bass Roll-off Checkbox & Slider
        self.chk_r608_comp = QCheckBox("R608: Suitcase Compression & Bass Roll-Off")
        self.chk_r608_comp.setChecked(False)
        self.chk_r608_comp.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        self.chk_r608_comp.stateChanged.connect(
            lambda s: setattr(self.dsp, "r608_comp_mode", bool(s))
        )
        grid.addWidget(self.chk_r608_comp, 21, 0, 1, 2)

        self.lbl_comp_int = QLabel("Compression & Bass Strip Level: 50%")
        self.sld_comp_int = QSlider(Qt.Orientation.Horizontal)
        self.sld_comp_int.setRange(0, 100)
        self.sld_comp_int.setValue(50)
        self.sld_comp_int.valueChanged.connect(
            lambda v: (setattr(self.dsp, "r608_comp_intensity", v / 100.0),
                       self.lbl_comp_int.setText(f"Compression & Bass Strip Level: {v}%"))
        )
        grid.addWidget(self.lbl_comp_int, 22, 0)
        grid.addWidget(self.sld_comp_int, 23, 0, 1, 2)

        riaa_label = QLabel(
            "RIAA: Disabled for ordinary digital files. "
            "Use only with pre-emphasized vinyl-cut audio."
        )
        riaa_label.setStyleSheet("color: #94a3b8;")
        grid.addWidget(riaa_label, 24, 0, 1, 2)
        frm_lay.addLayout(grid)
        
        # Audio Import & Export Toolbar
        io_frame = QFrame()
        io_frame.setStyleSheet("""
            QFrame {
                background: #0f1923;
                border: 1px solid #0284c7;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        io_lay = QHBoxLayout(io_frame)
        
        self.btn_load = QPushButton("LOAD NEW AUDIO")
        self.btn_load.setStyleSheet("background: #334155; color: white; padding: 6px 12px; border-radius: 6px;")
        self.btn_load.clicked.connect(self._load_new_audio)
        io_lay.addWidget(self.btn_load)

        io_lay.addWidget(QLabel("Format:"))
        self.cbo_format = QComboBox()
        self.cbo_format.addItems([
            "WAV (PCM 24-bit)", "WAV (PCM 16-bit)", "WAV (32-bit Float)"
        ])
        io_lay.addWidget(self.cbo_format)
        
        self.btn_export = QPushButton("EXPORT RENDERED VINYL")
        self.btn_export.setStyleSheet("""
            QPushButton {
                background: #0284c7;
                color: white;
                font-size: 11px;
                padding: 7px 14px;
                border-radius: 6px;
            }
        """)
        self.btn_export.clicked.connect(self._start_export)
        io_lay.addWidget(self.btn_export)
        frm_lay.addWidget(io_frame)
        
        lay.addWidget(frm)
        
        self.lbl_status = QLabel("Status: Tonearm Lifted")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #38bdf8; font-size: 11px;")
        lay.addWidget(self.lbl_status)
        
        self.setCentralWidget(root)

    def _init_audio(self):
        try:
            def callback(outdata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                outdata[:] = self.dsp.process_block(
                    frames, self.sld_vol.value()
                )
            self.stream = sd.OutputStream(
                samplerate=self.dsp.sr,
                channels=2,
                callback=callback,
                blocksize=self.block_size,
                dtype="float32"
            )
            self.stream.start()
        except Exception as e:
            QMessageBox.critical(
                self, "Audio Error", f"Cannot open audio stream:\n{e}"
            )
            raise

    def _on_fidelity_changed(self, index):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
        self.dsp.set_fidelity_mode(index)
        self._init_audio()
        self.timeline.setValue(0)
        self.lbl_status.setText(f"Status: Switched Sample Rate to {self.dsp.sr} Hz")

    def _load_new_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)"
        )
        if not path:
            return
        
        try:
            data, sr = sf.read(path, always_2d=True, dtype="float32")
        except Exception:
            try:
                from pydub import AudioSegment
                song = AudioSegment.from_file(path)
                sr = song.frame_rate
                raw = np.array(song.get_array_of_samples(), dtype=np.float32)
                max_value = float(2 ** (8 * song.sample_width - 1))
                raw /= max_value
                if song.channels == 2:
                    data = raw.reshape(-1, 2)
                else:
                    data = np.column_stack([raw, raw])
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Could not decode audio file:\n{e}")
                return

        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()

        self.dsp = TurntableDSP(data.astype(np.float32), sample_rate=sr)
        self.dsp.set_fidelity_mode(self.cbo_fidelity.currentIndex())
        self._init_audio()
        self.timeline.setValue(0)
        self.lbl_status.setText(f"Status: Loaded {os.path.basename(path)}")

    def _set_speed(self, ratio, active, inactive):
        self.dsp.set_target_speed(ratio)
        active.setStyleSheet("background:#0284c7;color:white;")
        inactive.setStyleSheet("")
        label = ("33 1/3" if ratio == 1.0 else "45")
        self.lbl_status.setText(f"Status: Playing at {label} RPM")

    def _toggle_stop(self):
        if self.dsp.target_speed_ratio > 0:
            self.dsp.set_target_speed(0)
            self.btn_stop.setStyleSheet("background:#991b1b;color:white;")
            self.lbl_status.setText("Status: Motor Braking")
        else:
            self.dsp.set_target_speed(1.0)
            self.btn_stop.setStyleSheet("")
            self.btn_33.setStyleSheet("background:#0284c7;color:white;")
            self.btn_45.setStyleSheet("")
            self.lbl_status.setText("Status: Motor Restarting")

    def _toggle_cue(self):
        self.dsp.tonearm_lifted = not self.dsp.tonearm_lifted
        if self.dsp.tonearm_lifted:
            self.btn_cue.setText("LOWER NEEDLE")
            self.btn_cue.setStyleSheet("")
            self.lbl_status.setText("Status: Tonearm Lifted")
        else:
            self.btn_cue.setText("LIFT NEEDLE")
            self.btn_cue.setStyleSheet("background:#0284c7;color:white;")
            self.dsp.needle_drop_trigger = True
            self.lbl_status.setText("Status: Needle Down")

    def _tl_pressed(self):
        self.dsp.slider_pressed = True
        self.last_slider_val = self.timeline.value()

    def _tl_moved(self, val):
        delta = val - self.last_slider_val
        self.last_slider_val = val
        if self.dsp.slider_scrub_mode:
            self.dsp.slider_scrub_velocity = float(
                np.clip(delta * 0.035, -2.5, 2.5)
            )
        else:
            self.dsp.slider_seek_pct = val / 1000.0

    def _tl_released(self):
        if not self.dsp.slider_scrub_mode:
            self.dsp.playhead = self.dsp.slider_seek_pct * self.dsp.total_samples
        self.dsp.slider_pressed = False
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
                self.dsp.hand_velocity = float(
                    np.clip(vel, -15.0, 15.0)
                )

    def _gui_tick(self):
        if (
            self.dsp.is_holding
            or (self.dsp.slider_pressed and self.dsp.slider_scrub_mode)
        ):
            self.current_visual_angle += (self.dsp.hand_velocity * 4.0)
        else:
            self.current_visual_angle += (self.dsp.current_speed_ratio * 3.33)
        self.platter.set_rotation(self.current_visual_angle)
        
        if not self.timeline.isSliderDown():
            pct = self.dsp.playhead / self.dsp.total_samples
            val = int(pct * 1000)
            self.last_slider_val = val
            self.timeline.setValue(val)

    def _start_export(self):
        fmt_map = {
            0: ("WAV PCM 24-bit", "PCM_24", ".wav"),
            1: ("WAV PCM 16-bit", "PCM_16", ".wav"),
            2: ("WAV 32-bit Float", "FLOAT", ".wav")
        }
        fmt_label, subtype, ext = fmt_map[self.cbo_format.currentIndex()]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Rendered Vinyl",
            f"vinyl_export{ext}",
            f"Audio Files (*{ext});;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        if os.path.exists(path):
            reply = QMessageBox.question(
                self,
                "File Exists",
                f"{os.path.basename(path)} already exists.\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
                
        self.btn_export.setEnabled(False)
        self._progress_dlg = QProgressDialog(
            f"Rendering vinyl DSP...\n{path}", "Cancel", 0, 100, self
        )
        self._progress_dlg.setWindowTitle("Exporting")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setValue(0)
        
        self._export_worker = ExportWorker(
            self.dsp, path, self.sld_vol.value(), subtype
        )
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
        size_mb = os.path.getsize(path) / 1_048_576
        QMessageBox.information(
            self,
            "Export Complete",
            f"Vinyl render saved successfully!\n\n"
            f"File: {path}\n"
            f"Size: {size_mb:.1f} MB"
        )

    def _export_error(self, msg):
        self._progress_dlg.close()
        self.btn_export.setEnabled(True)
        QMessageBox.critical(
            self, "Export Failed", f"Export error:\n\n{msg}"
        )

    def closeEvent(self, event):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.requestInterruption()
            self._export_worker.wait()
        event.accept()

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    initial_path, _ = QFileDialog.getOpenFileName(
        None, "Select Audio Track to Play", "", "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)"
    )
    
    data = None
    sample_rate = 44100
    
    if initial_path:
        try:
            data, sample_rate = sf.read(initial_path, always_2d=True, dtype="float32")
            print(f"Loaded {initial_path} at {sample_rate} Hz.")
        except Exception:
            try:
                from pydub import AudioSegment
                song = AudioSegment.from_file(initial_path)
                sample_rate = song.frame_rate
                raw = np.array(song.get_array_of_samples(), dtype=np.float32)
                max_value = float(2 ** (8 * song.sample_width - 1))
                raw /= max_value
                if song.channels == 2:
                    data = raw.reshape(-1, 2)
                else:
                    data = np.column_stack([raw, raw])
            except Exception as e:
                QMessageBox.critical(None, "Startup Error", f"Could not load audio file:\n{e}")
                sys.exit(1)
    else:
        t = np.linspace(0, 30, int(44100 * 30), endpoint=False)
        tone = (
            0.30 * np.sin(2 * np.pi * 220 * t)
            + 0.15 * np.sin(2 * np.pi * 440 * t)
            + 0.07 * np.sin(2 * np.pi * 880 * t)
        )
        data = np.column_stack([tone, tone * 0.9]).astype(np.float32)

    audio_data = data.astype(np.float32)
    dsp = TurntableDSP(audio_data, sample_rate=sample_rate)
    win = TurntableWindow(dsp)
    win.show()
    sys.exit(app.exec())