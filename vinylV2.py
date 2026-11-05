import sys
import math
import numpy as np
from scipy.signal import lfilter, butter
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSlider, QFrame, QMessageBox, QGridLayout, QCheckBox
)
import sounddevice as sd
import soundfile as sf

# ==========================================
# MASTER RIAA & SUBSONIC RUMBLE FILTER
# ==========================================
def apply_riaa_filter(audio, sr):
    nyq = sr / 2.0
    b_sub, a_sub = butter(2, 28.0 / nyq, btype='high')
    
    processed = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        sig = audio[:, ch]
        processed[:, ch] = lfilter(b_sub, a_sub, sig)
        
    return processed.astype(np.float32)


# ==========================================
# VECTORIZED RT85 DSP ENGINE + THUMP & IGD
# ==========================================

class TurntableDSP:
    def __init__(self, audio_data, sample_rate=44100):
        self.sr = sample_rate
        self.audio = audio_data.astype(np.float32)
        if self.audio.ndim == 1:
            self.audio = np.stack([self.audio, self.audio], axis=-1)
        self.total_samples = len(self.audio)
        
        # Pre-calculate crackle buffer
        self.crackle_buffer = np.zeros(self.total_samples, dtype=np.float32)
        pop_indices = np.random.choice(self.total_samples, size=int(self.total_samples * 0.00015))
        self.crackle_buffer[pop_indices] = np.random.choice([-1.0, 1.0, 0.5, -0.5], size=len(pop_indices)) * np.random.uniform(0.003, 0.015, size=len(pop_indices))

        # Platter & Motor State
        self.target_speed_ratio = 1.0 
        self.current_speed_ratio = 1.0
        self.inertia_mass = 0.995 
        self.playhead = 0.0
        
        # Tonearm & Cueing State
        self.tonearm_lifted = False
        self.cue_transition = 1.0 
        self.needle_drop_trigger = False
        
        # Slip-Mat & Hand-Brake State
        self.is_holding = False
        self.hand_velocity = 0.0
        self.smoothed_hand_vel = 0.0
        
        # Slider Seek & Scrub State
        self.slider_scrub_mode = False
        self.slider_pressed = False
        self.slider_scrub_velocity = 0.0
        self.slider_seek_pct = 0.0
        
        # Adjustable Audiophile Sliders
        self.wf_depth = 1.0         
        self.hiss_level = 0.00002   
        self.air_sparkle = 0.4      
        self.crosstalk_amt = 0.08   
        self.hum_level = 0.001      
        self.crackle_gain = 1.0     
        self.igd_amount = 0.35      
        
        # LFO phases
        self.wf_phase_055 = 0.0
        self.wf_phase_31 = 0.0
        self.hum_phase = 0.0
        self.block_counter = 0

    def set_target_speed(self, ratio):
        self.target_speed_ratio = ratio

    def process_block(self, block_size, volume_slider_val):
        self.block_counter += 1
        vol_norm = np.clip(volume_slider_val / 100.0, 0.01, 1.0)
        vol_curve = vol_norm ** 1.5

        target_cue = 0.0 if self.tonearm_lifted else 1.0
        self.cue_transition += 0.05 * (target_cue - self.cue_transition)

        # 1. PLAYBACK SPEEDS (VECTORIZED)
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
            
            t_indices = np.arange(block_size, dtype=np.float32) + (self.block_counter * block_size)
            self.wf_phase_055 += (2.0 * np.pi * 0.55 * block_size) / self.sr
            self.wf_phase_31 += (2.0 * np.pi * 3.1 * block_size) / self.sr
            
            wf_mod = (0.0003 * np.sin(2.0 * np.pi * 0.55 * t_indices / self.sr + self.wf_phase_055) + 
                      0.0002 * np.sin(2.0 * np.pi * 3.1 * t_indices / self.sr + self.wf_phase_31)) * self.wf_depth
            speeds = np.full(block_size, self.current_speed_ratio, dtype=np.float32) * (1.0 + wf_mod)

        # 2. PLAYHEAD TRAJECTORY
        playheads = self.playhead + np.cumsum(speeds)
        playheads %= self.total_samples
        self.playhead = playheads[-1]

        # 3. VECTORIZED FRACTIONAL INTERPOLATION
        idx_ints = playheads.astype(np.int64) % self.total_samples
        idx_fracs = playheads - idx_ints.astype(np.float32)
        idx_nexts = (idx_ints + 1) % self.total_samples

        s0 = self.audio[idx_ints]
        s1 = self.audio[idx_nexts]
        out_block = s0 * (1.0 - idx_fracs[:, np.newaxis]) + s1 * idx_fracs[:, np.newaxis]

        if self.slider_pressed and not self.slider_scrub_mode:
            out_block *= 0.0
        else:
            out_block *= self.cue_transition

        # --- NEEDLE DROP THUMP ---
        if self.needle_drop_trigger:
            thump_indices = np.arange(block_size, dtype=np.float32)
            thump_envelope = np.exp(-thump_indices / 40.0)
            thump_wave = np.sin(2.0 * np.pi * 35.0 * thump_indices / self.sr) * thump_envelope * 0.4
            out_block[:, 0] += thump_wave
            out_block[:, 1] += thump_wave * 0.85
            self.needle_drop_trigger = False

        # --- INNER-GROOVE DISTORTION (IGD) ---
        progress = (idx_ints.astype(np.float32) / self.total_samples)
        igd_curve = (np.maximum(0.0, progress - 0.7) / 0.3) ** 2.0 * self.igd_amount
        out_block += (out_block ** 3) * igd_curve[:, np.newaxis] * 0.25

        # --- CARTRIDGE CROSSTALK ---
        l_chan = out_block[:, 0] + out_block[:, 1] * self.crosstalk_amt
        r_chan = out_block[:, 1] + out_block[:, 0] * self.crosstalk_amt
        out_block[:, 0] = l_chan
        out_block[:, 1] = r_chan

        # --- MOTOR HUM & NOISE ---
        if self.target_speed_ratio > 0.0 or self.current_speed_ratio > 0.05:
            hum_indices = np.arange(block_size, dtype=np.float32) + (self.block_counter * block_size)
            self.hum_phase = (2.0 * np.pi * 60.0 * hum_indices[-1]) / self.sr
            hum_sig = np.sin(2.0 * np.pi * 60.0 * hum_indices / self.sr + self.hum_phase) * self.hum_level
            out_block[:, 0] += hum_sig
            out_block[:, 1] += hum_sig * 0.8

        white_hiss = np.random.normal(0, self.hiss_level, size=(block_size, 2))
        pops = self.crackle_buffer[idx_ints, np.newaxis] * self.crackle_gain

        out_block += white_hiss + pops
        out_block *= vol_curve

        # --- ORTOFON 2M BLUE HIGH-END AIR & SPARKLE ---
        if self.air_sparkle > 0.0:
            high_freqs = out_block - np.roll(out_block, 1, axis=0)
            out_block += high_freqs * self.air_sparkle

        out_block = np.nan_to_num(out_block, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out_block, -1.0, 1.0).astype(np.float32)


# ==========================================
# TURNTABLE GUI & WINDOW SUITE
# ==========================================

class VirtualTurntableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
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
            self.last_angle = self.get_angle_from_center(self.last_mouse_pos)
            if self.scratch_callback:
                self.scratch_callback(True, 0.0)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            current_pos = event.position()
            current_angle = self.get_angle_from_center(current_pos)
            delta_angle = current_angle - self.last_angle
            
            if delta_angle > 180: delta_angle -= 360
            elif delta_angle < -180: delta_angle += 360
                
            if self.scratch_callback:
                scratch_vel = delta_angle * 0.75
                self.scratch_callback(False, scratch_vel)
                
            self.last_angle = current_angle

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            if self.scratch_callback:
                self.scratch_callback(False, None)

    def get_angle_from_center(self, pos):
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        return math.degrees(math.atan2(pos.y() - center.y(), pos.x() - center.x()))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        center = QPointF(w / 2.0, h / 2.0)
        radius = min(w, h) / 2.0 - 15.0

        painter.setBrush(QBrush(QColor(25, 27, 32)))
        painter.setPen(QPen(QColor(12, 14, 18), 3))
        painter.drawRoundedRect(self.rect().adjusted(5, 5, -5, -5), 16, 16)

        painter.save()
        painter.translate(center)
        
        platter_grad = QRadialGradient(0, 0, radius)
        platter_grad.setColorAt(0.7, QColor(190, 220, 250, 50))
        platter_grad.setColorAt(1.0, QColor(8, 12, 18, 230))
        painter.setBrush(QBrush(platter_grad))
        painter.setPen(QPen(QColor(160, 195, 230, 200), 2))
        painter.drawEllipse(QPointF(0, 0), radius, radius)

        painter.rotate(self.angle_deg)
        vinyl_radius = radius - 12.0
        vinyl_brush = QRadialGradient(0, 0, vinyl_radius)
        vinyl_brush.setColorAt(0.0, QColor(8, 8, 8))
        vinyl_brush.setColorAt(0.72, QColor(35, 37, 42))
        vinyl_brush.setColorAt(1.0, QColor(3, 3, 3))
        painter.setBrush(QBrush(vinyl_brush))
        painter.setPen(QPen(QColor(40, 40, 40), 1))
        painter.drawEllipse(QPointF(0, 0), vinyl_radius, vinyl_radius)

        painter.setPen(QPen(QColor(25, 25, 25), 0.8))
        for r in np.linspace(vinyl_radius * 0.38, vinyl_radius * 0.86, 22):
            painter.drawEllipse(QPointF(0, 0), r, r)

        label_radius = vinyl_radius * 0.32
        label_grad = QRadialGradient(0, 0, label_radius)
        label_grad.setColorAt(0.0, QColor(14, 70, 150))
        label_grad.setColorAt(1.0, QColor(6, 30, 75))
        painter.setBrush(QBrush(label_grad))
        painter.setPen(QPen(QColor(200, 200, 200), 1.5))
        painter.drawEllipse(QPointF(0, 0), label_radius, label_radius)

        painter.setBrush(QBrush(QColor(230, 180, 50)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), 6.0, 6.0)
        painter.restore()


class TurntableWindow(QMainWindow):
    def __init__(self, dsp_engine):
        super().__init__()
        self.dsp = dsp_engine
        self.sr = dsp_engine.sr
        self.block_size = 2048
        
        self.setWindowTitle("Fluance RT85 Reference - Final Perfection Suite")
        self.setStyleSheet("background-color: #121418; color: #f1f5f9; font-family: 'Segoe UI', Arial;")
        self.resize(820, 880)
        
        self.current_visual_angle = 0.0
        self.stream = None
        self.last_slider_val = 0
        
        self.init_ui()
        self.init_audio_stream()
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_gui_loop)
        self.anim_timer.start(16)

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)

        title_label = QLabel("FLUANCE RT85 REFERENCE - FINAL PERFECTION SUITE")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #38bdf8; letter-spacing: 1.5px;")
        main_layout.addWidget(title_label)

        self.turntable_widget = VirtualTurntableWidget()
        self.turntable_widget.scratch_callback = self.handle_scratch_event
        main_layout.addWidget(self.turntable_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        timeline_layout = QVBoxLayout()
        timeline_top = QHBoxLayout()
        self.timeline_label = QLabel("Track Timeline Position")
        timeline_top.addWidget(self.timeline_label)
        
        self.chk_scrub_mode = QCheckBox("Slider Scrubbing / Rewind Mode")
        self.chk_scrub_mode.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.chk_scrub_mode.stateChanged.connect(lambda state: setattr(self.dsp, 'slider_scrub_mode', bool(state)))
        timeline_top.addWidget(self.chk_scrub_mode, alignment=Qt.AlignmentFlag.AlignRight)
        timeline_layout.addLayout(timeline_top)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderPressed.connect(self.on_slider_pressed)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        self.timeline_slider.sliderReleased.connect(self.on_slider_released)
        timeline_layout.addWidget(self.timeline_slider)
        main_layout.addLayout(timeline_layout)

        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame { background-color: #1a1d24; border-radius: 12px; padding: 12px; }
            QPushButton { background-color: #334155; color: #f8fafc; border: none; border-radius: 6px; padding: 8px 14px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background-color: #434c5e; }
            QPushButton#active_btn { background-color: #0284c7; color: white; }
            QLabel { font-size: 12px; color: #cbd5e1; }
        """)
        
        ctrl_layout = QVBoxLayout(control_frame)
        top_row = QHBoxLayout()
        
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Speed"), alignment=Qt.AlignmentFlag.AlignCenter)
        speed_btns = QHBoxLayout()
        self.btn_33 = QPushButton("33 ⅓")
        self.btn_33.setObjectName("active_btn")
        self.btn_33.clicked.connect(lambda: self.set_speed(1.0, self.btn_33, self.btn_45))
        self.btn_45 = QPushButton("45")
        self.btn_45.clicked.connect(lambda: self.set_speed(1.35, self.btn_45, self.btn_33))
        speed_btns.addWidget(self.btn_33)
        speed_btns.addWidget(self.btn_45)
        speed_layout.addLayout(speed_btns)
        top_row.addLayout(speed_layout)

        brake_layout = QVBoxLayout()
        brake_layout.addWidget(QLabel("Motor Control"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_stop = QPushButton("MOTOR STOP / BRAKE")
        self.btn_stop.clicked.connect(self.toggle_stop)
        brake_layout.addWidget(self.btn_stop)
        top_row.addLayout(brake_layout)

        cue_layout = QVBoxLayout()
        cue_layout.addWidget(QLabel("Tonearm Cue"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_cue = QPushButton("LOWER NEEDLE")
        self.btn_cue.clicked.connect(self.toggle_tonearm)
        cue_layout.addWidget(self.btn_cue)
        top_row.addLayout(cue_layout)

        vol_layout = QVBoxLayout()
        self.vol_label = QLabel("Master Volume: 80%")
        vol_layout.addWidget(self.vol_label)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.valueChanged.connect(lambda v: self.vol_label.setText(f"Master Volume: {v}%"))
        vol_layout.addWidget(self.vol_slider)
        top_row.addLayout(vol_layout)

        ctrl_layout.addLayout(top_row)

        slider_grid = QGridLayout()
        self.lbl_inertia = QLabel("Acrylic Platter Mass: High")
        self.sld_inertia = QSlider(Qt.Orientation.Horizontal)
        self.sld_inertia.setRange(980, 999)
        self.sld_inertia.setValue(995)
        self.sld_inertia.valueChanged.connect(lambda v: setattr(self.dsp, 'inertia_mass', v/1000.0) or self.lbl_inertia.setText(f"Acrylic Platter Mass: {v/1000.0:.3f}"))
        slider_grid.addWidget(self.lbl_inertia, 0, 0)
        slider_grid.addWidget(self.sld_inertia, 1, 0)

        self.lbl_crosstalk = QLabel("Cartridge Crosstalk: 8%")
        self.sld_crosstalk = QSlider(Qt.Orientation.Horizontal)
        self.sld_crosstalk.setRange(0, 20)
        self.sld_crosstalk.setValue(8)
        self.sld_crosstalk.valueChanged.connect(lambda v: setattr(self.dsp, 'crosstalk_amt', v/100.0) or self.lbl_crosstalk.setText(f"Cartridge Crosstalk: {v}%"))
        slider_grid.addWidget(self.lbl_crosstalk, 0, 1)
        slider_grid.addWidget(self.sld_crosstalk, 1, 1)

        self.lbl_hum = QLabel("Motor Hum & Rumble: Low")
        self.sld_hum = QSlider(Qt.Orientation.Horizontal)
        self.sld_hum.setRange(0, 50)
        self.sld_hum.setValue(10)
        self.sld_hum.valueChanged.connect(lambda v: setattr(self.dsp, 'hum_level', (v/100.0)*0.001) or self.lbl_hum.setText(f"Motor Hum & Rumble: {v}%"))
        slider_grid.addWidget(self.lbl_hum, 2, 0)
        slider_grid.addWidget(self.sld_hum, 3, 0)

        self.lbl_air = QLabel("Ortofon 2M Air Sparkle: Active")
        self.sld_air = QSlider(Qt.Orientation.Horizontal)
        self.sld_air.setRange(0, 120)
        self.sld_air.setValue(40)
        self.sld_air.valueChanged.connect(lambda v: setattr(self.dsp, 'air_sparkle', v/100.0) or self.lbl_air.setText(f"Ortofon 2M Air Sparkle: {v/100.0:.2f}x"))
        slider_grid.addWidget(self.lbl_air, 2, 1)
        slider_grid.addWidget(self.sld_air, 3, 1)

        self.lbl_crackle = QLabel("Thrift Crackle Density: Moderate")
        self.sld_crackle = QSlider(Qt.Orientation.Horizontal)
        self.sld_crackle.setRange(0, 100)
        self.sld_crackle.setValue(15)
        self.sld_crackle.valueChanged.connect(lambda v: setattr(self.dsp, 'crackle_gain', v/15.0) or self.lbl_crackle.setText(f"Thrift Crackle Density: {v}%"))
        slider_grid.addWidget(self.lbl_crackle, 4, 0)
        slider_grid.addWidget(self.sld_crackle, 5, 0)

        self.lbl_igd = QLabel("Inner-Groove Distortion (IGD): Moderate")
        self.sld_igd = QSlider(Qt.Orientation.Horizontal)
        self.sld_igd.setRange(0, 100)
        self.sld_igd.setValue(35)
        self.sld_igd.valueChanged.connect(lambda v: setattr(self.dsp, 'igd_amount', v/100.0) or self.lbl_igd.setText(f"Inner-Groove Distortion (IGD): {v}%"))
        slider_grid.addWidget(self.lbl_igd, 4, 1)
        slider_grid.addWidget(self.sld_igd, 5, 1)

        ctrl_layout.addLayout(slider_grid)
        main_layout.addWidget(control_frame)

        self.status_label = QLabel("Status: Final Perfection Suite Active | Tonearm Lifted (Ready)")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #38bdf8; font-size: 11px;")
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(central_widget)

    def init_audio_stream(self):
        try:
            def audio_callback(outdata, frames, time_info, status):
                if status: print(status, file=sys.stderr)
                outdata[:] = self.dsp.process_block(frames, self.vol_slider.value())

            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=2,
                callback=audio_callback,
                blocksize=self.block_size
            )
            self.stream.start()
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Audio stream failed:\n{str(e)}")
            sys.exit(1)

    def set_speed(self, ratio, active_btn, inactive_btn):
        self.btn_stop.setStyleSheet("")
        active_btn.setObjectName("active_btn")
        active_btn.setStyleSheet("")
        inactive_btn.setObjectName("")
        inactive_btn.setStyleSheet("")
        self.dsp.set_target_speed(ratio)
        self.status_label.setText(f"Status: Playing at {'33 1/3' if ratio==1.0 else '45'} RPM")

    def toggle_stop(self):
        if self.dsp.target_speed_ratio > 0.0:
            self.dsp.set_target_speed(0.0)
            self.btn_stop.setStyleSheet("background-color: #991b1b; color: white;")
            self.status_label.setText("Status: Motor Brake Engaged (Acrylic Platter Coasting)...")
        else:
            self.dsp.set_target_speed(1.0)
            self.btn_stop.setStyleSheet("")
            self.btn_33.setObjectName("active_btn")
            self.btn_33.setStyleSheet("")
            self.btn_45.setObjectName("")
            self.btn_45.setStyleSheet("")
            self.status_label.setText("Status: Motor Resumed...")

    def toggle_tonearm(self):
        self.dsp.tonearm_lifted = not self.dsp.tonearm_lifted
        if self.dsp.tonearm_lifted:
            self.btn_cue.setText("LOWER NEEDLE")
            self.btn_cue.setStyleSheet("")
            self.status_label.setText("Status: Tonearm Lifted (Hydraulic arm up)")
        else:
            self.btn_cue.setText("LIFT NEEDLE")
            self.btn_cue.setStyleSheet("background-color: #0284c7; color: white;")
            self.dsp.needle_drop_trigger = True
            self.status_label.setText("Status: Needle Dropped in Groove (Thump & Playback Active)")

    def on_slider_pressed(self):
        self.dsp.slider_pressed = True
        self.last_slider_val = self.timeline_slider.value()

    def on_slider_moved(self, val):
        delta = val - self.last_slider_val
        self.last_slider_val = val
        
        if self.dsp.slider_scrub_mode:
            self.dsp.slider_scrub_velocity = np.clip(delta * 0.035, -2.5, 2.5)
        else:
            self.dsp.slider_seek_pct = val / 1000.0

    def on_slider_released(self):
        if not self.dsp.slider_scrub_mode:
            self.dsp.playhead = self.dsp.slider_seek_pct * self.dsp.total_samples
        self.dsp.slider_pressed = False
        self.dsp.slider_scrub_velocity = 0.0

    def handle_scratch_event(self, start, scratch_vel):
        if not self.dsp.slider_scrub_mode:
            if start:
                self.dsp.is_holding = True
                self.dsp.hand_velocity = 0.0
                self.dsp.smoothed_hand_vel = 0.0
            elif scratch_vel is None:
                self.dsp.is_holding = False
            else:
                clamped_vel = np.clip(scratch_vel, -15.0, 15.0)
                self.dsp.hand_velocity = clamped_vel

    def update_gui_loop(self):
        if self.dsp.is_holding or (self.dsp.slider_pressed and self.dsp.slider_scrub_mode):
            self.current_visual_angle += self.dsp.hand_velocity * 4.0
        else:
            deg_step = self.dsp.current_speed_ratio * 3.33
            self.current_visual_angle += deg_step
        self.turntable_widget.set_rotation(self.current_visual_angle)

        if not self.timeline_slider.isSliderDown():
            pct = self.dsp.playhead / self.dsp.total_samples
            val = int(pct * 1000)
            self.last_slider_val = val
            self.timeline_slider.setValue(val)

    def closeEvent(self, event):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
        event.accept()


# ==========================================
# SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    audio_path = r"C:\Users\YOUR-MUSIC-FILE-HERE" 
    
    try:
        data, sample_rate = sf.read(audio_path, always_2d=True)
        print(f"Loaded '{audio_path}' at native sample rate: {sample_rate} Hz.")
        audio_data = apply_riaa_filter(data.astype(np.float32), sample_rate)
    except Exception as e:
        print(f"Could not load '{audio_path}' ({e}). Generating synthetic studio test tone...")
        sample_rate = 44100
        t = np.linspace(0, 30, int(sample_rate * 30), endpoint=False)
        synth = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)
        synth_stereo = np.stack([synth, synth * 0.9], axis=-1).astype(np.float32)
        audio_data = apply_riaa_filter(synth_stereo, sample_rate)

    dsp_engine = TurntableDSP(audio_data, sample_rate=sample_rate)
    window = TurntableWindow(dsp_engine)
    window.show()
    
    sys.exit(app.exec())