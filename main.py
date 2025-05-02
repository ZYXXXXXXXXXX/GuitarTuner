import tkinter as tk
import numpy as np
import scipy.fftpack
import sounddevice as sd
import threading, math, copy
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SAMPLE_FREQ = 48000  # sampling frequency
WINDOW_SIZE = 48000  # window size DFT
WINDOW_STEP = 12000  # step size
NUM_HPS = 5  # number of HPS
POWER_THRESH = 1e-6  # minimum signal power
CONCERT_PITCH = 440  # reference pitch A4
WHITE_NOISE_THRESH = 0.2  # threshold for noise suppression

HANN_WINDOW = np.hanning(WINDOW_SIZE)
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE
# octave bands for noise suppression:
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

# notes
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def find_closest_note(pitch):
    # find the closest musical note
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    return closest_note, closest_pitch


class GuitarTunerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Guitar Tuner")
        self.targets = {"E2": 82.41, "A2": 110.00, "D3": 146.83,
                        "G3": 196.00, "B3": 246.94, "E4": 329.63}
        self.target_string = "E2"
        self.target_frequency = self.targets[self.target_string]

        self.detected_frequency = None
        self.detected_note = ""
        # buffers
        self.window_samples = np.zeros(WINDOW_SIZE)
        self.magnitude_spec = np.zeros(WINDOW_SIZE // 2)
        self.noteBuffer = ["", ""]
        self.lock = threading.Lock()

        # GUI
        self.create_widgets()
        # audio stream
        self.audio_stream = sd.InputStream(channels=1, callback=self.audio_callback,
                                           blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ)
        self.audio_stream.start()
        self.update_gui()

    def create_widgets(self):
        # buttons
        self.left_frame = tk.Frame(self.master)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        self.right_frame = tk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.pack(side=tk.TOP, pady=10)
        self.string_buttons = []
        for string in ["E2", "A2", "D3", "G3", "B3", "E4"]:
            btn = tk.Button(self.button_frame, text=string, command=lambda s=string: self.set_target(s))
            btn.pack(side=tk.LEFT, padx=5)
            self.string_buttons.append(btn)
        # Auto Detect
        self.auto_var = tk.BooleanVar(value=False)
        self.auto_check = tk.Checkbutton(self.button_frame, text="Auto Detect", variable=self.auto_var,
                                         command=self.toggle_auto, indicatoron=False, onvalue=True, offvalue=False)
        self.auto_check.pack(side=tk.LEFT, padx=5)

        # Dial
        self.canvas_size = 300
        self.canvas = tk.Canvas(self.left_frame, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.dial_center = self.canvas_size / 2
        self.dial_radius = 120
        # circular dial
        self.canvas.create_oval(self.dial_center - self.dial_radius, self.dial_center - self.dial_radius,
                                self.dial_center + self.dial_radius, self.dial_center + self.dial_radius,
                                outline="black")
        self.target_text = self.canvas.create_text(self.dial_center, self.dial_center,
                                                   text=f"{self.target_frequency:.2f} Hz",
                                                   font=("Helvetica", 20))
        self.needle = self.canvas.create_line(self.dial_center, self.dial_center,
                                              self.dial_center, self.dial_center - self.dial_radius + 20,
                                              width=6, fill="black")
        # labels
        self.info_label = tk.Label(self.left_frame, text="Detected: --", font=("Helvetica", 16))
        self.info_label.pack(pady=10)

        # right plots
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_time = self.fig.add_subplot(2, 1, 1)
        self.ax_freq = self.fig.add_subplot(2, 1, 2)
        # Time-domain
        x_time = np.linspace(0, WINDOW_SIZE / SAMPLE_FREQ, WINDOW_SIZE)
        y_time = np.zeros(WINDOW_SIZE)
        self.line_time, = self.ax_time.plot(x_time, y_time, color='blue')
        self.ax_time.set_xlim(0, WINDOW_SIZE / SAMPLE_FREQ)
        self.ax_time.set_ylim(-1, 1)
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.set_title("Waveform")
        # frequency domain
        x_freq = np.arange(0, 1001)
        y_freq = np.zeros(len(x_freq))
        self.line_freq, = self.ax_freq.plot(x_freq, y_freq, color='orange')
        self.ax_freq.set_xlim(0, 1000)
        self.ax_freq.set_ylim(0, 1000)
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        # Tkinter canvas
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_plot.draw()

    def set_target(self, string):
        with self.lock:
            self.target_string = string
            self.target_frequency = self.targets[string]
            # update the dial
            self.canvas.itemconfig(self.target_text, text=f"{self.target_frequency:.2f} Hz")

    def toggle_auto(self):
        if self.auto_var.get():
            # auto mode
            for btn in self.string_buttons:
                btn.config(state=tk.DISABLED)
        else:
            # manual mode
            for btn in self.string_buttons:
                btn.config(state=tk.NORMAL)
            # default setting
            self.set_target("E2")

    def update_gui(self):
        """
        periodically update
        """
        with self.lock:
            detected_freq = self.detected_frequency
            detected_note = self.detected_note
            if self.auto_var.get() and detected_freq is not None:
                min_diff = float('inf')
                closest_string = None
                for s, freq in self.targets.items():
                    diff = abs(detected_freq - freq)
                    if diff < min_diff:
                        min_diff = diff
                        closest_string = s
                if closest_string is not None and closest_string != self.target_string:
                    self.target_string = closest_string
                    self.target_frequency = self.targets[closest_string]
                    new_target_freq = self.target_frequency
                else:
                    new_target_freq = None
            else:
                new_target_freq = None

            target_freq = self.target_frequency
            if detected_freq is not None:
                delta = detected_freq - target_freq
            else:
                delta = 0
            # map angles
            clamped_delta = max(min(delta, 50), -50)
            angle_deg = 90 - (clamped_delta / 50) * 45
            angle_rad = math.radians(angle_deg)
            needle_length = self.dial_radius - 20
            end_x = self.dial_center + needle_length * math.cos(angle_rad)
            end_y = self.dial_center - needle_length * math.sin(angle_rad)

            # pointer colors
            if detected_freq is None or detected_note == "":
                needle_color = "black"
            elif abs(detected_freq - target_freq) <= 2.0:
                needle_color = "green"
            else:
                needle_color = "red"

            # texts
            if detected_freq is not None:
                info_text = f"Detected: {detected_note} {detected_freq:.1f} Hz"
            else:
                info_text = "Detected: --"

            # plot waveforms
            waveform = self.window_samples.copy()
            if hasattr(self, 'magnitude_spec') and self.magnitude_spec is not None:
                spectrum = self.magnitude_spec[:1001].copy()
            else:
                spectrum = np.zeros(1001)

        # update new dial
        if new_target_freq is not None:
            self.canvas.itemconfig(self.target_text, text=f"{new_target_freq:.2f} Hz")
        # update needle position and color
        self.canvas.coords(self.needle, self.dial_center, self.dial_center, end_x, end_y)
        self.canvas.itemconfig(self.needle, fill=needle_color)
        # update text
        self.info_label.config(text=info_text)
        # waveform and spectrum
        self.line_time.set_ydata(waveform)
        self.line_freq.set_ydata(spectrum)
        self.canvas_plot.draw()
        self.master.after(100, self.update_gui)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
            return
        with self.lock:
            # append new and keep the latest
            self.window_samples = np.concatenate((self.window_samples, indata[:, 0]))
            self.window_samples = self.window_samples[len(indata[:, 0]):]

            # check if above threshold
            signal_power = np.linalg.norm(self.window_samples, ord=2) ** 2 / len(self.window_samples)
            if signal_power < POWER_THRESH: # too low ignore
                self.detected_frequency = self.target_frequency
                self.detected_note = ""
                self.magnitude_spec = np.zeros(WINDOW_SIZE // 2)
                return

            # Hann window
            hann_samples = self.window_samples * HANN_WINDOW
            magnitude_spec = np.abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

            # cancel low frequency 62Hz below
            for i in range(int(62 / DELTA_FREQ)):
                magnitude_spec[i] = 0

            # suppress white noise
            for j in range(len(OCTAVE_BANDS) - 1):
                ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
                ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
                ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
                avg_energy = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2) ** 2 /
                              (ind_end - ind_start)) ** 0.5
                for i in range(ind_start, ind_end):
                    if magnitude_spec[i] < WHITE_NOISE_THRESH * avg_energy:
                        magnitude_spec[i] = 0

            self.magnitude_spec = magnitude_spec

            # interpolate spectrum
            mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS),
                                      np.arange(0, len(magnitude_spec)), magnitude_spec)
            norm = np.linalg.norm(mag_spec_ipol, ord=2)
            if norm:
                mag_spec_ipol /= norm

            # HPS
            hps_spec = copy.deepcopy(mag_spec_ipol)
            for i in range(NUM_HPS):
                length = int(np.ceil(len(mag_spec_ipol) / (i + 1)))
                tmp = hps_spec[:length] * mag_spec_ipol[::(i + 1)]
                if not np.any(tmp):
                    break
                hps_spec = tmp

            # the fundamental frequency
            max_ind = np.argmax(hps_spec)
            max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

            closest_note, closest_pitch = find_closest_note(max_freq)
            max_freq = round(max_freq, 1)
            # stabilize based on buffers
            self.noteBuffer.insert(0, closest_note)
            if len(self.noteBuffer) > 2:
                self.noteBuffer.pop()
            if self.noteBuffer.count(self.noteBuffer[0]) == len(self.noteBuffer):
                self.detected_frequency = max_freq
                self.detected_note = closest_note


if __name__ == "__main__":
    root = tk.Tk()
    app = GuitarTunerGUI(root)
    root.mainloop()