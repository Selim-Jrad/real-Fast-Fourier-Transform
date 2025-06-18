import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
import sounddevice as sd
import librosa
from collections import deque
from scipy.fft import rfft, irfft, rfftfreq
import tkinter as tk
from tkinter import filedialog
import os

def run_visualizer(audio_path=None):
    plt.style.use('dark_background')
    
    if audio_path is None:
        root = tk.Tk()
        root.withdraw()
        audio_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(
                ("Audio files", "*.wav;*.mp3;*.flac;*.ogg"),
                ("All files", "*.*")
            )
        )
        root.destroy()
        
        if not audio_path:
            print("No file selected. Exiting.")
            return
    
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return
    
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    block_size = 2048
    display_samples = sr // 10
    update_interval = 30

    time_buffer = deque(maxlen=display_samples)
    spectrum_buffer = deque(maxlen=10)
    freqs = rfftfreq(block_size, 1/sr)
    window = np.blackman(block_size)

    lowcut = 100
    highcut = 1000
    filter_enabled = False

    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(14, 8), facecolor='#121212')
    plt.subplots_adjust(bottom=0.3)

    plot_colors = {
        'background': '#2b2b2b',
        'text': '#ffffff',
        'time_plot': '#00ffff',
        'freq_plot': '#ffff00',
        'button': '#404040',
        'input': '#000000',
        'selection': '#ff00ff'
    }

    filename = os.path.basename(audio_path)
    fig.canvas.manager.set_window_title(f"Audio Visualizer - {filename}")

    time_line, = ax_time.plot([], [], lw=1, color=plot_colors['time_plot'])
    ax_time.set_title('Time Domain Signal', color=plot_colors['text'])
    ax_time.set(xlabel='Time (s)', ylabel='Amplitude', 
               xlim=(0, display_samples/sr), ylim=(-1, 1),
               facecolor=plot_colors['background'])
    ax_time.tick_params(colors=plot_colors['text'])

    freq_line, = ax_freq.semilogx([], [], lw=1, color=plot_colors['freq_plot'])
    selected_band, = ax_freq.semilogx([], [], lw=2, color=plot_colors['selection'])
    ax_freq.set_title('Frequency Spectrum', color=plot_colors['text'])
    ax_freq.set(xlabel='Frequency (Hz)', ylabel='Magnitude (dB)',
               xlim=(20, sr/2), ylim=(0, 100),
               facecolor=plot_colors['background'])
    ax_freq.grid(True, which="both", ls="--", alpha=0.3, color=plot_colors['text'])
    ax_freq.tick_params(colors=plot_colors['text'])

    stream = None
    paused = True
    audio_position = 0

    def apply_bandpass(data, lowcut, highcut):
        fft_data = rfft(data * window)
        mask = (freqs >= lowcut) & (freqs <= highcut)
        fft_data[~mask] = 0
        return irfft(fft_data).real

    def audio_callback(outdata, frames, time, status):
        nonlocal audio_position
        if paused or audio_position >= len(y):
            outdata.fill(0)
            return
        
        start = audio_position
        end = min(start + frames, len(y))
        chunk = y[start:end]
        
        if len(chunk) == block_size:
            if filter_enabled:
                filtered_chunk = apply_bandpass(chunk, lowcut, highcut)
            else:
                filtered_chunk = chunk

            time_buffer.extend(filtered_chunk[::4])
            spectrum = 20 * np.log10(np.abs(rfft(filtered_chunk * window)) + 1e-9)
            spectrum_buffer.append(spectrum)
        
        outdata[:] = filtered_chunk.reshape(-1, 1)
        audio_position = end

    def update(frame):
        if not paused:
            time_data = np.array(time_buffer)
            time_x = np.linspace(0, display_samples/sr, len(time_data))
            time_line.set_data(time_x, time_data)
            
            if spectrum_buffer:
                avg_spectrum = np.mean(spectrum_buffer, axis=0)
                freq_line.set_data(freqs, avg_spectrum)
                mask = (freqs >= lowcut) & (freqs <= highcut)
                selected_band.set_data(freqs[mask], avg_spectrum[mask])
                valid_mags = avg_spectrum[(freqs >= 20) & (freqs <= 20000)]
                if len(valid_mags) > 0:
                    ax_freq.set_ylim(np.min(valid_mags)-5, np.max(valid_mags)+5)
        
        return time_line, freq_line, selected_band

    def toggle_filter(event):
        nonlocal filter_enabled
        filter_enabled = not filter_enabled
        filter_btn.color = '#00ff00' if filter_enabled else plot_colors['button']
        fig.canvas.draw_idle()

    def update_lowcut(text):
        nonlocal lowcut
        try: 
            lowcut = float(text)
            lowcut = max(20, min(sr/2, lowcut))
            if lowcut > highcut:
                highcut_box.set_val(str(lowcut))
        except ValueError:
            pass
        update(0)

    def update_highcut(text):
        nonlocal highcut
        try: 
            highcut = float(text)
            highcut = max(20, min(sr/2, highcut))
            if highcut < lowcut:
                lowcut_box.set_val(str(highcut))
        except ValueError:
            pass
        update(0)

    ax_filter = plt.axes([0.3, 0.15, 0.4, 0.04])
    filter_btn = Button(ax_filter, 'Toggle Filter', color=plot_colors['button'])
    filter_btn.on_clicked(toggle_filter)

    ax_lowcut = plt.axes([0.3, 0.10, 0.2, 0.04])
    lowcut_box = TextBox(ax_lowcut, 'Low Cut (Hz)', initial=str(lowcut), color=plot_colors['input'])
    lowcut_box.on_submit(update_lowcut)

    ax_highcut = plt.axes([0.55, 0.10, 0.2, 0.04])
    highcut_box = TextBox(ax_highcut, 'High Cut (Hz)', initial=str(highcut), color=plot_colors['input'])
    highcut_box.on_submit(update_highcut)

    def toggle_play(event):
        nonlocal paused, stream
        if paused:
            start_stream()
        else:
            stop_stream()

    def start_stream():
        nonlocal stream, paused
        stream = sd.OutputStream(samplerate=sr, blocksize=block_size,
                                channels=1, callback=audio_callback)
        stream.start()
        paused = False

    def stop_stream():
        nonlocal stream, paused
        if stream:
            stream.stop()
            stream.close()
        paused = True

    def reset(event):
        nonlocal audio_position
        audio_position = 0
        time_buffer.clear()
        spectrum_buffer.clear()
        update(0)
        
    def change_file(event):
        nonlocal y, sr, audio_position
        stop_stream()
        
        root = tk.Tk()
        root.withdraw()
        new_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(
                ("Audio files", "*.wav;*.mp3;*.flac;*.ogg"),
                ("All files", "*.*")
            )
        )
        root.destroy()
        
        if new_path and os.path.exists(new_path):
            try:
                y, sr = librosa.load(new_path, sr=None)
                filename = os.path.basename(new_path)
                fig.canvas.manager.set_window_title(f"Audio Visualizer - {filename}")
                reset(None)
            except Exception as e:
                print(f"Error loading audio file: {e}")

    ax_change = plt.axes([0.3, 0.20, 0.4, 0.04])
    change_btn = Button(ax_change, 'Change Audio File', color=plot_colors['button'])
    change_btn.on_clicked(change_file)

    btn_play = Button(plt.axes([0.3, 0.05, 0.2, 0.04]), 'Play/Pause', color=plot_colors['button'])
    btn_play.on_clicked(toggle_play)

    btn_reset = Button(plt.axes([0.55, 0.05, 0.2, 0.04]), 'Reset', color=plot_colors['button'])
    btn_reset.on_clicked(reset)

    ani = animation.FuncAnimation(fig, update, interval=update_interval,
                                blit=True, cache_frame_data=False)
    fig.canvas.mpl_connect('key_press_event', lambda e: toggle_play(None) if e.key == ' ' else None)
    fig.canvas.mpl_connect('close_event', lambda e: stop_stream())

    plt.show()

if __name__ == '__main__':
    run_visualizer()
     