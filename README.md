🎧 Real-Time Audio Visualizer (Python + PyAudio + Pygame)

A real-time audio visualizer that reacts to microphone input using waveform, neon-heart, and bar-style animations.
It analyzes loudness (RMS), spectral centroid, and mood, then visualizes everything in a smooth, dynamic UI.

👥 Contributors

Yash Khandelwal

Yash Pratap

Himank Singh

Pratik Tiwari

Chaitanya Singh

📌 Features

🎵 Three visualization modes

Waveform

Neon Heart

Bars

🎙️ Real-time audio processing with PyAudio

📊 Spectral Centroid (brightness of sound)

🔊 RMS-based dB level meter

😎 Mood classifier (silence, chill, energetic, sad)

🖼️ Dynamic wallpapers with selector

🔁 Device switching (cycle between microphones)

📁 JSON logging of listening sessions

💻 Full-screen responsive Pygame UI

🛠️ Tech Stack

Python 3.10+

PyAudio

NumPy

Pygame-CE

Pillow (PIL)

🚀 Installation & Setup
1. Clone the repository
git clone https://github.com/Yashkhandelwal328/Real-Time-Audio-Waveform.git
cd Real-Time-Audio-Waveform

2. Create Conda environment
conda create -n visualizer python=3.10 -y

3. Activate environment
conda activate visualizer

4. Install dependencies
pip install -r requirements.txt

5. Run the project
python frontend.py

🎮 Controls
Key	Action
1 / 2 / 3	Change visualization theme
M / P / N	Switch mode (music / podcast / noise)
Left / Right Arrow	Change wallpaper
L	Switch audio input device
F11	Toggle fullscreen
ESC	Quit
📐 How It Works (Short Explanation)
Audio Backend

Reads microphone data in chunks

Converts raw bytes → NumPy array

Normalizes audio

Calculates:

RMS → dBFS

Energy

Spectral Centroid (FFT)

Frontend

Gets samples & processed data every frame

Draws animated shapes based on amplitude

Applies glow, rotation, and smooth transitions

Shows UI overlays (dB meter, mode, username, etc.)

📂 Project Structure
Real-Time-Audio-Waveform/
│
├── backend.py             # audio processing
├── frontend.py            # pygame GUI + visualizer
├── log_viewer.py          # view JSON logs
├── listening_data.json    # auto-generated session logs
├── requirements.txt
├── environment.yml        # optional conda file
├── 1.jpg / 2.jpg / 3.png  # wallpapers
└── README.md


🧠 Future Improvements

ML-powered mood detection using training data

Music beat detection

Export visualization as video

Plugin for system audio 