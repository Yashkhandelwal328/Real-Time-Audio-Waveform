# frontend.py
import sys
import os
import math
from collections import deque

import pygame
import numpy as np
from PIL import Image, ImageOps

from backend import AudioBackend, make_amplitude_bins

# ========================= GLOBAL CONFIG =========================

pygame.init()

BG_COLOR = (3, 15, 20)
WAVE_COLOR = (0, 255, 140)
TEXT_COLOR = (190, 255, 200)
LINE_THICKNESS = 2
FPS = 60

MODES = {
    "music":   "MUSIC mode",
    "podcast": "PODCAST mode",
    "noise":   "NOISE mode",
}

THEME_NAMES = {
    1: "Waveform",
    2: "Neon Heart",
    3: "Bars",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ========================= IMAGE UTILS =========================

def load_and_fix_wallpaper(filename, width, height):
    full_path = os.path.join(BASE_DIR, filename)
    try:
        img = Image.open(full_path)
        img = ImageOps.exif_transpose(img)  # fix EXIF orientation
        img = img.resize((width, height), Image.LANCZOS)

        mode = img.mode
        data = img.tobytes()

        if mode == "RGBA":
            surf = pygame.image.frombytes(data, img.size, "RGBA")
        else:
            surf = pygame.image.frombytes(data, img.size, "RGB")

        return surf.convert()
    except Exception as e:
        print(f"[IMG] Failed to load {full_path}: {e}")
        return None


# ========================= FRONTEND / VISUALIZER =========================

def draw_center_glow(surface, x, y, max_radius, width, height):
    glow = pygame.Surface((width, height), pygame.SRCALPHA)
    steps = 18
    for i in range(steps):
        t = i / steps
        radius = int(max_radius * t)
        alpha_c = int(120 * (1.0 - t))
        color = (0, 255, 140, alpha_c)
        if radius > 0:
            pygame.draw.circle(glow, color, (x, y), radius, width=2)
    surface.blit(glow, (0, 0))


class VisualizerApp:
    def __init__(self):
        # fullscreen display
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Neon Audio Visualizer")

        self.width, self.height = self.screen.get_size()
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

        self.current_mode = "music"
        self.current_theme = 2
        self.frame = 0
        self.shape_history = deque(maxlen=14)

        self.wallpaper_files = ["1.jpg", "2.jpg", "3.png"]
        self.backgrounds = []
        self.bg_index = 0
        self.load_backgrounds()

        # audio backend, chunk = screen width so 1 sample per pixel
        self.backend = AudioBackend(rate=44100, chunk=self.width)

        username = self.login_screen()
        self.username = username
        self.backend.set_username(username)
        print(f"[LOGIN] '{self.username}'")

    # ---------- backgrounds ----------

    def load_backgrounds(self):
        self.backgrounds = []
        print(f"[PATH] Script dir: {BASE_DIR}")
        for fname in self.wallpaper_files:
            surf = load_and_fix_wallpaper(fname, self.width, self.height)
            if surf:
                self.backgrounds.append(surf)
                print(f"[IMG] Loaded background: {os.path.join(BASE_DIR, fname)}")
            else:
                print(f"[IMG] Skipped: {fname}")

        if not self.backgrounds:
            print("[IMG] No backgrounds loaded, using solid color only.")

    def set_background(self, index: int):
        if not self.backgrounds:
            return
        self.bg_index = index % len(self.backgrounds)

    def next_background(self):
        self.set_background(self.bg_index + 1)

    def prev_background(self):
        self.set_background(self.bg_index - 1)

    def get_bg_button_rects(self):
        n = max(1, len(self.backgrounds))
        panel_width = 260
        panel_height = 70
        panel_x = 16
        panel_y = 90   # moved down so it doesn't cover the dB bar

        button_margin = 8
        button_w = (panel_width - (button_margin * (n + 1))) // n
        button_h = panel_height - 24
        rects = []
        for i in range(n):
            x = panel_x + button_margin + i * (button_w + button_margin)
            y = panel_y + 12
            rects.append(pygame.Rect(x, y, button_w, button_h))
        return rects, pygame.Rect(panel_x, panel_y, panel_width, panel_height)

    # ---------- login ----------

    def login_screen(self):
        username = ""
        active = True
        while active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit(0)
                    elif event.key == pygame.K_RETURN:
                        if username.strip():
                            active = False
                            break
                    elif event.key == pygame.K_BACKSPACE:
                        username = username[:-1]
                    else:
                        if event.unicode.isprintable():
                            username += event.unicode

            self.screen.fill(BG_COLOR)
            title = self.font.render("Login", True, TEXT_COLOR)
            prompt = self.small_font.render(
                "Enter username and press Enter", True, TEXT_COLOR
            )
            user_text = self.font.render(username or "_", True, WAVE_COLOR)

            rect_title = title.get_rect(center=(self.center_x, self.center_y - 60))
            rect_prompt = prompt.get_rect(center=(self.center_x, self.center_y - 20))
            rect_user = user_text.get_rect(center=(self.center_x, self.center_y + 20))

            self.screen.blit(title, rect_title)
            self.screen.blit(prompt, rect_prompt)
            self.screen.blit(user_text, rect_user)

            pygame.display.flip()
            self.clock.tick(30)

        return username.strip()

    # ---------- main loop ----------

    def run(self):
        running = True
        while running:
            self.frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_F11:
                        self.toggle_fullscreen()

                    if event.key == pygame.K_m:
                        self.current_mode = "music"
                    elif event.key == pygame.K_p:
                        self.current_mode = "podcast"
                    elif event.key == pygame.K_n:
                        self.current_mode = "noise"

                    elif event.key == pygame.K_1:
                        self.current_theme = 1
                    elif event.key == pygame.K_2:
                        self.current_theme = 2
                    elif event.key == pygame.K_3:
                        self.current_theme = 3

                    elif event.key == pygame.K_RIGHT:
                        self.next_background()
                    elif event.key == pygame.K_LEFT:
                        self.prev_background()

                    # switch input device
                    elif event.key == pygame.K_l:
                        self.backend.next_device()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    rects, panel_rect = self.get_bg_button_rects()
                    if panel_rect.collidepoint(mx, my):
                        for i, r in enumerate(rects):
                            if r.collidepoint(mx, my):
                                self.set_background(i)
                                break

            self.backend.process_frame()
            self.draw_frame()

            pygame.display.flip()
            self.clock.tick(FPS)

    def toggle_fullscreen(self):
        flags = self.screen.get_flags()
        if flags & pygame.FULLSCREEN:
            self.screen = pygame.display.set_mode((1280, 720))
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        self.width, self.height = self.screen.get_size()
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # reload wallpapers at new resolution
        self.load_backgrounds()

    # ---------- drawing ----------

    def draw_frame(self):
        self.draw_background()

        glow_radius = int(
            min(self.width, self.height) * (0.25 + self.backend.energy * 1.5)
        )
        draw_center_glow(
            self.screen,
            self.center_x,
            self.center_y,
            glow_radius,
            self.width,
            self.height,
        )

        if self.current_theme == 1:
            self.draw_waveform()
        elif self.current_theme == 2:
            self.draw_neon_heart()
        elif self.current_theme == 3:
            self.draw_bars()

        self.draw_db_box()
        self.draw_info_box()
        self.draw_legend()
        self.draw_bg_selector()

    def draw_background(self):
        if self.backgrounds:
            self.screen.blit(self.backgrounds[self.bg_index], (0, 0))
        else:
            self.screen.fill(BG_COLOR)

    def draw_waveform(self):
        samples = self.backend.samples
        if len(samples) == 0:
            return

        energy = float(self.backend.energy)
        db = float(self.backend.ema_db)

        # basically silent → flat line
        if db < -35.0 and energy < 0.02:
            pygame.draw.line(
                self.screen,
                WAVE_COLOR,
                (0, self.center_y),
                (self.width, self.center_y),
                2,
            )
            return

        # stronger smoothing on the raw audio
        smooth_kernel = np.ones(5, dtype=np.float32) / 5.0
        samples_smoothed = np.convolve(samples, smooth_kernel, mode="same")

        total = len(samples_smoothed)
        x_coords = np.linspace(0, self.width - 1, total).astype(int)

        # amplitude based on dB but softer response
        t = (db + 50.0) / 40.0        # [-50..-10] -> [0..1]
        t = max(0.0, min(1.0, t))
        amp_scale = (t ** 0.8) * 0.9  # a bit more chill
        base_amp = self.height // 4
        amplitude = int(base_amp * amp_scale)
        if amplitude < 10:
            amplitude = 10

        ys = (samples_smoothed * amplitude + self.center_y).astype(np.float32)

        # extra smoothing on the y positions to remove jaggedness
        ys = np.convolve(ys, np.ones(5, dtype=np.float32) / 5.0, mode="same").astype(int)

        points = list(zip(x_coords, ys))
        if len(points) > 1:
            pygame.draw.lines(
                self.screen,
                WAVE_COLOR,
                False,
                points,
                2,
            )


        energy = float(self.backend.energy)
        db = float(self.backend.ema_db)

        # basically silent → flat line
        if db < -35.0 and energy < 0.02:
            pygame.draw.line(
                self.screen,
                WAVE_COLOR,
                (0, self.center_y),
                (self.width, self.center_y),
                2,
            )
            return

        # light smoothing
        smooth_kernel = np.ones(3, dtype=np.float32) / 3.0
        samples_smoothed = np.convolve(samples, smooth_kernel, mode="same")

        total = len(samples_smoothed)
        x_coords = np.linspace(0, self.width - 1, total).astype(int)

        # amplitude based on dB
        t = (db + 50.0) / 40.0        # [-50..-10] -> [0..1]
        t = max(0.0, min(1.0, t))
        amp_scale = t ** 0.7
        base_amp = self.height // 4
        amplitude = int(base_amp * amp_scale)
        if amplitude < 10:
            amplitude = 10

        ys = (samples_smoothed * amplitude + self.center_y).astype(int)

        points = list(zip(x_coords, ys))
        if len(points) > 1:
            pygame.draw.lines(
                self.screen,
                WAVE_COLOR,
                False,
                points,
                2,
            )

    def draw_neon_heart(self):
        samples = self.backend.samples
        num_bins = 140
        mags = make_amplitude_bins(samples, num_bins=num_bins)

        base_size = min(self.width, self.height) // 18
        beat = 1.0 + self.backend.energy * 2.8
        wobble_strength = 0.35
        angle_offset = self.frame * 0.01

        pts = []
        for i in range(num_bins):
            t = (2.0 * math.pi * i) / num_bins

            xh = 16 * math.sin(t) ** 3
            yh = (
                13 * math.cos(t)
                - 5 * math.cos(2 * t)
                - 2 * math.cos(3 * t)
                - math.cos(4 * t)
            )

            xh *= base_size
            yh *= -base_size

            mag = mags[i]
            disturb = wobble_strength * mag * base_size

            dx, dy = xh, yh
            length = math.hypot(dx, dy) or 1.0
            nx, ny = dx / length, dy / length

            x = xh * beat + disturb * nx
            y = yh * beat + disturb * ny

            ca = math.cos(angle_offset)
            sa = math.sin(angle_offset)
            rx = x * ca - y * sa
            ry = x * sa + y * ca

            pts.append((int(self.center_x + rx), int(self.center_y + ry)))

        self.shape_history.append(pts)

        history_list = list(self.shape_history)
        for idx, path in enumerate(history_list):
            age = len(history_list) - idx
            t_age = age / len(history_list)

            alpha_c = int(35 + 170 * (1.0 - t_age))
            width = 1 if age < len(history_list) - 1 else 2

            scale = 1.0 - 0.06 * t_age
            scaled_path = []
            for (x, y) in path:
                dx = x - self.center_x
                dy = y - self.center_y
                sx = int(self.center_x + dx * scale)
                sy = int(self.center_y + dy * scale)
                scaled_path.append((sx, sy))

            layer_surface = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA
            )
            col = (0, 255, 170, alpha_c)
            if len(scaled_path) > 1:
                pygame.draw.lines(layer_surface, col, True, scaled_path, width)
            self.screen.blit(layer_surface, (0, 0))

    def draw_bars(self):
        samples = self.backend.samples
        num_bins = 64
        mags = make_amplitude_bins(samples, num_bins=num_bins)

        bar_w = self.width / num_bins
        max_bar_h = self.height * 0.65
        bottom_y = self.height - 30

        for i in range(num_bins):
            h = mags[i] * max_bar_h
            x = int(i * bar_w)
            y = int(bottom_y - h)
            rect = pygame.Rect(x + 1, y, int(bar_w) - 2, int(h))
            pygame.draw.rect(self.screen, WAVE_COLOR, rect)

    def draw_db_box(self):
        box_w, box_h = 320, 70
        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((0, 0, 0, 170))

        # title
        title_surf = self.small_font.render("LEVEL", True, (150, 200, 180))
        box.blit(title_surf, (10, 6))

        # value text
        label = f"{self.backend.ema_db:.1f} dBFS"
        value_surf = self.font.render(label, True, TEXT_COLOR)
        box.blit(value_surf, (10, 24))

        # normalized loudness
        t_val = max(0.0, min(1.0, (self.backend.ema_db + 60.0) / 60.0))

        # color gradient green -> yellow -> red
        r = int(255 * (t_val ** 1.2))
        g = int(230 * (1.0 - max(0, t_val - 0.1)))
        b = 90
        bar_color = (r, max(0, g), b)

        bar_w, bar_h = box_w - 24, 14
        x0, y0 = 10, box_h - bar_h - 10

        # track
        pygame.draw.rect(
            box,
            (18, 40, 35),
            (x0, y0, bar_w, bar_h),
            border_radius=7,
        )

        # fill
        fill_w = int(bar_w * t_val)
        if fill_w > 0:
            pygame.draw.rect(
                box,
                bar_color,
                (x0, y0, fill_w, bar_h),
                border_radius=7,
            )

        # border
        pygame.draw.rect(
            box,
            (80, 120, 110),
            (x0, y0, bar_w, bar_h),
            width=1,
            border_radius=7,
        )

        self.screen.blit(box, (8, 8))

    def draw_info_box(self):
        info_w, info_h = 420, 160
        info = pygame.Surface((info_w, info_h), pygame.SRCALPHA)
        info.fill((0, 0, 0, 170))

        mode_text = MODES.get(self.current_mode, "Unknown mode")
        mode_label_surf = self.small_font.render("MODE", True, (130, 190, 160))
        mode_text_surf = self.font.render(mode_text, True, TEXT_COLOR)

        detail = f"{self.backend.ema_db:.1f} dBFS | Centroid ~ {int(self.backend.centroid)} Hz"
        detail_surf = self.small_font.render(detail, True, (210, 230, 220))

        user_surf = self.small_font.render(
            f"User: {self.username}", True, (200, 230, 220)
        )
        mood_surf = self.small_font.render(
            f"Mood: {self.backend.current_mood}", True, (180, 220, 200)
        )

        input_label = self.backend.get_input_label()
        input_surf = self.small_font.render(
            f"Input: {input_label}", True, (190, 230, 210)
        )

        pill_text = self.current_mode.upper()
        pill_color = (0, 200, 120)
        pill_surf = self.small_font.render(pill_text, True, (0, 0, 0))

        info.blit(mode_label_surf, (10, 6))
        info.blit(mode_text_surf, (10, 24))
        info.blit(detail_surf, (10, 48))
        info.blit(user_surf, (10, 72))
        info.blit(mood_surf, (10, 96))
        info.blit(input_surf, (10, 120))

        pill_rect = pygame.Rect(info_w - 110, 20, 100, 28)
        pygame.draw.rect(info, pill_color, pill_rect, border_radius=8)
        info.blit(pill_surf, (pill_rect.x + 8, pill_rect.y + 4))

        self.screen.blit(info, (self.width - info_w - 10, 10))

    def draw_legend(self):
        legend_text = (
            "M/P/N: Modes  |  1/2/3: Themes  |  L: Switch input  |  ←/→ or click Scenes: Background  |  F11: Fullscreen"
        )
        legend_surface = self.small_font.render(legend_text, True, (160, 210, 190))
        legend_rect = legend_surface.get_rect()
        legend_rect.bottomright = (self.width - 8, self.height - 8)
        self.screen.blit(legend_surface, legend_rect)

    def draw_bg_selector(self):
        if not self.backgrounds:
            return

        rects, panel_rect = self.get_bg_button_rects()
        panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))

        title = self.small_font.render("Scenes", True, (210, 230, 220))
        panel.blit(title, (10, 4))

        for i, r in enumerate(rects):
            local_rect = pygame.Rect(
                r.x - panel_rect.x, r.y - panel_rect.y, r.width, r.height
            )
            is_active = (i == self.bg_index)

            base_color = (60, 80, 90) if not is_active else (0, 200, 150)
            border_color = (140, 200, 190) if is_active else (100, 150, 150)

            pygame.draw.rect(panel, base_color, local_rect, border_radius=8)

            thumb = pygame.transform.smoothscale(
                self.backgrounds[i], (local_rect.width, local_rect.height)
            )
            thumb.set_alpha(210 if is_active else 170)
            panel.blit(thumb, local_rect)

            pygame.draw.rect(panel, border_color, local_rect, width=2, border_radius=8)

            label = f"{i+1}"
            text = self.small_font.render(label, True, (10, 20, 20))
            t_rect = text.get_rect()
            t_rect.midbottom = (local_rect.centerx, local_rect.bottom - 4)
            panel.blit(text, t_rect)

        self.screen.blit(panel, panel_rect.topleft)


# ========================= ENTRY =========================

if __name__ == "__main__":
    app = VisualizerApp()
    try:
        app.run()
    finally:
        try:
            app.backend.finalize_session()
        except Exception as e:
            print(f"[FINALIZE] Error while saving session: {e}")
        try:
            app.backend.close()
        except Exception:
            pass
        pygame.quit()
        sys.exit(0)
