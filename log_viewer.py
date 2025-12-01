import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

LOG_FILE = "listening_data.json"


class LogViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Listening History Viewer")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")

        self.data = []

        # --- Top Controls ---
        top = tk.Frame(root, bg="#1e1e1e")
        top.pack(fill="x", pady=10)

        tk.Label(top, text="User:", fg="white", bg="#1e1e1e").pack(side="left", padx=5)
        self.user_filter = tk.Entry(top, width=10)
        self.user_filter.pack(side="left")

        tk.Label(top, text="Mood:", fg="white", bg="#1e1e1e").pack(side="left", padx=5)
        self.mood_filter = tk.Entry(top, width=10)
        self.mood_filter.pack(side="left")

        ttk.Button(top, text="Apply Filter", command=self.apply_filter).pack(side="left", padx=5)
        ttk.Button(top, text="Reset", command=self.load_data).pack(side="left", padx=5)
        ttk.Button(top, text="Open File", command=self.pick_file).pack(side="right", padx=5)

        # --- Table Frame ---
        table_frame = tk.Frame(root, bg="#1e1e1e")
        table_frame.pack(fill="both", expand=True)

        columns = ("timestamp", "user", "mood", "avg_db")

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=20
        )

        # Styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2a2a2a", foreground="white", fieldbackground="#2a2a2a", rowheight=28)
        style.configure("Treeview.Heading", background="#333333", foreground="white")
        style.map("Treeview", background=[("selected", "#444444")])

        # Columns
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.heading("user", text="User")
        self.tree.heading("mood", text="Mood")
        self.tree.heading("avg_db", text="Average dB")

        self.tree.column("timestamp", width=260, anchor="w")
        self.tree.column("user", width=100, anchor="center")
        self.tree.column("mood", width=100, anchor="center")
        self.tree.column("avg_db", width=100, anchor="center")

        # Scrollbar
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        # Load initial data
        self.load_data()

    # --- Load JSON ---
    def load_data(self):
        if not os.path.exists(LOG_FILE):
            messagebox.showerror("Error", f"{LOG_FILE} does not exist.")
            return

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                self.data = json.load(f)
                if not isinstance(self.data, list):
                    raise ValueError("JSON root must be a list")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")
            return

        self.refresh_table(self.data)

    # --- Apply Filters ---
    def apply_filter(self):
        uf = self.user_filter.get().strip()
        mf = self.mood_filter.get().strip().lower()

        filtered = []
        for item in self.data:
            if uf and str(item["user"]) != uf:
                continue
            if mf and item["mood"].lower() != mf:
                continue
            filtered.append(item)

        self.refresh_table(filtered)

    # --- Refresh Table ---
    def refresh_table(self, rows):
        for x in self.tree.get_children():
            self.tree.delete(x)

        for row in rows:
            self.tree.insert(
                "", "end",
                values=(
                    row.get("timestamp", ""),
                    row.get("user", ""),
                    row.get("mood", ""),
                    row.get("avg_db", ""),
                )
            )

    def pick_file(self):
        global LOG_FILE
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            LOG_FILE = path
            self.load_data()


# --- RUN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LogViewer(root)
    root.mainloop()
