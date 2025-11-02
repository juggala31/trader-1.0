from __future__ import annotations
import json, time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from exosati_trader.config.loader import load as load_cfg

HEALTH_FILE = Path('logs/healthbeat.json')

class StatusLight(ttk.Frame):
    def __init__(self, master, label: str):
        super().__init__(master, padding=6)
        self.dot = tk.Canvas(self, width=18, height=18, highlightthickness=0)
        self.lbl = ttk.Label(self, text=label, width=20, anchor='w')
        self.dot.pack(side='left')
        self.lbl.pack(side='left', padx=(8,0))
        self._state = 'red'
        self._draw()

    def _draw(self):
        self.dot.delete('all')
        color = '#2ecc71' if self._state == 'green' else ('#f1c40f' if self._state=='yellow' else '#e74c3c')
        self.dot.create_oval(2,2,16,16, fill=color, outline='')

    def set_state(self, state: str):
        self._state = state
        self._draw()

class App(tk.Tk):
    def __init__(self, cfg_path: str = 'configs/ftmo.yaml'):
        super().__init__()
        self.title('Exosati Trader — Enhanced Dashboard (MVP)')
        self.geometry('720x420')
        self.resizable(False, False)
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        self.cfg = load_cfg(cfg_path)

        # Header
        header = ttk.Frame(self, padding=10)
        header.pack(fill='x')
        ttk.Label(header, text='Config', font=('Segoe UI', 12, 'bold')).pack(side='left')
        ttk.Label(header, text=f" | Broker: {self.cfg.broker.platform}@{self.cfg.broker.server} | Symbols: {', '.join(self.cfg.symbols)} | TF: {', '.join(self.cfg.timeframes)}", foreground='#555').pack(side='left')

        # Status lights
        lights = ttk.Frame(self, padding=10)
        lights.pack(fill='x')
        self.light_health = StatusLight(lights, 'Healthbeat')
        self.light_health.pack(anchor='w')

        # Body
        body = ttk.Frame(self, padding=10)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='Notes', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        notes = (
            "• This MVP uses a simple file-based heartbeat (no ZMQ yet).\n"
            "• Start the healthbeat service first, then launch this dashboard.\n"
            "• Edit configs/ftmo.yaml to change symbols, risk, and ports."
        )
        self.notes = tk.Text(body, height=6, wrap='word')
        self.notes.insert('1.0', notes)
        self.notes.configure(state='disabled')
        self.notes.pack(fill='x')

        # Footer controls
        footer = ttk.Frame(self, padding=10)
        footer.pack(fill='x')
        ttk.Button(footer, text='Open Config', command=self._open_cfg).pack(side='left')
        ttk.Button(footer, text='Exit', command=self.destroy).pack(side='right')

        # Start polling thread
        self._stop = False
        t = threading.Thread(target=self._poll_health, daemon=True)
        t.start()

    def _open_cfg(self):
        messagebox.showinfo('Config Path', 'configs/ftmo.yaml')

    def _poll_health(self):
        while not self._stop:
            try:
                data = json.loads(Path(HEALTH_FILE).read_text(encoding='utf-8'))
                state = 'green' if (time.time() - float(data.get('ts', 0))) < 5 else 'yellow'
            except Exception:
                state = 'red'
            self.after(0, lambda s=state: self.light_health.set_state(s))
            time.sleep(1.0)

def main():
    App().mainloop()

if __name__ == '__main__':
    main()