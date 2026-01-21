import tkinter as tk
import keyboard
import pyautogui
import pyperclip
import time
import csv
import os

CSV_FILE = "data/raw/reddata.csv"

# Ensure CSV exists with headers if new
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "score"])  # header


def copy_url():
    """Copy current browser URL into clipboard."""
    pyautogui.hotkey('ctrl', 'l')
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.1)
    url = pyperclip.paste()
    return url

def is_white(rgb, threshold=240):
    return all(c >= threshold for c in rgb)


def show_dialog(url):
    """Show tkinter dialog to enter score/value."""
    root = tk.Tk()
    root.title("Score Input")
    root.geometry("280x80")
    root.attributes('-topmost', True)
    root.focus_force()  # Force window to take focus

    label = tk.Label(root, text=f"URL captured!\nEnter score:")
    label.pack()

    entry = tk.Entry(root)
    entry.pack()
    entry.focus_set()  # Set focus to entry field
    root.after(10, lambda: entry.focus_force())  # Ensure entry gets focus after window is fully rendered

    
    def on_enter(event=None):
        score = entry.get().strip()
        if score:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([url, score])
        root.destroy()

    entry.bind("<Return>", on_enter)
    root.mainloop()


def on_up_arrow():
    url = copy_url()
    show_dialog(url)



print("[INFO] Tool running...")
print("[INFO] Press UP ARROW to capture URL & enter score.")
print("[INFO] Press CTRL+C in terminal to quit.")

keyboard.add_hotkey('up', on_up_arrow)
keyboard.wait()