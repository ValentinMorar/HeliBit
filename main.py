"""
HeliBit-AI: Primary Application Entry Point
Launches the Graphical Chat Window or retrains the model.

Usage:
  python main.py          (Launches the Desktop Chat GUI Window)
  python main.py --train  (Retrains the model on dataset.json)
"""

import sys
import os
import argparse
import tkinter as tk
from tkinter import scrolledtext
from helibit import HeliBitEngine, HeliBitTrainer


class HeliBitChatGUI:
    """
    Tkinter Graphical User Interface for interactive HeliBit-AI v2 model inference.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HeliBit-AI v2: Compositional Neuromorphic Engine")
        self.root.geometry("800x640")
        self.root.minsize(680, 520)

        # Color Palette (Dark Theme)
        self.BG_DARK = "#1E1E2E"
        self.CARD_BG = "#181825"
        self.INPUT_BG = "#313244"
        self.TEXT_COLOR = "#CDD6F4"
        self.ACCENT_BLUE = "#89B4FA"
        self.ACCENT_GREEN = "#A6E3A1"
        self.ACCENT_RED = "#F38BA8"

        self.root.configure(bg=self.BG_DARK)

        # Initialize HeliBit-AI Engine
        self.engine = HeliBitEngine()
        self.trainer = HeliBitTrainer(self.engine)

        checkpoint_file = "helibit_model.json"
        if os.path.exists(checkpoint_file):
            try:
                self.trainer.load_checkpoint(checkpoint_file)
            except Exception as e:
                print(f"Notice: Could not load checkpoint ({e}), using default model state.")

        self._build_ui()
        self._append_welcome_message()

    def _build_ui(self):
        """Constructs window layout and controls."""
        header_frame = tk.Frame(self.root, bg=self.CARD_BG, pady=10, padx=15)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="HeliBit-AI v2: Compositional Engine",
            font=("Segoe UI", 14, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_BLUE,
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header_frame,
            text="Structured Bitboard Roles | Zero-FPU Integer Math | Abstention Guard",
            font=("Segoe UI", 9),
            bg=self.CARD_BG,
            fg="#9399B2",
        )
        subtitle_lbl.pack(side=tk.RIGHT)

        self.chat_display = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self.BG_DARK,
            fg=self.TEXT_COLOR,
            insertbackground=self.TEXT_COLOR,
            relief=tk.FLAT,
            padx=15,
            pady=15,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        self.chat_display.tag_config("user_tag", foreground="#89B4FA", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("ai_tag", foreground="#A6E3A1", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("ai_unknown_tag", foreground="#F38BA8", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("meta_tag", foreground="#9399B2", font=("Consolas", 9, "italic"))
        self.chat_display.tag_config("system_tag", foreground="#F9E2AF", font=("Consolas", 9))

        input_frame = tk.Frame(self.root, bg=self.CARD_BG, pady=10, padx=15)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.entry_box = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            bg=self.INPUT_BG,
            fg=self.TEXT_COLOR,
            insertbackground=self.TEXT_COLOR,
            relief=tk.FLAT,
            bd=5,
        )
        self.entry_box.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 10))
        self.entry_box.bind("<Return>", self._on_send)
        self.entry_box.focus_set()

        send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 10, "bold"),
            bg=self.ACCENT_BLUE,
            fg="#11111B",
            activebackground="#74C7EC",
            activeforeground="#11111B",
            relief=tk.FLAT,
            padx=15,
            pady=4,
            command=self._on_send,
            cursor="hand2",
        )
        send_btn.pack(side=tk.RIGHT)

        self.telemetry_var = tk.StringVar(value="Status: Ready | Enter natural language query prompt (e.g. 'quarter past five')")
        telemetry_bar = tk.Label(
            self.root,
            textvariable=self.telemetry_var,
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#A6ADC8",
            anchor=tk.W,
            padx=15,
            pady=3,
        )
        telemetry_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _append_welcome_message(self):
        """Displays initial greeting and prompt guidance."""
        welcome_txt = (
            "======================================================================\n"
            " Welcome to HeliBit-AI v2 Compositional Neuro-Symbolic Engine!\n"
            " Features: Structured 64-bit Bitboard roles, zero-FPU arithmetic,\n"
            "           unseen combination generalization, and explicit UNKNOWN abstention.\n\n"
            " Try valid time expressions:\n"
            "   - 'quarter past five'       -> 05:15\n"
            "   - 'twenty-five to seven'    -> 06:35  (compositional generalization)\n"
            "   - '14:30'                   -> 14:30\n"
            "   - 'ten to midnight'         -> 23:50\n"
            " Try adversarial inputs:\n"
            "   - 'banana o'clock'          -> UNKNOWN (abstention guard)\n"
            "   - 'the room is cold'        -> UNKNOWN\n"
            "======================================================================\n\n"
        )
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, welcome_txt, "system_tag")
        self.chat_display.config(state=tk.DISABLED)

    def _on_send(self, event=None):
        """Handles message submission and inference."""
        user_text = self.entry_box.get().strip()
        if not user_text:
            return

        self.entry_box.delete(0, tk.END)

        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"You: {user_text}\n", "user_tag")

        result = self.engine.predict(user_text)

        tag = "ai_unknown_tag" if result.is_unknown() else "ai_tag"
        ai_response = f"HeliBit-AI: Output -> {result.predicted_target}\n"
        meta_info = (
            f"            [Confidence: {result.confidence:.2f} | Slots: {result.slots} | Latency: {result.latency_us:.1f} us | Overlap: {result.bitboard_overlap_score} bits]\n\n"
        )

        self.chat_display.insert(tk.END, ai_response, tag)
        self.chat_display.insert(tk.END, meta_info, "meta_tag")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

        self.telemetry_var.set(
            f"Last: {result.predicted_target} | Conf: {result.confidence:.2f} | Latency: {result.latency_us:.1f} us | Slots: {result.slots}"
        )


def launch_gui():
    root = tk.Tk()
    app = HeliBitChatGUI(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="HeliBit-AI v2 Main Application Entry Point")
    parser.add_argument("--train", action="store_true", help="Retrain model on dataset.json")
    args = parser.parse_args()

    if args.train:
        from train import run_training
        run_training()
    else:
        launch_gui()


if __name__ == "__main__":
    main()
