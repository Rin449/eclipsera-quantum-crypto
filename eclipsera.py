# ===================================================================
#   ECLIPSERA™ - Quantum Neural Cryptosystem v9
#   The Ultimate Post-Quantum Encryption Tool
#   Created by: ivan4154_4 | Ivan
#   © 2025 Vietnam Quantum Labs
# ===================================================================

import customtkinter as ctk
import torch
import torch.nn as nn
import threading
import json
import base64
import numpy as np
from tkinter import scrolledtext, messagebox, filedialog
from datetime import datetime

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

MSG_LEN = 16
KEY_LEN = 32
NOISE_STD = 0.07
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== MODELS ====================
class V9Alice(nn.Module):
    def __init__(self, msg_len=MSG_LEN, key_len=KEY_LEN, hidden=512):
        super().__init__()
        self.msg_proj = nn.Linear(msg_len, hidden)
        self.key_proj = nn.Linear(key_len, hidden)
        self.basis_proj = nn.Linear(msg_len, hidden)
        self.hadamard1 = nn.Linear(hidden, hidden)
        self.hadamard2 = nn.Linear(hidden, hidden)
        self.phase_gate = nn.Linear(hidden, hidden)
        self.res_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(0.08)
        ) for _ in range(6)])
        self.chaos_layer = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.output = nn.Linear(hidden, msg_len)

    def forward(self, msg, key, basis):
        msg_h = self.msg_proj(msg); key_h = self.key_proj(key); basis_h = self.basis_proj(basis)
        x = msg_h + key_h + basis_h * 0.25
        h1 = self.hadamard1(x); h2 = self.hadamard2(x)
        x = torch.sin(h1 * 3.1415926535/4) * torch.cos(h2 * 3.1415926535/4)
        phase = self.phase_gate(x)
        x = x * torch.cos(phase * 3.1415926535/4)
        residual = x
        for b in self.res_blocks: x = b(x) + residual * 0.08; residual = x
        chaos = self.chaos_layer(x)
        x = x * torch.sin(chaos * 3.1415926535) + torch.cos(chaos * 3.1415926535/2)
        return torch.sigmoid(self.output(x))

class V9Bob(nn.Module):
    def __init__(self, msg_len=MSG_LEN, key_len=KEY_LEN, hidden=512):
        super().__init__()
        self.joint_encoder = nn.Linear(msg_len + key_len, hidden)
        self.basis_decoder = nn.Linear(msg_len, hidden)
        self.transformer_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden*2), nn.ReLU(), nn.Dropout(0.08),
            nn.Linear(hidden*2, hidden), nn.LayerNorm(hidden), nn.Dropout(0.03)
        ) for _ in range(4)])
        self.measurement_proj = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, msg_len)

    def forward(self, cipher, key, basis):
        x = self.joint_encoder(torch.cat([cipher, key], dim=1))
        x = x + self.basis_decoder(basis) * 0.25
        residual = x
        for b in self.transformer_blocks: x = b(x) + residual * 0.85; residual = x
        x = x * torch.cos(self.measurement_proj(x) * 3.1415926535/4)
        return torch.sigmoid(self.output(x))

# Load models
try:
    alice = V9Alice().to(DEVICE)
    bob = V9Bob().to(DEVICE)
    alice.load_state_dict(torch.load("alice.pt", map_location=DEVICE))
    bob.load_state_dict(torch.load("bob.pt", map_location=DEVICE))
    alice.eval(); bob.eval()
except Exception as e:
    messagebox.showerror("LOAD ERROR", f"Failed to load neural models!\n{e}")
    exit()

# ==================== ENCRYPT ====================
def encrypt_text(text):
    bits = ''.join(format(ord(c), '08b') for c in text)
    bits += '0' * (-len(bits) % MSG_LEN)
    blocks = [bits[i:i+MSG_LEN] for i in range(0, len(bits), MSG_LEN)]
    msg = torch.tensor([[int(b) for b in block] for block in blocks], dtype=torch.float32, device=DEVICE)
    key = torch.randint(0, 2, (len(blocks), KEY_LEN), device=DEVICE).float()
    basis = torch.randint(0, 2, (len(blocks), MSG_LEN), device=DEVICE).float()

    with torch.no_grad():
        cipher = alice(msg, key, basis)

    cipher_b64 = base64.b64encode(cipher.flatten().cpu().numpy().tobytes()).decode()
    key_b64 = base64.b64encode(key.flatten().cpu().numpy().tobytes()).decode()

    result = {
        "ciphertext": cipher_b64,
        "key": key_b64,
        "original": text,
        "message": f"Eclipsera™ Encrypted: \"{text}\"",
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "system": "Eclipsera v9 GOD MODE",
        "author": "ivan4154_4 | Ivan"
    }
    return json.dumps(result, indent=2, ensure_ascii=False)

# ==================== DECRYPT ====================
def decrypt_ciphertext(ciphertext_b64, key_b64):
    try:
        cipher_bytes = base64.b64decode(ciphertext_b64)
        key_bytes = base64.b64decode(key_b64)

        if len(cipher_bytes) % (MSG_LEN * 4) != 0:
            return "ERROR: Invalid ciphertext length"

        n_blocks = len(cipher_bytes) // (MSG_LEN * 4)
        cipher_np = np.frombuffer(cipher_bytes, dtype=np.float32).copy()
        key_np = np.frombuffer(key_bytes, dtype=np.float32).copy()

        cipher = torch.from_numpy(cipher_np).reshape(n_blocks, MSG_LEN).to(DEVICE)
        key = torch.from_numpy(key_np).reshape(n_blocks, KEY_LEN).to(DEVICE)
        basis = torch.randint(0, 2, (n_blocks, MSG_LEN), device=DEVICE).float()

        with torch.no_grad():
            output = bob(cipher, key, basis)

        bits = (output > 0.5).int()
        bitstring = ''.join(str(b.item()) for b in bits.flatten())
        text = ''.join(chr(int(bitstring[i:i+8], 2)) for i in range(0, len(bitstring), 8)).rstrip('\x00')
        return text
    except Exception as e:
        return f"DECRYPTION FAILED:\n{str(e)}"

# ==================== ECLIPSERA GUI ====================
class Eclipsera(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Eclipsera™ - Quantum Neural Cryptosystem v9")
        self.geometry("1560x980")
        self.minsize(1200, 800)

        # Header
        header = ctk.CTkFrame(self, fg_color="#000000")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="ECLIPSERA™", font=("Impact", 64, "bold"), text_color="#00ff88").pack(pady=10)
        ctk.CTkLabel(header, text="Next-Generation Post-Quantum Encryption Engine", 
                     font=("Arial", 18, "italic"), text_color="#00ffaa").pack()
        ctk.CTkLabel(header, text="Created by ivan4154_4 | Ivan • Vietnam 2025", 
                     font=("Arial", 14), text_color="#888888").pack(pady=8)

        # Tabs
        tabview = ctk.CTkTabview(self, fg_color="#111111")
        tabview.pack(fill="both", expand=True, padx=25, pady=15)

        tab_encrypt = tabview.add("  ENCRYPT  ")
        tab_decrypt = tabview.add("  DECRYPT  ")

        self.build_encrypt_tab(tab_encrypt)
        self.build_decrypt_tab(tab_decrypt)

        # Footer
        footer = ctk.CTkLabel(self, text="© 2025 Eclipsera™ • All Rights Reserved • ivan4154_4", 
                              text_color="#444444", font=("Arial", 10))
        footer.pack(side="bottom", pady=10)

        self.last_json = ""

    def build_encrypt_tab(self, tab):
        ctk.CTkLabel(tab, text="ENCRYPT MESSAGE", font=("Impact", 38), text_color="#00ff88").pack(pady=25)

        frame = ctk.CTkFrame(tab)
        frame.pack(pady=15, padx=60, fill="both", expand=True)

        ctk.CTkLabel(frame, text="Enter your secret message (full UTF-8 + emoji support):", 
                     font=("Arial", 16)).pack(anchor="w", padx=25, pady=(15,8))
        self.txt_input = ctk.CTkTextbox(frame, height=130, font=("Segoe UI", 14))
        self.txt_input.pack(fill="x", padx=25, pady=8)
        self.txt_input.insert("1.0", "The future belongs to those who master quantum cryptography.")

        ctk.CTkButton(tab, text="ENCRYPT NOW", height=70, fg_color="#00ff44", hover_color="#00cc44",
                      font=("Arial", 32, "bold"), command=self.encrypt_action).pack(pady=30)

        res = ctk.CTkFrame(tab)
        res.pack(fill="both", expand=True, padx=60, pady=10)
        btns = ctk.CTkFrame(res)
        btns.pack(fill="x", pady=8)
        ctk.CTkButton(btns, text="COPY JSON", width=160, fg_color="#0066ff", 
                      command=self.copy_encrypt).pack(side="right", padx=25)

        self.encrypt_box = scrolledtext.ScrolledText(res, font=("Consolas", 11), bg="#001122", fg="#88ffaa", 
                                                   wrap="word", relief="flat", borderwidth=2)
        self.encrypt_box.pack(fill="both", expand=True, padx=25, pady=8)

    def build_decrypt_tab(self, tab):
        ctk.CTkLabel(tab, text="DECRYPT MESSAGE", font=("Impact", 38), text_color="#00ff88").pack(pady=25)

        frame = ctk.CTkFrame(tab)
        frame.pack(pady=15, padx=60, fill="x")

        ctk.CTkLabel(frame, text="Ciphertext (Base64):", font=("Arial", 14)).grid(row=0, column=0, sticky="w", padx=25, pady=(10,5))
        self.ct_input = ctk.CTkTextbox(frame, height=110)
        self.ct_input.grid(row=1, column=0, columnspan=2, sticky="ew", padx=25, pady=5)

        ctk.CTkLabel(frame, text="Key (Base64):", font=("Arial", 14)).grid(row=2, column=0, sticky="w", padx=25, pady=(15,5))
        self.key_input = ctk.CTkTextbox(frame, height=110)
        self.key_input.grid(row=3, column=0, columnspan=2, sticky="ew", padx=25, pady=5)

        btn_frame = ctk.CTkFrame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=25)
        ctk.CTkButton(btn_frame, text="LOAD JSON FILE", fg_color="#ff8800", width=200, 
                      command=self.load_json_file).pack(side="left", padx=30)
        ctk.CTkButton(btn_frame, text="DECRYPT NOW", height=55, fg_color="#00ff44", font=("Arial", 22, "bold"),
                      command=self.decrypt_action).pack(side="right", padx=30)

        frame.grid_columnconfigure(0, weight=1)

        res = ctk.CTkFrame(tab)
        res.pack(fill="both", expand=True, padx=60, pady=15)
        ctk.CTkLabel(res, text="DECRYPTED MESSAGE:", text_color="#00ff88", font=("Arial", 20, "bold")).pack(anchor="w", padx=25, pady=10)
        self.decrypt_box = scrolledtext.ScrolledText(res, height=12, font=("Segoe UI Emoji", 18), 
                                                   bg="#001100", fg="#00ff99")
        self.decrypt_box.pack(fill="both", expand=True, padx=25, pady=8)

    def encrypt_action(self):
        threading.Thread(target=self.do_encrypt, daemon=True).start()

    def do_encrypt(self):
        text = self.txt_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty", "Please enter a message to encrypt!")
            return
        json_out = encrypt_text(text)
        self.last_json = json_out
        self.after(0, lambda: self.encrypt_box.delete(1.0, "end"))
        self.after(0, lambda: self.encrypt_box.insert(1.0, json_out))

    def copy_encrypt(self):
        content = self.encrypt_box.get("1.0", "end").strip()
        if content:
            self.clipboard_clear()
            self.clipboard_append(content)
            messagebox.showinfo("Copied", "Eclipsera™ JSON copied to clipboard!")

    def decrypt_action(self):
        threading.Thread(target=self.do_decrypt, daemon=True).start()

    def do_decrypt(self):
        ct = self.ct_input.get("1.0", "end").strip()
        key = self.key_input.get("1.0", "end").strip()
        if not ct or not key:
            self.after(0, lambda: messagebox.showwarning("Missing Data", "Both ciphertext and key are required!"))
            return
        result = decrypt_ciphertext(ct, key)
        self.after(0, lambda: self.decrypt_box.delete(1.0, "end"))
        self.after(0, lambda: self.decrypt_box.insert(1.0, result))

    def load_json_file(self):
        file = filedialog.askopenfilename(title="Load Eclipsera JSON", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.ct_input.delete("1.0", "end")
                self.key_input.delete("1.0", "end")
                self.ct_input.insert("1.0", data.get("ciphertext", ""))
                self.key_input.insert("1.0", data.get("key", ""))
                messagebox.showinfo("Loaded", f"Eclipsera™ session loaded:\n{file.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")

if __name__ == "__main__":
    app = Eclipsera()
    app.mainloop()