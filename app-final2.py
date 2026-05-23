#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import json
import threading
import time
from datetime import datetime, date

import cv2
import numpy as np
import pandas as pd  # لإنشاء شيتات إكسيل الحقيقية
import pygame        # المشغل المعتمد والأسرع للملفات الصوتية المباشرة

import customtkinter as ctk
from PIL import Image
import face_recognition

# ==========================================
#  الإعدادات الأساسية (محسنة بالكامل للـ CPU)
# ==========================================
FACE_LOCATION_MODEL = "hog"  
TOLERANCE        = 0.45      
PROCESS_EVERY_N  = 7         
FRAME_SCALE      = 0.15      

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
EMP_DIR    = os.path.join(BASE_DIR, "employee")

for d in [DATA_DIR, CONFIG_DIR, LOGS_DIR, EMP_DIR]:
    os.makedirs(d, exist_ok=True)

# تعديل اسم ملف الأوزان والإعدادات ليتوافق مع الهوية الجديدة للمشروع
WEIGHTS_FILE   = os.path.join(DATA_DIR, "mecatronics_face_weights.pkl")
SETTINGS_FILE  = os.path.join(CONFIG_DIR, "mecatronics_settings.json")

COLORS = {
    "bg": "#0f0f1a", "card": "#1a1a2e", "accent": "#16213e",
    "primary": "#e94560", "success": "#00ff88", "warning": "#ffaa00",
    "danger": "#ff3333", "info": "#00d4ff", "text": "#ffffff", "muted": "#888888"
}

class MecatronicsAttendance:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Mecatronics Attendance | Smart System")
        self.root.geometry("600x500")
        self.root.configure(fg_color=COLORS["bg"])
        ctk.set_appearance_mode("dark")
        
        self.is_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.fps, self.frame_count, self.fps_time = 0, 0, time.time()
        self.last_greeted = {}
        self.latest_results = None

        # تهيئة الـ mixer الخاص بـ pygame لتشغيل ملفك الصوتي المخصص
        pygame.mixer.init()

        self.known_encodings, self.known_names, self.known_roles = [], [], []
        self.employee_dir = EMP_DIR
        self.on_time_limit = self.load_settings()

        # جلب داتا الموظفين وبناء ملف الأوزان للـ AI
        self.employees_data = self.scan_employees()
        if self.check_and_build_weights() == "loaded":
            self.show_main_menu()

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    return json.load(f).get("on_time", "09:00")
            except: pass
        return "09:00"

    def save_settings(self, val):
        self.on_time_limit = val
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"on_time": val}, f)

    def scan_employees(self):
        """قراءة الفولدرات، تجميع الصور، وتحديد مسار ملف الـ mp3 المخصص لكل موظف"""
        data = {}
        if not os.path.exists(self.employee_dir): return data
        for role_entry in os.listdir(self.employee_dir):
            role_path = os.path.join(self.employee_dir, role_entry)
            if not os.path.isdir(role_path): continue
            role_name = role_entry.replace("_", " ").replace("-", " ")
            for emp_entry in os.listdir(role_path):
                emp_path = os.path.join(role_path, emp_entry)
                if not os.path.isdir(emp_path): continue
                
                images = []
                voice_file = None
                
                # البحث عن الصور وعن ملف الـ MP3 المخصص جوه نفس الفولدر
                for f in os.listdir(emp_path):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        images.append(os.path.join(emp_path, f))
                    elif f.lower().endswith(".mp3"):
                        voice_file = os.path.join(emp_path, f)
                        
                if images:
                    data[emp_entry] = {
                        "name": emp_entry, 
                        "images": images, 
                        "role": role_name,
                        "voice_file": voice_file
                    }
        return data

    def check_and_build_weights(self):
        if os.path.exists(WEIGHTS_FILE) and len(self.employees_data) > 0:
            try:
                with open(WEIGHTS_FILE, "rb") as f:
                    saved = pickle.load(f)
                if set(saved.get("names", [])) == set(self.employees_data.keys()):
                    self.known_encodings = saved["encodings"]
                    self.known_names = saved["names"]
                    self.known_roles = saved["roles"]
                    return "loaded"
            except: pass
        if not self.employees_data:
            self.show_error("❌ No employee folders found inside 'employee/' folder!")
            return "error"
        self.train_model(self.employees_data)
        return "training"

    def train_model(self, employees):
        self.clear_screen()
        ctk.CTkLabel(self.root, text="⚡ MECATRONICS AI TRAINING (CPU)", font=("Arial", 24, "bold"), text_color=COLORS["primary"]).pack(pady=50)
        self.train_lbl = ctk.CTkLabel(self.root, text="Initializing...", font=("Consolas", 14))
        self.train_lbl.pack(pady=10)
        threading.Thread(target=self._training_worker, args=(employees,), daemon=True).start()

    def _training_worker(self, employees):
        encodings, names, roles = [], [], []
        items = list(employees.items())
        for idx, (name, info) in enumerate(items):
            self.root.after(0, lambda n=name, i=idx, t=len(items): self.train_lbl.configure(text=f"Training: {n} ({i+1}/{t})"))
            person_encs = []
            for img_path in info["images"]:
                try:
                    img = face_recognition.load_image_file(img_path)
                    locs = face_recognition.face_locations(img, model="hog")
                    enc = face_recognition.face_encodings(img, locs)
                    if enc: person_encs.append(enc[0])
                except: pass
            if person_encs:
                encodings.append(np.mean(person_encs, axis=0))
                names.append(name)
                roles.append(info["role"])
        with open(WEIGHTS_FILE, "wb") as f:
            pickle.dump({"encodings": encodings, "names": names, "roles": roles}, f)
        self.known_encodings, self.known_names, self.known_roles = encodings, names, roles
        self.root.after(500, self.show_main_menu)

    def show_main_menu(self):
        self.clear_screen()
        self.root.geometry("600x500")
        
        # تغيير الواجهة الرئيسية للاسم الجديد ميكاترونكس
        ctk.CTkLabel(self.root, text="MECATRONICS ATTENDANCE", font=("Arial", 24, "bold"), text_color=COLORS["text"]).pack(pady=40)
        ctk.CTkLabel(self.root, text="🚀 Custom Voice Mode Active", font=("Arial", 12), text_color=COLORS["success"]).pack(pady=5)
        
        ctk.CTkButton(self.root, text="▶  START ATTENDANCE", fg_color=COLORS["primary"], font=("Arial", 16, "bold"), height=50, width=250, command=self.open_camera_window).pack(pady=10)
        ctk.CTkButton(self.root, text="⚙  SETTINGS", fg_color=COLORS["accent"], font=("Arial", 14), height=40, width=250, command=self.open_settings).pack(pady=10)
        ctk.CTkButton(self.root, text="🚪  EXIT", fg_color=COLORS["danger"], font=("Arial", 14), height=40, width=250, command=self.quit_app).pack(pady=10)

    def open_settings(self):
        self.clear_screen()
        ctk.CTkLabel(self.root, text="⚙ Settings (24h format HH:MM)", font=("Arial", 20)).pack(pady=30)
        self.time_entry = ctk.CTkEntry(self.root, placeholder_text="e.g. 09:00", width=150)
        self.time_entry.insert(0, self.on_time_limit)
        self.time_entry.pack(pady=10)
        ctk.CTkButton(self.root, text="💾 SAVE", fg_color=COLORS["success"], command=lambda: (self.save_settings(self.time_entry.get()), self.show_main_menu())).pack(pady=20)

    def open_camera_window(self):
        self.clear_screen()
        self.root.geometry("800x650")
        
        self.fps_lbl = ctk.CTkLabel(self.root, text="FPS: --", font=("Consolas", 14), text_color=COLORS["success"])
        self.fps_lbl.pack(pady=5)
        
        self.video_label = ctk.CTkLabel(self.root, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=5)
        
        ctk.CTkButton(self.root, text="⏹ BACK TO MENU", fg_color=COLORS["danger"], command=self.stop_and_home).pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.frame_counter = 0
        
        threading.Thread(target=self.capture_loop, daemon=True).start()
        self.update_ui()

    def capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: continue
            self.frame_counter += 1
            if self.frame_counter % PROCESS_EVERY_N == 0 and self.known_encodings:
                self.process_frame(frame)
            with self.frame_lock:
                self.latest_frame = frame
            time.sleep(0.01)

    def process_frame(self, frame):
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        
        if not locs:
            with self.frame_lock: self.latest_results = None
            return

        encs = face_recognition.face_encodings(rgb, locs)
        names, roles, statuses, colors = [], [], [], []
        is_early = datetime.now().strftime("%H:%M") <= self.on_time_limit

        for enc in encs:
            dists = face_recognition.face_distance(self.known_encodings, enc)
            best_idx = np.argmin(dists)
            if dists[best_idx] <= TOLERANCE:
                name = self.known_names[best_idx]
                role = self.known_roles[best_idx]
                status = "EARLY ✅" if is_early else "LATE ⚠️"
                color = COLORS["success"] if is_early else COLORS["warning"]
                
                # تسجيل الحضور اليومي في الإكسيل ومنع التكرار
                is_new_log = self.log_to_excel(name, role, status)
                if is_new_log:
                    self.handle_custom_voice_greeting(name)
            else:
                name, role, status, color = "Unknown", "Visitor", "", COLORS["danger"]
            names.append(name)
            roles.append(role)
            statuses.append(status)
            colors.append(color)

        with self.frame_lock:
            self.latest_results = (locs, names, roles, statuses, colors)

    def log_to_excel(self, name, role, status):
        """إنشاء وحفظ شيت حضور إكسيل منظم بالتاريخ اليومي في مجلد logs"""
        today_str = date.today().isoformat()
        excel_file = os.path.join(LOGS_DIR, f"attendance_{today_str}.xlsx")
        current_time = datetime.now().strftime("%H:%M:%S")
        clean_name = name.replace("_", " ")

        new_row = {
            "التاريخ": [today_str],
            "الوقت": [current_time],
            "الاسم": [clean_name],
            "الوظيفة": [role],
            "الحالة": [status]
        }
        new_df = pd.DataFrame(new_row)

        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file)
                if clean_name in df["الاسم"].values:
                    return False  
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_excel(excel_file, index=False)
                print(f"📊 [Excel] Recorded attendance for: {clean_name}")
                return True
            except: return False
        else:
            try:
                new_df.to_excel(excel_file, index=False)
                print(f"📊 [Excel] Created new daily sheet for: {clean_name}")
                return True
            except: return False

    def handle_custom_voice_greeting(self, name):
        """تأكيد عدم تكرار النداء الصوتي متتالياً في نفس الدقيقة تفادياً للإزعاج"""
        now = time.time()
        if name in self.last_greeted and now - self.last_greeted[name] < 20: return
        self.last_greeted[name] = now
        threading.Thread(target=self._voice_worker, args=(name,), daemon=True).start()

    def _voice_worker(self, name):
        """تشغيل ملف الـ MP3 الموجود في مجلد الشخص مباشرة غصب عن أي نظام"""
        emp_info = self.employees_data.get(name, {})
        voice_path = emp_info.get("voice_file")

        if voice_path and os.path.exists(voice_path):
            try:
                print(f"🔊 [Audio Engine] Playing custom voice for: {name}")
                pygame.mixer.music.unload()  
                pygame.mixer.music.load(voice_path)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"❌ [Audio Error] Couldn't play file {voice_path}: {e}")
        else:
            print(f"⚠️ [Audio Warning] No custom MP3 found in folder for '{name}'. File log only.")

    def update_ui(self):
        if not self.is_running: return
        self.frame_count += 1
        t = time.time()
        if t - self.fps_time >= 1.0:
            self.fps = self.frame_count / (t - self.fps_time)
            self.frame_count, self.fps_time = 0, t
            self.fps_lbl.configure(text=f"FPS: {self.fps:.1f} (CPU Optimized)")

        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            results = self.latest_results

        if frame is None:
            self.root.after(10, self.update_ui)
            return

        if results is not None:
            locs, names, roles, statuses, colors = results
            scale = int(1 / FRAME_SCALE)
            for (top, right, bottom, left), name, role, status, color in zip(locs, names, roles, statuses, colors):
                top *= scale; right *= scale; bottom *= scale; left *= scale
                bgr = (int(color[5:7],16), int(color[3:5],16), int(color[1:3],16)) if len(color)==7 else (0,0,255)
                cv2.rectangle(frame, (left, top), (right, bottom), bgr, 2)
                cv2.putText(frame, f"{name.replace('_', ' ')}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.current_video_image = ctk.CTkImage(light_image=pil, size=(640, 480))
        self.video_label.configure(image=self.current_video_image)
        self.root.after(10, self.update_ui)

    def stop_and_home(self):
        self.is_running = False
        if hasattr(self, "cap") and self.cap: self.cap.release()
        self.show_main_menu()

    def clear_screen(self):
        for w in self.root.winfo_children(): w.destroy()

    def show_error(self, msg):
        self.clear_screen()
        ctk.CTkLabel(self.root, text=msg, font=("Arial", 14), text_color=COLORS["danger"]).pack(pady=50)

    def quit_app(self):
        self.is_running = False
        pygame.mixer.quit()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = ctk.CTk()
    app = MecatronicsAttendance(root)
    root.mainloop()