import os
import random
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# Base photos directory (nested photos folder)
BASE_PHOTOS_DIR = r'photos'

# Collect all image files from all beer type directories
image_files = []
image_paths = []  # Store full paths for later use

# Iterate through each beer type directory
for beer_type in os.listdir(BASE_PHOTOS_DIR):
    beer_type_path = os.path.join(BASE_PHOTOS_DIR, beer_type)
    
    # Skip if not a directory
    if not os.path.isdir(beer_type_path):
        continue
    
    print(f"Scanning {beer_type} directory...")
    
    # Collect images directly from the beer type directory
    for filename in os.listdir(beer_type_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            full_path = os.path.join(beer_type_path, filename)
            image_files.append(f"{beer_type}/{filename}")
            image_paths.append(full_path)

print(f"Found {len(image_files)} total images across all beer types")

# If there are more than 10 images, randomly select 10
if len(image_files) > 20:
    selected_indices = random.sample(range(len(image_files)), 20)
    image_files = [image_files[i] for i in selected_indices]
    image_paths = [image_paths[i] for i in selected_indices]

# Shuffle the images for the test
combined = list(zip(image_files, image_paths))
random.shuffle(combined)
image_files, image_paths = zip(*combined)

print(f"Selected {len(image_files)} images for testing")

# Define the possible classes (adjust as needed)
CLASSES = ['ipa', 'stout', 'cider', 'wheat', 'lager', 'not beer']

class ImageTestApp:
    def __init__(self, master, image_files, image_paths):
        self.master = master
        self.image_files = image_files
        self.image_paths = image_paths
        self.index = 0
        self.score = 0
        self.total = len(image_files)
        self.user_guesses = []
        self.correct_labels = []  # Store correct labels based on directory names
        
        # Extract correct labels from directory names
        for image_file in image_files:
            beer_type = image_file.split('/')[0]  # Get the beer type from path
            self.correct_labels.append(beer_type)

        # Configure the main window
        self.master.title("🍺 Beer Type Classification Test")
        self.master.geometry("800x800")
        self.master.resizable(False, False)
        self.master.configure(bg='#2c3e50')  # Dark blue background
        
        # Set window icon (if available)
        try:
            self.master.iconbitmap('default')
        except:
            pass

        # Create main container with rounded corners effect
        self.main_container = tk.Frame(master, bg='#34495e', relief='raised', bd=2)
        self.main_container.pack(expand=True, fill='both', padx=15, pady=15)

        # Header
        header_frame = tk.Frame(self.main_container, bg='#34495e')
        header_frame.pack(fill='x', pady=(15, 10))
        
        title_label = tk.Label(header_frame, text="🍺 Beer Classification Test", 
                              font=("Arial", 16, "bold"), 
                              fg='#ecf0f1', bg='#34495e')
        title_label.pack()

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(header_frame, variable=self.progress_var, 
                                           maximum=self.total, length=300)
        self.progress_bar.pack(pady=(10, 5))

        # Image display area with border
        image_frame = tk.Frame(self.main_container, bg='#2c3e50', relief='sunken', bd=3)
        image_frame.pack(pady=15, padx=20)
        
        self.img_label = tk.Label(image_frame, bg='#2c3e50', relief='flat')
        self.img_label.pack(padx=5, pady=5)

        # Radio buttons in a styled frame
        radio_frame = tk.Frame(self.main_container, bg='#34495e')
        radio_frame.pack(fill='x', padx=20, pady=10)

        # Radio button label
        tk.Label(radio_frame, text="Select Beer Type:", 
                font=("Arial", 12, "bold"), 
                fg='#ecf0f1', bg='#34495e').pack(anchor='w', pady=(0, 5))

        self.var = tk.StringVar()
        self.var.set(CLASSES[0])

        self.radio_buttons = []
        for i, c in enumerate(CLASSES):
            # Create a frame for each radio button with hover effect
            rb_frame = tk.Frame(radio_frame, bg='#34495e')
            rb_frame.pack(fill='x', pady=2)
            
            rb = tk.Radiobutton(rb_frame, text=c.title(), variable=self.var, value=c, 
                               font=("Arial", 11), fg='#ecf0f1', bg='#34495e',
                               selectcolor='#2c3e50', activebackground='#34495e',
                               activeforeground='#3498db')
            rb.pack(anchor='w', padx=10)
            self.radio_buttons.append(rb)

        # Submit button with modern styling
        button_frame = tk.Frame(self.main_container, bg='#34495e')
        button_frame.pack(pady=15)
        
        self.submit_btn = tk.Button(button_frame, text="Submit Answer", 
                                   command=self.submit, 
                                   font=("Arial", 12, "bold"),
                                   fg='white', bg='#3498db', 
                                   activebackground='#2980b9', activeforeground='white',
                                   relief='flat', bd=0, padx=30, pady=8)
        self.submit_btn.pack()

        # Status label with better styling
        self.status_label = tk.Label(self.main_container, text="", 
                                    font=("Arial", 10), 
                                    fg='#bdc3c7', bg='#34495e')
        self.status_label.pack(pady=10)

        # Score display
        self.score_label = tk.Label(self.main_container, text="", 
                                   font=("Arial", 11, "bold"), 
                                   fg='#f39c12', bg='#34495e')
        self.score_label.pack(pady=5)

        self.show_image()

    def show_image(self):
        img_path = self.image_paths[self.index]
        img = cv2.imread(img_path)
        if img is None:
            self.status_label.config(text=f"Could not read image: {self.image_files[self.index]}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        # Resize image to fit nicely in the frame
        img_pil = img_pil.resize((250, 250), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.img_label.config(image=self.photo)
        
        # Update progress and status
        progress = (self.index + 1) / self.total * 100
        self.progress_var.set(self.index + 1)
        self.status_label.config(text=f"Image {self.index+1} of {self.total} ({progress:.0f}% complete)")
        self.score_label.config(text=f"Current Score: {self.score}/{self.index}")

    def submit(self):
        guess = self.var.get()
        self.user_guesses.append(guess)
        
        # Check if guess is correct
        correct_label = self.correct_labels[self.index]
        is_correct = guess == correct_label
        
        if is_correct:
            self.score += 1
            # Show brief success feedback
            self.submit_btn.config(text="✓ Correct!", bg='#27ae60')
            self.master.after(500, lambda: self.submit_btn.config(text="Submit Answer", bg='#3498db'))
        else:
            # Show brief incorrect feedback
            self.submit_btn.config(text="✗ Wrong", bg='#e74c3c')
            self.master.after(500, lambda: self.submit_btn.config(text="Submit Answer", bg='#3498db'))
        
        self.index += 1
        if self.index < self.total:
            self.master.after(600, self.show_image)  # Small delay for feedback
        else:
            self.finish_test()

    def finish_test(self):
        # Calculate final score
        percentage = (self.score / self.total) * 100
        
        # Create a more detailed result message
        result_text = f"🎉 Test Complete! 🎉\n\n"
        result_text += f"📊 Final Score: {self.score}/{self.total} ({percentage:.1f}%)\n\n"
        
        if percentage >= 80:
            result_text += "🏆 Excellent! You're a beer expert!\n"
        elif percentage >= 60:
            result_text += "👍 Good job! You know your beers!\n"
        elif percentage >= 40:
            result_text += "📚 Not bad! Keep learning!\n"
        else:
            result_text += "📖 Keep practicing! You'll get better!\n"
        
        result_text += "\n📋 Detailed Results:\n"
        for i, fname in enumerate(self.image_files):
            correct = self.correct_labels[i]
            guess = self.user_guesses[i]
            status = "✓" if guess == correct else "✗"
            result_text += f"{i+1}. {status} Your guess: {guess.title()} | Correct: {correct.title()}\n"
        
        messagebox.showinfo("Test Results", result_text)
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTestApp(root, image_files, image_paths)
    root.mainloop()
