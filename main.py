import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

import predict

def create_main_frame():
    my_font1 = ('times', 36, 'bold')

    label_main = tk.Label(current_frame, text='NHAN DANG CAM XUC',
                          width=30, font=my_font1)
    label_main.pack()
    # Thêm khoảng trống giữa Upload và Predict
    tk.Label(current_frame, text='', height=15).pack()

    button_upload = tk.Button(current_frame, text='Upload File', font=(
        'times', 18, 'bold'), width=20, command=upload_file_dialog)
    button_upload.pack()
    # Thêm khoảng trống giữa Upload và Predict
    tk.Label(current_frame, text='', height=2).pack()
    button_real_time = tk.Button(current_frame, text='Real Time', font=(
        'times', 18, 'bold'), width=20, command=real_time_update)
    button_real_time.pack()


def create_upload_frame():
    global img_label_upload, img_upload, img_upload, uploaded_filename
    img_label_upload = tk.Label(current_frame)

    back_button = tk.Button(current_frame, text='Back', font=(
        'times', 18, 'bold'), command=show_main_frame)
    back_button.pack(side=tk.TOP, anchor='nw', padx=0, pady=0)
    upload_button = tk.Button(current_frame, text='Upload File', font=(
        'times', 18, 'bold'), command=upload_file_dialog)
    upload_button.pack(side=tk.TOP, anchor='n', padx=0, pady=0)
    # Thêm khoảng trống giữa Upload và Predict
    tk.Label(current_frame, text='', height=1).pack()
    if uploaded_filename:
        # img_upload = Image.open(uploaded_filename)
        # img_resized = img_upload.resize((480, 360))  # new width & height
        # img_tk_upload = ImageTk.PhotoImage(img_resized)

        img_upload = cv2.imread(uploaded_filename)
        img_resized = cv2.resize(img_upload, (480, 360), interpolation=cv2.INTER_AREA)

        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_tk_upload = ImageTk.PhotoImage(Image.fromarray(img_resized_rgb))

        # image = img_upload.convert("L")
        # image = np.expand_dims(img_to_array(img_upload.resize((480, 360))), [0])

        # Hiển thị hình ảnh trên nhãn
        img_label_upload.config(image=img_tk_upload)
        img_label_upload.image = img_tk_upload
        img_label_upload.pack()

        # Thêm khoảng trống giữa Upload và Predict
        tk.Label(current_frame, text='', height=1).pack()
        predict_button = tk.Button(current_frame, text='Predict', font=(
            'times', 18, 'bold'), command=predict_image_function)
        predict_button.pack(side=tk.TOP, anchor='n', padx=0, pady=0)


def show_upload_frame():
    clear_frame()
    create_upload_frame()

def show_main_frame():
    clear_frame()
    create_main_frame()


def upload_file_dialog():
    global uploaded_filename
    filename = filedialog.askopenfilename(title="Select an image file",
                                          filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif"), ("All files", "*.*")])
    if filename:
        uploaded_filename = filename
        show_upload_frame()


def real_time_update():
    predict.predict_real_time()


def clear_frame():
    for widget in current_frame.winfo_children():
        widget.destroy()


def print_text(text):
    label_main = tk.Label(current_frame, text=text,
                          width=30, font=('times', 36, 'bold'))
    label_main.pack()
    clear_frame()


def create_predict_frame(image):
    clear_frame()
    back_button = tk.Button(current_frame, text='Back', font=(
        'times', 18, 'bold'), command=show_main_frame)
    back_button.pack(side=tk.TOP, anchor='nw', padx=0, pady=0)
    upload_button = tk.Button(current_frame, text='Upload File', font=(
        'times', 18, 'bold'), command=upload_file_dialog)
    upload_button.pack(side=tk.TOP, anchor='n', padx=0, pady=0)

    tk.Label(current_frame, text='', height=1).pack()

    img_label_upload = tk.Label(current_frame)

    img_resized = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_tk_upload = ImageTk.PhotoImage(Image.fromarray(img_resized_rgb))


    # Hiển thị hình ảnh trên nhãn
    img_label_upload.config(image=img_tk_upload)
    img_label_upload.image = img_tk_upload
    img_label_upload.pack()


def predict_image_function():
    image = predict.predict_image(img_upload)
    create_predict_frame(image)

root = tk.Tk()
root.geometry("800x600")  # Kích thước cửa sổ
root.title('FER')

# Tạo frame chính
current_frame = tk.Frame(root)
current_frame.pack(fill='both', expand=True)

uploaded_filename = None  # Biến để theo dõi xem người dùng đã chọn tệp hay chưa

create_main_frame()

root.mainloop()
