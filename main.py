import os
import cv2  # OpenCV
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from tkinter import *
import PIL.Image, PIL.ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize the camera
cap = cv2.VideoCapture(0) 

# Use a pipeline
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Creates a Tkinter window to display image
window = Tk()
window.title("Capture Image")
canvas = Canvas(window, width=700, height=400)
canvas.pack()

# Create a figure for the pie chart
fig = plt.Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)

# Add the figure to the Tkinter window
chart = FigureCanvasTkAgg(fig, window)
chart.get_tk_widget().pack()


# CAPTURING IMAGE PROCESS
def capture_image():
    ret, frame = cap.read()
    cv2.imwrite('captured_image.jpeg', frame)
    result = pipe('captured_image.jpeg')

    # Separate labels and scores
    labels = [prediction['label'] for prediction in result]
    scores = [prediction['score'] for prediction in result]
    ax.clear()
    ax.pie(scores, labels=labels, autopct='%1.1f%%')
    chart.draw()

    # Delay for 2 seconds
    window.after(2000, capture_image)  # hehe 60fps

    # Delete the image after the usecase is done
    os.remove('captured_image.jpeg')


# Start capturing images
capture_image()


# TO UPDATE THE CANVAS IMAGE PROCESS
def update_image():
    ret, frame = cap.read()
    if ret:
        image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        canvas.photo = PIL.ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=canvas.photo, anchor=NW)

    window.after(10, update_image)


# Calling all the functions
update_image()
window.mainloop()
cap.release()
