#*******************************
#Author:
#u3253248 u3275890 Assessment 3_price_predictor_gui 20/ 10/2024
#Programming:
#*******************************

import sys
import os
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the saved model, scaler, and column names
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
Xcolumns = joblib.load('Xcolumns.pkl')

# Function to get the correct path for PyInstaller bundled files
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load the saved model, scaler, and column names using resource_path()
model = joblib.load(resource_path("model.pkl"))
scaler = joblib.load(resource_path("scaler.pkl"))
X_columns = joblib.load(resource_path("Xcolumns.pkl"))

# Function to predict price based on user input
def predict_price_gui():
    try:
        # Initialize DataFrame with the saved columns (all set to 0)
        input_data = pd.DataFrame(np.zeros((1, len(Xcolumns))), columns=Xcolumns)

        # Update relevant features based on user input
        if f"Brand_{brand_var.get()}" in input_data.columns:
            input_data.at[0, f"Brand_{brand_var.get()}"] = 1

        if f"Category_{category_var.get()}" in input_data.columns:
            input_data.at[0, f"Category_{category_var.get()}"] = 1

        if f"Color_{color_var.get()}" in input_data.columns:
            input_data.at[0, f"Color_{color_var.get()}"] = 1

        if f"Size_{size_var.get()}" in input_data.columns:
            input_data.at[0, f"Size_{size_var.get()}"] = 1

        if f"Material_{material_var.get()}" in input_data.columns:
            input_data.at[0, f"Material_{material_var.get()}"] = 1

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_scaled)[0]
        messagebox.showinfo("Prediction", f"Predicted Price: ${prediction:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create the Tkinter GUI application
app = tk.Tk()
app.title("Clothing Price Predictor")

# GUI Elements
tk.Label(app, text="Select Brand:").grid(row=0, column=0, padx=10, pady=10)
brand_var = tk.StringVar(app)
brand_var.set("Nike")
tk.OptionMenu(app, brand_var, "Nike", "New Balance", "Puma", "Reebok", "Under Armour").grid(row=0, column=1)

tk.Label(app, text="Select Category:").grid(row=1, column=0, padx=10, pady=10)
category_var = tk.StringVar(app)
category_var.set("Shoes")
tk.OptionMenu(app, category_var, "Shoes", "Sweater", "Jacket", "Jeans").grid(row=1, column=1)

tk.Label(app, text="Select Color:").grid(row=2, column=0, padx=10, pady=10)
color_var = tk.StringVar(app)
color_var.set("Red")
tk.OptionMenu(app, color_var, "Red", "Blue", "Green", "White").grid(row=2, column=1)

tk.Label(app, text="Select Size:").grid(row=3, column=0, padx=10, pady=10)
size_var = tk.StringVar(app)
size_var.set("M")
tk.OptionMenu(app, size_var, "S", "M", "L", "XL", "XS", "XXL").grid(row=3, column=1)

tk.Label(app, text="Select Material:").grid(row=4, column=0, padx=10, pady=10)
material_var = tk.StringVar(app)
material_var.set("Cotton")
tk.OptionMenu(app, material_var, "Cotton", "Silk", "Polyester", "Wool", "Denim", "Nylon").grid(row=4, column=1)

# Predict Button
tk.Button(app, text="Predict Price", command=predict_price_gui).grid(row=5, column=0, columnspan=2, pady=20)

# Start the Tkinter event loop
app.mainloop()
