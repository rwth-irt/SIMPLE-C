# The GUI boilerplate code has been created using ChatGPT.
import tkinter as tk
from tkinter import filedialog
import json

# GLOBALS
# root: tkinter root
# params: parameter dictionary
# entries: List of tkinter entries (text inputs)


def create_gui(params_, callback):
    # Create the main window
    global root
    global params
    params = params_
    root = tk.Tk()
    root.title("Point selection parameters")

    # Create labels and entry widgets for each input

    global entries
    entries = []

    for i, (label_text, default_value) in enumerate(params.items()):
        label = tk.Label(root, text=label_text)
        label.grid(row=i, column=0, sticky="e")
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entry.insert(0, default_value)  # Insert default value
        entries.append(entry)

    # Create the submit button
    submit_button = tk.Button(
        root, text="Calculate", command=lambda: submit_form(callback)
    )
    submit_button.grid(row=len(params), columnspan=2, pady=10)

    save_button = tk.Button(root, text="Save parameters to file", command=save_params)
    save_button.grid(row=len(params) + 1, column=0, padx=5, pady=10)

    load_button = tk.Button(root, text="Load parameters from file", command=load_params)
    load_button.grid(row=len(params) + 1, column=1, padx=5, pady=10)

    # Start the tkinter event loop
    root.mainloop()


def submit_form(callback):
    # This function retrieves the values from the entry widgets and prints them
    try:
        for entry, key in zip(entries, params.keys()):
            params[key] = float(entry.get())
    except ValueError:
        print("Could not parse input fields to floats. Check your input.")
        return
    callback(params)


def save_params(cwd="./"):
    file_path = filedialog.asksaveasfilename(initialdir=cwd, title="Select a File")
    try:
        with open(file_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved to {file_path}")
    except IOError:
        print("Could not save parameters")


def load_params(cwd="./"):
    global params
    file_path = filedialog.askopenfilename(initialdir=cwd, title="Select a File")
    try:
        with open(file_path, "r") as f:
            params = json.load(f)
        # update gui. No checks whatsoever...
        for val, entry in zip(params.values(), entries):
            entry.delete(0, tk.END)
            entry.insert(0, val)
        print(f"Parameters read from {file_path}")
    except IOError:
        print("Could not read parameters")
