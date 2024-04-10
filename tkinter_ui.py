# The GUI boilerplate code has been created using ChatGPT.
import tkinter as tk


def create_gui(params, callback):
    # Create the main window
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
        root, text="Calculate", command=lambda: submit_form(params, callback)
    )
    submit_button.grid(row=len(params), columnspan=2, pady=10)

    # Start the tkinter event loop
    root.mainloop()


def submit_form(params, callback):
    # This function retrieves the values from the entry widgets and prints them
    try:
        for entry, key in zip(entries, params.keys()):
            params[key] = float(entry.get())
    except ValueError:
        print("Could not parse input fields to floats. Check your input.")
        return
    callback(params)
