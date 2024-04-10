# The GUI boilerplate code has been created using ChatGPT. 
import tkinter as tk


def submit_form():
    # This function retrieves the values from the entry widgets and prints them
    for entry in entries:
        print(entry.get())


# Create the main window
root = tk.Tk()
root.title("Simple Form")

# Create labels and entry widgets for each input
labels = ["Name:", "Email:", "Phone:"]
entries = []

default_values = ["John Doe", "john@example.com", "1234567890"]

for i, (label_text, default_value) in enumerate(zip(labels, default_values)):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, sticky="e")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entry.insert(0, default_value)  # Insert default value
    entries.append(entry)

# Create the submit button
submit_button = tk.Button(root, text="Submit", command=submit_form)
submit_button.grid(row=len(labels), columnspan=2, pady=10)

# Start the tkinter event loop
root.mainloop()
