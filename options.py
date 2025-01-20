from tkinter import Tk, messagebox, simpledialog


def choose_input():
    global input_source
    root = Tk()
    root.withdraw()
    input_source = simpledialog.askstring(
        "Choose Input",
        "Select the input source:\n1 - File\n2 - RTSP/Stream",
        parent=root,
    )
    if input_source not in ["1", "2"]:
        messagebox.showerror("Error", "Invalid input source selected. Exiting...")
        exit()
    return "File" if input_source == "1" else "RTSP/Stream"
    
# Function to choose the mode: Dwelling Time or Speed and Track
def choose_mode():
    global mode
    root = Tk()
    root.withdraw()
    mode = simpledialog.askstring(
        "Choose Mode",
        "Select the mode:\n1 - Dwelling Time\n2 - Speed and Track",
        parent=root,
    )
    if mode not in ["1", "2"]:
        messagebox.showerror("Error", "Invalid mode selected. Exiting...")
        exit()
    # mode = "Dwelling Time" if mode == "1" else "Speed and Track"
    return "Dwelling Time" if mode == "1" else "Speed and Track"
    
    