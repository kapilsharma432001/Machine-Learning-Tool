from tkinter import *

root = Tk()


def a_k():
    root.minsize(800,800)
    root.title("RexD ML Tool")
    
    
    buttonFrame = Frame(root,borderwidth = 2, bg = "grey", relief = SUNKEN)
    buttonFrame.pack(side = TOP)
    
    startButton = Button(buttonFrame, text="START", fg="Green", bg = "orange", height = 2, width = 30)
    startButton.pack()
    

root.mainloop()