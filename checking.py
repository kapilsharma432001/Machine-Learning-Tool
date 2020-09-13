
from tkinter import *
root = Tk()
def regression():
    pass
def classification():
    pass
def clustering():
    pass

Button(root,text = "Regression",bg = "linen",fg = "black",height=2,width = 30,command = regression).grid(row=10,column=1)
Button(root,text = "Classification",bg = "linen",fg = "black",height=2,width = 30,command = classification).grid(row=10,column=3)
Button(root,text = "Clustering",bg = "linen",fg = "black",height=2,width = 30,command = clustering).grid(row=10,column=5)
                            
root.mainloop()
