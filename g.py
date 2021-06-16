from Tkinter import *
import mysql.connector
root = Tk()

scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

listbox = Listbox(root)
listbox.pack()
aa = mysql.connector.connect(host='localhost', port=3306, user="root", passwd="root", db="cardiac")
mm = aa.cursor()
mm.execute("SELECT * FROM cardiac1")


for i in result:
    listbox.insert(END, i)

# bind listbox to scrollbar
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)

mainloop()
