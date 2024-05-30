from tkinter import *
from try_chatting import *
from sentiment import * 
 

root = Tk()
root.title('CHAT')
root.geometry("700x500+50+50")
root.config(bg="skyblue") 



def button_pressed():
  
    
    text = write_txt.get(1.0,END)
    #ELIMINAR SALTOS DE LINEAS
    text = text.strip()
    sentiment = analyze_sentiment(text)
    
    print("-------------------USER ENTRY-------------------")
    print("TEXTO ESCRITO:" + repr(text))
    print("-------------------USER ENTRY-------------------")
    print("-------------------USER SENTIMENT---------------")
    print("SENTIMENT ANALIZADO:" + sentiment)
    print("-------------------USER SENTIMENT---------------")
    respuesta = update_chat(text);
    print("-------------------IA RESPONSE------------------")
    print("TEXTO RESPUESTA:"+respuesta)
    print("-------------------IA RESPONSE------------------")
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("------------------------------------------------")
    print("------------------------------------------------")
    chat_txt.configure(state='normal')
    chat_txt.insert(1.0,'Tú:\n' + text + '\n')
    chat_txt.insert(1.0,'IA:\n' + respuesta + '\n')
    chat_txt.configure(state='disabled')
    
    write_txt.delete(1.0,END)

def button_pressed_event(event):
    button_pressed()

def buton_pressed_clicked():
    button_pressed()


root.bind('<Return>',button_pressed_event)

chat_frame = Frame(root,width=680,height=340)
chat_frame.grid(row=0,column=0,padx=10,pady=10)
chat_txt = Text(chat_frame)
chat_txt.grid(row=0,column=0)
chat_txt.place(width=680,height=340)
chat_txt.configure(state='disabled')

#*frame que contiene la zona de escritua y el boton send
user_frame = Frame(root, width=680,height=100)
user_frame.config(bg='blue')
user_frame.grid(row=1,column=0,padx=10,pady=10)

#*zona de escritura para el usuario
write_frame = Frame(user_frame,width=600,height=100)
write_frame.grid(row=0,column=0,padx=10,pady=0)
write_txt = Text(user_frame)
write_txt.grid(row=0,column=0)
write_txt.place(width=600,height=100)


#* frame boton send
button_frame = Frame(user_frame, width=90,height=90)
button_frame.grid(row=0,column=1,padx=10,pady=10)
#*button send
send_button = Button(button_frame,text='Send',command=buton_pressed_clicked)
send_button.pack()



root.mainloop()