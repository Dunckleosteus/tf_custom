import PySimpleGUI as sg

def drop_down_menu(fields, message):
    choices = fields
    layout = [  [sg.Text(message) ],
                [sg.Listbox(choices,select_mode='extended', size = (20,20), key='-COLOR-', font=("consolas", 15))],
                [sg.Button('Ok')]  ]

    window = sg.Window(message, layout)

    while True:                  # the event loop
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == 'Ok':
            if values['-COLOR-']:    # if something is highlighted in the list
                value = values['-COLOR-'][0]
                break 
    window.close()
    return value 



def browse_files():
    # TODO: implement some sort of data verification 
    path = ""
    sg.theme("DarkTeal2")
    layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],[sg.Button("Submit")]]

    ###Building Window
    window = sg.Window('My File Browser', layout, size=(800,150))
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            break
        elif event == "Submit":
            print(values["-IN-"])
            path = (values["-IN-"])
            break
    window.close()    
    return path
    

def multiple_choice(list, message):
    layout=[[sg.Text(message,size=(20, 8), font=('consolas',15),justification='left')],
            [sg.Listbox(values=list, select_mode='extended', key='fac', font= ('consolas', 15), size=(25, 15))],
            [sg.Button('SAVE', font=('Times New Roman',12)),sg.Button('CANCEL', font=('Times New Roman',12))]]
    #Define Window
    win =sg.Window("press ctrl+click to select multiple",layout)
    #Read  values entered by user
    e,v=win.read() # is the i necessary
    #close first window
    win.close()
    #access the selected value in the list box and add them to a string
    output = []
    for val in v['fac']:
        output.append(val)
    return output