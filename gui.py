import PySimpleGUI as sg
import math

def main():
    sg.theme('DarkAmber')

    NUM_COLS = 6#24
    NUM_ROWS = 4#16
    CANVAS_HEIGHT = 400

    layout = [      
               [make_plate(NUM_COLS, NUM_ROWS, CANVAS_HEIGHT)],      
           ]      

    window = sg.Window('Graph test', layout)      
    window.Finalize()      

    graph = window['graph']
    buttons = draw_buttons(graph, NUM_COLS, NUM_ROWS, CANVAS_HEIGHT)

    hasSelected = False
    while True:      
        event, values = window.read()      
        if event == sg.WIN_CLOSED:      
            break      

        if event == "graph":
            #graph.TKCanvas.itemconfig(buttons[0][0], fill="Blue")
            change_color(values['graph'], NUM_COLS, NUM_ROWS, CANVAS_HEIGHT, buttons)
        if event == "graph+UP":
            print("mouse up")

    window.close()

def make_plate(cols, rows, height):
    ratio = cols/rows
    width = height * ratio
    size = (width, height)
    graph = sg.Graph(canvas_size=size, graph_bottom_left=(0,0), graph_top_right=size, background_color='white', key='graph',
                        enable_events=True, drag_submits=True)
    return graph

def draw_buttons(graph, cols, rows, height):
    ratio = cols/rows
    width = height * ratio
    padding = cols*rows / 100
    button_size = math.sqrt((height - padding ) * (width - padding) / (cols*rows)) - padding

    buttons = []

    bottom = padding
    for _ in range(rows):
        left = padding
        row = []
        for _ in range(cols):
            bottom_left = (left, bottom)
            top_right = (left + button_size, bottom + button_size)
            row.append(graph.DrawRectangle(bottom_left, top_right, fill_color='grey', line_color='black'))
            left += button_size + padding

        buttons.insert(0, row)
        bottom += button_size + padding

    return buttons

def change_color(cords, cols, rows, height, buttons):
    cords_to_tile(cords, cols, rows, height)

def cords_to_tile(cords, cols, rows, height):
    ratio = cols/rows
    width = height * ratio
    padding = cols*rows / 100
    button_size = math.sqrt((height - padding ) * (width - padding) / (cols*rows)) - padding

    x = cords[0]
    y = cords[1]
    
    col = int(x // button_size)
    row = int(rows - 1 - y // button_size)
    print(cords)
    print(row, col)


main()
