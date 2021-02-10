import PySimpleGUI as sg
import string
import group
import math

class GUI:

    cols = 24
    rows = 16
    height = 600

    def __init__(self):
        self.ratio = self.cols / self.rows
        self.width = self.height * self.ratio
        self.size = (self.width, self.height)
        self.padding = (self.cols * self.rows)/100
        self.offset = 5*self.padding
        self.button_size = self.get_button_size()
        self.groups = {}
        self.active_group = None

    def run(self):
        plate_win = None#self.make_plate_win()      
        choose_group_win = None
        group_win = None
        start_win = self.make_start_win()

        hasSelected = False
        start_row, start_col, end_row, end_col = -1,-1,-1,-1
        while True:      
            window, event, values = sg.read_all_windows()      

            if window == sg.WINDOW_CLOSED:
                break

            if event == sg.WIN_CLOSED:      
                window.close()
                if window == plate_win:
                    plate_win = None
                elif window == group_win:
                    group_win = None
                elif window == choose_group_win:
                    choose_group_win = None
                elif window == start_win:
                    start_win = None

            if self.active_group and event == "graph" and not hasSelected:
                start_row, start_col = self.cords_to_tile(values['graph'])
                hasSelected = True
            if self.active_group and event == "graph+UP" and hasSelected:
                end_row, end_col = self.cords_to_tile(values['graph'])
                self.fill_cells(start_row, start_col, end_row, end_col)
                hasSelected = False

            if event == 'New group':
                if not group_win:
                    group_win = self.make_group_win()
                    group_win.move(plate_win.current_location()[0], plate_win.current_location()[1])

            if event == 'Create group':
                color = values['color_in']
                name = values['group_name']
                self.add_group(color, name)
                group_win.close()
                group_win = None

            if len(self.groups) > 0 and event == 'Edit group':
                if not choose_group_win:
                    choose_group_win = self.make_choose_group_win()
                    choose_group_win.move(plate_win.current_location()[0], plate_win.current_location()[1])

            if event == 'Close':
                plate_win.close()

            if event == 'choose_group':
                self.active_group = self.groups[values['choose_group'][0]]
                choose_group_win.close()
                choose_group_win = None

            if event == 'file_choose':
                self.filename = values['file_choose']
                start_win.close()
                start_win = None
                plate_win = self.make_plate_win()
                self.graph = plate_win['graph']
                self.buttons = self.draw_buttons()

    def make_plate_win(self):
        menu_def = [['Groups', ['New group', 'Edit group', 'Finalize groups']],
                    ['Options',['Close']],]
        layout = [[sg.Menu(menu_def)],      
               [self.make_plate()],      
               ]      

        return sg.Window('Plate', layout, finalize=True)


    def make_group_win(self):
        default_name = "Group{:02d}".format(1+len(self.groups))
        layout = [[sg.Text('New Group')],
                [sg.Text('Color: '), sg.Input(key='color_in'), sg.ColorChooserButton('Choose color', target='color_in')],
              [sg.Text('Name: '), sg.Input(default_text=default_name, key='group_name')],
              [sg.Button('Create group')]]
        return sg.Window('New Group', layout, finalize=True)

    def make_choose_group_win(self):
        group_names = [name for name, group in self.groups.items()]
        layout = [[sg.Text("Choose group to edit")],
                [sg.Listbox(values=group_names, key='choose_group', enable_events=True,
                    size=(15,10), select_mode="LISTBOX_SELECT_MODE_SINGLE")]]

        return sg.Window('Choose Group', layout, finalize=True)

    def make_start_win(self):
        layout= [[sg.Text("Choose file",  size=(15,1))],
                 [sg.In(visible=False, enable_events=True, key='file_choose'), sg.FileBrowse()]]

        return sg.Window('Choose file', layout, finalize = True)

    def make_plate(self):
        graph = sg.Graph(canvas_size=self.size, graph_bottom_left=(0,0), graph_top_right=self.size, background_color='white', key='graph',
                            enable_events=True, drag_submits=True)
        return graph

    def get_button_size(self):
        height = self.height - self.padding - self.offset
        width = self.width - self.padding - self.offset
        return math.sqrt((width * height) / (self.cols*self.rows)) - self.padding

    def draw_buttons(self):
        buttons = []

        row_names = list(string.ascii_uppercase[:self.rows])
        y = self.button_size / 2
        for letter in row_names[::-1]:
            self.graph.draw_text(letter, (self.offset / 2, y))
            y += self.button_size + self.padding

        x = self.button_size + self.padding
        for num in range(self.cols):
            self.graph.draw_text(num + 1, (x, self.height - self.offset/2))
            x += self.button_size + self.padding

        bottom = self.padding
        for _ in range(self.rows):
            left = self.padding + self.offset
            row = []
            for _ in range(self.cols):
                bottom_left = (left, bottom)
                top_right = (left + self.button_size, bottom + self.button_size)
                row.append(self.graph.DrawRectangle(bottom_left, top_right, fill_color='grey', line_color='black'))
                left += self.button_size + self.padding

            buttons.insert(0, row)
            bottom += self.button_size + self.padding

        return buttons

    def fill_cells(self, start_row, start_col, end_row, end_col):
        if start_row > end_row:
            start_row, end_row = end_row, start_row

        if start_col > end_col:
            start_col, end_col = end_col, start_col

        for i in range(start_row, end_row+1):
            for j in range(start_col, end_col+1):
                self.change_color(i, j)

        self.active_group.coordinates.append( [(start_row, start_col), (end_row, end_col)] )

    def change_color(self, row, col):
        self.graph.TKCanvas.itemconfig(self.buttons[row][col], fill=self.active_group.color)

    def cords_to_tile(self, cords):
        x = cords[0] - self.padding - self.offset
        y = cords[1] - self.padding

        row = int(self.rows - 1 - y // (self.button_size + self.padding))
        col = int(x // (self.button_size+self.padding))

        return (row, col)

    def add_group(self, color, name):
        new_group = group.Group(color, name)
        self.groups[name] = new_group
        self.active_group = new_group
