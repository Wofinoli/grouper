import PySimpleGUI as sg
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import string
import math

import group
import data_process

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
        self.row = 0
        self.col = 0
        self.group_colors = {}

    def run(self):
        plate_win = None#self.make_plate_win()      
        choose_group_win = None
        group_win = None
        start_win = self.make_start_win()
        plot_win = None

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
                elif window == plot_win:
                    plot_win = None

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
                self.plate = data_process.Plate(self.cols, self.rows, self.filename)
                start_win.close()
                start_win = None
                plate_win = self.make_plate_win()
                self.graph = plate_win['graph']
                self.buttons = self.draw_buttons()

            if event == 'Finalize groups':
                plate_win.close()
                plate_win = None
                plot_win = self.make_plot_win()
                self.set_start_cell()
                self.draw_plot()
                self.finalize_groups()

            if event == 'excel':
                plot_win.close()
                plot_win = None
                self.plate.statistics = data_process.get_statistics(self.plate.accepted_fits)
                self.to_excel()


            if event in ['Accept', 'Reject', 'Next', 'Previous', 'Re-fit']:
                plt.close()
                if event == 'Previous':
                    self.prev_cell()
                elif event == 'Re-fit':
                    print("Reffiting")
                else:
                    if event == 'Accept':
                        self.accept_cell()
                    elif event == 'Reject':
                        self.reject_cell()

                    self.next_cell()

                self.draw_plot()

    def make_plate_win(self):
        menu_def = [['Groups', ['New group', 'Edit group', 'Finalize groups']],
                    ['Options',['Close']],]
        layout = [[sg.Menu(menu_def)],      
               [self.make_plate()],      
               ]      

        return sg.Window('Plate', layout, finalize=True)

    def make_plate(self):
        graph = sg.Graph(canvas_size=self.size, graph_bottom_left=(0,0), graph_top_right=self.size, background_color='white', key='graph',
                            enable_events=True, drag_submits=True)
        return graph

    def make_group_win(self):
        default_name = "G{:02d}".format(1+len(self.groups))
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

        return sg.Window('Choose file', layout, size=(200,75), finalize = True)

    def make_plot_win(self):
        layout = [[sg.Button("Accept"), sg.Button("Reject")],
                  [sg.Button("Previous"), sg.Button("Next")],
                  [sg.Button("Write to Excel", key="excel")],]

        return sg.Window('Plots', layout, size=(300,110), finalize = True)

    def draw_plot(self, p0=[70,0.4,0.6,1]):
        plate = self.plate
        sodium_sweeps = self.plate.sodium_sweeps
        potentials = self.plate.potentials
        row_names = self.plate.row_names

        fig = plt.figure(figsize=(6.4*2, 4.8)) # Based off matplotlib default size
        cell_ax = fig.add_subplot(121)
        cumul_ax = fig.add_subplot(122)
        plt.ion()

        fig.show()
        fig.canvas.draw()
        
        index = 1
        write_col = 1
        self.ydata = []
        self.title = row_names[self.row] + "{:02d}".format(self.col + 1)
        success = True
        for sweep in sodium_sweeps:
            self.ydata.append(sweep.iloc[self.row,self.col])
           
        x_range = np.arange(min(potentials), max(potentials), 0.01)
        try:
            #bounds = ((65,-np.inf,-np.inf,-np.inf),(85,np.inf,np.inf,np.inf))
            self.popt, pcov = curve_fit(data_process.func_IV_NA, potentials, self.ydata, p0=p0, maxfev=100000)
        except:
            print("Fit failed for " + self.title)
            index = 0 if pd.isnull(self.plate.failed.index.max()) else self.plate.failed.index.max() + 1
            self.plate.failed.loc[index] = self.title
            #plt.close()
            success = False
    
        label = 'fit: $v_{rev}=%5.3f, g_{max}=%5.3f, v_{0.5}=%5.3f, v_{slope}=%5.3f$' % tuple(self.popt)

        cell_ax.clear()
        cell_ax.set_title(self.active_group.name + "_" + self.title)
        cell_ax.set_xlabel("Potential (mV)")
        cell_ax.set_ylabel("Current (pA)")
        cell_ax.grid()

        if success:
            cell_ax.plot(potentials, self.ydata, 'b.', label="data")
            cell_ax.plot(x_range, data_process.func_IV_NA(x_range, *self.popt), 'r-', label=label)
            cell_ax.legend()
        

        self.get_cumul(self.active_group.name)
        color = self.active_group.color
        custom_legend = [Line2D([0],[0], color=color, lw=4),
                         Line2D([0],[0], color='black', lw=4)]
        for fit in self.get_cumul(self.active_group.name):
            cumul_ax.plot(x_range, data_process.func_IV_NA(x_range, *fit), '-', color=color)

        for fit in self.get_rejected_fits(self.active_group.name):
            cumul_ax.plot(x_range, data_process.func_IV_NA(x_range, *fit), '-', color='black')

        cumul_ax.grid()
        cumul_ax.set_title(self.active_group.name + " Cumulative")
        cumul_ax.set_xlabel("Potential (mV)")
        cumul_ax.set_ylabel("Current (pA)")
        cumul_ax.legend(custom_legend, ['Accepted', 'Rejected'])

        fig.canvas.draw()

    def get_cumul(self, group):
        current = self.plate.accepted_fits[self.plate.accepted_fits['Cell'].str.startswith(group)]
        return current.loc[:, current.columns != 'Cell'].to_numpy()

    def get_rejected_fits(self, group):
        current = self.plate.rejected_fits[self.plate.rejected_fits['Cell'].str.startswith(group)]
        return current.loc[:, current.columns != 'Cell'].to_numpy()

    def next_cell(self):
        #if self.row == self.rows - 1 and self.col == self.cols - 1:
        #    return

        #if self.col < self.cols - 1:
        #    self.col += 1
        #else:
        #    self.col = 0
        #    self.row += 1

        max_rows, max_cols = self.active_group.coordinates[self.pair][1]
        min_rows, min_cols = self.active_group.coordinates[self.pair][0]
        if(self.col < max_cols):
            self.col += 1
        elif(self.row < max_rows):
            self.col = min_cols
            self.row += 1 
        else:
            if(len(self.active_group.coordinates) > self.pair + 1):
                self.pair += 1
            else:
                next_group = None
                can_select = False
                for _, group in self.groups.items():
                    if can_select:
                        self.active_group = group
                        break

                    if(self.active_group == group):
                        print("Curent group found")
                        can_select = True

                if next_group == None:
                    return False


                self.pair = 0
            self.row, self.col = self.active_group.coordinates[self.pair][0]

        return True

    def prev_cell(self):
        max_rows, max_cols = self.active_group.coordinates[self.pair][1]
        min_rows, min_cols = self.active_group.coordinates[self.pair][0]
        if(self.col > min_cols):
            self.col -= 1
        elif(self.row > min_rows):
            self.col = max_cols
            self.row -= 1 
        else:
            if(len(self.active_group.coordinates) > self.pair + 1):
                self.pair += 1
            else:
                prev_group = None
                can_select = False
                for _, group in reversed(self.groups.items()):
                    if can_select:
                        self.active_group = group
                        break

                    if(self.active_group == group):
                        print("Curent group found")
                        can_select = True

                if prev_group == None:
                    return False

                self.pair = 0
            self.row, self.col = self.active_group.coordinates[self.pair][0]

        return True

    def accept_cell(self):
        index = 0 if pd.isnull(self.plate.accepted_fits.index.max()) else self.plate.accepted_fits.index.max() + 1
        title = self.title
        for name, group in self.groups.items():
            coords = group.coordinates
            color = group.color

            for coord in coords:
                start_row, start_col = coord[0]
                end_row, end_col = coord[1]

                if self.row >= start_row and self.row <= end_row and self.col >= start_col and self.col <= end_col:
                    title = name + "_" + title

        if title in self.plate.accepted_fits['Cell'].values:
            return

        self.plate.accepted_fits.loc[index, 'Cell'] =  title
        self.plate.accepted_fits.loc[index, 'v_rev':'v_slope'] = self.popt
        self.plate.source[title] = self.ydata

        driving_force = self.plate.source['Potential'] - self.plate.accepted_fits.loc[index, 'v_rev']
        conductance = self.plate.source[title] / driving_force
        g_max = conductance.max()
        normalized_g = conductance / g_max

        title += "_G"
        self.plate.source[title] = normalized_g


    def reject_cell(self):
        index = 0 if pd.isnull(self.plate.rejected_fits.index.max()) else self.plate.rejected_fits.index.max() + 1
        title = self.title
        for name, group in self.groups.items():
            coords = group.coordinates
            color = group.color

            for coord in coords:
                start_row, start_col = coord[0]
                end_row, end_col = coord[1]

                if self.row >= start_row and self.row <= end_row and self.col >= start_col and self.col <= end_col:
                    title = name + "_" + title

        if title in self.plate.rejected_fits['Cell'].values:
            return

        self.plate.rejected_fits.loc[index, 'Cell'] =  title
        self.plate.rejected_fits.loc[index, 'v_rev':'v_slope'] = self.popt
        
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

    def set_start_cell(self):
        if(len(self.groups) > 0):
            self.group_iter = iter(self.groups.items())
            self.pair = 0
            _, self.active_group = next(self.group_iter)
            start = self.active_group.coordinates[self.pair]
            self.row, self.col = start[0]

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

    def finalize_groups(self):
        for _, group in self.groups.items():
            coords = group.coordinates
            color = group.color

            for coord in coords:
                start_row, start_col = coord[0]
                end_row, end_col = coord[1]

                for row in range(start_row, end_row+1):
                    for col in range(start_col, end_col+1):
                        self.group_colors[(row,col)] = color

    def get_group_statistics(self):
        rows = len(self.groups) * self.plate.accepted_fits.shape[1]

        statistics = pd.DataFrame(index=np.arange(0, rows), columns=["Group", "Variable","Mean","Median","Std. Dev","Std. Err","Max","Min","N"])

        index = 0
        for name in self.groups.keys():
            current = self.plate.accepted_fits[self.plate.accepted_fits['Cell'].str.startswith(name)]

            statistics.iloc[index]['Group'] = name
            for label, content in current.items():
                if(label == "Cell"):
                    continue
                

                statistics.iloc[index]["Variable"] = label
                statistics.iloc[index]["Mean"] = np.mean(content)
                statistics.iloc[index]["Median"] = np.median(content)
                statistics.iloc[index]["Std. Dev"] = np.std(content, ddof=1)
                if len(current) > 1:
                    statistics.iloc[index]["Std. Err"] = stats.sem(content, axis=None, nan_policy="omit")
                statistics.iloc[index]["Max"] = np.max(content)
                statistics.iloc[index]["Min"] = np.min(content)
                statistics.iloc[index]["N"] = np.sum(content.count())
                index += 1
         
        return statistics


    def to_excel(self):
        last_sep = self.filename.rindex("/") + 1
        path = os.path.join(os.getcwd(), "output")
        if not os.path.exists(path):
            os.makedirs(path)

        filename = "RESULT_" + self.filename[last_sep:-4] + ".xlsx"
        filename = os.path.join(path, filename)
        with pd.ExcelWriter(filename) as writer: 
            name = "Result"
            workbook = writer.book
            worksheet = workbook.add_worksheet(name)
            writer.sheets[name] = worksheet

            startrow, startcol = 0, 0

            self.plate.accepted_fits.to_excel(writer,sheet_name=name,startrow=startrow, startcol=startcol)
            self.format_sheet(writer, workbook, worksheet, startrow, startcol, self.plate.accepted_fits, 'Cell')

            startcol = self.plate.accepted_fits.shape[1] + 2
            startrow = self.plate.statistics.shape[0] + 2
            self.plate.statistics.to_excel(writer,sheet_name=name,startrow=0, startcol=startcol)
            self.plate.failed.to_excel(writer,sheet_name=name, startrow=startrow, startcol=startcol)
            
            startrow += self.plate.failed.shape[0] + 2
            group_statistics = self.get_group_statistics()
            group_statistics.to_excel(writer, sheet_name=name, startrow=startrow, startcol=startcol)
            self.format_sheet(writer,workbook,worksheet,startrow,startcol,group_statistics,'Group')


            source_sheet = workbook.add_worksheet('source')
            writer.sheets['source'] = source_sheet

            source_width, source_height = self.plate.source.shape
            current_cols = [not name.endswith('_G') for name in self.plate.source.columns]
            self.plate.source.loc[:, current_cols].to_excel(writer, sheet_name='source', startrow = 0, startcol=0)

            conductance_cols = [not col for col in current_cols]
            conductance_cols[0] = True

            self.plate.source.loc[:, conductance_cols].to_excel(writer, sheet_name='source', startrow = source_height, startcol=0)
            
    def format_sheet(self, writer, workbook, worksheet, startrow, startcol, frame, key):
        if key == 'Cell':
            for index, cell in enumerate(frame[key]):
                name = cell[-3:]
                coords = self.cell_to_pair(name)
                if coords in self.group_colors:
                    color = self.group_colors[coords]
                    cell_format = workbook.add_format({'bg_color': color})

                    endcol = frame.shape[1] + startcol

                    worksheet.conditional_format(startrow + index + 1, startcol + 1, startrow + index + 1, endcol,
                            {'type': 'cell',
                             'criteria': '!=',
                             'value': '""',
                             'format': cell_format})
        elif key == 'Group':
            for name, group in self.groups.items():
                color = group.color
                cell_format = workbook.add_format({'bg_color': color})
                endcol = frame.shape[1] + startcol
                endrow = startrow + 4 #Number of variables, TODO: refactor for generality
                worksheet.conditional_format(startrow + 1, startcol + 1, endrow, endcol,
                        {'type': 'cell',
                         'criteria': '!=',
                         'value': '""',
                         'format': cell_format})

                startrow = endrow
 
    def cell_to_pair(self, cell):
        row = ord(cell[0]) - ord('A')
        col = int(cell[-2:]) - 1

        return (row, col)
