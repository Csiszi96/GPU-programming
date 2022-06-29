
from matplotlib import pyplot
import numpy
import PySimpleGUI as sg

length = 0
width = 0

sg.theme('DarkBlack')
layout = [
    [sg.Text('Simulation parameters:')],
    [sg.Text('length (x): '), sg.Input(length, key='x')],
    [sg.Text('width (y): '), sg.Input(width, key='y')]
]

vec =numpy.random.rand(50,50)


pyplot.imshow(vec, cmap='hot', interpolation='nearest')
pyplot.show()



############################################################################################

# import PySimpleGUI as sg
# import time


# sg.theme('DarkBlack')   # Add a touch of color
# # All the stuff inside your window.

# layout = [  [sg.Text('ENTER DESIRED AMOUNT OF ROLLS')],
#             [sg.Slider(range=(5, 1000), orientation='h', default_value=5)],
#             [sg.Button('Start'), sg.Button('Stop')],
#             [sg.Button('Results'),sg.Button('Cancel')] ]

# # Create the Window
# window = sg.Window('Dice Tester', layout).Finalize()
# #window.maximize()

# # Event Loop to process "events" and get the "values" of the inputs
# while True:
#     event, values = window.read()
#     if event in (None,'Cancel'):   # if user closes window or clicks cancel
#         print('Closing GUI')
#         break
#     elif event in ('Start'):  # if user clicks start button
#         Confirm = sg.popup_yes_no('Please Confirm Your Choice')
#         if Confirm == 'Yes':
#             window['Start'].update(disabled = True)
#             print('Starting Test:', values, 'Selected number of rolls')
#             for x in range(1, int(values[0])):
#                 if event in ('Stop'):  # if user clicks Stop button
#                     print('Kill Power to the solenoids')
#                     break

#                 event, values = window.read(timeout=1000)
#                 print(x)

#                 if int(x) == int(values[0]):
#                     break
#             window['Start'].update(disabled = False)
# window.close()

############################################################################################

# import PySimpleGUI as sg
# import numpy as np
# import matplotlib.pyplot as plt

# # Note the matplot tk canvas import
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# # VARS CONSTS:
# _VARS = {'window': False}


# # Helper method to draw figure from PysimpleGUI Demo:
# # https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Pyplot_Bar_Chart.py

# def draw_figure(canvas, figure):
#     figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
#     figure_canvas_agg.draw()
#     figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
#     return figure_canvas_agg

# # \\  -------- PYSIMPLEGUI -------- //


# AppFont = 'Any 16'
# sg.theme('LightGrey')

# layout = [[sg.Canvas(key='figCanvas')],
#           [sg.Button('Exit', font=AppFont)]]
# _VARS['window'] = sg.Window('Such Window',
#                             layout,
#                             finalize=True,
#                             resizable=True,
#                             element_justification="right")

# # \\  -------- PYSIMPLEGUI -------- //


# # \\  -------- PYPLOT -------- //

# # Make synthetic data
# dataSize = 1000
# xData = np.random.randint(100, size=dataSize)
# yData = np.linspace(0, dataSize, num=dataSize, dtype=int)
# # make fig and plot
# fig = plt.figure()
# plt.plot(xData, yData, '.k')
# # Instead of plt.show
# draw_figure(_VARS['window']['figCanvas'].TKCanvas, fig)

# # \\  -------- PYPLOT -------- //

# # MAIN LOOP
# while True:
#     event, values = _VARS['window'].read(timeout=200)
#     if event == sg.WIN_CLOSED or event == 'Exit':
#         break
# _VARS['window'].close()

############################################################################################

# import PySimpleGUI as sg

# sg.theme('DarkAmber')   # Add a touch of color
# # All the stuff inside your window.
# layout = [  [sg.Text('Some text on Row 1')],
#             [sg.Text('Enter something on Row 2'), sg.InputText()],
#             [sg.Button('Ok'), sg.Button('Cancel')] ]

# # Create the Window
# window = sg.Window('Window Title', layout)

# _VARS = {
#     'window': sg.Window(
#         'Such Window',
#         layout,
#         finalize=True,
#         resizable=True,
#         element_justification="right"
#     )
# }

# # Event Loop to process "events" and get the "values" of the inputs
# while True:
#     event, values = window.read()
#     if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
#         break
#     print('You entered ', values[0])

# window.close()

############################################################################################

# import random
# import time

# from matplotlib import pyplot as plt
# from matplotlib import animation


# class RegrMagic(object):
#     """Mock for function Regr_magic()
#     """
#     def __init__(self):
#         self.x = 0
#     def __call__(self):
#         time.sleep(random.random())
#         self.x += 1
#         return self.x, random.random()

# regr_magic = RegrMagic()

# def frames():
#     while True:
#         yield regr_magic()

# fig = plt.figure()

# x = []
# y = []
# def animate(args):
#     x.append(args[0])
#     y.append(args[1])
#     return plt.plot(x, y, color='g')


# anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000)
# plt.show()