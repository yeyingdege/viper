from main_simple_lib import *

im = load_image('https://viper.cs.columbia.edu/static/images/kids_muffins.jpg')
query = 'How many muffins can each kid have for it to be fair?'
# im = load_image('./data/qa3_nextStep_38193_f7.png')
# query = 'What might the next step be? select one from options:\n(1) wipe the dish\n(2) install the new bulb\n(3) strip the insulation\n(4) twist the paperclips by hands\n(5) use the needle to open the SIM card slot\nReturn only the index of the correct answer (e.g. 1, 2, 3, 4, or 5).'
# im = load_image('./data/qa18_TaskSameToolSamePurpose_17265_f3.png')
# query = 'What is the other task that use the tool in this video for the same purpose? select one from options:\n(1) Arc Weld\n(2) Make Pickles\n(3) Set Up A Hamster Cage\n(4) Wash Hair\n(5) Make Homemade Ice Cream\nReturn only the index of the correct answer (e.g. 1, 2, 3, 4, or 5).'

show_single_image(im)
code = get_code(query)
# print('code\n', code)
execute_code(code, im, show_intermediate_steps=True)
