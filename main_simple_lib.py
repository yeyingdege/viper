# General imports and variables, as well as config
import ast
import math
import sys
import time
import requests
import torch.multiprocessing as mp
from joblib import Memory
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.prompt import Prompt
from rich.syntax import Syntax
import traceback
from functools import partial
from rich import print
from rich.markup import escape as rich_escape

from IPython.display import update_display, clear_output, display
from PIL import Image
import matplotlib.pyplot as plt
from configs import config
from utils import extract_code, extract_code1, extract_code_deepseek

from IPython.display import update_display, clear_output
from IPython.core.display import HTML

cache = Memory('cache/' if config.use_cache else None, verbose=0)

mp.set_start_method('spawn', force=True)
from vision_processes import forward, finish_all_consumers  # This import loads all the models. May take a while
from image_patch import *
from video_segment import *
from datasets.my_dataset import MyDataset

console = Console(highlight=False, force_terminal=False)

time_wait_between_lines = 0.5


def inject_saver(code, show_intermediate_steps, syntax=None, time_wait_between_lines=None, console=None):
    injected_function_name = 'show_all'
    if injected_function_name in code:
        return code
    code = code.split("\n")
    newcode = []
    for n, codeline in enumerate(code):
        codeline, indent = split_codeline_and_indent_level(codeline)

        if codeline.startswith('#') or codeline == '':  # this will cause issues if you have lots of comment lines
            continue
        if '#' in codeline:
            codeline = codeline.split('#')[0]

        thing_to_show, code_type = get_thing_to_show_codetype(codeline)

        if code_type in ('assign', 'append', 'if', 'return', 'for', 'sort', 'add'):
            if '\'' in codeline:
                codeline.replace('\'', '\\\'')

            if show_intermediate_steps:
                escape_thing = lambda x: x.replace("'", "\\'")
                injection_string_format = \
                    lambda \
                        thing: f"{indent}{injected_function_name}(lineno={n},value=({thing}),valuename='{escape_thing(thing)}'," \
                               f"fig=my_fig,console_in=console,time_wait_between_lines=time_wait_between_lines); " \
                               f"CodexAtLine({n},syntax=syntax,time_wait_between_lines=time_wait_between_lines)"
            else:
                injection_string_format = lambda thing: f"{indent}CodexAtLine({n},syntax=syntax," \
                                                        f"time_wait_between_lines=time_wait_between_lines)"

            extension_list = []
            if isinstance(thing_to_show, list):
                injection_string_list = [injection_string_format(f"{thing}") for thing in thing_to_show]
                extension_list.extend(injection_string_list)
            elif code_type == 'for':
                injection_string = injection_string_format(f"{thing_to_show}")
                injection_string = " " * 4 + injection_string
                extension_list.append(injection_string)
            else:
                extension_list.append(injection_string_format(f"{thing_to_show}"))

            if code_type in ('if', 'return'):
                extension_list = extension_list + [f"{indent}{codeline}"]
            else:
                extension_list = [f"{indent}{codeline}"] + extension_list

            newcode.extend(extension_list)

        elif code_type == 'elif_else':
            newcode.append(f"{indent}{codeline}")
        else:
            newcode.append(f"{indent}{codeline}")
    return "\n".join(newcode)


def get_thing_to_show_codetype(codeline):
    # can output either a list of things to show, or a single thing to show
    things_to_show = []
    if codeline.startswith("if"):
        condition, rest = codeline[3:].split(":", 1)
        codeline = f"if {condition}:{rest}"
        code_type = "if"

        operators = ['==', '!=', '>=', '<=', '>', '<']
        things_to_show = []
        for op in operators:
            if op in condition:
                things_to_show = [x.strip() for x in condition.split(op)]
                # print(things_to_show)
                break
        # things_to_show.append(thing_to_show)
        thing_to_show = things_to_show + [condition.strip()]

    elif codeline.startswith("for"):
        code_type = 'for'
        thing_to_show = codeline.split("for ")[1].split(" in ")[0]

    elif codeline.startswith("return"):
        thing_to_show = codeline.split("return ")[1]
        code_type = 'return'

    elif ' = ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' = ')[0]
    elif ' += ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' += ')[0]
    elif ' -= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' -= ')[0]
    elif ' *= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' *= ')[0]
    elif ' /= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' /= ')[0]

    elif '.append(' in codeline:
        code_type = 'append'
        thing_to_show = codeline.split('.append(')[0] + '[-1]'
    elif '.add(' in codeline:
        code_type = 'add'
        thing_to_show = codeline.split('.add(')[0]

    elif '.sort(' in codeline:
        code_type = 'sort'
        thing_to_show = codeline.split('.sort(')[0]

    elif codeline.startswith("elif") or codeline.startswith("else"):
        thing_to_show = None
        code_type = 'elif_else'
    else:
        thing_to_show = None
        code_type = 'other'

    if isinstance(thing_to_show, list):
        thing_to_show = [thing if not (thing.strip().startswith("'") and thing.strip().endswith("'"))
                         else thing.replace("'", '"') for thing in thing_to_show if thing is not None]
    elif isinstance(thing_to_show, str):
        thing_to_show = thing_to_show if not (thing_to_show.strip().startswith("'") and
                                              thing_to_show.strip().endswith("'")) else thing_to_show.replace("'", '"')
    return thing_to_show, code_type


def split_codeline_and_indent_level(codeline):
    origlen = len(codeline)
    codeline = codeline.lstrip()
    indent = origlen - len(codeline)
    indent = " " * indent
    return codeline, indent


def show_one_image(image, ax):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)
        image = image.squeeze(0).permute(1, 2, 0)
    ax.imshow(image)


def CodexAtLine(lineno, syntax, time_wait_between_lines=1.):
    syntax._stylized_ranges = []
    syntax.stylize_range('on red', (lineno + 1, 0), (lineno + 1, 80))
    time.sleep(time_wait_between_lines)


def show_all(lineno, value, valuename, fig=None, usefig=True, disp=True, console_in=None, time_wait_between_lines=None,
             lastlineno=[-1]):
    time.sleep(0.1)  # to avoid race condition!

    if console_in is None:
        console_in = console

    thing_to_show = value

    if lineno is not None and lineno != lastlineno[0]:
        console_in.rule(f"[bold]Line {lineno}[/bold]", style="chartreuse2")
        lastlineno[0] = lineno  # ugly hack

    if usefig:
        plt.clf()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
    if isinstance(thing_to_show, Image.Image):
        if valuename:
            console_in.print(f'{rich_escape(valuename)} = ')
        show_one_image(thing_to_show, ax)
    elif str(type(thing_to_show)) == "<class 'image_patch.ImagePatch'>":
        if valuename:
            console_in.print(f'{rich_escape(valuename)} = ')
        show_one_image(thing_to_show.cropped_image, ax)
    elif isinstance(thing_to_show, list) or isinstance(thing_to_show, tuple):
        if len(thing_to_show) > 0:
            for i, thing in enumerate(thing_to_show):
                disp_ = disp or i < len(thing_to_show) - 1
                show_all(None, thing, f"{rich_escape(valuename)}[{i}]", fig=fig, disp=disp_, usefig=usefig)
            return
        else:
            console_in.print(f"{rich_escape(valuename)} is empty")
    elif isinstance(thing_to_show, dict):
        if len(thing_to_show) > 0:
            for i, (thing_k, thing_v) in enumerate(thing_to_show.items()):
                disp_ = disp or i < len(thing_to_show) - 1
                show_all(None, thing_v, f"{rich_escape(valuename)}['{thing_k}']", fig=fig, disp=disp_, usefig=usefig)
            return
        else:
            console_in.print(f"{rich_escape(valuename)} is empty")
    else:
        console_in.print(f"{rich_escape(valuename)} = {thing_to_show}")
        if time_wait_between_lines is not None:
            time.sleep(time_wait_between_lines / 2)
        return

    # display small
    if usefig:
        fig.set_size_inches(2, 2)
        if disp:
            display(fig)


def load_image(path):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        image = transforms.ToTensor()(image)
    else:
        image = Image.open(path)
        image = transforms.ToTensor()(image)
    return image


def get_code(query):
    model_name_codex = 'codellama' if config.codex.model == 'codellama' else 'codex'
    code = forward(model_name_codex, prompt=query, input_type="image")
    print(f"code before processing\n{code}")
    if "deepseek" in config.codex.model:
        # chat models give execute_command due to system behaviour
        code = extract_code_deepseek(code)
    else:
        code = extract_code(code)
        code = extract_code1(code)
        code = ast.unparse(ast.parse(code))
    code_for_syntax = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    syntax_1 = Syntax(code_for_syntax, "python", theme="monokai", line_numbers=True, start_line=0)
    console.print(syntax_1)
    code_for_syntax_2 = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    syntax_2 = Syntax(code_for_syntax_2, "python", theme="monokai", line_numbers=True, start_line=0)
    return code, syntax_2


def execute_code(code, im, show_intermediate_steps=True):
    code, syntax = code
    code_line = inject_saver(code, show_intermediate_steps, syntax, time_wait_between_lines, console)

    display(HTML("<style>.output_wrapper, .output {height:auto !important; max-height:1000000px;}</style>"))

    with Live(Padding(syntax, 1), refresh_per_second=10, console=console, auto_refresh=True) as live:
        my_fig = plt.figure(figsize=(4, 4))
        try:
            exec(compile(code_line, 'Codex', 'exec'), globals())
            result = execute_command(im, my_fig, time_wait_between_lines, syntax)  # The code is created in the exec()
        except Exception as e:
            print(f"Encountered error {e} when trying to run with visualizations. Trying from scratch.")
            exec(compile(code, 'Codex', 'exec'), globals())
            result = execute_command(im, my_fig, time_wait_between_lines, syntax)  # The code is created in the exec()

        plt.close(my_fig)

    def is_not_fig(x):
        if x is None:
            return True
        elif isinstance(x, str):
            return True
        elif isinstance(x, float):
            return True
        elif isinstance(x, int):
            return True
        elif isinstance(x, list) or isinstance(x, tuple):
            return all([is_not_fig(xx) for xx in x])
        elif isinstance(x, dict):
            return all([is_not_fig(xx) for xx in x.values()])
        return False

    f = None
    usefig = False
    if not is_not_fig(result):
        f = plt.figure(figsize=(4, 4))
        usefig = True

    console.rule(f"[bold]Final Result[/bold]", style="chartreuse2")
    show_all(None, result, 'Result', fig=f, usefig=usefig, disp=False, console_in=console, time_wait_between_lines=0)
    return result


def show_single_image(im):
    im = Image.fromarray((im.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
    im.copy()
    im.thumbnail((400, 400))
    display(im)


def get_code_video(query, input_type="video", extra_context=""):
    # generate and process code for video QA
    model_name_codex = 'codellama' if config.codex.model == 'codellama' else 'codex'
    code = forward(model_name_codex, prompt=query, input_type=input_type, extra_context=extra_context)
    print(f"code before processing\n{code}")
    if "deepseek" in config.codex.model:
        code = extract_code_deepseek(code)
    else:
        code = extract_code(code)
        code = extract_code1(code)
        code = ast.unparse(ast.parse(code))
    return code


queue_results = None
def run_program(parameters, queues_in_, input_type_, retrying=False):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from video_segment import VideoSegment

    global queue_results

    code, sample_id, image, possible_answers, query = parameters

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query, ' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match):\n' \
                  f'    # Answer is:'
    code = code_header + code

    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        try:
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code 
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e2:
            print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
            return None, code

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # Retry again with fixed code
        new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
                             retrying=True)[0]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, code

