import sys
import time
import os
import pathlib
import zipfile
from hydra.core.hydra_config import HydraConfig

TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T

def progress_bar(current, total, msg=None):

    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def begin_chart(chart_name, x_axis_name,save_path=None):
    if save_path is not None:
        with open(os.path.join(save_path,chart_name + '.tsv'),"w") as fd:
            fd.write(str(x_axis_name)+"\t"+chart_name+"\n")

    print(f'{{"chart":"{chart_name}", "axis": "{x_axis_name}"}}')

def begin_per_epoch_chart(chart_name,save_path=None):
    begin_chart(chart_name, 'Epoch',save_path=save_path)

def add_chart_point(chart_name, x, y,save_path=None):
    if save_path is not None:
        with open(os.path.join(save_path,chart_name + '.tsv'),"a+") as fd:
            fd.write(str(x)+"\t"+str(y)+"\n")
    
    print(f'{{"chart": "{chart_name}", "x":{x}, "y":{y}}}')

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def save_current_code(path: str):
    print(f"Saving current code to {path}")
    project_root = HydraConfig.get().runtime.cwd
    unwanted_dirs = ["venv", f"utils{os.path.sep}__pycache__",
                     "outputs", "results", ".idea", ".git", "runs", f"models{os.path.sep}__pycache__", "data"]
    unwanted_extensions = ["", "txt", "md"]
    with zipfile.ZipFile(os.path.join(path, "files.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(project_root):
            root = root.replace(project_root, "").lstrip(os.path.sep)
            if True in [root.startswith(x) for x in unwanted_dirs]:
                continue
            for file in files:
                if file.split(".")[-1] in unwanted_extensions:
                    continue
                z.write(os.path.join(project_root, root, file), os.path.join(root, file))
