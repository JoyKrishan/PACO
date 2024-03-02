import re

def get_diff_files(patch,type):
    with open(patch, 'r') as file:
        lines = ''
        flag = True
        for line in file:
            line = line.strip()
            if '*/' in line: # if multiline comments at the start of file, we skip
                flag = True
                continue
            if flag == False:
                continue
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                elif '/*' in line: # if multiline comments after hunk indicators, we skip
                    flag = False
                    continue
                elif type == 'buggy':
                    if line.startswith('---'):
                        line = line.split(' ')[1]
                        lines += line.strip() + ' '
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'): # if single line comments, we skip
                            continue
                        lines += line[1:].strip() + ' '
                    elif line.startswith('+'):
                        # do nothing
                        pass
                    else:
                        lines += line.strip() + ' '
                elif type == 'patched':
                    if line.startswith('+++'):
                        line = line.split(' ')[1]
                        lines += line.strip() + ' '
                    elif line.startswith('+'):
                        if line[1:].strip().startswith('//'): # if single line comments, we skip
                            continue
                        lines += line[1:].strip() + ' '
                    elif line.startswith('-'): 
                        # do nothing
                        pass
                    else:
                        lines += line.strip() + ' '
        return lines
    

def get_diff_files_frag(patch,type):
    with open(patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        flag = True
        for line in file:
            line = line.strip()
            if '*/' in line:
                flag = True
                continue
            if flag == False:
                continue
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index') or line.startswith('Binary'):
                    continue
                elif '/*' in line:
                    flag = False
                    continue
                elif type == 'buggy':
                    if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                        # continue
                        line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        lines += ' '.join(line) + ' '
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('+'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                elif type == 'patched':
                    if line.startswith('PATCH_DIFF_ORIG=---'):
                        continue
                    elif line.startswith('+++'):
                        # continue
                        line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        lines += ' '.join(line) + ' '
                    elif line.startswith('+'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('-'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
        return lines

def get_whole(path):
    with open(path, 'r') as f:
        lines = ''
        for line in f:
            line = line.strip()
            lines += line + ' '
    return lines


import os
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[2]
patch_path = os.path.join(PROJECT_DIR, 'data', 'raw', 'custom_patches', 'correct', 'patch1-Chart-1-AVATAR.patch')

print(type(get_diff_files(patch_path, 'buggy')))