import os
from pathlib import Path
import re

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')

def get_diff_files_modifying(patch, type):
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
                        if line[1:].strip().startswith('//'): # if single line comments after +, we skip
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
                        if line[1:].strip().startswith('//'): # if single line comments after +, we skip
                            continue
                        lines += line[1:].strip() + ' '
                    elif line.startswith('//'):
                        continue
                    elif line.startswith('-'): 
                        # do nothing
                        pass
                    else:
                        lines += line.strip() + ' '
        return lines

def get_diff_files_frag(patch,type):
    with open(patch, 'r') as file:
        lines = ''
        p =  r"\s+"
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
                        continue
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
                    elif line.startswith('//'):
                        continue
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
    
#print(get_diff_files_modifying("/home/tdy245/ResearchProjects/Winter 2024/CMPT-828/A_Implementation/PaCo/data/raw/custom_patches/overfitting/patch1-Math-70-SketchFix-plausible.patch", "patched"))  
print(get_diff_files_frag("/home/tdy245/ResearchProjects/Winter 2024/CMPT-828/A_Implementation/PaCo/data/raw/custom_patches/overfitting/patch14-Math-85-SequenceR.patch", "buggy"))     
