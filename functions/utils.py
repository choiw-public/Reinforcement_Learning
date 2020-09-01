import os
import re


def list_getter(dir_name, extension, must_include=None):
    def sort_nicely(a_list):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        a_list.sort(key=alphanum_key)

    file_list = []
    if dir_name:
        for path, subdirs, files in os.walk(dir_name):
            for name in files:
                if name.lower().endswith((extension)):
                    if must_include:
                        if must_include in name:
                            file_list.append(os.path.join(path, name))
                    else:
                        file_list.append(os.path.join(path, name))
        sort_nicely(file_list)
    return file_list
