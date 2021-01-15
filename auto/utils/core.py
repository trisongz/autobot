from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import unicodedata
from itertools import chain
from subprocess import check_output


_timer_formats = {
    'secs': ['sec', 'secs', 'second', 'seconds', 's'],
    'mins': ['min', 'mins', 'minute', 'minutes', 'm'],
    'hrs': ['hr', 'hrs', 'hour', 'hours', 'h'],
    'days': ['day', 'days', 'd'],
    'wks': ['wk', 'wks', 'week', 'weeks' 'w']
}

def FormatSize(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return bytes, f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def run_command(cmd):
    out = check_output(cmd, shell=True)
    if isinstance(out, bytes):
        out = out.decode('utf8')
    return out


class AutoArgs(dict):
    def __init__(self, config):
        for k,v in config.items():
            self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
