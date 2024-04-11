"""
This module defines the NewsDataLoader class for loading news-related datasets
 and a function to print a visual tree structure of a directory.

The NewsDataLoader class provides a method to load a dataset from a given path.
If the dataset has already been loaded, it retrieves the data from memory instead of reloading it.

The tree function prints a visual tree structure of a directory.
 It can be limited to directories only and can also have a length limit.
"""
from pathlib import Path
from itertools import islice
import sys

# prefix components:
SPACE =  '    '
BRANCH = '│   '
# pointers:
TEE =    '├── '
LAST =   '└── '

def tree(dir_path: Path, level: int=-1, limit_to_directories: bool=False,
         length_limit: int=1000):
    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path) # accept string coerceable to Path
    files = 0
    directories = 0
    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level:
            return # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [TEE] * (len(contents) - 1) + [LAST]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = BRANCH if pointer == TEE else SPACE
                yield from inner(path, prefix=prefix+extension, level=level-1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1
    yield dir_path.name
    iterator = inner(dir_path, level=level)
    yield from islice(iterator, length_limit)
    if next(iterator, None):
        yield f'... length_limit, {length_limit}, reached, counted:'
    yield f'\n{directories} directories' + (f', {files} files' if files else '')

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else '.'
    dir__path = Path(p)
    for line in tree(dir__path, limit_to_directories=True):
        print(line)
