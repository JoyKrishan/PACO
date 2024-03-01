from pathlib import Path
import os

import ast
import astpretty

PROJECT_DIR = Path(__file__).resolve().parents[2] 
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw') # raw source code is kept here


code = """
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    
    visited.add(start)
    while queue:
        vertex = queue.popleft()
        print(str(vertex) + " ", end="")
        
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
"""

parsed_code = ast.parse(code)

astpretty.pprint(parsed_code)