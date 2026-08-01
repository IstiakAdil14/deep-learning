import json
import sys
nb_path = sys.argv[1]
nb = json.load(open(nb_path))
cells = nb['cells']
for i, c in enumerate(cells):
    source = ''.join(c['source'])
    if source.strip():
        print(f'=== Cell {i} [{c["cell_type"]}] ===')
        print(source[:500])
        print()