import json
import sys

def dump_nb(path, out):
    nb = json.load(open(path,'r',encoding='utf-8'))
    with open(out,'w',encoding='utf-8') as f:
        for i, c in enumerate(nb['cells']):
            source = ''.join(c['source'])
            if source.strip():
                f.write(f'=== Cell {i} [{c["cell_type"]}] ===\n')
                f.write(source)
                f.write('\n\n')

dump_nb(sys.argv[1], sys.argv[2])