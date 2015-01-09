import sys
import pandas as pd

def getDoorStr(r):
    if 1 == r['indoor']:
        return 'indoor'
    elif 1 == r['outdoor, natural']:
        return 'outdoor, natural'
    elif 1 == r['outdoor, man-made']:
        return 'outdoor, man-made'

if __name__ == "__main__":
    excel = pd.read_excel('/clusterfs/cortex/scratch/shiry/hierarchy_three_levels/three_levels.xlsx', 'SUN908', skiprows=[0])
    d = {x.replace("'", "") : getDoorStr(excel[excel['category'] == x]) for x in excel['category']}

    print(d[sys.argv[1]])
