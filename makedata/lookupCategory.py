import sys
import pandas as pd
import dictionary_utils as du


def get_door_str(r):
    if 1 == r['indoor']:
        return 'indoor'
    elif 1 == r['outdoor, natural']:
        return 'outdoor, natural'
    elif 1 == r['outdoor, man-made']:
        return 'outdoor, man-made'
    else:
        raise Exception('No door specified in row = ' + str(r))

if __name__ == "__main__":
    excel = pd.read_excel('/clusterfs/cortex/scratch/shiry/hierarchy_three_levels/three_levels.xlsx', 'SUN908', skiprows=[0])
    d = {x.replace("'", "") : get_door_str(excel[excel['category'] == x]) for x in excel['category']}

    print(d[sys.argv[1]])
