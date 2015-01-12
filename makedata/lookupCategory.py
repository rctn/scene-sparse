import argparse
import sys
import pandas as pd
import dictionary_utils as du


def get_door_dict(r):
    # returns a dictionary for a given category row
    # '/a/abbey' -> {indoor -> False, outdoor, natural -> False, outdoor, man-made -> True}
    return {
        'indoor': 1 == r['indoor'],
        'outdoor, natural': 1 == r['outdoor, natural'],
        'outdoor, man-made': 1 == r['outdoor, man-made']
    }


def in_xor_out_door(_d):
    # returns true if the input is indoor or any sort of outdoor but not both (or none)
    # ^ is xor
    return _d['indoor'] ^ (_d['outdoor, natural'] or _d['outdoor, man-made'])


def natural_xor_man_made(_d):
    # returns true if the input is outdoor natural or outdoor man-made but not both (or none)
    # ^ is xor
    return _d['outdoor, natural'] ^ _d['outdoor, man-made']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process the SUN excel file, create a dictionary to reflect it and save it.')
    parser.add_argument('dictionary_filename', type=str, help='a filename for the output dictionary')
    parser.add_argument('--excel_filename', dest='excel_filename', type=str, help='the input excel sheet filename',
                        default='/clusterfs/cortex/scratch/shiry/hierarchy_three_levels/three_levels.xlsx')
    args = parser.parse_args()

    # use the first column (labelled category) as an index for the DataFrame
    excel = pd.read_excel(args.excel_filename, 'SUN908', skiprows=[0], index_col=0)
    # create a dictionary to reflect the excel file
    d = {category.replace("'", ""): get_door_dict(excel.loc[category]) for category in excel.index}

    # DEBUG: print the value in the dictionary of a given category
    # print(d[sys.argv[2]])

    # save the dictionary to file
    du.save_dictionary(d, args.dictionary_filename)