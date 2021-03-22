import csv

def does_contain_dublicates(file, col=0):
    # Make sure no dublicates!
    d = {}
    with open(file, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(row)
            if row[col] in d:
                return True
            d[row[col]] = 0
    return False
