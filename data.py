import csv

d = {}
with open('eggs.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(row)
        if row[0] in d:
            print('yup')
        d[row[0]] = 0
print(len(d))