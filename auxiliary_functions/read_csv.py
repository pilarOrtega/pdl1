import csv

def read_csv(file_path):
    with open(file_path, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        list = []
        for row in reader:
            list.append(row)
        return list
