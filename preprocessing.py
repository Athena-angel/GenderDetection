import os
import csv
from collections import defaultdict

# import chardet

folder_path = './dataset'

files_and_directories = os.listdir(folder_path)

file_paths = []
for item in files_and_directories:
    item_path = os.path.join(folder_path, item)
    if os.path.isfile(item_path):
        file_paths.append(item_path)
    elif os.path.isdir(item_path):
        for root, dirs, files in os.walk(item_path):
            for file in files:
                if '.csv' in file:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)

print("File paths:")
print(file_paths)

header = ['name', 'count', 'gender']
aio_data = [header]

for csv_path in file_paths:
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # chardet.detect(file.read())
        try:
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                if len(row) == 3:
                    aio_data.append(list(row))
                else:
                    continue
        except:
            print('skipping')

mix = defaultdict(lambda: {"count": 0, "gender": ""})
for name, count, gender in aio_data[1:]:
    mix[name]["count"] += int(count)
    mix[name]["gender"] = gender

result = [[name.lower(), data['count'], data['gender']] for name, data in sorted(mix.items()) if data['count'] != None]
result = [row for row in result if any(row)]
print(result)

with open('aio_data.csv', 'w', newline='', encoding='utf-8') as writer:
    writer = csv.writer(writer, delimiter=',')
    writer.writerow(header)
    writer.writerows(result)

print('________________________>>>Done!!!')
