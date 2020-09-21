"""
This script checks where do the cut script cut the data to less than 10 segments.
"""
from pathlib import Path

length_of_segment = 1
p = Path('.')
q = p.resolve().parent.parent

dest_dir = q / 'data' / ('cut_length_' + str(length_of_segment) + '_' + 'TAU-urban-acoustic-scenes-2019-development')
counter = 0
for file in dest_dir.iterdir():
    if file.is_file():
        index = file.stem[-1]
        if not(int(index) == counter % 10):
            print(file.name)
            break
    counter += 1

#%%
"""
This script makes a csv file from all files in directory.
It assumes that the label is one of the words in the files name.
paths, delimiter and name location changes are required.
"""
from pathlib import Path

indoor = ('room', 'hallway')
outdoor = ('outdoor')

p = Path('.')
data_path = p.resolve().parent / 'data' / '4students'  #p.resolve().parent.parent / 'data'

length_of_segment = 1
# data_type = 'cut_length_' + str(length_of_segment) + '_TAU-urban-acoustic-scenes-2019-development'
csv_path = data_path / 'evaluation_setup'  #data_path / data_type / 'evaluation_setup'
wav_files_path = data_path  # data_path / data_type / 'audio'

if not(csv_path.exists()):
    csv_path.mkdir()
csv_val_file_path = csv_path / 'fold1_val.csv'
csv_test_file_path = csv_path / 'fold1_test.csv'
csv_val_file = open(str(csv_val_file_path), 'w')
csv_test_file = open(str(csv_test_file_path), 'w')

delimiter = '.'
tag = 0

csv_val_file.write('filename\tscene_label\n')
csv_test_file.write('filename\tscene_label\n')

cur_csv = 0

for file in wav_files_path.iterdir():
    label = file.stem.split(delimiter)[tag]
    if label in indoor:
        label = 'shopping_mall'
    elif label in outdoor:
        label = 'park'
    else:
        continue
    if cur_csv == 0:
        csv_val_file.write('audio/' + file.name + '\t' + label + '\n')
    else:
        csv_test_file.write('audio/' + file.name + '\t' + label + '\n')
    cur_csv = not cur_csv

csv_val_file.close()
csv_test_file.close()

#%%
"""
This script makes a csv file from the DCASE origin csv file.
It goes through each line and duplicates it in the same way that the cut has been done. 
It robust to different segments length.
"""
from pathlib import Path

#which_fold = 'fold1_train.csv'
which_fold = 'fold1_evaluate.csv'

p = Path('.')
data_path = p.resolve().parent.parent / 'data'
orig_csv_dir = data_path /'TAU-urban-acoustic-scenes-2019-development' / 'evaluation_setup'
orig_csv_path = orig_csv_dir / which_fold
max_orig_length = 10

length_of_segment = 1
data_type = 'cut_length_' + str(length_of_segment) + '_TAU-urban-acoustic-scenes-2019-development'
new_csv_dir = data_path / data_type / 'evaluation_setup'
new_csv_path = new_csv_dir / which_fold

if not(new_csv_dir.exists()):
    new_csv_dir.mkdir()
new_csv_file = open(str(new_csv_path), 'w')
orig_csv_file = orig_csv_path.open()

lines = orig_csv_file.readlines()
new_csv_file.write(lines[0])
for line in lines[1:]:
    line_split = line.split('.')
    for i in range(max_orig_length):
        file_name = line_split[0] + '_' + str(i) + '.wav'
        if (data_path / data_type / file_name).exists():
            new_csv_file.write(line_split[0] + '_' + str(i) + '.' + line_split[1])

new_csv_file.close()
orig_csv_file.close()
