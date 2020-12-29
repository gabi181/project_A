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

indoor = ('airport', 'room', 'hallway')
outdoor = ('street_traffic', 'outdoor')

two_csv = 1
# data_source = 'rafael'
data_source = 'airport_str_traf'

delimiter = '-'

p = Path('.')
data_path = p.resolve().parent.parent / 'data' / data_source

length_of_segment = 1
# data_type = 'cut_length_' + str(length_of_segment) + '_TAU-urban-acoustic-scenes-2019-development'
csv_path = data_path / 'evaluation_setup'  #data_path / data_type / 'evaluation_setup'
wav_files_path = data_path / 'audio'  # data_path / data_type / 'audio'

if not(csv_path.exists()):
    csv_path.mkdir()
csv_val_file_path = csv_path / 'fold1_evaluate.csv'
# csv_test_file_path = csv_path / 'fold1_test.csv'
csv_test_file_path = csv_path / 'fold1_test_80-20.csv'
#csv_train_file_path = csv_path / 'in-air_out-street_traf.csv'
csv_train_file_path = csv_path / 'fold1_train_80-20.csv'

if two_csv:
    csv_train_file = open(str(csv_train_file_path), 'w')
    csv_test_file = open(str(csv_test_file_path), 'w')
    csv_train_file.write('filename\tscene_label\n')
    csv_test_file.write('filename\tscene_label\n')
else:
    csv_train_file = open(str(csv_train_file_path), 'w')
    csv_train_file.write('filename\tscene_label\n')

tag = 0


counter = 1

for file in wav_files_path.iterdir():
    label = file.stem.split(delimiter)[tag]
    if label in indoor:
        label = 'indoor'
    elif label in outdoor:
        label = 'outdoor'
    else:
        continue

    if two_csv:
        if counter % 10 == 0 or counter % 9 == 0:
            csv_test_file.write('audio/' + file.name + '\t' + label + '\n')
        else:
            csv_train_file.write('audio/' + file.name + '\t' + label + '\n')
        counter += 1
    else:
        csv_train_file.write('audio/' + file.name + '\t' + label + '\n')
if two_csv:
    csv_train_file.close()
    csv_test_file.close()
else:
    csv_train_file.close()

#%%
"""
This script makes a csv file from csv file.
It goes through each line and duplicates it in the same way that the cut has been done. 
It robust to different segments length.
"""
from pathlib import Path

# which_fold = 'fold1_train.csv'
which_fold = 'fold1_train_80-20.csv'
# which_fold = 'fold1_evaluate.csv'
# which_fold = 'fold1_test.csv'
# which_fold = 'fold1_test_80-20.csv'

new_csv = 'fold1_train_80-20.csv'
# new_csv = 'fold1_test_80-20.csv'

indoor = ('\tairport\n', '\troom\n', '\thallway\n', '\tindoor\n')
outdoor = ('\tstreet_traffic\n', '\toutdoor\n')

# speaker_proportion = '1-5'
# speaker_proportion = ''
length_of_segment = 6
# dec_factor = 3

# preprocess = 'filtered_' + 'speaker_' + speaker_proportion + '_cut_' + str(length_of_segment) + '_dec_' + str(dec_factor) + '_' + 'mono'
preprocess = 'cut_length_' + str(length_of_segment)


# data_source = 'DCASE'
data_source = 'rafael'
# data_source = 'airport_str_traf'

p = Path('.')
data_path = p.resolve().parent.parent / 'data'
orig_csv_dir = data_path / data_source / 'evaluation_setup'
orig_csv_path = orig_csv_dir / which_fold

max_orig_length = 42

data_type = preprocess + '_' + data_source
new_csv_dir = data_path / data_type / 'evaluation_setup'
new_csv_path = new_csv_dir / new_csv

if not(new_csv_dir.exists()):
    new_csv_dir.mkdir()
new_csv_file = open(str(new_csv_path), 'w')
orig_csv_file = orig_csv_path.open()

lines = orig_csv_file.readlines()
new_csv_file.write(lines[0])
for line in lines[1:]:
    line_split = line.split('.wav')
    for i in range(int(max_orig_length / length_of_segment)):
        file_name = line_split[0] + '_' + str(i) + '.wav'
        if (data_path / data_type / file_name).exists():
            if line_split[1] in indoor:
                label = 'indoor'
            elif line_split[1] in outdoor:
                label = 'outdoor'
            else:
                print(file_name)
                continue
            new_csv_file.write(file_name + '\t' + label + '\n')

new_csv_file.close()
orig_csv_file.close()
#%%
"""
This script reads a csv file and checks the length of the wav files.
Then it splits it according to the lengths.
"""
from pathlib import Path
import pandas as pd
import soundfile as sf

p = Path('.')
csv_path = p.resolve().parent.parent / 'data' / 'rafael' / 'evaluation_setup' / 'fold1_test_80-20.csv'
dir_short = p.resolve().parent.parent / 'data' / 'rafael' / 'short-0-8'
dir_medium = p.resolve().parent.parent / 'data' / 'rafael' / 'medium-8-12'
dir_long = p.resolve().parent.parent / 'data' / 'rafael' / 'long-12-40'

for dir_name in [dir_short, dir_medium, dir_long]:
    if not (dir_name.exists()):
        dir_name.mkdir()
        (dir_name / 'audio').mkdir()

csv = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
file_names = csv['filename'].tolist()

for file_name in file_names:
    file_path = p.resolve().parent.parent / 'data' / 'rafael' / file_name
    data, fs = sf.read(file_path)
    seconds = len(data)/fs
    dest_dir = str(dir_short) * (seconds < 8) + str(dir_medium) * (8 <= seconds < 12) + str(dir_long) * (seconds >= 12)
    sf.write(dest_dir + '/' + file_name, data, fs)


