"""
This script uses the filter_wav_file to filter all wav files in directory.
"""


from preprocessing_functions import filter_wav_file
from pathlib import Path

p = Path('.')
q = p.resolve().parent.parent / 'data' / 'TAU-urban-acoustic-scenes-2019-development' / 'audio'

dest_dir = q.parent.parent / ('filtered_' + q.parent.name)
if not dest_dir.exists():
    dest_dir.mkdir()


for src_file in q.iterdir():
    if src_file.is_file():
        dest_file = dest_dir / src_file.name  # filtered_
        filter_wav_file(src_file, dest_file, False, False)

#%%
"""
This script uses the cut_wav_file to cut all wav files in directory.
"""


from pathlib import Path
from preprocessing_functions import cut_wav_file

length_of_segment = 1  # seconds

p = Path('.')
src_dir = p.resolve().parent.parent / 'data' / 'TAU-urban-acoustic-scenes-2019-development' / 'audio'

dest_dir = src_dir.parent.parent / ('cut_length_' + str(length_of_segment) + '_' + src_dir.parent.name) / 'audio'
if not dest_dir.exists():
    dest_dir.mkdir()
for src_file in src_dir.iterdir():
    if src_file.is_file():
        cut_wav_file(src_file, dest_dir, length_of_segment)


#%%
"""
This script uses the decimate_wav_file function to decimate all wav files in directory.
"""


from pathlib import Path
from preprocessing_functions import decimate_wav_file

dec_factor = 3

# src_data_type = ''  # original
src_data_type = 'cut_length_1_'

dest_data_type = 'decimate_' + str(dec_factor) + '_'

p = Path('.')
src_dir = p.resolve().parent.parent / 'data' / (src_data_type + 'TAU-urban-acoustic-scenes-2019-development') / 'audio'

dest_dir = src_dir.parent.parent / (dest_data_type + src_dir.parent.name) / 'audio'
if not dest_dir.parent.exists():
    dest_dir.parent.mkdir()
if not dest_dir.exists():
    dest_dir.mkdir()

for src_file in src_dir.iterdir():
    if src_file.is_file():
        decimate_wav_file(src_file, dest_dir, dec_factor)
