# import soundfile as sf
# sf.read('../../data/TAU-urban-acoustic-scenes-2019-development/audio/airport-barcelona-0-0-a.wav')

"""
This script uses the filter_wav_file to filter all wav files in directory.
"""


from preprocessing_functions import filter_wav_file
from pathlib import Path
import soundfile as sf

p = Path('.')
q = p.resolve().parent.parent / 'data' / 'TAU-urban-acoustic-scenes-2019-development' / 'audio'

dest_dir = q.parent.parent / ('filtered_' + q.parent.name)
if not dest_dir.exists():
    dest_dir.mkdir()


for src_file in q.iterdir():
    if src_file.is_file():
        dest_file = dest_dir / src_file.name  # filtered_
        data, fs = sf.read(src_file)
        filtered_data = filter_wav_file(data, fs, False)
        sf.write(dest_file, filtered_data, fs)

#%%
"""
This script uses the cut_wav_file to cut all wav files in directory.
"""

from pathlib import Path
from preprocessing_functions import cut_n_write_wav_file
import soundfile as sf

length_of_segment = 6  # seconds
#dataSource = 'TAU-urban-acoustic-scenes-2019-development'
# dataSource = 'rafael_16667'
# dataSource = 'airport_str_traf'
dataSource = 'rafael'

p = Path('.')
src_dir = p.resolve().parent.parent / 'data' / dataSource / 'audio'

dest_dir = src_dir.parent.parent / ('cut_length_' + str(length_of_segment) + '_' + src_dir.parent.name) / 'audio'
if not dest_dir.parent.exists():
    dest_dir.parent.mkdir()
if not dest_dir.exists():
    dest_dir.mkdir()
for src_file in src_dir.iterdir():
    if src_file.is_file():
        data, fs = sf.read(src_file)
        cut_n_write_wav_file(data, fs, src_file, dest_dir, length_of_segment)

#%%
"""
Mono to streo
"""


from preprocessing_functions import filter_wav_file
from pathlib import Path
import soundfile as sf

dataSource = 'cut_length_3_airport_str_traf'

p = Path('.')
q = p.resolve().parent.parent / 'data' / dataSource / 'audio'

dest_dir = q.parent.parent / ('mono_' + q.parent.name)
if not dest_dir.exists():
    dest_dir.mkdir()


for src_file in q.iterdir():
    if src_file.is_file():
        dest_file = dest_dir / src_file.name  # filtered_
        data, fs = sf.read(src_file)
        data = data.sum(axis=1) / 2  # stereo to mono
        sf.write(dest_file, data, fs)


#%%
"""
This script uses the decimate_wav_file function to decimate all wav files in directory.
"""


from pathlib import Path
from preprocessing_functions import decimate_wav_file

dec_factor = 3

src_data_type = 'mono_cut_length_3_'  # original
# src_data_type = 'cut_length_1_'

#src_data = 'TAU-urban-acoustic-scenes-2019-development'
src_data = 'airport_str_traf'

dest_data_type = 'decimate_' + str(dec_factor) + '_'

p = Path('.')
src_dir = p.resolve().parent.parent / 'data' / (src_data_type + src_data) / 'audio'

dest_dir = src_dir.parent.parent / (dest_data_type + src_dir.parent.name) / 'audio'
if not dest_dir.parent.exists():
    dest_dir.parent.mkdir()
if not dest_dir.exists():
    dest_dir.mkdir()

for src_file in src_dir.iterdir():
    if src_file.is_file():
        data, fs = sf.read(src_file)
        dec_data, dec_fs = decimate_wav_file(data, fs, dec_factor)
        sf.write(dest_dir / src_file.name, dec_data, dec_fs)

#%%
"""
This script is using place_speaker function to place speaker in all given files.
"""
from pathlib import Path
import soundfile as sf
from preprocessing_functions import place_speaker

speaker_proportion = 2

src_data_type = ''  # original
# src_data_type = 'cut_length_1_'

dest_data_type = 'placed_speaker_prop_' + str(speaker_proportion) + '_'

p = Path('.')
src_dir = p.resolve().parent.parent / 'data' / (src_data_type + 'TAU-urban-acoustic-scenes-2019-development') / 'audio'

dest_dir = src_dir.parent.parent / (dest_data_type + src_dir.parent.name) / 'audio'
if not dest_dir.parent.exists():
    dest_dir.parent.mkdir()
if not dest_dir.exists():
    dest_dir.mkdir()

speaker_path = p.resolve().parent.parent / 'data' / 'speakers' / 'radio_44_1_to_48.wav'
speaker, fs = sf.read(speaker_path)

for src_file in src_dir.iterdir():
    if src_file.is_file():
        src_data, fs = sf.read(src_file)
        new_data = place_speaker(src_data, speaker, speaker_proportion)
        sf.write(dest_dir / src_file.name, new_data, fs)

#%%
"""
"""
from pathlib import Path
import soundfile as sf
import audioop
from preprocessing_functions import *

speaker_proportion = 1.5
length_of_segment = 3
dec_factor = 3
for length_of_segment in [2]:
    str_speaker_proportion = str(speaker_proportion).split('.')[0] + (type(speaker_proportion) == float)*('-' + str(speaker_proportion).split('.')[-1])
    dest_data_type = 'filtered_' + 'speaker_' + str_speaker_proportion + '_cut_' + str(length_of_segment) + '_dec_' + str(dec_factor) + '_mono'

    p = Path('.')
    src_dir = p.resolve().parent.parent / 'data' / 'airport_str_traf' / 'audio'

    dest_dir = src_dir.parent.parent / (dest_data_type + '_' + src_dir.parent.name) / 'audio'
    if not dest_dir.parent.exists():
        dest_dir.parent.mkdir()
    if not dest_dir.exists():
        dest_dir.mkdir()

    speaker_path = p.resolve().parent.parent / 'data' / 'speakers' / 'radio_44_1_to_48.wav'
    speaker, fs = sf.read(speaker_path)

    for src_file in src_dir.iterdir():
        if src_file.is_file():
            data, fs = sf.read(src_file)
            data = place_speaker(data, speaker, speaker_proportion)
            data = filter_wav_file(data, fs)
            data, fs = decimate_wav_file(data, fs, dec_factor)
            data = data.sum(axis=1) / 2  # stereo to mono
            cut_n_write_wav_file(data, fs, src_file, dest_dir, length_of_segment)


