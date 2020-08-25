from preprocessing_filter_functions import filter_wav_file
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