from preprocessing_filter_functions import filter_wav_file
from pathlib import Path

p = Path('.')
q = p.resolve().parent / 'data' / 'TAU-urban-acoustic-scenes-2019-development.audio.1' / 'audio'
for file in q.iterdir():
    if file.is_file():
        filter_wav_file(file, False, False)

