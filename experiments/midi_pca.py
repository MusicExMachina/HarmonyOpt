import numpy as np
from numpy import inf
from math import ceil
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mido
import copy
import scipy.spatial.distance as dist
import time

from sklearn.decomposition import PCA

direc="data/test_midi/bach_prelude_1.mid"

m = mido.MidiFile(direc)


def midifile_to_dict(mid):
    tracks = []
    for track in mid.tracks:
        tracks.append([vars(msg).copy() for msg in track])

    return {
        'ticks_per_beat': mid.ticks_per_beat,
        'tracks': tracks,
    }


mid = m
mid_dict_pure = midifile_to_dict(mid)

notes = []

midi_range = range(127)

mid_dict = copy.deepcopy(mid_dict_pure)

for i, track in enumerate(mid_dict['tracks']):
    tsum = 0
    for msg in track:
        tinc = msg['time']
        tsum += tinc
        #         if (msg['type']=='note_on' or msg['type']=='note_off') and msg['note']==60:
        #             print(tsum,msg['time'],msg['type'],msg['note'])
        #         else:
        #             print(tsum,msg['time'],msg['type'])

        if (msg['type'] == 'note_on' or msg['type'] == 'note_off') and msg['note'] == 60:
            print(msg, tsum)

        if msg['type'] == "note_on":
            notes.append([msg['note'], tsum, tsum])  # ,msg['velocity']
        elif msg['type'] == "note_off":
            for i in range(len(notes) - 1, -1, -1):
                if notes[i][0] == msg['note']:
                    notes[i][2] = tsum
                    break


# 1 measure = 384
# 1/16 note = 24
# pitch = notes[x][0] % 12

max_time = 0

for note in notes:
    max_time = max(max_time, note[2])

measures = ceil(max_time / 384.0)

measure_notes = np.zeros((measures*16 - 15, 12))

for i in range(0, len(measure_notes)):
    for note in notes:
        if note[1] >= i*24 + 384:
            break
        if note[2] > i*24:
            measure_notes[i][note[0] % 12] += min(note[2], i*24 + 384) - max(note[1], i*24)

pca = PCA(n_components=10)
pca.fit(measure_notes)
reduced_notes = pca.transform(measure_notes)

colors = cm.rainbow(np.linspace(0, 1, 16))

#plt.scatter(*zip(*reduced_notes), c=colors)
#plt.show()

measure_vectors = []

for i in range(0, len(reduced_notes)-1):
    measure_vectors.append(reduced_notes[i+1]-reduced_notes[i])

cosine_dist = []

for i in range(0, len(measure_vectors)-1):
    cosine_dist.append(dist.cosine(measure_vectors[i], measure_vectors[i+1]))
