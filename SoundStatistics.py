from pyAudioAnalysis import MidTermFeatures as aF
import os
import numpy as np
import matplotlib.pyplot as plt

dirs = ["C:\Users\jelle\OneDrive - Het Baken\Informatica\G6 periode 2 eindopdracht\alarm.wav", "C:\Users\jelle\OneDrive - Het Baken\Informatica\G6 periode 2 eindopdracht\crickets.wav"]
for d in dirs:
  f, files, fn = aF.directory_feature_extraction(d, m_win, m_step, s_win, s_step)
  features.append(f)
  
print(features[0].shape, features[1].shape)
f1 = np.array([features[0][:, fn.index('spectral_centroid_mean')], features[0][:, fn.index('energy_entropy_mean')]])
f2 = np.array([features[1][:, fn.index('spectral_centroid_mean')], features[1][:, fn.index('energy_entropy_mean')]])

x1 = f1[0, :]
y1 = f1[1, :]
x2 = f2[0, :]
y2 = f2[1, :]

plt.scatter(x1, y1, label="alarm")
plt.scatter(x2, y2, label="crickets")
plt.xlabel('spectral_centroid_mean')
plt.ylabel('energy_entropy_mean')

plt.legend()
plt.show()
