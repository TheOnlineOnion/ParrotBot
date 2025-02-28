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

x = f1[0, :]
y = f1[0, :]
plt.plot(x)
plt.show()
