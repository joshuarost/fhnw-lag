import numpy as np
from core import pp

v = np.array([8, 0, 15])
w = np.array([15, 0, -8])

pp("a) v", np.dot(v, v))
pp("a) w", np.dot(w, w))

pp("b ) Skalarprodukt", np.dot(v, w))
pp("b) Zwischenwinkel in Grad", np.degrees(np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))))

sum = v + w
pp("c) ", np.dot(sum, sum) - np.dot(v, v) - np.dot(w, w))

