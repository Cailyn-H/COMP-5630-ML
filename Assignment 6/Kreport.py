from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np




num_input = str(input("Enter image number: "))
#k_input = int(input("Enter K value: "))

img = "img/test" + num_input.zfill(2) + ".jpg"
im = Image.open(img)
img_width, img_height = im.size
px = im.load()
S = []
for x in range(0, img_width):
    for y in range(0, img_height):
        S.append(px[x, y])

intertia = []
K = range(1,21)
for k in K:
    km = KMeans(n_clusters=k).fit(S)
    intertia.append(km.inertia_)
plt.plot(K, intertia, marker= "x")
plt.xlabel('k')
plt.xticks(np.arange(21))
plt.ylabel('Objective')
plt.title('Elbow Method')
plt.show()

