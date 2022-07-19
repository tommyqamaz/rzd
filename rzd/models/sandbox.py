import cv2
import numpy as np
import matplotlib.pyplot as plt

# IMAGE_EX_PATH = (
#     "/home/qamaz/Documents/mlprojects/hackatons/rzd/img_0.0007435791908498857.png"
# )
IMAGE_EX_PATH = (
    /home/qamaz/Documents/mlprojects/hackatons/rzd/rzd/data/sample_solution/img_0.1719878008870872.png
)
image = cv2.imread(str(IMAGE_EX_PATH), cv2.IMREAD_GRAYSCALE)
plt.imshow(image)
print(np.unique(image))
