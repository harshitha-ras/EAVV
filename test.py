import matplotlib.pyplot as plt
import numpy as np

# Create a simple test image
plt.figure()
plt.imshow(np.random.rand(100, 100, 3))
plt.savefig('test_image.png')
print("Test image saved successfully")
