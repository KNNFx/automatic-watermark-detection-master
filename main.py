import cv2
from src.estimate_watermark import *

gx, gy, gxlist, gylist = estimate_watermark_gradients('./images/input')

cropped_gx, cropped_gy = crop_watermark_area(gx, gy)
W_m = poisson_image_reconstruction(cropped_gx, cropped_gy)

#
W_m_normalized = cv2.normalize(W_m, None, 0, 255, cv2.NORM_MINMAX)
W_m_uint8 = np.uint8(W_m_normalized)
#

laplacian = cv2.Laplacian(W_m_uint8, cv2.CV_64F)

# Normalize the Laplacian result for display
laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
laplacian_uint8 = ~np.uint8(laplacian_normalized)

# Display the image using matplotlib
plt.imshow(laplacian_uint8, cmap='gray')
plt.title('Reconstructed Watermark')
plt.axis('off')
plt.show()

# Define the output directory and file name
output_directory = './images/output'
output_image_name = 'reconstructed_watermark.png'  # Ensure it has a valid extension
output_image_path = os.path.join(output_directory, output_image_name)

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save the reconstructed image
cv2.imwrite(output_image_path, laplacian_uint8)
print(f'Reconstructed watermark image saved to {output_image_path}')