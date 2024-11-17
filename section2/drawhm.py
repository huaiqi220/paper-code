import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def generate_gaussian_focus_image(cm_coord, variance, output_path="gaussian_focus.png"):
    # Constants
    cm_to_pixel = 0.2  # Conversion factor: 0.2 cm per pixel
    img_size = 120  # Size of the output image (120x120)
    spatial_range = 6  # Corresponding range in cm [-6, 6]

    # Convert cm coordinates to pixel coordinates
    pixel_coord_x = int((cm_coord[0] + spatial_range) / cm_to_pixel)
    pixel_coord_y = int((cm_coord[1] + spatial_range) / cm_to_pixel)

    # Generate grid for the image space
    x = np.linspace(-spatial_range, spatial_range, img_size)
    y = np.linspace(-spatial_range, spatial_range, img_size)
    X, Y = np.meshgrid(x, y)

    # Calculate 2D Gaussian distribution
    gaussian = stats.multivariate_normal(mean=[cm_coord[0], cm_coord[1]],
                                          cov=[[variance, 0], [0, variance]])
    Z = gaussian.pdf(np.dstack((X, Y)))

    # Normalize the heatmap
    Z /= np.max(Z)

    # Plotting the heatmap in grayscale
    plt.figure(figsize=(6, 6))
    plt.imshow(Z, extent=[-spatial_range, spatial_range, -spatial_range, spatial_range], origin='lower', cmap='gray', vmin=0, vmax=1)
    # plt.colorbar(label='Probability Density')
    # plt.title('2D Gaussian Focus Heatmap')
    # plt.xlabel('X (cm)')
    # plt.ylabel('Y (cm)')

    # Save the output image
    plt.savefig(output_path)
    plt.close()

# Example usage
cm_coord = [2.0, -3.0]  # Coordinate in cm
variance = 0.1  # Variance of the Gaussian
output_path = "gaussian_focus.png"
generate_gaussian_focus_image(cm_coord, variance, output_path)
