import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for external window
import os
import pickle
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch  # type: ignore
from scipy.ndimage import gaussian_filter

# Global parameters
n_folds = 6
k = 8


def filter_function(k, theta, n_folds=n_folds):
    # Computes (cos(n_folds/2 * theta))^(2*k) using torch.
    # theta is expected to be a torch.Tensor.
    filt = torch.exp(k * torch.log((torch.cos(n_folds / 2 * theta)) ** 2))
    return filt


def normalize_min_max(data, lower_percentile=1, upper_percentile=100):
    """Robustly normalize a torch.Tensor to [0, 1] using percentile clipping.

    The output is always a torch.Tensor.
    """
    array = data.detach().cpu().numpy()
    lower_bound = np.percentile(array, lower_percentile)
    upper_bound = np.percentile(array, upper_percentile)
    if upper_bound == lower_bound:
        norm_array = np.zeros_like(array)
    else:
        clipped = np.clip(array, lower_bound, upper_bound)
        norm_array = (clipped - lower_bound) / (upper_bound - lower_bound)
    return torch.tensor(norm_array, dtype=torch.float32)


def mask_and_blur_images(array, sigma=1):
    """Masks signal inside radius of 14 and outside radius of 30 and applies
    gaussian blur."""
    x, y = np.meshgrid(np.arange(128), np.arange(128))
    radius = np.sqrt((x - 64) ** 2 + (y - 62) ** 2)
    mask1 = radius <= 14
    mask2 = radius >= 30
    masked_data = array.copy()
    masked_data[mask1] = 0
    masked_data2 = masked_data.copy()
    masked_data2[mask2] = 0
    blurred_data = gaussian_filter(masked_data2, sigma=sigma)
    return blurred_data


def calculate_azimuthal_angle(center, point):
    """Calculate the azimuthal angle (in degrees) between the center and a
    given point."""
    dx = point[0] - center[0]
    dy = -1 * (point[1] - center[1])
    angle = -np.degrees(np.arctan2(dy, dx))
    return angle


def record_angles(image_stack, output_path, domain):
    """Displays images and allows the user to click to record an azimuthal
    angle. Updates the frame on each click.

    For each frame:
      - Domain 1: Displays the processed image.
      - Domain 2: Subtracts a filtered signal (computed using DATA_THETA and the corresponding first_offset)
        from the processed image.

    A set of 6 radial (hex) spokes are drawn to aid visualization.

    :param image_stack: 3D (N,H,W) or 4D (N,H,W,C) numpy array of images.
    :param output_path: Path to save the recorded angles.
    :param domain: Integer (1 or 2) determining the processing logic.
    """
    angles = []
    num_images = image_stack.shape[0]

    # Determine image center (assume image_stack.shape = (N, H, W))
    height_idx = 1
    width_idx = 2
    center = (
        image_stack.shape[width_idx] // 2,
        image_stack.shape[height_idx] // 2,
    )

    # --- Compute DATA_THETA once based on the first image's shape ---
    h, w = image_stack[0].shape[:2]
    cx, cy = w // 2, h // 2
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    # Compute angle as arctan2(y - cy, x - cx)
    DATA_THETA = torch.tensor(
        np.arctan2(y_grid - cy, x_grid - cx), dtype=torch.float32
    )

    # Set up figure and display the first image
    fig, ax = plt.subplots()

    processed = mask_and_blur_images(image_stack[0])
    processed_tensor = torch.tensor(processed, dtype=torch.float32)
    if domain == 1:
        im0 = normalize_min_max(processed_tensor)
    elif domain == 2:
        offset_val = torch.tensor(
            np.deg2rad(first_offset[0]), dtype=torch.float32
        )
        filt_tensor = filter_function(
            k, DATA_THETA - offset_val + torch.tensor(np.deg2rad(30))
        )
        norm_proc = normalize_min_max(processed_tensor)
        diff_tensor = norm_proc - filt_tensor
        diff_tensor = torch.clamp(diff_tensor, min=0.0, max=1.0)
        im0 = diff_tensor
    else:
        print("Invalid domain. Must be 1 or 2.")
        sys.exit(1)

    # Create one image artist; we will update its data later.
    img_display = ax.imshow(im0.numpy(), cmap="viridis")
    ax.set_title(
        f"{numor}: Image 1/{num_images}: Click to record angle, press 'q' to quit"
    )

    # Create 6 radial (hex) spokes
    hex_lines = [ax.plot([], [], "r-", linewidth=1)[0] for _ in range(6)]

    def show_image(index):
        """Update the displayed image using frame index and domain, without
        creating new image objects."""
        processed = mask_and_blur_images(image_stack[index])
        processed_tensor = torch.tensor(processed, dtype=torch.float32)
        if domain == 1:
            im_tensor = normalize_min_max(processed_tensor)
        elif domain == 2:
            offset_val = torch.tensor(
                np.deg2rad(first_offset[index]), dtype=torch.float32
            )
            filt_tensor = filter_function(
                k, DATA_THETA - offset_val + torch.tensor(np.deg2rad(30))
            )
            norm_proc = normalize_min_max(processed_tensor)
            norm_filt = normalize_min_max(filt_tensor)
            diff_tensor = norm_proc - norm_filt
            diff_tensor = torch.clamp(diff_tensor, min=0, max=1)
            im_tensor = diff_tensor
        else:
            print("Invalid domain")
            return
        # Instead of creating a new image, update the data of the existing one.
        img_display.set_data(im_tensor.numpy())
        ax.set_title(
            f"{numor}: Image {index + 1}/{num_images}: Click to record angle, press 'q' to quit"
        )
        fig.canvas.draw_idle()

    def update_hex_lines(angle_deg, radius):
        """Update the six radial lines (hex spokes) based on current mouse
        angle and distance."""
        cx, cy = center
        angle_rad = np.radians(angle_deg)
        for i, ln in enumerate(hex_lines):
            this_angle = angle_rad + i * (np.pi / 3)
            x2 = cx + radius * np.cos(this_angle)
            y2 = cy + radius * np.sin(this_angle)
            ln.set_data([cx, x2], [cy, y2])
        fig.canvas.draw_idle()

    def on_move(event):
        """Event handler: update the hex spokes as the mouse moves."""
        if (
            event.inaxes == ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            cx, cy = center
            dx = event.xdata - cx
            dy = event.ydata - cy
            radius = np.sqrt(dx**2 + dy**2)
            angle_deg = np.degrees(np.arctan2(dy, dx))
            update_hex_lines(angle_deg, radius)

    def onclick(event):
        """
        Event handler: record the clicked angle and advance to the next image.
        Maintains angular continuity between clicks.
        """
        if (
            event.inaxes == ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            point = (event.xdata, event.ydata)
            angle = calculate_azimuthal_angle(center, point)

            if angles:
                previous_angle = angles[-1]
                delta_angle = angle - (previous_angle % 360)
                if delta_angle > 180:
                    delta_angle -= 360
                elif delta_angle < -180:
                    delta_angle += 360
                angles.append(previous_angle + delta_angle)
            else:
                angles.append(angle)
            print(f"Angle recorded: {angles[-1]:.2f} degrees")
            current_index = len(angles)
            if current_index < num_images:
                show_image(current_index)
            else:
                print("All images processed.")
                plt.close()

    def on_key(event):
        """Exit on pressing 'q'."""
        if event.key == "q":
            print("Exiting gracefully. Closing the figure.")
            plt.close()

    # Attach event handlers.
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()  # This call blocks until the plot window is closed.

    # Save recorded angles to the output path.
    with open(output_path, "wb") as f:
        pickle.dump(angles, f)


def run_record_angles(image_stack, domain):
    """Runs the record_angles function and returns the recorded angles.

    :param image_stack: 3D or 4D numpy array of images.
    :param domain: 1 or 2, determining processing.
    :return: List of recorded angles.
    """
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pkl"
    ) as output_file:
        output_path = output_file.name

    try:
        record_angles(image_stack, output_path, domain)
        with open(output_path, "rb") as f:
            recorded_angles = pickle.load(f)
        return recorded_angles
    finally:
        os.remove(output_path)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python record_angles_plotter.py <input_image_stack.pkl> <output_angles.pkl> <numor> <domain>"
        )
        sys.exit(1)

    input_image_stack_path = sys.argv[1]
    output_angles_path = sys.argv[2]
    numor = int(sys.argv[3])
    domain = int(sys.argv[4])
    print(f"Running on domain {domain}")

    # Load first_offset from file (required for domain 2)
    if domain == 2:
        first_offset_path = f"/Users/cadenmyers/billingelab/dev/sym_adapted_filts/working_code/seed_generating/seed_data/cm_{numor}_offset1_seed.npz"
        first_offset = np.load(first_offset_path)["data"]

    if domain not in [1, 2]:
        print("Invalid domain. Must be 1 or 2.")
        sys.exit(1)

    if not os.path.exists(input_image_stack_path):
        print(f"Input file '{input_image_stack_path}' does not exist.")
        sys.exit(1)

    with open(input_image_stack_path, "rb") as f:
        image_stack = pickle.load(f)

    record_angles(image_stack, output_angles_path, domain)
