import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import cv2
import argparse

def polar_to_cartesian(polar_image: np.ndarray, 
                       max_range: float, 
                       hfov: float, 
                       cmap: str) -> np.ndarray:
    """
    Convert a polar-mapped image to a Cartesian image.

    Parameters:
    - polar_image (np.ndarray): Input image in polar coordinates (H, W).
    - max_range (float): Maximum range corresponding to the last row in polar_image.
    - hfov (float): Horizontal field of view in degrees.

    Returns:
    - cartesian_image (np.ndarray): Output image in Cartesian coordinates.
    """

    H, W = polar_image.shape  # H: radial bins, W: angular bins
    cart_size = 2 * H  # Define Cartesian image size (square)
    cartesian_image = np.zeros((cart_size, cart_size))  # Output image

    # Create Cartesian coordinate grid centered at (H, H)
    x = np.linspace(0, max_range, H)
    y = np.linspace(-max_range, max_range, 2*H)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)  # Radius
    Theta = np.arctan2(Y, X)  # Angle in radians (-π, π)
    # Convert Cartesian coordinates (X, Y) -> Polar coordinates (r, theta)
    # r = np.linspace(0, max_range, H)  # Radius
    # theta = np.linspace(-np.radians(hfov/2), np.radians(hfov/2), W)  # Angle in radians (-π, π)
    # R, Theta = np.meshgrid(r, theta)

    # Convert Theta range from radians (-π to π) to index range (0 to W-1)
    theta_min = -np.radians(hfov / 2) # Corresponds to index 0
    theta_max = np.radians(hfov / 2)  # Corresponds to index W-1

    Theta_idx = ((Theta - theta_min) / (theta_max - theta_min)) * (W - 1)

    # Convert R range from (0 to max_range) to index range (0 to H-1)
    R_idx = (R / max_range) * (H - 1)

    # Clip indices to valid range
    R_idx = np.clip(R_idx, 0, H - 1)
    Theta_idx = np.clip(Theta_idx, 0, W - 1)

    # Interpolate using map_coordinates (bilinear interpolation)
    cartesian_image = map_coordinates(polar_image, [R_idx, Theta_idx], mode="constant", cval=1)
    cartesian_image[Theta > np.radians(hfov / 2)] = 0  # Zero out values outside the FOV
    cartesian_image[Theta < -np.radians(hfov / 2)] = 0  # Zero out values outside the FOV
    cartesian_image[R > max_range] = 0  # Zero out values beyond the maximum range
    #rotate the image -90 deg 
    cartesian_image = np.rot90(cartesian_image, 1)

    cartesian_cmap = plt.get_cmap(cmap)
    cartesian_image_3 = cartesian_cmap(cartesian_image)

    #make this into rgba image 
    alpha = np.ones_like(cartesian_image)
    alpha[cartesian_image == 0] = 0

    # print(cartesian_image_3[...,:3].shape)
    # print(alpha[...,None].shape)
    out_img = np.concatenate([cartesian_image_3[...,:3], alpha[...,None]], axis=-1)

    return out_img



#read either an image or .mp4 file and convert it to cartesian 
def main(input_path: str, output_path: str, hfov: float, max_range: float, cmap: str, crop_half_height: bool, rotate_90: bool):

    if input_path.endswith(".mp4"):
        #read the video frame by frame 
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        cart_frames = []
        while ret:
            frame = frame[:,:,0]
            if crop_half_height:
                frame = frame[0:frame.shape[0]//2,:]
            if rotate_90:
                frame = frame.transpose(1, 0)
            cartesian_image = polar_to_cartesian(frame, max_range, hfov, cmap)
            cart_frames.append(cartesian_image)
            ret, frame = cap.read()
        #save the cartesian frames as a video
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                            (cart_frames[0].shape[1], cart_frames[0].shape[0]), 
                            isColor=True)
        for frame in cart_frames:
            # Convert RGBA to BGR format that OpenCV expects
            frame_bgr = (255 * frame[:, :, :3][:, :, ::-1]).astype(np.uint8)
            out.write(frame_bgr)
        out.release()
    
    else:
        #read the image
        image = cv2.imread(input_path)
        image = image[:,:,0]
        if crop_half_height:
            image = image[image.shape[0]//2:,:,:]
        if rotate_90:
            image = image.transpose(1, 0)
        cartesian_image = polar_to_cartesian(image, max_range, hfov, cmap)
        #write as an rgba image 
        cv2.imwrite(output_path, (255*cartesian_image).astype(np.uint8))
        

if __name__ == "__main__":
    #read either an image or .mp4 file and convert it to cartesian 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--hfov", type=float, required=True)
    parser.add_argument("--max_range", type=float, required=True)
    parser.add_argument("--cmap", type=str, required=True)
    parser.add_argument("--crop_half_height", type=bool, required=False, default=False)
    parser.add_argument("--rotate_90", type=bool, required=False, default=False)
    args = parser.parse_args()

    main(args.input, args.output, args.hfov, args.max_range, args.cmap, args.crop_half_height, args.rotate_90)