import pyfiglet
import numpy as np
import cv2
import sys
import os

# Display the Panorama text
panorama_text = pyfiglet.figlet_format("Panorama")
print(panorama_text)
print("Initializing...")

class ImageStitching:
    """Contains the utilities required to stitch images"""

    def __init__(self):
        self.smoothing_window_size = 500

    def give_gray(self, image):
        """Converts the input image to grayscale."""
        photo_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image, photo_gray

    @staticmethod
    def _sift_detector(image):
        """Applies SIFT algorithm to the given image."""
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
        return keypoints, features

    def create_and_match_keypoints(self, features_train_image, features_query_image):
        """Creates and matches keypoints from the SIFT features using Brute Force matching."""
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        best_matches = bf.match(features_train_image, features_query_image)
        #print(best_matches)
        raw_matches = sorted(best_matches, key=lambda x: x.distance)
        return raw_matches

    def compute_homography(self, keypoints_train_image, keypoints_query_image, matches, reprojThresh):
        """Computes the Homography to map images to a single plane using RANSAC algorithm."""
        keypoints_train_image = np.float32([keypoint.pt for keypoint in keypoints_train_image])
        keypoints_query_image = np.float32([keypoint.pt for keypoint in keypoints_query_image])

        if len(matches) >= 4:
            points_train = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
            points_query = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
            H, status = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)
            return matches, H, status
        else:
            print("Minimum match count not satisfied; cannot get homography.")
            return None

    def create_mask(self, query_image, train_image, version):
        """Creates the mask using query and train images for blending."""
        height_query_photo = query_image.shape[0]
        width_query_photo = query_image.shape[1]
        width_train_photo = train_image.shape[1]
        height_panorama = height_query_photo
        width_panorama = width_query_photo + width_train_photo
        offset = int(self.smoothing_window_size / 2)
        barrier = query_image.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))

        if version == "left_image":
            mask[:, barrier - offset: barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset: barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, barrier + offset:] = 1

        return cv2.merge([mask, mask, mask])

    def blending_smoothing(self, query_image, train_image, homography_matrix):
        """Blends both query and train images via the homography matrix."""
        height_img1 = query_image.shape[0]
        width_img1 = query_image.shape[1]
        width_img2 = train_image.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        mask1 = self.create_mask(query_image, train_image, version="left_image")
        panorama1[0:query_image.shape[0], 0:query_image.shape[1], :] = query_image
        panorama1 *= mask1

        mask2 = self.create_mask(query_image, train_image, version="right_image")
        panorama2 = (
            cv2.warpPerspective(train_image, homography_matrix, (width_panorama, height_panorama))
            * mask2
        )

        result = panorama1 + panorama2

        # Remove extra blackspace
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1

        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result

    def draw_and_show_keypoints(self, image, keypoints, width=None, height=None):
        """Draws and shows keypoints on the image, with optional resizing."""
        if width is not None and height is not None:
           image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def read(image_dir_list):
    """Reads the images from the provided directory list and returns them as a list."""
    images_list = []
    for image_dir in image_dir_list:
        image = cv2.imread(image_dir)
        images_list.append(image)
    return images_list, len(images_list)

def recurse(image_list, no_of_images):
    """Recursive function to get panorama of multiple images."""
    if no_of_images == 2:
        result, mapped_image = forward(
            query_photo=image_list[no_of_images - 2],
            train_photo=image_list[no_of_images - 1],
        )
        return result, mapped_image
    else:
        result, _ = forward(
            query_photo=image_list[no_of_images - 2],
            train_photo=image_list[no_of_images - 1],
        )
        image_list[no_of_images - 2] = result
        return recurse(image_list, no_of_images - 1)

def forward(query_photo, train_photo):
    """Runs a forward pass using the ImageStitching class."""
    image_stitching = ImageStitching()
    _, query_photo_gray = image_stitching.give_gray(query_photo)
    _, train_photo_gray = image_stitching.give_gray(train_photo)

    keypoints_train_image, features_train_image = image_stitching._sift_detector(train_photo_gray)
    keypoints_query_image, features_query_image = image_stitching._sift_detector(query_photo_gray)

    # Show the keypoints on the images
    image_stitching.draw_and_show_keypoints(query_photo, keypoints_query_image,width=800, height=600)
    image_stitching.draw_and_show_keypoints(train_photo, keypoints_train_image,width=800, height=600)

    matches = image_stitching.create_and_match_keypoints(features_train_image, features_query_image)

    mapped_feature_image = cv2.drawMatches(
        train_photo,
        keypoints_train_image,
        query_photo,
        keypoints_query_image,
        matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    M = image_stitching.compute_homography(
        keypoints_train_image, keypoints_query_image, matches, reprojThresh=4
    )

    if M is None:
        return "Error: cannot stitch images"

    (matches, homography_matrix, status) = M

    result = image_stitching.blending_smoothing(query_photo, train_photo, homography_matrix)
    return result, mapped_feature_image

if __name__ == "__main__":
    try:
        num_images = int(input("Enter the number of images to stitch: "))
        image_paths = []

        for i in range(num_images):
            image_path = input(f"Enter the path for image {i + 1}: ")
            image_paths.append(image_path)

        def read_images(image_path):
            photo = cv2.imread(image_path)
            return photo

        images = [read_images(path) for path in image_paths]

        result, _ = recurse(images, num_images)

        # Ensure the output directory exists
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, "panorama_image.jpg")
        cv2.imwrite(output_path, result)
        print(f"Panorama created successfully and saved at {output_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")
