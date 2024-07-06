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
   

    def __init__(self):
        self.smoothing_window_size = 500

    def give_gray(self, image):
        
        photo_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image, photo_gray

    @staticmethod
    def _sift_detector(image):
        
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
        return keypoints, features

    def create_and_match_keypoints(self, features_train_image, features_query_image):
        
        matches = []
        
        # Loop through each descriptor in the training image
        for i, train_descriptor in enumerate(features_train_image):
            # Find the best match for this descriptor in the query image
            best_match = None
            best_distance = float('inf')
            
            # Loop through each descriptor in the query image
            for j, query_descriptor in enumerate(features_query_image):
                # Calculate Euclidean distance between the descriptors
                distance = np.linalg.norm(train_descriptor - query_descriptor)
                
                # Update the best match if the current distance is smaller
                if distance < best_distance:
                    best_distance = distance
                    best_match = cv2.DMatch(i, j, best_distance)
            
            # Add the best match for this descriptor
            matches.append(best_match)
        
        # Sort matches based on the distance
        raw_matches = sorted(matches, key=lambda x: x.distance)
        
        return raw_matches


    def compute_homography(self, keypoints_train_image, keypoints_query_image, matches, reprojThresh):
       
        def normalize(points):
            mean = np.mean(points, axis=0)
            std = np.std(points, axis=0)
            s = np.sqrt(2) / std
            transform = np.array([[s[0], 0, -s[0] * mean[0]],
                                  [0, s[1], -s[1] * mean[1]],
                                  [0, 0, 1]])
            normalized_points = np.dot(transform, np.concatenate((points.T, np.ones((1, points.shape[0])))))
            return transform, normalized_points.T
        
        def dlt(points1, points2):#dlt=direct linear transformation
            A = []
            for i in range(len(points1)):
                x, y = points1[i][0], points1[i][1]
                u, v = points2[i][0], points2[i][1]
                A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
                A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
            A = np.array(A)
            U, S, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            return H
        
        def compute_inliers(H, points1, points2, thresh):
            inliers = []
            points1_proj = np.dot(H, np.concatenate((points1.T, np.ones((1, points1.shape[0])))))
            points1_proj = points1_proj / points1_proj[2]
            points1_proj = points1_proj[:2].T
            distances = np.linalg.norm(points1_proj - points2, axis=1)
            for i, distance in enumerate(distances):
                if distance < thresh:
                    inliers.append(i)
            return inliers
        
        keypoints_train_image = np.float32([keypoint.pt for keypoint in keypoints_train_image])
        keypoints_query_image = np.float32([keypoint.pt for keypoint in keypoints_query_image])
        
        if len(matches) < 4:
            print("Minimum match count not satisfied; cannot get homography.")
            return None
        
        points_train = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
        
        best_H = None
        max_inliers = []
        
        for _ in range(1000):  # Number of iterations
            indices = np.random.choice(len(matches), 4, replace=False)
            points_train_sample = points_train[indices]
            points_query_sample = points_query[indices]
            
            T_train, normalized_train = normalize(points_train_sample)
            T_query, normalized_query = normalize(points_query_sample)
            
            H_normalized = dlt(normalized_train, normalized_query)
            H = np.dot(np.linalg.inv(T_query), np.dot(H_normalized, T_train))
            
            inliers = compute_inliers(H, points_train, points_query, reprojThresh)
            
            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_H = H
        
        if best_H is None or len(max_inliers) < 4:
            print("RANSAC could not find a valid homography.")
            return None
        
        inlier_points_train = points_train[max_inliers]
        inlier_points_query = points_query[max_inliers]
        
        T_train, normalized_train = normalize(inlier_points_train)
        T_query, normalized_query = normalize(inlier_points_query)
        
        H_normalized = dlt(normalized_train, normalized_query)
        best_H = np.dot(np.linalg.inv(T_query), np.dot(H_normalized, T_train))
        
        return matches, best_H, max_inliers

    def create_mask(self, query_image, train_image, version):
       
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
            cv2.warpPerspective(train_image, homography_matrix, (width_panorama, height_panorama))#cut the overlapping part from the train image using the homography matrix
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
        
        if width is not None and height is not None:
           image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def read(image_dir_list):
   
    images_list = []
    for image_dir in image_dir_list:
        image = cv2.imread(image_dir)
        images_list.append(image)
    return images_list, len(images_list)

def recurse(image_list, no_of_images):
   
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
    mapped_feature_image= cv2.resize(mapped_feature_image, (700, 700), interpolation=cv2.INTER_AREA)
    cv2.imshow("Matches", mapped_feature_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
