def create_and_match_keypoints(self, features_train_image, features_query_image):
        """Creates and matches keypoints from the SIFT features using Brute Force matching."""
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        best_matches = bf.match(features_train_image, features_query_image)
        #print(best_matches)
        raw_matches = sorted(best_matches, key=lambda x: x.distance)
        return raw_matches