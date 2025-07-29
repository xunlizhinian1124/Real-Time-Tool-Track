from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import time

class Kinect:
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        self.color_width = 1920
        self.color_height = 1080
        self.depth_width = 512
        self.depth_height = 424

        self.camera_matrix = np.array([[1092.29214, 0.0, 951.696366],
                                       [0.0, 1092.82326, 496.561675],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([[-0.00254295, 0.43106886, -0.01124835, 0.0046628, -0.75915729]], dtype=np.float32)

    def get_last_color(self):
        if self.kinect.has_new_color_frame():
            frame = self.kinect.get_last_color_frame()
            image = np.reshape(frame, [self.color_height, self.color_width, 4])
            image = image[:, :, 0:3][:,::-1,:].astype(np.uint8)
            return image
            # return cv2.resize(image, (960, 540))
        return None

    def get_last_depth(self):
        if self.kinect.has_new_depth_frame():
            frame = self.kinect.get_last_depth_frame()
            depth_image = np.reshape(frame, [self.depth_height, self.depth_width])[:,::-1].astype(np.uint16)
            return depth_image
        return None

    def draw_axis(self, image, rvec, tvec):
        axis_length = 0.05
        axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], [0, 0, 0]]).reshape(-1, 3)

        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)

        origin = tuple(imgpts[3].ravel().astype(int))
        image = cv2.line(image, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)
        image = cv2.line(image, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
        image = cv2.line(image, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)

        return image

    def detect_qrcode_with_corners(self, image, dictionary_type=cv2.aruco.DICT_4X4_50):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            if len(approx) == 4:
                outer_corners = np.squeeze(approx, axis=1)

                x, y, w, h = cv2.boundingRect(approx)
                roi = gray[y:y + h, x:x + w]
                corners, ids, _ = detector.detectMarkers(roi)

                if ids is not None:
                    aruco_corners = corners[0][0] + np.array([x, y])

                    sorted_outer_corners = [None] * 4
                    sorted_outer_corners[0] = outer_corners[np.argmin(np.sum(outer_corners, axis=1))]
                    sorted_outer_corners[1] = outer_corners[np.argmax(outer_corners[:, 0] - outer_corners[:, 1])]
                    sorted_outer_corners[2] = outer_corners[np.argmax(np.sum(outer_corners, axis=1))]
                    sorted_outer_corners[3] = outer_corners[np.argmin(outer_corners[:, 0] - outer_corners[:, 1])]

                    for idx, corner in enumerate(sorted_outer_corners):
                        cv2.circle(image, tuple(corner), 5, (0, 255, 0), -1)
                        cv2.putText(image, f"{idx}:{tuple(corner)}", tuple(corner), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 1)

                    return sorted_outer_corners

        return None

    def estimate_pen_tip(self, rvec, tvec, offset_y=-0.11):
        pen_tip_relative = np.array([[0], [offset_y], [0]], dtype=np.float32)  # (X=0, Y=-0.05, Z=0)

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        pen_tip_camera = rotation_matrix @ pen_tip_relative + tvec  # R * P + t

        pen_tip_2d, _ = cv2.projectPoints(pen_tip_camera.T, np.zeros((3, 1)), np.zeros((3, 1)),
                                          self.camera_matrix, self.dist_coeffs)

        return tuple(pen_tip_2d[0].ravel().astype(int))

    def sort_corners_with_aruco_axis(self, corners, rvec, tvec):
        corners = np.array(corners, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError(f"corners must have shape (4, 2), but got {corners.shape}")

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        aruco_coords = []
        for corner in corners:
            corner_homogeneous = np.array([[corner[0]], [corner[1]], [1]], dtype=np.float32)
            world_coord = np.linalg.inv(rotation_matrix) @ (corner_homogeneous - tvec)

            world_coord[1] *= -1
            aruco_coords.append(world_coord[:2].ravel())

        aruco_coords = np.array(aruco_coords)
        center = np.mean(aruco_coords, axis=0)

        def angle_from_center(point):
            delta = point - center
            return np.arctan2(delta[1], delta[0])

        sorted_indices = np.argsort([angle_from_center(point) for point in aruco_coords])
        sorted_corners = corners[sorted_indices]

        return sorted_corners

    def get_3d_coordinates(self, x, y, depth_frame):
        x_depth = int(x * (self.depth_width / self.color_width))
        y_depth = int(y * (self.depth_height / self.color_height))

        if 0 <= y_depth < self.depth_height and 0 <= x_depth < self.depth_width:
            z_depth = depth_frame[y_depth, x_depth] / 1000.0
        else:
            return None

        X = (x - self.camera_matrix[0, 2]) * z_depth / self.camera_matrix[0, 0]
        Y = (y - self.camera_matrix[1, 2]) * z_depth / self.camera_matrix[1, 1]
        Z = z_depth

        return np.array([X, Y, Z])

    def detect_and_estimate_pen_tip3(self, color_image, depth_image, dictionary_type=cv2.aruco.DICT_4X4_50,
                                     marker_length=0.0125, qrcode_size=0.03, pen_length=-0.097):
        outer_corners = self.detect_qrcode_with_corners(color_image, dictionary_type)

        sorted_corners = None
        pen_tip_3d_aruco = None
        pen_tip_3d_qrcode = None
        pen_tip_3d_depth = None
        pen_tip_3d_avg = None

        if outer_corners is not None:
            cv2.polylines(color_image, [np.int32(outer_corners)], True, (0, 255, 0), 3)

        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, corner in enumerate(corners):
                cv2.polylines(color_image, [np.int32(corner[0])], True, (0, 0, 255), 3)

                obj_points = np.array([
                    [-marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0],
                ], dtype=np.float32)

                _, rvec, tvec = cv2.solvePnP(obj_points, corner[0], self.camera_matrix, self.dist_coeffs)

                color_image = self.draw_axis(color_image, rvec, tvec)

                pen_tip_3d_aruco = tvec + np.array([0, pen_length, 0])
                pen_tip_2d_aruco = self.estimate_pen_tip(rvec, tvec, offset_y=pen_length)
                cv2.circle(color_image, pen_tip_2d_aruco, 5, (0, 255, 255), -1)
                cv2.putText(color_image,
                            f"Aruco Tip ({pen_tip_3d_aruco[0][0]:.2f},{pen_tip_3d_aruco[1][0]:.2f},{pen_tip_3d_aruco[2][0]:.2f})",
                            (pen_tip_2d_aruco[0] + 5, pen_tip_2d_aruco[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)

                if outer_corners is not None:
                    sorted_corners = self.sort_corners_with_aruco_axis(outer_corners, rvec, tvec)

                    cv2.polylines(color_image, [np.int32(sorted_corners)], True, (255, 255, 0), 3)

                    half_size = qrcode_size / 2
                    obj_points_qrcode = np.array([
                        [-half_size, half_size, 0],
                        [half_size, half_size, 0],
                        [half_size, -half_size, 0],
                        [-half_size, -half_size, 0],
                    ], dtype=np.float32)
                    _, rvec_qrcode, tvec_qrcode = cv2.solvePnP(obj_points_qrcode, sorted_corners, self.camera_matrix,
                                                               self.dist_coeffs)

                    color_image = self.draw_axis(color_image, rvec_qrcode, tvec_qrcode)

                    pen_tip_3d_qrcode = tvec_qrcode + np.array([0, pen_length, 0])
                    pen_tip_2d_qrcode = self.estimate_pen_tip(rvec_qrcode, tvec_qrcode, offset_y=pen_length)
                    cv2.circle(color_image, pen_tip_2d_qrcode, 5, (255, 20, 147), -1)
                    cv2.putText(color_image,
                                f"QRCode Tip ({pen_tip_3d_qrcode[0][0]:.2f},{pen_tip_3d_qrcode[1][0]:.2f},{pen_tip_3d_qrcode[2][0]:.2f})",
                                (pen_tip_2d_qrcode[0] + 5, pen_tip_2d_qrcode[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 20, 147), 1)

                if pen_tip_2d_aruco is not None:
                    pen_tip_3d_depth = self.get_3d_coordinates(int(pen_tip_2d_aruco[0]), int(pen_tip_2d_aruco[1]),
                                                               depth_image)
                    if pen_tip_3d_depth is not None:
                        cv2.circle(color_image, tuple(map(int, pen_tip_2d_aruco)), 5, (0, 128, 255), -1)
                        cv2.putText(color_image,
                                    f"Depth Tip ({pen_tip_3d_depth[0]:.2f},{pen_tip_3d_depth[1]:.2f},{pen_tip_3d_depth[2]:.2f})",
                                    (pen_tip_2d_aruco[0] + 5, pen_tip_2d_aruco[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 128, 255), 1)

                if pen_tip_3d_aruco is not None and pen_tip_3d_qrcode is not None and pen_tip_3d_depth is not None:
                    pen_tip_3d_avg = (pen_tip_3d_aruco + pen_tip_3d_qrcode + np.array(pen_tip_3d_depth).reshape(3,
                                                                                                                1)) / 3
                    avg_2d = (np.array(pen_tip_2d_aruco) + np.array(pen_tip_2d_qrcode)) // 2
                    cv2.circle(color_image, tuple(avg_2d), 5, (0, 0, 255), -1)
                    cv2.putText(color_image,
                                f"Avg Tip ({pen_tip_3d_avg[0][0]:.2f},{pen_tip_3d_avg[1][0]:.2f},{pen_tip_3d_avg[2][0]:.2f})",
                                (avg_2d[0] + 5, avg_2d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    cv2.putText(color_image, f"Avg Tip ({pen_tip_3d_avg[0][0]:.2f},{pen_tip_3d_avg[1][0]:.2f},{pen_tip_3d_avg[2][0]:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

        return color_image

    def write_tip_to_file(self, filename, tip_coordinates):
        with open(filename, 'a') as f:
            if len(tip_coordinates.shape) == 2 and tip_coordinates.shape[0] == 3:
                f.write(f"{tip_coordinates[0][0]:.4f}, {tip_coordinates[1][0]:.4f}, {tip_coordinates[2][0]:.4f}\n")
            elif len(tip_coordinates.shape) == 1 and tip_coordinates.shape[0] == 3:
                f.write(f"{tip_coordinates[0]:.4f}, {tip_coordinates[1]:.4f}, {tip_coordinates[2]:.4f}\n")
            else:
                print("Error: tip_coordinates is not in expected shape.")

    def show_frames(self):
        while True:
            start_time = time.time()

            color_frame = self.get_last_color()
            depth_frame = self.get_last_depth()

            if color_frame is not None and depth_frame is not None:
                color_frame = self.detect_and_estimate_pen_tip3(color_frame, depth_frame)
                # color_frame = self.detect_and_estimate_pen_tip_with_average(color_frame, depth_frame)
                cv2.imshow("RGB Image", color_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            time.sleep(max(1 / 10 - (time.time() - start_time), 0))

        cv2.destroyAllWindows()


if __name__ == "__main__":
    kinect = Kinect()
    kinect.show_frames()
