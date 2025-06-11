##这里就用来测试带回字的新aruco码设计的可行性了
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import time

class Kinect:
    def __init__(self):
        # 初始化 Kinect 设备，支持颜色、深度图
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
            #PyKinectRuntime来初始化kinect，括号内为要激活两个摄像头，返回一个PyKinectRuntime对象，包含PyKinectRuntime设备的所有接口
        self.color_width = 1920
        self.color_height = 1080
        self.depth_width = 512
        self.depth_height = 424  #kinect的深度影像默认分辨率

        # 摄像机内参矩阵（根据标定结果修改）
        self.camera_matrix = np.array([[1092.29214, 0.0, 951.696366],
                                       [0.0, 1092.82326, 496.561675],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
            # camera_matrix描述相机的内参（焦距和主点）
        self.dist_coeffs = np.array([[-0.00254295, 0.43106886, -0.01124835, 0.0046628, -0.75915729]], dtype=np.float32)
            # dist_coeffs 畸变系数，一个 4x1 的零矩阵，表示没有镜头畸变。

    def get_last_color(self):
        """ 获取 RGB 图像 """
        if self.kinect.has_new_color_frame():  #判断是否有新的rgb帧，返回的是布尔值，来源是PyKinectRuntime类，只是赋予名称为kinect
            frame = self.kinect.get_last_color_frame()  #一个【一维】数组，表示 RGB 图像的数据
            image = np.reshape(frame, [self.color_height, self.color_width, 4])  #通道数：4（包含 RGBA）
            image = image[:, :, 0:3][:,::-1,:].astype(np.uint8)  #忽略 Alpha 通道，并且镜像，将图像数据转换为8位无符号整数格式。
            return image
            # return cv2.resize(image, (960, 540))
        return None  #如果有新图像，就继续传图像，如果布尔值为0了，就返回None

    def get_last_depth(self):
        """ 获取原始 16 位深度图像 """
        if self.kinect.has_new_depth_frame():
            frame = self.kinect.get_last_depth_frame()
            depth_image = np.reshape(frame, [self.depth_height, self.depth_width])[:,::-1].astype(np.uint16)
            return depth_image  # 【16位】深度图像数据，单位【毫米】
        return None

    def draw_axis(self, image, rvec, tvec):
        """ 绘制三维坐标轴，用以验证pnp算法的正确性 """
        axis_length = 0.05  # 5cm  这是坐标轴显示的长度，不会影响具体的测试
        axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], [0, 0, 0]]).reshape(-1, 3)
            #这里的四个分别是，XYZ的各个轴显示的终点坐标，和原点坐标，最后把它们变成（4，3）这样的形状，4各点，每个点3个坐标值

        # 将3D坐标点投影到2D图像平面
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            #axis三维点坐标，rvec旋转向量，tvec平移向量，camera_matrix相机内参，dist_coeffs畸变系数
            #返回投影后的二维坐标imgpts，和投影矩阵，但一般用不到
            #本函数draw_axis的重点也就在这里了projectPoints

        origin = tuple(imgpts[3].ravel().astype(int))  # 原点
        #tuple创建元组，imgpts[3]为原点坐标，ravel将二维数组展平成一维数组，这里其实就是拿到原点坐标
        #【元组】是因为，cv2的line或者circle函数不接受numpy数组表示坐标，必须要用【元组】来作为二维坐标输入
        image = cv2.line(image, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)  # X轴-红
        image = cv2.line(image, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)  # Y轴-绿
        image = cv2.line(image, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)  # Z轴-蓝
        #用cv2.line画线，也就是画三个坐标轴

        return image

    def detect_qrcode_with_corners(self, image, dictionary_type=cv2.aruco.DICT_4X4_50):
        """
        检测回字外圈的四个角点，利用嵌套的 ArUco 码方向对角点排序并标记坐标。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #转灰度，cvtColor就是转色彩空间
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        #转二值，threshold返回两个值，一个是实际利用的阈值（一般不关心），和得到的二值图

        # 检测所有轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #返回值：contours包含轮廓点的列表，【每个轮廓】是一个二维坐标点的数组。  hierarchy层次结构，一般不关心

        # 初始化 ArUco 字典和检测器
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        for cnt in contours:
            # 逼近轮廓为多边形
            perimeter = cv2.arcLength(cnt, True)
            #arcLength计算轮廓的周长，True为限定轮廓是封闭的，【返回值】：轮廓的周长（浮点数）
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            #多边形拟合，0.02 * perimeter就是允许的误差，这一步是为了将轮廓拟合成四边形，True也是限定封闭轮廓
            #【返回值】：拟合后的多边形顶点数组。

            # 确保轮廓是一个四边形
            if len(approx) == 4:
                # 提取回字外圈的四个角点
                outer_corners = np.squeeze(approx, axis=1)
                #np.squeeze：移除多余的维度，确保 outer_corners 是一个形状为 [4, 2] 的数组，表示四个角点的坐标。

                # 在外圈内部寻找 ArUco 码
                x, y, w, h = cv2.boundingRect(approx)  #cv2.boundingRect：计算四边形的最小边界框。
                roi = gray[y:y + h, x:x + w]
                corners, ids, _ = detector.detectMarkers(roi)
                #这里的_返回是，检测到的附加信息，类似于存在二维码中的信息，这里我们不关注，信息本身也没有

                if ids is not None:
                    # 偏移 ArUco 码角点到全图坐标
                    aruco_corners = corners[0][0] + np.array([x, y])  # 第一个 ArUco 码角点

                    # 通过 ArUco 码的方向重新排序回字外框角点
                    sorted_outer_corners = [None] * 4
                    sorted_outer_corners[0] = outer_corners[np.argmin(np.sum(outer_corners, axis=1))]  # 左上
                    sorted_outer_corners[1] = outer_corners[np.argmax(outer_corners[:, 0] - outer_corners[:, 1])]  # 右上
                    sorted_outer_corners[2] = outer_corners[np.argmax(np.sum(outer_corners, axis=1))]  # 右下
                    sorted_outer_corners[3] = outer_corners[np.argmin(outer_corners[:, 0] - outer_corners[:, 1])]  # 左下

                    # 标记外圈角点及其坐标
                    for idx, corner in enumerate(sorted_outer_corners):
                        cv2.circle(image, tuple(corner), 5, (0, 255, 0), -1)  # 绿色点
                        cv2.putText(image, f"{idx}:{tuple(corner)}", tuple(corner), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 1)

                    return sorted_outer_corners  # 只返回最外圈的角点顺序

        return None  # 如果没有检测到回字外圈

    def estimate_pen_tip(self, rvec, tvec, offset_y=-0.11):  #这里是笔身长度
        """
        估算笔尖位置，基于 ArUco 码的三维姿态，沿 Y 轴负方向延伸指定长度。

        参数:
            rvec: ArUco 码的旋转向量 (3x1)
            tvec: ArUco 码的平移向量 (3x1)
            offset_y: 笔尖在 ArUco 坐标系 Y 轴负方向的偏移量 (单位: 米)

        返回:
            pen_tip_2d: 笔尖在图像坐标系中的二维坐标 (x, y)
        """
        # 笔尖相对于 【ArUco 坐标系】的三维坐标
        pen_tip_relative = np.array([[0], [offset_y], [0]], dtype=np.float32)  # (X=0, Y=-0.05, Z=0)

        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        #rvec是3*1形状的aruco坐标系相较于相机坐标系的旋转向量，用Rodrigues转化为3*3大小的旋转矩阵
        #【返回值】：rotation_matrix矩阵本身，‘_’是旋转矩阵的 Jacobian（这里未使用）

        # 计算笔尖在相机坐标系下的三维坐标，这里的tvec平移向量，就是两个坐标系【原点】的平移
        pen_tip_camera = rotation_matrix @ pen_tip_relative + tvec  # R * P + t

        # 将三维点投影到图像平面
        pen_tip_2d, _ = cv2.projectPoints(pen_tip_camera.T, np.zeros((3, 1)), np.zeros((3, 1)),
                                          self.camera_matrix, self.dist_coeffs)
        #其中pen_tip_camera.T，转置是为了把三维坐标变成1*3的形状，可能是之前R * P + t得到的坐标是3*1的
        #两个np.zeros((3, 1))就是tvec和rvec都为0矩阵，表示当前笔尖坐标其实都已经是在相机坐标系下了。
        #后面两个是self定义的参数，就是相机内参和畸变系数
        #projectPoints的【返回值】为：二维坐标值，和投影过程的 Jacobian（未使用）

        # 返回二维投影坐标 (x, y)
        return tuple(pen_tip_2d[0].ravel().astype(int))
        #pen_tip_2d[0]取二维坐标，展平，然后取元组变成(x,y)的形式

    def sort_corners_with_aruco_axis(self, corners, rvec, tvec):
        """
        利用 ArUco 的三维坐标轴划分象限，为回字外框的角点排序。
        确保顺序与 ArUco 的默认顺序一致：左上、右上、右下、左下。
        """
        # 确保 corners 是 numpy 数组
        corners = np.array(corners, dtype=np.float32)
        if corners.shape != (4, 2):  #确保角点的形状是4*2，也就是每一个点都是1*2的
            raise ValueError(f"corners must have shape (4, 2), but got {corners.shape}")

        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        #因为和aruco码在同一平面，享受同一个旋转和平移向量，用Rodrigues得到同一个旋转矩阵

        # 将角点从图像坐标系转换到 ArUco 坐标系
        aruco_coords = []
        for corner in corners:
            corner_homogeneous = np.array([[corner[0]], [corner[1]], [1]], dtype=np.float32)
            world_coord = np.linalg.inv(rotation_matrix) @ (corner_homogeneous - tvec)

            # 反转 Y 轴方向以符合 ArUco 的默认顺序
            world_coord[1] *= -1
            aruco_coords.append(world_coord[:2].ravel())  # 投影到 XOY 平面

        # 转换为 numpy 数组并划分象限
        aruco_coords = np.array(aruco_coords)
        center = np.mean(aruco_coords, axis=0)  # 计算中心点

        def angle_from_center(point):
            # 计算相对于中心点的角度
            delta = point - center
            return np.arctan2(delta[1], delta[0])  # 计算角度

        # 按角度排序（ArUco 默认逆时针顺序）
        sorted_indices = np.argsort([angle_from_center(point) for point in aruco_coords])
        sorted_corners = corners[sorted_indices]  # 修复索引问题：保证是 numpy 数组操作

        return sorted_corners

    def get_3d_coordinates(self, x, y, depth_frame):
        """
        根据像素坐标 (x, y) 和深度图，计算三维世界坐标。
        """
        # 映射到深度图分辨率
        x_depth = int(x * (self.depth_width / self.color_width))
        y_depth = int(y * (self.depth_height / self.color_height))

        # 获取深度值
        if 0 <= y_depth < self.depth_height and 0 <= x_depth < self.depth_width:
            z_depth = depth_frame[y_depth, x_depth] / 1000.0  # 转换为米
        else:
            return None

        # 使用相机内参计算世界坐标
        X = (x - self.camera_matrix[0, 2]) * z_depth / self.camera_matrix[0, 0]
        Y = (y - self.camera_matrix[1, 2]) * z_depth / self.camera_matrix[1, 1]
        Z = z_depth

        return np.array([X, Y, Z])

    # def detect_and_estimate_pen_tip3(self, color_image, depth_image, dictionary_type=cv2.aruco.DICT_4X4_50,
    #                                  marker_length=0.03, qrcode_size=0.07, pen_length=-0.10):
    def detect_and_estimate_pen_tip3(self, color_image, depth_image, dictionary_type=cv2.aruco.DICT_4X4_50,
                                     marker_length=0.0125, qrcode_size=0.03, pen_length=-0.097):
        """
        检测回字外框、嵌套的 ArUco 码，同时推理笔尖位置，绘制回字外框和 ArUco 的三维坐标轴，并排序回字角点。
        """
        # 检测回字外框及其四个角点
        outer_corners = self.detect_qrcode_with_corners(color_image, dictionary_type)

        sorted_corners = None  # 存储排序后的回字角点
        pen_tip_3d_aruco = None  # ArUco 笔尖三维坐标
        pen_tip_3d_qrcode = None  # 回字外框笔尖三维坐标
        pen_tip_3d_depth = None  # 深度推理笔尖三维坐标
        pen_tip_3d_avg = None  # 平均笔尖三维坐标

        if outer_corners is not None:
            # 绘制回字外框的绿色框
            cv2.polylines(color_image, [np.int32(outer_corners)], True, (0, 255, 0), 3)

        # 初始化 ArUco 字典和参数
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 检测 ArUco 码
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, corner in enumerate(corners):
                # 绘制红色框，表示 ArUco 码的外框
                cv2.polylines(color_image, [np.int32(corner[0])], True, (0, 0, 255), 3)

                # ArUco 码的三维角点坐标
                obj_points = np.array([
                    [-marker_length / 2, marker_length / 2, 0],  # 左上
                    [marker_length / 2, marker_length / 2, 0],  # 右上
                    [marker_length / 2, -marker_length / 2, 0],  # 右下
                    [-marker_length / 2, -marker_length / 2, 0],  # 左下
                ], dtype=np.float32)

                # 使用 PnP 进行姿态估计
                _, rvec, tvec = cv2.solvePnP(obj_points, corner[0], self.camera_matrix, self.dist_coeffs)

                # 绘制三维坐标轴
                color_image = self.draw_axis(color_image, rvec, tvec)

                # 推理笔尖位置 (基于 ArUco 坐标轴)
                pen_tip_3d_aruco = tvec + np.array([0, pen_length, 0])  # 笔尖在三维空间的坐标
                pen_tip_2d_aruco = self.estimate_pen_tip(rvec, tvec, offset_y=pen_length)
                cv2.circle(color_image, pen_tip_2d_aruco, 5, (0, 255, 255), -1)  # 标记笔尖位置（黄色圆点）
                cv2.putText(color_image,
                            f"Aruco Tip ({pen_tip_3d_aruco[0][0]:.2f},{pen_tip_3d_aruco[1][0]:.2f},{pen_tip_3d_aruco[2][0]:.2f})",
                            (pen_tip_2d_aruco[0] + 5, pen_tip_2d_aruco[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)

                # 排序回字外框角点
                if outer_corners is not None:
                    sorted_corners = self.sort_corners_with_aruco_axis(outer_corners, rvec, tvec)

                    # 绘制排序后的回字外框
                    cv2.polylines(color_image, [np.int32(sorted_corners)], True, (255, 255, 0), 3)

                    # 回字外框的 PnP 计算
                    half_size = qrcode_size / 2
                    obj_points_qrcode = np.array([
                        [-half_size, half_size, 0],  # 左上
                        [half_size, half_size, 0],  # 右上
                        [half_size, -half_size, 0],  # 右下
                        [-half_size, -half_size, 0],  # 左下
                    ], dtype=np.float32)
                    _, rvec_qrcode, tvec_qrcode = cv2.solvePnP(obj_points_qrcode, sorted_corners, self.camera_matrix,
                                                               self.dist_coeffs)

                    # 绘制回字外框的三维坐标轴
                    color_image = self.draw_axis(color_image, rvec_qrcode, tvec_qrcode)

                    # 推理笔尖位置 (基于回字坐标轴)
                    pen_tip_3d_qrcode = tvec_qrcode + np.array([0, pen_length, 0])  # 笔尖在三维空间的坐标
                    pen_tip_2d_qrcode = self.estimate_pen_tip(rvec_qrcode, tvec_qrcode, offset_y=pen_length)
                    cv2.circle(color_image, pen_tip_2d_qrcode, 5, (255, 20, 147), -1)  # 标记笔尖位置（粉色圆点）
                    cv2.putText(color_image,
                                f"QRCode Tip ({pen_tip_3d_qrcode[0][0]:.2f},{pen_tip_3d_qrcode[1][0]:.2f},{pen_tip_3d_qrcode[2][0]:.2f})",
                                (pen_tip_2d_qrcode[0] + 5, pen_tip_2d_qrcode[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 20, 147), 1)

                # 基于深度图的笔尖位置推理
                if pen_tip_2d_aruco is not None:
                    pen_tip_3d_depth = self.get_3d_coordinates(int(pen_tip_2d_aruco[0]), int(pen_tip_2d_aruco[1]),
                                                               depth_image)
                    if pen_tip_3d_depth is not None:
                        cv2.circle(color_image, tuple(map(int, pen_tip_2d_aruco)), 5, (0, 128, 255), -1)  # 深度推理的笔尖
                        cv2.putText(color_image,
                                    f"Depth Tip ({pen_tip_3d_depth[0]:.2f},{pen_tip_3d_depth[1]:.2f},{pen_tip_3d_depth[2]:.2f})",
                                    (pen_tip_2d_aruco[0] + 5, pen_tip_2d_aruco[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 128, 255), 1)

                # 计算三个笔尖位置的平均值
                if pen_tip_3d_aruco is not None and pen_tip_3d_qrcode is not None and pen_tip_3d_depth is not None:
                    pen_tip_3d_avg = (pen_tip_3d_aruco + pen_tip_3d_qrcode + np.array(pen_tip_3d_depth).reshape(3,
                                                                                                                1)) / 3
                    avg_2d = (np.array(pen_tip_2d_aruco) + np.array(pen_tip_2d_qrcode)) // 2
                    cv2.circle(color_image, tuple(avg_2d), 5, (0, 0, 255), -1)  # 标记平均位置（红色圆点）
                    cv2.putText(color_image,
                                f"Avg Tip ({pen_tip_3d_avg[0][0]:.2f},{pen_tip_3d_avg[1][0]:.2f},{pen_tip_3d_avg[2][0]:.2f})",
                                (avg_2d[0] + 5, avg_2d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    cv2.putText(color_image, f"Avg Tip ({pen_tip_3d_avg[0][0]:.2f},{pen_tip_3d_avg[1][0]:.2f},{pen_tip_3d_avg[2][0]:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

        return color_image

    def write_tip_to_file(self, filename, tip_coordinates):
        """ 将笔尖三维坐标写入文件 """
        with open(filename, 'a') as f:
            # 如果 tip_coordinates 是二维数组（3, 1），就提取每个分量
            if len(tip_coordinates.shape) == 2 and tip_coordinates.shape[0] == 3:
                f.write(f"{tip_coordinates[0][0]:.4f}, {tip_coordinates[1][0]:.4f}, {tip_coordinates[2][0]:.4f}\n")
            # 如果 tip_coordinates 是一维数组（3,），直接提取每个分量
            elif len(tip_coordinates.shape) == 1 and tip_coordinates.shape[0] == 3:
                f.write(f"{tip_coordinates[0]:.4f}, {tip_coordinates[1]:.4f}, {tip_coordinates[2]:.4f}\n")
            else:
                print("Error: tip_coordinates is not in expected shape.")

    def show_frames(self):
        """ 循环显示 RGB 和深度图像，同时检测回字外框和 ArUco 码 """
        while True:
            start_time = time.time()

            color_frame = self.get_last_color()
            depth_frame = self.get_last_depth()

            if color_frame is not None and depth_frame is not None:
                # 检测回字外框与嵌套的 ArUco 码
                color_frame = self.detect_and_estimate_pen_tip3(color_frame, depth_frame)
                # color_frame = self.detect_and_estimate_pen_tip_with_average(color_frame, depth_frame)
                cv2.imshow("RGB Image", color_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 'q' 键退出
                break

            time.sleep(max(1 / 10 - (time.time() - start_time), 0))

        cv2.destroyAllWindows()


if __name__ == "__main__":
    kinect = Kinect()  # 实例化 Kinect 类
    kinect.show_frames()  # 调用显示函数
