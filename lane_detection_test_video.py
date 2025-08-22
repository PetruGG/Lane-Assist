import cv2
import numpy as np


def to_grayscale(frame):
    height, width, _ = frame.shape

    gray_frame = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            b, g, r = frame[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_frame[i, j] = gray_value

    return gray_frame


def create_trapezoid(frame):
    height, width, _ = frame.shape

    upper_left = (int(width * 0.45), int(height * 0.75))
    upper_right = (int(width * 0.55), int(height * 0.75))
    lower_right = (int(width * 1.0), int(height * 0.98))
    lower_left = (int(width * 0.0), int(height * 0.98))

    points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    trapezoid_frame = np.zeros((height, width), dtype=np.uint8)

    cv2.fillConvexPoly(trapezoid_frame, points, 1)

    return trapezoid_frame


def stretch_frame(frame, trapezoid_frame):
    height, width, _ = frame.shape

    trapezoid_points = np.array([
        (int(width * 0.45), int(height * 0.75)),
        (int(width * 0.55), int(height * 0.75)),
        (int(width * 1.0), int(height * 0.98)),
        (int(width * 0.0), int(height * 0.98))
    ], dtype=np.float32)

    frame_points = np.array([
        (0, 0),
        (width, 0),
        (width, height),
        (0, height),
    ], dtype=np.float32)

    magic_matrix = cv2.getPerspectiveTransform(trapezoid_points, frame_points)
    stretched_frame = cv2.warpPerspective(trapezoid_frame, magic_matrix, (width, height))

    return stretched_frame


def apply_blur(frame, kernel_size=(5, 5)):
    blurred_frame = cv2.blur(frame, ksize=kernel_size)
    return blurred_frame


def apply_sobel_filter(frame):
    sobel_vertical = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    sobel_horizontal = np.transpose(sobel_vertical)

    frame_float32 = np.float32(frame)

    sobel_vertical_result = cv2.filter2D(frame_float32, -1, sobel_vertical)

    sobel_horizontal_result = cv2.filter2D(frame_float32, -1, sobel_horizontal)

    return sobel_vertical_result, sobel_horizontal_result


def combine_sobel_result(sobel_v, sobel_h):
    matrix3 = np.sqrt(np.square(sobel_v) + np.square(sobel_h))

    matrix3_uint8 = cv2.convertScaleAbs(matrix3)

    return matrix3_uint8


def binarize_frame(frame, threshold=255 // 2):
    _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    return binary_frame


def remove_edge_noise(binary_frame):
    height, width = binary_frame.shape
    margin = int(width * 0.1)

    clean_frame = binary_frame.copy()
    clean_frame[:, :margin] = 0
    clean_frame[:, -margin:] = 0

    return clean_frame


def extract_lane_coordinates(binary_frame):
    height, width = binary_frame.shape
    mid = width // 2

    left_half = binary_frame[:, :mid]
    left_points = np.argwhere(left_half == 255)
    left_y, left_x = left_points[:, 0], left_points[:, 1]

    right_half = binary_frame[:, mid:]
    right_points = np.argwhere(right_half == 255)
    right_y, right_x = right_points[:, 0], right_points[:, 1]
    right_x += mid

    return left_x, left_y, right_x, right_y


def find_lines(binary_frame, left_x, left_y, right_x, right_y):
    global prev_values
    height, width = binary_frame.shape

    b_left, a_left = np.polynomial.polynomial.polyfit(left_x, left_y, 1)
    b_right, a_right = np.polynomial.polynomial.polyfit(right_x, right_y, 1)

    left_top_y = 0
    left_top_x = int((left_top_y - b_left) / a_left)

    left_bottom_y = height
    left_bottom_x = int((left_bottom_y - b_left) / a_left)

    right_top_y = 0
    right_top_x = int((right_top_y - b_right) / a_right)

    right_bottom_y = height
    right_bottom_x = int((right_bottom_y - b_right) / a_right)

    if not -10 ** 8 <= left_top_x <= 10 ** 8:
        left_top_x = prev_values["left_top"]
    else:
        prev_values["left_top"] = left_top_x
    if not -10 ** 8 <= left_bottom_x <= 10 ** 8:
        left_bottom_x = prev_values["left_bottom"]
    else:
        prev_values["left_bottom"] = left_bottom_x
    if not -10 ** 8 <= right_top_x <= 10 ** 8:
        right_top_x = prev_values["right_top"]
    else:
        prev_values["right_top"] = right_top_x
    if not -10 ** 8 <= right_bottom_x <= 10 ** 8:
        right_bottom_x = prev_values["right_bottom"]
    else:
        prev_values["right_bottom"] = right_bottom_x

    lines_frame = binary_frame.copy()
    cv2.line(lines_frame, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (200, 0, 0), 5)
    cv2.line(lines_frame, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (100, 0, 0), 5)
    cv2.line(lines_frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

    return lines_frame


def final_visualization(frame, left_top_x, left_bottom_x, right_top_x, right_bottom_x):
    height, width = frame.shape[:2]

    trapezoid_points = np.array([
        (int(width * 0.45), int(height * 0.75)),
        (int(width * 0.55), int(height * 0.75)),
        (int(width * 1.0), int(height * 0.98)),
        (int(width * 0.0), int(height * 0.98))
    ], dtype=np.float32)

    frame_points = np.array([
        (0, 0),
        (width, 0),
        (width, height),
        (0, height),
    ], dtype=np.float32)

    left_blank = np.zeros((height, width), np.uint8)
    cv2.line(left_blank, (left_top_x, 0), (left_bottom_x, height), (255, 0, 0), 3)
    magic_matrix = cv2.getPerspectiveTransform(frame_points, trapezoid_points)
    left_warped = cv2.warpPerspective(left_blank, magic_matrix, (width, height))
    left_line_points = np.argwhere(left_warped == 255)
    left_line_y, left_line_x = left_line_points[:, 0], left_line_points[:, 1]

    right_blank = np.zeros((height, width), np.uint8)
    cv2.line(right_blank, (right_top_x, 0), (right_bottom_x, height), (255, 0, 0), 3)
    right_warped = cv2.warpPerspective(right_blank, magic_matrix, (width, height))
    right_line_points = np.argwhere(right_warped == 255)
    right_line_y, right_line_x = right_line_points[:, 0], right_line_points[:, 1]

    result_frame = frame.copy()

    for i in range(len(left_line_y)):
        y, x = left_line_y[i], left_line_x[i]
        if 0 <= y < height and 0 <= x < width:
            result_frame[y, x] = (50, 50, 250)

    for i in range(len(right_line_y)):
        y, x = right_line_y[i], right_line_x[i]
        if 0 <= y < height and 0 <= x < width:
            result_frame[y, x] = (50, 250, 50)

    return result_frame


cam = cv2.VideoCapture('Lane_detection_Test_Video_01_2.mp4')
prev_values = {"left_top": 0, "left_bottom": 0, "right_top": 0, "right_bottom": 0}

while True:
    ret, frame = cam.read()
    if ret is False:
        break

    frame_width, frame_height, x = frame.shape
    resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 6))

    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    trapezoid_frame = create_trapezoid(resized_frame)

    road_frame = gray_frame * trapezoid_frame

    stretched_frame = stretch_frame(resized_frame, road_frame)

    blurred_frame = apply_blur(stretched_frame, kernel_size=(5, 5))

    sobel_vertical, sobel_horizontal = apply_sobel_filter(blurred_frame)

    sobel_result = combine_sobel_result(sobel_vertical, sobel_horizontal)

    binary_frame = binarize_frame(sobel_result, threshold=220 // 2)

    clean_frame = remove_edge_noise(binary_frame)
    left_x, left_y, right_x, right_y = extract_lane_coordinates(binary_frame)

    lines_frame = find_lines(clean_frame, left_x, left_y, right_x, right_y)

    left_top_x = prev_values["left_top"]
    left_bottom_x = prev_values["left_bottom"]
    right_top_x = prev_values["right_top"]
    right_bottom_x = prev_values["right_bottom"]

    final_frame = final_visualization(resized_frame, left_top_x, left_bottom_x, right_top_x, right_bottom_x)

    cv2.imshow('Small', resized_frame)
    cv2.imshow('Grayscale', gray_frame)
    cv2.imshow('Trapezoid', 255 * trapezoid_frame)
    cv2.imshow('Road', road_frame)
    cv2.imshow('Top-Down', stretched_frame)
    cv2.imshow('Blur', blurred_frame)
    cv2.imshow('Sobel', sobel_result)
    cv2.imshow('Binary', binary_frame)
    cv2.imshow('Clean', clean_frame)
    cv2.imshow('Lines', lines_frame)
    cv2.imshow('Final', final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
