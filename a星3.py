import cv2
import numpy as np
import math
from PIL import Image


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0


def astar(start, end, map):
    open_list = []
    closed_list = []

    open_list.append(start)

    while open_list:
        current_node = open_list[0]
        current_index = 0

        for index, node in enumerate(open_list):
            if node.f < current_node.f:
                current_node = node
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node.x == end.x and current_node.y == end.y:
            path = []
            current = current_node
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]  # 返回反转的路径

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_x = current_node.x + new_position[0]
            node_y = current_node.y + new_position[1]

            if node_x < 0 or node_x >= len(map) or node_y < 0 or node_y >= len(map[0]):
                continue

            if map[node_x][node_y] == 1:  # 1 represents an obstacle
                continue

            new_node = Node(node_x, node_y, current_node)
            children.append(new_node)

        for child in children:
            if child in closed_list:
                continue

            child.g = current_node.g + 1
            child.h = math.sqrt((child.x - end.x) ** 2 + (child.y - end.y) ** 2)
            child.f = child.g + child.h

            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)
            map[child.x][child.y] = 1  # Mark the node as visited

    return None  # 如果无法找到路径，返回None


def convert_to_bw(image_path, threshold):
    # 打开图像
    image = Image.open(image_path)

    # 将图像转换为灰度
    bw_image = image.convert("L")

    # 应用阈值将灰度值转换为黑白
    bw_image = bw_image.point(lambda x: 0 if x < threshold else 255, "1")

    # 保存黑白图像
    bw_image.save("black_and_white.png")

    print("黑白图像已保存！")


# 调用函数，传入图像路径和阈值
convert_to_bw("img.png", 180)

# Read input image
image = cv2.imread("black_and_white.png", 0)  # Read the black and white image

# Set map dimensions
map_height = image.shape[0]
map_width = image.shape[1]

# Convert the image to map data
map_data = np.zeros((map_height, map_width), dtype=np.uint8)
for i in range(map_height):
    for j in range(map_width):
        if image[i][j] == 0:  # Black pixel
            map_data[i][j] = 1  # Represents an obstacle
        else:  # White pixel
            map_data[i][j] = 0  # Represents a passable area

# Define start and end nodes
start_node = Node(230, 260)  # Note: I swapped the x and y coordinates to match the image dimensions
end_node = Node(310, 310)  # Note: I swapped the x and y coordinates to match the image dimensions

# Run A* algorithm for pathfinding
path = astar(start_node, end_node, map_data)
if path:
    print("Found path:", path)

    # Mark the path on a new image
    img2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Create a color image copy

    for x, y in path:
        cv2.circle(img2, (y, x), 3, (0, 0, 255), -1)  # Draw a red circle on the points of the path

    # Save the image
    cv2.imwrite("img2.png", img2)
else:
    print("Unable to find a path")
