import numpy as np
from numba import njit


# @njit
def nms_viola_jones(boxes, iouThresh=0.5, overlapThresh=3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return boxes

    output_boxes = np.zeros((len(boxes), 5), dtype=np.int32)
    out_idx = 0

    # while there are boxes
    remaining_boxes = len(boxes)
    while remaining_boxes > 0:
        # find all the boxes that overlap with the current agg boxes more than iouThresh
        # loop over the boxes
        countOverlapping = 1
        sum_x1, sum_y1, sum_x2, sum_y2 = boxes[0]

        curr_idx = 0
        for i in range(1, remaining_boxes):
            # get the coordinates and the area of the current box
            xc1, yc1, xc2, yc2 = int(sum_x1 / countOverlapping), int(sum_y1 / countOverlapping), \
                int(sum_x2 / countOverlapping), int(sum_y2 / countOverlapping)
            areac = (xc2 - xc1 + 1) * (yc2 - yc1 + 1)

            # get the coordinates of the ith box
            xi1, yi1, xi2, yi2 = boxes[i]
            # calculate the area of the ith box
            areai = (xi2 - xi1 + 1) * (yi2 - yi1 + 1)
            # calculate the coordinates of the intersection of the ith box and the current box
            xo1, yo1 = max(xc1, xi1), max(yc1, yi1)
            xo2, yo2 = min(xc2, xi2), min(yc2, yi2)

            # calculate the length and the width of the intersection
            w = max(0, xo2 - xo1 + 1)
            h = max(0, yo2 - yo1 + 1)

            # calculate the area of the  overlap between the ith box and the current box
            area_intersection = w * h

            # calculate the area of the union of the ith box and the current box
            area_union = areac + areai - area_intersection

            # calculate the ratio of overlap (iou)
            iou = area_intersection / area_union

            # if the overlap is greater than iouThresh
            # increment the count of overlapping boxes
            # mark the ith box as overlapping
            if iou >= iouThresh:
                countOverlapping += 1
                sum_x1 += xi1
                sum_y1 += yi1
                sum_x2 += xi2
                sum_y2 += yi2
            else:
                boxes[curr_idx][0] = boxes[i][0]
                boxes[curr_idx][1] = boxes[i][1]
                boxes[curr_idx][2] = boxes[i][2]
                boxes[curr_idx][3] = boxes[i][3]
                curr_idx += 1

        # if number of overlapping boxes is greater than or equal to overlapThresh
        # get the average of the overlapping boxes to calculate the new coordinates of the current box and the score
        avg_x1 = int(sum_x1 / countOverlapping)
        avg_y1 = int(sum_y1 / countOverlapping)
        avg_x2 = int(sum_x2 / countOverlapping)
        avg_y2 = int(sum_y2 / countOverlapping)
        avg_w = avg_x2 - avg_x1 + 1
        avg_h = avg_y2 - avg_y1 + 1
        avg_side = int((avg_w + avg_h) / 2)
        # score = int(countOverlapping / avg_side * 96)
        # if avg_side <= 30:
        #     score = int(score * 2 / 3)
        if countOverlapping >= overlapThresh:
            # get the average of the overlapping boxes and the first box and add it to output boxes

            output_boxes[out_idx][0] = avg_x1
            output_boxes[out_idx][1] = avg_y1
            output_boxes[out_idx][2] = avg_x2
            output_boxes[out_idx][3] = avg_y2
            output_boxes[out_idx][4] = countOverlapping
            out_idx += 1

        # update the number of remaining boxes
        remaining_boxes = curr_idx

    # no more boxes remaining return the output boxes

    return output_boxes[:out_idx]


class DFS:

    def __init__(self):
        self.visited = set()
        self.graph = {}
        self.components = []

    def initialize_graph(self, boxes):
        for i, box1 in enumerate(boxes):
            self.graph[i] = set()
            for j, box2 in enumerate(boxes):
                if i != j and calculate_overlap(box1, box2) >= 0.5:
                    self.graph[i].add(j)

    def dfs(self, node):
        self.visited.add(node)
        self.current_component.append(node)
        for neighbor in self.graph[node]:
            if neighbor not in self.visited:
                self.dfs(neighbor)

    def get_connected_components(self):
        for node in self.graph:
            if node not in self.visited:
                self.current_component = []
                self.dfs(node)
                self.components.append(self.current_component)

        return self.components


def merge_overlapping_boxes(boxes, overlapThresh):
    # initialize the DFS class
    dfs = DFS()
    # initialize the graph
    dfs.initialize_graph(boxes)
    # get the connected components
    connected_components = dfs.get_connected_components()

    # initialize the output boxes
    output_boxes = []

    # loop over the connected components
    for connected_component in connected_components:
        x1, y1, x2, y2, score = 0, 0, 0, 0, 0
        # loop over the boxes in the connected component
        # weight the coordinates of the boxes by their scores
        for idx in connected_component:
            box = boxes[idx]
            x1 += box[0] * box[4]
            y1 += box[1] * box[4]
            x2 += box[2] * box[4]
            y2 += box[3] * box[4]
            score += box[4]

        # average the weighted coordinates
        x1 = int(x1 / score)
        y1 = int(y1 / score)
        x2 = int(x2 / score)
        y2 = int(y2 / score)

        # add the box to the output boxes
        output_boxes.append([x1, y1, x2, y2])

    return np.array(output_boxes)


def calculate_overlap(box1, box2):
    x_min1, y_min1, x_max1, y_max1, score1 = box1
    x_min2, y_min2, x_max2, y_max2, score2 = box2

    # Calculate the coordinates of the overlapping region
    x_min = max(x_min1, x_min2)
    y_min = max(y_min1, y_min2)
    x_max = min(x_max1, x_max2)
    y_max = min(y_max1, y_max2)

    # Calculate the area of the overlapping region
    overlap_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    # Calculate the areas of both boxes
    box1_area = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
    box2_area = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)

    # Calculate the overlap ratio
    overlap_ratio1 = overlap_area / float(
        box1_area + box2_area - overlap_area)  # iou of box1 and box2
    overlap_ratio2 = overlap_area / float(min(
        box1_area, box2_area))  # iou of box2 with box1

    return max(overlap_ratio1, overlap_ratio2)


@njit
def iou2(box1, box2):
    # calculate the intersection over union of two boxes
    # box1 given slightly less weight than box2 since box1 is the ground truth of the head which is larger than the face
    # box1 and box2 are numpy arrays of shape (4,)
    # box1 = [x1, y1, x2, y2]
    # box2 = [x1, y1, x2, y2]
    # intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area1 = intersection + (area1 - intersection) * 0.5
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union