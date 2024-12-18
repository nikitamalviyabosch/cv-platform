class IOUUtils:
    def calculate_iou(self, box1, box2):
        box1_dims = [box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]]
        box2_dims = [box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1]]
        x_intersection = max(box1_dims[0], box2_dims[0])
        y_intersection = max(box1_dims[1], box2_dims[1])
        w_intersection = min(box1_dims[0] + box1_dims[2], box2_dims[0] + box2_dims[2]) - x_intersection
        h_intersection = min(box1_dims[1] + box1_dims[3], box2_dims[1] + box2_dims[3]) - y_intersection
        area_intersection = max(0, w_intersection) * max(0, h_intersection)
        area_box1 = box1_dims[2] * box1_dims[3]
        area_box2 = box2_dims[2] * box2_dims[3]
        area_union = area_box1 + area_box2 - area_intersection
        # iou = area_intersection / max(area_union, 1e-6)
        ### to handle edge cases, (where one or both bounding boxes might have zero area, and preventing division by zero errors. This makes the IoU calculation more reliable in a variety of scenarios) took the iou approach as below 
        iou = area_intersection / max(min(area_box1,area_box2), 1e-6)
        return iou
    
    def remove_duplicates(self, main_list, sublist):
        ### removing the same BB which are processed for IoU
        return [item for item in main_list if item not in sublist]