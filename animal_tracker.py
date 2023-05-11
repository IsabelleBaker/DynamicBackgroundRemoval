import numpy as np
import torch
import copy
import pickle
import cv2


# This class is used to track objects (animals). I started this by studying a few other algorithms. However, this
# now has almost no resemblance directly to anything I've read.
class animal_tracker:
    def __init__(self,
                 score_threshold: float = 0.5,
                 lost_threshold: int = 120,
                 class_index: int = 0,
                 max_distance_moved: float = 5,
                 tracking_iou_threshold: float = 0.4,
                 save_trackers: bool = False
                 ):
        self.active_tracklets = {"animals": {}}
        self.finished_tracklets = {}
        self.score_threshold = score_threshold
        self.class_index = class_index
        self.save_trackers = save_trackers

        # How many frames can an animal not be detected before it is considered lost for good.
        self.lost_threshold = lost_threshold

        # The maximum distance an animal can move and still be considered the same animal as a
        # percentage of the diagonal distance of the animal's detection box. This auto-scales with
        # animals
        self.max_distance_moved = max_distance_moved

        # Internal class variable for ensuring animal ids do not get reused
        self.global_animal_id = 0

        self.tracking_iou_threshold = tracking_iou_threshold

    def track(self, detected_boxes, detected_pred_classes, detected_scores, detected_masks,
              frame_number, size=(100, 100),
              max_size: int = 500000):

        # Only include the animals (boxes) that have a detection score high enough and that match the
        # "type" of thing we are looking for with this tracker. If more than one type of thing (rat, larva, horse,
        # etc) are detected, they must be tracked in their own tracker.  That way you can have rat_1->rat_n,
        # larva_1->larva_n, etc. and consistently match the correct animal.  Also, this way you can have separate
        # thresholds for distance, lost threshold, etc.
        mask = torch.logical_and(
            detected_pred_classes == self.class_index, detected_scores > self.score_threshold
        )
        detected_boxes = detected_boxes[mask]

        # If animals were detected (boxes) then process them.  Otherwise, just check to see if any animals are
        # permanently lost.
        if len(detected_boxes) > 0:
            detected_scores = detected_scores.unsqueeze(dim=1)
            if len(mask) == 1:
                detected_scores = [detected_scores[mask], ]
            else:
                detected_scores = detected_scores[mask]
            temp_masks = []
            detected_contours = []
            for idx, mask_item in enumerate(detected_masks):
                if mask[idx]:
                    temp_masks.append(detected_masks[idx])
                    temp = detected_masks[idx].detach().cpu().numpy().astype(np.uint8)
                    contours, hierarchy = cv2.findContours(image=temp,
                                                           mode=cv2.RETR_EXTERNAL,
                                                           method=cv2.CHAIN_APPROX_NONE)
                    detected_contours.append(contours)
            detected_masks = temp_masks

            # If we are already tracking animals then analyze, otherwise they are all new
            if len(self.active_tracklets['animals']) > 0:
                animal_list = []
                mapped_animals = []
                leftover_animals = []
                # Loop through all detected boxes and calculate their centers
                for i in list(self.active_tracklets['animals']):
                    animal_list.append(self.active_tracklets['animals'][i]['animal_id'])
                temp_boxes = copy.deepcopy(detected_boxes)
                temp_masks = copy.deepcopy(detected_masks)

                # make iou matrix
                iou_list = []

                # For each existing animal, Make a matrix of IoU to all detected boxes
                for animal_index in animal_list:
                    animal_last_box = self.active_tracklets['animals'][animal_index][
                        'boxes'][-1]
                    animal_last_box = animal_last_box.unsqueeze(dim=0)

                    # Add the current animal's center to the first element of the box list
                    iou_checker = torch.cat((animal_last_box, temp_boxes), dim=0)
                    ious = self.get_iou(iou_checker)
                    ious = ious[1:]
                    iou_list.append(ious)
                iou_list = torch.stack(iou_list)
                animal_index = 0

                # Match IoU to existing animals while we haven't mapped them all and have more unmatched boxes
                while len(animal_list) > 0 and len(temp_boxes) > 0:
                    # find the best box for this animal based on iou
                    best_center_index = torch.argmax(iou_list[animal_index, :])
                    # check to make sure that the iou is greater than the minimum threshold
                    if iou_list[animal_index, best_center_index] > self.tracking_iou_threshold:
                        # find the animal with the best iou for this box
                        centers_best_animal_index = torch.argmax(iou_list[:, best_center_index])
                        # check if the current animal is the best match. If so, it's a MATCH!
                        if centers_best_animal_index == animal_index:
                            # make sure the value of iou is above the threshold...otherwise this is a lost animal
                            # We have a match!
                            mapped_animals.append((animal_list[animal_index], temp_boxes[best_center_index]))
                            animal_list.pop(animal_index)
                            temp_boxes = torch.cat(
                                (temp_boxes[0:best_center_index, :], temp_boxes[best_center_index + 1:, :]), dim=0)
                            temp_masks.pop(best_center_index)
                            iou_list = torch.cat((iou_list[0:animal_index, :], iou_list[animal_index + 1:, :]), dim=0)
                            iou_list = torch.cat(
                                (iou_list[:, 0:best_center_index], iou_list[:, best_center_index + 1:]), dim=1)
                            animal_index = 0
                        # If there was an animal that was a better match, change to that animal and try again.
                        else:
                            animal_index = centers_best_animal_index
                            # go back and try with this animal.....

                    # If there wasn't a good enough IoU match for this animal, add it to the leftovers list
                    else:
                        # add an animal because you did not find a match by distance when IoU ==0
                        leftover_animals.append(animal_list[animal_index])
                        animal_list.pop(animal_index)
                        iou_list = torch.cat((iou_list[0:animal_index, :], iou_list[animal_index + 1:, :]), dim=0)
                        animal_index = 0
                animal_index = 0

                # With the ones that didn't find a simple IoU match, this time find all of the potential matches that
                # are within the distance threshold and still not matched. Then, shift the potential matches
                # such that their box and the animal have the same x0, y0. Then see if it passes the IoU threshold.
                # This is kind of a fancy way to check relative size of new detected box to last known animal size
                while len(leftover_animals) > 0 and len(temp_masks) > 0:
                    animal_mask = self.active_tracklets['animals'][leftover_animals[animal_index]]['masks'][-1]
                    animal_last_box = self.active_tracklets['animals'][leftover_animals[animal_index]]['boxes'][-1]
                    distances = []
                    contours, _ = cv2.findContours(animal_mask.detach().cpu().numpy(), cv2.RETR_CCOMP,
                                                   cv2.CHAIN_APPROX_NONE)
                    fixed_contours = []
                    for contour_number in range(len(contours)):
                        fixed_contours = []
                        for contour in contours[contour_number]:
                            fixed_contours.append((
                                contour[0][0], contour[0][1]))
                    fixed_contours = np.array(fixed_contours)
                    if cv2.moments(fixed_contours)['m00'] > 0:
                        cx = int(cv2.moments(fixed_contours)['m10'] / cv2.moments(fixed_contours)['m00']) + \
                             animal_last_box[0]
                        cy = int(cv2.moments(fixed_contours)['m01'] / cv2.moments(fixed_contours)['m00']) + \
                             animal_last_box[1]
                    else:
                        cx = animal_last_box[0]
                        cy = animal_last_box[1]
                    animal_center = [cx.cpu(), cy.cpu()]

                    for i in range(len(temp_masks)):
                        contours, _ = cv2.findContours(temp_masks[i].detach().cpu().numpy(), cv2.RETR_CCOMP,
                                                       cv2.CHAIN_APPROX_NONE)
                        fixed_contours = []
                        for contour_number in range(len(contours)):
                            fixed_contours = []
                            for contour in contours[contour_number]:
                                fixed_contours.append((
                                    contour[0][0], contour[0][1]))
                        fixed_contours = np.array(fixed_contours)
                        if cv2.moments(fixed_contours)['m00'] > 0:
                            cx = int(
                                cv2.moments(fixed_contours)['m10'] / cv2.moments(fixed_contours)['m00']) + \
                                 temp_boxes[None][0][i][0]
                            cy = int(
                                cv2.moments(fixed_contours)['m01'] / cv2.moments(fixed_contours)['m00']) + \
                                 temp_boxes[None][0][i][1]
                        else:
                            cx = temp_boxes[None][0][i][0]
                            cy = temp_boxes[None][0][i][1]
                        current_test_center = [cx.cpu(), cy.cpu()]
                        animal_center = np.array(animal_center)
                        distances.append(np.linalg.norm(animal_center - current_test_center))
                    if (min(distances) < self.max_distance_moved) and len(temp_boxes) > 0 and len(distances) > 0:
                        temp = np.array(distances)
                        order = temp.argsort()
                        new_order = []
                        ################
                        for i in range(len(order)):
                            if distances[order[i]] <= self.max_distance_moved:
                                new_order.append(order[i])
                        # new order have a list of indexes into temp_boxes that are within distance
                        # animal_index is the index into left_overs for the current animal

                        # make the boxes as if they were at the origin to check for IoU
                        # if they had the same origin. Basically, checking relative size.
                        new_animal_last_box = [animal_last_box[0] - animal_last_box[0],
                                               animal_last_box[1] - animal_last_box[1],
                                               animal_last_box[2] - animal_last_box[0],
                                               animal_last_box[3] - animal_last_box[1], ]

                        temp_adjusted_boxes = []
                        for i in range(len(new_order)):
                            temp_adjusted_box = [temp_boxes[order[i]][0] - temp_boxes[order[i]][0],
                                                 temp_boxes[order[i]][1] - temp_boxes[order[i]][1],
                                                 temp_boxes[order[i]][2] - temp_boxes[order[i]][0],
                                                 temp_boxes[order[i]][3] - temp_boxes[order[i]][1], ]
                            temp_adjusted_boxes.append(temp_adjusted_box)
                        temp_adjusted_boxes=torch.Tensor(temp_adjusted_boxes)
                        new_animal_last_box = torch.Tensor(new_animal_last_box).unsqueeze(dim=0)

                         # Add the current animal's center to the first element of the box list
                        temp_adjusted_boxes = torch.cat((new_animal_last_box, temp_adjusted_boxes), dim=0)
                        temp_IoU_list = self.get_iou(temp_adjusted_boxes)
                        temp_IoU_list = temp_IoU_list[1:]
                        sort = temp_IoU_list.argsort()
                        if(temp_IoU_list[sort[-1]]) > self.tracking_iou_threshold:
                            mapped_animals.append((leftover_animals[animal_index], temp_boxes[new_order[sort[-1]]]))
                            leftover_animals.pop(animal_index)
                            temp_masks.pop(animal_index)
                            temp_boxes = torch.cat((temp_boxes[0:new_order[sort[-1]], :],
                                                    temp_boxes[new_order[sort[-1]] + 1:, :]), dim=0)
                        else:
                            # Remove an animal, it was not found.
                            leftover_animals.pop(animal_index)
                            animal_index = 0

                    else:
                        # remove the animal. It was not found
                        leftover_animals.pop(animal_index)
                        animal_index = 0

            else:
                mapped_animals = []
                temp_boxes = detected_boxes

            # Clean up the rest of temp_boxes, for anything that didn't have a match, map them to an animal of -1
            for i in range(len(temp_boxes)):
                mapped_animals.append((-1, temp_boxes[i]))

            # Set the visibility of everything to false.
            for i in list(self.active_tracklets['animals']):
                self.active_tracklets['animals'][i]['visible'] = False

            # FOr each animal that found a mapping, update its dictionary. Otherwise make a new animal.
            for animal in mapped_animals:
                animal_index = animal[0]
                box_index = detected_boxes.tolist().index(animal[1].tolist())
                if animal_index >= 0:
                    if self.save_trackers:
                        self.active_tracklets['animals'][animal_index]['boxes'].append(
                            detected_boxes[box_index])
                        self.active_tracklets['animals'][animal_index]['masks'] = [detected_masks[box_index], ]
                        self.active_tracklets['animals'][animal_index]['contours'].append(
                            detected_contours[box_index])
                        self.active_tracklets['animals'][animal_index]['scores'].append(
                            detected_scores[box_index])
                        self.active_tracklets['animals'][animal_index]['frame_number'].append(
                            frame_number)
                        self.active_tracklets['animals'][animal_index]['last_seen'] = frame_number
                        self.active_tracklets['animals'][animal_index]['visible'] = True
                    else:
                        self.active_tracklets['animals'][animal_index]['boxes'] = [detected_boxes[box_index], ]
                        self.active_tracklets['animals'][animal_index]['masks'] = [detected_masks[box_index], ]
                        self.active_tracklets['animals'][animal_index]['contours'] = [detected_contours[box_index], ]
                        self.active_tracklets['animals'][animal_index]['scores'] = [detected_scores[box_index], ]
                        self.active_tracklets['animals'][animal_index]['frame_number'] = [frame_number, ]
                        self.active_tracklets['animals'][animal_index]['last_seen'] = frame_number
                        self.active_tracklets['animals'][animal_index]['visible'] = True
                else:
                    # make new animal, when first found, don't make visible yet
                    self.active_tracklets['animals'][self.global_animal_id] = \
                        {"animal_id": self.global_animal_id,
                         "frame_number": [frame_number, ],
                         "visible": True,
                         "last_seen": frame_number,
                         "boxes": [detected_boxes[box_index], ],
                         "masks": [detected_masks[box_index], ],
                         "contours": [detected_contours[box_index], ],
                         "scores": [detected_scores[box_index].detach().cpu(), ],
                         "color": (self.get_color(self.global_animal_id)),
                         }
                    # Increment the animal id counter so that all new animals are given a new id
                    self.global_animal_id += 1
            for i in list(self.active_tracklets['animals']):
                if self.active_tracklets['animals'][i]['visible'] == False:
                    if (frame_number - self.active_tracklets['animals'][i]['last_seen']) > self.lost_threshold:
                        #print(f"Removed animal {i} last seen in frame {self.active_tracklets['animals'][i]['last_seen']} and it is now frame {frame_number}")
                        self.finished_tracklets[i] = self.active_tracklets['animals'][i]
                        del self.active_tracklets['animals'][i]
        # If we didn't find any animals (boxes), then check to see if we have officially lost any animals yet.
        # If an animal is permanently lost, move it to finished tracklets and from active tracklets
        else:
            for i in list(self.active_tracklets['animals']):
                self.active_tracklets['animals'][i]['visible'] = False
                if (frame_number - self.active_tracklets['animals'][i]['last_seen']) > self.lost_threshold:
                    self.finished_tracklets[i] = self.active_tracklets['animals'][i]
                    del self.active_tracklets['animals'][i]

    # Create a file that includes information about the video we are processing and the information from
    # active_tracklets and finished_tracklets. It is saved in a pickle (*.pkl) file. Using this file, you could
    # later recreate the entire detection box, center, mask, etc info without even having the video available.
    def save_tracklets(self, output_filename, video_info):
        self.active_tracklets.update(video_info)
        self.active_tracklets['animals'].update(self.finished_tracklets)
        f = open(output_filename, "wb")
        output = self.active_tracklets
        pickle.dump(output, f)

    # Alternative to creating a random color for animals because random colors sometimes don't look great. It
    # allocates the colors in order.  If there is a color that you would prefer, simply add it to the list.
    def get_color(self, index):
        color_list = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 255),
                      (0, 0, 255), (127, 0, 255), (255, 0, 255), (255, 0, 127),
                      (255, 153, 153), (255, 204, 153), (255, 255, 153), (204, 253, 153), (153, 255, 153),
                      (153, 255, 204), (153, 255, 255), (153, 204, 255), (153, 153, 255), (204, 153, 255),
                      (255, 153, 255), (255, 153, 204), (153, 0, 0), (102, 51, 0),
                      (102, 102, 0), (51, 102, 0), (0, 102, 0), (0, 102, 51), (0, 102, 102),
                      (0, 51, 102), (0, 0, 102), (51, 0, 102), (102, 0, 102), (102, 0, 51), ]
        length = len(color_list)
        while index >= length:
            index -= length
        return color_list[index]

    def get_iou(self, animal_boxes: torch.tensor):
        """
        modified version of the IoU algorithm from here:
        https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/

        This function processes a list of boxes and returns the IoU of all boxes relative to the first box in the list.
        Therefore, the first element in the list should always have an IoU of 1.
        """
        # animal_boxes = torch.stack(animal_boxes)
        # we extract coordinates for every
        # prediction box present in P
        x1 = animal_boxes[:, 0]
        y1 = animal_boxes[:, 1]
        x2 = animal_boxes[:, 2]
        y2 = animal_boxes[:, 3]

        # calculate area of every block in P
        areas = (x2 - x1) * (y2 - y1)

        # select coordinates of BBoxes according to
        # the indices in order
        # todo: remove these unnecessary variables.
        xx1 = x1
        xx2 = x2
        yy1 = y1
        yy2 = y2

        # find the coordinates of the intersection boxes
        xxx1 = torch.max(xx1, x1[0])
        yyy1 = torch.max(yy1, y1[0])
        xxx2 = torch.min(xx2, x2[0])
        yyy2 = torch.min(yy2, y2[0])

        # find height and width of the intersection boxes
        w = xxx2 - xxx1
        h = yyy2 - yyy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = areas

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[0] represents area of our box we are checking
        union = (rem_areas - inter) + areas[0]

        # find the IoU of every prediction in P with S
        IoU = inter / union

        return IoU


"""
# To load the data file (pickle file with extension *.pkl), use the following code, run it 
# in debug mode with a break on the print command and look in the data structure. I'll make a visualizer at
# some point in the future.

import pickle
filename = "./videos_output/animals_2023-02-01_17_04_52_629497/larva_tracklets_2023-02-01_17_04_52_629497.pkl"
with open(filename, 'rb') as f:
    data = pickle.load(f)
    print(data)
"""
