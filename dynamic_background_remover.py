import cv2
import json
import os
import torch
import torchvision
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
from animal_tracker import animal_tracker
import copy
import math


class dynamic_background_remover:
    def __init__(self,
                 model_path: str,
                 thing_names_path: str,
                 detection_threshold: float = 0.25,
                 iou: float = 0.4,
                 sigmoid_threshold: float = 0.55,
                 inference_size: int = 640) -> None:
        self.current_frame = 0
        self.batch_size = 1
        self.header_center = None
        self.header_font_size = None
        self.model = None
        self.model_things = None
        self.original_tensor = []
        self.processed_image_tensor = []
        self.outputs = None
        self.trackers = None
        self.animal_frames = {}
        self.detection_threshold = detection_threshold
        self.iou = iou
        self.sigmoid_threshold = sigmoid_threshold
        self.frame_count = 0
        self.start_frame = 0
        self.duration = 0
        self.percent_complete = 0
        self.save_original = False
        self.stop = False
        self.inference_size = inference_size
        self.ort_sess = None
        self._load_model(model_path)
        self._load_thing_names(thing_names_path)

    # Load the Model. It checks what kind of local resources you have and loads the best option for you
    # to optimize speed of processing
    def _load_model(self, model_path):
        # Check if any hardware acceleration devices are present and if so use them. Otherwise, use the CPU.
        if torch.cuda.is_available():
            self.model = torch.jit.load(model_path, map_location='cuda:0')
            self.device = torch.device('cuda:0')
        elif False:
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.device = torch.device('mps')
            self.model.to(self.device)
        else:
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.device = torch.device('cpu')

    # Load in the dictionary of real world names to detected indexes that we saved while exporting the
    # original model.
    def _load_thing_names(self, thing_names_path):
        with open(thing_names_path) as f:
            imported_data = f.read()

        # Reconstruct the data as a dictionary in index, name pairs {'0': 'rat', '1': 'larva', etc)
        self.model_things = json.loads(imported_data)

        # Invert the references so we can easily reverse lookup
        self.model_things_lookup = {}
        for key in self.model_things.keys():
            self.model_things_lookup[self.model_things[key]] = key

    # Takes the input from the model then filters it and resizes the outputs (masks, boxes)
    # to match the original image. This is also where I'll add conversions for other model types in
    # such as Yolov8 which could be used in the future.
    def _normalize_model_outputs(self, model_outputs, view_classes, temp_view_class_indexes, scale):

        outputs = []
        for i in range(len(model_outputs)):
            # Resize the boxes and masks to match the original image before storing them in the internal structure.
            # shape[1] is the height of the images and shape[0] is the width.
            boxes = model_outputs[i]['pred_boxes']

            # Rescale the boxes to fit the original image, put scale ratio onto the correct device
            new_boxes = boxes * scale[i]

            # Scale the masks.
            masks = model_outputs[i]["pred_masks"]
            new_masks = []

            # Changing dimensions with "unsqueeze" is necessary for the interpolate command.
            # This is required because the animal masks are all sent back
            # as 28x28 pixel images that must be scaled up to the size of their bounding box.
            for boxes_index in range(len(new_boxes)):
                height = int(new_boxes[boxes_index][3]) - int(new_boxes[boxes_index][1])
                width = int(new_boxes[boxes_index][2]) - int(new_boxes[boxes_index][0])
                mask = masks[boxes_index]
                mask = torch.unsqueeze(mask, dim=0)
                mask = torch.nn.functional.interpolate(mask,
                                                       size=(height, width),
                                                       mode="bicubic",
                                                       align_corners=False)
                mask = (mask.sigmoid() > self.sigmoid_threshold)
                mask = mask * 255
                new_masks.append(torch.squeeze(mask))

                # normalize the boxes to catch boundary condition at maximum width and height
                if int(new_boxes[boxes_index][3]) <= self.height:
                    new_boxes[boxes_index][3] = int(new_boxes[boxes_index][3])
                else:
                    new_boxes[boxes_index][3] = self.height
                new_boxes[boxes_index][1] = int(new_boxes[boxes_index][3]-height)
                if int(new_boxes[boxes_index][2]) <= self.width:
                    new_boxes[boxes_index][2] = int(new_boxes[boxes_index][2])
                else:
                    new_boxes[boxes_index][2] = self.width
                new_boxes[boxes_index][0] = int(new_boxes[boxes_index][2] - width)

            # Filter the scores and predicted classes based on the NMS output
            new_scores = model_outputs[i]['scores']
            new_prediction_classes = model_outputs[i]["pred_classes"]
            outputs.append({'boxes': new_boxes,
                            'scores': new_scores,
                            'masks': new_masks,
                            'pred_classes': new_prediction_classes,
                            'view_classes': view_classes,
                            'view_classes_indexes': temp_view_class_indexes,
                            })
        return outputs

    # Run inference on the provided image or image path and update internal variables accordingly.
    def run_inference(self, img_path=None, view_classes=None, input_scale=1):

        # If img_path a string is passed in, assume it is the path to the image.
        original_tensor = []
        scale = []
        processed_image_tensor = []
        model_input = []
        if isinstance(img_path, str):
            i = 0
            original_tensor.append(torchvision.io.read_image(img_path))
            pad = (0, 0, 0, 0)

            # Create a pad around the image to make it square and divisible by 32
            # then resize it to the inference size
            if original_tensor[i].shape[2] > original_tensor[i].shape[1]:
                bottom_pad = original_tensor[i].shape[2] % 32
                right_pad = original_tensor[i].shape[2] - original_tensor[i].shape[1] + bottom_pad
                pad = (0, bottom_pad, 0, right_pad)
            elif original_tensor[i].shape[1] > original_tensor[i].shape[2]:
                right_pad = original_tensor[i].shape[1] % 32
                bottom_pad = original_tensor[i].shape[1] - original_tensor[i].shape[2] + right_pad
                pad = (0, bottom_pad, 0, right_pad)

            resize = torchvision.transforms.Resize(size=(self.inference_size, self.inference_size),
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                   antialias=True)

            processed_image_tensor.append(resize(torch.nn.functional.pad(original_tensor[i], pad)))

            input_image = torch.nn.functional.pad(original_tensor[i], pad)
            model_input.append({'image': processed_image_tensor[i], })
            temp4 = input_image.shape[1]
            temp5 = processed_image_tensor[i].shape[1]
            scale.append((temp4 / temp5), )

        # If img_path is not a string, assume it is a numpy array from opencv
        # which has a format of BGR. Required to process video frames
        # from opencv (cv2.VideoCapture)
        else:
            model_input = []
            scale = []
            for i in range(len(img_path)):
                model_input.append({'image': img_path[i]}, )
                scale.append(input_scale)

        with torch.no_grad():
            outputs = self.model(model_input)

        # If no prediction classes were selected, show all animals.
        if view_classes is None:
            view_classes = list(self.model_things.values())

        # Get the 'index' values for the text strings to map them
        # to integer class values (animals) found during inferencing.
        temp_view_class_indexes = []
        key_list = list(self.model_things.keys())
        val_list = list(self.model_things.values())

        # For each animal type you are looking at, check whether to add it to self.outputs,
        # the list of animals you want to see.
        for name in view_classes:

            # This confirms that there is a valid mapping for the animal you entered within the list of things to view.
            if name in val_list:
                temp_view_class_indexes.append(int(key_list[val_list.index(name)]))

        outputs = self._normalize_model_outputs(outputs, view_classes, temp_view_class_indexes, scale)
        return outputs

    def get_original_with_masks(self,
                                 image_path=None,
                                 view_classes: list = None,
                                 frame_number: int = 0,
                                 model_output: dict = {},
                                 draw_bbox: bool = True,
                                 draw_text: bool = True,
                                 draw_masks: bool = True):
        # If an image path was passed in, then run inference on the image before proceeding.
        if isinstance(image_path, str):
            self.original_tensor = torchvision.io.read_image(image_path)

            # Assume that inference must be re-run and get the first dictionary in the return list
            width = self.original_tensor.shape[1]
            height = self.original_tensor.shape[2]
            font_scale, font_width = self.get_font_size(image_width=width, image_height=height)
            self.header_font_scale = font_scale*.25
            self.header_font_width = int(font_width * .4)
            self.header_center = width / 2 - (width * .15)

            model_output = self.run_inference(image_path, view_classes)[0]
            self.update_animal_data(tracking=False,
                                    frame_number=0,
                                    model_output=model_output)
        if 'original_tensor' not in model_output.keys():
            model_output['original_tensor'] = self.original_tensor

        if view_classes is None:
            view_classes = list(self.model_things.values())

        # Get the original image array
        original_image = np.array(model_output['original_tensor'])

        # Fixed the dimensions of the numpy coming from a tensor, move first dimension to end.
        original_image = np.transpose(original_image, (1, 2, 0)).copy()

        pil_comp = Image.fromarray(original_image)

        # Create a holder from the masks and then load its pixel map into a variable
        mask_holder = Image.new('RGB', Image.fromarray(original_image,
                                                       mode='RGB').size)
        holder_pixels = mask_holder.load()  # Create the pixel map

        if len(model_output['boxes'] > 0):
            if draw_masks:
                # For each type of animal that we are trying to track, process their tracking info.
                for animal_class in self.trackers:

                    # If there aren't any tracked animals, stop processing this animal set and move on.
                    if len(animal_class.active_tracklets['animals']) == 0:
                        continue

                    # For each key in the animal tracker,
                    # make the masked images, individual animal image, and contour images
                    for z in animal_class.active_tracklets['animals'].keys():

                        # Check to be sure an animal that is being tracked is visible in this frame before processing
                        if animal_class.active_tracklets['animals'][z]['visible']:

                            # 'Boxes' get the coordinates out of the boxes variable
                            x_min = int(animal_class.active_tracklets['animals'][z]['boxes'][-1][0])
                            y_min = int(animal_class.active_tracklets['animals'][z]['boxes'][-1][1])

                            # Image prep completed
                            temp_mask = animal_class.active_tracklets['animals'][z]['masks'][-1]
                            temp_mask = torch.stack([temp_mask, temp_mask, temp_mask], dim=2)
                            temp_mask2 = temp_mask.detach().cpu().numpy()
                            data = temp_mask2.astype(np.uint8)
                            red, green, blue = data.T  # Separate the channels of the image

                            # Create a filter to find non-back color and then replaces black with the chosen color
                            non_black_areas = (red > 0) | (blue > 0) | (green > 0)
                            data[..., :][non_black_areas.T] = animal_class.active_tracklets['animals'][z]['color']
                            mask = Image.fromarray(data)
                            pixels = mask.load()
                            for i in range(mask.size[0]):
                                for j in range(mask.size[1]):
                                    if holder_pixels[i + x_min, j + y_min] == (0, 0, 0):

                                        # Change if white, change to color
                                        holder_pixels[i + x_min, j + y_min] = pixels[i, j]

                brightness = ImageEnhance.Brightness(pil_comp)
                pil_comp = brightness.enhance(1.2)
                pil_comp = Image.blend(pil_comp.convert('RGB'), mask_holder, 0.3)

            # Now that the mask has been blended, revert back to numpy array.
            return_image = np.array(pil_comp)
            for animal_class in self.trackers:
                if len(animal_class.active_tracklets['animals']) == 0:
                    continue
                for z in animal_class.active_tracklets['animals'].keys():
                    if animal_class.active_tracklets['animals'][z]['visible']:

                        # 'Boxes' is a tensor with a gradient, so you have to detach it and
                        # convert it to a numpy to get access to the values
                        tempbox = animal_class.active_tracklets['animals'][z]['boxes'][-1]  # .numpy()
                        x_min = int(tempbox[0])
                        x_max = int(tempbox[2])
                        y_min = int(tempbox[1])
                        y_max = int(tempbox[3])
                        height = y_max - y_min
                        width = x_max - x_min
                        if draw_bbox or draw_text:
                            outline_color = animal_class.active_tracklets['animals'][z]['color']
                        if draw_text:
                            score = np.round(animal_class.active_tracklets['animals'][z]['scores'][-1].detach().cpu(),
                                             3) * 100
                            score = '%s' % float('%.3g' % score)
                            text = f" {self.model_things[str(animal_class.class_index)]}{str(z)}: {str(score)}% "
                            font_scale, font_width = self.get_font_size(image_width=int(width),
                                                                        image_height=int(height))
                            return_image = cv2.putText(img=return_image,
                                                       text=text,
                                                       org=(int(tempbox[0]), int(tempbox[1])),
                                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                       fontScale=font_scale,
                                                       color=outline_color,
                                                       thickness=font_width
                                                       )
                        if draw_bbox:
                            cv2.rectangle(img=return_image,
                                          pt1=(int(tempbox[0]), int(tempbox[1])),
                                          pt2=(int(tempbox[2]), int(tempbox[3])),
                                          color=outline_color,
                                          thickness=1
                                          )
        else:
            return_image = original_image
        if draw_text:
            text = f"Frame Number: {frame_number}"
            return_image = cv2.putText(img=return_image,
                                       text=text,
                                       org=(int(self.header_center), int(30 * self.header_font_scale)),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=self.header_font_scale,
                                       color=(255, 0, 0),  # Put the frame number in red.
                                       thickness=self.header_font_width
                                       )
        return_image = cv2.cvtColor(return_image, cv2.COLOR_RGB2BGR)
        return return_image

    def get_masked_image(self,
                         image_path=None,
                         view_classes: list = None,
                         frame_number: int = 0,
                         draw_text: bool = True,
                         model_output: dict = {}):

        if isinstance(image_path, str):
            self.original_tensor = torchvision.io.read_image(image_path)

            # Assume that inference must be re-run and get the first dictionary in the return list
            width = self.original_tensor.shape[1]
            height = self.original_tensor.shape[2]
            font_scale, font_width = self.get_font_size(image_width=width, image_height=height)
            self.header_font_scale = font_scale*.25
            self.header_font_width = int(font_width * .4)
            self.header_center = width / 2 - (width * .15)

            # Assume that inference must be re-run.
            model_output = self.run_inference(image_path, view_classes)[0]
            self.update_animal_data(tracking=False,
                                    frame_number=0,
                                    model_output=model_output)

        if 'original_tensor' not in model_output.keys():
            model_output['original_tensor'] = self.original_tensor
        if view_classes is None:
            view_classes = list(self.model_things.values())

        # Get the original image array
        original_image = np.array(model_output['original_tensor'])

        # Fixed the dimensions of the numpy coming from a tensor, move first dimension to end.
        original_image = np.transpose(original_image, (1, 2, 0)).copy()
        mask_holder = np.zeros_like(original_image)

        # Look through the boxes that were found
        for animal_name in view_classes:
            animal_id = self.model_things_lookup[animal_name]

            value_mask = torch.logical_and(model_output['pred_classes'] ==
                                           torch.tensor(int(animal_id), dtype=torch.int8),
                                           model_output['scores'] > self.detection_threshold)

            for z in range(len(model_output['boxes'])):

                # If the box met the criteria for valid animal type and score
                if value_mask[z]:
                    # Get the coordinates of the bounding box for each animal
                    box = model_output['boxes'][z]
                    temp_mask = model_output['masks'][z].detach().cpu().numpy().astype(np.uint8)

                    # Check a pixel in the mask_holder and if it is zero (black), copy the corresponding
                    # value from the mask into it. This code also catch the boundary
                    # condition at the extremes of the frame
                    top = int(math.floor(box[1]))
                    bottom = top + temp_mask.shape[0]
                    left = int(math.floor(box[0]))
                    right = left + temp_mask.shape[1]
                    mask_holder[top:bottom, left:right] += temp_mask.clip(max=1)[..., None]

        # Make a true/false array out of the mask_holder numpy. True means not white
        mask_holder = mask_holder[..., :] == 0  # 255

        # Everywhere that mask_holder is True (not white), set the original equals to zero
        original_image[mask_holder] = 0

        if draw_text:
            text = f"Frame Number: {frame_number}"
            original_image = cv2.putText(img=original_image,
                                       text=text,
                                       org=(int(self.header_center), int(30 * self.header_font_scale)),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=self.header_font_scale,
                                       color=(255, 0, 0),  # Put the frame number in red.
                                       thickness=self.header_font_width
                                       )

        # Turn the color channels back to BRG from RGB for opencv
        composite = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        # Return the image
        return composite

    def get_original_with_contours(self,
                                   image_path=None,
                                   view_classes: list = None,
                                   frame_number: int = 0,
                                   model_output: dict = {},
                                   draw_bbox: bool = True,
                                   draw_text: bool = True,
                                   draw_contour: bool = True):
        # If an image path was passed in, then run inference on the image before proceeding.
        if isinstance(image_path, str):
            self.original_tensor = torchvision.io.read_image(image_path)

            # Assume that inference must be re-run. and get the first dictionary in the return list
            width = self.original_tensor.shape[1]
            height = self.original_tensor.shape[2]
            font_scale, font_width = self.get_font_size(image_width=width, image_height=height)
            self.header_font_scale = font_scale*.25
            self.header_font_width = int(font_width * .4)
            self.header_center = width / 2 - (width * .15)

            model_output = self.run_inference(image_path, view_classes)[0]
            self.update_animal_data(tracking=False,
                                    frame_number=0,
                                    model_output=model_output)
        if 'original_tensor' not in model_output.keys():
            model_output['original_tensor'] = self.original_tensor

        if view_classes is None:
            view_classes = list(self.model_things.values())

        # Get the original image array
        original_image = np.array(model_output['original_tensor'])

        # Fixed the dimensions of the numpy coming from a tensor, move first dimension to end.
        original_image = np.transpose(original_image, (1, 2, 0)).copy()
        if len(model_output['boxes'] > 0):
            for animal_class in self.trackers:
                if len(animal_class.active_tracklets['animals']) == 0:
                    continue
                for z in animal_class.active_tracklets['animals'].keys():
                    if animal_class.active_tracklets['animals'][z]['visible']:

                        # 'Boxes' is a tensor with a gradient, so you have to detach it and
                        # convert it to a numpy to get access to the values
                        tempbox = animal_class.active_tracklets['animals'][z]['boxes'][-1]  # .numpy()
                        x_min = int(tempbox[0])
                        x_max = int(tempbox[2])
                        y_min = int(tempbox[1])
                        y_max = int(tempbox[3])
                        height = y_max - y_min
                        width = x_max - x_min

                        outline_color = animal_class.active_tracklets['animals'][z]['color']
                        temp_mask = animal_class.active_tracklets['animals'][z]['masks'][-1]
                        contours, _ = cv2.findContours(temp_mask.detach().cpu().numpy(), cv2.RETR_CCOMP,
                                                       cv2.CHAIN_APPROX_NONE)
                        if draw_contour:
                            cv2.drawContours(image=original_image,
                                             contours=contours ,
                                             contourIdx=-1,
                                             color=outline_color,
                                             thickness=1,
                                             offset=(int(x_min), int(y_min))
                                             )

                        if draw_bbox:
                            cv2.rectangle(img=original_image,
                                          pt1=(int(tempbox[0]), int(tempbox[1])),
                                          pt2=(int(tempbox[2]), int(tempbox[3])),
                                          color=outline_color,
                                          thickness=1
                                          )

                        if draw_text:
                            score = np.round(animal_class.active_tracklets['animals'][z]['scores'][-1].detach().cpu(),
                                             3) * 100
                            score = '%s' % float('%.3g' % score)
                            text = f" {self.model_things[str(animal_class.class_index)]}{str(z)}: {str(score)}% "
                            font_scale, font_width = self.get_font_size(image_width=int(width),
                                                                        image_height=int(height))
                            original_image = cv2.putText(img=original_image,
                                                         text=text,
                                                         org=(int(tempbox[0]), int(tempbox[1])),
                                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                         fontScale=font_scale,
                                                         color=outline_color,
                                                         thickness=font_width
                                                         )
        if draw_text:
            text = f"Frame Number: {frame_number}"
            original_image = cv2.putText(img=original_image,
                                         text=text,
                                         org=(int(self.header_center), int(30 * self.header_font_scale)),
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=self.header_font_scale,
                                         color=(255, 0, 0),  # Put the frame number in red.
                                         thickness=self.header_font_width
                                         )
        return_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        return return_image

    def update_animal_data(self,
                           tracking: bool = False,
                           frame_number: int = 0,
                           model_output: dict = {},
                           keep_history: bool = False):

        # Get the original image array
        original_image = np.array(model_output['original_tensor']) if keep_history else []

        # If you aren't using this library's tracking algorithm, it will just create a dummy tracking item
        # to store the info about the animals.
        if not tracking:
            self.trackers = []
            for animal_class in model_output['view_classes_indexes']:
                self.trackers.append(animal_tracker(score_threshold=self.detection_threshold,
                                                    lost_threshold=1,
                                                    class_index=animal_class,
                                                    max_distance_moved=5,
                                                    save_trackers=False))
            for tracker in self.trackers:
                tracker.track(model_output['boxes'], model_output['pred_classes'],
                              model_output['scores'], model_output['masks'], 0)

        # For each type of animal that we are trying to track, process their tracking info.
        for animal_class in self.trackers:

            # If there aren't any tracked animals, stop processing this animal set and move on
            if len(animal_class.active_tracklets['animals']) == 0:
                continue

            # For each key in the animal tracker, make the masked images, individual animal image, and contour images
            for z in animal_class.active_tracklets['animals'].keys():

                # Check to be sure an animal that is being tracked is visible in this frame before processing
                if animal_class.active_tracklets['animals'][z]['visible']:

                    # 'Boxes' get the coordinates out of the boxes variable
                    tempbox = animal_class.active_tracklets['animals'][z]['boxes'][-1]

                    # Image prep completed
                    temp_mask = animal_class.active_tracklets['animals'][z]['masks'][-1].detach().cpu().numpy()
                    animal_id = animal_class.active_tracklets['animals'][z]['animal_id']
                    if len(self.animal_frames) == 0:
                        for class_id in self.model_things.keys():
                            self.animal_frames[int(class_id)] = {"Frames": {}}
                    else:
                        for class_id in self.model_things.keys():
                            if len(self.animal_frames[int(class_id)]) == 0:
                                self.animal_frames[int(class_id)] = {"Frames": {}}
                    # If the animal is not being tracked then set its first element, otherwise append to the
                    # element.
                    if (animal_id not in self.animal_frames[animal_class.class_index].keys()) or \
                            (keep_history is False):
                        if keep_history:
                            self.animal_frames[animal_class.class_index]['Frames'][f"{frame_number}"] = original_image
                            self.animal_frames[animal_class.class_index][animal_id] = {"masks": [temp_mask, ],
                                                                                       "boxes": [tempbox, ],
                                                                                       "frame_number": [frame_number, ]}
                        else:
                            self.animal_frames[animal_class.class_index]['Frames']["0"] = original_image
                            self.animal_frames[animal_class.class_index][animal_id] = {"masks": [temp_mask, ],
                                                                                       "boxes": [tempbox, ],
                                                                                       "frame_number": [frame_number, ]}

                    else:
                        self.animal_frames[animal_class.class_index]['Frames'][f"{frame_number}"] = original_image
                        self.animal_frames[animal_class.class_index][animal_id]["masks"].append(temp_mask)
                        self.animal_frames[animal_class.class_index][animal_id]["boxes"].append(tempbox)
                        self.animal_frames[animal_class.class_index][animal_id]["frame_number"].append(frame_number)

        return

    def resize_frame(self, frame, inference_size=None):
        if not inference_size:
            inference_size = self.inference_size
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_to_tensor = torchvision.transforms.ToTensor()(img_rgb)

        # Convert the image to a tensor and scale it
        img_to_tensor = img_to_tensor * 255

        # Make the tensor an uint8 type
        frame = img_to_tensor.type(torch.uint8)
        pad = (0, 0, 0, 0)

        # Resize the image to feed into the model by padding it to make it square and divisible by 32.
        # Then resize to the inference size.
        if frame.shape[2] > frame.shape[1]:
            bottom_pad = frame.shape[2] % 32
            right_pad = frame.shape[2] - frame.shape[1] + bottom_pad
            pad = (0, bottom_pad, 0, right_pad)
        elif frame.shape[1] > frame.shape[2]:
            right_pad = frame.shape[1] % 32
            bottom_pad = frame.shape[1] - frame.shape[2] + right_pad
            pad = (0, bottom_pad, 0, right_pad)
        resize = torchvision.transforms.Resize(size=(inference_size, inference_size),
                                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                               antialias=True)
        processed_frame = resize(torch.nn.functional.pad(frame, pad))
        return processed_frame

    def find_scale(self, width, height, inference_size):
        if height >= width:
            bottom_pad = height % 32
        else:
            right_pad = width % 32
            bottom_pad = width - height + right_pad
        new_height = height + bottom_pad
        scale = (new_height / inference_size)
        return scale

    def get_tracked_video(self,
                          duration: int,
                          view_classes: list,
                          input_video_name: str,
                          video_output_path: str = os.getcwd(),
                          blanks: bool = True,
                          batch_size: int = 1,
                          progress: bool = False,
                          original_with_masks: bool = False,
                          masked_original: bool = False,
                          original_with_contours: bool = False,
                          save_progress_videos: bool = False,
                          decorate_videos: bool = True,
                          save_tracker_info: bool = False,
                          mini_clip_length: int = 0,
                          lost_threshold: int = 10,
                          max_distance_moved: int = 5,
                          video_start: int = 0,
                          debug_masks: bool = False,
                          frame_skip: int = 0,
                          tracking_iou: float = 0.5) -> None:

        # If the requested length of individual animal clips is > 0, set the flag to make individual videos and
        # contours.
        self.stop = False
        if mini_clip_length == 0:
            individual_videos = False
        else:
            individual_videos = True

        if len(view_classes) == 0:
            view_classes = list(self.model_things.values())
        self.batch_size = batch_size

        # Get the 'index' values for the text strings that were sent in so that we can map those
        # to animals found during inferencing.
        temp_view_class_indexes = []
        key_list = list(self.model_things.keys())
        val_list = list(self.model_things.values())
        for name in view_classes:
            if name in val_list:
                temp_view_class_indexes.append(int(key_list[val_list.index(name)]))

        # Set up the storage for individual animal frames if we will make individual videos/contours
        if individual_videos:
            self.animal_frames = {}

        cap = cv2.VideoCapture(input_video_name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.width = width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.height = height
        frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_distance_moved = math.sqrt(width ** 2 + height ** 2) * max_distance_moved / 100

        # Set up trackers
        self.trackers = []
        for animal_class in temp_view_class_indexes:
            self.trackers.append(animal_tracker(score_threshold=self.detection_threshold,
                                                lost_threshold=lost_threshold,
                                                class_index=animal_class,
                                                max_distance_moved=max_distance_moved,
                                                tracking_iou_threshold=tracking_iou,
                                                save_trackers=save_tracker_info))

        # Get the Date-Time string and replace characters we don't want with underscores.
        date_time = datetime.now()
        date_time = f"date_{date_time.year}_{date_time.month}_{date_time.day}_time_" \
                    f"{date_time.hour}_{date_time.minute}_{date_time.second}"

        project_path = os.path.join(video_output_path, f"animals_{os.path.split(input_video_name)[1][:-4]}_{date_time}")

        os.makedirs(project_path,
                    exist_ok=True)

        # Set the video output format
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        if original_with_masks and save_progress_videos:
            original_with_masks_result = os.path.join(project_path,
                                                      "original_with_masks_" +
                                                      os.path.split(input_video_name)[1][:-4] + '_' +
                                                      date_time + "_result.avi")
            output_writer_original_with_masks = cv2.VideoWriter(original_with_masks_result, fourcc,
                                                                frames_per_second, (width, height))
        if original_with_contours and save_progress_videos:
            original_with_contours_result = os.path.join(project_path,
                                                         "original_with_contours_" +
                                                         os.path.split(input_video_name)[1][:-4] + '_' +
                                                         date_time + "_result.avi")

            output_writer_original_with_contours = cv2.VideoWriter(original_with_contours_result, fourcc,
                                                                   frames_per_second, (width, height))
        if masked_original and save_progress_videos:
            masked_original_result = os.path.join(project_path,
                                                  "masked_original_" +
                                                  os.path.split(input_video_name)[1][:-4] + '_' +
                                                  date_time + "_result.avi")

            output_writer_masked_original = cv2.VideoWriter(masked_original_result, fourcc,
                                                            frames_per_second, (width, height))
        if num_frames == 0:
            cap.release()
            assert num_frames == 0, 'video not found or empty!'
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        assert int(video_start * frames_per_second) <= video_length, "Check the start time and ensure it is within " \
                                                                     "the duration of the video"

        # If a specific duration is not sent in for video output, use the entire video.
        if duration is None or duration <= 0:
            if video_length - video_start * frames_per_second > 0:
                duration = video_length - video_start * frames_per_second
            else:
                duration = 0
        else:
            if (duration * frames_per_second) > video_length:
                duration = int(video_length - video_start * frames_per_second)
            else:
                duration = int(duration * frames_per_second)
        start_frame = int(video_start * frames_per_second)
        self.start_frame = start_frame
        self.duration = duration
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if decorate_videos:

            # Find the right size for the Frame Number text
            font_scale, font_width = self.get_font_size(image_width=width, image_height=height)
            self.header_font_scale = font_scale*.25
            self.header_font_width = int(font_width * .4)
            self.header_center = width / 2 - (width * .15)

        # Loop through the video and create the appropriate output files.
        scale = width / self.inference_size if width > height else height / self.inference_size
        frame_count = 0
        with tqdm(total=duration, position=0, leave=True,
                  desc=f"Processing video: {os.path.basename(input_video_name)}") as pbar:
            while frame_count < duration:
                if self.stop:
                    break
                self.current_frame = frame_count + start_frame
                batch = []
                for i in range(self.batch_size):
                    _, frame = cap.read()
                    if frame is None:
                        break
                    if batch_size == 1:
                        original_frame = copy.deepcopy(frame)
                    frame = self.resize_frame(frame)
                    batch.append(frame, )
                if frame is None and len(batch) == 0:
                    if frame_count < duration:
                        print(f'Video file metadata or content is corrupted. Skipping any remaining content after corruption.')
                        print('Exiting cleanly now. Please check the results.')
                    break
                # if no animals were found in the frame, move on to the next frame
                outputs = self.run_inference(batch, view_classes, input_scale=scale)

                # This is an "expensive" operation so only do it if you are batching images. Then the cost
                # would be worth it.
                if (original_with_contours or original_with_masks or individual_videos or masked_original) and \
                        batch_size > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                for batch_idx, output in enumerate(outputs):
                    if original_with_contours or original_with_masks or individual_videos or masked_original:
                        if (original_with_contours or original_with_masks or individual_videos or masked_original) and \
                                batch_size > 1:
                            _, temp_frame = cap.read()
                        else:
                            temp_frame = original_frame
                        img_rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                        img_to_tensor = torchvision.transforms.ToTensor()(img_rgb)
                        img_to_tensor = img_to_tensor * 255
                        temp_frame = img_to_tensor.type(torch.uint8)
                        output['original_tensor'] = temp_frame
                    if len(output['boxes']) <= 0:
                        if original_with_masks:
                            foreground = self.get_original_with_masks(frame_number=self.current_frame,
                                                                      view_classes=view_classes,
                                                                      model_output=output,
                                                                      draw_bbox=decorate_videos,
                                                                      draw_text=decorate_videos,
                                                                      draw_masks=True)

                            if save_progress_videos:
                                output_writer_original_with_masks.write(foreground)
                            if progress:
                                cv2.imshow('video progress: showing original with masks', foreground)
                                cv2.waitKey(1)
                        if original_with_contours:
                            foreground = self.get_original_with_contours(frame_number=self.current_frame,
                                                                         view_classes=view_classes,
                                                                         model_output=output,
                                                                         draw_bbox=decorate_videos,
                                                                         draw_text=decorate_videos,
                                                                         draw_contour=True)
                            if save_progress_videos:
                                output_writer_original_with_contours.write(foreground)
                            if progress:
                                cv2.imshow('video progress: showing original with contours', foreground)
                                cv2.waitKey(1)
                        if masked_original:
                            foreground = self.get_masked_image(frame_number=self.current_frame,
                                                               view_classes=view_classes,
                                                               draw_text=decorate_videos,
                                                               model_output=output)
                            if save_progress_videos:
                                output_writer_masked_original.write(foreground)
                            if progress:
                                cv2.imshow('video progress: masked original', foreground)
                                cv2.waitKey(1)
                        for animal_class_id in self.animal_frames.keys():
                            self.animal_frames[animal_class_id] = {}
                        self.current_frame += 1
                        continue
                    for tracker in self.trackers:
                        tracker.track(output['boxes'], output['pred_classes'],
                                      output['scores'], output['masks'], self.current_frame, (height, width))
                    self.update_animal_data(tracking=True,
                                            frame_number=self.current_frame,
                                            model_output=output,
                                            keep_history=individual_videos)
                    if original_with_masks:
                        foreground = self.get_original_with_masks(frame_number=self.current_frame,
                                                                  view_classes=view_classes,
                                                                  model_output=output,
                                                                  draw_bbox=decorate_videos,
                                                                  draw_text=decorate_videos,
                                                                  draw_masks=True)
                        if save_progress_videos:
                            output_writer_original_with_masks.write(foreground)
                        if progress:
                            cv2.imshow('video progress: showing original with masks', foreground)
                            cv2.waitKey(1)

                    if original_with_contours:
                        foreground = self.get_original_with_contours(frame_number=self.current_frame,
                                                                     view_classes=view_classes,
                                                                     model_output=output,
                                                                     draw_bbox=decorate_videos,
                                                                     draw_text=decorate_videos,
                                                                     draw_contour=True)
                        if save_progress_videos:
                            output_writer_original_with_contours.write(foreground)
                        if progress:
                            cv2.imshow('video progress: showing original with contours', foreground)
                            cv2.waitKey(1)
                    if masked_original:
                        foreground = self.get_masked_image(frame_number=self.current_frame,
                                                           view_classes=view_classes,
                                                           draw_text=decorate_videos,
                                                           model_output=output)
                        if save_progress_videos:
                            output_writer_masked_original.write(foreground)
                        if progress:
                            cv2.imshow('video progress: masked original', foreground)
                            cv2.waitKey(1)
                    if individual_videos:
                        if (self.current_frame + 1) % mini_clip_length == 0:
                            for animal_class_id in self.animal_frames.keys():
                                animal_name = self.model_things[str(animal_class_id)]
                                for idx1, animal in enumerate(self.animal_frames[animal_class_id].keys()):
                                    if animal != 'Frames' and len(
                                            self.animal_frames[animal_class_id][animal]['boxes']) == mini_clip_length:
                                        xmin = np.inf
                                        xmax = 0
                                        ymin = np.inf
                                        ymax = 0
                                        for box in self.animal_frames[animal_class_id][animal]['boxes']:
                                            if box[0] < xmin:
                                                xmin = int(box[0])
                                            if box[1] < ymin:
                                                ymin = int(box[1])
                                            if box[2] > xmax:
                                                xmax = int(box[2])
                                            if box[3] > ymax:
                                                ymax = int(box[3])

                                        # Make the output images squares
                                        height = ymax - ymin
                                        width = xmax - xmin
                                        if height > width:
                                            side_length = height
                                            offset_x = int((side_length - width) / 2)
                                            offset_y = 0
                                        else:
                                            side_length = width
                                            offset_x = 0
                                            offset_y = int((side_length - height) / 2)

                                        temp_mask_mini = np.zeros([side_length, side_length, 3], dtype=np.uint8)
                                        temp_contours = np.zeros_like(temp_mask_mini)
                                        frame_reference = self.current_frame - mini_clip_length + 1
                                        for idx, animal_frame in enumerate(
                                                self.animal_frames[animal_class_id][animal]['masks']):
                                            os.makedirs(os.path.join(project_path,
                                                                     f"{animal_name}",
                                                                     f"{animal}"),
                                                        exist_ok=True)
                                            if debug_masks:
                                                os.makedirs(os.path.join(project_path,
                                                                         f"{animal_name}",
                                                                         f"{animal}",
                                                                         f"debug"),
                                                            exist_ok=True)
                                            if idx == 0:
                                                filename = os.path.join(project_path,
                                                                        f"{animal_name}",
                                                                        f"{animal}",
                                                                        f"{animal}_{self.current_frame}.avi")
                                                mini_animal_writer = cv2.VideoWriter(filename,
                                                                                     fourcc,
                                                                                     frames_per_second,
                                                                                     (temp_contours.shape[1],
                                                                                      temp_contours.shape[
                                                                                          0]))
                                            box = self.animal_frames[animal_class_id][animal]['boxes'][idx]
                                            temp_frame = copy.deepcopy(
                                                self.animal_frames[animal_class_id]['Frames'][f'{frame_reference}'])

                                            # Important: need to make a deepcopy
                                            # so that we don't modify the original array
                                            temp_frame2 = copy.deepcopy(temp_frame)
                                            temp_frame2 = temp_frame2[:, int(box[1]):int(box[3]),
                                                          int(box[0]):int(box[2])]
                                            debug_keeper = copy.deepcopy(
                                                np.transpose(temp_frame2, (1, 2, 0))) if debug_masks else None
                                            temp_animal_mask = self.animal_frames[animal_class_id][animal]['masks'][
                                                idx].astype(np.uint8)
                                            temp_frame2 = np.transpose(temp_frame2, (1, 2, 0))

                                            # Multiply the image and the mask, but set the max value in the mask image
                                            # to 1. So all pixels in the image are multiplied by 0 or 1.
                                            temp_frame2 *= temp_animal_mask.clip(max=1)[..., None]
                                            temp_mask_mini[int(box[1]) - ymin + offset_y:int(box[3]) - ymin + offset_y,
                                            int(box[0]) - xmin + offset_x:int(box[2]) - xmin + offset_x] = temp_frame2

                                            # Write the masked mini frame after turning the image back to BGR for opencv
                                            mini_animal_writer.write(temp_mask_mini)
                                            temp_mask_mini.fill(0)
                                            frame_reference += 1

                                            # Saves Mask to a "debug" folder to see the mask vs original framed box
                                            if debug_masks:
                                                filename_mask = os.path.join(project_path,
                                                                             f"{animal_name}",
                                                                             f"{animal}",
                                                                             "debug",
                                                                             f"{animal}_{self.current_frame - mini_clip_length + idx + 1}_mask.jpg")
                                                filename_mini_frame = os.path.join(project_path,
                                                                                   f"{animal_name}",
                                                                                   f"{animal}",
                                                                                   "debug",
                                                                                   f"{animal}_{self.current_frame - mini_clip_length + idx + 1}_frame.jpg")
                                                cv2.imwrite(filename_mask, temp_animal_mask)
                                                cv2.imwrite(filename_mini_frame, debug_keeper)

                                            contours, hierarchy = cv2.findContours(image=temp_animal_mask,
                                                                                   mode=cv2.RETR_EXTERNAL,
                                                                                   method=cv2.CHAIN_APPROX_NONE)
                                            color = self.getContourColor(idx, mini_clip_length)
                                            contour_thickness = int(max(abs(int(box[3]) - int(box[1])), abs(int(box[0]) - int(box[2])))/150 + 1)
                                            cv2.drawContours(image=temp_contours,
                                                             contours=contours,
                                                             contourIdx=-1,
                                                             color=color,
                                                             thickness=contour_thickness,
                                                             offset=(int(box[0]) - xmin + offset_x, int(box[1]) - ymin + offset_y)

                                                             )
                                        mini_animal_writer.release()
                                        filename = os.path.join(project_path,
                                                                f"{animal_name}",
                                                                f"{animal}",
                                                                f"{animal}_{self.current_frame}.jpg")
                                        cv2.imwrite(filename, temp_contours)
                                self.animal_frames[animal_class_id] = {}
                    self.current_frame += 1
                frame_count += self.batch_size
                pbar.update(self.batch_size)
        pbar.close()
        if original_with_masks:
            if save_progress_videos:
                output_writer_original_with_masks.release()
            if progress:
                cv2.destroyWindow('video progress: showing original with masks')
        if original_with_contours:
            if save_progress_videos:
                output_writer_original_with_contours.release()
            if progress:
                cv2.destroyWindow('video progress: showing original with contours')
        if masked_original:
            if save_progress_videos:
                output_writer_masked_original.release()
            if progress:
                cv2.destroyWindow('video progress: masked original')
        cap.release()
        if save_tracker_info:
            for tracker in self.trackers:
                video_info = {"video_info": {},
                              'video_info': {'filename': os.path.basename(input_video_name), 'fps': frames_per_second,
                                             'width': self.width, 'height': self.height}}
                filename = os.path.join(project_path,
                                        f"{self.model_things[str(tracker.class_index)]}_tracklets_"
                                        f"{os.path.split(input_video_name)[1][:-4]}_{date_time}.pkl")
                tracker.save_tracklets(output_filename=filename, video_info=video_info)
        return

    def get_animal_arrays(self,
                          image_path: str = "",
                          view_classes: list = [],
                          inners: bool = True,
                          model_output: list = [],
                          debug: bool = False):

        # If view_classes isn't specified, use all animals in model
        if len(view_classes) == 0:
            view_classes = list(self.model_things.values())

        if len(image_path) > 0:
            self.original_tensor = torchvision.io.read_image(image_path)

            # Assume that inference must be re-run.
            outputs = self.run_inference(image_path, view_classes)[0]
            original_tensor = self.original_tensor
        else:
            outputs = model_output
            original_tensor = outputs['original_tensor']
        new_boxes = []
        new_scores = []
        new_pred_classes = []
        new_masks = []
        new_contours = []
        new_inners = []
        for animal_name in view_classes:
            animal_id = self.model_things_lookup[animal_name]

            # Bool array for whether this box will be an animals we want based on class id and threshold
            value_mask = torch.logical_and(outputs['pred_classes'] ==
                                           torch.tensor(int(animal_id), dtype=torch.int8),
                                           outputs['scores'] > self.detection_threshold)

            # Loop through the boxes
            for z in range(len(outputs['boxes'])):

                # Is it a box we are interested in
                if value_mask[z]:

                    # Append to new boxes, prediction classes, scores, and masks.
                    new_boxes.append(outputs['boxes'][z])
                    new_pred_classes.append(outputs['pred_classes'][z])
                    new_scores.append(outputs['scores'][z])
                    new_masks.append(outputs['masks'][z])

                    # Make blank BGR image, find the original mask contours, then create the contour image
                    contour_image = np.zeros((new_masks[z].shape[0], new_masks[z].shape[1], 3), np.uint8)

                    # Use opencv to find the contour(s) list in the mask and then draw the contours in red and
                    # append to the return variable
                    contours, hierarchy = cv2.findContours(image=np.array(new_masks[z]).astype(np.uint8),
                                                           mode=cv2.RETR_EXTERNAL,
                                                           method=cv2.CHAIN_APPROX_NONE)
                    fixed_contours = [sorted(contours, key=cv2.contourArea, reverse=True)[0]]
                    new_contours.append(fixed_contours, )

                    if debug:

                        # Testing purposes only, to show individual contours
                        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
                        cv2.imshow('contours', contour_image)
                        cv2.waitKey(0)

                    # If we want the inner image contours, get the actual image and not just the mask.
                    # Then, mask the image and find the inner contours.
                    if inners:
                        tempbox = outputs['boxes'][z]

                        # Get the full original image and put it in a numpy array.
                        animal_image = np.array(original_tensor)

                        # Fixed the dimensions of the numpy coming from a tensor, move first dimension to end.
                        animal_image = np.transpose(animal_image, (1, 2, 0))

                        # Cut the current animal out of the original image using its bounding box
                        mini_image = animal_image[int(tempbox[1]):int(tempbox[3]), int(tempbox[0]):int(tempbox[2])]

                        # Make a numpy mask using the stored animal image mask
                        # Everywhere in the mask that the value is not white, set it to True
                        animal_mask = outputs['masks'][z][..., :] < 255

                        # Everywhere the mask is true, set that place in the array to 0 (black)
                        # The image with the animal now has its background removed
                        mini_image[animal_mask] = 0

                        gray = cv2.cvtColor(np.uint8(mini_image), cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (3, 3), 0)
                        edges = cv2.Canny(blur, 20, 75, apertureSize=3, L2gradient=True)
                        contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                        new_inners.append(contours, )

                        if debug:
                            contour_image = np.zeros((new_masks[z].shape[0], new_masks[z].shape[1], 3), np.uint8)
                            cv2.drawContours(image=contour_image,
                                             contours=contours,
                                             contourIdx=-1,
                                             color=(255, 255, 255),
                                             thickness=1,
                                             )
                            cv2.imshow('inners', contour_image)
                            cv2.waitKey(0)
        if debug:
            cv2.destroyWindow('contours')
            cv2.destroyWindow('inners')

        # Return the new lists including contours
        return new_scores, new_pred_classes, new_contours, new_inners

    def get_font_size(self, image_width, image_height, font_scale=2e-3, thickness_scale=5e-3):
        font_scale = (image_height+image_width) * font_scale
        font_scale = font_scale if font_scale > 0.5 else 0.5
        thickness = int(math.ceil(min(image_height, image_width) * thickness_scale))
        return font_scale, thickness

    # This is taken from the original LabGym code to keep the colors consistent from the original LabGym
    # to the contours created here.
    def getContourColor(self, index, length):

        # Use different colors to indicate the sequence of contours
        if index < length / 4:
            d = index * int((255 * 4 / length))
            color = (255, d, 9)
        elif index < length / 2:
            d = int((index - length / 4) * (255 * 4 / length))
            color = (255, 255, d)
        elif index < 3 * length / 4:
            d = int((index - length / 2) * (255 * 4 / length))
            color = (255, 255 - d, 255)
        else:
            d = int((index - 3 * length / 4) * (255 * 4 / length))
            color = (255 - d, 0, 255)
        return color


""""
# Copied straight from here: https://realpython.com/python-timer/#a-python-timer-class with a minor modification 
# to print a custom message. 
# This is inserted so that I can easily test the timing within the code.
# I just declare a name=Timer("name of my timer or message") and then I can start and stop it 
# anywhere in the code.  When stopped, it prints out my message that I initialized it with above. 

"""

import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self, name):
        self._start_time = None
        self.name = name

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f'({self.name} timer Elapsed time: {elapsed_time:0.4f} seconds)')
