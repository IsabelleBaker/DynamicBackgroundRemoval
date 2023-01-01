import os
import numpy as np
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class thing_masker:
    def __init__(self, things=[], dataset_name='', \
                 dataset_path=''):
        self.classes = things
        dict_temp = {'thing_classes': things}

        # register a catalog
        if str("thing_masker_" + dataset_name) in DatasetCatalog.list():
            DatasetCatalog.remove(str("thing_masker_" + dataset_name))
        DatasetCatalog.register(
            str("thing_masker_" + dataset_name),
            lambda d: dict_temp
        )
        MetadataCatalog.get(str("thing_masker_" + \
                                dataset_name)).set(thing_classes=things)

        # Set the thing_masker internal variables
        self.metadata = MetadataCatalog.get(str("thing_masker_" + \
                                                dataset_name))
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_path = dataset_path + dataset_name + '/'
        self.test_directory = self.data_path + 'test/'
        self.video_test_directory = self.data_path + 'videos/'
        self.video_output_directory = self.data_path + 'videos_output/'
        self.cfg = get_cfg()

        # This will be updated later to load a local config file
        # for now, re-load the coco base config.
        self.outputs = None

        # update the detection threshold

    def update_prediction_percentage(self, percentage):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = percentage
        self.predictor = DefaultPredictor(self.cfg)

    def init_predictor(self, threshold=0.5, cuda_device='cpu', weights=None, \
                       config=None):
        if not config:
            self.cfg.merge_from_file(model_zoo.get_config_file( \
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.DATASETS.TEST = ("thing_masker",)
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(len(self.classes))
        else:
            self.cfg.merge_from_file(config)
        self.cfg.MODEL.DEVICE = cuda_device
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.MODEL.WEIGHTS = weights
        self.predictor = DefaultPredictor(self.cfg)

        # get the list of files to test based on the provided extensions

    def get_test_files(self, extensions=[], directory="./"):
        images = []
        for ext in extensions:
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    if name.endswith(ext):
                        images.append(root + name)
        return images

        # Given an input image, find the animals and return them without
        # a background

    def get_masked_image(self, img=None, view_classes=[]):
        # get the masks
        mask_instances = self.outputs['instances']

        i = 0
        for i in range(len(self.classes)):
            if not self.classes[i] in view_classes:
                mask_instances = mask_instances[mask_instances.pred_classes != i]
        masks = mask_instances.pred_masks.to("cpu")
        composite = Image.new('RGB', Image.fromarray(img, mode='RGB').size)

        # loop through the masks and paste their section from the original
        # image into a blank canvas
        for item_mask in masks:
            # Get the true bounding box of the mask
            segmentation = np.where(item_mask == True)
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            # create a cropped image from just the portion of the image we want
            cropped = Image.fromarray(img[y_min:y_max, x_min:x_max, :],
                                      mode='RGB')

            # create a PIL image out of the mask
            mask = Image.fromarray(np.uint8(item_mask * 255))

            # Crop the mask to match the cropped image
            cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

            # load in a background image and choose a paste position
            paste_position = (x_min, y_min)

            # create a new foreground image as large as the composite and
            # past the cropped image on top
            new_fg_image = Image.new('RGB', composite.size)
            new_fg_image.paste(cropped, paste_position)

            # create a new alpha mask as large as the composite and paste
            # the cropped mask
            new_alpha_mask = Image.new('L', composite.size, color=0)
            new_alpha_mask.paste(cropped_mask, paste_position)

            # compose the new foreground and background using an alpha mask
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)
        return composite

    def get_original_with_masks(self, image=None, view_classes=None):
        # todo: build a custom visualizer, and add
        # Non Maximum Suppression
        if not view_classes: view_classes = self.classes
        v = Visualizer(image[:, :, ::-1],
                       metadata=self.metadata,
                       scale=1.0,
                       instance_mode=ColorMode.SEGMENTATION
                       )

        display_filter = self.outputs['instances']
        i = 0
        for i in range(len(self.classes)):
            if not self.classes[i] in view_classes:
                display_filter = display_filter[display_filter.pred_classes != i]

        v = v.draw_instance_predictions(display_filter.to('cpu'))
        return v

    def update_outputs(self, img=None):
        self.outputs = self.predictor(img)
        return
