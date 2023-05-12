import wx
import wx.lib.agw.pygauge as PG
import time
import threading
import shutil
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import torch, torchvision, cv2, random
from datetime import datetime
from typing import Dict, List
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
import detectron2
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.export import (
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from torch import Tensor, nn


# Class: ModelTrainerInitialWindow
# Description: This class was taken from LabGym and then modified for User input
class ModelTrainerInitialWindow(wx.Frame):

    def __init__(self, title):
        wx.Frame.__init__(self, parent=None, title=title)
        self.panel = ModelTrainerPanel(self)
        self.frame_sizer = wx.BoxSizer(wx.VERTICAL)
        self.frame_sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(self.frame_sizer)
        self.Size = (self.panel.BestVirtualSize[0] + 30, self.panel.BestVirtualSize[1] + 40)
        self.Move(wx.Point(50, 50))
        self.Show()


class ModelTrainerPanel(wx.ScrolledWindow):

    def __init__(self, parent):
        wx.ScrolledWindow.__init__(self, parent, id=-1, pos=wx.DefaultPosition, size=wx.DefaultSize,
                                   style=wx.HSCROLL | wx.VSCROLL,
                                   name="scrolledWindow")
        self.SetScrollbars(1, 1, 600, 400)

        # Set up the variables that we want to capture
        self.dataset_path = None
        self.inference_size = None
        self.animal = None
        self.training_annotation_path = None
        self.prediction_threshold = None
        self.output_path = None
        self.percent_video_complete = 0
        self.keep_training = True
        self.training_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.progressbar_update, self.training_timer)
        self.Bind(EVT_TRAIN, self.OnTrain)

        # ADDING IN BUTTONS AND LABELS

        # Create the text that says "Enter Inputs for Model Training" and add it to the vertical window container
        main_label = wx.StaticText(self, label='Enter Inputs for Model Training')

        # Add the button to get the dataset directory and bind its event function
        get_dataset_button = wx.Button(self, label='Select Dataset Folder')
        get_dataset_button.SetToolTip('Select the folder containing the images '
                                      'that your annotations were done on.')

        get_dataset_button.Bind(wx.EVT_BUTTON, self.evt_get_dataset_path)
        self.get_dataset_label = wx.TextCtrl(self, value='', style=wx.TE_LEFT, size=(300, -1))
        self.get_dataset_label.SetHint('Select the folder containing the images '
                                       'that your annotations were done on.')
        self.get_dataset_label.SetHint('{your dataset}')

        # Add the button to get the training annotation directory and bind its event function
        get_training_annotation_button = wx.Button(self, label='Select Training Annotation File (json)')
        get_training_annotation_button.SetToolTip('Select the json file containing your training '
                                                  'annotations.')
        get_training_annotation_button.Bind(wx.EVT_BUTTON, self.evt_get_training_annotation_path)
        self.get_training_annotation_label = wx.TextCtrl(self, value='', style=wx.TE_LEFT, size=(300, -1))
        self.get_training_annotation_label.SetToolTip('Select the json file containing your training '
                                                      'annotations.')
        self.get_training_annotation_label.SetHint('{your training annotation}')

        # Add the button to get the output path and bind its event function
        get_output_path_button = wx.Button(self, label='Select Output Folder')
        get_output_path_button.SetToolTip('Select the folder in which you would like your '
                                          'model and the corresponding txt file to be placed.')
        get_output_path_button.Bind(wx.EVT_BUTTON, self.evt_get_output_path)
        self.get_output_path_label = wx.TextCtrl(self, value='', style=wx.TE_LEFT, size=(300, -1))
        self.get_output_path_label.SetToolTip('Select the folder in which you would like your '
                                              'model and the corresponding txt file to be placed.')
        self.get_output_path_label.SetHint('{your output path}')

        # Add the text for inference size and bind its event function
        inference_size_text = wx.StaticText(self, label='Training Size')
        inference_size_text.SetToolTip('Select the size (in pixels) that you would like '
                                       'your model to be trained at. The usage of the model '
                                       'will be optimized at this size.')
        self.inference_size_widget = wx.SpinCtrlDouble(self, initial=256, min=64, max=1280, inc=64)
        self.inference_size_widget.SetToolTip('Select the size (in pixels) that you would like '
                                              'your model to be trained at. The usage of the model '
                                              'will be optimized at this size.')
        self.inference_size_widget.Bind(wx.EVT_SPINCTRLDOUBLE, self.evt_set_inference_size)
        self.inference_size = self.inference_size_widget.GetValue()

        # Add the text for prediction threshold and bind its event function
        prediction_threshold_text = wx.StaticText(self, label='Prediction Threshold')
        prediction_threshold_text.SetToolTip('Select a prediction threshold 0.00-1.00 that '
                                             'will be the lowest possible detection threshold '
                                             'for your model. (0.5 is a good baseline)')
        self.prediction_threshold_widget = wx.SpinCtrlDouble(self, min=0.00, max=1.00, inc=0.01, initial=0.50)
        self.prediction_threshold_widget.SetToolTip('Select a prediction threshold 0.00-1.00 that '
                                                    'will be the lowest possible detection threshold '
                                                    'for your model. (0.5 is a good baseline)')
        self.prediction_threshold_widget.Bind(wx.EVT_SPINCTRLDOUBLE, self.evt_set_prediction_threshold)
        self.prediction_threshold = self.prediction_threshold_widget.GetValue()

        # Add the text for max iterations and bind its event function
        max_iterations_text = wx.StaticText(self, label='Max Iterations')
        max_iterations_text.SetToolTip('Select a number for max iterations which decides '
                                       'for how many loops your model will be trained, '
                                       'but a greater number requires more time for the training.')
        self.max_iterations_widget = wx.SpinCtrlDouble(self, min=50, max=99999, inc=50, initial=200)
        self.max_iterations_widget.SetToolTip('Select a number for max iterations which decides '
                                              'for how many loops your model will be trained, '
                                              'but a greater number requires more time for the training.')
        self.max_iterations_widget.Bind(wx.EVT_SPINCTRLDOUBLE, self.evt_set_max_iterations)
        self.max_iterations = self.max_iterations_widget.GetValue()

        self.enable_mps_checkbox = wx.CheckBox(self,
                                               label='Enable Support for \n Apple Silicon (Experimental)')
        self.enable_mps_checkbox.SetValue(False)
        self.enable_mps_checkbox.SetToolTip(
            'Apple Silicon support in Pytorch is still a work in progress. Use at your own risk')

        # Add text for start training and bind its event function
        self.train_model_button = wx.Button(self, label='Train Model')
        self.train_model_button.Bind(wx.EVT_BUTTON, self.evt_train_model)

        # Done button
        done_button = wx.Button(self, label='Done')
        done_button.Bind(wx.EVT_BUTTON, self.evt_done)
        self.gauge_label = wx.StaticText(self, label='Progress: Idle')
        self.gauge_widget = PG.PyGauge(self, range=100, size=(250, 25), style=wx.GA_HORIZONTAL)
        self.gauge_widget.SetBarColor((0, 0, 255))  # blue bar
        self.gauge_widget.SetBorderColor(wx.BLACK)
        # FORMATTING THE PANEL

        # Set up the container (BoxSizer) for the overall display window. Within this window, we will
        # place additional containers for sets of input and capabilities.
        overall_window_vertical = wx.BoxSizer(wx.VERTICAL)
        overall_window_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        overall_window_vertical.Add(0, 15)
        overall_window_vertical.Add(main_label)

        # Set up the Main Part of the Gui
        main_parameter_sizer_vertical = wx.StaticBox(self)
        main_parameter_options_vertical = wx.StaticBoxSizer(main_parameter_sizer_vertical, wx.VERTICAL)
        main_parameter_options = wx.BoxSizer(wx.HORIZONTAL)

        # Make the Button to get the Dataset
        get_dataset_sizer_vertical = wx.StaticBox(self)
        get_dataset_options_vertical = wx.StaticBoxSizer(get_dataset_sizer_vertical, wx.VERTICAL)
        get_dataset_options = wx.BoxSizer(wx.HORIZONTAL)

        get_dataset_options.Add(get_dataset_button)
        get_dataset_options.Add(10, 0)
        get_dataset_options.Add(self.get_dataset_label, wx.EXPAND)
        get_dataset_options_vertical.Add(0, 5)
        get_dataset_options_vertical.Add(get_dataset_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

        # Make the Button to get the Annotation File
        get_training_annotation_sizer_vertical = wx.StaticBox(self)
        get_training_annotation_options_vertical = wx.StaticBoxSizer(get_training_annotation_sizer_vertical,
                                                                     wx.VERTICAL)
        get_training_annotation_options = wx.BoxSizer(wx.HORIZONTAL)

        get_training_annotation_options.Add(get_training_annotation_button)
        get_training_annotation_options.Add(10, 0)
        get_training_annotation_options.Add(self.get_training_annotation_label, wx.EXPAND)
        get_training_annotation_options_vertical.Add(0, 5)
        get_training_annotation_options_vertical.Add(get_training_annotation_options, wx.ALIGN_CENTER_VERTICAL,
                                                     wx.EXPAND)

        # Make the Button to get the output path
        get_output_path_sizer_vertical = wx.StaticBox(self)
        get_output_path_options_vertical = wx.StaticBoxSizer(get_output_path_sizer_vertical, wx.VERTICAL)
        get_output_path_options = wx.BoxSizer(wx.HORIZONTAL)

        get_output_path_options.Add(get_output_path_button)
        get_output_path_options.Add(10, 0)
        get_output_path_options.Add(self.get_output_path_label, wx.EXPAND)
        get_output_path_options_vertical.Add(0, 5)
        get_output_path_options_vertical.Add(get_output_path_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

        training_options_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        # Add the inference size widgets
        inference_size_box = wx.StaticBox(self)
        inference_size_sizer = wx.StaticBoxSizer(inference_size_box, wx.VERTICAL)
        inference_size_sizer.Add(inference_size_text, flag=wx.ALIGN_CENTER_HORIZONTAL)
        inference_size_sizer.Add(self.inference_size_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

        # Add the prediction threshold widgets
        prediction_threshold_box = wx.StaticBox(self)
        prediction_threshold_sizer = wx.StaticBoxSizer(prediction_threshold_box, wx.VERTICAL)
        prediction_threshold_sizer.Add(prediction_threshold_text, flag=wx.ALIGN_CENTER_HORIZONTAL)
        prediction_threshold_sizer.Add(self.prediction_threshold_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

        # Add the max iterations widgets
        max_iterations_box = wx.StaticBox(self)
        max_iterations_sizer = wx.StaticBoxSizer(max_iterations_box, wx.VERTICAL)
        max_iterations_sizer.Add(max_iterations_text, flag=wx.ALIGN_CENTER_HORIZONTAL)
        max_iterations_sizer.Add(self.max_iterations_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

        # Add the mps enable widgets
        enable_mps_box = wx.StaticBox(self)
        enable_mps_sizer = wx.StaticBoxSizer(enable_mps_box, wx.VERTICAL)
        enable_mps_sizer.Add(self.enable_mps_checkbox, flag=wx.ALIGN_CENTER_HORIZONTAL)

        training_options_horizontal.Add(inference_size_sizer, flag=wx.EXPAND)
        training_options_horizontal.Add(prediction_threshold_sizer, flag=wx.EXPAND)
        training_options_horizontal.Add(max_iterations_sizer, flag=wx.EXPAND)
        training_options_horizontal.Add(enable_mps_sizer, flag=wx.EXPAND)

        # Place the train model and stop train model button
        train_model_button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        train_model_button_sizer.Add(10, 0)
        train_model_button_sizer.Add(self.train_model_button, wx.LEFT)
        train_model_button_sizer.Add(10, 0)

        # Place all items into the box.
        main_parameter_options_vertical.Add(get_dataset_options_vertical, flag=wx.EXPAND)
        main_parameter_options_vertical.Add(0, 5)
        main_parameter_options_vertical.Add(get_training_annotation_options_vertical, flag=wx.EXPAND)
        main_parameter_options_vertical.Add(0, 5)
        main_parameter_options_vertical.Add(get_output_path_options_vertical, flag=wx.EXPAND)
        main_parameter_options_vertical.Add(0, 5)
        main_parameter_options_vertical.Add(training_options_horizontal, flag=wx.EXPAND)
        main_parameter_options_vertical.Add(0, 5)
        main_parameter_options_vertical.Add(train_model_button_sizer, wx.LEFT)

        # Add the main options to the vertical window container
        overall_window_vertical.Add(main_parameter_options_vertical, flag=wx.EXPAND)
        overall_window_vertical.Add(0, 5)

        # Add the done button at the bottom of the panel
        done_button_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        done_box = wx.StaticBox(self)
        done_sizer = wx.StaticBoxSizer(done_box, wx.VERTICAL)
        done_sizer.Add(self.gauge_label, wx.ALIGN_CENTER)
        done_sizer.Add(self.gauge_widget)
        done_sizer.Add(0, 10)
        done_sizer.Add(done_button)
        done_button_horizontal.Add(done_sizer)
        overall_window_vertical.Add(done_button_horizontal, wx.LEFT)
        overall_window_vertical.Add(0, 5)
        overall_window_horizontal.Add(15, 0)
        overall_window_horizontal.Add(overall_window_vertical, wx.EXPAND)
        overall_window_horizontal.Add(15, 0)
        self.SetSizer(overall_window_horizontal)

    # SET UP BUTTON FUNCTIONALITY/EVENTS

    # Event for get dataset button
    def evt_get_dataset_path(self, event):
        dlg = wx.DirDialog(None, "Choose dataset folder containing the image samples", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.dataset_path = dlg.GetPath()
            self.get_dataset_label.LabelText = " " + self.dataset_path
        dlg.Destroy()

    # Event for get training annotation button
    def evt_get_training_annotation_path(self, event):
        """
        Create and show the Open FileDialog
        """
        wildcard = "Model Files (*.json)|*.json"
        dlg = wx.FileDialog(
            self, message="Choose the json file for your training annotations",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.training_annotation_path = path
            self.get_training_annotation_label.SetValue(os.path.basename(path))
        dlg.Destroy()

    # Event for get dataset button
    def evt_get_output_path(self, event):
        dlg = wx.DirDialog(None, "Choose output folder", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.output_path = dlg.GetPath()
            self.get_output_path_label.LabelText = " " + self.output_path
        dlg.Destroy()

    # Event for inference size button
    def evt_set_inference_size(self, event):
        self.inference_size = self.inference_size_widget.GetValue()

    # Event for prediction threshold button
    def evt_set_prediction_threshold(self, event):
        self.prediction_threshold = self.prediction_threshold_widget.GetValue()

    # Event for max iterations button
    def evt_set_max_iterations(self, event):
        self.max_iterations = int(self.max_iterations_widget.GetValue())

    # Event for train model button
    def evt_train_model(self, event):
        self.animal = 'model'  # str(self.get_animal_mapping_label.GetValue())
        if not self.dataset_path:
            dlg = wx.GenericMessageDialog(None, 'No dataset path has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.training_annotation_path:
            dlg = wx.GenericMessageDialog(None, 'No training annotation path has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.output_path:
            dlg = wx.GenericMessageDialog(None, 'No output path has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.inference_size:
            dlg = wx.GenericMessageDialog(None, 'No inference size has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.prediction_threshold:
            dlg = wx.GenericMessageDialog(None, 'No prediction threshold has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.max_iterations:
            dlg = wx.GenericMessageDialog(None, 'No max iterations has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        # thread = threading.Thread(target=self.get_trained_model)
        # thread.run()

        self.get_trained_model()

    # Export the Detectron2 model to Torchscript
    def export_scripting(self, torch_model, modelname, output_path):
        assert TORCH_VERSION >= (1, 8)
        fields = {
            "proposal_boxes": Boxes,
            "objectness_logits": Tensor,
            "pred_boxes": Boxes,
            "scores": Tensor,
            "pred_classes": Tensor,
            "pred_masks": Tensor,
            "pred_keypoints": torch.Tensor,
            "pred_keypoint_heatmaps": torch.Tensor,
        }

        class ScriptableAdapterBase(nn.Module):
            # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
            # by not returning instances but dicts. Otherwise, the exported model is not deployable
            def __init__(self):
                super().__init__()
                self.model = torch_model
                self.eval()

        if isinstance(torch_model, GeneralizedRCNN):
            class ScriptableAdapter(ScriptableAdapterBase):
                def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                    instances = self.model.inference(inputs, do_postprocess=False)
                    return [instance.get_fields() for instance in instances]

        ts_model = scripting_with_instances(ScriptableAdapter(), fields)
        with PathManager.open(modelname, "wb") as file_pointer:
            torch.jit.save(ts_model, file_pointer)
        dump_torchscript_IR(ts_model, output_path)
        return None

    # Train the background removal model
    def get_trained_model(self):
        # See if GPUs can be used
        if torch.cuda.is_available():
            device = 'cuda'
        elif self.enable_mps_checkbox.GetValue():
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

        # Declare variables for training
        my_inference_size_min = int(self.inference_size)
        my_inference_size_max = int(self.inference_size)
        my_dataset_path = self.dataset_path
        my_training_annotation_path = self.training_annotation_path
        my_output_path = self.output_path
        my_max_iterations = int(self.max_iterations)
        my_IMS_PER_BATCH = 4

        if str('internal_train') in DatasetCatalog.list():
            DatasetCatalog.remove('internal_train')
        register_coco_instances('internal_train', {},
                                my_training_annotation_path,
                                my_dataset_path)
        my_dataset_metadata = MetadataCatalog.get('internal_train')
        dataset_dicts = DatasetCatalog.get('internal_train')
        my_things = MetadataCatalog.get('internal_train').thing_classes
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.OUTPUT_DIR = os.path.join(my_output_path, 'logs')
        cfg.DATASETS.TRAIN = ('internal_train',)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(len(my_things))
        cfg.MODEL.RETINANET.NUM_CLASSES = int(len(my_things))
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = int(len(my_things))
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(len(my_things))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TEST = ('internal_train',)
        cfg.SOLVER.MAX_ITER = int(my_max_iterations)
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.WARMUP_ITERS = int(my_max_iterations * .1)
        cfg.SOLVER.STEPS = (int(my_max_iterations * .4), int(my_max_iterations * .8),)
        cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.IMS_PER_BATCH = my_IMS_PER_BATCH
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MIN_SIZE_TEST = my_inference_size_min
        cfg.INPUT.MAX_SIZE_TEST = my_inference_size_max
        cfg.INPUT.MIN_SIZE_TRAIN = my_inference_size_min
        cfg.INPUT.MAX_SIZE_TRAIN = my_inference_size_max
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(my_output_path, 'models'), exist_ok=True)
        self.trainer = DefaultTrainer(cfg)
        self.trainer.resume_or_load(False)
        self.training_timer.Start(10000)
        self.gauge_label.SetLabel(f'Progress: starting...')

        threaded_trainer = TrainingThread(self, self.trainer)
        threaded_trainer.start()

    def OnTrain(self, event):
        print('Done Training')
        self.training_timer.Stop()
        current_time = datetime.now()
        date_time = f'_{str(current_time.month)}_{str(current_time.day)}_{str(current_time.year)}_' \
                    f'{str(current_time.hour)}_{str(current_time.minute)}_{str(current_time.second)}'

        os.makedirs(os.path.join(self.output_path, 'models', 'model' + date_time), exist_ok=True)
        my_things = MetadataCatalog.get('internal_train').thing_classes
        print('Progress: evaluating model')
        self.gauge_label.SetLabel('Progress: evaluating model')
        self.gauge_label.Refresh()
        evaluator = detectron2.evaluation.COCOEvaluator('internal_train',
                                                        distributed=False, output_dir=self.trainer.cfg.OUTPUT_DIR)
        val_loader = detectron2.data.build_detection_test_loader(self.trainer.cfg, 'internal_train')
        print(detectron2.evaluation.inference_on_dataset(self.trainer.model, val_loader, evaluator))

        # Export the animal names
        print('Progress: exporting model to torchscript')
        self.gauge_label.SetLabel('Progress: exporting model to torchscript')
        self.gauge_label.Refresh()
        class_list_names = os.path.join(self.output_path, 'models', 'model' + date_time,
                                        'model' + date_time + '_class_list.txt')

        class_list = {'animal_mapping': {}, 'training_size': int(self.inference_size)}
        for i in range(len(my_things)):
            class_list['animal_mapping'][i] = my_things[i]

        with open(class_list_names, 'w') as class_file:
            class_file.write(json.dumps(class_list))

        predictor = DefaultPredictor(self.trainer.cfg)
        model = predictor.model
        DetectionCheckpointer(model).resume_or_load(os.path.join(self.trainer.cfg.OUTPUT_DIR, "model_final.pth"))
        model.eval()
        # Create the exported model name and location
        standalone_model = os.path.join(self.output_path, 'models', 'model' + date_time, 'model' + date_time + '.ts')

        model_location = os.path.join(self.output_path, 'logs')

        # Export the scripted model
        self.export_scripting(model, standalone_model, model_location)
        print('Progress: exporting model config')
        self.gauge_label.SetLabel('Progress: exporting model config')
        self.gauge_label.Refresh()
        config_file = os.path.join(self.output_path, 'models', 'model' + date_time, 'model' + date_time + '.yaml')
        with open(config_file, "w") as f:
            f.write(self.trainer.cfg.dump())

        self.gauge_widget.SetValue(0)
        self.gauge_widget.Refresh()
        self.gauge_label.SetLabel("Progress: Idle")

    # Event for done button
    def evt_done(self, event):
        self.Parent.Destroy()

    def progressbar_update(self, event):
        update_gauge_percentage = int(
            (self.trainer.iter / self.trainer.max_iter) * 100)
        self.gauge_widget.SetValue(update_gauge_percentage)
        self.gauge_widget.Refresh()
        self.gauge_label.SetLabel(f'Progress: {str(update_gauge_percentage)}%')


myEVT_TRAIN = wx.NewEventType()
EVT_TRAIN = wx.PyEventBinder(myEVT_TRAIN, 1)


class TrainingFinishedEvent(wx.PyCommandEvent):

    def __init__(self, etype, eid, value=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self._value = value

    def GetValue(self):
        return self._value


class TrainingThread(threading.Thread):
    def __init__(self, parent, mytrainer):
        threading.Thread.__init__(self)
        self._parent = parent
        self._trainer = mytrainer
        self._value = False

    def run(self):
        self._trainer.train()
        self._value = True
        evt = TrainingFinishedEvent(myEVT_TRAIN, -1, self._value)
        wx.PostEvent(self._parent, evt)


# Run the program
if __name__ == '__main__':
    app = wx.App()
    ModelTrainerInitialWindow("Baker's Mask R-CNN Model Trainer")
    app.MainLoop()
