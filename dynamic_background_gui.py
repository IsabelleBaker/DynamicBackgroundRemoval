import os
import cv2
import wx
import json
import threading
from dynamic_background_remover import dynamic_background_remover


# Class: DynamicBackgroundInitialWindow
# Description: This class was taken from LabGym and then modified for User input
class DynamicBackgroundInitialWindow(wx.Frame):

    def __init__(self, title, mode='MIN'):
        wx.Frame.__init__(self, parent=None, title=title)
        self.panel = DynamicBackgroundPanel(self, mode=mode)
        self.frame_sizer = wx.BoxSizer(wx.VERTICAL)
        self.frame_sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(self.frame_sizer)
        self.Size = (self.panel.BestVirtualSize[0] + 30, self.panel.BestVirtualSize[1] + 40)
        self.Move(wx.Point(50, 50))
        self.Show()


class DynamicBackgroundPanel(wx.ScrolledWindow):

    def __init__(self, parent, mode='TEST'):
        wx.ScrolledWindow.__init__(self, parent, id=-1, pos=wx.DefaultPosition, size=wx.DefaultSize,
                                   style=wx.HSCROLL | wx.VSCROLL,
                                   name="scrolledWindow")
        self.SetScrollbars(1, 1, 600, 400)

        # Set up the variables that we want to capture
        self.inference_size = None
        self.animals = None
        self.thing_choice = None
        self.model_path = None
        self.model_folder_path = None
        self.thing_names_path = None
        self.video_path = None
        self.detection_threshold = None
        self.output_path = os.path.join(os.getcwd(), 'videos_output')
        self.image_path = None
        self.display_things = []
        self.percent_video_complete = 0
        self.mode = mode

        # Start of Step 1 GUI

        # Create the text that says "Step 1...." and add it to the vertical window container
        note = wx.StaticText(self, label='*Hover over any button or parameter for a description')
        step_1 = wx.StaticText(self, label='Set Up Your Model')

        # Add the button to get the input directory and bind its event function
        get_model_button = wx.Button(self, label='Select A Model Folder')
        get_model_button.SetToolTip('Navigate to the location of the desired model '
                                    '(for dynamic background removal) and select it')
        get_model_button.Bind(wx.EVT_BUTTON, self.evt_get_model)
        self.get_model_label = wx.TextCtrl(self, value='{your model}', style=wx.TE_LEFT, size=(300, -1))
        self.get_model_label.SetToolTip('Navigate to the location of the desired model '
                                        '(for dynamic background removal) and select it')

        self.inference_size_text = wx.StaticText(self, label='Frame Processing Size')
        self.inference_size_text.SetToolTip('Enter the size (in pixels) that you would like '
                                       'your video to be processed at. A smaller size is more '
                                       'efficient but produces lower quality results.')
        self.inference_size_widget = wx.SpinCtrlDouble(self, initial=0, min=64, max=1280, inc=64)
        self.inference_size_widget.SetToolTip('Enter the size (in pixels) that you would like '
                                              'your video to be processed at. A smaller size is more '
                                              'efficient but produces lower quality results.')
        self.inference_size_widget.Bind(wx.EVT_SPINCTRLDOUBLE, self.evt_set_inference_size)
        self.inference_size = self.inference_size_widget.GetValue()
        self.inference_size_widget.Disable()
        self.inference_size_text.Disable()

        self.detection_threshold_text = wx.StaticText(self, label='Detection Threshold')
        self.detection_threshold_text.SetToolTip('Enter the percent confidence for detection of an animal '
                                            'you would like (0.00-1.00)')
        self.detection_threshold_widget = wx.SpinCtrlDouble(self, min=0.00, max=1.00, inc=0.01, initial=0.80)
        self.detection_threshold_widget.SetToolTip('Enter the percent confidence for detection of an animal '
                                                   'you would like (0.00-1.00)')
        self.detection_threshold_widget.Bind(wx.EVT_SPINCTRLDOUBLE, self.evt_set_detection_threshold)
        self.detection_threshold = self.detection_threshold_widget.GetValue()
        self.detection_threshold_text.Disable()
        self.detection_threshold_widget.Disable()

        self.enable_mps_checkbox = wx.CheckBox(self,
                                               label='Enable Support for \n Apple Silicon (Experimental)')  # Create the checkbox
        self.enable_mps_checkbox.SetValue(False)  # Set the default to "checked
        self.enable_mps_checkbox.SetToolTip(
            'Apple Silicon support in Pytorch is still a work in progress. Use at your own risk')
        self.enable_mps_checkbox.Disable()
        # End of Step 1 GUI

        if self.mode == 'TEST':
            # Start of Step 2 GUI

            # Add the text that says "Step 2...." to the vertical window container
            step_2 = wx.StaticText(self, label='Test Model On An Image (optional)')

            # Add the button to get the input directory and bind its event function
            get_image_button = wx.Button(self, label='Select An Image')
            get_image_button.Bind(wx.EVT_BUTTON, self.evt_get_image)
            get_image_button.SetToolTip('Specify an Image to use to test the model')
            self.get_image_label = wx.TextCtrl(self, value='{your image}', style=wx.TE_LEFT, size=(300, -1))
            self.get_image_label.SetToolTip('Specify an Image to use to test the model')
            # Add the checkbox widgets in their own row.

            self.image_with_masks_checkbox = wx.CheckBox(self, label='Masks')  # Create the checkbox
            self.image_with_masks_checkbox.SetValue(True)  # Set the default to "checked
            self.image_with_masks_checkbox.SetToolTip('Produce an image showing the original with solid masks overlaid')

            self.image_masked_checkbox = wx.CheckBox(self, label='Masked')  # Create the checkbox
            self.image_masked_checkbox.SetValue(True)  # Set the default to "checked"
            self.image_masked_checkbox.SetToolTip(
                'Produce an image showing only the detected animals in the original image')

            self.image_with_contours_checkbox = wx.CheckBox(self, label='Outlined')  # Create the checkbox
            self.image_with_contours_checkbox.SetValue(True)  # Set the default to "checked"
            self.image_with_contours_checkbox.SetToolTip('Produce an image with the detected animals outlined')

            self.image_decorate_checkbox = wx.CheckBox(self, label='Decorate')  # Create the checkbox
            self.image_decorate_checkbox.SetValue(True)  # Set the default to "checked"
            self.image_decorate_checkbox.SetToolTip(
                'When producing images show bounding boxes, animal names, and confidence values')

            process_image_button = wx.Button(self, label='Process Image')
            process_image_button.Bind(wx.EVT_BUTTON, self.evt_process_image)
            process_image_button.SetToolTip('Run the image through the specified model')

            ###### End of Step 2 GUI

        if self.mode == 'TEST' or self.mode == 'SAMPLES':
            ###### Start of Step 3 GUI

            # Add the text that says "Step 3...." to the vertical window container
            step_3 = wx.StaticText(self, label='Process a Video')

            # add the button to get the input directory and bind its event function
            get_video_path_button = wx.Button(self, label='Select A Video')
            get_video_path_button.Bind(wx.EVT_BUTTON, self.evt_get_video_path)
            self.get_video_path_label = wx.TextCtrl(self, value='{your video}', style=wx.TE_LEFT, size=(300, -1))
            self.get_video_path_label.SetToolTip('Navigate to the location of the video '
                                                 'you would like to process and select it')
            get_video_path_button.SetToolTip('Navigate to the location of the video '
                                             'you would like to process and select it')

            # Add the button to get the output directory and bind its event function
            get_video_output_directory_button = wx.Button(self, label='Select Output Path')
            get_video_output_directory_button.Bind(wx.EVT_BUTTON, self.evt_get_video_output_directory)
            self.get_video_output_directory_label = wx.TextCtrl(self, value=self.output_path, style=wx.TE_LEFT,
                                                                size=(300, -1))
            get_video_output_directory_button.SetToolTip('Navigate to the location where you would like '
                                                         'to place your output and select it')
            self.get_video_output_directory_label.SetToolTip('Navigate to the location where you would like '
                                                             'to place your output and select it')

            self.video_masked_checkbox = wx.CheckBox(self, label='Masked')  # Create the checkbox
            self.video_masked_checkbox.SetValue(True)  # Set the default to "checked"
            self.video_masked_checkbox.SetToolTip('Produce a video showing only the detected animals in '
                                                  'the original image')

            self.video_with_masks_checkbox = wx.CheckBox(self, label='With Masks')  # Create the checkbox
            self.video_with_masks_checkbox.SetValue(True)
            self.video_with_masks_checkbox.SetToolTip('Produce a video showing the original with solid masks overlaid')

            self.video_contours_checkbox = wx.CheckBox(self, label='Outlined')  # Create the checkbox
            self.video_contours_checkbox.SetValue(True)  # Set the default to "checked"
            self.video_contours_checkbox.SetToolTip('Produce a video with the detected animals outlined')

            self.video_decorate_checkbox = wx.CheckBox(self, label='Decorate Videos')  # Create the checkbox
            self.video_decorate_checkbox.SetValue(True)
            self.video_decorate_checkbox.SetToolTip('When producing videos show bounding boxes, animal names, and '
                                                    'confidence values')

            self.video_show_progress_checkbox = wx.CheckBox(self, label='Show Progress')  # Create the checkbox
            self.video_show_progress_checkbox.SetValue(True)
            self.video_show_progress_checkbox.SetToolTip('Display the progress videos while analyzing the video')

            self.video_save_tracker_info_checkbox = wx.CheckBox(self, label='Save Tracker Info')  # Create the checkbox
            self.video_save_tracker_info_checkbox.SetValue(True)
            self.video_save_tracker_info_checkbox.SetToolTip('Save all animal tracking info into a pkl file when the '
                                                             'analysis session is complete. You may use this file '
                                                             'within one of my other tools to see the tracking. ')

            self.video_debug_masks_checkbox = wx.CheckBox(self, label='Debug Masks')  # Create the checkbox
            self.video_debug_masks_checkbox.SetToolTip('Save frames of individual animals into a special debug folder.'
                                                       ' This is useful for understand exactly how the animation videos'
                                                       ' and contour images are being produced.')
            self.video_save_progress_videos_checkbox = wx.CheckBox(self, label='Save Progress Videos')
            self.video_save_progress_videos_checkbox.SetValue(True)
            self.video_save_progress_videos_checkbox.SetToolTip('Save the progress videos (masked, with masks, '
                                                                'and/or outlined) when analysis is complete.')

            if self.mode != 'TEST':
                self.video_debug_masks_checkbox.Hide()
                self.video_save_progress_videos_checkbox.Hide()
                self.video_decorate_checkbox.Hide()
                self.video_with_masks_checkbox.Hide()
                self.video_show_progress_checkbox.Hide()

            video_start_time_text = wx.StaticText(self, label='Start \nTime (s)', style=wx.ALIGN_CENTER_HORIZONTAL)
            self.video_start_time_widget = wx.SpinCtrl(self, initial=0, min=0, max=1000000)
            video_start_time_text.SetToolTip('Enter the time at which you would like to start '
                                             'processing the video')
            self.video_start_time_widget.SetToolTip('Enter the time at which you would like to start '
                                                    'processing the video')
            video_duration_text = wx.StaticText(self, label='Duration \n(s)', style=wx.ALIGN_CENTER_HORIZONTAL)
            video_duration_text.SetToolTip('Enter the amount of time in the video you would '
                                           'like to be processed')
            self.video_duration_widget = wx.SpinCtrl(self, initial=-1, min=-1, max=1000000)
            self.video_duration_widget.SetToolTip('Enter the amount of time in the video you would '
                                                  'like to be processed')
            animal_clip_length_text = wx.StaticText(self, label='Mini-Clips \n(frames)',
                                                    style=wx.ALIGN_CENTER_HORIZONTAL)
            self.animal_clip_length_widget = wx.SpinCtrl(self, initial=15, min=0, max=10000)
            animal_clip_length_text.SetToolTip('Enter the number of frames you would like '
                                               'in each mini-clip for behaviors')
            self.animal_clip_length_widget.SetToolTip('Enter the number of frames you would like '
                                                      'in each mini-clip for behaviors')
            lost_threshold_text = wx.StaticText(self, label='Lost Threshold \n(frames)',
                                                style=wx.ALIGN_CENTER_HORIZONTAL)
            lost_threshold_text.SetToolTip('Maximum time (in frames) that an animal can go undetected and then'
                                           'reappear. After this threshold an animal will be marked as permanently'
                                           ' lost.')
            self.lost_threshold_widget = wx.SpinCtrl(self, initial=120, min=0, max=10000)
            self.lost_threshold_widget.SetToolTip('Maximum time (in frames) that an animal can go undetected and then '
                                                  'reappear. After this threshold an animal will be marked as permanently'
                                                  ' lost.')
            batch_size_text = wx.StaticText(self, label='Batch Size \n(frames)', style=wx.ALIGN_CENTER_HORIZONTAL)
            self.batch_size_widget = wx.SpinCtrl(self, initial=1, min=1, max=100)
            batch_size_text.SetToolTip('Batches can speed up the video processing '
                                       'if GPUs are used. Else, keep the value as 1.')
            self.batch_size_widget.SetToolTip('Batches can speed up the video processing '
                                              'if GPUs are used. Else, keep the value as 1.')
            tracking_iou_text = wx.StaticText(self, label='Tracking IoU \nThreshold', style=wx.ALIGN_CENTER_HORIZONTAL)
            tracking_iou_text.SetToolTip(
                'IoU Threshhold used to accept a detected animal as the same animal previously detected. '
                'Do not modify unless you understand the tracking algorithm.')
            self.tracking_iou_widget = wx.SpinCtrlDouble(self, min=0.00, max=1.00, inc=0.01, initial=0.40)
            tracking_iou_text.SetToolTip(
                'IoU Threshhold used to accept a detected animal as the same animal previously detected. '
                'Do not modify unless you understand the tracking algorithm.')

            tracking_distance_text = wx.StaticText(self, label='Max Distance \nMoved(%)',
                                                   style=wx.ALIGN_CENTER_HORIZONTAL)
            tracking_distance_text.SetToolTip("Maximum distance the center of a detected animal's bounding box can "
                                              "move and still be considered the same animal for tracking purposes. "
                                              "This value is a percentage of the image's diagonal length to "
                                              "compensate for differing sizes of animals within given videos. For a "
                                              "large animal in a frame, the value should be larger. "
                                              "Default works well for small animals within a frame. Only used if "
                                              "IoU tracking fails.")

            self.tracking_distance_widget = wx.SpinCtrl(self, initial=5, min=0, max=100)
            self.tracking_distance_widget.SetToolTip("Maximum distance the center of a detected animal's bounding box "
                                                     "can move and still be considered the same animal for tracking "
                                                     "purposes. "
                                                     "This value is a percentage of the image's diagonal length to "
                                                     "compensate for differing sizes of animals within given videos. "
                                                     "For a "
                                                     "large animal in a frame, the value should be larger. "
                                                     "Default works well for small animals within a frame. Only"
                                                     " used if "
                                                     "IoU tracking fails.")

            self.process_video_button = wx.Button(self, label='Process Video')
            self.process_video_button.Bind(wx.EVT_BUTTON, self.evt_process_video)

            stop_processing_video_button = wx.Button(self, label='Stop Processing')
            stop_processing_video_button.Bind(wx.EVT_BUTTON, self.evt_stop_video)
            stop_processing_video_button.SetToolTip('This button stops the current video analysis. Note: This is NOT'
                                                    ' an immediate action. It will gracefully stop the processing '
                                                    'and cleanly close all files. Because of system load, you may'
                                                    ' need to click it several times and then wait. There is no harm'
                                                    ' in clicking it multiple times.')

        done_button = wx.Button(self, label='Done')
        done_button.Bind(wx.EVT_BUTTON, self.evt_done)
        done_button.SetToolTip('Close the application')

        # Set up the container (BoxSizer) for the overall display window. Within this window, we will
        # place additional containers for sets of input and capabilities.
        overall_window_vertical = wx.BoxSizer(wx.VERTICAL)
        overall_window_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        overall_window_vertical.Add(0, 15)
        overall_window_vertical.Add(note)
        overall_window_vertical.Add(0, 15)
        overall_window_vertical.Add(step_1)

        # Set up the Model Part of the Gui
        model_parameter_sizer_vertical = wx.StaticBox(self)
        model_parameter_options_vertical = wx.StaticBoxSizer(model_parameter_sizer_vertical, wx.VERTICAL)
        model_parameter_options = wx.BoxSizer(wx.HORIZONTAL)

        # Make the Button to get the Model
        get_model_sizer_vertical = wx.StaticBox(self)
        get_model_options_vertical = wx.StaticBoxSizer(get_model_sizer_vertical, wx.VERTICAL)
        get_model_options = wx.BoxSizer(wx.HORIZONTAL)

        get_model_options.Add(get_model_button)
        get_model_options.Add(10, 0)
        get_model_options.Add(self.get_model_label, wx.EXPAND)
        get_model_options_vertical.Add(0, 5)
        get_model_options_vertical.Add(get_model_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

        # Add the inference size widgets in their own row.
        inference_size_box = wx.StaticBox(self)
        inference_size_sizer = wx.StaticBoxSizer(inference_size_box, wx.VERTICAL)

        inference_size_sizer.Add(self.inference_size_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

        inference_size_sizer.Add(self.inference_size_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

        model_parameter_options.Add(inference_size_sizer, flag=wx.ALIGN_CENTER)
        model_parameter_options.Add(5, 0)

        # Add the threshold selection widgets in their own row.
        detection_threshold_box = wx.StaticBox(self)
        detection_threshold_sizer = wx.StaticBoxSizer(detection_threshold_box, wx.VERTICAL)

        detection_threshold_sizer.Add(self.detection_threshold_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

        detection_threshold_sizer.Add(self.detection_threshold_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

        model_parameter_options.Add(detection_threshold_sizer, flag=wx.ALIGN_CENTER)
        model_parameter_options.Add(5, 0)

        # Add the Apple Silicon selection widgets in their own row.
        enable_mps_box = wx.StaticBox(self)
        enable_mps_sizer = wx.StaticBoxSizer(enable_mps_box, wx.VERTICAL)

        enable_mps_sizer.Add(self.enable_mps_checkbox, flag=wx.ALIGN_CENTER_HORIZONTAL)

        model_parameter_options.Add(enable_mps_sizer, flag=wx.ALIGN_CENTER)
        model_parameter_options.Add(5, 0)

        # Place the Model setup items into the box.
        model_parameter_options_vertical.Add(get_model_options_vertical, flag=wx.EXPAND)
        model_parameter_options_vertical.Add(0, 5)
        model_parameter_options_vertical.Add(model_parameter_options, flag=wx.EXPAND)

        # Add the model options to the vertical window container
        overall_window_vertical.Add(model_parameter_options_vertical, flag=wx.EXPAND)

        overall_window_vertical.Add(0, 5)

        if self.mode == 'TEST':
            overall_window_vertical.Add(step_2)

            # Build the image options UI
            image_parameter_sizer_vertical = wx.StaticBox(self)
            image_parameter_options_vertical = wx.StaticBoxSizer(image_parameter_sizer_vertical, wx.VERTICAL)
            image_parameter_options = wx.BoxSizer(wx.HORIZONTAL)

            # Make the Button to get the image
            get_image_sizer_vertical = wx.StaticBox(self)
            get_image_options_vertical = wx.StaticBoxSizer(get_image_sizer_vertical, wx.VERTICAL)

            get_image_options = wx.BoxSizer(wx.HORIZONTAL)
            get_image_options.Add(get_image_button)
            get_image_options.Add(10, 0)
            get_image_options.Add(self.get_image_label, wx.EXPAND)
            get_image_options_vertical.Add(0, 5)
            get_image_options_vertical.Add(get_image_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

            image_parameters_column_1_box = wx.StaticBox(self)
            image_parameters_column_1_sizer = wx.StaticBoxSizer(image_parameters_column_1_box, wx.VERTICAL)

            image_parameters_column_1_sizer.Add(self.image_with_masks_checkbox, flag=wx.ALIGN_CENTER_HORIZONTAL)

            image_parameters_column_2_box = wx.StaticBox(self)
            image_parameters_column_2_sizer = wx.StaticBoxSizer(image_parameters_column_2_box, wx.VERTICAL)

            image_parameters_column_2_sizer.Add(self.image_masked_checkbox, flag=wx.ALIGN_CENTER_HORIZONTAL)

            image_parameters_column_3_box = wx.StaticBox(self)
            image_parameters_column_3_sizer = wx.StaticBoxSizer(image_parameters_column_3_box, wx.VERTICAL)

            image_parameters_column_3_sizer.Add(self.image_with_contours_checkbox, flag=wx.ALIGN_CENTER_HORIZONTAL)

            image_parameters_column_4_box = wx.StaticBox(self)
            image_parameters_column_4_sizer = wx.StaticBoxSizer(image_parameters_column_4_box, wx.VERTICAL)

            image_parameters_column_4_sizer.Add(self.image_decorate_checkbox, flag=wx.ALIGN_CENTER_HORIZONTAL)

            image_parameter_options.Add(image_parameters_column_1_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            image_parameter_options.Add(5, 0)
            image_parameter_options.Add(image_parameters_column_2_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            image_parameter_options.Add(5, 0)
            image_parameter_options.Add(image_parameters_column_3_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            image_parameter_options.Add(5, 0)
            image_parameter_options.Add(image_parameters_column_4_sizer, flag=wx.ALIGN_CENTER_VERTICAL)

            # Place the image setup items into the box.
            image_parameter_options_vertical.Add(get_image_options_vertical, flag=wx.EXPAND)
            image_parameter_options_vertical.Add(0, 5)
            image_parameter_options_vertical.Add(image_parameter_options, flag=wx.EXPAND)

            # Place the "Process image" button
            process_image_button_sizer = wx.BoxSizer(wx.HORIZONTAL)

            image_parameter_options_vertical.Add(0, 5)
            process_image_button_sizer.Add(10, 0)
            process_image_button_sizer.Add(process_image_button, wx.LEFT)
            image_parameter_options_vertical.Add(0, 5)
            image_parameter_options_vertical.Add(process_image_button_sizer, wx.ALIGN_CENTER)

            # Add the image options to the vertical window container
            overall_window_vertical.Add(image_parameter_options_vertical, flag=wx.EXPAND)

        if self.mode == 'TEST' or self.mode == 'SAMPLES':
            overall_window_vertical.Add(0, 15)
            overall_window_vertical.Add(step_3)

            video_parameter_sizer_vertical = wx.StaticBox(self)
            video_parameter_options_vertical = wx.StaticBoxSizer(video_parameter_sizer_vertical, wx.VERTICAL)

            # Make the Button to get the video
            get_video_path_sizer_vertical = wx.StaticBox(self)
            get_video_path_options_vertical = wx.StaticBoxSizer(get_video_path_sizer_vertical, wx.VERTICAL)
            get_video_path_options = wx.BoxSizer(wx.HORIZONTAL)

            get_video_path_options.Add(get_video_path_button)
            get_video_path_options.Add(10, 0)
            get_video_path_options.Add(self.get_video_path_label, wx.EXPAND)
            get_video_path_options_vertical.Add(0, 5)
            get_video_path_options_vertical.Add(get_video_path_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

            video_parameter_options_vertical.Add(get_video_path_options_vertical, flag=wx.EXPAND)

            # Make the Button to get the video
            get_video_output_directory_sizer_vertical = wx.StaticBox(self)
            get_video_output_directory_options_vertical = wx.StaticBoxSizer(get_video_output_directory_sizer_vertical,
                                                                            wx.VERTICAL)
            get_video_output_directory_options = wx.BoxSizer(wx.HORIZONTAL)

            get_video_output_directory_options.Add(get_video_output_directory_button)
            get_video_output_directory_options.Add(10, 0)
            get_video_output_directory_options.Add(self.get_video_output_directory_label, wx.EXPAND)
            get_video_output_directory_options_vertical.Add(0, 5)
            get_video_output_directory_options_vertical.Add(get_video_output_directory_options,
                                                            wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

            video_parameter_options_vertical.Add(0, 5)
            video_parameter_options_vertical.Add(get_video_output_directory_options_vertical, flag=wx.EXPAND)

            # Set up the first column of checkboxes
            video_masked_sizer = wx.BoxSizer(wx.VERTICAL)
            video_masked_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            video_masked_sizer.Add(self.video_masked_checkbox, 0,
                                   flag=wx.ALIGN_LEFT)  # Add this checkbox to the column
            video_masked_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            video_masked_sizer.Add(self.video_with_masks_checkbox, 0,
                                   flag=wx.ALIGN_LEFT)  # Add this checkbox to the column

            # Set up the second column of checkboxes
            video_contours_sizer = wx.BoxSizer(wx.VERTICAL)
            video_contours_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            video_contours_sizer.Add(self.video_contours_checkbox, 0,
                                     flag=wx.ALIGN_LEFT)  # Add this checkbox to the column
            video_contours_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            video_contours_sizer.Add(self.video_decorate_checkbox, 0,
                                     flag=wx.ALIGN_LEFT)  # Add this checkbox to the column

            # Set up the third column of checkboxes
            video_save_progress_sizer = wx.BoxSizer(wx.VERTICAL)
            video_save_progress_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            video_save_progress_sizer.Add(self.video_save_progress_videos_checkbox, 0,
                                          flag=wx.ALIGN_LEFT)
            video_save_progress_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            video_save_progress_sizer.Add(self.video_show_progress_checkbox, 0, flag=wx.ALIGN_LEFT)

            # Set up the fourth column of checkboxes
            video_save_tracker_info_sizer = wx.BoxSizer(wx.VERTICAL)
            video_save_tracker_info_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            video_save_tracker_info_sizer.Add(self.video_save_tracker_info_checkbox, 0,
                                              flag=wx.ALIGN_LEFT)  # Add this checkbox to the column
            video_save_tracker_info_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column

            # Set up the fifth column of checkboxes
            video_debug_masks_sizer = wx.BoxSizer(wx.VERTICAL)
            video_debug_masks_sizer.Add(0, 10, 0)  # Add some vertical empty space to the column
            video_debug_masks_sizer.Add(self.video_debug_masks_checkbox, 0, flag=wx.ALIGN_LEFT)

            # Set up the Horizontal space for the checkbox columns
            horizontal_video_parameter_checkboxes = wx.BoxSizer(wx.HORIZONTAL)
            horizontal_video_parameter_checkboxes.Add(10, 0, 0)
            horizontal_video_parameter_checkboxes.Add(video_masked_sizer)
            horizontal_video_parameter_checkboxes.Add(10, 0, 0)
            horizontal_video_parameter_checkboxes.Add(video_contours_sizer)
            horizontal_video_parameter_checkboxes.Add(10, 0, 0)
            horizontal_video_parameter_checkboxes.Add(video_save_progress_sizer)
            horizontal_video_parameter_checkboxes.Add(10, 0, 0)
            horizontal_video_parameter_checkboxes.Add(video_save_tracker_info_sizer)
            horizontal_video_parameter_checkboxes.Add(10, 0, 0)
            horizontal_video_parameter_checkboxes.Add(video_debug_masks_sizer)

            # Add the checkbox columns to the layout
            video_parameter_options_vertical.Add(horizontal_video_parameter_checkboxes, flag=wx.EXPAND)

            # Place the image setup items into the box.
            video_parameter_options_vertical.Add(0, 5)

            # Add the video option widgets in their own row.
            video_parameters_horizontal = wx.BoxSizer(wx.HORIZONTAL)

            video_start_time_box = wx.StaticBox(self)
            video_start_time_sizer = wx.StaticBoxSizer(video_start_time_box, wx.VERTICAL)

            video_start_time_sizer.Add(video_start_time_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

            video_start_time_sizer.Add(self.video_start_time_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

            video_duration_box = wx.StaticBox(self)
            video_duration_sizer = wx.StaticBoxSizer(video_duration_box, wx.VERTICAL)

            video_duration_sizer.Add(video_duration_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

            video_duration_sizer.Add(self.video_duration_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)
            animal_clip_length_box = wx.StaticBox(self)
            animal_clip_length_sizer = wx.StaticBoxSizer(animal_clip_length_box, wx.VERTICAL)

            animal_clip_length_sizer.Add(animal_clip_length_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

            animal_clip_length_sizer.Add(self.animal_clip_length_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

            lost_threshold_box = wx.StaticBox(self)
            lost_threshold_sizer = wx.StaticBoxSizer(lost_threshold_box, wx.VERTICAL)

            lost_threshold_sizer.Add(lost_threshold_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

            lost_threshold_sizer.Add(self.lost_threshold_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

            batch_size_box = wx.StaticBox(self)
            batch_size_sizer = wx.StaticBoxSizer(batch_size_box, wx.VERTICAL)

            batch_size_sizer.Add(batch_size_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

            batch_size_sizer.Add(self.batch_size_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

            video_parameters_horizontal.Add(video_start_time_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            video_parameters_horizontal.Add(5, 0)
            video_parameters_horizontal.Add(video_duration_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            video_parameters_horizontal.Add(5, 0)
            video_parameters_horizontal.Add(animal_clip_length_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            video_parameters_horizontal.Add(5, 0)
            video_parameters_horizontal.Add(lost_threshold_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            video_parameters_horizontal.Add(5, 0)
            video_parameters_horizontal.Add(batch_size_sizer, flag=wx.ALIGN_CENTER_VERTICAL)

            # Place the video setup items into the box.
            video_parameter_options_vertical.Add(0, 5)
            video_parameter_options_vertical.Add(video_parameters_horizontal)
            video_parameter_options_vertical.Add(0, 5)

            #########
            # Add the video option widgets in their own row.
            tracking_parameters_horizontal = wx.BoxSizer(wx.HORIZONTAL)

            tracking_parameters_column_1_box = wx.StaticBox(self)
            tracking_parameters_column_1_sizer = wx.StaticBoxSizer(tracking_parameters_column_1_box, wx.VERTICAL)

            tracking_parameters_column_1_sizer.Add(tracking_iou_text, flag=wx.ALIGN_CENTER_HORIZONTAL)

            tracking_parameters_column_1_sizer.Add(self.tracking_iou_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)
            tracking_parameters_column_2_box = wx.StaticBox(self)
            tracking_parameters_column_2_sizer = wx.StaticBoxSizer(tracking_parameters_column_2_box, wx.VERTICAL)
            tracking_parameters_column_2_sizer.Add(tracking_distance_text, flag=wx.ALIGN_CENTER_HORIZONTAL)
            tracking_parameters_column_2_sizer.Add(self.tracking_distance_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)
            video_parameters_horizontal.Add(5, 0)
            video_parameters_horizontal.Add(tracking_parameters_column_1_sizer, flag=wx.ALIGN_CENTER_VERTICAL)
            video_parameters_horizontal.Add(5, 0)
            video_parameters_horizontal.Add(tracking_parameters_column_2_sizer, flag=wx.ALIGN_CENTER_VERTICAL)

            # Remove unnecessary user inputs
            if self.mode == 'SAMPLES':
                self.lost_threshold_widget.Hide()
                lost_threshold_text.Hide()
                lost_threshold_box.Hide()

                tracking_parameters_column_1_box.Hide()
                self.tracking_iou_widget.Hide()
                tracking_iou_text.Hide()

                tracking_parameters_column_2_box.Hide()
                self.tracking_distance_widget.Hide()
                tracking_distance_text.Hide()

            # Place the video setup items into the box.
            video_parameter_options_vertical.Add(0, 5)
            video_parameter_options_vertical.Add(tracking_parameters_horizontal)
            video_parameter_options_vertical.Add(0, 5)

            ###################
            # Place the "Process video" button
            process_video_button_sizer = wx.BoxSizer(wx.HORIZONTAL)

            video_parameter_options_vertical.Add(0, 15)
            process_video_button_sizer.Add(10, 0)
            process_video_button_sizer.Add(self.process_video_button, wx.LEFT)
            video_parameter_options_vertical.Add(0, 5)
            process_video_button_sizer.Add(10, 0)
            video_parameter_options_vertical.Add(process_video_button_sizer, wx.ALIGN_CENTER_HORIZONTAL)

            process_video_button_sizer.Add(stop_processing_video_button, wx.LEFT)
            overall_window_vertical.Add(video_parameter_options_vertical, flag=wx.EXPAND)
            overall_window_vertical.Add(0, 5)

            ###### End of Step 3 GUI
        done_botton_horizontal = wx.BoxSizer(wx.HORIZONTAL)

        done_box = wx.StaticBox(self)
        done_sizer = wx.StaticBoxSizer(done_box, wx.VERTICAL)
        done_sizer.Add(done_button)
        done_botton_horizontal.Add(done_sizer)
        overall_window_vertical.Add(done_botton_horizontal, wx.LEFT)
        overall_window_vertical.Add(0, 5)
        overall_window_horizontal.Add(15, 0)
        overall_window_horizontal.Add(overall_window_vertical, wx.EXPAND)
        overall_window_horizontal.Add(15, 0)
        self.SetSizer(overall_window_horizontal)

    def evt_stop_video(self, event):
        self.animals.stop = True

    def evt_set_inference_size(self, event):
        self.inference_size = self.inference_size_widget.GetValue()

    def evt_done(self, event):
        self.Parent.Destroy()

    # Function: evt_get_input_directory
    # Description: basic modal directory dialog box to get the input directory
    def evt_set_detection_threshold(self, event):
        self.detection_threshold = self.detection_threshold_widget.GetValue()

    # Function: evt_get_input_directory
    # Description: basic modal directory dialog box to get the input directory
    def evt_get_model(self, event):
        dlg = wx.DirDialog(None, "Choose model folder containing its ts and txt files", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.model_path = None
            self.thing_names_path = None
            self.model_folder_path = dlg.GetPath()
            filenames = os.listdir(self.model_folder_path)
            for filename in filenames:
                file = os.path.join(self.model_folder_path, filename)
                if file.endswith('.ts'):
                    if self.model_path is not None:
                        dlg = wx.GenericMessageDialog(None, 'Multiple ts files were found!', caption='Error',
                                                      style=wx.OK | wx.CENTER)
                        dlg.ShowModal()
                        self.model_path = None
                        self.thing_names_path = None
                        return
                    self.model_path = file
                    self.get_model_label.SetValue(os.path.basename(self.model_path)[:-3])
                if file.endswith('.txt'):
                    if self.thing_names_path is not None:
                        dlg = wx.GenericMessageDialog(None, 'Multiple txt files were found!', caption='Error',
                                                      style=wx.OK | wx.CENTER)
                        dlg.ShowModal()
                        self.model_path = None
                        self.thing_names_path = None
                        return
                    self.thing_names_path = file
        if self.model_path is None:
            dlg = wx.GenericMessageDialog(None, 'No ts file was found!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            self.model_path = None
            self.thing_names_path = None
            return
        if self.thing_names_path is None:
            dlg = wx.GenericMessageDialog(None, 'No txt file was found!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            self.model_path = None
            self.thing_names_path = None
            return
        else:
            with open(self.thing_names_path) as f:
                imported_data = f.read()

                # reconstruct the data as a dictionary in index, name pairs {'0': 'rat', '1': 'larva', etc)
                class_storage = json.loads(imported_data)
                if 'training_size' in class_storage:
                    model_things = class_storage['animal_mapping']
                    self.inference_size_widget.SetValue(int(class_storage['training_size']))
                else:
                    model_things = class_storage
                    self.inference_size_widget.Enable()
                    self.inference_size_text.Enable()
                    self.inference_size_widget.SetValue(256)
                    self.inference_size = 256
                self.detection_threshold_widget.Enable()
                self.detection_threshold_text.Enable()
            # Invert the references so we can easily reverse lookup
            val_list = list(model_things.values())
            self.display_things = val_list  # []
            # self.display_things.append(str(val_list[0]))

            self.animals = self.display_things  # []
            # self.animals.append(self.display_things[0])
            self.enable_mps_checkbox.Enable()

        dlg.Destroy()

    # Function: evt_get_input_directory
    # Description: basic modal directory dialog box to get the input directory
    def evt_get_video_path(self, event):
        """
        Create and show the Open FileDialog
        """
        wildcard = "Videos (*.mp4, *.mov, *.avi)|*.mp4;*.mov;*.avi"
        dlg = wx.FileDialog(
            self, message="Choose a Video",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.video_path = path
            self.get_video_path_label.SetValue(os.path.basename(path))
        dlg.Destroy()

    # Function: evt_get_input_directory
    # Description: basic modal directory dialog box to get the input directory
    def evt_get_image(self, event):
        wildcard = "Image Files (*.png, *.jpg, *.jpeg, *.tiff)|*.png;*.jpg;*.jpeg;*.tiff"
        dlg = wx.FileDialog(
            self, message="Choose an Image",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.image_path = path
            self.get_image_label.SetValue(os.path.basename(path))
        dlg.Destroy()

    # Function: evt_get_input_directory
    # Description: basic modal directory dialog box to get the input directory
    def evt_process_image(self, event):

        if not self.model_path:
            dlg = wx.GenericMessageDialog(None, 'No model has been selected!', caption='Error', style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.thing_names_path:
            dlg = wx.GenericMessageDialog(None, 'No animal mapping file has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.image_path:
            dlg = wx.GenericMessageDialog(None, 'No image has been selected!', caption='Error', style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        animals = dynamic_background_remover(model_path=self.model_path,
                                             detection_threshold=self.detection_threshold_widget.GetValue(),
                                             thing_names_path=self.thing_names_path,
                                             enable_mps=self.enable_mps_checkbox.GetValue())

        # test sending in the images and view classes directly
        animals.inference_size = int(self.inference_size_widget.GetValue())
        if self.image_masked_checkbox.GetValue():
            cv2.imshow('masked animals! (press space bar to continue)',
                       animals.get_masked_image(image_path=self.image_path,
                                                view_classes=self.display_things,
                                                draw_text=self.image_decorate_checkbox.GetValue(), ))
            cv2.waitKey(0)
            cv2.destroyWindow('masked animals! (press space bar to continue)')

        if self.image_with_masks_checkbox.GetValue():
            cv2.imshow('animals with masks! (press space bar to continue)',
                       animals.get_original_with_masks(image_path=self.image_path,
                                                       view_classes=self.display_things,
                                                       draw_bbox=self.image_decorate_checkbox.GetValue(),
                                                       draw_text=self.image_decorate_checkbox.GetValue(),
                                                       draw_masks=True))
            cv2.waitKey(0)
            cv2.destroyWindow('animals with masks! (press space bar to continue)')

        if self.image_with_contours_checkbox.GetValue():
            cv2.imshow('animals with masks! (press space bar to continue)',
                       animals.get_original_with_contours(image_path=self.image_path,
                                                          view_classes=self.display_things,
                                                          draw_bbox=self.image_decorate_checkbox.GetValue(),
                                                          draw_text=self.image_decorate_checkbox.GetValue(),
                                                          draw_contour=True))
            cv2.waitKey(0)
            cv2.destroyWindow('animals with masks! (press space bar to continue)')

    # Function: evt_get_video_output_directory
    # Description: basic modal directory dialog box to get the output directory
    def evt_get_video_output_directory(self, event):
        dlg = wx.DirDialog(None, "Choose output directory", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.output_path = dlg.GetPath()
            self.get_video_output_directory_label.LabelText = ": " + self.output_path
        dlg.Destroy()

    def evt_process_video(self, event):
        if not self.model_path:
            dlg = wx.GenericMessageDialog(None, 'No model has been selected!', caption='Error', style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.thing_names_path:
            dlg = wx.GenericMessageDialog(None, 'No animal mapping file has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        if not self.video_path:
            dlg = wx.GenericMessageDialog(None, 'No video has been selected!', caption='Error', style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return

        thread = threading.Thread(target=self.get_tracked_video)
        thread.run()

    def get_tracked_video(self):
        self.animals = dynamic_background_remover(model_path=self.model_path,
                                                  detection_threshold=self.detection_threshold_widget.GetValue(),
                                                  thing_names_path=self.thing_names_path,
                                                  enable_mps=self.enable_mps_checkbox.GetValue())
        self.animals.inference_size = int(self.inference_size_widget.GetValue())
        self.percent_video_complete = self.animals.percent_complete
        self.animals.get_tracked_video(input_video_name=self.video_path,
                                       video_output_path=self.output_path,
                                       view_classes=self.display_things,
                                       original_with_contours=self.video_contours_checkbox.GetValue(),
                                       decorate_videos=self.video_decorate_checkbox.GetValue(),
                                       batch_size=self.batch_size_widget.GetValue(),
                                       tracking_iou=self.tracking_iou_widget.GetValue(),
                                       progress=self.video_show_progress_checkbox.GetValue(),
                                       save_progress_videos=self.video_save_progress_videos_checkbox.GetValue(),
                                       original_with_masks=self.video_with_masks_checkbox.GetValue(),
                                       masked_original=self.video_masked_checkbox.GetValue(),
                                       save_tracker_info=self.video_save_tracker_info_checkbox.GetValue(),
                                       mini_clip_length=self.animal_clip_length_widget.GetValue(),
                                       lost_threshold=self.lost_threshold_widget.GetValue(),
                                       max_distance_moved=self.tracking_distance_widget.GetValue(),
                                       debug_masks=self.video_debug_masks_checkbox.GetValue(),
                                       video_start=self.video_start_time_widget.GetValue(),
                                       duration=self.video_duration_widget.GetValue()
                                       )


# Run the program
if __name__ == '__main__':
    app = wx.App()
    DynamicBackgroundInitialWindow("Baker's Dynamic Background Removal Interface", mode='TEST')
    # DynamicBackgroundInitialWindow("Baker's Dynamic Background Removal Interface", mode='SAMPLES')
    # DynamicBackgroundInitialWindow("Baker's Dynamic Background Removal Interface", mode='')
    app.MainLoop()
