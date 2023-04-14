import copy
import os, cv2, wx, pickle, math, threading
import numpy as np


# Class: SortingHat_InitialWindow
# Description: This class was taken from LabGym and then modified for User input for sortinghat
# Its goal is to get the input and output paths as well as the category name <-> key mappings
class DynamicBackgroundTrackletPlayerInitialWindow(wx.Frame):

    def __init__(self, title):
        # if want to adjust the size, add arg 'size=(x,y)' but this size seems fine
        wx.Frame.__init__(self, parent=None, title=title)
        self.panel = MyPanel(self)
        self.frame_sizer = wx.BoxSizer(wx.VERTICAL)
        self.frame_sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(self.frame_sizer)
        self.Size = (self.panel.BestVirtualSize[0] + 20, self.panel.BestVirtualSize[1] + 30)
        self.Move(wx.Point(50, 50))
        self.Show()


class MyPanel(wx.ScrolledWindow):

    def __init__(self, parent):
        wx.ScrolledWindow.__init__(self, parent, id=-1, pos=wx.DefaultPosition, size=wx.DefaultSize,
                                   style=wx.HSCROLL | wx.VSCROLL,
                                   name="scrolledWindow")
        self.SetScrollbars(1, 1, 600, 400)
        # Set up the variables that we want to capture

        self.refinement_path = None
        self.header_center = None
        self.header_font_width = None
        self.header_font_scale = None
        self.tracklet_stop = False
        self.video_path = None
        self.animal_tracking_dictionary = None  # self._load_pickle(file_path)
        self.pickle_path = None  # file_path

        # Set up the container (BoxSizer) for the overall display window. Within this window, we will
        # place additional containers for sets of input and capabilities.
        overall_window_vertical = wx.BoxSizer(wx.VERTICAL)
        overall_window_horizontal = wx.BoxSizer(wx.HORIZONTAL)

        ##### Start of Step 1 GUI

        # Create the tex that says "Step 1...." and add it to the vertical window container
        overall_window_vertical.Add(0, 15)
        step_1 = wx.StaticText(self, label='Pick Your File(s)')
        overall_window_vertical.Add(step_1)

        # Make the Button to get the tracklet file

        get_tracklet_sizer_vertical = wx.StaticBox(self)
        get_tracklet_vertical = wx.StaticBoxSizer(get_tracklet_sizer_vertical, wx.VERTICAL)
        get_tracklet_options = wx.BoxSizer(wx.HORIZONTAL)

        # add the button to get the input directory and bind its event function
        get_tracklet_button = wx.Button(self, label='Select A Tracklet File')
        get_tracklet_button.SetToolTip("Select the *.pkl file holding your animal tracks")
        get_tracklet_button.Bind(wx.EVT_BUTTON, self.evt_get_tracklet)
        self.get_tracklet_label = wx.TextCtrl(self, value='', style=wx.TE_LEFT, size=(300, -1))
        self.get_tracklet_label.SetHint('{your tracklet file}')
        get_tracklet_options.Add(get_tracklet_button, 0, flag=wx.ALIGN_CENTER)
        get_tracklet_options.Add(10, 0)
        get_tracklet_options.Add(self.get_tracklet_label, 0, flag=wx.ALIGN_CENTER)
        get_tracklet_vertical.Add(0, 5)
        get_tracklet_vertical.Add(get_tracklet_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

        overall_window_vertical.Add(get_tracklet_vertical, flag=wx.EXPAND)
        overall_window_vertical.Add(10, 0)

        # Make the Button to get the video file
        get_video_sizer_vertical = wx.StaticBox(self)
        get_video_vertical = wx.StaticBoxSizer(get_video_sizer_vertical, wx.VERTICAL)
        get_video_options = wx.BoxSizer(wx.HORIZONTAL)

        # add the button to get the input directory and bind its event function
        get_video_button = wx.Button(self, label='Select A Video File')
        get_video_button.SetToolTip('Select the video that matches your animal tracks')
        get_video_button.Bind(wx.EVT_BUTTON, self.evt_get_video_path)
        self.get_video_label = wx.TextCtrl(self, value='', style=wx.TE_LEFT, size=(300, -1))
        self.get_video_label.SetHint('{optional: your video}')
        get_video_options.Add(get_video_button, 0, flag=wx.ALIGN_CENTER)
        get_video_options.Add(10, 0)
        get_video_options.Add(self.get_video_label, 0, flag=wx.ALIGN_CENTER)
        get_video_vertical.Add(0, 5)
        get_video_vertical.Add(get_video_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

        overall_window_vertical.Add(get_video_vertical, flag=wx.EXPAND)
        overall_window_vertical.Add(10, 0)

        # Make the Button to get the video file
        refinement_sizer_vertical = wx.StaticBox(self)
        refinement_vertical = wx.StaticBoxSizer(refinement_sizer_vertical, wx.VERTICAL)
        refinement_options = wx.BoxSizer(wx.HORIZONTAL)

        # add the button to get the input directory and bind its event function
        self.make_refinement_button = wx.CheckBox(self, label='Make Refinement Files')
        self.make_refinement_button.SetToolTip('Check if you would like to export images to help refine model training')
        refinement_options.Add(self.make_refinement_button, 0, flag=wx.ALIGN_CENTER)
        refinement_options.Add(10, 0)

        self.animal_number_widget = wx.SpinCtrl(self, initial=-1, max=100, min=-1)
        self.animal_number_widget.SetToolTip('How many animals are in the scene')

        refinement_options.Add(self.animal_number_widget, flag=wx.ALIGN_CENTER)
        animal_number_text = wx.StaticText(self, label='Number of Animals')
        animal_number_text.SetToolTip('How many animals are in the scene')
        refinement_options.Add(animal_number_text, flag=wx.ALIGN_CENTER)

        refinement_options.Add(10, 0)
        self.max_speed_button = wx.CheckBox(self, label='Max Speed')
        self.max_speed_button.SetToolTip('Process as fast as possible ignoring framerate')

        refinement_options.Add(self.max_speed_button, flag=wx.ALIGN_CENTER)

        refinement_vertical.Add(0, 5)
        refinement_vertical.Add(refinement_options, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

        refinement_vertical.Add(0, 5)

        refinement_output = wx.BoxSizer(wx.HORIZONTAL)
        refinement_output_button = wx.Button(self, label='Output Path')
        refinement_output_button.SetToolTip('Location to store refinement images')
        refinement_output_button.Bind(wx.EVT_BUTTON, self.evt_get_refinement_directory)
        self.refinement_output_label = wx.TextCtrl(self, value='',
                                                   style=wx.TE_LEFT, size=(300, -1))
        self.refinement_output_label.SetHint('{refinement directory}')
        refinement_output.Add(refinement_output_button, flag=wx.ALIGN_CENTER)

        refinement_output.Add(10, 0)
        refinement_output.Add(self.refinement_output_label, flag=wx.ALIGN_CENTER)

        refinement_vertical.Add(0, 5)
        refinement_vertical.Add(refinement_output, wx.ALIGN_CENTER_VERTICAL, wx.EXPAND)

        overall_window_vertical.Add(refinement_vertical, flag=wx.EXPAND)
        overall_window_vertical.Add(10, 0)

        start_stop_tracklet_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        start_stop_tracklet_box = wx.StaticBox(self)
        start_stop_tracklet_options_vertical_sizer = wx.StaticBoxSizer(start_stop_tracklet_box, wx.VERTICAL)


        #play_start_stop_options_horizontal= wx.BoxSizer(wx.HORIZONTAL)
        video_start_frame_box = wx.StaticBox(self)
        video_start_frame_sizer = wx.StaticBoxSizer(video_start_frame_box, wx.VERTICAL)

        video_start_frame_text = wx.StaticText(self, label='Start Frame', style=wx.ALIGN_CENTER_HORIZONTAL)
        video_start_frame_sizer.Add(video_start_frame_text, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.video_start_frame_widget = wx.SpinCtrl(self, initial=0, min=0, max=10000000)
        video_start_frame_sizer.Add(self.video_start_frame_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

        start_stop_tracklet_horizontal.Add(video_start_frame_sizer)

        video_stop_frame_box = wx.StaticBox(self)
        video_stop_frame_sizer = wx.StaticBoxSizer(video_stop_frame_box, wx.VERTICAL)

        video_stop_frame_text = wx.StaticText(self, label='Duration in Frames', style=wx.ALIGN_CENTER_HORIZONTAL)
        video_stop_frame_sizer.Add(video_stop_frame_text, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.video_stop_frame_widget = wx.SpinCtrl(self, initial=-1, min=-1, max=10000000)
        video_stop_frame_sizer.Add(self.video_stop_frame_widget, flag=wx.ALIGN_CENTER_HORIZONTAL)

        start_stop_tracklet_horizontal.Add(video_stop_frame_sizer)

        overall_window_vertical.Add(start_stop_tracklet_horizontal, flag=wx.EXPAND)




        play_tracklet_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        play_tracklet_box = wx.StaticBox(self)
        play_tracklet_options_vertical_sizer = wx.StaticBoxSizer(play_tracklet_box, wx.VERTICAL)

        ###################
        # Place the "Play tracklet" button

        self.play_tracklet_button = wx.Button(self, label='Play')
        self.play_tracklet_button.Bind(wx.EVT_BUTTON, self.evt_process_tracklet)
        self.step_button = wx.CheckBox(self, label='Step')
        self.step_button.SetToolTip('Step through the track file one frame at a time. Space bar advances to next frame')
        play_tracklet_parts_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        play_tracklet_parts_horizontal.Add(self.play_tracklet_button, wx.ALIGN_CENTER)
        play_tracklet_parts_horizontal.Add(10, 0)
        play_tracklet_parts_horizontal.Add(self.step_button, wx.ALIGN_CENTER)
        play_tracklet_options_vertical_sizer.Add(play_tracklet_parts_horizontal, wx.LEFT)
        play_tracklet_horizontal.Add(play_tracklet_options_vertical_sizer, wx.ALIGN_CENTER_HORIZONTAL)

        stop_tracklet_box = wx.StaticBox(self)
        stop_tracklet_options_vertical_sizer = wx.StaticBoxSizer(stop_tracklet_box, wx.VERTICAL)
        self.stop_tracklet_button = wx.Button(self, label='Stop')
        self.stop_tracklet_button.Bind(wx.EVT_BUTTON, self.evt_stop_tracklet)
        stop_tracklet_options_vertical_sizer.Add(self.stop_tracklet_button, wx.LEFT)
        play_tracklet_horizontal.Add(stop_tracklet_options_vertical_sizer, wx.ALIGN_CENTER_HORIZONTAL)

        overall_window_vertical.Add(play_tracklet_horizontal, flag=wx.EXPAND)
        overall_window_vertical.Add(0, 15)

        overall_window_horizontal.Add(15, 0)
        overall_window_horizontal.Add(overall_window_vertical, wx.EXPAND)
        overall_window_horizontal.Add(15, 0)
        self.SetSizer(overall_window_horizontal)

    def evt_stop_tracklet(self, event):
        self.tracklet_stop = True

    # Function: evt_get_input_directory
    # Description: basic modal directory dialog box to get the input directory
    def evt_get_tracklet(self, event):
        wildcard = "Track Files (*.pkl)|*.pkl"
        dlg = wx.FileDialog(
            self, message="Choose a Track File",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.pickle_path = path
            self.get_tracklet_label.SetValue(os.path.basename(path))
            self.animal_tracking_dictionary = self._load_pickle(self.pickle_path)
        dlg.Destroy()

    # Function: evt_get_input_directory
    # Description: basic modal directory dialog box to get the input directory
    def evt_get_video_path(self, event):
        if not self.pickle_path:
            dlg = wx.GenericMessageDialog(None, 'First select your trackl file!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        wildcard = "Videos (*.mp4, *.mov, *.avi)|*.mp4;*.mov;*.avi"
        dlg = wx.FileDialog(
            self, message="Choose a Video",
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            if os.path.basename(path) != self.animal_tracking_dictionary['video_info']['filename']:
                dlg = wx.GenericMessageDialog(None, "Video file name does not match track file!", caption='Error',
                                              style=wx.OK | wx.CENTER)
                dlg.ShowModal()
                return
            self.video_path = path
            self.get_video_label.SetValue(os.path.basename(path))
        dlg.Destroy()

    # Function: evt_get_video_output_directory
    # Description: basic modal directory dialog box to get the output directory
    def evt_get_refinement_directory(self, event):
        dlg = wx.DirDialog(None, "Choose output directory", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.refinement_path = dlg.GetPath()
            self.refinement_output_label.LabelText = self.refinement_path
        dlg.Destroy()

    def evt_process_tracklet(self, event):
        if not self.pickle_path:
            dlg = wx.GenericMessageDialog(None, 'No track file has been selected!', caption='Error',
                                          style=wx.OK | wx.CENTER)
            dlg.ShowModal()
            return
        thread = threading.Thread(target=self.play_tracklet)
        thread.run()

    def play_tracklet(self):
        if self.make_refinement_button.IsChecked():
            if self.video_path is None:
                dlg = wx.GenericMessageDialog(None, 'No video file has been selected!', caption='Error',
                                              style=wx.OK | wx.CENTER)
                dlg.ShowModal()
                return
            elif self.refinement_path is None:
                dlg = wx.GenericMessageDialog(None, 'No refinement path has been selected!', caption='Error',
                                              style=wx.OK | wx.CENTER)
                dlg.ShowModal()
                return
            elif self.animal_number_widget.GetValue() < 1:
                dlg = wx.GenericMessageDialog(None, 'Please select a number of visible animals!', caption='Error',
                                              style=wx.OK | wx.CENTER)
                dlg.ShowModal()
                return
        original_file_name = self.animal_tracking_dictionary['video_info']['filename']
        path_base = os.path.dirname(self.pickle_path)
        video_filename = self.video_path
        exists = False if self.video_path is None else True
        frame_rate = self.animal_tracking_dictionary['video_info']['fps']
        true_frame_rate = frame_rate
        wait_time = int(1000 / true_frame_rate)
        width = self.animal_tracking_dictionary['video_info']['width']
        height = self.animal_tracking_dictionary['video_info']['height']
        font_scale, font_width = self.get_font_size(image_width=width, image_height=height)
        self.header_font_scale = font_scale * .25
        self.header_font_width = int(font_width * .4)
        self.header_center = width / 2 - (width * .15)
        if self.video_stop_frame_widget.GetValue() <= 0:
            requested_length = np.inf
        else:
            requested_length = self.video_stop_frame_widget.GetValue()
        max_frame_number = self.find_max_frame_number()
        if max_frame_number > (requested_length + self.video_start_frame_widget.GetValue()-1):
            max_frame_number = requested_length + self.video_start_frame_widget.GetValue() -1
        if exists:
            cap = cv2.VideoCapture(self.video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.video_start_frame_widget.GetValue()))
        else:
            num_frames = max_frame_number
        frame_number = int(self.video_start_frame_widget.GetValue())
        while frame_number <= num_frames and frame_number <= max_frame_number and not self.tracklet_stop:
            if exists:
                _, frame = cap.read()
                if frame is None:
                    cap.release()
                    break
                display_frame = self.draw_frame(frame_number, frame=frame)
            else:
                display_frame = self.draw_frame(frame_number)
            if not self.max_speed_button.IsChecked():
                cv2.imshow(f'Simulating {original_file_name} analysis', display_frame)
                if self.step_button.IsChecked():
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(wait_time)
            frame_number += (1)
        if not self.max_speed_button.IsChecked():
            if not self.tracklet_stop: cv2.waitKey(0)
            cv2.destroyWindow(f'Simulating {original_file_name} analysis')
        self.tracklet_stop = False
        return None

    def draw_frame(self, frame_number, frame=None):
        decorate_video = True
        if frame is not None:
            frame_temp = frame
        else:
            frame_temp = np.zeros((self.animal_tracking_dictionary['video_info']['height'],
                                   self.animal_tracking_dictionary['video_info']['width'], 3), np.uint8)
        if self.make_refinement_button.IsChecked():
            original_frame = copy.deepcopy((frame))
        number_of_animals = 0
        for animal in self.animal_tracking_dictionary['animals']:
            if frame_number in self.animal_tracking_dictionary['animals'][animal]['frame_number']:
                number_of_animals += 1
                frame_index = self.animal_tracking_dictionary['animals'][animal]['frame_number'].index(frame_number)
                color = self.animal_tracking_dictionary['animals'][animal]['color']
                # swap to BGR from RGB
                color = (color[2], color[1], color[0])
                cv2.drawContours(image=frame_temp,
                                 contours=self.animal_tracking_dictionary['animals'][animal]['contours'][frame_index],
                                 contourIdx=-1,
                                 color=color,
                                 thickness=1,
                                 offset=(
                                     int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][0]),
                                     int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1]))
                                 )
                if decorate_video:
                    """cv2.rectangle(img=frame_temp,
                                  color=color,
                                  thickness=2,
                                  pt1=(
                                      int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][0]),
                                      int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1])),
                                  pt2=(
                                      int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][2]),
                                      int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][3])),
                                  )"""
                    x_org = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][0])
                    y_org = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1]) - 5 if \
                        int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1]) - 5 > 0 else 0
                    box_width = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][2] - \
                                    self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][0])
                    box_height = int(self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][3] - \
                                     self.animal_tracking_dictionary['animals'][animal]['boxes'][frame_index][1])
                    score = round(
                        float(self.animal_tracking_dictionary["animals"][animal]["scores"][frame_index]) * 100, 2)
                    text = f'Animal {self.animal_tracking_dictionary["animals"][animal]["animal_id"]}: ' \
                           f'{score}'
                    animal_font_scale, animal_font_width = self.get_font_size(image_width=int(box_width),
                                                                              image_height=int(box_height))
                    cv2.putText(img=frame_temp,
                                text=text,
                                org=(x_org, y_org),
                                color=color,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=animal_font_scale,
                                thickness=animal_font_width,
                                )

        text = f'Frame Number: {frame_number}'
        cv2.putText(img=frame_temp,
                    text=text,
                    org=(int(self.header_center), int(30 * self.header_font_scale)),
                    color=(0, 0, 255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=self.header_font_scale,
                    thickness=self.header_font_width,
                    )
        if self.make_refinement_button.IsChecked():

            if number_of_animals != int(self.animal_number_widget.GetValue()):
                filename_original = os.path.join(self.refinement_path,
                                                 f'{os.path.basename(self.video_path)[:-4]}_'
                                                 f'{frame_number}_original_frame_.jpg')
                filename_outlined = os.path.join(self.refinement_path,
                                                 f'{os.path.basename(self.video_path)[:-4]}_'
                                                 f'{frame_number}_outlined_frame_.jpg')
                cv2.imwrite(filename_original, original_frame)
                cv2.imwrite(filename_outlined, frame_temp)
        return frame_temp

    def _load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            animal_data = pickle.load(f)
        return animal_data

    def find_max_frame_number(self):
        # use the self.animal_tracking_dictionary to find that maximum frame number to know how long to loop
        max_frame_count = 0
        for animal in self.animal_tracking_dictionary['animals']:
            local_max = max(self.animal_tracking_dictionary['animals'][animal]['frame_number'])
            max_frame_count = local_max if local_max > max_frame_count else max_frame_count
        return max_frame_count

    def get_font_size(self, image_width, image_height, font_scale=2e-3, thickness_scale=5e-3):
        font_scale = (image_height + image_width) * font_scale
        font_scale = font_scale if font_scale > 0.5 else 0.5
        thickness = int(math.ceil(min(image_height, image_width) * thickness_scale))
        return font_scale, thickness


# Run the program
if __name__ == '__main__':
    app = wx.App()
    DynamicBackgroundTrackletPlayerInitialWindow('Playback Animal Track File')
    app.MainLoop()
