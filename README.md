# Welcome to my dynamic background removal GitHub. 
 
I am an undergraduate student working on a research project at my university. As part of my responsibilities, I was asked to find a method for removing a 'dynamic' background in a set of videos that include animals, leaving behind only the animals. After searching around for solutions, including how to make the [matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN) code work natively on my Mac M2, I stumbled upon a [youtube video](https://www.youtube.com/watch?v=9a_Z14M-msc) explaining how to work with Meta's detectron2. Detectron2 was immediately appealing and made me realize that the problem statement I was working with needed to be adjusted. I mentally changed my assignment to "given a video with animals in it, return only the animals without any other background content." This may sound the same as my original assignment, but to me, the subtle difference in thought process was significant to me. So I decided to train and use detectron2 as the base for identifying the animals in my videos, creating instance masks, extracting the animals from videos based on these masks, pasting them onto a blank canvas, and then ultimately saving the newly constructed canvas as a video frame in a new video. 


To make this notebook and learn about Machine Vision models, I studied many sources scattered around the internet. Therefore, let me know if you see something that looks like your code without acknowledgment, and I am happy to include a reference.


I have included the lessons and methods I learned during this process, even if they weren't strictly related to my original goal. I spent **A LOT** of time researching solutions to what seemed like simple problems with detectron2 and Colab, which ultimately required rather complex solutions. I sincerely hope this notebook helps someone else save time in the future. Let me know if you find something I am missing, and I'll include it as I have time.


Within this repository, I have the following files:

1. A Jupyter Notebook (Baker_Training_Exporting_Mask_R_CNN.ipynb) that trains Mask R-CNN model, evaluates, and then exports it to torchscript from within Google Colab.
2. Installation Instructions (StandAlone-Install-Instructions.docx) for installing my standalone software framework.
3. A User Interface written in wxpython for training Mask R-CNN models (Track_player_gui.py)
4. A User Interface written in wxpython for using these models to remove background from video and create behavior animations/contour images compatible with Ye Labs LabGyn project. (dynamic_background_gui.py)
5. Two support libraries I wrote to enable the above capabilities. The first (dynamic_background_remover.py) handles all of the major code for inferencing and image processing. The second (animal_tracker.py) is an animal tracking algorithm I wrote which uses a combination of IoU tracking layered on top of a distance between centers tracking method.
6. A User Interface written in wxpython for playing back animal track files saved into pkl files during analysis (Track_player_gui.py). This UI is able to play back the highly portable pkl file that holds the outline of each animal and its path without the need for the original video.




Problems I had to solve:
1. Hacking together a static version of the core training/inference code was relatively easy. Making it generic so that I/you can make zero/minimal code changes and train 1 to N classes, took a lot of effort. This was the main reason I have this notebook's Global Variables at the start of code sections. Make sure you understand these variables and their usage before modifying the code in the notebook.
2. I only found 1 sample of code explaining how to take an instance mask and use it to get only that object back out of an image. I ended up creating a new framework for use with my research. More on that later.
3. I did not find any working examples of subclassing the default trainer to add the augmentations. One example was close to working, but its info needed to be completed. My functional example is shown below in the MyTrainer class. In addition, I have added the augmentation list to the Global Variables so that it can be easily modified for your purposes.
4. Auto-saving the newly created model out of colab. This may sound minor, but it is vital when trying to conserve compute credits. I save a time/date marked version of the model and model configuration to my Google drive. At the end of the code, there is an auto-terminate function to delete the runtime and stop consuming credits. You should be abler to adjust the Global Variables while offline and then connect the runtime, hit run-all, and walk away knowing that it won't waste your compute credits. You can change one of the Global Variables if you and stay connected after training if that is your desire.
5. I could not find any *simple* method to display only the thing classes I wanted. The objective is that regardless of what was detected, only mask and highlight certain classes of things. All of the examples I read showed filtering to a single classes or not at all, but I implemented a method to filter N classes. This is done by recursively removing all detected things that are not in the un-ordered "my_display_things" list. Three lines of code accomplish what seemed very complex initially. However, as this notebook evolved, I have moved that specific functionality over to my framework in GitHub. 
6. I did not locate any complete examples of preparing a dataset from start to finish. The data preparation section of this doc may seem excessively detailed, but it comes from frustration at not finding a good tutorial.
7. An ***EASY*** way to export a pytorch model to torchscript from detectron2. Every tutorial I found made it overly complicated. This is a tiny bit of customization in code borrowed from detectron2, it's very simple.
8. A way to 'batch' load images into an exported torchscript model. This is really subtle in this notebook but the export_scripting method that I use is slightly modified from the code in detectron2. I have changed Tuple to List in the forward function.  I never figured out how to build Tuple[Dict[str, torch.Tensor]] so I changed it to List[Dict[str, torch.Tensor]]. Like magic, it worked. If anyone reading this understands how to make the original structure and load multiple images/frames simultaneously please let me know.  
9. I will add more items here as I remember additional challenges I encountered and solved. 


## About the Jupiter Notebook

Within my notebook I do the following:


1.  Explain how to get your dataset ready to train your model.
2.  Give the code required to train your model, including the flexibility to train an individual thing, or many things, automagically with a simple change to configuration parameters.
3.  Create inference output with images using your new model.
4.  Export the model to torchscript format for portability.


Happy training!

<font size = '2' color='red'>*This was the actual goal of my work when I started.</font>


### FYI: A copy of the dataset, annotations, a video input, etc. are available [here](https://drive.google.com/drive/folders/1XPPQ7phosdoSiQS7dVN9heVVWOrPkFx5?usp=sharing).




## About the Standalone Software Framework
For now, I will not talk much about the standalone framework.  I am in the process of creating some additional documentation for the UIs, but for now, run them and experiment with them. Almost every button and text entry has a tip for its usage if you hover over it. 

I suggest starting without installing the trainer. That installation is a lot more involved. Begin with dynamic_background_gui.py using the models in my google drive and the larva avi file I provide. Once it successfully processes a few seconds of a video to completion, run the Track_payer_gui.py to get a sense of how the video is being processed. Only then, install all the software and libraries required to train your model. I still prefer to train my models in Google Colab and then I perform the background removal on my local machine. However, with the software I have provided, the choice is up to you. 


### If you have any feedback on this software, please reach out here or through my LinkedIn. 



 
