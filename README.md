# Welcome to my background removal with detectron2 notebook
 
 I am an undergraduate student working on a research project at my university. As part of my responsibilities, I was asked to find a method for removing a 'dynamic' background in a set of videos which include animals, leaving behind only the animals. After searching around for solutions, including how make the [matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN) code work natively on my Mac M1, I stumbled upon a [youtube video](https://www.youtube.com/watch?v=9a_Z14M-msc) explaining how to work with detectron2. Detectron2 was immediately appealing and made me realize that the problem statement I was working with needed to be adjusted. I mentally changed my assignment to "given a video with animals in it, return only the animals without any other background content." This may sound the same as my original assignment, but to me, the subtle difference in thought process was significant. So I decided to train and use detectron2 as the base for identifying the animals in my videos, creating instance masks, extracting the animals from videos based on these masks, pasting them onto a blank canvas, and then ultimately saving the newly constructed canvas as a video frame in a new video.  

This notebook includes information from many sources, which I pulled together quickly, mainly over a Christmas break. Therefore, let me know if you see some of your code here without acknowledgment as I am happy to include a reference. 

I have included the lessons and methods I learned during this process, even if they were not strictly related to my original goal. I spent **A LOT** of time researching solutions to what seemed like simple problems with detectron2 and Colab, which ultimately required rather complex solutions. I sincerely hope this notebook helps someone else save time in the future! If you find something I am missing, let me know and I'll include it as I have time.

Within this notebook, I will: 

1.   Explain how to get your dataset ready to train your model. 
2.   Give the code required to train your model, including the flexibility to train an individual thing, or many things, automagically with a simple change to configuration parameters. 
3.   Create inference output with images using your new model.
4.   Create a set of videos to demonstrate the masking capabilities of this notebook.<font color='red'>*</font>

Problems I had to solve:
1.  Hacking together a static version of the core training/inference code was relatively easy. Making it generic so that I/you can make zero/minimal code changes and train 1 to N classes, took a lot of effort. This was the main reason I have this notebook's Global Variables at the start of code sections. Make sure you understand these variables and their usage before modifiying the code in the notebook. 
2. I only found 1 sample of code explaining how to take an instance mask and use it to get only that object back out of an image. I adapted that code and included it within thing_masker.get_masked_image(). More on that later.
3. I did not find any working examples of subclassing the default trainer to add the augmentations. One example was close to working, but its info needed to be completed. My functional example is shown below in the MyTrainer class. In addition, I have added the augmentation list to the Global Variables so that it can be easily modified for your purposes.
4. Auto-saving a newly created model out of colab. This may sound minor, but it is vital when trying to conserve compute credits. I saved a time/date marked version of the model and model configuration to my Google drive. At the end of the code, there is an auto-terminate function to delete the runtime and stop consuming credits. If no errors occur, you can connect the runtime, hit run-all, and walk away knowing that it won't waste your compute credits.
5. I could not find any *simple* method to display only the thing classes I wanted. The objective is that regardless of what was detected, only mask and highlight certain classes of things. All of the examples I read showed filtering to a single classes, but I implemented a method to filter N classes. This is done by recursively removing all detected things that are not in the un-ordered "my_display_things" list. Three lines of code accomplish what seemed very complex initially. 
6. I did not locate any complete examples of preparing a dataset from start to finish. The data preparation section of this doc may seem excessively detailed, but it comes from frustration at not finding a good tutorial.  
7. I will add more items here as I remember additional challenges I encountered and solved.    


Happy training!

<font size = '2' color='red'>*This was the actual goal of my work when I started.</font>


## FYI: A copy of the dataset, these notebooks, annotations, videos input, sample video outputs, etc. are available [here](https://drive.google.com/drive/folders/1PhGNe1x4vHVVNPzkfftruOMD5HlovET5?usp=sharing).
