# face_movie
Create a timelapse movie for faces (An adaptation of the repo https://github.com/andrewdcampbell/face-movie)

Create a video warp sequence of human faces. Can be used, for example, to create a time-lapse video showing someone's face change over time. 

# requirments

* OpenCV
* Dlib
* MoviePy
* Scipy
* Numpy
* Matplotlib
* pillow

# Installation

1. Clone the Repo
2. Download the trained face detector model from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Unzip it and place it in the root directory of the repo.

# Creating a face movie - reccomended workflow
1. Make a directory `<FACE_MOVIE_DIR>` in the root directory of the repo with the desired face images. The images must feature a clear frontal view of the desired face (other faces can be present too). The image filenames must be in lexicographic order of the order in which they are to appear in the video.
2. Create a directory `<ALIGN_OUTPUT>`. Then align the faces in the images with
```
python face-movie/align.py -images <FACE_MOVIE_DIR> -target <BASE_IMAGE> 
                           [-overlay] [-border <BORDER>] -outdir <ALIGN_OUTPUT>
```
The output will be saved to the provided <ALIGN_OUTPUT> directory. BASE_IMAGE is the image to which all other images will be aligned to. It should represent the "typical" image of all your images - it will determine the output dimensions and facial position.

The optional -overlay flag places subsequent images on top of each other (recommended). The optional -border <BORDER> argument adds a white border <BORDER> pixels across to all the images for aesthetics. I think around 5 pixels looks good.

If your images contain multiple faces, a window will appear with the faces annotated and you will be prompted to enter the index of the correct face on the command line.

At this point you should inspect the output images and re-run the alignment with new parameters until you're satisfied with the result.
3. Morph the sequence with
```
python face-movie/main.py -morph -images <ALIGN_OUTPUT> -td <TRANSITION_DUR> 
                          -pd <PAUSE_DUR> -fps <FPS> -out <OUTPUT_NAME>.mp4
```

This will create a video `OUTPUT_NAME.mp4` in the root directory with the desired parameters. Note that `TRANSITION_DUR` and `PAUSE_DUR` are floating point values while FPS is an integer.

You may again be prompted to choose the correct face.
