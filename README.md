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

