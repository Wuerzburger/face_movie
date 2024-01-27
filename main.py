# USAGE: python face-movie/main.py (-morph | -average) -images IMAGES [-td TD] [-pd PD] -fps FPS -out OUT
from io import BytesIO
from PIL import Image
from face_morph import morph_images
from subprocess import Popen
import argparse
import os
import cv2
import time

def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-morph", help="Create morph sequence", action='store_true')
    group.add_argument("-average", help="Create average face", action='store_true')
    ap.add_argument("-images", help="Directory of input images", required=True)
    ap.add_argument("-td", type=float, help="Transition duration (in seconds)", default=3.0)
    ap.add_argument("-pd", type=float, help="Pause duration (in seconds)", default=0.0)
    ap.add_argument("-fps", type=int, help="Frames per second", default=25)
    ap.add_argument("-out", help="Output file name", required=True)
    args = vars(ap.parse_args())

    im_dir = args["images"]
    frame_rate = args["fps"]
    duration = args["td"]
    pause_duration = args["pd"]
    output_name = args["out"]

    valid_formats = [".jpg", ".jpeg", ".png"]
    im_files = [f for f in os.listdir(im_dir) if os.path.splitext(f)[1].lower() in valid_formats]
    im_files = sorted(im_files, key=lambda x: x.lower())
    if args["morph"]:
        morph_images(im_dir, im_files, duration, frame_rate, pause_duration, output_name)
    #else:
    #    average_images(im_dir, im_files, output_name)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
