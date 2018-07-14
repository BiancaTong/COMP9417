This project is self-coded using
	Python 3.5.
The libraries used in this project are: 
	OpenCV 3.4, Random, Math, Numpy and Copy.

There are two code files in total: 
traffic_simulation.py and test.py
(1) The first file named 'traffic_simulation.py' involves all the simulation algorithms.
(2) The second file named 'test.py' is used to run the 'traffic_simulation.py', which wil simulate fix model first and then Q-learning model.

There are another two empty folders in total:
'images_fix' and 'images_q'
(1) The 'images_fix' is used to save the images of each time-step generated in fix model.
(2) The 'image_q' is used to save the images of each time-step generated in Q-learning model.

The command to implement the simulation is:
	python3 test.py

All you need to do is to put those two files and another two empty folders into a same directory.

Input:
The screen will ask you to input several parameters: seed, gamma, alpha, epsilon and episode.

Output:
The screen will output something like:
	Fix model start simulation
	The result of fix model
	Fix model finished
	Q-learning model start simulation
	The progress of Q-learning model
	Q-learning model finished

There will be two video files output in the same directory, one is fix.avi for fix model simulation and the other is q.avi for Q-learning model simulation.
If there is some problem with the output vedio files, you could also view these two videos on YouTube.
(1) Fix model video, please goto: https://youtu.be/KSVe5CjPbsk
(2) Q-learning model video, please goto: https://youtu.be/ezEEBDhFZmY



