# image-completion
a C++ implementation of image completion with Criminisi algorithm

dependency: **OpenCV**

algorithm: **Criminisi algorithm**

## Result
in bin file directory, run `demo.exe`. Input the `golf.png` and enter you will see 

![golf](./bin/golf.png)

use the mouse to smear the woman, it looks like

![golf_ruined](./bin/golf_mask.png)

press space (must on the image window not the cmd window), the program will fix it and display. The woman disappeared. The program used the surrounding infomation to complete the blank.

![golf_result](./bin/golf_result.png)

## Other Pictures
<img src="./res/panorama.png" width = "60%"/>
<img src="./res/sky.png" width = "60%"/>

The program doesn't work well for below pictures. The first has plenty of blank, the second has the repeat unit which cannot be handled well because every part looks too similar.

<img src="./res/beach.png" width = "60%"/><img src="./res/fence.png" width = "60%"/>

There are many pictures that this program cannot fix well. (Literally, it means **most** pictures in **most** cases.)
