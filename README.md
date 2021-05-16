

Speed up image selection, tagging and bounding boxing of relevant images to then use in machine learning models.
```
1. The program reads frames/images from a video resizes the original images down into a grid.
2. Allows the user to select a span of images, place bounding boxes and tag the images with text. 
3. A text file containing the frames tagging and boxes is produced on each press of the space bar.
4. The next frames from the video are fed into the grid on Space bar.
```
Keyboard and mouse shortcuts:
```
Esc = Exit the program.
Left mouse click = Select span of images.
Right mouse drag = Create rectangle bounding boxes.
Middle mouse click = Undo selection, or bounding box placement.
Spacebar = Go to next set of images and store tagged images + frame numbers in text file.
```

```Example of loading images on spacebar:```

![load_images](https://user-images.githubusercontent.com/49783296/113673561-43098b00-970d-11eb-865b-dc85971a7f9f.gif)


```Example of selecting relevant span of images:```

![Annotate_images](https://user-images.githubusercontent.com/49783296/113673605-5157a700-970d-11eb-8fc4-1c5bc5b92fe2.gif)

```Example of having placed bounding boxes:```
![bbox](https://user-images.githubusercontent.com/49783296/113673683-692f2b00-970d-11eb-89d7-c4d25622d779.png)

```Text file produced on spacebar containing: image/frame number, tagging and bounding box coordinates. ```
![List_of_tagged_images](https://user-images.githubusercontent.com/49783296/113673801-87952680-970d-11eb-8a6b-28f38e0d617b.png)




