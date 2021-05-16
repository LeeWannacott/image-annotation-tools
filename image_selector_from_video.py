'''
Author: Lee Wannacott
Github: https://github.com/LeeWannacott/
'''
from cv2 import cv2
import numpy as np
import math
import easygui
import sys
import pyautogui

''' 
Speed up image selection, tagging and bounding boxing of relevant images to then use in machine learning models.

1. The program reads frames/images from a video resizes the original images down into a grid.
2. Allows the user to select a span of images, place bounding boxes and tag the images with text. 
3. A text file containing the frames tagging and boxes is produced on each press of the space bar.
4. The next frames from the video are fed into the grid on Space bar.

Esc = Exit the program.
Left mouse click = Select span of images.
Right mouse click = Select bounding boxes.
Middle mouse click = Undo selection.
Space bar = Go to next set of images and store tagged images + frame numbers in text file.
'''

class CreateGrid: pass
grid = CreateGrid()

# ************************************************************************
# variables to be changed by the user:

# Select file path to read video file from.
cap = cv2.VideoCapture('/home/lee/Desktop/image-annotation-tool/video.mp4')

# Number of rows is changeable value. Default value is 7.
grid.number_of_rows = 5

# Variables containing default window size for the grid - default values assume 1080p.
grid.window_width = 1800
grid.window_height = 1000

# Original images from video that will be resized down to fit into grid.
grid.initial_img_width = 1920
grid.initial_img_height = 1080

# ************************************************************************

grid.cell_height= 0
grid.cell_width = 0
grid.number_of_cells= 0
grid.number_of_columns = 0
grid.image_resize_x = 0
grid.image_resize_y = 0

frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frames in video: ' + str(frames_in_video))

def recalculate_window_stuff():
    (_, _, temp_window_width, temp_window_height) = cv2.getWindowImageRect('image_selector_from_video')
    if not (temp_window_width == -1 or temp_window_height == -1):
        grid.window_width = temp_window_width
        grid.window_height = temp_window_height
    print(f"Window size: {grid.window_width}x{grid.window_height}")

    grid.cell_height = grid.window_height // grid.number_of_rows
    grid.cell_width = int(grid.cell_height * (grid.initial_img_width / grid.initial_img_height))

    # cell_aspect_ratio and image_aspect_ratio used for assert statement; for debugging.
    cell_aspect_ratio = grid.cell_width / grid.cell_height
    image_aspect_ratio = grid.initial_img_width / grid.initial_img_height
    assert abs(cell_aspect_ratio - image_aspect_ratio) < 0.01, f"cell_aspect_ratio={cell_aspect_ratio} \
        image_aspect_ratio={image_aspect_ratio}"

    grid.number_of_columns = grid.window_width // grid.cell_width
    grid.image_resize_x = grid.cell_width / grid.initial_img_width
    grid.image_resize_y = grid.cell_height / grid.initial_img_height
    grid.number_of_cells = grid.number_of_rows * grid.number_of_columns
    print(f"Cell width: {str(grid.cell_width)}, Cell Height: {str(grid.cell_height)}, "
          f"Number of rows: {str(grid.number_of_rows)},Number of Columns: {str(grid.number_of_columns)}")

class ClickFunctionality: pass
mouse_click = ClickFunctionality()
mouse_click.last_mouse_button_clicked = []
mouse_click.enable_draw_on_grid = False
mouse_click.coordinates = []


class ImageSelection: pass
image_selection = ImageSelection()
image_selection.drawn_one_cell_or_span = []
image_selection.tagged_as = ''
image_selection.new_image = ''
image_selection.image_list_temporary = []
image_selection.image_list = []
image_selection.cell_numbers_selection_temporary = []
image_selection.cell_numbers_selection_for_drawing_text = []

class CreateBoundingBox: pass
bounding_box = CreateBoundingBox()
bounding_box.bounding_box_start_coordinates_x_y = ()
bounding_box.list_of_bounding_box_coordinates = []
bounding_box.temp_list_cells_with_bboxes = []
bounding_box.temp_dict_and_cell_number_bboxes = {}
bounding_box.perm_dict_of_cell_num_and_bbox = {}

class CreateTextFile: pass
create_text_file = CreateTextFile()
create_text_file.frame_numbers_list = []
create_text_file.list_of_frames_to_keep = []
create_text_file.image_list_to_print = []
create_text_file.image_list_to_keep = []
create_text_file.cell_numbers_list_for_each_grid = []

def click_event(event, x, y, flags, param):
    # Mouse click for left button
    global mouseX
    global mouseY
    global global_click_event

    global_click_event = event
    mouseX = x
    mouseY = y
    if event == cv2.EVENT_LBUTTONDOWN and image_selection.tagged_as == '' and \
        mouse_click.enable_draw_on_grid == True:
        mouse_click.last_mouse_button_clicked.append('left')
        # Gets row and column number on left mouse click
        col_number = x / grid.cell_width
        x1 = math.trunc(col_number)
        row_number = y / grid.cell_height
        y1 = math.trunc(row_number)

        # Get cell number based on coordinates x1, y1
        def coordinate_to_cell(x, y):
            # !SO 9816024/coordinates-to-grid-box-number
            cell_number = int(x + (y * grid.number_of_columns))

            # Temporary list for drawing rectangles
            image_selection.cell_numbers_selection_temporary.append(cell_number)
            image_selection.cell_numbers_selection_for_drawing_text.append(cell_number)
            create_text_file.cell_numbers_list_for_each_grid.append(cell_number)
            return cell_number

        # For use in drawing rectangle function.
        cell = coordinate_to_cell(x1, y1)

        # Making copy of image for undo function.
        image_selection.new_image = param[0].copy()
        image_selection.image_list.append(image_selection.new_image)

        # Uses the cell number converts to coordinates and draws rectangles
        def draw_rectangles(cell_number):
            # !SO 8669189/converting-numbers-within-grid-to-their-corresponding-x-y-coordinates
            x2 = cell_number % grid.number_of_columns
            y2 = cell_number // grid.number_of_columns
            cell_x_position = x2 * grid.cell_width
            cell_y_position = y2 * grid.cell_height
            draw_to_x = cell_x_position + grid.cell_width
            draw_to_y = cell_y_position + grid.cell_height

            cv2.rectangle(param[0], (int(cell_x_position), int(cell_y_position)), (int(draw_to_x), int(draw_to_y)),
                          (152,251,152), 2)
            cv2.imshow('image_selector_from_video', param[0])

        # draw rectangles between two grid images
        def draw_rectangles_span():
            # Draws rectangle in the first cell
            if len(image_selection.cell_numbers_selection_temporary) == 1:
                draw_rectangles(cell)
                image_selection.drawn_one_cell_or_span.append('one')

            if len(image_selection.cell_numbers_selection_temporary) >= 2:
                between_backwards = list(
                    range(int(image_selection.cell_numbers_selection_temporary[-1]), \
                        int(image_selection.cell_numbers_selection_temporary[-2] + 1)))  # backwards
                between_forwards = list(
                    range(int(image_selection.cell_numbers_selection_temporary[-2]), \
                         int(image_selection.cell_numbers_selection_temporary[-1] + 1)))  # forwards

                # Draw rectangles backwards
                if image_selection.cell_numbers_selection_temporary[-1] < \
                    image_selection.cell_numbers_selection_temporary[-2]:
                    for next_cell1 in between_backwards:
                        draw_rectangles(next_cell1)

                # Draw rectangles forwards
                if image_selection.cell_numbers_selection_temporary[-1] > \
                     image_selection.cell_numbers_selection_temporary[-2]:
                    for next_cell2 in between_forwards:
                        draw_rectangles(next_cell2)

                # Clear temporary list so another two squares can be selected
                image_selection.cell_numbers_selection_temporary.clear()

                image_selection.drawn_one_cell_or_span.append('span')

                if image_selection.tagged_as == '':
                    # Window_open = True
                    image_selection.tagged_as = 'window open'
                    image_selection.tagged_as = easygui.enterbox("What is tagged in the images?")
                    # replace
                    if image_selection.tagged_as is not None:
                        image_selection.tagged_as = image_selection.tagged_as.replace(" ","")
                        draw_label_on_selection_span_of_images()
                        ####

                if image_selection.tagged_as != 'window open':
                    if image_selection.tagged_as is None:
                        image_selection.tagged_as = ''
                    image_selection.image_list_temporary.append(image_selection.tagged_as)

                    image_selection.tagged_as = ''

        def draw_label(cell_number):
            # https://stackoverflow.com/questions/8669189/converting-numbers-within-grid-to-their-corresponding-x-y-coordinates
            x2 = cell_number % grid.number_of_columns
            y2 = cell_number // grid.number_of_columns
            cell_x_position = x2 * grid.cell_width
            cell_y_position = y2 * grid.cell_height
            cv2.putText(img=param[0], text=image_selection.tagged_as, org=(cell_x_position+5, cell_y_position+20),
                        fontFace=cv2.FONT_ITALIC, fontScale=0.5, color=(156,217,107),
                        thickness=1, lineType=cv2.LINE_AA)
            cv2.imshow('image_selector_from_video', param[0])

        def draw_label_on_selection_span_of_images():
            if len(image_selection.cell_numbers_selection_for_drawing_text) >= 2:
                between_backwards = list(
                    range(int(image_selection.cell_numbers_selection_for_drawing_text[-1]),
                          int(image_selection.cell_numbers_selection_for_drawing_text[-2] + 1)))  # backwards
                between_forwards = list(
                    range(int(image_selection.cell_numbers_selection_for_drawing_text[-2]),
                          int(image_selection.cell_numbers_selection_for_drawing_text[-1] + 1)))  # forwards

                # Draw rectangles backwards
                if image_selection.cell_numbers_selection_for_drawing_text[-1] < \
                    image_selection.cell_numbers_selection_for_drawing_text[
                    -2]:
                    for backward_cell in between_backwards:
                        draw_label(backward_cell)

                # Draw rectangles forwards
                if image_selection.cell_numbers_selection_for_drawing_text[-1] > \
                    image_selection.cell_numbers_selection_for_drawing_text[
                    -2]:
                    for forward_cell in between_forwards:
                        draw_label(forward_cell)

        draw_rectangles_span()

    elif event == cv2.EVENT_MBUTTONUP and image_selection.tagged_as == '' and \
        mouse_click.enable_draw_on_grid == True and len(mouse_click.last_mouse_button_clicked) > 0 \
        and mouse_click.last_mouse_button_clicked[-1] == 'left':


        # Allows undo if one image selected.
        if len(image_selection.image_list) >= 1 and image_selection.drawn_one_cell_or_span[-1] == 'one':
            param[0] = image_selection.image_list[-1]
            cv2.imshow('image_selector_from_video', image_selection.image_list[-1])
            cv2.waitKey(1)
            image_selection.image_list.pop()
            create_text_file.cell_numbers_list_for_each_grid.pop()
            mouse_click.last_mouse_button_clicked.pop()
            image_selection.drawn_one_cell_or_span.pop()
            image_selection.cell_numbers_selection_temporary.clear()

        # Allows undo if span of images selected.
        elif len(image_selection.image_list) >= 2 and image_selection.drawn_one_cell_or_span[-1] == 'span':
            param[0] = image_selection.image_list[-2]
            cv2.imshow('image_selector_from_video', image_selection.image_list[-2])
            cv2.waitKey(1)

            # Popping lists to undo selection span of images
            image_selection.image_list.pop()
            image_selection.image_list.pop()
            mouse_click.last_mouse_button_clicked.pop()
            mouse_click.last_mouse_button_clicked.pop()
            image_selection.drawn_one_cell_or_span.pop()
            image_selection.drawn_one_cell_or_span.pop()
            create_text_file.cell_numbers_list_for_each_grid.pop()
            create_text_file.cell_numbers_list_for_each_grid.pop()

            if len(image_selection.image_list_temporary) > 0:
                image_selection.image_list_temporary.pop()

    elif event == cv2.EVENT_MBUTTONUP and mouse_click.enable_draw_on_grid == True and \
        len(mouse_click.last_mouse_button_clicked) > 0 and mouse_click.last_mouse_button_clicked[-1] == 'right':
        if len(image_selection.image_list) >= 1 and mouse_click.last_mouse_button_clicked[-1] == 'right':
            param[0] = image_selection.image_list[-1]
            cv2.imshow('image_selector_from_video', image_selection.image_list[-1])
            cv2.waitKey(1)

            image_selection.image_list.pop()
            mouse_click.last_mouse_button_clicked.pop()
            # Allows for the potential to have multiple boundary boxes in same cell
            if len(bounding_box.temp_dict_and_cell_number_bboxes[bounding_box.temp_list_cells_with_bboxes[-1]]) > 4:

                # Removing four coordinates if more than one boundary box in cell.
                for i in range(4):
                    del bounding_box.temp_dict_and_cell_number_bboxes[bounding_box.temp_list_cells_with_bboxes[-1]][-1]

            # Removes boundary boxes if in a single cell
            elif len(bounding_box.temp_dict_and_cell_number_bboxes[bounding_box.temp_list_cells_with_bboxes[-1]]) == 4:
                bounding_box.temp_dict_and_cell_number_bboxes.pop(bounding_box.temp_list_cells_with_bboxes[-1])


            # Removes cell number from temporary list
            bounding_box.temp_list_cells_with_bboxes.pop()

    # Allow bounding boxes to be places over images
    elif global_click_event == cv2.EVENT_RBUTTONDOWN and len(image_selection.drawn_one_cell_or_span) > 0 \
        and image_selection.drawn_one_cell_or_span[-1] == 'span':

        def get_bounding_box_start_coordinates(x, y):
            x_start_boundary = x
            y_start_boundary = y
            # Calculates starting cell number
            col_number = x / grid.cell_width
            x1 = math.trunc(col_number)
            row_number = y / grid.cell_height
            y1 = math.trunc(row_number)
            cell_number_on_start_of_drawing = int(x1 + (y1 * grid.number_of_columns))
            return x_start_boundary, y_start_boundary, cell_number_on_start_of_drawing

        bounding_box.bounding_box_start_coordinates_x_y = get_bounding_box_start_coordinates(x, y)

        # Handle visual drawing of placing bounding box; to give user feedback of where they are placing bbox.
        global allow_draw_bbox
        allow_draw_bbox = True
        while allow_draw_bbox == True:
            draw_bbox_images = param[0].copy()
            cv2.rectangle(draw_bbox_images, (bounding_box.bounding_box_start_coordinates_x_y[0], \
            bounding_box.bounding_box_start_coordinates_x_y[1]),(mouseX, mouseY), (32,178,170), 2)
            cv2.imshow('image_selector_from_video',draw_bbox_images)
            cv2.waitKey(10)


    elif event == cv2.EVENT_RBUTTONUP and len(image_selection.drawn_one_cell_or_span) > 0 \
        and image_selection.drawn_one_cell_or_span[-1] == 'span':
        # Get cell number at end of bounding box
        col_number = x / grid.cell_width
        cell_x = math.trunc(col_number)
        row_number = y / grid.cell_height
        cell_y = math.trunc(row_number)
        cell_number_on_end_of_drawing = int(cell_x + (cell_y * grid.number_of_columns))
        # Minus x,y positions to get cells relative position
        cell_x_position = cell_x * grid.cell_width
        cell_y_position = cell_y * grid.cell_height
        allow_draw_bbox = False

        def draw_boundary_box(x, y, start_boundary_x_and_y):
            # Making copy for un-drawing bounding box
            new_image_boundary = param[0].copy()
            image_selection.image_list.append(new_image_boundary)

            # Draws bounding box
            end_boundary_x_and_y = (x, y)
            cv2.rectangle(param[0], (start_boundary_x_and_y[0], start_boundary_x_and_y[1]),
                          (end_boundary_x_and_y[0], end_boundary_x_and_y[1]), (0,255,127), 2)
            cv2.imshow('image_selector_from_video', param[0])

            # Used for un-drawing bounding box logic
            mouse_click.last_mouse_button_clicked.append('right')

            # Makes the coordinates relative to within that cell rather than the whole window.
            cell_start_relative_position_x = start_boundary_x_and_y[0] - cell_x_position
            cell_start_relative_position_y = start_boundary_x_and_y[1] - cell_y_position
            cell_end_relative_position_x = end_boundary_x_and_y[0] - cell_x_position
            cell_end_relative_position_y = end_boundary_x_and_y[1] - cell_y_position

            # Resizing coordinates back to the those of the original sized images rounded to whole pixels.
            cell_start_relative_position_x_resized = round(cell_start_relative_position_x / grid.image_resize_x)
            cell_start_relative_position_y_resized = round(cell_start_relative_position_y / grid.image_resize_y)
            cell_end_relative_position_x_resized = round(cell_end_relative_position_x / grid.image_resize_x)
            cell_end_relative_position_y_resized = round(cell_end_relative_position_y / grid.image_resize_y)

            # Gets bounding box x,y start and x,y end, relative to that individual cell.
            list_bounding_box_coordinates = [cell_start_relative_position_x_resized, cell_start_relative_position_y_resized,
                                                    cell_end_relative_position_x_resized, cell_end_relative_position_y_resized]

            # Cells the user have selected that contain bounding boxes
            bounding_box.temp_list_cells_with_bboxes.append(bounding_box.bounding_box_start_coordinates_x_y[2])




            # Checks if there is already a key from there already being a bounding box in the cell
            if bounding_box.bounding_box_start_coordinates_x_y[2] in bounding_box.temp_dict_and_cell_number_bboxes:
                bounding_box.temp_dict_and_cell_number_bboxes[bounding_box.bounding_box_start_coordinates_x_y[2]] \
                    .extend(list_bounding_box_coordinates)

            else:
                bounding_box.temp_dict_and_cell_number_bboxes[bounding_box.bounding_box_start_coordinates_x_y[2]] = \
                    list_bounding_box_coordinates

        # Checks if its still within the same cell and if so draws bounding box
        if bounding_box.bounding_box_start_coordinates_x_y[2] == cell_number_on_end_of_drawing and \
            len(create_text_file.cell_numbers_list_for_each_grid) > 0:
            # Check if drawing boundary boxes in span of images that has been selected.
            if bounding_box.bounding_box_start_coordinates_x_y[2] in range(create_text_file.cell_numbers_list_for_each_grid[-2], \
                 create_text_file.cell_numbers_list_for_each_grid[-1]+1)  \
            or bounding_box.bounding_box_start_coordinates_x_y[2] in range(create_text_file.cell_numbers_list_for_each_grid[-1], \
                 create_text_file.cell_numbers_list_for_each_grid[-2]+1):
                draw_boundary_box(x, y, bounding_box.bounding_box_start_coordinates_x_y)


# Check if camera opened successfully.
if not cap.isOpened():
    print("Error opening video stream or file")
    sys.exit()

index = 0

cv2.namedWindow('image_selector_from_video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image_selector_from_video', grid.window_width, grid.window_height)


# Read until video is completed
def image_grid(index, x_offset=0, y_offset=0, i=0):
    cap.set(1, index)
    recalculate_window_stuff()
    # Resetting large image/grid to black each time.
    l_img = np.zeros((grid.window_height, grid.window_width, 3), np.uint8)
    cv2.imshow('image_selector_from_video', l_img)
    cv2.waitKey(1)
    index_for_frame_list = index

    while i < grid.number_of_cells:

        mouse_click.enable_draw_on_grid = False
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:

                #  Resize image
                s_image = cv2.resize(frame, (0, 0), None, grid.image_resize_x, grid.image_resize_y)

                # Put small images onto large image
                x_offset = (i % grid.number_of_columns) * int(grid.cell_width)
                y_offset = (i // grid.number_of_columns) * int(grid.cell_height)
                l_img[y_offset:y_offset + s_image.shape[0], x_offset:x_offset + s_image.shape[1]] = s_image

                # Show each small images drawn
                cv2.imshow('image_selector_from_video', l_img)
                cv2.waitKey(1)

                # i for count of images, index keeps track of where you are in frames
                i += 1
                index += 1
                if i == grid.number_of_cells or index == frames_in_video:
                    mouse_click.enable_draw_on_grid = True
                    # Parameters passed to mouse click function
                    param = [l_img, index_for_frame_list]
                    while True:

                        cv2.setMouseCallback('image_selector_from_video', click_event, param)
                        c = cv2.waitKey(1)

                        if c == 27:  # Escape
                            print('Esc pressed to Exit')
                            sys.exit()

                        elif c == 32:  # Space
                            print('Spacebar pressed')

                            # Getting rid of an uneven number of cells in lists
                            if len(image_selection.cell_numbers_selection_temporary) % 2 == 0:
                                pass
                            else:
                                image_selection.cell_numbers_selection_temporary.pop()
                            if len(create_text_file.cell_numbers_list_for_each_grid) % 2 == 0:
                                pass
                            else:
                                create_text_file.cell_numbers_list_for_each_grid.pop()

                            # Clears lists for the case of single selected rectangle
                            image_selection.cell_numbers_selection_temporary.clear()
                            image_selection.image_list.clear()

                            # Creates a image list from the temporary image lists of each grid of images.
                            def create_permanent_image_list():
                                for images in image_selection.image_list_temporary:
                                    create_text_file.image_list_to_keep.append(images)

                            create_permanent_image_list()

                            # Transforms cell number in temporary grid into frame and appends to frame numbers list
                            def each_grid_cells_into_frames_list():
                                for cell in create_text_file.cell_numbers_list_for_each_grid:
                                    frame_number = int(param[1]) + int(cell)
                                    create_text_file.frame_numbers_list.append(frame_number)

                            each_grid_cells_into_frames_list()



                            # Calculates frames spans backwards and forwards
                            def make_list_of_frames_to_keep(image_count=0):
                                # Putting into a list of two's for calculating frame spans
                                frame_numbers_list_sliced = zip(create_text_file.frame_numbers_list[0::2], \
                                    create_text_file.frame_numbers_list[1::2])
                                # Clear lists when space pressed
                                create_text_file.list_of_frames_to_keep.clear()
                                create_text_file.image_list_to_print.clear()

                                for numbers in frame_numbers_list_sliced:
                                    # Forward frame spans
                                    if numbers[0] < numbers[1]:
                                        for frames1 in range(numbers[0], numbers[1] + 1):
                                            create_text_file.list_of_frames_to_keep.append(frames1)
                                            create_text_file.image_list_to_print.append(create_text_file. \
                                                image_list_to_keep[image_count])
                                        image_count += 1

                                    # Backward frame spans
                                    elif numbers[0] > numbers[1]:
                                        for frames2 in range(numbers[1], numbers[0] + 1):
                                            create_text_file.list_of_frames_to_keep.append(frames2)
                                            create_text_file.image_list_to_print.append(create_text_file. \
                                                image_list_to_keep[image_count])
                                        image_count += 1

                            make_list_of_frames_to_keep()

                            def bounding_box_cell_keys_to_frames():
                                # Changing the key of dict: cell numbers into frame numbers.
                                temp_dict_to_append = {k + int(param[1]): v for (k, v) in bounding_box.\
                                    temp_dict_and_cell_number_bboxes.items()}
                                bounding_box.perm_dict_of_cell_num_and_bbox.update(temp_dict_to_append)
                            bounding_box_cell_keys_to_frames()

                            # Clear temp dict of bounding boxes ready for next lot of frames to have bounding boxes.
                            bounding_box.temp_dict_and_cell_number_bboxes.clear()

                            # Clearing temporary lists, so ready for the next lot of images.
                            create_text_file.cell_numbers_list_for_each_grid.clear()
                            image_selection.image_list_temporary.clear()

                            def creates_text_file():
                                # Creating text file with frame numbers and what user tags images as.
                                file = open('List_of_images.txt', 'w')

                                for i, frame in enumerate(create_text_file.list_of_frames_to_keep):
                                    if frame in bounding_box.perm_dict_of_cell_num_and_bbox.keys():
                                        file.write(f'{frame} {create_text_file.image_list_to_print[i]} \
                                            {bounding_box.perm_dict_of_cell_num_and_bbox[frame]} \n')
                                    else:
                                        file.write(f'{frame} {create_text_file.image_list_to_print[i]} \n')

                                file.close()

                            # Output text file on each press of Space.
                            creates_text_file()

                            # Continue calling function if hasn't hit end of video
                            if frames_in_video != index:
                                return image_grid(index)

                            # If Space bar pressed and end of the video exit out of loop and save tagging.
                            elif (frames_in_video == index and index >
                            grid.number_of_cells):
                                creates_text_file()
                                return


# Calls function with index of 0.
image_grid(index)

# When everything done, release the video capture object.
cap.release()

# Closes all the windows.
cv2.destroyAllWindows()

# Shuts program down.
sys.exit()