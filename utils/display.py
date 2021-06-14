import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys

def get_color_palette(class_names):
    cmaps = [
        plt.get_cmap('tab20'),
        plt.get_cmap('tab20b'),
        plt.get_cmap('tab20c'),
        plt.get_cmap('tab20'),
        ]
    colors = []
    for i in range(len(class_names)):
        ptr = i%20
        cmap = cmaps[i//20]
        steps = np.linspace(0, 1, 20)
        color = (np.array(cmap(steps[ptr])[:3])*255).astype(np.uint8)
        colors.append(tuple(color.tolist()))

    return colors

def get_color_mask(mask, color):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask >= 125], g[mask >= 125], b[mask >= 125] = color
    color_mask = np.stack([r, g, b], axis=2)
    return color_mask

def get_color(tag, hue_step=0.41):
    tag = int(tag)
    h, v = (tag*hue_step) % 1, 1. - (int(tag*hue_step)%4)/5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(r*255), int(255*g), int(255*b)

def draw_mask(frame, bbox, mask):
    canvas = np.zeros_like(frame).astype(np.uint8)
    xmin, ymin = int(bbox[0]), int(bbox[1])
    xmax, ymax = int(bbox[2]), int(bbox[3])
    canvas[ymin:ymax, xmin:xmax] = mask
    frame[:, :, :] = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

def draw_boxes(frame, boxes, class_names):
    """Draw bounding boxes on image frame

    Arguments:
        frame (ndarray): numpy image frame
        boxes (ndarray): boxes of shape (N, 6)
        class_names (list): list of class names

    Returns:
        image frame with bounding boxesimage

    NOTES: box format is (x1, y1, x2, y2, conf, class)
    """
    # Crreate color palette
    colors = get_color_palette(class_names)
    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        conf, cls = float(box[4]), int(box[5])
        name, color = class_names[cls], colors[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
        draw_text(frame, name, (x1, y1),
                bgcolor=color, fgcolor=(255, 255, 255),
                fontScale=1, thickness=1, margin=1)
    return frame

def draw_text(frame, text, position,
            fgcolor=(85, 125, 255),
            bgcolor=(85, 135, 255),
            fontScale=1, thickness=3, margin=5):
    """Draw text on the specified frame

    Args:
        frame (ndarray): processing frame
        text (string): text to render
        position (tuple): text position (tl_x, tl_y)
        fgcolor (tuple): BGR color palette for font color
        bgcolor (tuple): BGR color palette for background color
        fontScale (int): font scale
        thickness (int): line thickness
        margin (int): space between texts
    """
    # opencv doesn't handle `\n` in the text
    # therefore we handle it line by line
    lines = text.split('\n')
    text_widths = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][0]
                    for line in lines ]
    text_heights = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][1]
                    for line in lines ]
    max_width = int(max(text_widths))
    max_height = int(max(text_heights))
    tl_x = int(position[0])
    tl_y = int(position[1])

    # draw background
    cv2.rectangle(frame,
            (tl_x, tl_y),
            (tl_x+max_width, tl_y+max_height*len(lines)),
            bgcolor, -1)

    # draw text line by line
    for j, line in enumerate(lines):
        cv2.putText(frame, line,
                (tl_x+margin, tl_y+(max_height*(j+1))-margin),
                color=fgcolor,
                fontScale=fontScale,
                thickness=thickness,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

