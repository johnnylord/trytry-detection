import torch


def tlbr2xywh(boxes):
    """Convert bounding box format from 'tlbr' to 'xywh'

    Arguements:
        boxes (tensor): tensor of shape (N, 4)

    NOTE: box format is (x1, y1, x2, y2)
    """
    x = (boxes[..., 0:1]+boxes[..., 2:3])/2 # (N,1)
    y = (boxes[..., 1:2]+boxes[..., 3:4])/2 # (N,1)
    w = (boxes[..., 2:3]-boxes[..., 0:1])   # (N,1)
    h = (boxes[..., 3:4]-boxes[..., 1:2])   # (N,1)
    xywh = torch.cat([x, y, w, h], dim=-1)
    return xywh

def xywh2tlbr(boxes):
    """Convert bounding box format from 'xywh' to 'tlbr'

    Arguements:
        boxes (tensor): tensor of shape (N, 4)

    NOTE: box format is (x, y, w, h)
    """
    x1 = boxes[..., 0:1]-(boxes[..., 2:3]/2)    # (N,1)
    y1 = boxes[..., 1:2]-(boxes[..., 3:4]/2)    # (N,1)
    x2 = boxes[..., 0:1]+(boxes[..., 2:3]/2)    # (N,1)
    y2 = boxes[..., 1:2]+(boxes[..., 3:4]/2)    # (N,1)
    tlbr = torch.cat([x1, y1, x2, y2], dim=-1)
    return tlbr

def cells_to_boxes(cells, scale):
    """Transform the coordinate system of prediction to image coordiante system

    Arguments:
        cells (tensor): tensor of shape (N, 3, scale, scale, 6(7))
        scale (int): the scale of image

    Returns:
        tensor of shape (N, 3, scale, scale, 6(7))
        the format of output is (x, y, w, h, conf, class, [mask])

    NOTES: the cells format is (x_offset, y_offset, w_cell, h_cell, conf, class)
    """
    N = cells.size(0)
    # Extract each dimension
    x_cells = cells[..., 0:1]   # (N, 3, scale, scale, 1)
    y_cells = cells[..., 1:2]   # (N, 3, scale, scale, 1)
    w_cells = cells[..., 2:3]   # (N, 3, scale, scale, 1)
    h_cells = cells[..., 3:4]   # (N, 3, scale, scale, 1)
    conf = cells[..., 4:5]      # (N, 3, scale, scale, 1)
    cls = cells[..., 5:6]       # (N, 3, scale, scale, 1)
    if cells.size(4) > 6:
        tails = cells[..., 6:]  # (N, 3, scale, scale, N)
    # Cell coordinates
    cell_indices = (            # (N, 3, scale, scale, 1)
        torch.arange(scale)
        .repeat(N, 3, scale, 1)
        .unsqueeze(-1)
        .to(cells.device)
        )
    # Convert coordinates
    x = (1/scale)*(x_cells+cell_indices)
    y = (1/scale)*(y_cells+cell_indices.permute(0, 1, 3, 2, 4))
    w = (1/scale)*(w_cells)
    h = (1/scale)*(h_cells)
    if cells.size(4) > 6:
        boxes = torch.cat([x, y, w, h, conf, cls, tails], dim=-1)
    else:
        boxes = torch.cat([x, y, w, h, conf, cls], dim=-1)
    return boxes
