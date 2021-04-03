import torch


def cells_to_bboxes(preds, anchors, scale, is_preds=True):
    """Scale the predictions coming from the model

    Arguments:
        preds (tensor): tensor of shape (N, 3, scale, scale, 5+n_classes)
        anchors (tensor): tensor of shape (3, 2)
        scale (int): the number of cells the image is divided on width & height
        is_preds (bool): whether the input is predictions or the true bboxes

    Returns:
        list of converted bboxes of shape (N, 3*scale*scale, 6)
        6 -> (class_id, prob_score, x, y, w, h)
    """
    batch_size = preds.shape[0]
    n_anchors = len(anchors)
    pred_bboxes = preds[..., 1:5] # (N, 3, scale, scale, 4)
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) # (1, 3, 1, 1, 2)
        pred_bboxes[..., 0:2] = (2.0*torch.sigmoid(pred_bboxes[..., 0:2]))-0.5
        pred_bboxes[..., 2:] = ((2.0*torch.sigmoid(pred_bboxes[..., 2:]))**2)*anchors
        scores = torch.sigmoid(preds[..., 0:1])
        best_class = torch.argmax(preds[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = preds[..., 0:1]
        best_class = preds[..., 5:6]

    cell_indices = ( # (N, 3, scale, scale, 1)
        torch.arange(scale)
        .repeat(batch_size, n_anchors, scale, 1)
        .unsqueeze(-1)
        .to(pred_bboxes.device)
        )
    x = (1/scale)*(pred_bboxes[..., 0:1]+cell_indices)
    y = (1/scale)*(pred_bboxes[..., 1:2]+cell_indices.permute(0, 1, 3, 2, 4))
    wh = (1/scale)*(pred_bboxes[..., 2:4])
    bboxes = (
            torch.cat((best_class, scores, x, y, wh), dim=-1)
            .reshape(batch_size, n_anchors*scale*scale, 6)
            )
    return bboxes
