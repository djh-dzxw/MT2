import warnings
import torch
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def slide_inference(model, img, img_meta, rescale, args, valid_region=None, pred_cls=None):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """

    h_stride, w_stride = args.infer_stride
    h_crop, w_crop = args.crop_size
    batch_size, _, h_img, w_img = img.size()

    # min_crop = min(h_img,w_img)
    # h_crop = min(min_crop,h_crop)
    # w_crop = min(min_crop,w_crop)
    # if h_crop != 512 or w_crop != 512:
    #     h_crop = 256
    #     w_crop = 256
    # h_stride = min(h_crop, h_stride)
    # w_stride = min(w_crop, w_stride)

    num_classes = args.net_num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img)).cpu().detach()
    heat_preds = img.new_zeros((batch_size, 1, h_img, w_img)).cpu().detach()
    boundary_preds = img.new_zeros((batch_size, 1, h_img, w_img)).cpu().detach()
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img)).cpu().detach()
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                if pred_cls == None:
                    pred, heatmap, boundary = model(crop_img)
                else:
                    pred, heatmap, boundary = model(crop_img, pred_cls)
                if isinstance(pred, list):
                    pred = pred[-1]
                    heatmap = heatmap[-1]
                    boundary = boundary[-1]
            #     crop_seg_logit = torch.softmax(pred,dim=1)
            # crop_seg_logit = F.interpolate(crop_seg_logit, (h_crop, w_crop), mode='bilinear')
            crop_seg_logit = F.interpolate(pred, (h_crop, w_crop), mode='bilinear')
            heatmap = F.interpolate(heatmap, (h_crop, w_crop), mode='bilinear')
            boundary = F.interpolate(boundary, (h_crop, w_crop), mode='bilinear')
            preds += F.pad(crop_seg_logit.cpu().detach(),
                           (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
            heat_preds += F.pad(heatmap.cpu().detach(),
                                (int(x1), int(heat_preds.shape[3] - x2), int(y1), int(heat_preds.shape[2] - y2)))
            boundary_preds += F.pad(boundary.cpu().detach(), (
            int(x1), int(boundary_preds.shape[3] - x2), int(y1), int(boundary_preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
    preds = preds / count_mat
    heat_preds = heat_preds / count_mat
    boundary_preds = boundary_preds / count_mat

    if valid_region is not None:
        h_valid = valid_region[0]
        w_valid = valid_region[1]
        preds = preds[:, :, :h_valid, :w_valid]
        heat_preds = heat_preds[:, :, :h_valid, :w_valid]
        boundary_preds = boundary_preds[:, :, :h_valid, :w_valid]

    if rescale:
        preds = resize(
            preds,
            size=img_meta['ori_shape'][:2],
            mode='bilinear',
            align_corners=False,
            warning=False)
        heat_preds = resize(
            heat_preds,
            size=img_meta['ori_shape'][:2],
            mode='bilinear',
            align_corners=False,
            warning=False)
        boundary_preds = resize(
            boundary_preds,
            size=img_meta['ori_shape'][:2],
            mode='bilinear',
            align_corners=False,
            warning=False)
    return preds, heat_preds, boundary_preds
