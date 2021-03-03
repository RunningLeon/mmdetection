import torch


def add_dummy_nms_for_onnx(boxes,
                           scores,
                           max_output_boxes_per_class,
                           iou_threshold=0.5,
                           score_threshold=0.05):
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    # turn off tracing
    state = torch._C._get_tracing_state()
    # dummy indices of nms's output
    indices = torch.randint(10, (100, 3))
    output = indices
    setattr(DymmyONNXNMSop, 'output', output)
    # open tracing
    torch._C._set_tracing_state(state)
    indices = DymmyONNXNMSop.apply(boxes, scores, max_output_boxes_per_class,
                                   iou_threshold, score_threshold)
    return boxes, scores, indices


class DymmyONNXNMSop(torch.autograd.Function):
    """DymmyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):

        return DymmyONNXNMSop.output

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)
