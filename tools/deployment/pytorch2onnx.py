import argparse
import os.path as osp
import warnings
from functools import partial

import mmcv
import numpy as np
import onnx
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

from mmdet.core.export import prepare_inputs
from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
from mmdet.models import build_detector


def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 test_pipeline,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 test_img=None,
                 do_simplify=False,
                 dynamic_export=None):

    model.cpu().eval()
    img_list, img_meta_list = prepare_inputs(
        input_img, test_pipeline, shape=input_shape)

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=False)

    output_names = ['dets', 'labels']
    if model.with_mask:
        output_names.append('masks')
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        if model.with_mask:
            dynamic_axes['masks'] = {0: 'batch', 1: 'num_dets'}

    torch.onnx.export(
        model,
        img_list,
        output_file,
        input_names=['input'],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    model.forward = origin_forward

    # get the custom op path
    ort_custom_op_path = ''
    try:
        from mmcv.ops import get_onnxruntime_op_path
        ort_custom_op_path = get_onnxruntime_op_path()
    except (ImportError, ModuleNotFoundError):
        warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with ONNXRuntime from source.')

    if do_simplify:
        import onnxsim

        from mmdet import digit_version

        min_required_version = '0.3.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

        input_dic = {'input': img_list[0].detach().cpu().numpy()}
        onnxsim.simplify(
            output_file, input_data=input_dic, custom_lib=ort_custom_op_path)
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # wrap onnx model
        onnx_model = ONNXRuntimeDetector(output_file, model.CLASSES, 0)
        if dynamic_export:
            # scale up to test dynamic shape
            input_shape = [int((_ * 1.5) // 32 * 32) for _ in input_shape]

        if test_img is None:
            test_img = input_img

        # prepare test inputs, keep_ratio if dynamic export, rescale=True
        img_list, img_meta_list = prepare_inputs(
            test_img,
            test_pipeline,
            shape=input_shape,
            keep_ratio=dynamic_export,
            rescale=True)

        # get pytorch output
        pytorch_results = model(
            img_list, img_metas=img_meta_list, return_loss=False,
            rescale=True)[0]
        # get onnx output
        img_list = [_.cuda().contiguous() for _ in img_list]
        if dynamic_export:
            img_list = img_list + [_.flip(-1).contiguous() for _ in img_list]
            img_meta_list = img_meta_list * 2
        onnx_results = onnx_model(
            img_list, img_metas=img_meta_list, return_loss=False)[0]
        # visualize predictions
        if show:
            from mmdet.apis import show_result_pyplot
            show_img = mmcv.imread(test_img)
            show_result_pyplot(
                model, show_img, pytorch_results, title='Pytorch')
            show_result_pyplot(
                model, show_img, onnx_results, title='ONNXRuntime')

        # compare a part of result
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = args.shape
    else:
        raise ValueError('invalid input shape')

    # build the model and load checkpoint
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmdet.datasets import DATASETS
        dataset = DATASETS.get(cfg.data.test['type'])
        assert (dataset is not None)
        model.CLASSES = dataset.CLASSES

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../../demo/demo.jpg')

    # convert model to onnx file
    pytorch2onnx(
        model,
        args.input_img,
        input_shape,
        cfg.test_pipeline,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        test_img=args.test_img,
        do_simplify=args.simplify,
        dynamic_export=args.dynamic_export)
