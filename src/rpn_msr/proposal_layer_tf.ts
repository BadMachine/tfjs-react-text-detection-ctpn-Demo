import * as tf from '@tensorflow/tfjs';
import {nms} from "../nms/nms";
import { generate_anchors } from "./generate_anchors";
import {bbox_transform_inv, clip_boxes} from '../fast_rcnn/bbox_transform';
import {ravel} from "../utils/ravel";
import { argSort } from "../utils/argsort";
import { configInterface } from '../main';

export async function proposal_layer <T extends tf.Tensor4D>(cfg: configInterface, rpn_cls_prob_reshape: T, rpn_bbox_pred:T, im_info:T , test:string, _feat_stride = [16]){
//different rpn_CLS_PROB
    const _anchors = generate_anchors( 16, [0.5, 1, 2], cfg.ANCHOR_SCALES);

    const _num_anchors = _anchors.shape[0];

    if(rpn_cls_prob_reshape.shape[0] !== 1) console.error('Only single item batches are supported');
    const [height, width] = [ rpn_cls_prob_reshape.shape[1], rpn_cls_prob_reshape.shape[2]  ];

    const reshape = tf.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2]);

    let scores = tf.slice(reshape, [0,0,0,0,1],[1, height, width, _num_anchors,1]).reshape([1, height, width, _num_anchors])

    let bbox_deltas = rpn_bbox_pred.clone();
    let shift_x = tf.range(0, width).mul(_feat_stride);
    let shift_y = tf.range(0, height).mul(_feat_stride);
    [shift_x, shift_y] = tf.meshgrid(shift_x, shift_y);
    const shifts = tf.transpose(tf.stack([ravel(shift_x), ravel(shift_y), ravel(shift_x), ravel(shift_y)], 0));

    const A = _num_anchors;
    const K = shifts.shape[0];
    let anchors = _anchors.reshape([1, A, 4]).add(shifts.reshape([1, K, 4]).transpose([1, 0, 2]) );
    anchors = anchors.reshape([K * A, 4]);

    bbox_deltas = bbox_deltas.reshape([-1, 4]);
    scores = scores.reshape([-1, 1]);
    anchors = anchors.cast('int32');
    // Convert anchors into proposals via bbox transformations
    let proposals = bbox_transform_inv(anchors, bbox_deltas);

   // proposals.print()
    // 2. clip predicted boxes to image

    // @ts-ignore
    proposals = clip_boxes(proposals, im_info.arraySync()[0].slice(0,2));//
    let keep = await _filter_boxes(proposals, cfg.min_size * 1);
    keep = ravel(keep);
    proposals = tf.gather(proposals, keep.cast('int32'));
    scores = tf.gather(scores, keep.cast('int32'));
    bbox_deltas = tf.gather(bbox_deltas, keep.cast('int32'));

    //scores.print()
    let order = argSort(ravel(scores)).reverse();
    if(cfg.pre_nms_topN > 0){
        order = tf.slice(order,0, cfg.pre_nms_topN);
    }
    proposals = tf.gather(proposals, order.cast('int32'));

    scores = tf.gather(scores, order.cast('int32'));

    bbox_deltas = tf.gather(bbox_deltas, order.cast('int32'));

    keep = await nms({ dets: proposals, scores: scores, thresh:cfg.nms_thresh, method: cfg.NMS_FUNCTION} );
    console.table( tf.memory() );
    if (cfg.post_nms_topN > 0 && keep.shape[0] > cfg.post_nms_topN){
        keep = keep.slice(0, cfg.post_nms_topN);
    }
    proposals = tf.gather(proposals, keep.cast('int32')); //proposals = proposals[keep, :]
    scores = tf.gather(scores, keep.cast('int32')); //scores = scores[keep]
    bbox_deltas = tf.gather(bbox_deltas, keep.cast('int32')); //bbox_deltas=bbox_deltas[keep,:]
    return [ ravel(scores.cast('float32')), proposals.cast('float32'), bbox_deltas];
}


function _filter_boxes<T extends tf.Tensor, X extends number>(boxes: T, min_size: X): Promise<tf.Tensor>{
// Remove all boxes with any side smaller than min_size.
    const _ws_part_one =  tf.slice(boxes, [0,2], [boxes.shape[0],1]).reshape([boxes.shape[0]])//.squeeze();
    const _ws_part_two =  tf.slice(boxes, [0,0], [boxes.shape[0],1]).reshape([boxes.shape[0]])//.squeeze();
    const ws = _ws_part_one.sub(_ws_part_two ).add(1);  // const ws = boxes[:, 2] - boxes[:, 0] + 1;

    const _hs_part_one =  tf.slice(boxes, [0,3], [boxes.shape[0],1]).reshape([boxes.shape[0]])//.squeeze();
    const _hs_part_two =  tf.slice(boxes, [0,1], [boxes.shape[0],1]).reshape([boxes.shape[0]])//.squeeze();
    const hs = _hs_part_one.sub(_hs_part_two ).add(1);  // hs = boxes[:, 3] - boxes[:, 1] + 1

    const cond_part_one = tf.greaterEqual(ws, min_size );
    const cond_part_two = tf.greaterEqual(hs, min_size );

    const bitwise = tf.logicalAnd(cond_part_one, cond_part_two);

    return tf.whereAsync( bitwise ); // different result
}
