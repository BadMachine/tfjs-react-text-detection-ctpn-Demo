import * as tf from '@tensorflow/tfjs';
import { proposal_layer } from './rpn_msr/proposal_layer_tf';
import { resize_im } from "./utils/resize";
import { TextDetector } from './text_connector/detectors';
import { RGB2BGR } from './utils/RGB2BGR';
import { _get_blobs } from "./fast_rcnn/inference_blob";
import {GraphModel} from '@tensorflow/tfjs';

export interface configInterface extends Object{
    NMS_FUNCTION: string,
    ANCHOR_SCALES: number[],
    PIXEL_MEANS: tf.Tensor,
    SCALES: number[] ,
    MAX_SIZE:  number,
    HAS_RPN: true,
    DETECT_MODE: string,
    pre_nms_topN: number,
    post_nms_topN: number,
    nms_thresh: number,
    min_size: number,
}

export default class CTPN{
    model: Promise<GraphModel>;
    cfg: configInterface;
    constructor(config: configInterface) {
        this.model = tf.loadGraphModel('https://cdn.jsdelivr.net/gh/BadMachine/tfjs-text-detection-ctpn/ctpn_web/model.json'); //tf.loadGraphModel('file://./ctpn_web/model.json');
        this.cfg = config;
    }

    async predict(image: HTMLImageElement): Promise<[tf.Tensor, number]>{
        tf.engine().startScope()
        const image_swapped = RGB2BGR(tf.browser.fromPixels(image, 3).cast('float32')) as tf.Tensor3D;
        const [img, scale] = resize_im(image_swapped, 600, 1200);

        const [blobs, im_scales] = _get_blobs(img as tf.Tensor3D, null, this.cfg);

        if (this.cfg.HAS_RPN){
            const im_blob = blobs.data as tf.Tensor4D;
            blobs.im_info = tf.tensor( [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]]);
        }
        const model = await this.model;
        const raw = await model.executeAsync(img.expandDims());
        const [cls_prob, box_pred] = raw as Array<tf.Tensor4D>;
        let [scores, proposals, bbox_deltas] = await proposal_layer(this.cfg, cls_prob, box_pred, blobs.im_info as any,'TEST');
        const boxes = tf.div(proposals, im_scales[0]); // bixes a bit different
        const textDetector = new TextDetector(this.cfg);
        const _boxes = await textDetector.detect(boxes, scores.reshape([scores.shape[0],1]), img.shape.slice(0,2));
        return [_boxes, scale];
        tf.engine().endScope()
    }

    async draw<T extends tf.Tensor>(canvas: HTMLCanvasElement, _boxes: T, scale: number, color: string){

        const boxes = _boxes.arraySync() as number[][];
        for(let box of boxes){

            const ctx = canvas.getContext('2d');
            ctx!.beginPath();
            ctx!.strokeStyle = color;
            ctx!.lineWidth = 4;
            ctx!.moveTo(box[0]/ scale, box[1]/ scale);
            ctx!.lineTo(box[2] / scale, box[3] / scale);

            ctx!.lineTo(box[0] / scale, box[1] / scale);
            ctx!.lineTo(box[4] / scale, box[5] / scale);

            ctx!.lineTo(box[6] / scale, box[7] / scale);
            ctx!.lineTo(box[2] / scale, box[3] / scale);

            ctx!.stroke();
            ctx!.closePath();

        }

    }

}
