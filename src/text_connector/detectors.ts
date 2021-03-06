import {TextProposalConnectorOriented} from "./text_proposal_connector_oriented";
import {TextLineCfg} from './TextLineCfg';
import * as tf from '@tensorflow/tfjs';
import {nms} from '../nms/nms';
import {ravel} from "../utils/ravel";
import {argSort} from "../utils/argsort";
import {configInterface} from '../main';

export class TextDetector{
    mode: string;
    NMS_FUNCTION: string;
    text_proposal_connector: TextProposalConnectorOriented;
    constructor(cfg: configInterface){
        this.mode = cfg.DETECT_MODE;
        this.NMS_FUNCTION = cfg.NMS_FUNCTION;
        //this.text_proposal_connector = (mode =='O') ? new textProposalConnectorOriented() : new textProposalConnectorOriented();
        this.text_proposal_connector = new TextProposalConnectorOriented();
    }

    async detect<T extends tf.Tensor>(text_proposals: T, scores: T, size: number[]){
        //console.log(this.text_proposal_connector)
        const scoresFlat = ravel(scores);
        let keep_inds: tf.Tensor1D | tf.Tensor2D = await tf.whereAsync(scoresFlat.greater(TextLineCfg.TEXT_PROPOSALS_MIN_SCORE));
        keep_inds = ravel(keep_inds).cast('int32');
        [text_proposals, scores] = [tf.gather(text_proposals, keep_inds), tf.gather(scores, keep_inds) ];
        const sorted_indices = argSort(ravel(scores)).reverse().cast('int32'); // sorted_indices=np.argsort(scores.ravel())[::-1]
        [text_proposals, scores] = [tf.gather(text_proposals, sorted_indices), tf.gather(scores, sorted_indices) ] // text_proposals, scores=text_proposals[sorted_indices], scores[sorted_indices]
        keep_inds = await nms({ dets: text_proposals, scores: scores, thresh: TextLineCfg.TEXT_PROPOSALS_NMS_THRESH, method: this.NMS_FUNCTION} );
        keep_inds = keep_inds.cast('int32');
        [text_proposals, scores] = [tf.gather(text_proposals, keep_inds), tf.gather(scores, keep_inds) ];

        const text_recs = await this.text_proposal_connector.get_text_lines(text_proposals, scores, size);// text_recs=self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        keep_inds = await this.filter_boxes(text_recs);
        const _keep_inds = keep_inds.unstack(1)[0];
        return text_recs.gather(_keep_inds);

    }

    async filter_boxes<T extends tf.Tensor>(boxes: T){

        let heights: tf.TensorBuffer<tf.Rank.R2> | tf.Tensor2D = tf.buffer( [boxes.shape[0], 1] );//heights=np.zeros((len(boxes), 1), np.float)
        let widths: tf.TensorBuffer<tf.Rank.R2> | tf.Tensor2D = tf.buffer( [boxes.shape[0], 1] ); //widths=np.zeros((len(boxes), 1), np.float)
        let scores: tf.TensorBuffer<tf.Rank.R2> | tf.Tensor2D = tf.buffer( [boxes.shape[0], 1] );//scores=np.zeros((len(boxes), 1), np.float)
        let index = 0;
        for(let i = 0; i < boxes.shape[0]; i++){
            // @ts-ignore
            heights.set( tf.abs(tf.gatherND(boxes,[i, 5]).sub(tf.gatherND(boxes,[i, 1]))).add(tf.abs(tf.gatherND(boxes,[i, 7]).sub(tf.gatherND(boxes, [i, 3]) ))).div(2.0).add(1).arraySync(), 0,index); // heights[index]=(abs(box[5]-box[1])+abs(box[7]-box[3]))/2.0+1
            // @ts-ignore
            widths.set( tf.abs(tf.gatherND(boxes,[i, 2]).sub(tf.gatherND(boxes,[i, 0]))).add(tf.abs(tf.gatherND(boxes,[i, 6]).sub(tf.gatherND(boxes, [i, 4]) ))).div(2.0).add(1).arraySync(), 0,index); // heights[index]=(abs(box[5]-box[1])+abs(box[7]-box[3]))/2.0+1
            // @ts-ignore
            scores.set(tf.gatherND(boxes,[i, 8]).arraySync(), 0,index)//scores[index] = box[8]

            index+=1;
        }
        heights = heights.toTensor();
        widths = widths.toTensor();
        scores = scores.toTensor();
        return tf.whereAsync( tf.logicalAnd( tf.greater(widths.div(heights), TextLineCfg.MIN_RATIO), scores.greater(TextLineCfg.LINE_MIN_SCORE) ).logicalAnd( widths.greater(tf.mul(TextLineCfg.TEXT_PROPOSALS_WIDTH, TextLineCfg.MIN_NUM_PROPOSALS)) ) );
      }
}

// module.exports = TextDetector;
