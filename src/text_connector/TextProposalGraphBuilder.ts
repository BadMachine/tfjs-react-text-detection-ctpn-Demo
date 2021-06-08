import {TextLineCfg} from './TextLineCfg';
import {Graph} from './other';
import * as tf from '@tensorflow/tfjs';
import {argmax} from '../utils/argmax';
import {ravel} from "../utils/ravel";

export class TextProposalGraphBuilder{
    private text_proposals: number[][] ;
    private boxes_table: any;
    private im_size: number[];
    private heights: number[];
    private scores: tf.Tensor;
    constructor() {
        this.text_proposals= [[]];
        this.im_size = [];
        this.heights = [];
        this.scores = tf.tensor([]);
    }

    get_successions(index: number){

        const box = this.text_proposals[index];
        const results=[];
        for(let left = Math.round(box[0])+1; left < Math.min(Math.round(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP+1, this.im_size[1]); left++){ // for left in range(int(box[0])+1, min(int(box[0])+TextLineCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
           const adj_box_indices = this.boxes_table[left]; // adj_box_indices=self.boxes_table[left]
            for (let adj_box_index of adj_box_indices){
                if (this.meet_v_iou(adj_box_index, index)) results.push(adj_box_index);
            }

            if (results.length!==0) return results;
        }
        return results;
    }

    get_precursors(index: number) {
       const box = this.text_proposals[index];
       const results = [];

        for(let left = Math.round(box[0])-1; left > Math.max(Math.round(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) -1; left--){
            const adj_box_indices = this.boxes_table[left];
            for (let adj_box_index of adj_box_indices){
                if (this.meet_v_iou(adj_box_index, index)) results.push(adj_box_index);
            }
            if (results.length!==0) return results;
        }
        return results;
    }

    meet_v_iou(index1: number, index2: number){
        const overlaps_v = (index1: number, index2: number) => {
            const h1 = this.heights[index1];
            const h2 = this.heights[index2];
            const y0 = Math.max(this.text_proposals[index2][1], this.text_proposals[index1][1]);
            const y1 = Math.min(this.text_proposals[index2][3], this.text_proposals[index1][3]);
            return Math.max(0, y1-y0+1)/Math.min(h1, h2);
        }

        const size_similarity = (index1: number, index2: number) => {
           const h1 = this.heights[index1]
           const h2 = this.heights[index2]
            return Math.min(h1, h2) / Math.max(h1, h2);
        }

        return overlaps_v(index1, index2)>=TextLineCfg.MIN_V_OVERLAPS && size_similarity(index1, index2)>=TextLineCfg.MIN_SIZE_SIM;
    }

    is_succession_node(index: number, succession_index: number) {
        const precursors = this.get_precursors(succession_index);
        return tf.greaterEqual(this.scores.gather(index), tf.max(this.scores.gather(precursors)) ).arraySync();// here
    }
    build_graph<T extends tf.Tensor>(text_proposals: T, scores: T, im_size: number[]){
        this.text_proposals = text_proposals.arraySync() as number[][];
        this.scores = ravel(scores);
        this.im_size = im_size;
        const h1 = text_proposals.slice([0,3], [text_proposals.shape[0],1]).reshape([text_proposals.shape[0]]);
        const h2 = text_proposals.slice([0,1], [text_proposals.shape[0],1]).reshape([text_proposals.shape[0]]);
        this.heights = tf.add(tf.sub(h1,h2), 1).arraySync() as number[];
        const boxes_table =  Array.from(Array(im_size[1]), () => []);
        this.text_proposals.forEach((item, index)=>{
             // @ts-ignore
            boxes_table[Math.round(item[0])].push(index);
         })
        this.boxes_table = boxes_table;
        let graph = tf.buffer([text_proposals.shape[0], text_proposals.shape[0]], 'bool');
        for(let index = 0; index < this.text_proposals.length; index++){
            let successions = this.get_successions(index);
            if (successions.length === 0) continue;

            let succession_index;
            if (successions.length > 1) {
                succession_index = successions[argmax(this.scores.gather(successions).arraySync() as number[])];
            }else {
                succession_index = successions[0];
            }

           if (this.is_succession_node(index, succession_index)) {
               graph.set(true, index, succession_index);
            }

        }
        const _graph = graph.toTensor();
        return new Graph(_graph);
    }
}
