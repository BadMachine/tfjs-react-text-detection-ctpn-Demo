import * as tf from '@tensorflow/tfjs';
import {TextProposalGraphBuilder} from './TextProposalGraphBuilder';
import numeric from 'numeric';

//reimplement this to tf

class poly1d{
    argA: tf.Tensor;
    argB: tf.Tensor;
    constructor(args: tf.Tensor) {
        this.argA = tf.gatherND(args,[0]);
        this.argB = tf.gatherND(args,[1]);
    }
    solve(x: number){
        // return this.argA * x + this.argB;
        return this.argA.mul(x).add(this.argB);
    }
    get equation(){
        return `${this.argA.arraySync()} x + ${this.argB.arraySync()};`
    }
}

function polyfit<T extends tf.Tensor>(_x: T, _y: T, order: number){
    let xArray = _x.arraySync() as number[];
    let yArray = _y.arraySync() as number[];
    if (xArray.length <= order) console.warn("Warning: Polyfit may be poorly conditioned.")
    let xMatrix = []
    let yMatrix = yArray;//numeric.transpose([yArray])

    for (let i = 0; i < xArray.length; i++) {
        let temp = [];
        for (let j = 0; j <= order; j++) {
            temp.push(Math.pow(xArray[i], j));
        }
        xMatrix.push(temp);
    }
    let xMatrixT = numeric.transpose(xMatrix);
    let dot1 = numeric.dot(xMatrixT, xMatrix);
    let dot2 = numeric.dot(xMatrixT, yMatrix);
    let dotInv = numeric.inv(<number[][]>dot1)
    return tf.tensor(numeric.dot(dotInv, dot2)).reverse();
}

export class TextProposalConnectorOriented{
    graph_builder: TextProposalGraphBuilder;
    constructor() {
        this.graph_builder = new TextProposalGraphBuilder();
    }

    async group_text_proposals<T extends tf.Tensor>(text_proposals: T, scores: T, im_size: number[]){
        const graph = this.graph_builder.build_graph(text_proposals, scores, im_size);
        return await graph.sub_graphs_connected();
    }

    fit_y<T extends tf.Tensor>(X: T, Y: T, x1: T, x2: T){
        if( X.gather(0).equal(X.sum().div(X.shape[0])).arraySync()){
            return [tf.gatherND(Y,[0]), tf.gatherND(Y,[0])];
        }
        const p = new poly1d(polyfit(X, Y, 1));//p=np.poly1d(np.polyfit(X, Y, 1))
        return [p.solve(x1.arraySync() as number), p.solve(x2.arraySync() as number)]
    }

    async get_text_lines<T extends tf.Tensor>(text_proposals: T, scores: T, im_size: number[]){
       const tp_groups = await this.group_text_proposals(text_proposals, scores, im_size) as number[][];
       //let text_lines = tf.zeros([tp_groups.length, 8], 'float32');
       //console.log('text_lines shape:', text_lines.shape)
       let text_lines = tf.buffer( [tp_groups.length, 8]);

        tp_groups.forEach((tp_indices,index)=>{
            const text_line_boxes = tf.gather(text_proposals, tp_indices);
            //console.log(`index: ${index}`)
            const X = tf.add(text_line_boxes.slice([0,0], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]), text_line_boxes.slice([0,2], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]) ).div(2);// X = (text_line_boxes[:,0] + text_line_boxes[:,2]) / 2# ???????????????????????????x???y??????
            const Y = tf.add(text_line_boxes.slice([0,1], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]), text_line_boxes.slice([0,3], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]) ).div(2);// X = (text_line_boxes[:,0] + text_line_boxes[:,2]) / 2# ???????????????????????????x???y??????

            const z1 = polyfit(X, Y, 1);

            const x0 = tf.min(text_line_boxes.slice([0,0], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]])); //x0=np.min(text_line_boxes[:, 0])
            const x1 = tf.max(text_line_boxes.slice([0,2], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]));
            const offset = tf.sub( tf.gatherND(text_line_boxes, [0,2]), tf.gatherND(text_line_boxes, [0,0]) ).mul(0.5);//offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5
            //offset.print();
            const [lt_y, rt_y] = this.fit_y(text_line_boxes.slice([0,0], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]), text_line_boxes.slice([0,1], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]), x0.add(offset), x1.sub(offset));
            const [lb_y, rb_y] = this.fit_y(text_line_boxes.slice([0,0], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]), text_line_boxes.slice([0,3], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]), x0.add(offset), x1.sub(offset));
            const score = scores.gather(tp_indices).sum().div(tp_indices.length);//score=scores[list(tp_indices)].sum()/float(len(tp_indices))
            //score.print()
            text_lines.set(x0.arraySync() as number, index, 0);//text_lines[index, 0]=x0
            text_lines.set(tf.minimum(lt_y, rt_y).arraySync() as number, index, 1) //text_lines[index, 1]=min(lt_y, rt_y)#??????????????? ?????? ???y???????????????
            text_lines.set(x1.arraySync() as number, index, 2);//text_lines[index, 2]=x1
            text_lines.set(tf.maximum(lb_y, rb_y).arraySync() as number, index, 3)//text_lines[index, 3]=max(lb_y, rb_y)#??????????????? ?????? ???y???????????????
            text_lines.set(score.arraySync() as number, index, 4);//text_lines[index, 4]=score#???????????????
            text_lines.set(z1.gather([0]).arraySync() as number, index, 5);//text_lines[index, 5]=z1[0]#?????????????????????????????????k???b
            text_lines.set(z1.gather([1]).arraySync() as number, index, 6); //text_lines[index, 6]=z1[1]
            const height = tf.mean( text_line_boxes.slice([0,3], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]]).sub(text_line_boxes.slice([0,1], [text_line_boxes.shape[0],1]).reshape([text_line_boxes.shape[0]])) )
            text_lines.set(height.add(2.5).arraySync() as number, index, 7);//text_lines[index, 7]= height + 2.5
            //text_lines.toTensor().print()

        });
        const _text_lines = text_lines.toTensor();
        let text_recs = tf.buffer( [text_lines.shape[0], 9] );
        let index = 0;
        for(let i = 0; i< text_lines.shape[0]; i++){

            //const b1 = tf.sub(text_lines.get(i, 6), tf.div(text_lines.get(i, 7), 2) );
            const b1 = tf.sub(tf.gatherND(_text_lines,[i, 6]), tf.div(tf.gatherND(_text_lines,[i, 7]), 2) );
            const b2 = tf.add(tf.gatherND(_text_lines,[i, 6]), tf.div(tf.gatherND(_text_lines,[i, 7]), 2) );
            let x1 = tf.gatherND(_text_lines,[i, 0]);//x1 = line[0]
            let y1 = tf.gatherND(_text_lines,[i, 5]).mul(tf.gatherND(_text_lines,[i, 0])).add(b1);//y1 = line[5] * line[0] + b1
            let x2 = tf.gatherND(_text_lines,[i, 2]);//x2 = line[2]
            let y2 = tf.gatherND(_text_lines,[i, 5]).mul(tf.gatherND(_text_lines,[i, 2])).add(b1);
            let x3 = tf.gatherND(_text_lines,[i, 0]);//x1 = line[3]
            let y3 = tf.gatherND(_text_lines,[i, 5]).mul(tf.gatherND(_text_lines,[i, 0])).add(b2);//y3 = line[5] * line[0] + b2
            let x4 = tf.gatherND(_text_lines,[i, 2]);//x4 = line[2]
            let y4 = tf.gatherND(_text_lines,[i, 5]).mul(tf.gatherND(_text_lines,[i, 2])).add(b2);//y4 = line[5] * line[2] + b2
            const disX = x2.sub(x1);//disX = x2 - x1
            const disY = y2.sub(y1);//disY = y2 - y1
            const width = tf.sqrt( tf.add (disX.mul(disX), disY.mul(disY)) );//width = np.sqrt(disX * disX + disY * disY)
            const fTmp0 = y3.sub(y1);//fTmp0 = y3 - y1
            const fTmp1 = fTmp0.mul(disY).div(width);//fTmp1 = fTmp0 * disY / width
            const x = tf.abs(fTmp1.mul(disX).div(width) );//x = np.fabs(fTmp1 * disX / width)
            const y = tf.abs(fTmp1.mul(disY).div(width) );//y = np.fabs(fTmp1 * disY / width)

            if (tf.gatherND(_text_lines,[i, 5]).less(0).arraySync()){
                x1 = x1.sub(x)//x1 -= x
                y1 = y1.add(y);
                x4 = x4.add(x);//x4 += x
                y4 = y4.sub(y);
            }else{
                x2 = x2.add(x);//x2 += x
                y2 = y2.add(y);
                x3 = x3.sub(x);
                y3 = y3.sub(y);
            }
            text_recs.set(x1.arraySync() as number, index, 0);
            text_recs.set(y1.arraySync() as number, index, 1);
            text_recs.set(x2.arraySync() as number, index, 2);
            text_recs.set(y2.arraySync() as number, index, 3);
            text_recs.set(x3.arraySync() as number, index, 4);
            text_recs.set(y3.arraySync() as number, index, 5);
            text_recs.set(x4.arraySync() as number, index, 6);
            text_recs.set(y4.arraySync() as number, index, 7);
            text_recs.set(tf.gatherND(_text_lines,[i, 4]).arraySync() as number, index, 8);
            index+=1;

        }

        return text_recs.toTensor();
    }

}
