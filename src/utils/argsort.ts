import * as tf from '@tensorflow/tfjs';

export function argSort <T extends tf.Tensor>(tensor: T){
    const array = tensor.arraySync() as number[];
    const initial = Array.from(array);
    const sorted = array.sort((a, b)=>{return a-b});
    const args = sorted.map( item=>{ return initial.indexOf(item)})
    return tf.tensor1d(args);
}
