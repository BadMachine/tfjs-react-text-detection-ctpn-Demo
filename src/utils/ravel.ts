import * as tf from '@tensorflow/tfjs';

export function ravel<T extends tf.Tensor>(tensor: T){
    return tf.tensor( tf.util.flatten(tensor.arraySync()));
}
