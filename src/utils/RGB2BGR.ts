import * as tf from '@tensorflow/tfjs';

export function RGB2BGR<T extends tf.Tensor>(image: T){
    return tf.reverse(image, -1);
}
