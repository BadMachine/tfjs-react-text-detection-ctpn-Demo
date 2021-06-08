import * as tf from '@tensorflow/tfjs';

export function resize_im(im: tf.Tensor3D, scale: number, max_scale: number |null): [tf.Tensor3D, number] {
    let f = scale / Math.min(im.shape[0], im.shape[1]);
    if (max_scale != null && f * Math.max(im.shape[0], im.shape[1]) > max_scale) {
        f = max_scale / Math.max(im.shape[0], im.shape[1]);
    }
    const [newH, newW] = [ ~~(im.shape[0] * f), ~~(im.shape[1] * f)];
    console.log([newH, newW])
    return [tf.image.resizeBilinear(im, [newH, newW]), f]
}
