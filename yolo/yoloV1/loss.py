import tensorflow as tf
import numpy as np

def iou(y_true, y_pred, n_bouding_box):

    w_true = y_true[:, :, 2:3]
    h_true = y_true[:, :, 3:4]

    w_pred = y_pred[:, :, :, :, :, 2:3]
    h_pred = y_pred[:, :, :, :, :, 3:4]
    
    xi1 = tf.math.maximum(y_true[:, :, :1] - w_true/2,
                          y_pred[:, :, :, :, :, :1] - w_pred/2)

    yi1 = tf.math.maximum(y_true[:, :, 1:2] - h_true/2,
                          y_pred[:, :, :, :, :, 1:2] - h_pred/2)

    xi2 = tf.math.minimum(y_true[:, :, :1] + w_true/2,
                          y_pred[:, :, :, :, :, :1] + w_pred/2)

    yi2 = tf.math.minimum(y_true[:, :, 1:2] + h_true/2,
                          y_pred[:, :, :, :, :, 1:2] + h_pred/2)

    inter_area = tf.math.maximum((xi2 - xi1), 0) * tf.math.maximum((yi2 - yi1), 0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = w_pred * h_pred
    box2_area = w_true * h_true
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou


#@tf.function
def yolo_loss(y_true_coord,
              y_pred_coord,
              y_true_class,
              y_pred_class, 
              n_bouding_box=1,
              lambda_coord=1, 
              lambda_noobj=1):
    
    y_true_coord = tf.cast(y_true_coord, 'float')
    y_pred_coord = tf.cast(y_pred_coord[:, :, :, :, tf.newaxis, :], 'float')

    iou_result = iou(y_true_coord, y_pred_coord, n_bouding_box)

    i_obj = tf.where((iou_result==tf.reduce_max(iou_result, axis=3, keepdims=True)) & (iou_result!=0), 1., 0.)

    i_i = tf.reduce_sum(i_obj, axis=-2)

    i_noobj = 1 - i_i

    i = tf.reduce_max(i_i, axis=3)

    y = tf.square(y_true_coord - y_pred_coord)
    
    element1 = lambda_coord * tf.reduce_sum(
                                i_obj * (y[:, :, :, :, :, :1] +
                                y[:, :, :, :, :, 1:2]),
                                axis=tf.range(1,tf.rank(y)-1)
                                )
    
    element2 = lambda_coord * tf.reduce_sum(
                                i_obj * (y[:, :, :, :, :, 2:3] +
                                y[:, :, :, :, :, 3:4]),
                                axis=tf.range(1,tf.rank(y)-1)
                                )
 
    element3 = tf.reduce_sum(
                    i_obj * y[:, :, :, :, :, 4:5],
                    axis=tf.range(1,tf.rank(y)-1)
                    )

    element4 = lambda_noobj * tf.reduce_sum(
                    i_noobj * tf.square(tf.reduce_max(y_pred_coord[:, :, :, :, :, 4:5], axis=-2)),
                    axis=tf.range(1,tf.rank(y_pred_coord)-2)
                    
                    )

    element5 = tf.reduce_sum( 
                    i * tf.reduce_sum(tf.square(y_pred_class - y_true_class), axis=-1, keepdims=True),
                    axis=tf.range(1,tf.rank(y_pred_class)-1)
                    )

    return element1 + element2 + element3 + element4 + element5
