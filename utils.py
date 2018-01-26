import tensorflow as tf

def compute_PSNR1(outputs, labels):
    """ compute PSNR of outpus and labels
    args: 
    - outputs: tensor of shape (N, H, W, C)
    - labels: tensor of shape (N, H, W, C)
    return: average PSNR
    """

    N = tensor.shape(outputs)[0]
    diff = outputs - labels
    diff = tf.reshape(diff, [N, -1])
    rmse = tf.sqrt(tf.reduce_mean(diff**2, axis=1))
    psnr = 20*tf.log(255/rmse)/tf.log(10)
    psnr = tf.reduce_mean(psnr)
    return psnr

def clean_and_create_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)

        
