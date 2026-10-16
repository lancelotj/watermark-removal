"""Local, TensorFlow 2 compatible replacements for the neuralgym ops used here.

The pretrained checkpoints were produced with TensorFlow 1.15 + neuralgym, so
this module preserves the original op semantics and, crucially, the original
variable names (e.g. ``conv1/kernel`` and ``conv1/bias``). Entrypoints must
select ``tensorflow.compat.v1`` before importing this module.
"""
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name="resize"):
    """Resize ``x`` by ``scale`` or to an explicit ``to_shape``."""
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1] * scale, tf.int32),
                  tf.cast(xs[2] * scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]], align_corners=align_corners)
    return x


def flatten(x, name="flatten"):
    """Flatten all dimensions but the batch dimension."""
    with tf.variable_scope(name):
        return tf.reshape(x, [tf.shape(x)[0], -1])


def kernel_spectral_norm(kernel, iteration=1, name="kernel_sn"):
    """Apply spectral normalisation to a convolution kernel."""
    def l2_norm(input_x, epsilon=1e-12):
        return input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)

    with tf.variable_scope(name):
        w_shape = kernel.get_shape().as_list()
        w_mat = tf.reshape(kernel, [-1, w_shape[-1]])
        u = tf.get_variable(
            "u", shape=[1, w_shape[-1]],
            initializer=tf.truncated_normal_initializer(),
            trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1

        u_hat, v_hat, _ = power_iteration(u, iteration)
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        w_mat = w_mat / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


class Conv2DSepctralNorm(tf.layers.Conv2D):
    """Conv2D whose kernel is spectrally normalised during ``build``."""

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = kernel_spectral_norm(self.kernel)


def conv2d_spectral_norm(
        inputs, filters, kernel_size, strides=(1, 1), padding="valid",
        data_format="channels_last", dilation_rate=(1, 1), activation=None,
        use_bias=True, kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, trainable=True,
        name=None, reuse=None):
    """Spectral-normalised 2D convolution (used by the discriminator)."""
    layer = Conv2DSepctralNorm(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, data_format=data_format, dilation_rate=dilation_rate,
        activation=activation, use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
        trainable=trainable, name=name, dtype=inputs.dtype.base_dtype,
        _reuse=reuse, _scope=name)
    return layer.apply(inputs)


def collection_to_dict(collection):
    """Build a name -> tensor mapping for a graph collection."""
    return {v.name[:v.name.rfind(":")]: v for v in collection}


def _get_sess(sess=None):
    if sess is None:
        sess = tf.get_default_session()
    assert sess, "sess should not be None."
    return sess


def scalar_summary(name, value, sess=None, summary_writer=None, step=None):
    """Add a scalar summary, routing constant values through a placeholder."""
    def is_tensor_or_var(value):
        return isinstance(value, (tf.Tensor, tf.Variable))

    scope_name = tf.get_variable_scope().name
    full_name = name if not scope_name else scope_name + "/" + name
    collection = collection_to_dict(tf.get_collection(tf.GraphKeys.SUMMARIES))
    if full_name in collection:
        if not is_tensor_or_var(value):
            ph_collection = collection_to_dict(
                tf.get_collection("SUMMARY_PLACEHOLDERS"))
            op_collection = collection_to_dict(
                tf.get_collection("SUMMARY_OPS"))
            ph = ph_collection[full_name + "_ph"]
            op = op_collection[full_name + "_op"]
            sess = _get_sess(sess)
            sess.run(op, feed_dict={ph: value})
        summary = collection[full_name]
    else:
        if not is_tensor_or_var(value):
            print("To write summary, create tensor for non-tensor value: "
                  "%s_var." % name)
            value = tf.Variable(value, name=name + "_var")
            ph = tf.placeholder(value.dtype, name=name + "_ph")
            op = tf.assign(value, ph, name=name + "_op")
            tf.add_to_collection("SUMMARY_PLACEHOLDERS", ph)
            tf.add_to_collection("SUMMARY_OPS", op)
            sess = _get_sess(sess)
            sess.run(tf.variables_initializer([value]))
        with tf.device("/cpu:0"):
            summary = tf.summary.scalar(name, value)
    if summary_writer is not None:
        assert step is not None, "step must not be None when writing a summary."
        sess = _get_sess(sess)
        summary_writer.add_summary(sess.run(summary), step)


def images_summary(images, name, max_outs, color_format="BGR"):
    """Summary images. Input should be scaled to [-1, 1] (or [0, 1] grey)."""
    with tf.variable_scope(name), tf.device("/cpu:0"):
        if color_format == "BGR":
            img = tf.clip_by_value(
                (tf.reverse(images, [-1]) + 1.) * 127.5, 0., 255.)
        elif color_format == "RGB":
            img = tf.clip_by_value((images + 1.) * 127.5, 0., 255.)
        elif color_format == "GREY":
            img = tf.clip_by_value(images * 255., 0., 255.)
        else:
            raise NotImplementedError("color format is not supported.")
        tf.summary.image(name, img, max_outputs=max_outs)


def gradients_summary(y, x, norm=tf.abs, name="gradients_y_wrt_x"):
    """Summarise the sum of ``norm(grad(y, x))``."""
    grad = tf.reduce_sum(norm(tf.gradients(y, x)))
    scalar_summary(name, grad)


def gan_hinge_loss(pos, neg, value=1., name="gan_hinge_loss"):
    """Hinge GAN loss used by the spectral-normalised discriminator."""
    with tf.variable_scope(name):
        hinge_pos = tf.reduce_mean(tf.nn.relu(1 - pos))
        hinge_neg = tf.reduce_mean(tf.nn.relu(1 + neg))
        scalar_summary("pos_hinge_avg", hinge_pos)
        scalar_summary("neg_hinge_avg", hinge_neg)
        d_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)
        g_loss = -tf.reduce_mean(neg)
        scalar_summary("d_loss", d_loss)
        scalar_summary("g_loss", g_loss)
    return g_loss, d_loss


def random_interpolates(x, y, alpha=None, dtype=tf.float32):
    """Linear interpolation between ``x`` and ``y`` (first dim is batch)."""
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1], dtype=dtype)
    interpolates = x + alpha * (y - x)
    return tf.reshape(interpolates, shape)
