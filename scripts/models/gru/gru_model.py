import tensorflow as tf


class GRURegressionModel(tf.keras.Model):
    def __init__(self, dropout_rate: float, hidden_dim: int, output_dim: int):
        super(GRURegressionModel, self).__init__()
        self.gru = tf.keras.layers.GRU(
            hidden_dim,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0.0,
            use_bias=True,
            unroll=False,
            reset_after=True,
            return_sequences=False,
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=False):
        x = self.gru(inputs, training=training)
        x = self.dropout(x, training=training)
        return self.dense(x)


class ListNetLoss(tf.keras.losses.Loss):
    def __init__(self, name="listnet_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        p_true = tf.nn.softmax(y_true)
        p_pred = tf.nn.softmax(y_pred)
        loss = -tf.reduce_sum(p_true * tf.math.log(p_pred + 1e-10), axis=-1)
        return tf.reduce_mean(loss)
