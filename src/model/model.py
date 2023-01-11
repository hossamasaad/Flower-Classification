# imports
import tensorflow as tf


def resnet_encoder(inputs, freeze=True):
    """
    Resnet50 encoder to be used as feature extractor
    :param inputs: inputs
    :param freeze: freeze layers or not
    :return: Resnet50 encoder
    """
    resnet = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # freeze layers
    if freeze:
        for layer in resnet.layers:
            layer.trainable = False

    return resnet(inputs)


def classifier(inputs, units, dropout_rate):
    """
    Classification layers to be added to the encoder
    :param inputs: inputs
    :param units: number of Dense layer units
    :param dropout_rate: Dropout rate
    :return: classification layers
    """
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(102, activation="softmax", name="classification")(x)
    return x


def final_model(encoder, inputs, units, dropout_rate, freeze):
    """
    Combine encoder and classifier
    :param encoder: encoder type
    :param inputs: inputs
    :param units: number of units for Dense layer
    :param dropout_rate: Dropout layer
    :param freeze: freeze encoder layers or not
    :return: final model
    """
    encoders = {
        'resnet': resnet_encoder,
        'vgg': vgg_encoder,
        'inception': inception_encoder,
        'mobilenet': mobilenet_encoder
    }

    my_encoder = encoders[encoder]

    x = my_encoder(inputs=inputs, freeze=freeze)
    x = classifier(x, units=units, dropout_rate=dropout_rate)
    return x


def define_compile_model(encoder='resnet', optimizer='Adam', units=128, dropout_rate=0, freeze=True):
    """
    Dfine model and compile it
    :param encoder: encoder
    :param optimizer: optimizer
    :param units: number of units for Dense layer
    :param dropout_rate: dropout layer rate
    :param freeze: Freeze encoder layers or not
    :return: return final model
    """
    # inputs and outputs
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    output = final_model(encoder, inputs, units, dropout_rate, freeze=freeze)

    # create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model