from threading import Thread

from matplotlib import pyplot as plt
import numpy as np
import tempfile
import tensorflow as tf
import matplotlib as mpl
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from gtts import gTTS
from pygame import mixer
import sys
import warnings
import logging
from PIL import Image
import pickle
from hanziconv import HanziConv

from IPython.display import Audio  # Import Audio method from IPython's Display Class

tf.enable_eager_execution()

# mpl.use('TkAgg')
mpl.rcParams['font.family'] = ['Microsoft JhengHei']
mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input

hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# image_features_extract_model.summary()


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


BATCH_SIZE = 64
BUFFER_SIZE = 1000
max_length = 39
embedding_dim = 256
units = 512
vocab_size = 5001
features_shape = 512
attention_features_shape = 64


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.train.AdamOptimizer()


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2 + 1, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


#
# def image_plot(image):
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg
#     img = mpimg.imread(image)
#     imgplot = plt.imshow(img)
#     plt.show()


# def voice(res):
#     tts = gTTS(' '.join(res).rsplit(' ', 1)[0])
#     tts.save('1.mp3')
#     sound_file = './1.mp3'
#     return Audio(sound_file, autoplay=True)


def predict(train_captions_path, checkpoint_path, Image_path):
    dbfile = open(train_captions_path, 'rb')
    train_captions = pickle.load(dbfile)
    dbfile.close()

    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # restoring the model
    checkpoint_path = checkpoint_path
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt.restore(checkpoint_path)

    # attention_plot = None
    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                attention_plot = attention_plot[:len(result), :]

                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]

        return result, attention_plot

    new_img = Image_path

    result, my_att_plt = evaluate(new_img)
    # Thread(plot_attention, args=(new_img, result, my_att_plt))
    plot_attention(new_img, result, my_att_plt)
    for i in result:
        if i == "<unk>":
            result.remove(i)
        else:
            pass

    # plot_attention(new_img, result, attention_plot)

    # print('I guess: ', ' '.join(result).rsplit(' ', 1)[0])

    # image_plot(new_img)

    # real_caption = ' '.join(result).rsplit(' ', 1)[0]
    #

    # tts = gTTS(' '.join(result).rsplit(' ', 1)[0])
    # tts.save('1.mp3')
    # sound_file = './1.mp3'
    return ''.join(result)[:-5]


def speak(text):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=text, lang='zh-tw')
        tts.save('{}.mp3'.format(fp.name))
        mixer.init()
        mixer.music.load('{}.mp3'.format(fp.name))
        mixer.music.play()


def retranslateUi(Window):
    _translate = QtCore.QCoreApplication.translate
    Window.setWindowTitle(_translate("Window", "Chinese Image Caption"))
    # win.open_image_button.setText(_translate("Window", "開啟圖片"))
    win.generate_caption_button.setText(_translate("Window", "生成句子"))


def open_img():
    win.dlg = QFileDialog()
    win.dlg.setNameFilter("Images (*.png *.jpg)")
    # win.dlg.getOpenFileName(win)
    # print(win.dlg)
    # self.dlg.selectNameFilter("Images (*.png *.jpg)")
    # self.img = self.dlg.getOpenFileName()
    if win.dlg.exec_():
        win.img = win.dlg.selectedFiles()
        win.img_path = win.img[0]
        win.img = QtGui.QPixmap(win.img[0])
        win.scaled_img = win.img.scaled(800, 370, QtCore.Qt.KeepAspectRatio)
        win.show_picture.setPixmap(QtGui.QPixmap(win.scaled_img))
        win.img_shown = 1


def generate():
    if win.img_shown is 0:
        win.textBrowser.setText("請先開啟圖片")
        win.textBrowser.setFont(QtGui.QFont("Noto Sans Mono CJK TC", 17))
    else:
        win.textBrowser.setText('請稍等...')
        win.textBrowser.setFont(QtGui.QFont("Noto Sans Mono CJK TC", 17))
        predicted_cap = HanziConv.toTraditional(predict('./train_captions', "./ckpt-20", win.img_path))
        win.textBrowser.setText(predicted_cap)
        # win.textBrowser.setFont(win.def_font)
        speak(predicted_cap)


class ExtendedQLabel(QLabel):
    def __init__(self, parent):
        QLabel.__init__(self, parent)

    def mouseReleaseEvent(self, ev):
        open_img()


palette = QtGui.QPalette()
palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Base, QtGui.QColor(80, 80, 80))
palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)

palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197).lighter())
palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

app = QApplication([])
app.setStyle('Fusion')
app.setPalette(palette)
# win = Ui_Window()
win = QMainWindow()
win.img_path = ''

win.img_shown = 0
def_font = QtGui.QFont("Noto Sans Mono CJK TC", 12)
def_font.setItalic(True)

win.setObjectName("Window")
win.setEnabled(True)
win.resize(800, 500)
win.setFixedSize(QtCore.QSize(800, 500))
win.setInputMethodHints(QtCore.Qt.ImhNone)

win.show_picture = ExtendedQLabel(win)
win.show_picture.setGeometry(QtCore.QRect(0, 8, 800, 370))
win.show_picture.setText("點擊開啟圖片")
win.show_picture.setObjectName("show_picture")
win.show_picture.setFont(def_font)
win.show_picture.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter)

win.verticalLayoutWidget = QtWidgets.QWidget(win)
win.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 390, 781, 106))
win.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
win.verticalLayout = QtWidgets.QVBoxLayout(win.verticalLayoutWidget)
win.verticalLayout.setContentsMargins(0, 0, 0, 0)
win.verticalLayout.setObjectName("verticalLayout")

win.line = QtWidgets.QFrame(win.verticalLayoutWidget)
win.line.setFrameShape(QtWidgets.QFrame.HLine)
win.line.setFrameShadow(QtWidgets.QFrame.Sunken)
win.line.setObjectName("line")

win.verticalLayout.addWidget(win.line)

# win.open_image_button = QtWidgets.QPushButton(win.verticalLayoutWidget)
# win.open_image_button.setObjectName("open_image_button")
# win.open_image_button.setFont(win.def_font)
# win.verticalLayout.addWidget(win.open_image_button)

win.horizontalLayout_4 = QtWidgets.QHBoxLayout()
win.horizontalLayout_4.setObjectName("horizontalLayout_4")

# win.def_font = QtGui.QFont("Noto Sans Mono CJK TC", 10)

win.generate_caption_button = QtWidgets.QPushButton(win.verticalLayoutWidget)
win.generate_caption_button.setObjectName("generate_caption_button")
win.generate_caption_button.setFont(def_font)
win.horizontalLayout_4.addWidget(win.generate_caption_button)

win.textBrowser = QtWidgets.QTextBrowser(win.verticalLayoutWidget)
win.textBrowser.setObjectName("textBrowser")

win.horizontalLayout_4.addWidget(win.textBrowser)
win.verticalLayout.addLayout(win.horizontalLayout_4)

retranslateUi(win)
QtCore.QMetaObject.connectSlotsByName(win)

# win.open_image_button.clicked.connect(open_img)
win.generate_caption_button.clicked.connect(generate)

win.show()
sys.exit(app.exec_())
