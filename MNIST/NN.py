import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

epoches = 1000000 # 迭代次数
batch_size = 64 # 批大小
learning_rate = 0.001 # 学习率
smooth = 0.1
alpha = 0.01

image_dim = 784 # 28 * 28
g_hidden_dim = 128
d_hidden_dim = 128

noise_size = 100
sample_size = mnist.train.images[0].shape[0]

sess = tf.InteractiveSession()


d_w1 = tf.Variable(tf.truncated_normal([image_dim, d_hidden_dim], stddev=0.1))
d_b1 = tf.Variable(tf.zeros([d_hidden_dim]))
d_w2 = tf.Variable(tf.truncated_normal([d_hidden_dim, 1], stddev=0.1))
d_b2 = tf.Variable(tf.zeros([1]))

g_w1 = tf.Variable(tf.truncated_normal([noise_size, g_hidden_dim], stddev=0.1))  # 标准差0.1
g_b1 = tf.Variable(tf.zeros([g_hidden_dim]))
g_w2 = tf.Variable(tf.truncated_normal([g_hidden_dim, image_dim], stddev=0.1))  # 标准差0.1
g_b2 = tf.Variable(tf.zeros([image_dim]))

d_vars = [d_w1, d_b1, d_w2, d_b2]
g_vars = [g_w1, g_b1, g_w2, g_b2]

g_input = tf.placeholder(tf.float32, shape=[None, noise_size], name='g_input')
d_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='d_input')

def discriminator(images):
    hidden_layer = tf.matmul(images, d_w1) + d_b1
    hidden_layer = tf.maximum(alpha * hidden_layer, hidden_layer)
    output_layer = tf.matmul(hidden_layer, d_w2) + d_b2
    return output_layer

def generator(images):
    hidden_layer = tf.matmul(images, g_w1) + g_b1
    hidden_layer = tf.maximum(alpha * hidden_layer, hidden_layer)
    output_layer = tf.tanh(tf.matmul(hidden_layer, g_w2) + g_b2)
    return output_layer

#============================================================
#Generator
g_sample = generator(g_input)
#Discriminator
d_real = discriminator(d_input)
d_fake = discriminator(g_sample)
#============================================================
#Loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)) * (1 - smooth))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)) * (1 - smooth))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

d_loss = tf.add(d_loss_real, d_loss_fake)
#=============================================================
#Optimizer
train_vars = tf.trainable_variables()
# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list = d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = g_vars)
#=============================================================
plt_loss_d = []
plt_loss_d_real = []
plt_loss_d_fake = []
plt_loss_g = []
samples = []

sess.run(tf.global_variables_initializer())
for i in range(1, epoches+1):
    batch_x, _ = mnist.train.next_batch(batch_size)
    batch_x = batch_x * 2 - 1
    batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
    _ = sess.run(d_train_opt, feed_dict={d_input:batch_x, g_input:batch_noise})
    _ = sess.run(g_train_opt, feed_dict={g_input:batch_noise})

    if i % 10000 == 0:

        # 每一轮结束计算loss
        train_loss_d = sess.run(d_loss, feed_dict={d_input: batch_x, g_input:batch_noise})
        # real img loss
        train_loss_d_real = sess.run(d_loss_real, feed_dict={d_input: batch_x,g_input: batch_noise})
        # fake img loss
        train_loss_d_fake = sess.run(d_loss_fake, feed_dict={d_input: batch_x,g_input: batch_noise})
        # generator loss
        train_loss_g = sess.run(g_loss, feed_dict={g_input: batch_noise})

        #save the loss
        plt_loss_d.append(train_loss_d)
        plt_loss_d_real.append(train_loss_d_real)
        plt_loss_d_fake.append(train_loss_d_fake)
        plt_loss_g.append(train_loss_g)

        print("Epoch {}/{}...".format(i, epoches),
            "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real,train_loss_d_fake),
                "Generator Loss: {:.4f}".format(train_loss_g))
        #=================================================================================
    if i % 200000 == 0:
        sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
        gen_samples = sess.run(g_sample, feed_dict={g_input: sample_noise})
        samples.append(gen_samples)


print("=========Training End!=============")

def draw_loss():
   fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plt_loss_d, label='Discriminator Total Loss')
    ax.plot(plt_loss_d_real, label='Discriminator Real Loss')
    ax.plot(plt_loss_d_fake, label='Discriminator Fake Loss')
    ax.plot(plt_loss_g, label='Generator')

    ax.legend(loc='best')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    #ax.set_ylim(0, 3)
    ax.set_title("GAN_MNIST")
    plt.show()

epoch_idx = [0, 1, 2, 3, 4]  # 一共300轮，不要越界
show_imgs = []
for i in epoch_idx:
    show_imgs.append(samples[i])
# 指定图片形状
rows, cols = 5, 25
fig, axes = plt.subplots(figsize=(30, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(show_imgs, axes):
    for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()

draw_loss()