#多层感知机
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#定义算法公式。
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
sess = tf.InteractiveSession()
in_units = 784
hl_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units, hl_units], stddev = 0.1))
b1 = tf.Variable(tf.zeros([hl_units]))
w2 = tf.Variable(tf.zeros([hl_units, 10]))
b2 = tf.Variable([tf.zeros([10])])
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
hiddenl = tf.nn.relu(tf.matmul(x, w1) + b1)
hiddenl_drop = tf.nn.dropout(hiddenl, keep_prob)
y = tf.nn.softmax(tf.matmul(hiddenl_drop, w2) + b2)
y_ = tf.placeholder(tf.float32, [None, 10])
#定义损失函数，以及优化算法。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#训练网络
tf.global_variables_initializer().run()
for i in range(3000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x:batch_xs, y_: batch_ys, keep_prob:0.75})
#准确率测评。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(str(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})))
