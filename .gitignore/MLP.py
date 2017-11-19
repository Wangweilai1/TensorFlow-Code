# 多层感知机
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
x = tf.placeholder(tf.float32, [None, in_units], name = 'input_x')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
hiddenl = tf.nn.relu(tf.matmul(x, w1) + b1)
hiddenl_drop = tf.nn.dropout(hiddenl, keep_prob)
y = tf.nn.softmax(tf.matmul(hiddenl_drop, w2) + b2, name = 'output')
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
'''
#加载模型，并输出测试结果.
ckpt = tf.train.get_checkpoint_state('/tmp/')
saver = tf.train.Saver()
#saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta') #加载Saver模型对象(可不加载)。
saver.restore(sess, ckpt.model_checkpoint_path)

result = sess.run(y, feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
res = sess.run(tf.argmax(result, 1))
for i in res:
	print(i)
'''	

#保存模型结果.
#方法1. 保存*.pb文件，用于Android平台(需要事先把文件手动建立出来，然后才能输入到文件里面).
#参考文件： http://blog.csdn.net/cxq234843654/article/details/71171293
#output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
#with tf.gfile.FastGFile('/model/cxq.pb', mode='wb') as f:
    #f.write(output_graph_def.SerializeToString())

#方法2. 保存model.ckpt文件，用于PC端(需要事先把tmp文件夹手动建立出来，然后才能输入到文件里面).
#参考文件： https://www.cnblogs.com/hellcat/p/6925757.html
saver = tf.train.Saver()
tf.add_to_collection('pred_network', y)
save_path = saver.save(sess, "/tmp/model.ckpt")
sess.close()


'''
#不需重新定义网络结构的方法从而载入模型.
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#定义算法公式。
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
	new_saver.restore(sess, '/tmp/model.ckpt')
	# tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
	y = tf.get_collection('pred_network')[0]

	graph = tf.get_default_graph()

	# 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
	input_x = graph.get_operation_by_name('input_x').outputs[0]
	keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

	# 使用y进行预测  
	result = sess.run(y, feed_dict={input_x:mnist.test.images,  keep_prob:1.0})
	res = sess.run(tf.argmax(result, 1))
	for i in res:
		print(i)
'''
