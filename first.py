import tensorflow as tf

x = tf.constant([35,40,45],name='x')
y = tf.Variable(x+20,name='y')

#model = tf.global_variables_initializer()

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/basic", session.graph)
    model =  tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))

#print(y)
