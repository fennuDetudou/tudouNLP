import tensorflow as tf

flags=tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('input_file',default=None,help="input model file")
flags.DEFINE_string('output_file',None,help="output model file")

sess = tf.Session()
imported_meta = tf.train.import_meta_graph('{}.meta'.format(FLAGS.input_file))
imported_meta.restore(sess, FLAGS.input_file)
my_vars = []
for var in tf.global_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
        my_vars.append(var)
saver = tf.train.Saver(my_vars)
saver.save(sess, FLAGS.output_file)
