import os
import numpy as np
import shutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from learn.tf.log import LOG

ENABLE_ASSERTIONS=True

# MISC UTILITIES -------------------------------------------------------------------------------------------------------
def assert_shape(x, dims=-1, shape_dict={}, message=''):
    '''
    Checks the dimensions of a tensor, if ENABLE_ASSERTIONS = True (else a noop).
    :param x: input tensor
    :param dims: expected rank of a tensor (or -1 to skip check)
    :param shape_dict: dictionary from idx to size, where shape[idx] is asserted to be size
    :param message: message to attach to assertions
    :return: x
    '''
    if not ENABLE_ASSERTIONS:
        return x

    sh = tf.shape(x)
    assertions = []
    if dims > -1:
        assertions.append(tf.assert_equal(x.rank(), dims, message=message))
    for idx in shape_dict.keys():
        assertions.append(tf.assert_equal(sh[idx], shape_dict[idx], message=message))

    with tf.control_dependencies(assertions):
        x = tf.identity(x)
    return x


def evaluate_var_dict(sess, vars, input_dict):
    '''
    Runs a session to evaluate named variables in a dictionary and outputs a dictionary of values with same keys.
    :param sess: tensorflow Session object
    :param vars: dictionary of string : tensorflow node
    :param input_dict: input dictionary to pass to session
    :return:
    Dictionary of evaluated values.
    '''
    ignored_types = [list, dict, tuple]
    keys = [ k for k in vars.keys() if type(vars[k]) not in ignored_types]  # TODO: instead check if tensor
    vals = [ v for v in vars.values() if type(v) not in ignored_types]
    res = sess.run(vals, input_dict)
    res_dict = {}
    for i in range(len(keys)):
        res_dict[keys[i]] = res[i]
    return res_dict


def add_vars_summaries(variables=None, collections=None):
    # LOG.debug('Adding summaries')
    if variables is None:
        variables = tf.trainable_variables()
    for v in variables:
        add_var_summaries(v, v.name.replace(':', '_'), collections=collections)


def add_var_summaries(var, name, collections, types=['mean', 'std', 'max', 'min', 'hist']):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # LOG.debug('Adding summaries to var %s' % var.name)
    if ('mean' in types) or ('std' in types):
        with tf.name_scope('summaries'): #_%s' % var.name.replace('/', '__').replace(':', '__')):
            mean = tf.reduce_mean(var)
        if 'std' in types:
            with tf.name_scope('std'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    if 'mean' in types:
        tf.summary.scalar(name + '/mean', mean, collections=collections)
    if 'std' in types:
        tf.summary.scalar(name + '/stddev', stddev, collections=collections)
    if 'max' in types:
        tf.summary.scalar(name + '/max', tf.reduce_max(var), collections=collections)
    if 'min' in types:
        tf.summary.scalar(name + '/min', tf.reduce_min(var), collections=collections)
    if 'hist' in types:
        tf.summary.histogram(name + '/hist', var, collections=collections)


def add_grads_summaries(grads, collections=None):
    with tf.name_scope('grad_summaries'):
        for grad, var in grads:
            if grad is None:
                LOG.warning('Gradient for variable %s is None' % var.name)
            else:
                l2 = tf.nn.l2_loss(grad)
                tf.summary.histogram(var.name.replace(':', '_') + '/GRAD', grad, collections=collections)
                tf.summary.scalar(var.name.replace(':', '_') + '/GRAD_L2', l2, collections=collections)


def print_all_vars(scope=None, only_trainable=True):
    if only_trainable:
        vars = tf.trainable_variables(scope)
    else:
        vars = tf.global_variables(scope)

    for v in vars:
        LOG.info('%s: %s' % (str(v.name), str(v.value)))


def init_l2_reg(scope=None, exclude_biases=False):
    loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables(scope)
                       if not exclude_biases or 'bias' not in v.name])
    return loss


def get_clipped_grads(optimizer, loss, max_norm=10.0, exclude=None, include=None):
    excludes = exclude if exclude is not None else []
    includes = include if include is not None else [ '' ]
    def is_excluded(name):
        for key in excludes:
            if key in name:
                return True
        return False

    def is_included(name):
        for key in includes:
            if key in name:
                return True
            return False

    grads = optimizer.compute_gradients(loss)
    clipped_grads = []
    with tf.name_scope('grad_clip'):
        for g, v in grads:
            if g is not None:
                if is_included(v.name) and not is_excluded(v.name):
                    clipped_grads.append((tf.clip_by_norm(g, max_norm), v))
                    LOG.info('Including gradient for variable %s' % v.name)
                else:
                    LOG.info('NOT including gradient for variable %s' % v.name)
            else:
                LOG.warning('Gradient for variable %s is None' % v.name)
    train = optimizer.apply_gradients(clipped_grads)  # grads)
    return clipped_grads, train


# TRAINING HELPER ------------------------------------------------------------------------------------------------------
class TrainHelper(object):
    '''
    Helper class that manages logging, writing of summaries and saving model
    checkpoints.
    '''
    def __init__(self, run_dir, sess, overwrite=False,
                 log_vars = [],
                 save_summary_every_n=100,
                 save_model_every_n=200,
                 log_every_n=10):
        # Clear dir if requested
        if overwrite:
            if os.path.isdir(run_dir):
                print('Overwriting directory %s' % run_dir)
                shutil.rmtree(run_dir)
        # Create dir if DNE
        if not os.path.isdir(run_dir):
            print('Creating directory %s' % run_dir)
            os.makedirs(run_dir)

        # Iteration number
        self.step = tf.Variable(-1, name='global_step', trainable=False, dtype=tf.int32)
        self.increment_step = tf.assign(self.step, self.step + 1)

        # Paths
        self.run_dir = run_dir
        self.log_file = os.path.join(self.run_dir, 'log.txt')
        self.model_prefix = os.path.join(self.run_dir, 'model.ckpt')

        # Options
        self.log_vars = log_vars
        self.summary_interval = save_summary_every_n
        self.model_interval = save_model_every_n
        self.log_interval = log_every_n

        # Various savers
        self.sess = sess
        self.summaries = { 'train' : tf.summary.merge_all('train'),
                           'test' : tf.summary.merge_all('test') }
        self.summary_writer = tf.summary.FileWriter(self.run_dir, sess.graph)
        self.model_saver = tf.train.Saver()
        LOG.redirect_to_file(self.log_file)


    def restore_model(self):
        ''' If exists, restores model from file. '''
        if os.path.isfile(self.model_prefix + '.index'):
            self.model_saver.restore(self.sess, self.model_prefix)
            print('Model for iteration %d restored from prefix: %s' % (self.sess.run(self.step), self.model_prefix))
        else:
            print('No model with prefix: %s' % self.model_prefix)


    def process_iteration(self, train_input_dict, test_input_dicts=None, force=False):
        '''
        Processes a single iteration, saving summaries and checkpoints at
        intervals specified to the constructor; allows splitting test data into batches (e.g. if test set is too big).
        test_input_dicts: [ ("name", [ {inputdict}, {inputdict}]), ("name2", [ {inputdict} ]) ]
        force: forces summaries to be written
        '''
        i = self.sess.run(self.increment_step, {})
        if force or i % self.summary_interval == 0:
            self.__process_summaries(i, train_input_dict, 'train')
            if test_input_dicts is not None:
                for test_name,v in test_input_dicts:
                    # Get summary for the first batch of the test set
                    self.__process_summaries(i, v[0], test_name)
                # Also, use the first batch of the first test as the proxy for "test" in visualizations, etc
                self.__process_summaries(i, test_input_dicts[0][1][0], 'test')
            self.summary_writer.flush()
        if force or i % self.model_interval == 0:
            self.model_saver.save(self.sess, self.model_prefix)
            LOG.info('Saved model for iteration %d' % i)
            LOG.flush()
        if force or i % self.log_interval == 0:
            self.__process_log(i, train_input_dict, 'train')
            self.evaluate_test_batches(test_input_dicts)


    def __process_summaries(self, i, input_dict, name):
        if name not in self.summaries:
            self.summaries[name] = tf.summary.merge_all(name)
        summ = self.sess.run(self.summaries[name], input_dict)
        self.summary_writer.add_summary(summ, i)


    def evaluate_test_batches(self, test_input_dicts):
        for test_name, v in test_input_dicts:
            all_vals = []
            for input_dict in v:  # process all the batches of the testset
                vals = self.sess.run(self.log_vars, input_dict)
                all_vals.append(vals)
            all_vals = np.array(all_vals)
            for i in range(len(self.log_vars)):  # Log result for every var
                var = self.log_vars[i]
                vals = all_vals[:, i]  # ith column is all the batch's values for var
                var_values_str = ' '.join([ ('%0.4f' % x) for x in vals ])
                LOG.info('TESTSETLOG %s %d %s: %s' % (test_name, i, var.name, var_values_str))


    def evaluate_all_files(self, data_readers, make_input_dict_func, ofile):
        ofile.write('TESTSET FILENAME %s\n' % ' '.join([v.name for v in self.log_vars]))
        for name, dr in data_readers.iteritems():
            for i in range(len(dr.filenames)):
                orig_fname = dr.orig_filenames[i]
                data = dr.get_specific_batch([i])
                input_dict = make_input_dict_func(data)
                vals = self.sess.run(self.log_vars, input_dict)
                ofile.write('%s %s %s\n' % (name, orig_fname, ' '.join([('%0.6f' % v) for v in vals])))
                ofile.flush()


    def __process_log(self, i, input_dict, name):
        if self.log_vars:
            vals = self.sess.run(self.log_vars, input_dict)
            LOG.info('Ep %d (%s): %s' % (i, name, ', '.join([str(x) for x in vals])))
            LOG.flush()
