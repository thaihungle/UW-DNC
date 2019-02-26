import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from memory import Memory
import utility
import os

class DNC:

    def __init__(self, controller_class, input_encoder_size, input_decoder_size, output_size,
                 memory_words_num = 256, memory_word_size = 64, memory_read_heads = 1,
                 batch_size = 1,hidden_controller_dim=256, use_emb_encoder=True,
                 use_emb_decoder=True, train_emb=True,
                 use_mem=True, decoder_mode=False, emb_size=64, parallel_rnn=1,
                 write_protect=False,  hold_mem_mode=0,
                 dual_controller=False, dual_emb=True, controller_cell_type="lstm",
                 use_teacher=False, cache_attend_dim=0,
                 use_encoder_output=False, clip_output=0,
                 pass_encoder_state=True,
                 memory_read_heads_decode=None, enable_drop_out=False,
                 enable_rnn_drop_out=False, batch_norm=False,
                 nlayer=1, name='UW'):
        """
        constructs a complete DNC architecture as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html
        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        memory_words_num: int
            the number of words that can be stored in memory
        memory_word_size: int
            the size of an individual word in memory
        memory_read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        saved_args = locals()
        print("saved_args is", saved_args)
        self.name=name
        self.parallel_rnn=parallel_rnn
        self.input_encoder_size = input_encoder_size
        self.input_decoder_size = input_decoder_size
        self.output_size = output_size
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_norm=batch_norm
        self.clip_output=clip_output

        if memory_read_heads_decode is None:
            self.read_heads_decode = memory_read_heads
        else:
            self.read_heads_decode = memory_read_heads_decode


        self.batch_size = batch_size
        self.unpacked_input_encoder_data = None
        self.unpacked_input_decoder_data = None
        self.packed_output = None
        self.packed_output_encoder = None
        self.packed_memory_view_encoder = None
        self.packed_memory_view_decoder = None
        self.decoder_mode = decoder_mode
        self.emb_size = emb_size
        self.emb_size2 = emb_size
        self.dual_emb = dual_emb
        self.use_mem = use_mem
        self.controller_cell_type = controller_cell_type
        self.use_emb_encoder = use_emb_encoder
        self.use_emb_decoder = use_emb_decoder
        self.hidden_controller_dim = hidden_controller_dim
        self.cache_attend_dim = cache_attend_dim
        self.use_teacher = use_teacher
        self.teacher_force = tf.placeholder(tf.bool,[None], name='teacher')
        self.hold_mem_mode=hold_mem_mode
        if self.hold_mem_mode>0:
            self.hold_mem = tf.placeholder(tf.bool, [None], name='hold_mem')
        else:
            self.hold_mem=None

        self.use_encoder_output=use_encoder_output
        self.pass_encoder_state=pass_encoder_state
        self.clear_mem = tf.placeholder(tf.bool,None, name='clear_mem')
        self.drop_out_keep = tf.placeholder_with_default(1.0, None, name='drop_out_keep')
        self.drop_out_rnn_keep = tf.placeholder_with_default(1.0, None, name='drop_out_rnn_keep')

        self.nlayer=nlayer
        self.drop_out_v = 1
        self.drop_out_rnnv = 1
        if enable_drop_out:
            self.drop_out_v = self.drop_out_keep

        if enable_rnn_drop_out:
            self.drop_out_rnnv = self.drop_out_rnn_keep



        self.controller_out = self.output_size



        if self.use_emb_encoder is False:
            self.emb_size=input_encoder_size

        if self.use_emb_decoder is False:
            self.emb_size2=input_decoder_size #pointer mode not use


        if self.cache_attend_dim>0:


            self.cW_a = tf.get_variable('cW_a', [self.hidden_controller_dim, self.cache_attend_dim],
                                      initializer=tf.random_normal_initializer(stddev=0.1))

            value_size = self.hidden_controller_dim

            self.cU_a = tf.get_variable('cU_a', [value_size, self.cache_attend_dim],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
            if self.use_mem:
                self.cV_a = tf.get_variable('cV_a', [self.read_heads*self.word_size, self.cache_attend_dim],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            self.cv_a = tf.get_variable('cv_a', [self.cache_attend_dim],
                                  initializer=tf.random_normal_initializer(stddev=0.1))

        # DNC (or NTM) should be structurized into 2 main modules:
        # all the graph is setup inside these twos:
        self.W_emb_encoder = tf.get_variable('embe_w', [self.input_encoder_size, self.emb_size], trainable=train_emb,
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        if self.dual_emb:
            self.W_emb_decoder = tf.get_variable('embd_w', [self.output_size, self.emb_size2],trainable=train_emb,
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        with tf.variable_scope('controller_scope'):
            self.controller = controller_class(self.emb_size, self.controller_out, self.read_heads,
                                               self.word_size, self.batch_size, use_mem,
                                               cell_type=controller_cell_type, batch_norm=batch_norm,
                                               hidden_dim=hidden_controller_dim, nlayer=nlayer,
                                               drop_out_keep=self.drop_out_rnnv, clip_output=self.clip_output)

        self.dual_controller = dual_controller
        if self.dual_controller:
            with tf.variable_scope('controller2_scope'):
                if use_mem:
                    self.controller2 = controller_class(self.emb_size2, self.controller_out, self.read_heads_decode,
                                                       self.word_size, self.batch_size, use_mem,
                                                        cell_type=controller_cell_type, batch_norm=batch_norm, clip_output=self.clip_output,
                                                        hidden_dim=hidden_controller_dim, drop_out_keep=self.drop_out_rnnv, nlayer=nlayer)
                else:

                    self.controller2 = controller_class(self.emb_size2+hidden_controller_dim, self.controller_out, self.read_heads_decode,
                                                        self.word_size, self.batch_size, use_mem,
                                                        cell_type=controller_cell_type, batch_norm=batch_norm, clip_output=self.clip_output,
                                                        hidden_dim=hidden_controller_dim, drop_out_keep=self.drop_out_rnnv, nlayer=nlayer)
        self.write_protect = write_protect


        # input data placeholders

        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')

        self.input_encoder = tf.placeholder(tf.float32, [batch_size, None, input_encoder_size], name='input_encoder')

        self.input_decoder = tf.placeholder(tf.float32, [batch_size, None, input_decoder_size], name='input_decoder')

        self.mask = tf.placeholder(tf.bool, [batch_size, None], name='mask')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')# variant length?
        self.decode_length = tf.placeholder(tf.int32, name='decode_length')  # variant length?


        self.build_graph()



    # The nature of DNC is to process data by step and remmeber data at each time step when necessary
    # If input has sequence format --> suitable with RNN core controller --> each time step in RNN equals 1 time step in DNC
    # or just feed input to MLP --> each feed is 1 time step
    def _step_op_encoder(self,time, time2, step, memory_state, controller_state=None, cache_controller_hidden=None):
        """
        performs a step operation on the input step data
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6] # read values from memory
        last_read_weights = memory_state[5]
        pre_output, interface, nn_state = None, None, None
        var = {"time2": time2, "step": step, "compute_interface":True}
        cache_controller_hidden = cache_controller_hidden.write(time, controller_state)

        def let_compute():
            var["compute_interface"] = True
            ns2 = controller_state

            if self.cache_attend_dim > 0:
                # values = utility.pack_into_tensor(cache_controller_hidden, axis=1)
                values = cache_controller_hidden.gather(tf.range(time-time2, time+1))

                value_size = self.hidden_controller_dim

                encoder_outputs = \
                    tf.reshape(values, [self.batch_size, -1, value_size])  # bs x Lin x h
                v = tf.reshape(tf.matmul(tf.reshape(encoder_outputs, [-1, value_size]), self.cU_a),
                               [self.batch_size, -1, self.cache_attend_dim])

                if self.use_mem:
                    v += tf.reshape(
                        tf.matmul(tf.reshape(last_read_vectors, [-1, self.read_heads * self.word_size]),
                                  self.cV_a),
                        [self.batch_size, 1, self.cache_attend_dim])
                ns, statetype = self.get_hidden_value_from_state(controller_state)
                print("state typeppppp")
                print(controller_state)
                print(ns)
                v += tf.reshape(
                    tf.matmul(tf.reshape(ns, [-1, self.hidden_controller_dim]), self.cW_a),
                    [self.batch_size, 1, self.cache_attend_dim])  # bs.Lin x h_att
                print('state include only h')

                v = tf.reshape(tf.tanh(v), [-1, self.cache_attend_dim])
                eijs = tf.matmul(v, tf.expand_dims(self.cv_a, 1))  # bs.Lin x 1
                eijs = tf.reshape(eijs, [self.batch_size, -1])  # bs x Lin
                alphas = tf.nn.softmax(eijs)

                att = tf.reduce_sum(encoder_outputs * tf.expand_dims(alphas, 2), 1)  # bs x h x 1
                att = tf.reshape(att, [self.batch_size, value_size])  # bs x h
                # step = tf.concat([var["step"], att], axis=-1)  # bs x (encoder_input_size + h)
                # step = tf.matmul(step, self.cW_ah) # bs x encoder_input_size (or emb_size)
                if statetype==1:
                    ns2=list(controller_state)
                    ns2[-1][-1]=att
                    ns2=tuple(ns2)
                elif statetype==2 or statetype==3:
                    # ns2 = list(controller_state)
                    ns2 = LSTMStateTuple(controller_state[0],att)
                    # ns2 = tuple(ns2)
                elif statetype==4:
                    return att

            return ns2


        def hold_compute():
            var["compute_interface"] =False
            return controller_state

        if self.hold_mem_mode>0:
            controller_state = tf.cond(self.hold_mem[time], hold_compute, let_compute)

            #controller_state = let_compute()
        # compute oututs from controller

        if self.controller.has_recurrent_nn:
            compute_interface = var["compute_interface"]
            # controller state is the rnn cell state pass through each time step
            if not self.use_emb_encoder:
                step2 = tf.reshape(step, [-1, self.input_encoder_size])
                pre_output, interface, nn_state=  self.controller.process_input(step2, last_read_vectors,
                                                                                controller_state,
                                                                                compute_interface=compute_interface)
            else:
                pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors,
                                                                                controller_state,
                                                                                compute_interface=compute_interface)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first

        var["state"]=nn_state

        if self.hold_mem_mode >0:
            def hold_write():
                var["time2"] += 1
                if self.hold_mem_mode >2:
                    return self.memory.write(
                        memory_state[0], memory_state[1], memory_state[5],
                        memory_state[4], memory_state[2], memory_state[3],
                        interface['write_key'],
                        interface['write_strength'],
                        interface['free_gates'],
                        interface['allocation_gate'],
                        interface['write_gate'],
                        interface['write_vector'],
                        interface['erase_vector']
                    )
                else:
                    return memory_state[1], memory_state[4], memory_state[0], memory_state[3], memory_state[2]

            def let_write():
                # interface["write_gate"] = (1 - self.max_lambda**tf.cast(var["time2"],tf.float32))
                var["time2"] = 0
                # var["state"] = self.controller.zero_state()



                return  self.memory.write(
                    memory_state[0], memory_state[1], memory_state[5],
                    memory_state[4], memory_state[2], memory_state[3],
                    interface['write_key'],
                    interface['write_strength'],
                    interface['free_gates'],
                    interface['allocation_gate'],
                    interface['write_gate'],
                    interface['write_vector'],
                    interface['erase_vector']
                    )

            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector=\
                tf.cond(self.hold_mem[time], hold_write, let_write)
        else:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector= \
                self.memory.write(
                    memory_state[0], memory_state[1], memory_state[5],
                    memory_state[4], memory_state[2], memory_state[3],
                    interface['write_key'],
                    interface['write_strength'],
                    interface['free_gates'],
                    interface['allocation_gate'],
                    interface['write_gate'],
                    interface['write_vector'],
                    interface['erase_vector']
                )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading
        if self.hold_mem_mode>1:
            def hold_read():
                return last_read_weights, last_read_vectors

            def let_read():
                return self.memory.read(
                    memory_matrix,
                    memory_state[5],
                    interface['read_keys'],
                    interface['read_strengths'],
                    link_matrix,
                    interface['read_modes'],
                )

            read_weightings, read_vectors = tf.cond(self.hold_mem[time], hold_read, let_read)

        else:

            read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
            )
        fout=None
        if self.use_encoder_output:
            fout = self.controller.final_output(pre_output, read_vectors)

            if self.clip_output>0:
                fout = tf.clip_by_value(fout, -self.clip_output, self.clip_output)

        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix, #0

            # neccesary for next step to compute memory stuffs
            usage_vector, #1
            precedence_vector, #2
            link_matrix, #3
            write_weighting, #4
            read_weightings, #5
            read_vectors, #6

            # the final output of dnc
            fout, #7

            # the values public info to outside
            interface['read_modes'], #8
            interface['allocation_gate'], #9
            interface['write_gate'], #10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state if nn_state is not None else tf.zeros(1), #11
            var["time2"], #12
            cache_controller_hidden #13
        ]

    def _step_op_decoder(self, time, step, memory_state,
                         controller_state=None, controller_hiddens=None):
        """
        performs a step operation on the input step data
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_weights = memory_state[5]
        last_read_vectors = memory_state[6]  # read values from memory
        pre_output, interface, nn_state = None, None, None

        if self.dual_controller:
            controller=self.controller2
        else:
            controller=self.controller

        # compute outputs from controller
        if controller.has_recurrent_nn:
            if not self.use_emb_decoder:
                step2 = tf.reshape(step, [-1, self.output_size])
            else:
                step2 = step
            pre_output, interface, nn_state = controller.process_input(step2, last_read_vectors, controller_state)

        else:
            pre_output, interface = controller.process_input(step, last_read_vectors)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first
        if self.write_protect:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector \
                =memory_state[1], memory_state[4], memory_state[0], memory_state[3], memory_state[2]

        else:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
                memory_state[0], memory_state[1], memory_state[5],
                memory_state[4], memory_state[2], memory_state[3],
                interface['write_key'],
                interface['write_strength'],
                interface['free_gates'],
                interface['allocation_gate'],
                interface['write_gate'],
                interface['write_vector'],
                interface['erase_vector']
            )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading


        read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
            )
        fout = controller.final_output(pre_output, read_vectors) # bs x output_size

        if self.clip_output>0:
            fout = tf.clip_by_value(fout, -self.clip_output, self.clip_output)

        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix,  # 0

            # neccesary for next step to compute memory stuffs
            usage_vector,  # 1
            precedence_vector,  # 2
            link_matrix,  # 3
            write_weighting,  # 4
            read_weightings,  # 5
            read_vectors,  # 6

            # the final output of dnc
            fout,  # 7

            # the values public info to outside
            interface['read_modes'],  # 8
            interface['allocation_gate'],  # 9
            interface['write_gate'],  # 10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state if nn_state is not None else tf.zeros(1),  # 11
        ]

    '''
    THIS WRAPPER FOR ONE STEP OF COMPUTATION --> INTERFACE FOR SCAN/WHILE LOOP
    '''
    def _loop_body_encoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                   read_weightings, write_weightings, usage_vectors, controller_state,
                   outputs_cache, controller_hiddens, time2, cache_controller_hiddens):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input

        if self.use_emb_encoder:
            step_input = tf.matmul(self.unpacked_input_encoder_data.read(time), self.W_emb_encoder)
        else:
            step_input = self.unpacked_input_encoder_data.read(time)

        # compute one step of controller
        op = self._step_op_encoder

        output_list = op(time, time2, step_input, memory_state, controller_state, cache_controller_hiddens)
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11] #state  hidden values
        hstate, _ = self.get_hidden_value_from_state(new_controller_state)

        controller_hiddens = controller_hiddens.write(time, hstate)

        if self.use_encoder_output:
            outputs = outputs.write(time, output_list[7])# new output is updated
            outputs_cache = outputs_cache.write(time, output_list[7])# new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1, #0
            new_memory_state, #1
            outputs, #2
            free_gates,allocation_gates, write_gates, #3 4 5
            read_weightings, write_weightings, usage_vectors, #6 7 8
            new_controller_state, #9
            outputs_cache,  #10
            controller_hiddens, #11
            output_list[-2], #12
            output_list[-1], #13
        )

    def _loop_body_decoder(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                           read_weightings, write_weightings, usage_vectors, controller_state,
                           outputs_cache, controller_hiddens,
                           encoder_write_weightings, encoder_controller_hiddens):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input
        if self.decoder_mode:
            def fn1():
                return tf.zeros([self.batch_size, self.output_size])
            def fn2():
                def fn2_1():
                    return self.target_output[:, time - 1, :]

                def fn2_2():
                    inds = tf.argmax(outputs_cache.read(time - 1), axis=-1)
                    return tf.one_hot(inds, depth=self.output_size)

                if self.use_teacher:
                    return tf.cond(self.teacher_force[time - 1], fn2_1, fn2_2)
                else:
                    return  fn2_2()

            feed_value = tf.cond(time>0,fn2,fn1)


            if not self.use_emb_decoder:
                r = tf.reshape(feed_value, [self.batch_size, self.input_decoder_size])
                step_input = r
            elif self.dual_emb:
                step_input = tf.matmul(feed_value, self.W_emb_decoder)
            else:
                step_input = tf.matmul(feed_value, self.W_emb_encoder)

        else:
            if self.use_emb_decoder:
                if self.dual_emb:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_decoder)
                else:
                    step_input = tf.matmul(self.unpacked_input_decoder_data.read(time), self.W_emb_encoder)
            else:
                step_input = self.unpacked_input_decoder_data.read(time)
                print(step_input.shape)
                print('ssss')

        # compute one step of controller
        output_list = self._step_op_decoder(time, step_input, memory_state, controller_state)
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = output_list[11] # state hidden  values

        if self.nlayer>1:
            try:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1][-1])
                print('state include c and h')
            except:
                controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
                print('state include only h')
        else:
            controller_hiddens = controller_hiddens.write(time, new_controller_state[-1])
            print('single layer')
        outputs = outputs.write(time, output_list[7])  # new output is updated
        outputs_cache = outputs_cache.write(time, output_list[7])  # new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1,  # 0
            new_memory_state,  # 1
            outputs,  # 2
            free_gates, allocation_gates, write_gates,  # 3 4 5
            read_weightings, write_weightings, usage_vectors,  # 6 7 8
            new_controller_state,  # 9
            outputs_cache,  # 10
            controller_hiddens,  # 11
            encoder_write_weightings, #12
            encoder_controller_hiddens, #13
        )

    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        # make dynamic time step length tensor
        self.unpacked_input_encoder_data = utility.unpack_into_tensorarray(self.input_encoder, 1, self.sequence_length)

        # want to store all time step values of these variables
        eoutputs = tf.TensorArray(tf.float32, self.sequence_length)
        eoutputs_cache = tf.TensorArray(tf.float32, self.sequence_length)
        efree_gates = tf.TensorArray(tf.float32, self.sequence_length)
        eallocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        ewrite_gates = tf.TensorArray(tf.float32, self.sequence_length)
        eread_weightings = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        ewrite_weightings = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        eusage_vectors = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        econtroller_hiddens = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)
        cache_econtroller_hiddens = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False, dynamic_size=True)

        # make dynamic time step length tensor
        self.unpacked_input_decoder_data = utility.unpack_into_tensorarray(self.input_decoder, 1, self.decode_length)

        # want to store all time step values of these variables
        doutputs = tf.TensorArray(tf.float32, self.decode_length)
        doutputs_cache = tf.TensorArray(tf.float32, self.decode_length)
        dfree_gates = tf.TensorArray(tf.float32, self.decode_length)
        dallocation_gates = tf.TensorArray(tf.float32, self.decode_length)
        dwrite_gates = tf.TensorArray(tf.float32, self.decode_length)
        dread_weightings = tf.TensorArray(tf.float32, self.decode_length)
        dwrite_weightings = tf.TensorArray(tf.float32, self.decode_length, clear_after_read=False)
        dusage_vectors = tf.TensorArray(tf.float32, self.decode_length)
        dcontroller_hiddens = tf.TensorArray(tf.float32, self.decode_length, clear_after_read=False)

        # inital state for RNN controller
        controller_state = self.controller.zero_state()
        print(controller_state)
        memory_state = self.memory.init_memory()



        # final_results = None
        with tf.variable_scope("sequence_encoder_loop"):
            time = tf.constant(0, dtype=tf.int32)
            time2 = tf.constant(0, dtype=tf.int32)
            # use while instead of scan --> suitable with dynamic time step
            encoder_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body_encoder,
                loop_vars=(
                    time, memory_state, eoutputs,
                    efree_gates, eallocation_gates, ewrite_gates,
                    eread_weightings, ewrite_weightings,
                    eusage_vectors, controller_state,
                    eoutputs_cache, econtroller_hiddens, time2, cache_econtroller_hiddens
                ), # do not need to provide intial values, the initial value lies in the variables themselves
                parallel_iterations=self.parallel_rnn,
                swap_memory=True
            )

        memory_state2 = self.memory.init_memory(self.read_heads_decode)
        if self.read_heads_decode!=self.read_heads:
            encoder_results_state=(encoder_results[1][0],encoder_results[1][1],encoder_results[1][2],
                                encoder_results[1][3],encoder_results[1][4], memory_state2[5],memory_state2[6])
        else:
            encoder_results_state=encoder_results[1]



        with tf.variable_scope("sequence_decoder_loop"):
            time = tf.constant(0, dtype=tf.int32)
            nstate = controller_state
            if self.pass_encoder_state:
                nstate = encoder_results[9]

            self.final_encoder_state = nstate
            self.final_encoder_readw = encoder_results[6].read(encoder_results[0]-1)
            self.final_encoder_memory_mat = encoder_results[1][0]

            # use while instead of scan --> suitable with dynamic time step
            final_results = tf.while_loop(
                cond=lambda time, *_: time < self.decode_length,
                body=self._loop_body_decoder,
                loop_vars=(
                    time, encoder_results_state, doutputs,
                    dfree_gates, dallocation_gates, dwrite_gates,
                    dread_weightings, dwrite_weightings,
                    dusage_vectors, nstate,
                    doutputs_cache, dcontroller_hiddens,
                    encoder_results[7], encoder_results[11]
                ),  # do not need to provide intial values, the initial value lies in the variables themselves
                parallel_iterations=self.parallel_rnn,
                swap_memory=True
            )


        dependencies = []
        if self.controller.has_recurrent_nn:
            # tensor array of pair of hidden and state values of rnn
            dependencies.append(self.controller.update_state(final_results[9]))

        with tf.control_dependencies(dependencies):
            # convert output tensor array to normal tensor
            self.packed_output = utility.pack_into_tensor(final_results[2], axis=1)
            if self.use_encoder_output:
                self.packed_output_encoder = utility.pack_into_tensor(encoder_results[2], axis=1)

            self.packed_memory_view_encoder = {
                'free_gates': utility.pack_into_tensor(encoder_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(encoder_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(encoder_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(encoder_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(encoder_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(encoder_results[8], axis=1),
                'final_controller_ch': encoder_results[9],
            }


            self.packed_memory_view_decoder = {
                'free_gates': utility.pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(final_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(final_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(final_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(final_results[8], axis=1),
                'final_controller_ch':final_results[9],
            }



    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view
        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        if self.use_encoder_output:
            return self.packed_output_encoder, self.packed_memory_view_encoder, self.packed_memory_view_decoder
        return self.packed_output, self.packed_memory_view_encoder, self.packed_memory_view_decoder

    def get_hidden_value_from_state(self, state):
        print("state")
        if self.nlayer > 1:
            if 'lstm' in self.controller_cell_type:
                ns = state[-1][-1]
                print('multilayer state include c and h')
                statetype = 1
            else:
                ns = state[-1]
                print('multilayer state include only h')
                statetype = 2
        else:
            if 'lstm' in self.controller_cell_type:
                ns = state[-1]
                statetype = 3
            else:
                ns = state
                statetype = 4
            print('single layer')
        print("{}:{}".format(statetype, ns))
        return ns, statetype

    def get_single_output(self):
        h, _ = self.get_hidden_value_from_state(self.final_encoder_state)
        h = tf.nn.relu(h)
        if not self.use_mem:
            self.sWo = tf.get_variable('sWo', [self.hidden_controller_dim, self.hidden_controller_dim//4],
                                        initializer=tf.random_normal_initializer(stddev=0.1))
        else:
            self.sWo = tf.get_variable('sWo', [self.hidden_controller_dim+self.word_size*self.read_heads,
                                               self.hidden_controller_dim//4],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            readv = self.memory.update_read_vectors(self.final_encoder_memory_mat, self.final_encoder_readw)
            h = tf.concat([h,tf.reshape(readv,[self.batch_size,-1])], axis=-1)


        output = tf.matmul(h, self.sWo)
        output = tf.nn.dropout(output, keep_prob=self.drop_out_v)
        output = tf.nn.relu(output)
        self.sWo2 = tf.get_variable('sWo2', [self.hidden_controller_dim//4, self.output_size],
                                   initializer=tf.random_normal_initializer(stddev=0.1))
        output = tf.matmul(output, self.sWo2)
        return tf.expand_dims(output, axis=1)

    def assign_pretrain_emb_encoder(self, sess, lookup_mat):
        assign_op_W_emb_encoder = self.W_emb_encoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_encoder])

    def assign_pretrain_emb_decoder(self, sess, lookup_mat):
        assign_op_W_emb_decoder = self.W_emb_decoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_decoder])


    def build_loss_function(self, optimizer=None, clip_s=10):
        print('build loss....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _, _ = self.get_outputs()

        prob = tf.nn.softmax(output, dim=-1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_output,
            logits=output, dim=-1))


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                if isinstance(clip_s, list):
                    gradients[i] = (tf.clip_by_value(grad, clip_s[0], clip_s[1]), var)
                else:
                    gradients[i] = (tf.clip_by_norm(grad, clip_s), var)


        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients

    def build_loss_function_regression(self, optimizer=None, clip_s=10):
        print('build loss....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _, _ = self.get_outputs()


        loss = tf.reduce_mean(tf.squared_difference(
            self.target_output,
            output))


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                if isinstance(clip_s, list):
                    gradients[i] = (tf.clip_by_value(grad, clip_s[0], clip_s[1]), var)
                else:
                    gradients[i] = (tf.clip_by_norm(grad, clip_s), var)


        apply_gradients = optimizer.apply_gradients(gradients)
        return output, output, loss, apply_gradients

    def build_loss_function_multi_label(self, optimizer=None, clip_s=10, prefer_one_class=False, is_neat=False):
        print('build loss multi label....')
        if self.use_mem:
            is_neat=False
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        if not is_neat:
            output, _, _ = self.get_outputs()
        else:
            output = self.get_single_output()
        print("ooooo")
        print(output)
        if prefer_one_class:
            prob = tf.nn.softmax(output)
            fn = tf.nn.softmax_cross_entropy_with_logits
        else:
            prob = tf.nn.sigmoid(output)
            fn = tf.nn.sigmoid_cross_entropy_with_logits

        loss = tf.reduce_mean(fn(
            labels=tf.slice(self.target_output, [0, 0, 0],
                            [self.batch_size, 1, self.output_size]),

            logits=tf.slice(output, [0, 0, 0],
                            [self.batch_size, 1, self.output_size]))
        )

        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                if isinstance(clip_s, list):
                    gradients[i] = (tf.clip_by_value(grad, clip_s[0], clip_s[1]), var)
                else:
                    gradients[i] = (tf.clip_by_norm(grad, clip_s), var)

        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients


    def print_config(self):
        return '{}.{}cell_{}mem_{}dec_{}dua_{}wrp_{}wsz_{}msz_{}tea_{}hid_{}nread_{}nlayer'.\
            format(self.name, self.controller_cell_type, self.use_mem,
                   self.decoder_mode,
                   self.dual_controller,
                   self.write_protect,
                   self.words_num,
                   self.word_size,
                   self.use_teacher,
                   self.hidden_controller_dim,
                   self.read_heads_decode,
                   self.nlayer)

    @staticmethod
    def get_size_model():
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    @staticmethod
    def save(session, ckpts_dir, name):
        """
        saves the current values of the model's parameters to a checkpoint
        Parameters:
        ----------
        session: tf.Session
            the tensorflow session to save
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        checkpoint_dir = os.path.join(ckpts_dir, name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver(tf.global_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))





    @staticmethod
    def restore(session, ckpts_dir, name):
        """
        session: tf.Session
            the tensorflow session to restore into
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        tf.train.Saver(tf.global_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))

    @staticmethod
    def get_bool_rand(size_seq, prob_true=0.1):
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_incremental(size_seq, prob_true_min=0, prob_true_max=0.25):
        ret = []
        for i in range(size_seq):
            prob_true=(prob_true_max-prob_true_min)/size_seq*i
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_curriculum(size_seq, epoch, k=0.99, type='exp'):
        if type=='exp':
            prob_true = k**epoch
        elif type=='sig':
            prob_true = k / (k + np.exp(epoch / k))
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)
