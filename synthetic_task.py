import tensorflow as tf
import numpy as np
import pickle
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from uw_dnc import DNC
from recurrent_controller import StatelessRecurrentController
import visual_util

def exact_acc(target_batch, predict_batch, stop_S=-1, pprint=1.0):
    acc=[]
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []

        for ti, t in enumerate(target_batch[b]):
            if t != stop_S:
                trim_target.append(t)

        for ti, t in enumerate(predict_batch[b]):
            if t != stop_S:
                trim_predict.append(t)

        if np.random.rand()>pprint or b==0:
            print('{} vs {}'.format(trim_target, trim_predict))
        ac=0
        for n1,n2 in zip(trim_predict, trim_target):
            if n1==n2:
                ac+=1

        acc.append(float(ac/max(len(trim_target), len(trim_predict))))#have to be correct all
    return np.mean(acc)

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    # print('-----')
    # print(index)
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec




def copy_sample(vocab_lower, vocab_upper, length_from, length_to):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    seed = np.random.choice(list(range(int(vocab_lower),int(vocab_upper))),
                      int(random_length()), replace=True)
    inp = seed.tolist()
    inp = inp + [0]
    out = seed.tolist()
    out = out + [0]

    return inp, out



def sum_sample(vocab_lower, vocab_upper, length_from, length_to):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    seed = np.random.choice(list(range(int(vocab_lower),int(vocab_upper))),
                      int(random_length()), replace=True)
    inp = seed.tolist()
    out=[]
    for i in range(len(inp)//2):
        out.append((inp[i]+inp[-1-i])//2)
    inp = inp + [0]
    out = out + [0]

    return inp, out


def reverse_sample(vocab_lower, vocab_upper, length_from, length_to):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    seed = np.random.choice(list(range(int(vocab_lower),int(vocab_upper))),
                      int(random_length()), replace=True)
    inp = seed.tolist()
    out = inp[::-1]
    inp = inp + [0]
    # out1 = seed[:len(seed)//2].tolist()
    # out2 = seed[len(seed)//2:].tolist()
    out = out + [0]
    # out = sorted(out1) + sorted(out2, reverse=True) + [0]

    return inp, out

def double_sample(vocab_lower, vocab_upper, length_from, length_to):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    seed = np.random.choice(list(range(int(vocab_lower),int(vocab_upper))),
                      int(random_length()), replace=True)
    inp = seed.tolist()

    out=inp+inp

    inp = inp + [0]
    out = out + [0]

    return inp, out

def max_sample(vocab_lower, vocab_upper, length_from, length_to):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    seed = np.random.choice(list(range(int(vocab_lower),int(vocab_upper))),
                      int(random_length()), replace=True)
    inp = seed.tolist()
    out=[]
    for i in range(len(inp)//2):
        if inp[i*2]>inp[i*2+1]:
            out.append(inp[i*2])
        else:
            out.append(inp[i*2+1])


    inp = inp + [0]
    out = out + [0]

    return inp, out

def prepare_batch(bs, vocab_size, length_from, length_to, args):
    length=np.random.randint(length_from, length_to + 1)
    inps=np.zeros(shape= [bs,length+1, vocab_size])
    lout=length
    if "sum" in args.task:
        lout=length//2
    if "max" in args.task:
        lout = length // 2
    if "double" in args.task:
        lout=length*2
    oups=np.zeros(shape=[bs,lout+1, vocab_size])
    oups2=np.zeros(shape=[bs,lout+1, vocab_size])
    hold_mem = np.zeros(length + 1, dtype=bool)
    if args.hold_mem_mode>0:
        hold_mem = np.ones(length+1, dtype=bool)
        # print(hold_mem)

        holdstep=(length+1)//(args.mem_size+1)
        holdstep=min(holdstep, args.cache_size)

        if "random" in args.memo_type:
            hold_mem=global_var["hold_mem_random"]
        else:
            if holdstep>0:
                for iii in range(holdstep, int(length+1), holdstep):
                    hold_mem[iii] = False
            else:
                hold_mem[(length+1)//2] = False
    # print(hold_mem)
    lin=[]
    lou=[]
    for b in range(bs):
        if "copy" in args.task:
            i,o=copy_sample(1,vocab_size,length, length)
        elif "sum" in args.task:
            i,o=sum_sample(1,vocab_size,length, length)
        elif "double" in args.task:
            i,o=double_sample(1,vocab_size,length, length)
        elif "reverse" in args.task:
            i,o=reverse_sample(1, vocab_size, length, length)
        elif "max" in args.task:
            i,o=max_sample(1,vocab_size,length, length)


        lin.append(i)
        lou.append(o)
        c=0
        for c1 in i :
            inps[b, c, :]=onehot(c1, vocab_size)
            c+=1
        c = 0
        for c2 in o:
            oups[b, c, :] = onehot(c2, vocab_size)
            c += 1



    return inps, oups, oups2, length+1, lout+1, lin, lou, hold_mem

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


def synthetic_task(args):
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/save/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_{}_task'.format(args.task))

    llprint("Loading Data ... ")

    batch_size = args.batch_size
    input_size = args.number_range
    output_size = args.number_range
    print('dim out {}'.format(output_size))
    words_count = args.mem_size
    word_size = args.word_size
    read_heads = args.read_heads

    learning_rate = args.learning_rate
    momentum = 0.9

    iterations = args.iterations
    start_step = 0

    config = tf.ConfigProto(device_count={'CPU': args.cpu_num})
    config.intra_op_parallelism_threads = args.cpu_num

    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = args.gpu_ratio
    graph = tf.Graph()
    with graph.as_default():
        tf.contrib.framework.get_or_create_global_step()
        with tf.Session(graph=graph, config=config) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                output_size,
                words_count,
                word_size,
                read_heads,
                batch_size,
                use_mem=args.use_mem,
                controller_cell_type=args.cell_type,
                use_emb_encoder=False,
                use_emb_decoder=False,
                hold_mem_mode=args.hold_mem_mode,
                hidden_controller_dim=args.hidden_dim,
                cache_attend_dim=args.cache_attend_dim,
                nlayer=args.nlayer,
                clip_output=20,
                batch_norm=args.batch_norm,
                pass_encoder_state=True,
                parallel_rnn=10,
                name='m'+str(args.cache_attend_dim)+str(args.hold_mem_mode)+args.memo_type+str(args.cache_size)
            )



            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


            _, prob, loss, apply_gradients = ncomputer.build_loss_function(optimizer, clip_s=5)

            llprint("Done!\n")
            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")
            variables_names = [v.name for v in tf.trainable_variables()]
            values = session.run(variables_names)
            print("SHOW VARIABLES")
            for k, v in zip(variables_names, values):
                print("Variable: {} shape {} ".format(k, v.shape))
                # print (v)
                print("*************")


            if args.from_checkpoint is not '':
                if args.from_checkpoint == 'default':
                    from_checkpoint = ncomputer.print_config()
                else:
                    from_checkpoint = args.from_checkpoint
                llprint("Restoring Checkpoint %s ... " % from_checkpoint)
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")

            last_100_losses = []

            print('no param {}'.format(ncomputer.get_size_model()))

            start = 1 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            if args.mode == 'test':
                start = 0
                end = start


            start_time_100 = time.time()

            avg_100_time = 0.
            avg_counter = 0
            if args.mode == 'train':
                log_dir = './data/summary/log_{}/'.format(args.task)
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                log_dir = '{}/{}/'.format(log_dir, ncomputer.print_config())
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                train_writer = tf.summary.FileWriter(log_dir, session.graph)
            min_tloss = 0
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_data, target_output, itarget, seq_len, decoder_length, _, _, hold = \
                        prepare_batch(batch_size,args.number_range, args.length_from, args.length_to,args)
                    fd={
                        ncomputer.input_encoder: input_data,
                        ncomputer.input_decoder: itarget,
                        ncomputer.target_output: target_output,
                        ncomputer.sequence_length: seq_len,
                        ncomputer.decode_length: decoder_length,
                    }
                    if args.hold_mem_mode>0:
                        fd[ncomputer.hold_mem]=hold
                    summerize = (i % args.valid_time == 0)
                    if args.mode == 'train':
                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict=fd)
                        last_100_losses.append(loss_value)
                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        trscores_acc = []


                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))

                        for ii in range(5):
                            input_data, target_output, itarget, seq_len, decoder_length, brin, brout, hold = \
                                prepare_batch(batch_size, args.number_range, args.length_from, args.length_to, args)

                            fd = {
                                ncomputer.input_encoder: input_data,
                                ncomputer.input_decoder: itarget,
                                ncomputer.target_output: target_output,
                                ncomputer.sequence_length: seq_len,
                                ncomputer.decode_length: decoder_length,
                            }
                            if args.hold_mem_mode > 0:
                                fd[ncomputer.hold_mem] = hold

                            out ,emem_v, dmem_v = session.run([prob,
                                                               ncomputer.packed_memory_view_encoder,
                                                               ncomputer.packed_memory_view_decoder], feed_dict=fd)

                            out = np.reshape(np.asarray(out), [-1, decoder_length, output_size])
                            out = np.argmax(out, axis=-1)
                            bout_list = []
                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(out.shape[1]):
                                    # if out[b][io] == 0:
                                    #     break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)
                            trscores_acc.append(exact_acc(np.asarray(brout), np.asarray(bout_list), pprint=1))
                            # visual_util.plot_memory(emem_v, dmem_v, hold, brin)

                        tpre=np.mean(trscores_acc)
                        print('acc {}'.format(tpre))
                        if args.mode == 'train':
                            summary.value.add(tag='train_acc', simple_value=tpre)
                            train_writer.add_summary(summary, i)
                            train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.
                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []
                        if args.mode == 'train' and tpre > min_tloss:
                            min_tloss = tpre

                            llprint("\nSaving Checkpoint ... "),

                            ncomputer.save(session, ckpts_dir, ncomputer.print_config())

                            llprint("Done!\n")


                except KeyboardInterrupt:
                    sys.exit(0)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

global_var={"hold_mem_random":None}


def limit_copy(args):
    args.task = "copy500"
    args.cell_type = "lstm"
    args.mem_type = "dnc"
    args.hidden_dim=256
    args.mem_size = 49
    args.word_size = 64
    args.batch_size = 64
    args.number_range = 10
    args.length_from = 500
    args.length_to = 500
    args.iterations = 200000
    # args.hold_mem_mode = 2
    args.cache_sze = 1000
    return args


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--use_mem', default=True, type=str2bool)
    parser.add_argument('--cell_type', default="nlstm")
    parser.add_argument('--mem_type', default="dnc")
    parser.add_argument('--task', default="copy", help="support 5 tasks: copy/reverse/double/sum/max")
    parser.add_argument('--from_checkpoint', default="")
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--cache_attend_dim', default=0, type=int)
    parser.add_argument('--mem_size', default=4, type=int)
    parser.add_argument('--word_size', default=64, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--read_heads', default=1, type=int)
    parser.add_argument('--read_heads_decode', default=1, type=int)
    parser.add_argument('--batch_norm', default=True, type=str2bool)
    parser.add_argument('--number_range', default=10, type=int)
    parser.add_argument('--length_from', default=50, type=int)
    parser.add_argument('--length_to', default=50, type=int)
    parser.add_argument('--memo_type', default="", type=str)
    parser.add_argument('--hold_mem_mode', default=0, type=int)
    parser.add_argument('--cache_size', default=100, type=int)
    parser.add_argument('--nlayer', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--lr_decay_step', default=10000, type=float)
    parser.add_argument('--lr_decay_rate', default=0.9, type=float)
    parser.add_argument('--iterations', default=10000, type=int)
    parser.add_argument('--valid_time', default=100, type=int)
    parser.add_argument('--gpu_ratio', default=0.4, type=float)
    parser.add_argument('--cpu_num', default=5, type=int)
    parser.add_argument('--gpu_device', default="1,2,3", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device





    # args = limit_copy(args)

    print(args)

    hold_mem_random = np.ones(args.length_to + 1, dtype=bool)
    c=0
    if "random" in args.memo_type:
        for iii in range(int(args.length_to + 1)):
            if np.random.rand() > 0.5:
                hold_mem_random[iii] = False
                c+=1

    global_var["hold_mem_random"]=hold_mem_random

    synthetic_task(args)