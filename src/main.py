import argparse
import os
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.data_loader import load_data, create_dataloder
from src.main_utils import epoch_check, init_model, init_active_method, init_logger, need_query
from src.utils import set_seed

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def parsers_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_type', type=str, default='None', help='None, txt, tb')
    parser.add_argument('--save_folder_name', type=str, default='txt_logs')
    # regular training args
    parser.add_argument('--dataset_str', type=str, default='imbalanced_cf10',
                        help='[cf10, mnist, svhn, caltech101; imbalanced_cf10].')
    parser.add_argument('--MSE', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum.')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--base_model', type=str, default="mlp", help='[cnn_avgpool, vgg, mlp, cnn, resnet]')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--small_bz', type=int, default=50, help='args for the package BackPack')
    parser.add_argument('--grad_compute_batch_size', type=int, default=1024, help='args for the package BackPack')
    # imbalance setting args, dynamicAL is better under this wild setting
    parser.add_argument('--train_imbalance', type=int, default=0)
    parser.add_argument('--pool_imbalance', type=int, default=1)
    parser.add_argument('--test_imbalance', type=int, default=0)
    # ablation study args
    parser.add_argument('--wo_de_inf', type=int, default=0)
    parser.add_argument('--norm_inf', type=int, default=0)
    parser.add_argument('--inf_type', type=str, default='full', help='residual, full')
    parser.add_argument('--wo_train_sim', type=int, default=0)
    # active args
    parser.add_argument('--final_extra_epochs', type=int, default=150)
    parser.add_argument('--init_label_perC', type=int, default=None)
    parser.add_argument('--init_label_num', type=float, default=None)
    parser.add_argument('--budget_num_per_query', type=float, default=500)
    parser.add_argument('--total_query_times', type=int, default=5,
                        help='how many times we can do queries, -1 denotes for no query time restriction')
    parser.add_argument('--query_interval', nargs='+', type=int, default=[])
    parser.add_argument('--fixed_query_interval', type=int, default=5)
    parser.add_argument('--starting_epoch', type=int, default=1)
    parser.add_argument('--reinit', type=int, default=0)

    parser.add_argument('--active_method', type=str, default="ntk_memo")
    parser.add_argument('--pesudo', type=str, default='max', help='max, sample')

    parser.add_argument('--grad_update_size', type=int, default=100)
    parser.add_argument('--train_sim_times', type=int, default=2)

    # add in rebuttal, (for fast caltech)
    parser.add_argument('--train_acc_threshold', type=float, default=0.97)


    # scaling and perturb
    parser.add_argument('--s_p', type=int, default=0,
                        help='not used in the original paper, but it is an interesting trick.')
    parser.add_argument('--scaling', type=float, default=0.4)
    parser.add_argument('--perturb', type=float, default=0.05)
    args = parser.parse_args()

    # before return, args sanity check
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.budget_num_per_query % args.grad_update_size == 0
    args = epoch_check(args)
    return args


def test(args, model, test_loader, epoch, writer, active_times):
    correct = 0
    total = 0
    testing_loss = []
    model.eval()
    with torch.no_grad():
        for idxs, images, labels in test_loader:
            if model.device == 'cuda':
                images = images.to(model.device)
                labels = labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if model.MSE:
                labels = torch.nn.functional.one_hot(labels)
                loss_test = model.compute_MSELoss(outputs, labels)
            else:
                loss_test = model.compute_CELoss(outputs, labels)
            testing_loss.append(loss_test.item())

    test_acc = correct / total
    if model.best_test_acc < test_acc:
        model.best_test_acc = test_acc
    if test_acc > model.prev_best_test_acc:
        model.better_than_prev_epochs += 1

    if args.log_type == 'tb':
        writer.add_scalar('test/loss', sum(testing_loss) / len(testing_loss), epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
        writer.add_scalar('test/best_acc', model.best_test_acc, epoch)
    print('%d [Epoch: %d] Accuracy on the test images: %.5f %%' % (active_times, epoch, 100 * correct / total))


def train(args, model, optimizer, train_loader, epoch, writer):
    running_loss = []
    model.train()
    correct = 0
    total = 0
    for idxs, inputs, labels in train_loader:
        if model.device == 'cuda':
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if model.MSE:
            labels = torch.nn.functional.one_hot(labels)
            loss_train = model.compute_MSELoss(outputs, labels)
        else:
            loss_train = model.compute_CELoss(outputs, labels)

        loss_train.backward()
        optimizer.step()
        running_loss.append(loss_train.item())

        # train acc
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total

    if train_acc > args.train_acc_threshold:
        model.train_converge_epochs += 1

    if args.log_type == 'tb':
        writer.add_scalar('train/loss', sum(running_loss)/len(running_loss), epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
    print("[Epoch:{}] Train Acc:{}".format(epoch, train_acc))
    return train_acc


def main(args):
    train_data, test_data, pool_data = load_data(args)
    train_loader = create_dataloder(args, train_data, shuffle=True)
    test_loader = create_dataloder(args, test_data, shuffle=False)
    pool_loader = create_dataloder(args, pool_data, shuffle=False)

    model = init_model(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.total_query_times > 0:
        active_policy = init_active_method(args, model)
    writer, writer_title = init_logger(args)

    epoch = 0
    final_epoch = 0
    active_times = 0
    active_now = 0
    curr_round_epoch = 0

    while True:
        epoch += 1
        curr_round_epoch += 1
        if epoch % args.test_interval == 0 or active_now:
            test(args, model, test_loader, epoch, writer, active_times)
            active_now = 0
        train_acc = train(args, model, optimizer, train_loader, epoch, writer)
        model.current_round_epochs += 1

        if args.total_query_times == active_times:
            final_epoch += 1

        if final_epoch > args.final_extra_epochs:
            if args.log_type == 'txt':
                with open(writer, 'a+') as file_io:
                    file_io.write('{}\t{}\n'.format(active_times, model.best_test_acc))
                    file_io.write('Finished in: final_epoch > args.final_extra_epochs')
            break

        if need_query(args, model, epoch, curr_round_epoch, active_times):
            train_loader, pool_loader, train_data, pool_data = \
                active_policy.query(args, model, train_loader, pool_loader, train_data, pool_data)
            torch.cuda.empty_cache()
            if args.log_type == 'txt':
                with open(writer, 'a+') as file_io:
                    file_io.write('{}\t{}\n'.format(active_times, model.best_test_acc))
            model.prev_best_test_acc = model.best_test_acc
            model.best_test_acc = 0
            model.better_than_prev_epochs = 0
            model.train_converge_epochs = 0
            model.current_round_epochs = 0
            active_now = 1
            final_epoch = 0
            curr_round_epoch = 0
            active_times += 1

            if args.reinit:
                print("Reinit model..")
                del model, active_policy
                torch.cuda.empty_cache()
                model = init_model(args)
                if args.total_query_times > 0:
                    active_policy = init_active_method(args, model)

            if args.s_p: # scaling and perturb trick, https://arxiv.org/pdf/1910.08475.pdf
                print('Scaling & Perturbing Model..')
                temp_model = init_model(args)
                params1 = model.parameters()
                params2 = temp_model.parameters()
                for p1, p2 in zip(*[params1, params2]):
                    p1.data = deepcopy(args.perturb * p2.data + args.scaling * p1.data)

            del optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.log_type == 'txt':
        with open(writer, 'a+') as file_io:
            file_io.write('{}\t{}\n'.format(active_times, model.best_test_acc))
            file_io.write("Finished!")


if __name__ == '__main__':
    args = parsers_parser()
    set_seed(args.seed)
    print(args)
    main(args)