import os
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import datetime
import shutil

from conv_net import *
from rdd_net import *
from shared_hyperparams import *

parser = argparse.ArgumentParser()
parser.add_argument('folder_prefix', help='Prefix of folder name where data will be saved')
parser.add_argument("-n_epochs", type=int, help="Number of epochs", default=100)
parser.add_argument("-batch_size", type=int, help="Batch size", default=32)
parser.add_argument("-lr", help="Learning rate", type=float, default=0.01)
parser.add_argument("-momentum", type=float, help="Momentum", default=0.9)
parser.add_argument("-weight_decay", type=float, help="Weight decay", default=0)
parser.add_argument("-rdd_time", type=float, help="RDD time (s)", default=90)
parser.add_argument('-do_rdd', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-rdd_every_epoch', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-use_backprop', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-info", type=str, help="Any other information about the simulation", default="")

args=parser.parse_args()

folder_prefix           = args.folder_prefix
n_epochs                = args.n_epochs
batch_size              = args.batch_size
lr                      = args.lr
momentum                = args.momentum
weight_decay            = args.weight_decay
rdd_time                = args.rdd_time
do_rdd                  = args.do_rdd
rdd_every_epoch         = args.rdd_every_epoch
use_backprop            = args.use_backprop
info                    = args.info

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_set    = torchvision.datasets.CIFAR10(root='../Data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_set    = torchvision.datasets.CIFAR10(root='../Data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

train_set_2    = torchvision.datasets.CIFAR10(root='../Data', train=True, download=True, transform=transform_test)
train_loader_2 = torch.utils.data.DataLoader(train_set_2, batch_size=100, shuffle=True, num_workers=2)

net = ConvNet(input_channels=3, use_backprop=use_backprop).to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

def train(epoch):
    print("\nEpoch {}.".format(epoch))

    net.train()

    train_loss = 0
    correct    = 0
    total      = 0

    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if not use_backprop and weight_decay > 0:
            net.decay_fb_weights(weight_decay)

        train_loss  += loss.item()
        _, predicted = outputs.max(1)
        total       += targets.size(0)
        correct     += predicted.eq(targets).sum().item()

        progress_bar.set_description("Train Loss: {:.3f} | Acc: {:.3f}%% ({:d}/{:d})".format(train_loss/(batch_idx+1), 100*correct/total, correct, total))

    return 100*(1 - correct/total), train_loss

def test(epoch, train_set=False):
    net.eval()

    loss      = 0
    correct   = 0
    total     = 0
    with torch.no_grad():
        if train_set:
            progress_bar = tqdm(train_loader_2)
        else:
            progress_bar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss        += loss.item()
            _, predicted = outputs.max(1)
            total       += targets.size(0)
            correct     += predicted.eq(targets).sum().item()

            if train_set:
                progress_bar.set_description("Train Loss: {:.3f} | Acc: {:.3f}%% ({:d}/{:d})".format(loss/(batch_idx+1), 100*correct/total, correct, total))
            else:
                progress_bar.set_description("Test Loss: {:.3f} | Acc: {:.3f}%% ({:d}/{:d})".format(loss/(batch_idx+1), 100*correct/total, correct, total))

    return 100*(1 - correct/total), loss

# create the RDD net
rdd_net = RDDNet()

if folder_prefix is not None:
    # generate a name for the folder where data will be stored
    folder = "{} - {} - {} - {} - {}".format(folder_prefix, lr, batch_size, momentum, weight_decay) + " - BP"*(use_backprop == True)  + " - {}".format(info)*(info != "")
else:
    folder = None

if folder is not None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save a human-readable text file containing simulation details
    timestamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    with open(os.path.join(folder, "params.txt"), "w") as f:
        f.write("Simulation run @ {}\n".format(timestamp))
        f.write("Number of epochs: {}\n".format(n_epochs))
        f.write("Batch size: {}\n".format(batch_size))
        f.write("Learning rates: {}\n".format(lr))
        f.write("Momentum: {}\n".format(momentum))
        f.write("Weight decay: {}\n".format(weight_decay))
        f.write("RDD time: {} s\n".format(rdd_time))
        f.write("Do RDD every epoch: {} s\n".format(rdd_every_epoch))
        f.write("Using backprop: {}\n".format(use_backprop))
        if info != "":
            f.write("Other info: {}\n".format(info))
    filename = os.path.basename(__file__)
    if filename.endswith('pyc'):
        filename = filename[:-1]
        shutil.copyfile(os.path.abspath(__file__)[:-1], os.path.join(folder, filename))
    shutil.copyfile(os.path.abspath(__file__), os.path.join(folder, filename))

symm_losses   = [ [] for i in range(3) ]
decay_losses  = [ [] for i in range(3) ]
sparse_losses = [ [] for i in range(3) ]
self_losses   = [ [] for i in range(3) ]
amp_losses    = [ [] for i in range(3) ]
info_losses   = [ [] for i in range(3) ]
corr_percents = [ [] for i in range(3) ]

def train_fb():
    rdd_net.reset()

    rdd_net.copy_weights_from([net.conv1, net.conv2, net.fc1, net.fc2, net.fc3])

    RDD_time = int(rdd_time/dt)

    print("Performing RDD pre-training for {} s...".format(rdd_time))

    for i in range(3):
        weight    = rdd_net.classification_layers[-3+i].weight_orig
        fb_weight = rdd_net.classification_layers[-4+i].fb_weight
        x         = np.random.uniform(0, 1, size=(weight.shape[1], 1))

        if np.sum(fb_weight != 0) > 0:
            corr_percent = 100*np.sum((np.sign(fb_weight.T) == np.sign(weight)) & (fb_weight.T != 0))/np.sum(fb_weight.T != 0)
        else:
            corr_percent = 0

        corr_percents[i].append(corr_percent)

        decay_loss  = np.linalg.norm(np.dot(weight, x))**2 + np.linalg.norm(np.dot(x.T, fb_weight))**2
        sparse_loss = np.linalg.norm(weight)**2 + np.linalg.norm(fb_weight)**2
        self_loss   = -np.trace(np.dot(fb_weight, weight))
        amp_loss    = -np.trace(np.dot(x.T, np.dot(fb_weight, np.dot(weight, x))))
        info_loss   = np.linalg.norm(x - np.dot(fb_weight, np.dot(weight, x)))**2
        symm_loss   = np.linalg.norm(weight - fb_weight.T)**2

        decay_losses[i].append(decay_loss)
        sparse_losses[i].append(sparse_loss)
        self_losses[i].append(self_loss)
        amp_losses[i].append(amp_loss)
        info_losses[i].append(info_loss)
        symm_losses[i].append(symm_loss)

    text = "Time: {}/{} s. Correct: {:.2f}% / {:.2f}% / {:.2f}%. Trace:  {:.2f} / {:.2f} / {:.2f}.".format(0*dt, RDD_time*dt, corr_percents[0][-1], corr_percents[1][-1], corr_percents[2][-1], self_losses[0][-1], self_losses[1][-1], self_losses[2][-1])

    print(text)

    indices_1 = np.random.choice(1600, int(.2*1600), replace=False)
    indices_2 = np.random.choice(384, int(.2*384), replace=False)
    indices_3 = np.random.choice(192, int(.2*192), replace=False)

    driving_rates_1 = np.zeros((1600, 1))
    driving_rates_2 = np.zeros((384, 1))
    driving_rates_3 = np.zeros((192, 1))

    driving_rates_1[indices_1] = input_rate
    driving_rates_2[indices_2] = input_rate
    driving_rates_3[indices_3] = input_rate

    driving_spike_hist_1 = np.zeros((1600, mem), dtype=int)
    driving_spike_hist_2 = np.zeros((384, mem), dtype=int)
    driving_spike_hist_3 = np.zeros((192, mem), dtype=int)

    spike_rates_1 = np.zeros(1600)
    spike_rates_2 = np.zeros(384)
    spike_rates_3 = np.zeros(192)
    spike_rates_4 = np.zeros(10)

    for i in range(RDD_time):
        if (i+1) % (0.1/dt) == 0:
            indices_1 = np.random.choice(1600, int(.2*1600), replace=False)
            indices_2 = np.random.choice(384, int(.2*384), replace=False)
            indices_3 = np.random.choice(192, int(.2*192), replace=False)

            driving_rates_1 = np.zeros((1600, 1))
            driving_rates_2 = np.zeros((384, 1))
            driving_rates_3 = np.zeros((192, 1))
            
            driving_rates_1[indices_1] = input_rate
            driving_rates_2[indices_2] = input_rate
            driving_rates_3[indices_3] = input_rate

        if i < RDD_time/3:
            driving_spike_hist_1 = np.concatenate([driving_spike_hist_1[:, 1:], np.random.poisson(driving_rates_1)], axis=-1)
        elif i < 2*RDD_time/3:
            driving_spike_hist_2 = np.concatenate([driving_spike_hist_2[:, 1:], np.random.poisson(driving_rates_2)], axis=-1)
        else:
            driving_spike_hist_3 = np.concatenate([driving_spike_hist_3[:, 1:], np.random.poisson(driving_rates_3)], axis=-1)

        if i < RDD_time/3:
            rdd_net.out(driving_spike_hist_1, None, None)
        elif i < 2*RDD_time/3:
            rdd_net.out(None, driving_spike_hist_2, None)
        else:
            rdd_net.out(None, None, driving_spike_hist_3)

        spike_rates_1 += rdd_net.classification_layers[0].spike_hist[:, -1]
        spike_rates_2 += rdd_net.classification_layers[1].spike_hist[:, -1]
        spike_rates_3 += rdd_net.classification_layers[2].spike_hist[:, -1]
        spike_rates_4 += rdd_net.classification_layers[3].spike_hist[:, -1]

        if (i+1) % (10.0/dt) == 0:
            rdd_net.update_fb_weights()
            
            spike_rates_1 /= 10
            spike_rates_2 /= 10
            spike_rates_3 /= 10
            spike_rates_4 /= 10

            for j in range(3):
                weight    = rdd_net.classification_layers[-3+j].weight_orig
                fb_weight = rdd_net.classification_layers[-4+j].fb_weight
                beta      = rdd_net.classification_layers[-4+j].beta
                x         = np.random.uniform(0, 1, size=(weight.shape[1], 1))

                if np.sum(fb_weight != 0) > 0:
                    corr_percent = 100*np.sum((np.sign(fb_weight.T) == np.sign(weight)) & (fb_weight.T != 0))/np.sum(fb_weight.T != 0)
                else:
                    corr_percent = 0

                corr_percents[j].append(corr_percent)

                decay_loss  = np.linalg.norm(np.dot(weight, x))**2 + np.linalg.norm(np.dot(x.T, fb_weight))**2
                sparse_loss = np.linalg.norm(weight)**2 + np.linalg.norm(fb_weight)**2
                self_loss   = -np.trace(np.dot(fb_weight, weight))
                amp_loss    = -np.trace(np.dot(x.T, np.dot(fb_weight, np.dot(weight, x))))
                info_loss   = np.linalg.norm(x - np.dot(fb_weight, np.dot(weight, x)))**2
                symm_loss   = np.linalg.norm(weight - fb_weight.T)**2

                decay_losses[j].append(decay_loss)
                sparse_losses[j].append(sparse_loss)
                self_losses[j].append(self_loss)
                amp_losses[j].append(amp_loss)
                info_losses[j].append(info_loss)
                symm_losses[j].append(symm_loss)

                if folder is not None:
                    np.save(os.path.join(folder, "weight_{}.npy".format(j+1)), weight)
                    np.save(os.path.join(folder, "fb_weight_{}.npy".format(j+1)), fb_weight)
                    np.save(os.path.join(folder, "beta_{}.npy".format(j+1)), beta)

            text = "Time: {}/{} s. Correct: {:.2f}% / {:.2f}% / {:.2f}%. Trace: {:.2f} / {:.2f} / {:.2f}. Rates: {:.2f}Hz / {:.2f}Hz / {:.2f}Hz / {:.2f}Hz. ".format((i+1)*dt, RDD_time*dt, corr_percents[0][-1], corr_percents[1][-1], corr_percents[2][-1], self_losses[0][-1], self_losses[1][-1], self_losses[2][-1], np.mean(spike_rates_1), np.mean(spike_rates_2), np.mean(spike_rates_3), np.mean(spike_rates_4))
            
            print(text)

            if folder is not None:
                np.save(os.path.join(folder, "correct_1.npy"), np.array(corr_percents[0]))
                np.save(os.path.join(folder, "correct_2.npy"), np.array(corr_percents[1]))
                np.save(os.path.join(folder, "correct_3.npy"), np.array(corr_percents[2]))
                np.save(os.path.join(folder, "decay_1.npy"), np.array(decay_losses[0]))
                np.save(os.path.join(folder, "decay_2.npy"), np.array(decay_losses[1]))
                np.save(os.path.join(folder, "decay_3.npy"), np.array(decay_losses[2]))
                np.save(os.path.join(folder, "sparse_1.npy"), np.array(sparse_losses[0]))
                np.save(os.path.join(folder, "sparse_2.npy"), np.array(sparse_losses[1]))
                np.save(os.path.join(folder, "sparse_3.npy"), np.array(sparse_losses[2]))
                np.save(os.path.join(folder, "self_1.npy"), np.array(self_losses[0]))
                np.save(os.path.join(folder, "self_2.npy"), np.array(self_losses[1]))
                np.save(os.path.join(folder, "self_3.npy"), np.array(self_losses[2]))
                np.save(os.path.join(folder, "amp_1.npy"), np.array(amp_losses[0]))
                np.save(os.path.join(folder, "amp_2.npy"), np.array(amp_losses[1]))
                np.save(os.path.join(folder, "amp_3.npy"), np.array(amp_losses[2]))
                np.save(os.path.join(folder, "info_1.npy"), np.array(info_losses[0]))
                np.save(os.path.join(folder, "info_2.npy"), np.array(info_losses[1]))
                np.save(os.path.join(folder, "info_3.npy"), np.array(info_losses[2]))
                np.save(os.path.join(folder, "symm_1.npy"), np.array(symm_losses[0]))
                np.save(os.path.join(folder, "symm_2.npy"), np.array(symm_losses[1]))
                np.save(os.path.join(folder, "symm_3.npy"), np.array(symm_losses[2]))

            spike_rates_1 *= 0
            spike_rates_2 *= 0
            spike_rates_3 *= 0
            spike_rates_4 *= 0

    rdd_net.copy_weights_to([net.conv1, net.conv2, net.fc1, net.fc2, net.fc3], device)

test_err   = np.zeros(n_epochs+1)
test_cost  = np.zeros(n_epochs+1)
train_err  = np.zeros(n_epochs+1)
train_cost = np.zeros(n_epochs+1)
correct_1  = np.zeros(n_epochs+1)

# test_err[0], test_cost[0]   = test(-1)
# train_err[0], train_cost[0] = test(-1, train_set=True)

for epoch in range(n_epochs):
    if do_rdd and (rdd_every_epoch or epoch == 0):
        train_fb()

    if (not do_rdd) and (not use_backprop):
        # for feedback alignment, measure weight symmetry every epoch
        for i in range(3):
            if i == 0:
                layer = net.fc1
            elif i == 1:
                layer = net.fc2
            else:
                layer = net.fc3

            weight    = layer.weight.data.cpu().numpy()
            fb_weight = layer.fb_weight.data.cpu().numpy().T
            x         = np.random.uniform(0, 1, size=(weight.shape[1], 1))

            if np.sum(fb_weight != 0) > 0:
                corr_percent = 100*np.sum((np.sign(fb_weight.T) == np.sign(weight)) & (fb_weight.T != 0))/np.sum(fb_weight.T != 0)
            else:
                corr_percent = 0

            corr_percents[i].append(corr_percent)

            decay_loss  = np.linalg.norm(np.dot(weight, x))**2 + np.linalg.norm(np.dot(x.T, fb_weight))**2
            sparse_loss = np.linalg.norm(weight)**2 + np.linalg.norm(fb_weight)**2
            self_loss   = -np.trace(np.dot(fb_weight, weight))
            amp_loss    = -np.trace(np.dot(x.T, np.dot(fb_weight, np.dot(weight, x))))
            info_loss   = np.linalg.norm(x - np.dot(fb_weight, np.dot(weight, x)))**2

            decay_losses[i].append(decay_loss)
            sparse_losses[i].append(sparse_loss)
            self_losses[i].append(self_loss)
            amp_losses[i].append(amp_loss)
            info_losses[i].append(info_loss)

            if folder is not None:
                np.save(os.path.join(folder, "weight_{}.npy".format(i+1)), weight)
                np.save(os.path.join(folder, "fb_weight_{}.npy".format(i+1)), fb_weight)

        text = "Epoch {}/{}. Correct: {:.2f}% / {:.2f}% / {:.2f}%. Trace:  {:.2f} / {:.2f} / {:.2f}.".format(epoch, n_epochs, corr_percents[0][-1], corr_percents[1][-1], corr_percents[2][-1], self_losses[0][-1], self_losses[1][-1], self_losses[2][-1])

        print(text)

        if folder is not None:
            np.save(os.path.join(folder, "correct_1.npy"), np.array(corr_percents[0]))
            np.save(os.path.join(folder, "correct_2.npy"), np.array(corr_percents[1]))
            np.save(os.path.join(folder, "correct_3.npy"), np.array(corr_percents[2]))
            np.save(os.path.join(folder, "decay_1.npy"), np.array(decay_losses[0]))
            np.save(os.path.join(folder, "decay_2.npy"), np.array(decay_losses[1]))
            np.save(os.path.join(folder, "decay_3.npy"), np.array(decay_losses[2]))
            np.save(os.path.join(folder, "sparse_1.npy"), np.array(sparse_losses[0]))
            np.save(os.path.join(folder, "sparse_2.npy"), np.array(sparse_losses[1]))
            np.save(os.path.join(folder, "sparse_3.npy"), np.array(sparse_losses[2]))
            np.save(os.path.join(folder, "self_1.npy"), np.array(self_losses[0]))
            np.save(os.path.join(folder, "self_2.npy"), np.array(self_losses[1]))
            np.save(os.path.join(folder, "self_3.npy"), np.array(self_losses[2]))
            np.save(os.path.join(folder, "amp_1.npy"), np.array(amp_losses[0]))
            np.save(os.path.join(folder, "amp_2.npy"), np.array(amp_losses[1]))
            np.save(os.path.join(folder, "amp_3.npy"), np.array(amp_losses[2]))
            np.save(os.path.join(folder, "info_1.npy"), np.array(info_losses[0]))
            np.save(os.path.join(folder, "info_2.npy"), np.array(info_losses[1]))
            np.save(os.path.join(folder, "info_3.npy"), np.array(info_losses[2]))

    _, _ = train(epoch)
    test_err[epoch+1], test_cost[epoch+1] = test(epoch)
    train_err[epoch+1], train_cost[epoch+1] = test(epoch, train_set=True)

    if folder is not None:
        np.save(os.path.join(folder, "train_err.npy"), train_err)
        np.save(os.path.join(folder, "train_cost.npy"), train_cost)
        np.save(os.path.join(folder, "test_err.npy"), test_err)
        np.save(os.path.join(folder, "test_cost.npy"), test_cost)

# Measure weight symmetry one final time
if do_rdd and rdd_every_epoch:
    for i in range(3):
        if i == 0:
            layer = net.fc1
        elif i == 1:
            layer = net.fc2
        else:
            layer = net.fc3

        weight    = layer.weight.data.cpu().numpy()
        fb_weight = layer.fb_weight.data.cpu().numpy().T
        x         = np.random.uniform(0, 1, size=(weight.shape[1], 1))

        if np.sum(fb_weight != 0) > 0:
            corr_percent = 100*np.sum((np.sign(fb_weight.T) == np.sign(weight)) & (fb_weight.T != 0))/np.sum(fb_weight.T != 0)
        else:
            corr_percent = 0

        corr_percents[i].append(corr_percent)

        decay_loss  = np.linalg.norm(np.dot(weight, x))**2 + np.linalg.norm(np.dot(x.T, fb_weight))**2
        sparse_loss = np.linalg.norm(weight)**2 + np.linalg.norm(fb_weight)**2
        self_loss   = -np.trace(np.dot(fb_weight, weight))
        amp_loss    = -np.trace(np.dot(x.T, np.dot(fb_weight, np.dot(weight, x))))
        info_loss   = np.linalg.norm(x - np.dot(fb_weight, np.dot(weight, x)))**2

        decay_losses[i].append(decay_loss)
        sparse_losses[i].append(sparse_loss)
        self_losses[i].append(self_loss)
        amp_losses[i].append(amp_loss)
        info_losses[i].append(info_loss)

        if folder is not None:
            np.save(os.path.join(folder, "weight_{}.npy".format(i+1)), weight)
            np.save(os.path.join(folder, "fb_weight_{}.npy".format(i+1)), fb_weight)
            np.save(os.path.join(folder, "beta_{}.npy".format(i+1)), beta)

    if folder is not None:
        np.save(os.path.join(folder, "correct_1.npy"), np.array(corr_percents[0]))
        np.save(os.path.join(folder, "correct_2.npy"), np.array(corr_percents[1]))
        np.save(os.path.join(folder, "correct_3.npy"), np.array(corr_percents[2]))
        np.save(os.path.join(folder, "decay_1.npy"), np.array(decay_losses[0]))
        np.save(os.path.join(folder, "decay_2.npy"), np.array(decay_losses[1]))
        np.save(os.path.join(folder, "decay_3.npy"), np.array(decay_losses[2]))
        np.save(os.path.join(folder, "sparse_1.npy"), np.array(sparse_losses[0]))
        np.save(os.path.join(folder, "sparse_2.npy"), np.array(sparse_losses[1]))
        np.save(os.path.join(folder, "sparse_3.npy"), np.array(sparse_losses[2]))
        np.save(os.path.join(folder, "self_1.npy"), np.array(self_losses[0]))
        np.save(os.path.join(folder, "self_2.npy"), np.array(self_losses[1]))
        np.save(os.path.join(folder, "self_3.npy"), np.array(self_losses[2]))
        np.save(os.path.join(folder, "amp_1.npy"), np.array(amp_losses[0]))
        np.save(os.path.join(folder, "amp_2.npy"), np.array(amp_losses[1]))
        np.save(os.path.join(folder, "amp_3.npy"), np.array(amp_losses[2]))
        np.save(os.path.join(folder, "info_1.npy"), np.array(info_losses[0]))
        np.save(os.path.join(folder, "info_2.npy"), np.array(info_losses[1]))
        np.save(os.path.join(folder, "info_3.npy"), np.array(info_losses[2]))