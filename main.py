import pandas as pd
from dataset import *
import torch
import scipy.spatial.distance as distance
from scipy import stats
from evaluation import predict_dataset, eval_predictions_multi
from utils import set_seeds, min_max_norm
import kornia
from losses import *
import json
import argparse


def main(args, seed=42):

    train_gpu = torch.cuda.is_available()
    set_seeds(seed, train_gpu)

    dir_dataframe_train = './local_data/datasets/SICAPv2/partition/Validation/Val1/Train.xlsx'
    dir_dataframe_val = './local_data/datasets/SICAPv2/partition/Validation/Val1/Test.xlsx'
    dir_dataframe_test = './local_data/datasets/SICAPv2/partition/Test/Test.xlsx'
    classes = ['NC', 'G3', 'G4', 'G5']
    augmentation = True
    temperature = 1/1
    Ta = 0.01
    la_loss = 'l2'
    sr_mode = 'last'  # 'last', 'spatial'

    experiment_name = args.experiment
    experiment_name_teacher = args.experiment_name_teacher
    dir_images = args.dir_images
    target_shape = args.target_shape
    bs = args.bs
    lr = args.lr
    epochs = args.epochs
    input_shape = args.input_shape
    alpha_kd = args.alpha_kd
    alpha_fm = args.alpha_fm
    alpha_La = args.alpha_La
    alpha_SR = args.alpha_Sr
    normalization = args.normalization
    level_attention = int(args.level_attention)

    # Prepare folders
    dir_results = './results/' + experiment_name + '/'
    if not os.path.isdir(dir_results):
        os.makedirs(dir_results)

    # Prepare dataset and train generator
    dataset_train = Dataset(dir_images, pd.read_excel(dir_dataframe_train), classes=classes,
                            input_shape=(3, input_shape, input_shape),
                            labels=len(classes), augmentation=False, preallocate=True)
    dataset_val = Dataset(dir_images, pd.read_excel(dir_dataframe_val), classes=classes,
                          input_shape=(3, input_shape, input_shape),
                          labels=len(classes), augmentation=False, preallocate=True)
    dataset_test = Dataset(dir_images, pd.read_excel(dir_dataframe_test), classes=classes,
                           input_shape=(3, input_shape, input_shape),
                           labels=len(classes), augmentation=False, preallocate=True)

    train_generator = Generator(dataset_train, bs, shuffle=True, balance=True)

    # Prepare model backbone
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(model.features[0:5],
                                   model.features[5:10],
                                   model.features[10:17],
                                   model.features[17:24],
                                   model.features[24:29],
                                   torch.nn.AdaptiveAvgPool2d(1))
    classifier = torch.nn.Linear(512, len(classes))
    proj = torch.nn. Sequential(
                                torch.nn.Conv2d(int(512 / (2 ** (int(np.log2(input_shape/target_shape)) - 1))), 512, (1, 1)),
                                torch.nn.Conv2d(512, len(classes), (1, 1))
                                )

    # Prepare augmentations module
    transforms = torch.nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.RandomRotation(degrees=45, p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, scale=(0.95, 1.20), p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, translate=(0.05, 0), p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0., p=0.5),
    )

    # Set loss function
    Lce = torch.nn.CrossEntropyLoss()

    # Prepare optimizer
    opt = torch.optim.Adam(lr=lr, params=list(backbone.parameters()) + list(classifier.parameters()) +
                                         list(proj.parameters()))

    # Load teacher model
    if (alpha_kd + alpha_fm + alpha_La + alpha_SR) > 0:
        teacher_dir = './results/' + experiment_name_teacher + '/'\
                      + experiment_name_teacher + '_' + str(seed) + '_model_best.pth'
        teacher = torch.load(teacher_dir)
        teacher.eval()

    if train_gpu:
        backbone = backbone.cuda()
        transforms = transforms.cuda()
        classifier = classifier.cuda()
        if (alpha_kd + alpha_fm + alpha_La + alpha_SR) > 0:
            teacher = teacher.cuda()
            if alpha_SR > 0.:
                proj = proj.cuda()

    best_val = 0.0
    for i_epoch in range(epochs):
        L_epoch = 0.0
        Lce_epoch = 0.0
        Lkd_epoch = 0.0
        Lfm_epoch = 0.0
        La_epoch = 0.0
        Lsr_epoch = 0.0
        Y_t = []
        Yhat_t = []

        for i_iteration, (X, Y) in enumerate(train_generator):

            ####################
            # --- Training epoch
            model.train()

            X = X.float()
            Y = Y.float()

            # Move to cuda
            if train_gpu:
                X = X.cuda()
                Y = Y.cuda()

            Y_ohot = Y.clone()
            Y = torch.argmax(Y, -1)

            # Augmentation
            if augmentation:
                X = transforms(X.clone())

            # Downsample input image
            X_student = torch.nn.AvgPool2d(int(input_shape / target_shape))(X)
            X_student = torch.nn.UpsamplingBilinear2d(int(input_shape))(X_student)

            # Forward Student
            F_s = []
            for i in range(0, len(backbone)):
                if i == 0:
                    x = backbone[i](X_student)
                else:
                    x = backbone[i](x)

                if i >= (len(backbone) + level_attention - 1):
                    F_s.append(x)
                else:
                    F_s.append([])

            logits_s = classifier(torch.squeeze(F_s[-1]))

            # Forward Teacher
            if (alpha_kd + alpha_fm + alpha_La + alpha_SR) > 0:
                F_t = []
                for i in range(0, len(teacher[0])):
                    if i == 0:
                        x = teacher[0][i](X)
                    else:
                        x = teacher[0][i](x)

                    if i >= (len(teacher[0]) + level_attention - 1):
                        F_t.append(x)
                    else:
                        F_t.append([])

                logits_t = teacher[2](torch.squeeze(F_t[-1]))

            # CE loss
            Lcrossentropy = Lce(logits_s, Y)
            L_iteration = Lcrossentropy

            # Knowledge distillation loss
            if alpha_kd > 0:
                target = torch.softmax(logits_t/temperature, -1)
                target = target.detach()
                Lkd = torch.mean(torch.sum(-target * torch.log_softmax(logits_s, dim=-1), dim=-1))
                L_iteration += alpha_kd * Lkd

            # Feature matching loss
            if alpha_fm > 0:

                F_target = F_t[-2].detach()
                F_student = torch.nn.functional.interpolate(F_s[-2],
                                                            size=(F_target.shape[-1], F_target.shape[-1]),
                                                            mode='bilinear',
                                                            align_corners=True)

                Lfm = torch.sqrt(torch.mean(torch.square(F_student - F_target)))
                L_iteration += alpha_fm * Lfm

            # Proposed Attention matching loss
            if alpha_La > 0:
                acts_student = []
                acts_teacher = []
                for iClass in np.arange(len(classes)):

                    # Obtain attention for student
                    output_s = logits_s[:, iClass]
                    activation_s = F_s[-1+level_attention]

                    gradients = \
                    torch.autograd.grad(torch.sum(output_s), activation_s, grad_outputs=None, retain_graph=True,
                                        create_graph=True, only_inputs=True, allow_unused=True)[0]
                    gradients = torch.mean(gradients, dim=[2, 3])
                    gradients = gradients.unsqueeze(-1).unsqueeze(-1)
                    attention_student = torch.sum(torch.relu(gradients * activation_s), 1)

                    # Obtain attention for teacher
                    output_t = logits_t[:, iClass]
                    activation_t = F_t[-1+level_attention]

                    gradients = \
                    torch.autograd.grad(torch.sum(output_t), activation_t, grad_outputs=None, retain_graph=True,
                                        create_graph=True, only_inputs=True, allow_unused=True)[0]
                    gradients = torch.mean(gradients, dim=[2, 3])
                    gradients = gradients.unsqueeze(-1).unsqueeze(-1)
                    attention_teacher = torch.sum(torch.relu(gradients * activation_t), 1)
                    attention_teacher = attention_teacher.detach()

                    acts_student.append(attention_student.unsqueeze(-1))
                    acts_teacher.append(attention_teacher.unsqueeze(-1))

                acts_student = torch.cat(acts_student, -1)
                acts_teacher = torch.cat(acts_teacher, -1)

                # Normalize attention maps
                if 'softmax' in normalization:
                    acts_teacher = torch.softmax(acts_teacher.view((bs, -1))/Ta, -1).view(
                        (bs, acts_teacher.shape[-1], acts_teacher.shape[-1]))
                    acts_student = torch.softmax(acts_student.view((bs, -1))/Ta, -1).view(
                        (bs, acts_teacher.shape[-1], acts_teacher.shape[-1]))
                if 'minmax' in normalization:
                    acts_teacher = min_max_norm(acts_teacher)
                    acts_student = min_max_norm(acts_student)

                if la_loss == 'bce':
                    La = torch.nn.BCELoss()(acts_student.view(bs, -1), acts_teacher.view(bs, -1))
                elif la_loss == 'kl':
                    La = torch.nn.KLDivLoss(reduction="batchmean")(torch.log(acts_student.view((bs, -1))),
                                                                   acts_teacher.view((bs, -1)))
                elif la_loss == 'l2':
                    La = torch.mean(torch.sqrt(torch.mean(torch.square(acts_teacher - acts_student).view((bs, -1)), -1)))

                L_iteration += alpha_La * La

            # Softmax regression loss
            if alpha_SR > 0.:
                if sr_mode == 'last':
                    target = logits_t.detach()
                    logits_s_to_t = teacher[2](torch.squeeze(F_s[-1]))

                elif sr_mode == 'spatial':

                    scale = int(np.log2(input_shape/target_shape))
                    x = F_s[-2-scale]
                    logits_s_to_t = torch.squeeze(proj(x))

                    target = F_t[-2].permute((0, 2, 3, 1))
                    target = teacher[2](target).permute((0, 3, 1, 2)).detach()

                Lsr = torch.sqrt(torch.mean(torch.square(logits_s_to_t - target)))
                L_iteration += alpha_SR * Lsr

            # Backward and weights update
            L_iteration.backward()  # Backward
            opt.step()              # Update weights
            opt.zero_grad()         # Clear gradients

            L_epoch += L_iteration.cpu().detach().numpy() / len(train_generator)
            Lce_epoch += Lcrossentropy.cpu().detach().numpy() / len(train_generator)
            if alpha_kd > 0:
                Lkd_epoch += Lkd.cpu().detach().numpy() / len(train_generator)
            if alpha_fm > 0:
                Lfm_epoch += Lfm.cpu().detach().numpy() / len(train_generator)
            if alpha_La > 0:
                La_epoch += La.cpu().detach().numpy() / len(train_generator)
            if alpha_SR > 0.:
                Lsr_epoch += Lsr.cpu().detach().numpy() / len(train_generator)
            Y_t.append(Y.cpu().detach().numpy())
            Yhat_t.append(torch.softmax(logits_s, dim=1).cpu().detach().numpy())

            # Display training information per iteration
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.6f}".format(
                i_epoch + 1, epochs, i_iteration + 1, len(train_generator), Lcrossentropy.cpu().detach().numpy())
            if alpha_kd > 0:
                info += " || Lkd={:.6f}".format(Lkd.cpu().detach().numpy())
            if alpha_fm > 0:
                info += " || Lfm={:.6f}".format(Lfm.cpu().detach().numpy())
            if alpha_La > 0:
                info += " || La={:.6f}".format(La.cpu().detach().numpy())
            if alpha_SR > 0.:
                info += " || Lsr={:.6f}".format(Lsr.cpu().detach().numpy())
            print(info, end='\r')

        # Train evaluation
        Y_t = np.concatenate(Y_t)
        Yhat_t = np.concatenate(Yhat_t)
        k_tr, acc_tr = eval_predictions_multi(Y_t, np.argmax(Yhat_t, 1), print_conf=False)

        # Val evaluation
        print('Val evaluation...', end='\n')
        Y, Yhat = predict_dataset(dataset_val,
                                  torch.nn.Sequential(torch.nn.AvgPool2d(int(input_shape/target_shape)),
                                                      torch.nn.UpsamplingBilinear2d(int(input_shape)),
                                                      backbone, torch.nn.Flatten(), classifier, torch.nn.Softmax(dim=1)),
                                  use_cuda=train_gpu)
        k_val, acc_val = eval_predictions_multi(np.argmax(Y, 1), np.argmax(Yhat, 1), print_conf=True)

        # Test evaluation
        Y, Yhat = predict_dataset(dataset_test,
                                  torch.nn.Sequential(torch.nn.AvgPool2d(int(input_shape/target_shape)),
                                                      torch.nn.UpsamplingBilinear2d(int(input_shape)),
                                                      backbone, torch.nn.Flatten(), classifier, torch.nn.Softmax(dim=1)),
                                  use_cuda=train_gpu)
        k_te, acc_te = eval_predictions_multi(np.argmax(Y, 1), np.argmax(Yhat, 1), print_conf=True)

        if (k_val - best_val) > 0.01:
            print('Val loss improved... ', end='\n')
            best_val = k_val
            torch.save(torch.nn.Sequential(backbone, torch.nn.Flatten(), classifier, torch.nn.Softmax(dim=1)),
                       dir_results + experiment_name + '_' + str(seed) + '_model_best.pth')
            metrics = {'acc_val': acc_val, 'k_val': k_val, 'acc_te': acc_te, 'k_te': k_te}
            with open(dir_results + experiment_name + '_' + str(seed) + '_metrics_best.json', 'w') as fp:
                json.dump(metrics, fp)

        # Info display (end epoch)
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} -- TRAIN: Acc_tr={:.4f} || k_tr={:.4f} " \
               "-- VAL: Acc_val={:.4f} || k_val={:.4f} -- TEST: Acc_te={:.4f} || k_te={:.4f} -- EAT: "\
            .format(i_epoch + 1, epochs, len(train_generator), len(train_generator), L_epoch,
                    k_tr, acc_tr, acc_val, k_val, acc_te, k_te)
        if alpha_kd > 0:
            info += " || Lkd={:.6f}".format(Lkd_epoch)
        if alpha_fm > 0:
            info += " || Lfm={:.6f}".format(Lfm_epoch)
        if alpha_La > 0:
            info += " || La={:.6f}".format(La_epoch)
        if alpha_SR > 0.:
            info += " || Lsr={:.6f}".format(Lsr_epoch)
        print(info, end='\n')

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Settings
    parser.add_argument("--dir_images", default="./local_data/datasets/SICAPv2/images/", type=str)
    parser.add_argument("--experiment", default="experiment_id", type=str)
    parser.add_argument("--experiment_name_teacher", default="512_KD_0_FM_0_SR_0_AM_0", type=str)
    # Hyper-params training
    parser.add_argument("--input_shape", default=512, type=int)
    parser.add_argument("--target_shape", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--bs", default=32, type=int)
    # Different terms
    parser.add_argument("--alpha_kd", default=0., type=float)
    parser.add_argument("--alpha_fm", default=0., type=float)
    parser.add_argument("--alpha_La", default=0., type=float)
    parser.add_argument("--alpha_Sr", default=0., type=float)
    parser.add_argument("--level_attention", default=-1., type=float)
    parser.add_argument("--normalization", default="minmax", type=str)

    args = parser.parse_args()

    seeds = [22, 32, 43]
    metrics_all = np.zeros((len(seeds), 4))
    for i_seed in np.arange(0, 3):
        metrics = main(args, seed=seeds[i_seed])
        metrics_all[i_seed, :] = np.array([metrics['acc_val'], metrics['k_val'], metrics['acc_te'], metrics['k_te']])

    metrics = {'acc_val': np.mean(metrics_all, 0)[0],
               'acc_val_std': np.std(metrics_all, 0)[0],
               'k_val': np.mean(metrics_all, 0)[1],
               'k_val_std': np.std(metrics_all, 0)[1],
               'acc_te': np.mean(metrics_all, 0)[2],
               'acc_te_std': np.std(metrics_all, 0)[2],
               'k_te': np.mean(metrics_all, 0)[3],
               'k_te_std': np.std(metrics_all, 0)[3]}

    experiment_name = args.experiment
    dir_results = './results/' + experiment_name + '/'
    with open(dir_results + experiment_name + '_' + 'avg' + '_metrics_best.json', 'w') as fp:
        json.dump(metrics, fp)