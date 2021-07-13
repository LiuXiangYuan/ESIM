import torch
import torch.nn as nn

from sklearn import metrics


def train(model, args, train_loader, val_loader, criterion, optimizer, lr_scheduler=None):
    total_iters = len(train_loader)
    print(f'total_iters: { total_iters }')

    best_score = 0.0
    best_epoch = 0
    patience_counter = 0
    max_patience = args.patience

    for epoch in range(1, args.max_epochs):
        model.train()

        print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        print('Fold{} Epoch {}/{}'.format(args.fold + 1, epoch, args.max_epochs))
        print('-' * 10)

        count = 0
        train_loss = []

        for i, batch in enumerate(train_loader):
            count += 1
            premises = batch["premise"].to(args.device)
            premises_lengths = batch["premise_length"].to(args.device)
            hypotheses = batch["hypothesis"].to(args.device)
            hypotheses_lengths = batch["hypothesis_length"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()

            logits, _ = model(premises, premises_lengths, hypotheses, hypotheses_lengths)
            logits = logits.squeeze(1)

            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
            optimizer.step()

            if i % 1000 == 0 or logits.size()[0] < args.train_batch_size:
                print(
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f}'.format(
                        args.fold + 1, epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'])
                )

            train_loss.append(loss.item())

        print(
            ' Fold:{} Epoch:{}({}/{}) avg_train_loss:{:.3f} '.format(
                args.fold + 1, epoch, count, total_iters, sum(train_loss) / len(train_loader))
        )

        model.eval()
        pres_list = []
        labels_list = []
        for _, batch in enumerate(val_loader):
            premises = batch["premise"].to(args.device)
            premises_lengths = batch["premise_length"].to(args.device)
            hypotheses = batch["hypothesis"].to(args.device)
            hypotheses_lengths = batch["hypothesis_length"].to(args.device)
            labels = batch["label"].to(args.device)

            logits, _ = model(premises, premises_lengths, hypotheses, hypotheses_lengths)
            logits = logits.squeeze(1)
            pres_list += logits.sigmoid().detach().cpu().numpy().tolist()
            labels_list += labels.detach().cpu().numpy().tolist()

        val_auc = metrics.roc_auc_score(labels_list, pres_list, multi_class='ovo')
        val_loss = metrics.log_loss(labels_list, pres_list)
        print('val LogLoss: {:.4f} valAuc: {:.4f}'.format(val_loss, val_auc))

        if lr_scheduler != None:
            lr_scheduler.step(val_auc)

        if val_auc < best_score:
            patience_counter += 1
        else:
            patience_counter = 0

            best_score = val_auc
            best_epoch = epoch
            best_model_out_path = args.output_dir + 'fold_' + str(args.fold + 1) + '_best' + '.pth'
            torch.save(model.state_dict(), best_model_out_path)
            print("save best epoch: {} best auc: {} best log loss: {}".format(best_epoch, val_auc, val_loss))

        if patience_counter >= max_patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    print('Fold{} Best auc score: {:.3f} Best epoch:{}'.format(args.fold + 1, best_score, best_epoch))
    return best_score
