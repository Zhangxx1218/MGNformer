import os, sys, time
import logging, logzero
from logzero import logger
from tqdm import tqdm
from pathlib import Path
import click
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_printoptions(precision=4, linewidth=400, threshold=sys.maxsize, sci_mode=False)


from model import TFGAT, make_dataloader, make_dataset, make_optimizer, make_scheduler, STUNET
from model.configs import get_default_configs, get_modified_configs
from model.utils import load_checkpoint, save_checkpoint, backup_file, set_seed, draw
from model.loss import masked_mse, masked_mse_along_dim, multimodal_masked_mse_along_dim, masked_focal
from model.loss import maskedNLL
from model.model_args import args
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@click.group()
def clk():
    pass


@clk.command()
@click.option("-c", "--config-file", type=str, default="")
@click.option("-i", "--info-msg", type=str, default="")
@click.argument("cmd-config", nargs=-1)
def train(config_file, info_msg, cmd_config):
    cfg = get_modified_configs(config_file, cmd_config)
    set_seed(cfg)

    if not os.path.exists(cfg.workspace):
        os.makedirs(cfg.workspace)
    logzero.logfile(f"{cfg.workspace}/train.log")
    logzero.loglevel(level=logging.INFO)
    logger.info(info_msg)
    logger.info("Starting train process...")
    logger.info(cfg)

    logger.info("Building dataloaders...")
    trSet, valSet, _ = make_dataset(cfg.dataset)
    trDataLoader, valDataLoader = make_dataloader(trSet, valSet, batch_size=cfg.training.batch_size,
                                                  num_workers=cfg.training.num_workers,
                                                  multi_agents=cfg.network.multi_agents)

    logger.info("Building model...")
    model = STUNET(cfg.network, trSet)
    
    teacher.eval()
    
    logger.info(f"Loading checkpoint from {cfg.workspace}")
    start_epoch = load_checkpoint(cfg.workspace, device=cfg.device, epoch=0, model=model)
    if start_epoch > cfg.training.epochs:
        logger.info(f"The training corresponding to config file {Path(config_file).name} was over.")
        return
    elif start_epoch > 1:
        logger.info(f"Loaded checkpoint successfully! Start epoch is {start_epoch}.")
    else:
        logger.info(f"Cannot find pre-trained checkpoint. Start training from epoch 1.")
        backup_file(cfg)
        logger.info(f"Created backup successfully!")

    _NUM_CUDA_DEVICES = 0
    if cfg.device == 'cuda':
        _NUM_CUDA_DEVICES = torch.cuda.device_count()
        if _NUM_CUDA_DEVICES < 1:
            raise ValueError(f"cannot perform cuda training due to insufficient cuda device.")
        logger.info(f"{_NUM_CUDA_DEVICES} cuda device found!")
        model = model.cuda()
        teacher = teacher.cuda()
        if _NUM_CUDA_DEVICES > 1:
            model = nn.DataParallel(model)
            logger.info(f"Parallelized model on {_NUM_CUDA_DEVICES} cuda devices.")

    logger.info("Building optimizer...")
    optimizer = make_optimizer(model, cfg.optimizer)

    logger.info("Building scheduler...")
    last_epoch = start_epoch - 1
    if cfg.training.scheduled_by_steps:
        last_epoch *= len(trDataLoader)
    scheduler = make_scheduler(optimizer=optimizer, cfg=cfg.training, last_epoch=last_epoch)

    focal = masked_focal
    alpha = torch.Tensor(cfg.training.alpha, dtype=torch.float, device=cfg.device) if len(
        cfg.training.alpha) == trSet.num_maneuvers else None
    logger.info("Start training!")
    

    for epoch in range(start_epoch, cfg.training.epochs + 1):

        t0 = time.time()

        running_loss = 0
        running_loss1 = 0
        running_loss2 = 0
        running_acc = 0
        data_time = 0
        forward_time = 0
        backward_time = 0
        train_time = 0
        teacher.eval()
        model.train()

        # train loop
        for iter, batch in enumerate(tqdm(trDataLoader, ncols=70)):
            t1 = time.time()
            data_time += t1 - t0

            batch = [item.to(cfg.device) for item in batch]
            hist, fut, pad_mask, fut_mask, maneuver = batch
            hist = hist[:, :cfg.network.num_agents]
            # print(hist.shape) 128,20,16,5
            pad_mask = pad_mask[:, :, :cfg.network.num_agents]
            if cfg.network.multi_agents:
                maneuver = maneuver[:, :cfg.network.num_agents]
                fut = fut[:, :cfg.network.num_agents]
                fut_mask = fut_mask[:, :cfg.network.num_agents]
            track_xy, maneuver_prob, eha_s, token_s, lm_s, lp_s = model(hist, pad_mask, fut, maneuver)
            maneuver = maneuver.argmax(-1)

            t2 = time.time()
            if cfg.network.use_nll and epoch > cfg.training.pre_epochs and epoch < cfg.training.end_epochs:
                loss1 = maskedNLL(track_xy, fut, fut_mask)
            else:
                loss1 = masked_mse(track_xy[..., :2], fut, fut_mask) 

            running_loss1 = 0.9 * running_loss1 + 0.1 * loss1.item()

            if cfg.network.maneuver:
                if cfg.network.multi_agents:
                    bool_mask = pad_mask[:, 0, :].bool()
                    loss2 = focal(maneuver_prob, maneuver, mask=bool_mask, alpha=alpha, gamma=cfg.training.gamma)
                    acc = (torch.sum(
                        torch.max(maneuver_prob[bool_mask].data, -1)[1] == maneuver[bool_mask].data)).item() / \
                          maneuver[bool_mask].shape[0]
                else:
                    loss2 = focal(maneuver_prob, maneuver, mask=None, alpha=alpha, gamma=cfg.training.gamma)
                    acc = (torch.sum(torch.max(maneuver_prob.data, -1)[1] == maneuver.data)).item() / maneuver.shape[0]
                running_acc = 0.9 * running_acc + 0.1 * acc
                running_loss2 = 0.9 * running_loss2 + 0.1 * loss2.item()
            else:
                loss = loss1
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()

            forward_time += t2 - t1
            backward_time += t3 - t2
            train_time += t3 - t0

            if (iter % 1000) == 0 and iter > 0:
                eta = train_time / iter * (len(trDataLoader) - iter)
                progress = iter / len(trDataLoader) * 100
                if cfg.network.maneuver:
                    logger.info(f"Epoch: {epoch:04d}, Progress: {progress:5.2f} [%], ETA: {eta:8.2f} [s] | "
                                f"Loss: {running_loss:8.3f}, Acc: {(running_acc * 100): 8.3f} [%], Learning Rate: {(cfg.optimizer.base_lr * scheduler.get_last_lr_factor()):.3e}")
                    logger.info(f"MSE Loss velocity: {running_loss1:.3f}, CE Loss: {running_loss2:.3f}")
                else:
                    logger.info(f"Epoch: {epoch:04d}, Progress: {progress:5.2f} [%], ETA: {eta:8.2f} [s] | "
                                f"Loss: {running_loss:8.3f}, Learning Rate: {(cfg.optimizer.base_lr * scheduler.get_last_lr_factor()):.3e}")
            if cfg.training.scheduled_by_steps:
                scheduler.step()

            t0 = time.time()

        if not cfg.training.scheduled_by_steps:
            scheduler.step()

        logger.info(f"Epoch {epoch:04d} completed, Time: {train_time:8.2f} [s]. "
                    f"Data time:{data_time:8.2f} [s], Forward time: {forward_time:8.2f} [s], Backward time: {backward_time:8.2f} [s]")
        logger.info(f"Saving checkpoint {epoch}")
        save_checkpoint(epoch, cfg.workspace, model=model)
        logger.info("-" * 20)

        t0 = time.time()
        rmse_list = []
        ade_list = []
        fde_list = []
        val_loss1 = 0
        val_loss2 = 0
        val_loss3 = 0
        val_acc = 0
        val_loss_time = 0
        val_div_time = 0
        val_self_loss_time = 0
        val_self_div_time = 0
        model.eval()
        len_val_dataloader = len(valDataLoader)
        indexs = torch.LongTensor([i for i in range(cfg.evaluation.batch_size)])
        # validation loop
        with torch.no_grad():
            for batch in tqdm(valDataLoader, ncols=70):
                batch = [item.to(cfg.device) for item in batch]
                hist, fut, pad_mask, fut_mask, maneuver = batch
                hist = hist[:, :cfg.network.num_agents]
                pad_mask = pad_mask[:, :, :cfg.network.num_agents]
                if cfg.network.multi_agents:
                    maneuver = maneuver[:, :cfg.network.num_agents]
                    fut = fut[:, :cfg.network.num_agents]
                    fut_mask = fut_mask[:, :cfg.network.num_agents]

                track, maneuver_prob, eha_t, token_t, lm_t, lp_t = model(hist, pad_mask, fut, maneuver)
                maneuver = maneuver.argmax(-1)
                if cfg.network.mult_traj:
                    #print("track shape:", len(track))
                    track = torch.stack(track)
                    #print("track shape:", len(track))
                    #print("maneuver shape:", maneuver.shape)
                    #print("indexs shape:", indexs.shape)
                    track_xy = track[maneuver, indexs, ...]
                else:
                    track_xy = track
                # print(hist[3, :3, :, 1])
                # print(hist_xy[3, :, :3, 1].T)
                # print(fut[3, :3, :, 1])
                # print(fut_xy[3, :3, :, 1])
                val_loss1 += masked_mse(track_xy[..., :2], fut, fut_mask).item()
                val_loss2 = val_loss1
                #指标计算-RMSE
                rmse = torch.sqrt(torch.tensor(masked_mse(track_xy[..., :2], fut, fut_mask).item())).item()
                rmse_list.append(rmse)
                ade = torch.mean(torch.sqrt(torch.sum((track_xy[..., :2] - fut) ** 2, dim=-1))).item()
                ade_list.append(ade)
                fde = torch.sqrt(torch.sum((track_xy[..., -1, :2] - fut[..., -1, :2]) ** 2, dim=-1)).mean().item()
                fde_list.append(fde)
                if cfg.network.maneuver:
                    if cfg.network.multi_agents:
                        bool_mask = pad_mask[:, 0, :].bool()
                        loss3 = focal(maneuver_prob, maneuver, mask=bool_mask, alpha=alpha, gamma=cfg.training.gamma)
                        acc = (torch.sum(
                            torch.max(maneuver_prob[bool_mask].data, -1)[1] == maneuver[bool_mask].data)).item() / \
                              maneuver[bool_mask].shape[0]
                    else:
                        loss3 = focal(maneuver_prob, maneuver, mask=None, alpha=alpha, gamma=cfg.training.gamma)
                        acc = (torch.sum(torch.max(maneuver_prob.data, -1)[1] == maneuver.data)).item() / \
                              maneuver.shape[0]
                    val_loss3 += loss3
                    val_acc += acc
                if cfg.evaluation.multimodal and cfg.network.maneuver:
                    loss_time, div_time = multimodal_masked_mse_along_dim(track_xy[..., :2], fut, dim=-2,
                                                                          mask=fut_mask,
                                                                          reduction='none')
                else:
                    if cfg.network.multi_agents:
                        self_loss_time, self_div_time, loss_time, div_time = masked_mse_along_dim(track_xy[..., :2],
                                                                                                  fut,
                                                                                                  dim=-2, mask=fut_mask,
                                                                                                  reduction='none',
                                                                                                  multi_agents=cfg.network.multi_agents)
                    else:
                        self_loss_time, self_div_time = masked_mse_along_dim(track_xy[..., :2], fut, dim=-2,
                                                                             mask=fut_mask,
                                                                             reduction='none',
                                                                             multi_agents=cfg.network.multi_agents)

                val_self_loss_time += self_loss_time
                val_self_div_time += self_div_time
                if cfg.network.multi_agents:
                    val_loss_time += loss_time
                    val_div_time += div_time
        average_rmse = sum(rmse_list) / len(rmse_list) * 0.3048
        average_ade = sum(ade_list) / len(ade_list) * 0.3048
        average_fde = sum(fde_list) / len(fde_list) * 0.3048

        logger.info(f"Average RMSE: {average_rmse:.3f}")
        logger.info(f"Average ADE: {average_ade:.3f}")
        logger.info(f"Average FDE: {average_fde:.3f}")
        
        # print(val_self_loss_time)
        # print(val_self_div_time)
        val_time = time.time() - t0
        mse_along_time = (((val_self_loss_time / val_self_div_time).cpu().numpy()) ** 0.5) * 0.3048
        logger.info(f"self MSE Loss along time: {mse_along_time}")
        if cfg.network.multi_agents:
            mse_along_time = (((val_loss_time / val_div_time).cpu().numpy()) ** 0.5) * 0.3048
            logger.info(f"MSE Loss along time: {mse_along_time}")
        if cfg.network.maneuver:
            val_loss = val_loss1 + val_loss2 + val_loss3
            logger.info(
                f"Validation time: {val_time:8.2f} [s], Loss: {(val_loss / len_val_dataloader):8.2f}, Acc: {(val_acc * 100 / len_val_dataloader):8.2f} [%]")
            logger.info(
                f"MSE Loss track: {(val_loss1 / len_val_dataloader):.3f}, MSE Loss velocity {(val_loss2 / len_val_dataloader):.3f}, CE Loss: {(val_loss3 / len_val_dataloader):.3f}")
        else:
            val_loss = val_loss1 + val_loss2
            logger.info(f"Validation time: {val_time:8.2f} [s], Loss: {(val_loss / len_val_dataloader):8.2f}")
            logger.info(
                f"MSE Loss track: {(val_loss1 / len_val_dataloader):.3f}, MSE Loss velocity {(val_loss2 / len_val_dataloader):.3f}")
        logger.info("-" * 20)


@clk.command()
@click.option("-c", "--config-file", type=str, default="")
@click.argument("cmd-config", nargs=-1)
def eval(config_file, cmd_config):
    cfg = get_modified_configs(config_file, cmd_config)

    if not os.path.exists(cfg.workspace):
        raise FileNotFoundError
    logzero.logfile(f"{cfg.workspace}/eval.log")
    logzero.loglevel(level=logging.INFO)
    logger.info("Starting evaluation process...")
    logger.info(cfg.evaluation)

    logger.info("Building dataloader...")
    _, _, tsSet = make_dataset(cfg.dataset)
    tsDataLoader = make_dataloader(tsSet, batch_size=cfg.evaluation.batch_size, num_workers=cfg.evaluation.num_workers,
                                   multi_agents=cfg.network.multi_agents)

    logger.info("Building model...")
    #model = TFGAT(cfg.network, tsSet)
    model = STUNET(cfg.network, tsSet)

    logger.info(f"Loading checkpoint from {cfg.workspace}")
    try:
        epoch = load_checkpoint(cfg.workspace, device=cfg.device, epoch=cfg.evaluation.epoch, model=model)
    except FileNotFoundError:
        logger.error(f"Failed to load checkpoint at epoch {cfg.evaluation.epoch}.")
        return
    except IndexError:
        logger.error(f"Failed to load the {-cfg.evaluation.epoch}th checkpoint from the end.")
        return
    logger.info(f"Loaded checkpoint at epoch {epoch - 1} successfully!")

    if cfg.device == 'cuda':
        _NUM_CUDA_DEVICES = torch.cuda.device_count()
        if _NUM_CUDA_DEVICES < 1:
            raise ValueError(f"cannot perform cuda training due to insufficient cuda device.")
        logger.info(f"{_NUM_CUDA_DEVICES} cuda device found!")
        model = model.cuda()
        if _NUM_CUDA_DEVICES > 1:
            model = nn.DataParallel(model)
            logger.info(f"Parallelized model on {_NUM_CUDA_DEVICES} cuda devices.")

    model.eval()
    logger.info("Start evaluation!")
    focal = masked_focal
    alpha = torch.Tensor(cfg.training.alpha, dtype=torch.float, device=cfg.device) if len(
        cfg.training.alpha) == tsSet.num_maneuvers else None
    # evaluation loop
    t0 = time.time()
    rmse_list = []
    ade_list = []
    fde_list = []
    val_loss1 = 0
    val_loss2 = 0
    val_loss3 = 0
    val_acc = 0
    val_loss_time = 0
    val_div_time = 0
    val_self_loss_time = 0
    val_self_div_time = 0
    model.eval()
    len_val_dataloader = len(tsDataLoader)
    indexs = torch.LongTensor([i for i in range(cfg.evaluation.batch_size)])
    with torch.no_grad():
        for batch in tqdm(tsDataLoader, ncols=70):
            batch = [item.to(cfg.device) for item in batch]
            hist, fut, pad_mask, fut_mask, maneuver = batch
            hist = hist[:, :cfg.network.num_agents]
            pad_mask = pad_mask[:, :, :cfg.network.num_agents]
            if cfg.network.multi_agents:
                maneuver = maneuver[:, :cfg.network.num_agents]
                fut = fut[:, :cfg.network.num_agents]
                fut_mask = fut_mask[:, :cfg.network.num_agents]

            track, maneuver_prob, eha_s, token_s, lm_s, lp_s = model(hist, pad_mask, fut, maneuver)
            maneuver = maneuver.argmax(-1)
            if cfg.network.mult_traj:
                track = torch.stack(track)
                track_xy = track[maneuver, indexs, ...]
            else:
                track_xy = track
            # 计算均方误差
            val_mse = masked_mse(track_xy[..., :2], fut, fut_mask).item()
            val_loss1 += val_mse
            val_loss2 = val_loss1
            #指标计算-RMSE
            rmse = torch.sqrt(torch.tensor(val_mse)).item()
            rmse_list.append(rmse)
            # 计算平均位移误差（ADE）
            ade = torch.mean(torch.sqrt(torch.sum((track_xy[..., :2] - fut) ** 2, dim=-1))).item()
            ade_list.append(ade)

            # 计算最终位移误差（FDE）
            fde = torch.sqrt(torch.sum((track_xy[..., -1, :2] - fut[..., -1, :2]) ** 2, dim=-1)).mean().item()
            fde_list.append(fde)

            if cfg.evaluation.draw:
                draw(cfg.network.mult_traj, hist, fut, track, maneuver, pad_mask, val_mse)
            if cfg.network.maneuver:
                loss3 = masked_focal(maneuver_prob, maneuver, mask=None, alpha=alpha, gamma=cfg.training.gamma)
                acc = (torch.sum(torch.max(maneuver_prob.data, -1)[1] == maneuver.data)).item() / \
                      maneuver.shape[0]
                val_loss3 += loss3
                val_acc += acc

            self_loss_time, self_div_time = masked_mse_along_dim(track_xy[..., :2], fut, dim=-2,
                                                                 mask=fut_mask,
                                                                 reduction='none',
                                                                 multi_agents=cfg.network.multi_agents)

            val_self_loss_time += self_loss_time
            val_self_div_time += self_div_time

        # 计算平均值
        average_rmse = sum(rmse_list) / len(rmse_list) * 0.3048
        average_ade = sum(ade_list) / len(ade_list) * 0.3048
        average_fde = sum(fde_list) / len(fde_list) * 0.3048

        # 输出指标信息
        logger.info(f"Average RMSE: {average_rmse:.3f}")
        logger.info(f"Average ADE: {average_ade:.3f}")
        logger.info(f"Average FDE: {average_fde:.3f}")

        val_time = time.time() - t0
        mse_along_time = (((val_self_loss_time / val_self_div_time).cpu().numpy()) ** 0.5) * 0.3048
        logger.info(f"self MSE Loss along time: {mse_along_time}")
        if cfg.network.maneuver:
            val_loss = val_loss1 + val_loss2 + val_loss3
            logger.info(
                f"Validation time: {val_time:8.2f} [s], Loss: {(val_loss / len_val_dataloader):8.2f}, Acc: {(val_acc * 100 / len_val_dataloader):8.2f} [%]")
            logger.info(
                f"MSE Loss track: {(val_loss1 / len_val_dataloader):.3f}, MSE Loss velocity {(val_loss2 / len_val_dataloader):.3f}, CE Loss: {(val_loss3 / len_val_dataloader):.3f}")
        else:
            val_loss = val_loss1 + val_loss2
            logger.info(f"Validation time: {val_time:8.2f} [s], Loss: {(val_loss / len_val_dataloader):8.2f}")
            logger.info(
                f"MSE Loss track: {(val_loss1 / len_val_dataloader):.3f}, MSE Loss velocity {(val_loss2 / len_val_dataloader):.3f}")
        logger.info("-" * 20)

if __name__ == '__main__':
    clk()
