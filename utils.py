from tensorboard_logger import log_value
from torch.nn import DataParallel
import torch


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, opts):
    """
    Print values on the screen and log values to tensorboard.
    """
    avg_cost = cost.mean().item()
    print(avg_cost)
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}, reinforce_loss: {}'.format(epoch, batch_id, avg_cost, reinforce_loss))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
    if opts.baseline == 'critic':
        print('grad_norm_critic: {}, clipped: {}'.format(grad_norms[1], grad_norms_clipped[1]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        log_value('avg_cost', avg_cost, step)

        log_value('actor_loss', reinforce_loss.item(), step)
        log_value('nll', -log_likelihood.mean().item(), step)

        log_value('grad_norm', grad_norms[0], step)
        log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            log_value('critic_loss', bl_loss.item(), step)
            log_value('critic_grad_norm', grad_norms[1], step)
            log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)


def maybe_cuda_model(model, cuda, parallel=True):
    """
    Transforms the model to a parallel model with the cuda paramters from the options.
    """
    if cuda:
        model.cuda()

    if parallel and torch.cuda.device_count() > 1:
        model = DataParallel(model)

    return model