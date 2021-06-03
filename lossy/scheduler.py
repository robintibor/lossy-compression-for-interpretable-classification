from torch.optim.lr_scheduler import LambdaLR


class AlsoScheduleWeightDecay(object):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.base_weight_decays = [
            g['weight_decay'] for g in scheduler.optimizer.param_groups]
        self.adjust_weight_decays()


    def step(self, *args, **kwargs):
        # First make step of original scheduler,
        # for learning rate and now you can access last_lr
        # to know multiplier for this step
        self.scheduler.step(*args, **kwargs)
        self.adjust_weight_decays()

    def adjust_weight_decays(self):
        # Now also schedule weight decay
        for param_group, base_lr, last_lr, base_wd in zip(
                self.scheduler.optimizer.param_groups,
                self.scheduler.base_lrs,
                self.scheduler.get_last_lr(),
                self.base_weight_decays):
            multiplier = last_lr / base_lr
            param_group['weight_decay'] = multiplier * base_wd


class WarmupBefore(object):
    def __init__(self, scheduler, n_warmup_steps):
        self.base_lrs = scheduler.base_lrs
        self.scheduler = scheduler
        self.optimizer = self.scheduler.optimizer
        # for this to work properly,
        # have to assume other optimizer left
        # original learning rates untouched,
        # or at least that we want to warm up to these learning rates now
        self.n_warmup_steps = n_warmup_steps
        self.warmup_scheduler = LambdaLR(
            scheduler.optimizer, lr_lambda=lambda i_step: float(i_step) / n_warmup_steps)
        self.last_epoch = 0

    def step(self, *args, **kwargs):
        if self.last_epoch < self.n_warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()
        self.last_epoch += 1

    def get_last_lr(self):
        if self.last_epoch < self.n_warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.scheduler.get_last_lr()


class NoOpScheduler(LambdaLR):
    def __init_(self, optimizer):
        super().__init__(optimizer, lr_lambda=lambda *args: 1)
