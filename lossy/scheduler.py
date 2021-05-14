class AlsoScheduleWeightDecay(object):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.base_weight_decays = [
            g['weight_decay'] for g in scheduler.optimizer.param_groups]

    def step(self, *args, **kwargs):
        # First make step of original scheduler,
        # for learning rate and now you can access last_lr
        # to know multiplier for this step
        self.scheduler.step(*args, **kwargs)
        # Now also schedule weight decay
        for param_group, base_lr, last_lr, base_wd in zip(
                self.scheduler.optimizer.param_groups,
                self.scheduler.base_lrs,
                self.scheduler.get_last_lr(),
                self.base_weight_decays):
            multiplier = last_lr / base_lr
            param_group['weight_decay'] = multiplier * base_wd
