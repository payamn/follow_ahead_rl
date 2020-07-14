from tensorboardX import SummaryWriter
import wandb
import logging

logger = logging.getLogger(__name__)


class Logger(object):

    def __init__(self, log_dir, name, project_name="follow_ahead_d4pg_v0.1newplannerdense600numa41", use_wandb=True, reinit=False):
        """
        General logger.

        Args:
            log_dir (str): log directory
        """

        if not use_wandb:
            self.writer = SummaryWriter(log_dir)
        else:
            wandb.init(project=project_name, name=name, reinit=reinit)

        self.use_wandb = use_wandb
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning

    def image_summar(self, tag, image, step):
        if self.use_wandb:
            wandb.log({tag: wandb.Image(image)},step=step)
        else:
            self.writer.add_image(tag, image, step, dataformats='HWC')

    def scalar_summary(self, tag, value, step):
        """
        Log scalar value to the disk.
        Args:
            tag (str): name of the value
            value (float): value
            step (int): update step
        """
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
        else:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.use_wandb:
            wandb.join()
        else:
            self.writer.close()

