from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class CapsuleNetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(CapsuleNetTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        val_loop = tqdm(range(self.config.num_iter_val))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)

        val_losses, val_accs = [], []
        for _ in val_loop:
            X_batch, y_batch = self.data.next_batch_val(self.config.batch_size)
            loss_val, acc_val = self.sess.run(
                [self.model.loss, self.model.accuracy],
                feed_dict={
                    self.model.X: X_batch.reshape([-1, 28, 28, 1]),
                    self.model.y: y_batch
                })
            val_losses.append(loss_val)
            val_accs.append(acc_val)

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {'train_loss': loss, 'validation_loss': val_loss,
                          'validation_accuracy': val_acc}

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self):
        batch_x, batch_y = self.data.next_batch(self.config.batch_size)
        feed_dict = {
            self.model.X: batch_x.reshape(-1, 28, 28, 1),
            self.model.y: batch_y
        }
        _, loss = self.sess.run([self.model.training_step, self.model.loss],
                                feed_dict=feed_dict)
        return loss
