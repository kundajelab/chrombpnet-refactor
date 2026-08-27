"""Small framework-neutral training log helper."""


class LossHistory:
    """Retain the historical callback API without depending on Keras."""

    def __init__(self, model_output_path_logs_name, to_track):
        self.model_output_path_logs_name = model_output_path_logs_name
        self.to_track = list(to_track)
        self.outf = open(model_output_path_logs_name, "w")
        self.outf.write("Epoch\tBatch\t" + "\t".join(self.to_track) + "\n")

    def on_train_begin(self, logs=None):
        self.losses = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.losses[epoch] = {name: [] for name in self.to_track}
        self.cur_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for name in self.to_track:
            self.losses[self.cur_epoch][name].append(logs.get(name))

    def on_epoch_end(self, epoch, logs=None):
        for batch in range(len(self.losses[epoch][self.to_track[0]])):
            row = [epoch, batch]
            row.extend(self.losses[epoch][name][batch] for name in self.to_track)
            self.outf.write("\t".join(map(str, row)) + "\n")
        self.outf.flush()

    def on_train_end(self, logs=None):
        self.outf.close()
